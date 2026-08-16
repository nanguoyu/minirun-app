import Foundation
import StorageCore
import XCTest

@testable import MinirunKit

/// The drive assessment, checked on fixtures rather than on whatever drive
/// happens to be plugged into the machine running the suite.
///
/// Every assertion here is a sentence the report card will say to an operator
/// who is deciding whether to spend a day copying 1.56 TB onto a drive. The one
/// arm that touches a real volume is env-gated and lives in
/// `ReadSpotIntegrationTests`.
final class VolumeAssessmentTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-assess-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// A file of `bytes` zeroes, written without holding it all in memory.
    @discardableResult
    private func makeFile(_ name: String, bytes: Int) throws -> URL {
        let url = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        let chunk = Data(count: 1 << 20)
        var written = 0
        while written < bytes {
            let take = min(chunk.count, bytes - written)
            try handle.write(contentsOf: chunk.prefix(take))
            written += take
        }
        return url
    }

    // MARK: - The classification table

    /// The two ceilings this project measured, and the band between them that is
    /// deliberately not attributed to either.
    func testTheClassificationTableEdges() {
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 0.30e9), .belowUSB3Gen2)
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 0.599e9), .belowUSB3Gen2)
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 0.60e9), .usb3Gen2)
        // The rate the record actually measured behind the dock's hub.
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 0.95e9), .usb3Gen2)
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 1.15e9), .usb3Gen2)
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 1.151e9), .ambiguous)
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 2.29e9), .ambiguous)
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 2.30e9), .usb4OrThunderbolt)
        // The rate the same enclosure delivered over the USB4 tunnel.
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 2.84e9), .usb4OrThunderbolt)
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 3.99e9), .usb4OrThunderbolt)
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 4.00e9), .internalNVMe)
        // The Mac's internal SSD plateau.
        XCTAssertEqual(LinkClassifier.linkClass(forBytesPerSecond: 6.9e9), .internalNVMe)
    }

    /// The phone's port sits inside the USB3 Gen2 band, which is the truthful
    /// place for it, and it is a known ceiling rather than a class of its own.
    func testTheIPhonePortCeilingLandsInTheUSB3Band() {
        XCTAssertEqual(
            LinkClassifier.linkClass(
                forBytesPerSecond: LinkClassifier.iPhonePortCeiling.bytesPerSecond),
            .usb3Gen2)
        XCTAssertFalse(LinkClassifier.iPhonePortCeiling.source.isEmpty)
    }

    func testEveryTableRowNamesTheRecordItCameFrom() {
        for reference in LinkClassifier.table {
            XCTAssertTrue(
                reference.source.hasPrefix("docs/experiments/"),
                "\(reference.name) cites '\(reference.source)'")
            XCTAssertGreaterThan(reference.bytesPerSecond, 0)
        }
    }

    /// No measurement, no claim — the same rule the projection layer keeps.
    func testWithoutAMeasurementTheClassificationIsUndetermined() {
        let classification = LinkClassifier.classify(bytesPerSecond: nil)
        XCTAssertEqual(classification.linkClass, .ambiguous)
        XCTAssertEqual(classification.provenance, .undetermined)
        XCTAssertNil(classification.measuredBytesPerSecond)
        XCTAssertNil(classification.fasterTierMultiple)
        XCTAssertFalse(classification.isDegraded)
        XCTAssertTrue(classification.sentence.contains("undetermined"))
    }

    /// Hardware alone names a family and never a speed.
    func testHardwareAloneNeverProducesARate() {
        let facts = HardwareLinkFacts(
            interconnect: "USB", location: "External", bsdName: "disk4s2")
        let classification = LinkClassifier.classify(bytesPerSecond: nil, hardware: facts)
        XCTAssertEqual(classification.provenance, .hardware)
        XCTAssertNil(classification.measuredBytesPerSecond)
        XCTAssertTrue(classification.sentence.contains("USB"))
        XCTAssertTrue(classification.sentence.contains("family and not a speed"))
        XCTAssertFalse(classification.sentence.contains("USB4"))
    }

    func testMeasurementAndHardwareTogetherSayBoth() {
        let facts = HardwareLinkFacts(
            interconnect: "PCI-Express", location: "Internal", bsdName: "disk3s1")
        let classification = LinkClassifier.classify(bytesPerSecond: 6.5e9, hardware: facts)
        XCTAssertEqual(classification.provenance, .both)
        XCTAssertEqual(classification.linkClass, .internalNVMe)
        XCTAssertTrue(classification.sentence.contains("PCI-Express"))
    }

    /// The multiple is a ratio of two measured numbers, and exists only for a
    /// link the table has a faster tier for.
    func testTheDegradedMultipleIsTheRatioOfTwoMeasuredNumbers() {
        let classification = LinkClassifier.classify(bytesPerSecond: 0.95e9)
        XCTAssertTrue(classification.isDegraded)
        XCTAssertEqual(classification.fasterTierMultiple ?? 0, 2.84 / 0.95, accuracy: 1e-9)
        XCTAssertEqual(classification.fasterTierName, "USB4 tunnel, 40 Gb/s")

        let fast = LinkClassifier.classify(bytesPerSecond: 2.84e9)
        XCTAssertFalse(fast.isDegraded)
        XCTAssertNil(fast.fasterTierMultiple)

        // The band between the ceilings gets no multiple either: there is no
        // measured tier to compare an unattributable rate against.
        XCTAssertNil(LinkClassifier.classify(bytesPerSecond: 1.8e9).fasterTierMultiple)
    }

    /// 3.4× is the difference the record recorded on the token. The link ratio
    /// this table produces is the ~3× that explains it.
    func testTheDegradedSentenceQuotesTheMeasuredNumbers() throws {
        let sentence = try XCTUnwrap(makeReport(bytesPerSecond: 0.95e9).degradedLinkSentence)
        // `MRFormat`'s rule, kept here: under 1 GB/s a rate reads in MB/s.
        XCTAssertTrue(sentence.contains("950 MB/s"))
        XCTAssertTrue(sentence.contains("2.84 GB/s"))
        XCTAssertTrue(sentence.contains("3.0×"))
        XCTAssertNil(makeReport(bytesPerSecond: 2.84e9).degradedLinkSentence)
    }

    // MARK: - Fit arithmetic

    func testAModelThatFitsReportsExactlyWhatIsSpare() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "K3-flagship", catalogBytes: 1_000_000_000_000,
            bytesOnDisk: nil, availableBytes: 2_000_000_000_000, marginBytes: 8 << 30)
        XCTAssertEqual(
            fit.verdict, .fits(spareBytes: 2_000_000_000_000 - 1_000_000_000_000 - (8 << 30)))
        XCTAssertTrue(fit.fits)
        XCTAssertFalse(fit.requiredIsMeasured)
    }

    /// Exactly on the margin fits; one byte under does not. The boundary is
    /// stated rather than left to a `>` nobody checked.
    func testTheMarginBoundaryIsInclusive() {
        let need: Int64 = 1_000_000_000_000 + (8 << 30)
        let exact = FitCheck.check(
            model: .kimiK3, displayName: "K3", catalogBytes: 1_000_000_000_000, bytesOnDisk: nil,
            availableBytes: need, marginBytes: 8 << 30)
        XCTAssertEqual(exact.verdict, .fits(spareBytes: 0))

        let oneShort = FitCheck.check(
            model: .kimiK3, displayName: "K3", catalogBytes: 1_000_000_000_000, bytesOnDisk: nil,
            availableBytes: need - 1, marginBytes: 8 << 30)
        XCTAssertEqual(oneShort.verdict, .short(byBytes: 1))
        XCTAssertEqual(oneShort.shortfallBytes, 1)
        XCTAssertFalse(oneShort.fits)
    }

    func testNegativeAvailableCapacityIsRefusedAsUnknown() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "K3", catalogBytes: 100, bytesOnDisk: nil,
            availableBytes: -1, marginBytes: 10)

        XCTAssertFalse(fit.fits)
        XCTAssertNil(fit.shortfallBytes)
        guard case .unknown(let reason) = fit.verdict else {
            return XCTFail("expected .unknown")
        }
        XCTAssertTrue(reason.contains("negative"), reason)
    }

    func testMaximumCatalogSizeCannotWrapIntoAFit() {
        let exactSignedLimit = FitCheck.check(
            model: .kimiK3, displayName: "K3",
            catalogBytes: UInt64(Int64.max), bytesOnDisk: nil,
            availableBytes: .max, marginBytes: 0)
        XCTAssertEqual(exactSignedLimit.verdict, .fits(spareBytes: 0))

        let largerThanSignedCapacity = FitCheck.check(
            model: .kimiK3, displayName: "K3", catalogBytes: .max, bytesOnDisk: nil,
            availableBytes: .max, marginBytes: 0)
        XCTAssertEqual(
            largerThanSignedCapacity.verdict,
            .short(byBytes: UInt64.max - UInt64(Int64.max)))
        XCTAssertFalse(largerThanSignedCapacity.fits)

        let overflowingNeed = FitCheck.check(
            model: .kimiK3, displayName: "K3", catalogBytes: .max, bytesOnDisk: nil,
            availableBytes: .max, marginBytes: 1)
        XCTAssertFalse(overflowingNeed.fits)
        XCTAssertNil(overflowingNeed.shortfallBytes)
        guard case .unknown(let reason) = overflowingNeed.verdict else {
            return XCTFail("expected .unknown")
        }
        XCTAssertTrue(reason.contains("overflow"), reason)
    }

    /// A volume too small by a terabyte must not read as a comfortable fit,
    /// which is what an unsigned subtraction would have made it.
    func testAVolumeFarTooSmallDoesNotWrapIntoAFit() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "K3", catalogBytes: 1_559_976_181_760, bytesOnDisk: nil,
            availableBytes: 250_000_000_000, marginBytes: 8 << 30)
        XCTAssertFalse(fit.fits)
        XCTAssertEqual(
            fit.shortfallBytes, UInt64(1_559_976_181_760 + (8 << 30) - 250_000_000_000))
    }

    /// Files win over metadata: an artifact that is here is measured.
    func testAnArtifactAlreadyHereIsMeasuredNotEstimated() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "K3", catalogBytes: 1_559_976_181_760,
            bytesOnDisk: 1_559_000_000_000, availableBytes: 10_000_000_000)
        XCTAssertEqual(fit.verdict, .alreadyHere(bytesOnDisk: 1_559_000_000_000))
        XCTAssertEqual(fit.requiredBytes, 1_559_000_000_000)
        XCTAssertTrue(fit.requiredIsMeasured)
        XCTAssertTrue(fit.fits)
    }

    func testAnUnprobeableVolumeIsUnknownAndNotFine() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "K3", catalogBytes: 100, bytesOnDisk: nil,
            availableBytes: nil)
        XCTAssertFalse(fit.fits)
        guard case .unknown = fit.verdict else { return XCTFail("expected .unknown") }
    }

    func testTheMarginIsTheDownloadersOwnHeadroom() {
        XCTAssertEqual(FitCheck.defaultMarginBytes, DownloadOptions().freeSpaceHeadroomBytes)
    }

    // MARK: - Projection

    func testNoProjectionWithoutAMeasurement() {
        let workload = TokenWorkload(
            deterministicBytesPerToken: 111_200_000_000,
            expertBytesPerToken: 77_500_000_000, computeSecondsPerToken: 160.64)
        XCTAssertNil(TokenTimeProjector.project(workload: workload, bytesPerSecond: nil))
        XCTAssertNil(TokenTimeProjector.project(workload: workload, bytesPerSecond: 0))
        XCTAssertNil(TokenTimeProjector.project(workload: workload, bytesPerSecond: .nan))
    }

    /// The serial shape is compute plus both byte terms, at K3's measured terms
    /// over the link the record measured. Nothing is hidden and nothing is
    /// cached, which is what "serial" means.
    func testTheSerialShapeIsComputePlusBothByteTerms() throws {
        let workload = TokenWorkload(
            deterministicBytesPerToken: 111_200_000_000,
            expertBytesPerToken: 77_500_000_000, computeSecondsPerToken: 160.64)
        let projection = try XCTUnwrap(
            TokenTimeProjector.project(workload: workload, bytesPerSecond: 0.92e9))
        XCTAssertEqual(projection.mid, 160.64 + 188_700_000_000.0 / 0.92e9, accuracy: 1e-9)
        XCTAssertEqual(projection.secondsPerTokenLow, projection.mid * 0.88, accuracy: 1e-9)
        XCTAssertEqual(projection.secondsPerTokenHigh, projection.mid * 1.12, accuracy: 1e-9)
        XCTAssertFalse(projection.deterministicHidden)
    }

    /// The step in the curve: with the staged pair funded the token costs the
    /// longer of the two paths, not their sum.
    func testTheStagedShapeTakesTheLongerPathNotTheSum() throws {
        let workload = TokenWorkload(
            deterministicBytesPerToken: 111_200_000_000,
            expertBytesPerToken: 77_500_000_000, computeSecondsPerToken: 160.64,
            expertCacheFraction: 0, deterministicOverlapsCompute: true)
        let projection = try XCTUnwrap(
            TokenTimeProjector.project(workload: workload, bytesPerSecond: 0.92e9))
        let deterministic = 111_200_000_000.0 / 0.92e9
        let other = 160.64 + 77_500_000_000.0 / 0.92e9
        XCTAssertEqual(projection.mid, max(deterministic, other), accuracy: 1e-9)
        XCTAssertTrue(projection.deterministicHidden)
    }

    /// A faster link buys exactly the byte term, and nothing off the compute
    /// term. The projection must not promise otherwise.
    func testAFasterLinkOnlyBuysTheByteTerm() throws {
        let workload = TokenWorkload(
            deterministicBytesPerToken: 111_200_000_000,
            expertBytesPerToken: 77_500_000_000, computeSecondsPerToken: 160.64)
        let slow = try XCTUnwrap(
            TokenTimeProjector.project(workload: workload, bytesPerSecond: 0.95e9))
        let fast = try XCTUnwrap(
            TokenTimeProjector.project(workload: workload, bytesPerSecond: 2.84e9))
        XCTAssertLessThan(fast.mid, slow.mid)
        XCTAssertGreaterThan(fast.mid, 160.64)
        XCTAssertLessThan(slow.mid / fast.mid, 2.84 / 0.95)
    }

    // MARK: - The report

    /// Whole seconds, because these documents encode dates as ISO 8601 without
    /// a fractional part — the persisted report is therefore equal to the
    /// original to the second, and the test says which second rather than
    /// pretending the round trip is bit-exact.
    func testAReportSurvivesARoundTripThroughJSON() throws {
        let report = makeReport(
            bytesPerSecond: 0.95e9, at: Date(timeIntervalSince1970: 1_770_000_000))
        let data = try MinirunKitJSON.encoder(pretty: false).encode(report)
        let decoded = try MinirunKitJSON.decoder().decode(
            VolumeAssessmentReport.self, from: data)
        XCTAssertEqual(decoded, report)
        XCTAssertEqual(decoded.measuredBytesPerSecond, report.measuredBytesPerSecond)
    }

    func testAReportGoesStaleAfterSevenDays() {
        let now = Date()
        let fresh = makeReport(bytesPerSecond: 0.95e9, at: now)
        XCTAssertFalse(fresh.isStale(now: now))
        XCTAssertFalse(fresh.isStale(now: now.addingTimeInterval(6 * 86400)))
        XCTAssertTrue(fresh.isStale(now: now.addingTimeInterval(7 * 86400 + 1)))
    }

    func testTheLedgerRemembersOneReportPerLocation() {
        let ledger = InMemoryAssessmentLedger()
        let report = makeReport(bytesPerSecond: 0.95e9)
        ledger.record(report)
        XCTAssertEqual(ledger.report(forLocationPath: report.locationPath), report)
        ledger.forget(locationPath: report.locationPath)
        XCTAssertNil(ledger.report(forLocationPath: report.locationPath))
    }

    /// A verdict line is what an operator reads before spending a day copying.
    func testTheVerdictLineSaysBothHalvesOfTheAnswer() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "K3-flagship", catalogBytes: 1_559_976_181_760,
            bytesOnDisk: nil, availableBytes: 2_000_000_000_000)
        let projection = TokenTimeProjector.project(
            workload: TokenWorkload(
                deterministicBytesPerToken: 111_200_000_000,
                expertBytesPerToken: 77_500_000_000, computeSecondsPerToken: 160.64),
            bytesPerSecond: 0.95e9)
        let line = ModelVerdict(fit: fit, projection: projection).line
        XCTAssertTrue(line.hasPrefix("K3-flagship: 1.56 TB fits"))
        XCTAssertTrue(line.contains("≈"))
        XCTAssertTrue(line.contains("/ token"))
        XCTAssertTrue(line.contains("950 MB/s"))
    }

    func testAVerdictWithoutAMeasurementShowsNoTokenTime() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "K3-flagship", catalogBytes: 100, bytesOnDisk: nil,
            availableBytes: 10_000_000_000_000)
        let line = ModelVerdict(fit: fit, projection: nil).line
        XCTAssertTrue(line.contains("No rate was measured"))
        XCTAssertFalse(line.contains("≈"))
    }

    // MARK: - Choosing what to read

    func testTargetsAreTheLargestFilesAndBookkeepingIsSkipped() throws {
        try makeFile("layer00/w1.mxfp4tile", bytes: 24 << 20)
        try makeFile("layer01/w1.mxfp4tile", bytes: 20 << 20)
        try makeFile("layer02/w1.mxfp4tile", bytes: 8 << 20)  // under the minimum
        try makeFile(".cache/huge.part", bytes: 40 << 20)  // the downloader's tree
        try makeFile(".DS_Store", bytes: 1)

        let test = ReadSpotTest(configuration: reducedConfiguration(maximumTargets: 4))
        let targets = try test.targets(under: root)
        XCTAssertEqual(targets.count, 2)
        XCTAssertEqual(targets.map(\.sizeBytes), [24 << 20, 20 << 20])
        XCTAssertFalse(targets.contains { $0.path.contains(".cache") })
    }

    func testTargetsAreCappedAndReproducible() throws {
        for index in 0..<6 {
            try makeFile("layer0\(index)/w.mxfp4tile", bytes: 20 << 20)
        }
        let test = ReadSpotTest(configuration: reducedConfiguration(maximumTargets: 3))
        let first = try test.targets(under: root)
        let second = try test.targets(under: root)
        XCTAssertEqual(first.count, 3)
        // Equal sizes break on path, so two runs read the same three files.
        XCTAssertEqual(first, second)
    }

    /// The volume that matters most is read-only. Nothing may be written to make
    /// a test possible, and the refusal has to say so.
    func testAnEmptyLocationIsARefusalAndNothingIsWritten() throws {
        let test = ReadSpotTest(configuration: reducedConfiguration())
        let before = try FileManager.default.contentsOfDirectory(atPath: root.path)
        do {
            _ = try test.run(on: root)
            XCTFail("a location with nothing to read must refuse")
        } catch let refusal as ReadSpotRefusal {
            guard case .noTargets = refusal else { return XCTFail("expected .noTargets") }
            XCTAssertTrue(refusal.description.contains("Nothing was written"))
        }
        let after = try FileManager.default.contentsOfDirectory(atPath: root.path)
        XCTAssertEqual(before, after)
    }

    // MARK: - Running the test

    func testASpotTestReadsWhatItPlannedAndReportsARate() throws {
        try makeFile("layer00/w1.mxfp4tile", bytes: 24 << 20)
        try makeFile("layer01/w1.mxfp4tile", bytes: 24 << 20)

        var configuration = reducedConfiguration()
        configuration.sequentialBytes = 16 << 20
        configuration.scatteredReadCount = 8
        let result = try ReadSpotTest(configuration: configuration).run(on: root)

        XCTAssertEqual(result.sequentialBytesRead, 16 << 20)
        XCTAssertEqual(result.scatteredReadCount, 8)
        XCTAssertEqual(result.scatteredBytesRead, UInt64(8 * configuration.scatteredReadBytes))
        XCTAssertEqual(result.shortReadCount, 0)
        XCTAssertGreaterThan(result.sequentialBytesPerSecond, 0)
        XCTAssertGreaterThan(result.scatteredBytesPerSecond, 0)
        XCTAssertEqual(result.referenceBytesPerSecond, result.sequentialBytesPerSecond)
        XCTAssertEqual(result.totalBytesRead, result.sequentialBytesRead + result.scatteredBytesRead)
    }

    func testProgressReachesTheScatteredPhaseAndNeverExceedsThePlan() throws {
        try makeFile("layer00/w1.mxfp4tile", bytes: 24 << 20)
        var configuration = reducedConfiguration()
        configuration.sequentialBytes = 8 << 20
        configuration.scatteredReadCount = 4

        var phases: Set<ReadSpotProgress.Phase> = []
        var worst = 0.0
        _ = try ReadSpotTest(configuration: configuration).run(on: root) { progress in
            phases.insert(progress.phase)
            worst = max(worst, progress.fraction)
        }
        XCTAssertTrue(phases.contains(.sequential))
        XCTAssertTrue(phases.contains(.scattered))
        XCTAssertLessThanOrEqual(worst, 1.0)
    }

    func testCancellationStopsTheTestWithinAChunk() throws {
        try makeFile("layer00/w1.mxfp4tile", bytes: 24 << 20)
        var configuration = reducedConfiguration()
        configuration.sequentialBytes = 24 << 20

        var reads = 0
        do {
            _ = try ReadSpotTest(configuration: configuration).run(
                on: root,
                isCancelled: {
                    reads += 1
                    return reads > 2
                })
            XCTFail("a cancelled test must not return a result")
        } catch let refusal as ReadSpotRefusal {
            XCTAssertEqual(refusal, .cancelled)
        }
    }

    // MARK: - The assessor

    func testAssessingALocationCombinesEveryHalfOfTheAnswer() throws {
        try makeFile("k3/layer00/w1.mxfp4tile", bytes: 24 << 20)
        var configuration = reducedConfiguration()
        configuration.sequentialBytes = 8 << 20
        configuration.scatteredReadCount = 4

        let assessor = VolumeAssessor(
            spotTest: ReadSpotTest(configuration: configuration),
            hardwareProbe: UndeterminedHardwareProbe())
        let report = assessor.assess(
            location: root, scan: nil, catalog: ModelCatalog.bundled,
            workloads: [
                .kimiK3: TokenWorkload(
                    deterministicBytesPerToken: 111_200_000_000,
                    expertBytesPerToken: 77_500_000_000, computeSecondsPerToken: 160.64)
            ])

        XCTAssertNotNil(report.spot)
        XCTAssertNil(report.caution)
        XCTAssertEqual(report.classification.provenance, .measurement)
        XCTAssertEqual(report.verdicts.count, ModelCatalog.bundled.models.count)
        let k3 = try XCTUnwrap(report.verdicts.first { $0.fit.model == .kimiK3 })
        XCTAssertNotNil(k3.projection)
        // Every other model was given no workload, so none of them gets a
        // projected token time — a projection is never invented for a model
        // whose terms nobody measured.
        for verdict in report.verdicts where verdict.fit.model != .kimiK3 {
            XCTAssertNil(verdict.projection)
        }
    }

    func testALocationWithNothingToReadIsACautionAndNotAThrow() {
        let assessor = VolumeAssessor(
            spotTest: ReadSpotTest(configuration: reducedConfiguration()),
            hardwareProbe: UndeterminedHardwareProbe())
        let report = assessor.assess(
            location: root, scan: nil, catalog: ModelCatalog.bundled, workloads: [:])
        XCTAssertNil(report.spot)
        XCTAssertNotNil(report.caution)
        XCTAssertEqual(report.classification.provenance, .undetermined)
        // The fit checks still ran: a volume that cannot be timed can still be
        // too small, and that is worth saying.
        XCTAssertFalse(report.verdicts.isEmpty)
        XCTAssertTrue(report.verdicts.allSatisfy { $0.projection == nil })
    }

    /// The spot test reads the artifact's own bytes when there is an artifact,
    /// because those are the bytes a run will stream.
    func testTheReadRootIsTheLargestArtifactWhenThereIsOne() {
        let assessor = VolumeAssessor(hardwareProbe: UndeterminedHardwareProbe())
        let small = artifact(at: root.appendingPathComponent("small").path, bytes: 10)
        let big = artifact(at: root.appendingPathComponent("big").path, bytes: 1_000_000)
        let scan = LocationScan(
            rootPath: root.path, displayName: "fixture", isMounted: true, storageKey: nil,
            artifacts: [small, big], unreadable: [:], scannedAt: Date())
        XCTAssertEqual(assessor.readRoot(for: root, scan: scan).path, big.rootPath)
        XCTAssertEqual(assessor.readRoot(for: root, scan: nil).path, root.path)
    }

    // MARK: - Helpers

    private func reducedConfiguration(maximumTargets: Int = 8) -> ReadSpotTest.Configuration {
        ReadSpotTest.Configuration(
            sequentialBytes: 4 << 20, sequentialReadBytes: 1 << 20,
            scatteredReadCount: 4, scatteredReadBytes: 1 << 20,
            minimumTargetBytes: 16 << 20, maximumTargets: maximumTargets, seed: 7)
    }

    private func artifact(at path: String, bytes: UInt64) -> DiscoveredArtifact {
        DiscoveredArtifact(
            rootPath: path, locationPath: root.path, index: .unreadable, model: .kimiK3,
            displayName: "K3", bytesOnDisk: bytes, fileCount: 1, expectedFileCount: nil,
            expectedBytes: nil, verification: .unverified, verifiedAt: nil, scannedAt: Date())
    }

    private func makeReport(bytesPerSecond: Double, at date: Date = Date())
        -> VolumeAssessmentReport
    {
        let spot = ReadSpotResult(
            locationPath: "/Volumes/K3NVME", targets: [
                SpotTestTarget(path: "/Volumes/K3NVME/k3/layer00/w1.mxfp4tile", sizeBytes: 1 << 30)
            ],
            sequentialBytesRead: 2_000_000_000,
            sequentialSeconds: 2_000_000_000 / bytesPerSecond,
            scatteredBytesRead: 1_073_741_824, scatteredSeconds: 1.5,
            scatteredReadCount: 256, scatteredReadBytes: 4 << 20,
            shortReadCount: 0, cacheBypassed: true, startedAt: date, durationSeconds: 4)
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "K3-flagship", catalogBytes: 1_559_976_181_760,
            bytesOnDisk: nil, availableBytes: 2_000_000_000_000)
        return VolumeAssessmentReport(
            locationPath: "/Volumes/K3NVME", volumeName: "K3NVME",
            volumeMountPath: "/Volumes/K3NVME",
            space: FreeSpace(
                optimisticBytes: 2_100_000_000_000, dependableBytes: 2_000_000_000_000,
                totalBytes: 4_000_000_000_000, volumeName: "K3NVME", isReadOnly: true,
                isRemovable: true),
            assessedAt: date, spot: spot, caution: nil,
            classification: LinkClassifier.classify(bytesPerSecond: bytesPerSecond),
            verdicts: [ModelVerdict(fit: fit, projection: nil)])
    }
}
