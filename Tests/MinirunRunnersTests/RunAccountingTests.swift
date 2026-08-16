import Foundation
import MinirunKit
import ModelAdapters
import XCTest

@testable import MinirunRunners

/// The pieces of `MinirunKit` that carry an invariant rather than a value, seen
/// from the side that a real run uses them from.
///
/// `MinirunKitTests` owns these types' own suite; what is checked here is the
/// reading the runner depends on — an invariant that only holds inside a K3 run
/// is an invariant nobody can read.
final class RunAccountingTests: XCTestCase {

    /// Every byte in exactly one bucket — and "we do not know" reported as such
    /// rather than as zero.
    func testByteAccountingSaysWhenItsIdentityHolds() {
        let balanced = ByteAccounting(
            totalBytesRead: 300, deterministicBytesRead: 200, expertBytesRead: 100)
        XCTAssertEqual(balanced.unattributedBytes, 0)
        XCTAssertTrue(balanced.isBalanced)

        let short = ByteAccounting(
            totalBytesRead: 300, deterministicBytesRead: 200, expertBytesRead: 50)
        XCTAssertEqual(short.unattributedBytes, 50)
        XCTAssertFalse(short.isBalanced)

        // Unattributed and unknown are different claims, and neither is zero.
        // `isBalanced` reads true here because an unavailable split is not an
        // imbalance: the identity is not broken, it is not yet stated. A
        // consumer that needs the stronger reading asks `unattributedBytes`,
        // which is nil, and that is the distinction being kept.
        let unknown = ByteAccounting(totalBytesRead: 300, deterministicBytesRead: 200)
        XCTAssertNil(unknown.unattributedBytes)
        XCTAssertTrue(unknown.isBalanced)

        // Over-attribution has no remainder to report; it has a broken
        // identity, and reporting 0 there would hide it.
        let over = ByteAccounting(
            totalBytesRead: 300, deterministicBytesRead: 200, expertBytesRead: 200)
        XCTAssertNil(over.unattributedBytes)
        XCTAssertFalse(over.isBalanced)
    }

    func testByteAccountingSubtractsMonotonicTokenBoundaries() throws {
        let prefill = ByteAccounting(
            totalBytesRead: 140, deterministicBytesRead: 100, expertBytesRead: 40)
        let secondToken = ByteAccounting(
            totalBytesRead: 240, deterministicBytesRead: 170, expertBytesRead: 70)

        XCTAssertEqual(
            try XCTUnwrap(secondToken.subtracting(prefill)),
            ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 70, expertBytesRead: 30))
        XCTAssertEqual(
            ByteAccounting(totalBytesRead: 240).subtracting(
                ByteAccounting(totalBytesRead: 140)),
            ByteAccounting(totalBytesRead: 100))

        XCTAssertNil(prefill.subtracting(secondToken))
        XCTAssertNil(
            ByteAccounting(
                totalBytesRead: 250, deterministicBytesRead: 90, expertBytesRead: 160
            ).subtracting(prefill),
            "a component moving backwards must not become an unsigned decode cost")
    }

    /// The V4 phase brackets are cumulative for the same reason the byte
    /// counters are, and the engine turns them into one pass the same way:
    /// a difference of two boundaries, stated against the wall it measured.
    func testPhaseMetricsSubtractMonotonicPassBoundaries() throws {
        var atPrefillEnd = DeepSeekV4PhaseMetrics()
        atPrefillEnd.deterministicReadSeconds = 20
        atPrefillEnd.tileDigestSeconds = 6
        atPrefillEnd.expertPhaseSeconds = 30
        atPrefillEnd.expertIOWaitSeconds = 24
        atPrefillEnd.expertGatherComputeSeconds = 4
        atPrefillEnd.deterministicReadCount = 43

        var atDecodeEnd = atPrefillEnd
        atDecodeEnd.deterministicReadSeconds = 35
        atDecodeEnd.tileDigestSeconds = 11
        atDecodeEnd.expertPhaseSeconds = 52
        atDecodeEnd.expertIOWaitSeconds = 42
        atDecodeEnd.expertGatherComputeSeconds = 7
        atDecodeEnd.deterministicReadCount = 86

        let pass = try XCTUnwrap(
            atDecodeEnd.subtracting(atPrefillEnd)?.withPassSeconds(53))
        XCTAssertEqual(pass.deterministicReadSeconds, 15, accuracy: 1e-9)
        XCTAssertEqual(pass.tileDigestSeconds, 5, accuracy: 1e-9)
        XCTAssertEqual(pass.expertPhaseSeconds, 22, accuracy: 1e-9)
        XCTAssertEqual(pass.deterministicReadCount, 43)
        XCTAssertEqual(pass.attributedSeconds, 42, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(pass.unattributedSeconds), 11, accuracy: 1e-9)
        XCTAssertTrue(pass.isAccountingBalanced)

        // The decode boundary cannot precede the prefill one, and a pass whose
        // named terms exceed its wall is refused rather than rounded — both are
        // evidence the two samples do not describe one pass.
        XCTAssertNil(atPrefillEnd.subtracting(atDecodeEnd))
        XCTAssertFalse(pass.withPassSeconds(30).isAccountingBalanced)
    }

    /// The peak is quoted against the stated budget, never against physical
    /// memory, and a run that broke its promise says so.
    func testTelemetryQuotesThePeakAgainstTheStatedBudget() {
        func telemetry(peak: UInt64, budget: UInt64) -> RunTelemetry {
            RunTelemetry(
                at: Date(), elapsed: 1, phase: "decode 1", tokensPerSecond: 1,
                bytes: ByteAccounting(totalBytesRead: 0), bytesPerSecond: 0, bytesPerToken: 0,
                declaredBudgetBytes: budget, footprintBytes: peak, peakFootprintBytes: peak,
                residentBytes: peak, availableBytes: nil, mlxActiveBytes: 0, mlxCacheBytes: 0,
                mlxPeakBytes: 0, thermalState: .nominal, lowPowerMode: false, batteryLevel: nil)
        }
        XCTAssertTrue(telemetry(peak: 5_000, budget: 5_000).budgetRespected)
        XCTAssertFalse(telemetry(peak: 5_001, budget: 5_000).budgetRespected)
        XCTAssertEqual(telemetry(peak: 2_500, budget: 5_000).budgetFraction, 0.5)
        // A budget of zero is not a division to attempt.
        XCTAssertEqual(telemetry(peak: 1, budget: 0).budgetFraction, 0)
    }

    func testTelemetryDistinguishesUnreportedBytesFromAMeasuredZero() throws {
        let unavailable = RunTelemetry(
            at: Date(), elapsed: 1, phase: "prefill", tokensPerSecond: nil,
            bytes: ByteAccounting(totalBytesRead: 0), bytesPerSecond: nil,
            bytesPerToken: nil, byteAccountingReported: false,
            declaredBudgetBytes: 4_000, footprintBytes: 100, peakFootprintBytes: 100,
            residentBytes: 100, availableBytes: nil, mlxActiveBytes: 0,
            mlxCacheBytes: 0, mlxPeakBytes: 0, thermalState: .nominal,
            lowPowerMode: false, batteryLevel: nil)

        XCTAssertFalse(unavailable.reportsByteAccounting)
        let encoded = try JSONEncoder().encode(unavailable)
        let decoded = try JSONDecoder().decode(RunTelemetry.self, from: encoded)
        XCTAssertEqual(decoded.byteAccountingReported, false)
        XCTAssertFalse(decoded.reportsByteAccounting)

        var legacyObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        legacyObject.removeValue(forKey: "byteAccountingReported")
        let legacy = try JSONDecoder().decode(
            RunTelemetry.self, from: JSONSerialization.data(withJSONObject: legacyObject))
        XCTAssertNil(legacy.byteAccountingReported)
        XCTAssertTrue(
            legacy.reportsByteAccounting,
            "records written before the availability field must retain their old meaning")
    }

    func testPayloadFlowSampleRejectsInvalidIntervalsAndOverflow() throws {
        let valid = RunPayloadFlowSample(
            sequence: 2, at: Date(timeIntervalSince1970: 4), intervalSeconds: 0.5,
            deterministicBytes: 100, expertBytes: 50)
        XCTAssertEqual(valid.totalBytes, 150)
        XCTAssertEqual(valid.deterministicBytesPerSecond, 200)
        XCTAssertEqual(valid.expertBytesPerSecond, 100)
        XCTAssertTrue(valid.isValid)
        XCTAssertEqual(
            try JSONDecoder().decode(
                RunPayloadFlowSample.self, from: JSONEncoder().encode(valid)),
            valid)

        let zeroInterval = RunPayloadFlowSample(
            sequence: 3, at: Date(), intervalSeconds: 0,
            deterministicBytes: 1, expertBytes: 1)
        XCTAssertFalse(zeroInterval.isValid)
        XCTAssertNil(zeroInterval.deterministicBytesPerSecond)

        let overflow = RunPayloadFlowSample(
            sequence: 4, at: Date(), intervalSeconds: 0.5,
            deterministicBytes: .max, expertBytes: 1)
        XCTAssertNil(overflow.totalBytes)
        XCTAssertFalse(overflow.isValid)
    }

    /// The kernel's peak is process-lifetime state. A newly raised kernel peak
    /// still captures a spike between callbacks, while a new RunSession whose
    /// lifetime baseline is already high reports only its own live samples.
    func testRunFootprintPeakUsesLifetimeDeltaWithoutInheritingOldPeak() {
        var firstRun = K3RunEngine.FootprintPeak(processLifetimePeakBytes: 100)
        XCTAssertEqual(
            firstRun.observe(currentBytes: 400, processLifetimePeakBytes: 900), 900)
        XCTAssertEqual(
            firstRun.observe(currentBytes: 350, processLifetimePeakBytes: 900), 900)

        var secondRun = K3RunEngine.FootprintPeak(processLifetimePeakBytes: 900)
        XCTAssertEqual(
            secondRun.observe(currentBytes: 120, processLifetimePeakBytes: 900), 120)
        XCTAssertEqual(
            secondRun.observe(currentBytes: 180, processLifetimePeakBytes: 900), 180)
        XCTAssertEqual(
            secondRun.observe(currentBytes: 140, processLifetimePeakBytes: 900), 180)
    }

    func testRunFootprintPeakFallsBackToCurrentSamplesWithoutBaseline() {
        var run = K3RunEngine.FootprintPeak(processLifetimePeakBytes: nil)
        XCTAssertEqual(run.observe(currentBytes: 120, processLifetimePeakBytes: 900), 120)
        XCTAssertEqual(run.observe(currentBytes: 180, processLifetimePeakBytes: 1_200), 180)
    }

    /// An unrecognised thermal name becomes `.unknown` rather than `.nominal`:
    /// "we could not read the state" and "the state was fine" are different
    /// things to put in a record.
    func testAnUnreadableThermalStateIsUnknownAndNotNominal() {
        XCTAssertEqual(ThermalStateName(name: "serious"), .serious)
        XCTAssertEqual(ThermalStateName(name: "not a state"), .unknown)
    }

    /// The latch is idempotent and readable from any thread — it is polled by a
    /// blocking forward pass that cannot await anything.
    func testTheCancellationLatchIsOneWayAndThreadSafe() {
        let latch = RunCancellationLatch()
        XCTAssertFalse(latch.isCancelled)
        XCTAssertNoThrow(try latch.check())

        DispatchQueue.concurrentPerform(iterations: 64) { _ in latch.cancel() }
        XCTAssertTrue(latch.isCancelled)
        XCTAssertThrowsError(try latch.check()) { error in
            XCTAssertEqual(error as? RunError, .cancelled)
        }
    }

    /// A scope releases once, and only once. `deinit` after an explicit
    /// `release()` must not revoke a grant this scope no longer owns.
    func testAStorageScopeReleasesExactlyOnce() {
        let scope = StorageScope(url: URL(fileURLWithPath: NSTemporaryDirectory()))
        XCTAssertFalse(scope.isReleased)
        // `isScoped` is reported, never asserted on: macOS hands out an
        // extension for an ordinary directory that iOS would refuse, and the
        // whole point of reporting it is that the caller decides what that
        // means. What must hold on both is the balance below.
        scope.release()
        XCTAssertTrue(scope.isReleased)
        scope.release()
        XCTAssertTrue(scope.isReleased)
    }
}
