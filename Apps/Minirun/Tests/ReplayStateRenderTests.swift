import MinirunKit
import MinirunRunners
import SwiftUI
import XCTest

@testable import MinirunApp

/// A conversation being replayed says so, in the two places prefill progress
/// already appears: the line under the pending reply, and the Instruments
/// stage.
@MainActor
final class ReplayStateTests: XCTestCase {
    private func running(phase: RunPhase) -> RunSnapshot {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.phase = phase
        return snapshot
    }

    func testAReplayIsASentenceAndNothingElseIs() {
        let replay = running(
            phase: RunPhase(
                name: "prefill", generationStage: .prefill, fraction: 0.4,
                detail: "block 16/40", replay: RunReplayProgress(token: 12, of: 37)))
        XCTAssertEqual(replay.replaySentence, "Restoring the conversation, token 12 of 37")
        XCTAssertEqual(
            InstrumentPanelView(snapshot: replay).stageDetail,
            "Restoring the conversation, token 12 of 37")
        XCTAssertEqual(InstrumentPanelView(snapshot: replay).stageTitle, "Prefill")
        XCTAssertNil(
            replay.prefillRemainingSecondsEstimate,
            "a pass's layers say nothing about how many passes a replay has left")

        let prefill = running(
            phase: RunPhase(
                name: "prefill", generationStage: .prefill, fraction: 0.4,
                detail: "block 16/40"))
        XCTAssertNil(prefill.replaySentence)
        XCTAssertEqual(InstrumentPanelView(snapshot: prefill).stageDetail, "Reading prompt")

        var finished = replay
        finished.state = .finished
        XCTAssertNil(finished.replaySentence, "a finished run is not replaying")
    }
}

#if os(macOS)
    /// The replay state drawn, at the Mac inspector's and the phone's widths
    /// for Instruments and at the conversation column's and the phone's for
    /// the pending reply, in both appearances.
    @MainActor
    final class ReplayStateRenderTests: XCTestCase {

        private func liveSample() async -> RunSnapshot {
            let entry = CatalogFixtures.deepseekV41Flash
            let controller = RunController()
            let ended = expectation(description: "turn ended")
            controller.onTurnEnded = { _ in ended.fulfill() }
            controller.start(
                runner: MockDecodeRunner(entry: entry, reviewSeconds: 1),
                request: RunRequest(
                    model: entry.id,
                    artifact: ArtifactReference(root: URL(fileURLWithPath: "/tmp/model")),
                    prompt: .tokenIDs([17529]),
                    memoryBudgetBytes: 14_900_000_000,
                    maximumNewTokens: 2,
                    workingDirectory: URL(fileURLWithPath: NSTemporaryDirectory())),
                linkCeiling: 0.92e9,
                layerCount: 40, storedBytesPerLayer: 180_000_000,
                widenedBytesPerLayer: 180_000_000)
            await fulfillment(of: [ended], timeout: 60)
            let live = controller.snapshot
            controller.clear()
            return live
        }

        /// Twelve ids of thirty-seven, block 16 of 40 in flight.
        private func replaying(_ sample: RunTelemetry, acceptance: RunAcceptance?) -> RunSnapshot {
            var snapshot = RunSnapshot()
            snapshot.acceptance = acceptance
            snapshot.state = .running
            snapshot.startedAt = Date()
            snapshot.ladder = (0..<40).map {
                LayerCell(
                    index: $0, status: $0 < 16 ? .done : ($0 == 16 ? .computing : .pending),
                    storedBytes: 180_000_000, widenedBytes: 180_000_000, wallSeconds: nil,
                    wasStaged: false)
            }
            snapshot.phase = RunPhase(
                name: "prefill", generationStage: .prefill, fraction: 0.4,
                detail: "block 16/40", replay: RunReplayProgress(token: 12, of: 37))
            snapshot.telemetry = RunTelemetry(
                at: sample.at, elapsed: 64, phase: "prefill, replay token 12 of 37",
                tokensPerSecond: nil, generationStage: .prefill,
                bytes: ByteAccounting(
                    totalBytesRead: 21_400_000_000, deterministicBytesRead: 7_200_000_000,
                    expertBytesRead: 14_200_000_000),
                bytesPerSecond: 21_400_000_000 / 64,
                completedTokenBytes: nil, prefillBytes: nil, bytesPerToken: nil,
                prefillSeconds: nil, decodeSeconds: 0, decodeTokensCompleted: 0,
                byteAccountingReported: true,
                declaredBudgetBytes: 14_900_000_000,
                footprintBytes: 12_600_000_000, peakFootprintBytes: 12_900_000_000,
                entryFootprintBytes: nil,
                residentBytes: 12_600_000_000, availableBytes: sample.availableBytes,
                mlxActiveBytes: sample.mlxActiveBytes, mlxCacheBytes: sample.mlxCacheBytes,
                mlxPeakBytes: sample.mlxPeakBytes, thermalState: .nominal,
                lowPowerMode: false, batteryLevel: nil,
                instrumentation: nil)
            return snapshot
        }

        private func render<V: View>(_ content: V, width: CGFloat, scheme: ColorScheme) throws
            -> NSImage
        {
            let renderer = ImageRenderer(
                content: content
                    .padding(MRSpace.s4)
                    .frame(width: width)
                    .background(MRColor.abyss)
                    .environment(\.colorScheme, scheme))
            renderer.scale = 2
            return try XCTUnwrap(renderer.nsImage)
        }

        @discardableResult
        private func write(_ image: NSImage, named name: String) throws -> URL {
            let directory = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-replay-states", isDirectory: true)
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let tiff = try XCTUnwrap(image.tiffRepresentation)
            let png = try XCTUnwrap(
                NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
            let url = directory.appendingPathComponent("\(name).png")
            try png.write(to: url)
            let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
            attachment.name = "\(name).png"
            attachment.lifetime = .keepAlways
            add(attachment)
            print("rendered \(name).png at \(url.path)")
            return url
        }

        func testTheReplayStateDrawsInInstrumentsAndUnderThePendingReply() async throws {
            let live = await liveSample()
            let sample = try XCTUnwrap(live.telemetry)
            let snapshot = replaying(sample, acceptance: live.acceptance)
            var drawn = 0
            for scheme in [ColorScheme.light, ColorScheme.dark] {
                for (width, device) in [(CGFloat(372), "mac"), (CGFloat(390), "phone")] {
                    let image = try render(
                        InstrumentPanelView(snapshot: snapshot).sections,
                        width: width, scheme: scheme)
                    let url = try write(image, named: "replay-panel-\(device)-\(scheme)")
                    XCTAssertGreaterThan(image.size.height, 300, url.path)
                    drawn += 1
                }
                for (width, device) in [(CGFloat(620), "mac"), (CGFloat(390), "phone")] {
                    let image = try render(
                        PendingTurnView(snapshot: snapshot, isRunning: true),
                        width: width, scheme: scheme)
                    let url = try write(image, named: "replay-turn-\(device)-\(scheme)")
                    XCTAssertGreaterThan(image.size.height, 20, url.path)
                    drawn += 1
                }
            }
            XCTAssertEqual(drawn, 8)
        }
    }
#endif
