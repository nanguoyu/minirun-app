#if os(macOS)
    import MinirunKit
    import MinirunRunners
    import SwiftUI
    import XCTest

    @testable import MinirunApp

    /// The whole Instruments panel, live and finished, on both budget bases, and
    /// the footer under a finished reply.
    ///
    /// These are the surfaces whose figure rows carry three or four numbers each,
    /// so they are the ones a change to how figures sit beside each other can
    /// break without any value test noticing: a row that wraps mid-figure at the
    /// inspector's width, or a caption that runs off the edge, is only visible
    /// in a picture. The PNGs land where a reviewer can look at them, in both
    /// appearances, at the Mac inspector's width and at the phone's.
    @MainActor
    final class InstrumentPanelStatesRenderTests: XCTestCase {

        private func finishedSnapshot() async -> RunSnapshot {
            let entry = CatalogFixtures.deepseekV4Flash
            let controller = RunController()
            let ended = expectation(description: "turn ended")
            controller.onTurnEnded = { _ in ended.fulfill() }
            controller.start(
                runner: MockDecodeRunner(entry: entry, reviewSeconds: 1),
                request: RunRequest(
                    model: entry.id,
                    artifact: ArtifactReference(root: URL(fileURLWithPath: "/tmp/model")),
                    prompt: .tokenIDs([17529]),
                    memoryBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
                    maximumNewTokens: min(3, DeepSeekV4ProductMemoryBudget.maximumNewTokens),
                    workingDirectory: URL(fileURLWithPath: NSTemporaryDirectory())),
                linkCeiling: 0.92e9,
                layerCount: 43, storedBytesPerLayer: 1_200_000_000,
                widenedBytesPerLayer: 2_400_000_000)
            await fulfillment(of: [ended], timeout: 60)
            let snapshot = controller.snapshot
            controller.clear()
            return snapshot
        }

        /// The same sample at another stage and on another basis. Every other
        /// field is carried across, so the four panels differ only in what the
        /// row has to decide.
        private func restaged(
            _ sample: RunTelemetry, stage: RunGenerationStage, entryFootprintBytes: UInt64?
        ) -> RunTelemetry {
            // The peak is on the budget's basis; the footprint is the absolute
            // process figure, which on the added basis includes the entry floor.
            let peak = sample.declaredBudgetBytes / 100 * 85
            let footprint: UInt64 =
                stage == .terminal
                ? 161_000_000 : (entryFootprintBytes ?? 0) + sample.declaredBudgetBytes / 100 * 78
            return RunTelemetry(
                at: sample.at, elapsed: sample.elapsed, phase: sample.phase,
                tokensPerSecond: sample.tokensPerSecond, generationStage: stage,
                bytes: sample.bytes, bytesPerSecond: sample.bytesPerSecond,
                completedTokenBytes: sample.completedTokenBytes,
                prefillBytes: sample.prefillBytes, bytesPerToken: sample.bytesPerToken,
                prefillSeconds: sample.prefillSeconds, decodeSeconds: sample.decodeSeconds,
                decodeTokensCompleted: sample.decodeTokensCompleted,
                byteAccountingReported: sample.byteAccountingReported,
                declaredBudgetBytes: sample.declaredBudgetBytes,
                footprintBytes: footprint, peakFootprintBytes: peak,
                entryFootprintBytes: entryFootprintBytes,
                residentBytes: footprint, availableBytes: sample.availableBytes,
                mlxActiveBytes: sample.mlxActiveBytes, mlxCacheBytes: sample.mlxCacheBytes,
                mlxPeakBytes: sample.mlxPeakBytes, thermalState: sample.thermalState,
                lowPowerMode: sample.lowPowerMode, batteryLevel: sample.batteryLevel,
                instrumentation: sample.instrumentation)
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
                .appendingPathComponent("minirun-panel-states", isDirectory: true)
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

        func testThePanelDrawsLiveAndFinishedOnBothBases() async throws {
            let finished = await finishedSnapshot()
            let sample = try XCTUnwrap(finished.telemetry)
            let states: [(String, RunGenerationStage, UInt64?)] = [
                ("live-absolute", .decode, nil),
                ("live-added", .decode, 10_800_000_000),
                ("finished-absolute", .terminal, nil),
                ("finished-added", .terminal, 10_800_000_000),
            ]
            var heights: [String: CGFloat] = [:]
            for (name, stage, entry) in states {
                var snapshot = finished
                snapshot.telemetry = restaged(sample, stage: stage, entryFootprintBytes: entry)
                for (width, device) in [(CGFloat(372), "mac"), (CGFloat(390), "phone")] {
                    for scheme in [ColorScheme.light, ColorScheme.dark] {
                        let drawn = try render(
                            InstrumentPanelView(snapshot: snapshot).sections,
                            width: width, scheme: scheme)
                        let url = try write(drawn, named: "panel-\(name)-\(device)-\(scheme)")
                        XCTAssertGreaterThan(drawn.size.height, 400, "\(name) \(url.path)")
                        heights["\(name)-\(device)-\(scheme)"] = drawn.size.height
                    }
                }
            }
            XCTAssertEqual(heights.count, 16)
        }

        func testAFinishedReplyDrawsItsFooter() throws {
            var reply = Message(
                role: .assistant, text: "The capital of Austria is **Vienna**.")
            reply.telemetry = MessageTelemetry(
                wallSeconds: 265, tokensPerSecond: 0.21, bytesPerToken: 4_510_000_000,
                peakFootprintBytes: 14_700_000_000, declaredBudgetBytes: 14_900_000_000,
                budgetRespected: true,
                logitsDigest: "afb7ac5e2c63b5e17713636053e65aacde7e8bfa4377b9dd16507bc66b3c3a0d")
            var breached = reply
            breached.telemetry?.peakFootprintBytes = 15_200_000_000
            breached.telemetry?.budgetRespected = false
            let turn = VStack(alignment: .leading, spacing: MRSpace.s5) {
                MessageBlock(message: reply, modelName: "DeepSeek V4.1")
                MessageBlock(message: breached, modelName: "DeepSeek V4.1")
            }
            for (width, device) in [(CGFloat(620), "mac"), (CGFloat(390), "phone")] {
                for scheme in [ColorScheme.light, ColorScheme.dark] {
                    let drawn = try render(turn, width: width, scheme: scheme)
                    let url = try write(drawn, named: "reply-footer-\(device)-\(scheme)")
                    XCTAssertGreaterThan(drawn.size.height, 120, "\(device) \(url.path)")
                }
            }
        }
    }
#endif
