#if os(macOS)
    import MinirunKit
    import MinirunRunners
    import SwiftUI
    import XCTest

    @testable import MinirunApp

    /// The section is drawn, not merely computed.
    ///
    /// This project has already shipped a layout bug through a green suite —
    /// the launch layout that hid the entire sidebar — so a section that
    /// silently stops drawing has to fail something. Rendering the panel's own
    /// section list offscreen, with and without a decomposition, is that
    /// something: the two renders must differ, and the PNGs are written where a
    /// reviewer can look at them in both appearances.
    @MainActor
    final class InstrumentPanelRenderTests: XCTestCase {

        private func snapshot(_ entry: CatalogEntry, budget: UInt64, tokens: Int) async
            -> RunSnapshot
        {
            let controller = RunController()
            let ended = expectation(description: "turn ended \(entry.id.rawValue)")
            controller.onTurnEnded = { _ in ended.fulfill() }
            controller.start(
                runner: MockDecodeRunner(entry: entry, reviewSeconds: 1),
                request: RunRequest(
                    model: entry.id,
                    artifact: ArtifactReference(root: URL(fileURLWithPath: "/tmp/model")),
                    prompt: .tokenIDs([17529]),
                    memoryBudgetBytes: budget,
                    maximumNewTokens: tokens,
                    workingDirectory: URL(fileURLWithPath: NSTemporaryDirectory())),
                linkCeiling: 0.92e9,
                layerCount: 93, storedBytesPerLayer: 1_200_000_000,
                widenedBytesPerLayer: 2_400_000_000)
            await fulfillment(of: [ended], timeout: 60)
            let snapshot = controller.snapshot
            controller.clear()
            return snapshot
        }

        private func render(
            _ snapshot: RunSnapshot, scheme: ColorScheme
        ) throws -> NSImage {
            let renderer = ImageRenderer(
                content: InstrumentPanelView(snapshot: snapshot).sections
                    .padding(MRSpace.s4)
                    .frame(width: 372)
                    .background(MRColor.abyss)
                    .environment(\.colorScheme, scheme))
            renderer.scale = 2
            return try XCTUnwrap(renderer.nsImage)
        }

        private func write(_ image: NSImage, named name: String) throws {
            let directory = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-phase-split", isDirectory: true)
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let tiff = try XCTUnwrap(image.tiffRepresentation)
            let png = try XCTUnwrap(
                NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
            try png.write(to: directory.appendingPathComponent("\(name).png"))
        }

        /// K3's eight-term split and V4's sixteen-term one both have to fit the
        /// same narrow inspector column, so both are rendered.
        func testThePhaseSectionDrawsForBothProductRuntimes() async throws {
            let runs = [
                (
                    "k3",
                    await snapshot(
                        CatalogFixtures.kimiK3,
                        budget: CatalogFixtures.k3ProductMinimumBudget,
                        tokens: min(3, K3ProductMemoryBudget.currentPolicy.maximumNewTokens))
                ),
                (
                    "v4",
                    await snapshot(
                        CatalogFixtures.deepseekV4Flash,
                        budget: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
                        tokens: min(3, DeepSeekV4ProductMemoryBudget.maximumNewTokens))
                ),
            ]

            for (name, withPhases) in runs {
                XCTAssertFalse(withPhases.phaseSummaries.isEmpty, name)
                XCTAssertEqual(withPhases.phaseBars.count, 2, "\(name): prefill and decode")
                var withoutPhases = withPhases
                withoutPhases.phaseSummaries = []

                for scheme in [ColorScheme.light, ColorScheme.dark] {
                    let drawn = try render(withPhases, scheme: scheme)
                    let bare = try render(withoutPhases, scheme: scheme)
                    try write(drawn, named: "panel-\(name)-\(scheme)")
                    XCTAssertGreaterThan(drawn.size.width, 0, "\(name) \(scheme)")
                    XCTAssertGreaterThan(
                        drawn.size.height, bare.size.height + 40,
                        "\(name) \(scheme): the section did not take any room on the panel")
                }
            }
        }
    }

    /// The two surfaces this change touched, drawn where a reviewer can look at
    /// them: the footprint row on both bases, and a two-sided chat turn whose
    /// model half is Markdown.
    ///
    /// Both are rendered in both appearances because both carry colour
    /// decisions — the gauge's watermark and refusal, and the inline-code
    /// ground behind an answer — and a picture is the only thing that catches a
    /// row that has silently stopped drawing.
    @MainActor
    final class FootprintAndAnswerRenderTests: XCTestCase {

        /// Writes the PNG beside the other render output **and** attaches it to
        /// the result bundle, so the picture reaches a reviewer who never sees
        /// this machine's temporary directory.
        @discardableResult
        private func write(_ image: NSImage, named name: String) throws -> URL {
            let directory = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-v41-footprint", isDirectory: true)
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

        func testTheFootprintRowDrawsOnBothBases() throws {
            let card = VStack(alignment: .leading, spacing: MRSpace.s5) {
                Text("V4.1 · budget bounds what the run adds").mrLabel(MRColor.secondary)
                BudgetGauge(
                    footprintBytes: 12_900_000_000,
                    peakBytes: 14_700_000_000,
                    declaredBudgetBytes: 14_900_000_000,
                    processBytes: 23_700_000_000,
                    entryFootprintBytes: 10_800_000_000)
                Text("V4 · budget bounds the process").mrLabel(MRColor.secondary)
                BudgetGauge(
                    footprintBytes: 7_150_000_000,
                    peakBytes: 7_150_000_000,
                    declaredBudgetBytes: 8_000_000_000)
                Text("V4.1 · the promise did not hold").mrLabel(MRColor.secondary)
                BudgetGauge(
                    footprintBytes: 15_200_000_000,
                    peakBytes: 15_200_000_000,
                    declaredBudgetBytes: 14_900_000_000,
                    processBytes: 26_000_000_000,
                    entryFootprintBytes: 10_800_000_000,
                    latchedBreach: true)
            }

            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let drawn = try render(card, width: 372, scheme: scheme)
                let url = try write(drawn, named: "footprint-row-\(scheme)")
                XCTAssertGreaterThan(drawn.size.width, 0, "\(scheme) \(url.path)")
                XCTAssertGreaterThan(
                    drawn.size.height, 160,
                    "\(scheme): three rows, each with a caption line, drew almost nothing")
            }
        }

        /// The same row after the run, which is where it went wrong: the last
        /// sample is taken after teardown, so a released working set read as a
        /// run that had added nothing — `0 MB of 14.9 GB · peak 14.7 GB` under
        /// `Process 161 MB · 10.8 GB before this run`.
        ///
        /// Drawn from the telemetry rather than from hand-passed fields, so the
        /// picture is of the mapping the panel uses and not of a second one.
        func testTheFinishedFootprintRowDrawsOnBothBases() throws {
            func sample(
                footprintBytes: UInt64, peakFootprintBytes: UInt64, declaredBudgetBytes: UInt64,
                entryFootprintBytes: UInt64?
            ) -> RunTelemetry {
                RunTelemetry(
                    at: Date(), elapsed: 612, phase: "finished", tokensPerSecond: 0.03,
                    generationStage: .terminal,
                    bytes: ByteAccounting(totalBytesRead: 1), bytesPerSecond: nil,
                    bytesPerToken: nil,
                    declaredBudgetBytes: declaredBudgetBytes,
                    footprintBytes: footprintBytes,
                    peakFootprintBytes: peakFootprintBytes,
                    entryFootprintBytes: entryFootprintBytes,
                    residentBytes: footprintBytes, availableBytes: nil, mlxActiveBytes: 0,
                    mlxCacheBytes: 0, mlxPeakBytes: 0, thermalState: .nominal,
                    lowPowerMode: false, batteryLevel: nil)
            }

            let card = VStack(alignment: .leading, spacing: MRSpace.s5) {
                Text("V4.1 · finished, budget bounded what the run added")
                    .mrLabel(MRColor.secondary)
                BudgetGauge(
                    telemetry: sample(
                        footprintBytes: 161_000_000,
                        peakFootprintBytes: 14_730_000_000,
                        declaredBudgetBytes: 14_900_000_000,
                        entryFootprintBytes: 10_800_000_000),
                    latchedBreach: false)
                Text("V4 · finished, budget bounded the process").mrLabel(MRColor.secondary)
                BudgetGauge(
                    telemetry: sample(
                        footprintBytes: 161_000_000,
                        peakFootprintBytes: 7_150_000_000,
                        declaredBudgetBytes: 8_000_000_000,
                        entryFootprintBytes: nil),
                    latchedBreach: false)
                Text("V4.1 · finished, the promise did not hold").mrLabel(MRColor.secondary)
                BudgetGauge(
                    telemetry: sample(
                        footprintBytes: 161_000_000,
                        peakFootprintBytes: 15_200_000_000,
                        declaredBudgetBytes: 14_900_000_000,
                        entryFootprintBytes: 10_800_000_000),
                    latchedBreach: true)
            }

            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let drawn = try render(card, width: 372, scheme: scheme)
                let url = try write(drawn, named: "footprint-row-finished-\(scheme)")
                XCTAssertGreaterThan(drawn.size.width, 0, "\(scheme) \(url.path)")
                XCTAssertGreaterThan(
                    drawn.size.height, 160,
                    "\(scheme): three finished rows, each with a caption, drew almost nothing")
            }
        }

        func testAChatTurnDrawsItsMarkdown() throws {
            let answer = """
                The capital of Austria is **Vienna** (German: *Wien*).

                A few notes:

                - It has been the capital since **1920**
                - The endonym is `Wien`

                ```swift
                let capital = "Vienna"
                ```
                """
            let turn = VStack(alignment: .leading, spacing: MRSpace.s5) {
                MessageBlock(
                    message: Message(
                        role: .user,
                        text: "What is the capital of Austria? Keep the **asterisks** I typed."),
                    modelName: "DeepSeek V4.1")
                MessageBlock(
                    message: Message(role: .assistant, text: answer),
                    modelName: "DeepSeek V4.1")
                StreamingText(
                    text: "The capital of Austria is **Vien", caretActive: true)
            }

            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let drawn = try render(turn, width: 620, scheme: scheme)
                let url = try write(drawn, named: "chat-turn-\(scheme)")
                XCTAssertGreaterThan(drawn.size.width, 0, "\(scheme) \(url.path)")
                XCTAssertGreaterThan(
                    drawn.size.height, 240,
                    "\(scheme): the turn drew almost nothing")

                // The same source through the old path. The two pictures must
                // not be the same picture — that is the whole of the defect,
                // and pixels are the only place it showed.
                let flat = try render(
                    Text(answer).font(MRType.body).fixedSize(horizontal: false, vertical: true),
                    width: 620, scheme: scheme)
                let drawnAnswer = try render(
                    AnswerText(source: answer), width: 620, scheme: scheme)
                XCTAssertNotEqual(
                    drawnAnswer.tiffRepresentation, flat.tiffRepresentation,
                    "\(scheme): the answer rendered as one flat run of source text")
            }
        }
    }
#endif
