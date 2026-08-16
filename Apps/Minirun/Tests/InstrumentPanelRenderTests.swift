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
#endif
