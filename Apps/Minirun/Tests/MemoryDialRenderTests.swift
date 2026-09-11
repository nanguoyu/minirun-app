#if os(macOS)
    import MinirunKit
    import MinirunRunners
    import SwiftUI
    import XCTest

    @testable import MinirunApp

    /// The dial is drawn, not merely computed.
    ///
    /// `BudgetPlanTests` proves the arithmetic; this proves that Chat settings
    /// puts it on screen for DeepSeek V4.1, where until now the same control
    /// drew three presets that were all the floor. Rendering the dial offscreen
    /// at each preset, in both appearances, is what would fail if the streaming
    /// branch ever stopped drawing its pinned tier — the failure mode this
    /// project has already shipped once, through a green suite, with the launch
    /// layout that hid the whole sidebar.
    ///
    /// The PNGs are written where a reviewer can look at them.
    @MainActor
    final class MemoryDialRenderTests: XCTestCase {

        private let entry = CatalogFixtures.deepseekV41Flash

        private func plan(budget: UInt64) -> BudgetPlan {
            BudgetPlan(
                model: .deepseekV41Flash, modelName: entry.descriptor.displayName,
                profile: entry.memory, budgetBytes: budget,
                maximumNewTokens: DeepSeekV41ProductMemoryBudget.maximumNewTokens,
                deviceCeilingBytes: 34_400_000_000, readAheadDepth: 1)
        }

        /// Chat settings' own call: the compact presentation, the conversation
        /// role, and no read-ahead control, because a bounded streaming run has
        /// no deterministic read-ahead to state.
        private func render(_ plan: BudgetPlan, scheme: ColorScheme) throws -> NSImage {
            let renderer = ImageRenderer(
                content: MemoryDialView(
                    plan: plan, entry: entry, role: .conversation,
                    presentation: .compact, readAheadDepth: nil, onChange: { _ in }
                )
                .padding(MRSpace.s4)
                .frame(width: 372)
                .background(MRColor.abyss)
                .environment(\.colorScheme, scheme))
            renderer.scale = 2
            return try XCTUnwrap(renderer.nsImage)
        }

        private func write(_ image: NSImage, named name: String) throws -> URL {
            let directory = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-v41-dial", isDirectory: true)
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let tiff = try XCTUnwrap(image.tiffRepresentation)
            let png = try XCTUnwrap(
                NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
            let url = directory.appendingPathComponent("\(name).png")
            try png.write(to: url)
            return url
        }

        func testTheV41DialDrawsAtEveryPreset() throws {
            let base = plan(budget: DeepSeekV41ProductMemoryBudget.minimumBudgetBytes)
            var images = [String: NSImage]()

            for preset in BudgetPlan.Preset.allCases {
                let budget = base.budget(for: preset)
                let candidate = base.with(budgetBytes: budget)
                XCTAssertTrue(candidate.isRunnable, "\(preset.rawValue) is not runnable")
                XCTAssertTrue(
                    candidate.tiersAccountForBudget,
                    "\(preset.rawValue) draws tiers that do not sum to its budget")

                for scheme in [ColorScheme.dark, .light] {
                    let image = try render(candidate, scheme: scheme)
                    XCTAssertGreaterThan(image.size.width, 0)
                    XCTAssertGreaterThan(image.size.height, 0)
                    let appearance = scheme == .dark ? "dark" : "light"
                    let name = "\(preset.rawValue.lowercased())-\(appearance)"
                    images[name] = image
                    print("[v41-dial] wrote \(try write(image, named: name).path)")
                }
            }

            // Floor and Balanced are different residencies — nothing held
            // against all forty blocks and the head — so they cannot draw the
            // same picture. They did before the dial reached V4.1, because they
            // were the same budget.
            let floor = try XCTUnwrap(images["floor-dark"]?.tiffRepresentation)
            let balanced = try XCTUnwrap(images["balanced-dark"]?.tiffRepresentation)
            XCTAssertNotEqual(floor, balanced)
            XCTAssertNotEqual(
                base.budget(for: .floor), base.budget(for: .balanced),
                "three presets that are one budget are not a dial")
        }

        /// **The dial a phone draws, drawn on the Mac.**
        ///
        /// The iPhone's Floor is 1.9 GB and its other two presets are the first
        /// rung of a ladder no phone can reach, so this renders all three at the
        /// iPhone policy's prices and at a ceiling of the shape
        /// `os_proc_available_memory()` reports. What it would have caught is
        /// the thing that shipped: a phone whose Floor was the Mac's 3.4 GB,
        /// drawn as a perfectly ordinary dial, on a build whose own runner
        /// floor was 1.9 GB.
        ///
        /// Floor is the only runnable position, so it is the only one rendered
        /// as a plan; the other two are checked as budgets and deficits, which
        /// is what the disabled presets draw from.
        func testTheV41DialOnAPhoneDrawsThePhonesFloor() throws {
            let iOS = DeepSeekV41ProductMemoryBudget.iOSProductPolicy
            let phone = BudgetPlan(
                model: .deepseekV41Flash, modelName: entry.descriptor.displayName,
                profile: CatalogFixtures.deepSeekV41Memory(policy: iOS),
                budgetBytes: iOS.minimumBudgetBytes,
                maximumNewTokens: iOS.maximumNewTokens,
                deviceCeilingBytes: 5_000_000_000, readAheadDepth: 1)

            XCTAssertEqual(phone.budget(for: .floor), 1_900_000_000)
            XCTAssertTrue(phone.isRunnable)
            XCTAssertTrue(phone.tiersAccountForBudget)
            XCTAssertEqual(phone.pinnedLayerCount, 0)
            for preset in [BudgetPlan.Preset.balanced, .generous] {
                XCTAssertNotNil(
                    phone.presetDeficit(preset),
                    "\(preset.rawValue) is out of reach on a phone and must say so")
            }

            for scheme in [ColorScheme.dark, .light] {
                let image = try render(phone, scheme: scheme)
                XCTAssertGreaterThan(image.size.width, 0)
                XCTAssertGreaterThan(image.size.height, 0)
                let appearance = scheme == .dark ? "dark" : "light"
                print(
                    "[v41-dial] wrote "
                        + (try write(image, named: "iphone-floor-\(appearance)").path))
            }
        }

        /// The sentence under the bar names the model that is running. It used
        /// to say "V4" unconditionally, which was wrong for the other half of
        /// the models that draw it.
        func testTheStreamingSentenceNamesThisModel() throws {
            let balanced = plan(budget: 14_860_526_016)
            let text = try XCTUnwrap(
                ImageRenderer(content: MemoryDialView(
                    plan: balanced, entry: entry, role: .conversation,
                    presentation: .compact, readAheadDepth: nil, onChange: { _ in })
                ).nsImage)
            XCTAssertGreaterThan(text.size.height, 0)
            XCTAssertEqual(balanced.modelName, "DeepSeek V4.1 Flash")
            XCTAssertEqual(balanced.residentUnitsLabel, "40 of 40 + output head")
        }
    }
#endif
