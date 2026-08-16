#if os(macOS)
    import AppKit
    import SwiftUI
    import XCTest

    @testable import MinirunApp

    /// The inspector belongs to the current window. Showing it divides the
    /// workspace; it never grows or shrinks the outer frame.
    @MainActor
    final class ConversationInspectorLayoutTests: XCTestCase {

        func testHostedUnitTestsNeverLaunchTheLiveProductModel() {
            XCTAssertTrue(
                MinirunProcessContext.isUnitTestHost(
                    environment: [
                        "XCTestConfigurationFilePath": "/tmp/tests.xctestconfiguration"
                    ]))
            XCTAssertTrue(
                MinirunProcessContext.isUnitTestHost(
                    environment: ["XCTestBundlePath": "/tmp/MinirunTests.xctest"]))
            XCTAssertFalse(MinirunProcessContext.isUnitTestHost(environment: [:]))
        }

        func testWorkspaceStartsAtAPracticalSizeWithInspectorClosed() {
            let model = AppModel.preview()

            XCTAssertNil(model.conversationPanel)
            XCTAssertEqual(ConversationWorkspaceLayout.defaultWindowWidth, 1_120)
            XCTAssertEqual(ConversationWorkspaceLayout.defaultWindowHeight, 720)
            XCTAssertEqual(ConversationWorkspaceLayout.minimumWindowWidth, 960)
            XCTAssertEqual(
                ConversationWorkspaceLayout.sidebarWidth, 260,
                "opening the right information panel must not resize Chats")
            XCTAssertGreaterThan(
                ConversationWorkspaceLayout.defaultWindowWidth,
                ConversationWorkspaceLayout.minimumWindowWidth)
            XCTAssertGreaterThan(
                ConversationWorkspaceLayout.defaultWindowHeight,
                ConversationWorkspaceLayout.minimumWindowHeight)
        }

        func testInspectorHasADraggableWidthRange() {
            XCTAssertLessThan(
                ConversationWorkspaceLayout.inspectorMinimumWidth,
                ConversationWorkspaceLayout.inspectorIdealWidth)
            XCTAssertLessThan(
                ConversationWorkspaceLayout.inspectorIdealWidth,
                ConversationWorkspaceLayout.inspectorMaximumWidth)
            XCTAssertGreaterThanOrEqual(
                ConversationWorkspaceLayout.inspectorMinimumWidth, 300)
            XCTAssertEqual(
                ConversationWorkspaceLayout.clampedInspectorWidth(100),
                ConversationWorkspaceLayout.inspectorMinimumWidth)
            XCTAssertEqual(
                ConversationWorkspaceLayout.clampedInspectorWidth(420), 420)
            XCTAssertEqual(
                ConversationWorkspaceLayout.clampedInspectorWidth(900),
                ConversationWorkspaceLayout.inspectorMaximumWidth)
            XCTAssertEqual(
                ConversationWorkspaceLayout.maximumInspectorWidth(
                    availableWindowWidth: ConversationWorkspaceLayout.minimumWindowWidth),
                319)
            XCTAssertEqual(
                ConversationWorkspaceLayout.clampedInspectorWidth(
                    900,
                    availableWindowWidth: ConversationWorkspaceLayout.minimumWindowWidth),
                319)
            XCTAssertGreaterThanOrEqual(
                ConversationWorkspaceLayout.minimumWindowWidth
                    - ConversationWorkspaceLayout.sidebarWidth
                    - ConversationWorkspaceLayout.sidebarDividerWidth
                    - ConversationWorkspaceLayout.maximumInspectorWidth(
                        availableWindowWidth: ConversationWorkspaceLayout.minimumWindowWidth),
                ConversationWorkspaceLayout.minimumConversationWidth)
        }

        func testInspectorBoundaryStartsBelowTheSharedHeader() {
            XCTAssertEqual(ConversationInspectorBoundaryLayout.lineWidth, 1)
            XCTAssertGreaterThan(
                ConversationInspectorBoundaryLayout.resizeTargetWidth,
                ConversationInspectorBoundaryLayout.lineWidth)
            XCTAssertEqual(
                ConversationInspectorBoundaryLayout.topInset,
                MacColumnHeaderLayout.height + 1,
                "the visible detail/panel boundary must begin below the title row and its divider")
        }

        func testPanelToggleUsesAControlConsoleSymbolInsteadOfInfoOrSidebar() {
            XCTAssertEqual(
                ConversationPanelToggleStyle.symbolName, "slider.horizontal.3")
            XCTAssertNotEqual(ConversationPanelToggleStyle.symbolName, "info.circle")
            XCTAssertNotEqual(ConversationPanelToggleStyle.symbolName, "sidebar.trailing")
            XCTAssertNotNil(
                NSImage(
                    systemSymbolName: ConversationPanelToggleStyle.symbolName,
                    accessibilityDescription: nil))
            XCTAssertTrue(ConversationPanelToggleStyle.showHelp.contains("controls"))
            XCTAssertTrue(ConversationPanelToggleStyle.showHelp.contains("instruments"))
        }

        func testChatAndSettingsShareOneLeftAlignedSidebarHeaderGeometry() {
            XCTAssertEqual(MacSidebarHeaderLayout.windowControlClearanceHeight, 28)
            XCTAssertEqual(MacSidebarHeaderLayout.titleRowHeight, 44)
            XCTAssertEqual(MacSidebarHeaderLayout.horizontalPadding, MRSpace.s4)
            XCTAssertEqual(MacSidebarHeaderLayout.totalHeight, 72)
            XCTAssertEqual(MacColumnHeaderLayout.height, 52)
            XCTAssertLessThan(
                MacSidebarHeaderLayout.horizontalPadding,
                76,
                "sidebar titles must align with navigation content, not indent around traffic lights")
        }

        #if DEBUG
            func testVisualReviewLaunchPolicyIsAbsentValidOrFailClosed() {
                XCTAssertEqual(
                    MinirunInitialModel.resolve(arguments: ["Minirun"]), .live)
                XCTAssertEqual(
                    MinirunInitialModel.resolve(
                        arguments: ["Minirun", "-MinirunReviewPreview", "yes"]),
                    .launchRefused)
                XCTAssertEqual(
                    MinirunInitialModel.resolve(
                        arguments: ["Minirun", "-MinirunReviewPreview", "YES"]),
                    .visualReviewPreview)
                XCTAssertEqual(
                    MinirunInitialModel.resolve(
                        arguments: ["Minirun", "-MinirunReviewPreview"]),
                    .launchRefused)
                XCTAssertEqual(
                    MinirunInitialModel.resolve(
                        arguments: ["Minirun", "-MinirunReviewPreviewTypo", "YES"]),
                    .launchRefused)
                XCTAssertEqual(
                    MinirunInitialModel.resolve(
                        arguments: [
                            "Minirun", "-MinirunReviewPreview", "YES",
                            "-MinirunReviewPreview", "YES",
                        ]),
                    .launchRefused)
            }

            func testReleasePolicyRefusesEvenAnExactlySpelledReviewRequest() {
                XCTAssertEqual(
                    MinirunInitialModel.resolve(
                        arguments: ["Minirun", "-MinirunReviewPreview", "YES"],
                        visualReviewAvailable: false),
                    .launchRefused)
            }

            func testVisualReviewModelIsRunnableEphemeralAndPanelAddressable() throws {
                let suiteName = "minirun.review-tests.\(UUID().uuidString)"
                let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
                defer { defaults.removePersistentDomain(forName: suiteName) }
                defaults.set(Data([1, 2, 3]), forKey: "minirun.newChatDefaults")
                defaults.set(Data([4, 5, 6]), forKey: "minirun.kit.location.saved")
                let before = defaults.persistentDomain(forName: suiteName) as NSDictionary?
                let arguments = [
                    "Minirun",
                    "-MinirunReviewPreview", "YES",
                    "-MinirunPanel", "Instruments",
                    // This is deliberately ignored in visual-review mode. A
                    // layout fixture cannot mint a real storage grant.
                    "-MinirunAddLocation", "/Volumes/SHOULD-NOT-BE-REGISTERED",
                ]
                let model = MinirunInitialModel.resolve(arguments: arguments)
                    .makeModel(arguments: arguments, userDefaults: defaults)

                XCTAssertFalse(model.conversationStore.isPersistent)
                XCTAssertFalse(model.persistsDefaults)
                XCTAssertNil(model.launchRefusal)
                XCTAssertNotNil(model.visualReviewNotice)
                XCTAssertTrue(model.visualReviewNotice?.contains("sample data") == true)
                XCTAssertEqual(model.conversationPanel, .instruments)
                XCTAssertEqual(model.modelAvailability(for: .kimiK3), .available)
                XCTAssertNotNil(model.selectedConversation)
                XCTAssertTrue(model.installed.locationKeys.isEmpty)
                XCTAssertTrue(model.history.isEmpty)
                XCTAssertTrue(model.measuredPoints(for: .kimiK3).isEmpty)
                XCTAssertEqual(
                    defaults.persistentDomain(forName: suiteName) as NSDictionary?, before)
            }

            func testInvalidReviewRequestDoesNotOpenOrMutateProductState() throws {
                let suiteName = "minirun.refusal-tests.\(UUID().uuidString)"
                let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
                defer { defaults.removePersistentDomain(forName: suiteName) }
                defaults.set(Data([1, 2, 3]), forKey: "minirun.newChatDefaults")
                defaults.set(["artifact-location.saved"], forKey: "minirun.kit.ledger.__keys")
                defaults.set(Data([4, 5, 6]), forKey: "minirun.kit.location.saved")
                let before = defaults.persistentDomain(forName: suiteName) as NSDictionary?
                let arguments = ["Minirun", "-MinirunReviewPreview", "yes"]
                let launch = MinirunInitialModel.resolve(arguments: arguments)
                let model = launch.makeModel(arguments: arguments, userDefaults: defaults)

                XCTAssertEqual(launch, .launchRefused)
                XCTAssertNotNil(model.launchRefusal)
                XCTAssertFalse(model.persistsDefaults)
                XCTAssertFalse(model.conversationStore.isPersistent)
                XCTAssertFalse(
                    FileManager.default.fileExists(atPath: model.conversationStore.directory.path))
                XCTAssertTrue(model.conversations.isEmpty)
                XCTAssertTrue(model.installed.locationKeys.isEmpty)
                XCTAssertEqual(
                    defaults.persistentDomain(forName: suiteName) as NSDictionary?, before)
            }
        #endif

        func testLegacyWindowRestorationMigratesOnceThenPreservesManualFrames() throws {
            let suiteName = "minirun.window-tests.\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
            defer { defaults.removePersistentDomain(forName: suiteName) }
            let oldFrame =
                "NSWindow Frame SwiftUI.ModifiedContent<MinirunApp.RootView>-1-AppWindow-1"
            let oldSplit =
                "NSSplitView Subview Frames SwiftUI.ModifiedContent<MinirunApp.RootView>"
                + "-1-AppWindow-1, SidebarNavigationSplitView"
            let unrelated = "NSWindow Frame NSNavPanelAutosaveName"
            defaults.set("0 179 1920 721", forKey: oldFrame)
            defaults.set(["260", "1659"], forKey: oldSplit)
            defaults.set("880 747 800 448", forKey: unrelated)
            defaults.set(true, forKey: "minirun.window.panelAttached")
            defaults.set(1_200, forKey: "minirun.window.widthWithoutPanel")

            let first = MinirunWindowRestorationMigration.applyIfNeeded(
                for: .live, defaults: defaults)
            guard case .migrated(let removed) = first else {
                return XCTFail("expected a first-run restoration migration")
            }
            XCTAssertTrue(removed.contains(oldFrame))
            XCTAssertTrue(removed.contains(oldSplit))
            XCTAssertNil(defaults.object(forKey: oldFrame))
            XCTAssertNil(defaults.object(forKey: oldSplit))
            XCTAssertNil(defaults.object(forKey: "minirun.window.panelAttached"))
            XCTAssertNil(defaults.object(forKey: "minirun.window.widthWithoutPanel"))
            XCTAssertEqual(defaults.string(forKey: unrelated), "880 747 800 448")

            // AppKit writes the operator's next manual resize under the same
            // scene key. The version marker makes every later launch preserve it.
            defaults.set("180 180 1180 740", forKey: oldFrame)
            XCTAssertEqual(
                MinirunWindowRestorationMigration.applyIfNeeded(
                    for: .live, defaults: defaults),
                .alreadyCurrent)
            XCTAssertEqual(defaults.string(forKey: oldFrame), "180 180 1180 740")
        }

        func testReviewAndRefusalLaunchesCannotRunTheWindowDefaultsMigration() throws {
            let suiteName = "minirun.window-review-tests.\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
            defer { defaults.removePersistentDomain(forName: suiteName) }
            let oldFrame =
                "NSWindow Frame SwiftUI.ModifiedContent<MinirunApp.RootView>-1-AppWindow-1"
            defaults.set("0 179 1920 721", forKey: oldFrame)
            let before = defaults.persistentDomain(forName: suiteName) as NSDictionary?

            XCTAssertEqual(
                MinirunWindowRestorationMigration.applyIfNeeded(
                    for: .launchRefused, defaults: defaults),
                .notPermitted)
            #if DEBUG
                XCTAssertEqual(
                    MinirunWindowRestorationMigration.applyIfNeeded(
                        for: .visualReviewPreview, defaults: defaults),
                    .notPermitted)
            #endif
            XCTAssertEqual(
                defaults.persistentDomain(forName: suiteName) as NSDictionary?, before)
        }

        func testNinetyThreeLayerLadderFitsTheMinimumInspectorContentWidth() {
            let contentWidth = ConversationWorkspaceLayout.inspectorMinimumWidth
                - MRSpace.s4 * 2
            let metrics = LayerLadderGeometry.metrics(
                availableWidth: contentWidth, cellCount: 93)

            XCTAssertGreaterThan(metrics.cellWidth, 0)
            XCTAssertLessThanOrEqual(
                metrics.occupiedWidth(cellCount: 93), contentWidth + 0.001)
            XCTAssertLessThan(
                metrics.occupiedWidth(cellCount: 93),
                93 * 3 + 92,
                "the minimum-width inspector must not inherit the old fixed 371 pt ladder")
        }

        func testOpeningAndSwitchingInspectorPreservesOuterWindowFrame() async {
            let model = AppModel.preview()
            var inspectorSizes: [CGSize] = []
            ConversationInspectorLayoutObservation.onLayout = { inspectorSizes.append($0) }
            defer { ConversationInspectorLayoutObservation.onLayout = nil }
            let window = NSWindow(
                contentRect: NSRect(
                    x: 180, y: 180,
                    width: ConversationWorkspaceLayout.minimumWindowWidth,
                    height: ConversationWorkspaceLayout.defaultWindowHeight),
                styleMask: [.titled, .resizable], backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.contentViewController = NSHostingController(
                rootView: RootView()
                    .environment(model)
                    .frame(
                        minWidth: ConversationWorkspaceLayout.minimumWindowWidth,
                        minHeight: ConversationWorkspaceLayout.minimumWindowHeight))
            // Reproduce the WindowGroup's concrete size and minimum rather
            // than measuring the hosting view's intrinsic placeholder.
            window.setContentSize(
                NSSize(
                    width: ConversationWorkspaceLayout.minimumWindowWidth,
                    height: ConversationWorkspaceLayout.defaultWindowHeight))
            window.orderFront(nil)
            await settleLayout(in: window)
            let original = window.frame
            XCTAssertTrue(window.isVisible)

            model.conversationPanel = .chatSettings
            await settleLayout(in: window)
            XCTAssertEqual(window.frame, original)
            XCTAssertTrue(
                inspectorSizes.contains {
                    $0.width >= ConversationWorkspaceLayout.inspectorMinimumWidth - 2
                        && $0.width <= ConversationWorkspaceLayout.inspectorMaximumWidth + 2
                },
                "a real inspector pane should be visible inside the minimum-width window")

            model.conversationPanel = .instruments
            await settleLayout(in: window)
            XCTAssertEqual(window.frame, original)

            model.conversationPanel = nil
            await settleLayout(in: window)
            XCTAssertEqual(window.frame, original)

            window.orderOut(nil)
        }

        private func settleLayout(in window: NSWindow) async {
            for _ in 0..<6 { await Task.yield() }
            window.contentView?.layoutSubtreeIfNeeded()
        }

    }
#endif
