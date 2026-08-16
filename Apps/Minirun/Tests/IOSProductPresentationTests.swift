import Foundation
import MinirunRunners
import SwiftUI
import XCTest

#if canImport(UIKit)
    import UIKit
#endif

@testable import MinirunApp

/// Static product-presentation contracts for code that cannot be exercised by
/// the macOS-hosted app test target. These assertions keep compact iOS layout
/// decisions visible when the Xcode project is regenerated from project.yml.
final class IOSProductPresentationTests: XCTestCase {
    func testDestinationUsesACompactSheetAndDoesNotOfferIgnoredPolicies() throws {
        let source = try screen("DestinationPicker.swift")

        XCTAssertTrue(source.contains("#if os(macOS)\n            .frame(minWidth: 560"))
        XCTAssertFalse(source.contains("PreconditionCard("))
        XCTAssertFalse(source.contains("powerAcknowledged"))
        XCTAssertFalse(source.contains("allow metered networks"))
        XCTAssertFalse(source.contains("require external power"))
        XCTAssertFalse(source.contains("ASM2464"))
        XCTAssertTrue(source.contains("Some external drives may need their own power"))
    }

    func testFindModelsUsesNativeIOSDiscoveryControls() throws {
        let source = try screen("ModelCatalogView.swift")

        XCTAssertTrue(source.contains(".searchable("))
        XCTAssertTrue(source.contains("text: $query"))
        XCTAssertTrue(source.contains("ToolbarItem(placement: .cancellationAction)"))
        XCTAssertTrue(source.contains("ToolbarItem(placement: .primaryAction)"))
        XCTAssertTrue(source.contains(".contentShape(Rectangle())"))
    }

    func testModelAndDownloadEvidenceUseCompactCardsOnIOS() throws {
        let modelDetail = try screen("ModelDetailView.swift")
        let downloadDetail = try screen("DownloadDetailView.swift")

        XCTAssertTrue(modelDetail.contains("compactHistoryCard"))
        XCTAssertTrue(modelDetail.contains("desktopHistoryTable(records)"))
        XCTAssertTrue(downloadDetail.contains("compactFileCard"))
        XCTAssertTrue(downloadDetail.contains("desktopFileRow(file)"))
    }

    func testRemovingStorageAccessExplainsTheImpactBeforeActing() throws {
        let source = try screen("StorageSettingsView.swift")

        XCTAssertTrue(source.contains("Stop using this folder?"))
        XCTAssertTrue(source.contains("No files will be deleted"))
        XCTAssertTrue(source.contains("pendingLocationRemoval"))
    }

    func testThemeAndVersionFollowThePlatformAndBuildSettings() throws {
        let settings = try screen("SettingsView.swift")
        let iosPlist = try projectFile("Info-iOS.plist")
        let macOSPlist = try projectFile("Info-macOS.plist")
        let project = try projectFile("project.yml")

        XCTAssertTrue(settings.contains("dynamicTypeSize.isAccessibilitySize"))
        XCTAssertTrue(settings.contains("horizontalSizeClass == .compact"))
        for plist in [iosPlist, macOSPlist] {
            XCTAssertTrue(plist.contains("$(MARKETING_VERSION)"))
            XCTAssertTrue(plist.contains("$(CURRENT_PROJECT_VERSION)"))
        }
        // The numbers themselves move every release; the shape does not: a
        // marketing version like 0.2 and a ten-digit YYYYMMDDNN build, stamped
        // identically on both platform targets.
        let marketing = try XCTUnwrap(
            project.firstMatch(of: #"MARKETING_VERSION: "(\d+\.\d+)""#))
        let build = try XCTUnwrap(
            project.firstMatch(of: #"CURRENT_PROJECT_VERSION: "(\d{10})""#))
        XCTAssertEqual(
            project.components(separatedBy: "MARKETING_VERSION: \"\(marketing)\"").count, 3)
        XCTAssertEqual(
            project.components(separatedBy: "CURRENT_PROJECT_VERSION: \"\(build)\"").count, 3)
        XCTAssertFalse(project.contains("CURRENT_PROJECT_VERSION: \"1\""))
    }

    func testPreferenceRowsStackOnlyWhenWidthOrTypeRequiresIt() {
        XCTAssertTrue(
            PreferenceRowLayoutPolicy.usesVerticalLayout(
                horizontalSizeClass: .compact,
                dynamicTypeSize: .large))
        XCTAssertFalse(
            PreferenceRowLayoutPolicy.usesVerticalLayout(
                horizontalSizeClass: .regular,
                dynamicTypeSize: .large))
        XCTAssertTrue(
            PreferenceRowLayoutPolicy.usesVerticalLayout(
                horizontalSizeClass: .regular,
                dynamicTypeSize: .accessibility1))
    }

    func testEmptyChatsUsesOneClearPrimaryActionForEachReadinessState() {
        let storage = EmptyChatsPresentation(
            readiness: .needsStorage,
            modelName: { _ in "Kimi K3" })
        XCTAssertEqual(storage.title, "Choose a model folder")
        XCTAssertEqual(storage.primary, EmptyChatsButton(title: "Choose Storage", action: .storage))
        XCTAssertEqual(storage.secondary, EmptyChatsButton(title: "Find Models", action: .models))

        let verification = EmptyChatsPresentation(
            readiness: .needsVerification(model: .kimiK3),
            modelName: { _ in "Kimi K3" })
        XCTAssertEqual(verification.title, "Verify Kimi K3")
        XCTAssertEqual(verification.primary?.action, .models)
        XCTAssertFalse(verification.showsProgress)

        let ready = EmptyChatsPresentation(
            readiness: .ready(model: .kimiK3),
            modelName: { _ in "Kimi K3" })
        XCTAssertEqual(ready.primary, EmptyChatsButton(title: "New Chat", action: .newChat))
        XCTAssertNil(ready.secondary)

        let loading = EmptyChatsPresentation(
            readiness: .loadingCatalog,
            modelName: { _ in "Kimi K3" })
        XCTAssertTrue(loading.showsProgress)
        XCTAssertNil(loading.primary)
    }

    func testAboutUsesFullWidthProductLinkRowsInsteadOfStandaloneButtons() throws {
        let source = try screen("AboutView.swift")

        XCTAssertEqual(AboutPresentation.linksTitle, "Links")
        XCTAssertEqual(AboutPresentation.websiteTitle, "Website")
        XCTAssertEqual(AboutPresentation.privacyTitle, "Privacy Policy")
        XCTAssertEqual(AboutPresentation.termsTitle, "Terms of Service")
        XCTAssertTrue(source.contains("SectionHeader(title: AboutPresentation.linksTitle)"))
        XCTAssertTrue(source.contains("linkRow("))
        XCTAssertTrue(source.contains(".contentShape(Rectangle())"))
        XCTAssertTrue(source.contains(".buttonStyle(.plain)"))
        XCTAssertFalse(source.contains(".buttonStyle(.bordered)"))
    }

    /// ADR 0011 admits two output tokens on iPhone and sixty-four on the Mac.
    /// The review fixture used to compile in `64`, so a phone build advertised
    /// a ceiling the product policy refuses. The ceiling shown beside the
    /// response-limit field is the runner's own capability, so the fixture has
    /// to report the platform's.
    func testResponseCeilingComesFromTheRunnerCapabilityAndFollowsThePlatformTier() throws {
        let entry = try XCTUnwrap(
            CatalogFixtures.productPreview.first { $0.id == .kimiK3 })
        let runner = MockDecodeRunner(entry: entry)

        XCTAssertEqual(
            runner.capabilities.maximumNewTokens,
            K3ProductMemoryBudget.currentPolicy.maximumNewTokens)
        XCTAssertEqual(
            runner.capabilities.minimumBudgetBytes,
            K3ProductMemoryBudget.minimumBudgetBytes)

        #if os(iOS)
            XCTAssertEqual(runner.capabilities.maximumNewTokens, 2)
            XCTAssertTrue(K3ProductMemoryBudget.currentPolicy.isExperimental)
        #else
            XCTAssertEqual(runner.capabilities.maximumNewTokens, 64)
            XCTAssertFalse(K3ProductMemoryBudget.currentPolicy.isExperimental)
        #endif

        // The screen reads the ceiling from capabilities rather than restating
        // a number, which is what makes the value above reach the label.
        let source = try screen("ConversationSettingsView.swift")
        XCTAssertTrue(source.contains("capabilities?.maximumNewTokens ?? 1"))
        XCTAssertTrue(source.contains("Text(\"1 to \\(ceiling) tokens\")"))
        XCTAssertTrue(source.contains("up to \\(ceiling) output tokens"))
    }

    #if os(iOS)
        /// The composer's two states are one slot, and on iPhone that slot is a
        /// compact circular control rather than a labelled desktop button. This
        /// renders all three states offscreen so a visual review has something
        /// to look at without driving a live conversation, and it fails if a
        /// state stops drawing or the send and stop states become identical.
        @MainActor
        func testPhoneComposerRendersItsSendStopAndStoppingStates() throws {
            let directory = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-composer-states", isDirectory: true)
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)

            var images: [String: UIImage] = [:]
            for state in ComposerRenderState.allCases {
                let renderer = ImageRenderer(
                    content: state.view.frame(width: 390).background(MRColor.abyss))
                renderer.scale = 3
                let image = try XCTUnwrap(renderer.uiImage, state.rawValue)
                images[state.rawValue] = image
                if let data = image.pngData() {
                    try? data.write(
                        to: directory.appendingPathComponent("\(state.rawValue).png"))
                }
            }

            XCTAssertEqual(images.count, ComposerRenderState.allCases.count)
            // A 44-point control plus a 38-point field, in one band.
            for (name, image) in images {
                XCTAssertGreaterThanOrEqual(image.size.height, 44, name)
                XCTAssertLessThanOrEqual(image.size.height, 80, name)
            }
            XCTAssertNotEqual(
                images["send-enabled"]?.pngData(),
                images["stop-running"]?.pngData(),
                "the running state must not draw the same control as the idle one")
        }

        private enum ComposerRenderState: String, CaseIterable {
            case sendDisabled = "send-disabled"
            case sendEnabled = "send-enabled"
            case stopRunning = "stop-running"
            case stopStopping = "stop-stopping"

            @MainActor @ViewBuilder var view: some View {
                switch self {
                case .sendDisabled:
                    ConversationComposerBar(
                        text: .constant(""), isRunning: false, isStopping: false,
                        canSend: false, isInputDisabled: false, onSend: {}, onStop: {})
                case .sendEnabled:
                    ConversationComposerBar(
                        text: .constant("Where is the capital of China?"),
                        isRunning: false, isStopping: false,
                        canSend: true, isInputDisabled: false, onSend: {}, onStop: {})
                case .stopRunning:
                    ConversationComposerBar(
                        text: .constant(""), isRunning: true, isStopping: false,
                        canSend: false, isInputDisabled: true, onSend: {}, onStop: {})
                case .stopStopping:
                    ConversationComposerBar(
                        text: .constant(""), isRunning: true, isStopping: true,
                        canSend: false, isInputDisabled: true, onSend: {}, onStop: {})
                }
            }
        }
    #endif

    private func screen(_ name: String) throws -> String {
        try projectFile("Sources/Screens/\(name)")
    }

    private func projectFile(_ relativePath: String) throws -> String {
        let projectRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: projectRoot.appendingPathComponent(relativePath),
            encoding: .utf8)
    }
}

extension String {
    /// The first capture group of `pattern`, or nil.
    fileprivate func firstMatch(of pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let match = regex.firstMatch(
                in: self, range: NSRange(startIndex..., in: self)),
            let range = Range(match.range(at: 1), in: self)
        else { return nil }
        return String(self[range])
    }
}
