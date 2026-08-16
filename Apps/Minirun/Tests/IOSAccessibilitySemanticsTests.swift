import XCTest

@testable import MinirunApp

final class IOSAccessibilitySemanticsTests: XCTestCase {
    func testHelpLabelsNameTheirSubject() {
        XCTAssertEqual(
            MRAccessibility.helpLabel(subject: " storage overlap "),
            "About storage overlap")
        XCTAssertEqual(MRAccessibility.helpLabel(subject: "  "), "More information")
    }

    func testLegacyHelpLabelUsesTheFirstPlainLanguageClause() {
        XCTAssertEqual(
            MRAccessibility.inferredHelpLabel(
                from: "Shows model data read from storage. This is not network traffic."),
            "More information: Shows model data read from storage")
    }

    func testIOSTouchTargetHasAFortyFourPointFloorWithoutChangingMacSizing() {
        XCTAssertEqual(
            MRAccessibility.interactiveDimension(22, onIOS: true),
            MRAccessibility.minimumIOSTouchTarget)
        XCTAssertEqual(MRAccessibility.interactiveDimension(60, onIOS: true), 60)
        XCTAssertEqual(MRAccessibility.interactiveDimension(22, onIOS: false), 22)
    }

    func testLayerProgressUsesOneBasedProductIndices() {
        let cells = [
            cell(index: 0, status: .done),
            cell(index: 1, status: .computing),
            cell(index: 2, status: .staged),
        ]

        XCTAssertEqual(LayerLadderAccessibility.completedCount(in: cells), 1)
        XCTAssertEqual(
            LayerLadderAccessibility.visibleSummary(for: cells),
            "1 complete · layer 2 of 3")
        XCTAssertEqual(
            LayerLadderAccessibility.spokenValue(for: cells),
            "1 of 3 layers complete, layer 2 of 3 is computing")
        XCTAssertTrue(LayerLadderAccessibility.tooltip(for: cells[1]).hasPrefix("layer 2 ·"))
        XCTAssertFalse(LayerLadderAccessibility.spokenValue(for: cells).contains("layer 1 of 3"))
    }

    func testLayerProgressNamesAStagedLayerWhenNoLayerIsComputing() {
        let cells = [
            cell(index: 0, status: .done),
            cell(index: 1, status: .staged),
            cell(index: 2, status: .pending),
        ]

        XCTAssertEqual(
            LayerLadderAccessibility.spokenValue(for: cells),
            "1 of 3 layers complete, layer 2 of 3 is staged")
    }

    func testReduceMotionSelectsImmediateGeometryAndStaticRibbon() {
        let policy = MRAccessibleMotionPolicy(reduceMotion: true)

        XCTAssertEqual(policy, .reduced)
        XCTAssertFalse(policy.animatesGeometry)
        XCTAssertTrue(policy.usesStaticRibbon)
        XCTAssertNil(policy.tierAnimation)
    }

    func testStandardMotionKeepsTheDesignedGeometryTransition() {
        let policy = MRAccessibleMotionPolicy(reduceMotion: false)

        XCTAssertEqual(policy, .standard)
        XCTAssertTrue(policy.animatesGeometry)
        XCTAssertFalse(policy.usesStaticRibbon)
        XCTAssertNotNil(policy.tierAnimation)
    }

    func testMissingEvidenceIsNeutralAndSpokenAsMissing() {
        XCTAssertEqual(MetricEvidencePresentation.tone(for: .notReported), .neutral)
        XCTAssertEqual(
            MetricEvidencePresentation.accessibilityValue(for: .notReported),
            "not reported")
        XCTAssertEqual(MetricEvidencePresentation.tone(for: .valid), .ok)
        XCTAssertEqual(
            MetricEvidencePresentation.tone(for: .failed("bytes do not balance")),
            .refuse)
    }

    func testSharedMemoryPresetsDoNotLayerASelectionOutlineOverANativeButton() throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/MemoryDialView.swift"),
            encoding: .utf8)

        let presetSection = try XCTUnwrap(
            source.range(
                from: "private var presetButtons: some View",
                through: "@ViewBuilder private var presetDeficits"))
        let presetSource = String(source[presetSection])

        XCTAssertTrue(presetSource.contains(".buttonStyle(.plain)"))
        XCTAssertTrue(presetSource.contains("Image(systemName: \"checkmark\")"))
        XCTAssertFalse(presetSource.contains(".buttonStyle(.bordered)"))
        XCTAssertFalse(presetSource.contains("isSelected ? MRColor.tierPinned : Color.clear"))
    }

    private func cell(index: Int, status: LayerCell.Status) -> LayerCell {
        LayerCell(
            index: index,
            status: status,
            storedBytes: 1,
            widenedBytes: 2,
            wallSeconds: nil,
            wasStaged: status == .staged)
    }
}

private extension String {
    func range(from start: String, through end: String) -> Range<String.Index>? {
        guard let lower = range(of: start)?.lowerBound,
              let upper = range(of: end, range: lower..<endIndex)?.upperBound else {
            return nil
        }
        return lower..<upper
    }
}
