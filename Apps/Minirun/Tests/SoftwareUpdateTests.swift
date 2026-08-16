import Foundation
import XCTest

@testable import MinirunApp

/// Who is allowed to run the updater, and what the screen says when nobody is.
///
/// The interesting part of shipping outside the App Store is not the download —
/// Sparkle does that — it is the two ways a well-meaning build can quietly do
/// the wrong thing. A capture script can start a real updater against the
/// operator's own defaults domain, and a development build can advertise an
/// update mechanism it has no key to verify. Both are decided by
/// `SoftwareUpdatePolicy`, which is deliberately free of Sparkle, AppKit and
/// the network so both can be stated as arithmetic here.
///
/// The suite compiles on both hosts. `SoftwareUpdatePolicy` is cross-platform
/// for that reason: the rule is a product rule, even though only the Mac has an
/// updater to apply it to.
final class SoftwareUpdateTests: XCTestCase {

    // MARK: - The key has to be a key

    /// The literal that ships in the repository is the whole point of the
    /// check: an export, a fresh clone and a contributor's first build all have
    /// it, and none of them can verify an update.
    func testThePlaceholderPublicKeyIsNotAConfiguredKey() {
        XCTAssertFalse(
            SoftwareUpdatePolicy.isConfiguredPublicKey(
                SoftwareUpdatePolicy.publicKeyPlaceholder))
        XCTAssertFalse(SoftwareUpdatePolicy.isConfiguredPublicKey(nil))
        XCTAssertFalse(SoftwareUpdatePolicy.isConfiguredPublicKey(""))
    }

    /// A truncated or padded paste is the realistic way this goes wrong by
    /// hand, and it is why the check is the key's shape rather than merely
    /// "not the placeholder". 32 raw bytes, base64, is exactly 44 characters.
    func testOnlyA32ByteBase64KeyCounts() {
        let key = Data(repeating: 0x2a, count: 32).base64EncodedString()
        XCTAssertEqual(key.count, 44)
        XCTAssertTrue(SoftwareUpdatePolicy.isConfiguredPublicKey(key))
        XCTAssertTrue(SoftwareUpdatePolicy.isConfiguredPublicKey("  \(key)\n"))

        XCTAssertFalse(SoftwareUpdatePolicy.isConfiguredPublicKey(String(key.dropLast())))
        XCTAssertFalse(
            SoftwareUpdatePolicy.isConfiguredPublicKey(
                Data(repeating: 0x2a, count: 31).base64EncodedString()))
        XCTAssertFalse(
            SoftwareUpdatePolicy.isConfiguredPublicKey(
                Data(repeating: 0x2a, count: 33).base64EncodedString()))
        // 44 characters, but not base64.
        XCTAssertFalse(
            SoftwareUpdatePolicy.isConfiguredPublicKey(String(repeating: "!", count: 44)))
    }

    // MARK: - Whose launch it is

    /// A review capture and a hosted test host must not touch the operator's
    /// update schedule, for the same reason they do not touch the conversation
    /// directory or the bookmark ledger.
    func testOnlyALiveLaunchWithARealKeyStartsTheUpdater() {
        let key = Data(repeating: 0x7f, count: 32).base64EncodedString()

        XCTAssertTrue(SoftwareUpdatePolicy.startsUpdater(for: .live, publicKey: key))
        XCTAssertFalse(
            SoftwareUpdatePolicy.startsUpdater(
                for: .live, publicKey: SoftwareUpdatePolicy.publicKeyPlaceholder))
        XCTAssertFalse(SoftwareUpdatePolicy.startsUpdater(for: .launchRefused, publicKey: key))
        #if DEBUG
            XCTAssertFalse(
                SoftwareUpdatePolicy.startsUpdater(for: .visualReviewPreview, publicKey: key))
        #endif
    }

    // MARK: - What the screen says

    /// Every state has a sentence, and none of them is blank. A disabled
    /// toggle with no explanation is the failure this vocabulary prevents.
    func testEveryUpdateStateHasASentence() {
        for text in [
            SoftwareUpdatePresentation.runningExplanation,
            SoftwareUpdatePresentation.unsignedBuildExplanation,
            SoftwareUpdatePresentation.reviewLaunchExplanation,
        ] {
            XCTAssertFalse(text.isEmpty)
        }
        XCTAssertEqual(SoftwareUpdatePresentation.menuItemTitle, "Check for Updates…")
        XCTAssertEqual(
            SoftwareUpdatePresentation.automaticTitle, "Automatically check for updates")
        XCTAssertEqual(SoftwareUpdatePresentation.checkNowTitle, "Check now")
    }

    /// The running sentence names the host it contacts and denies sending
    /// anything, because that is the claim the product makes and the one a
    /// reviewer will check against the code.
    func testTheRunningSentenceNamesTheFeedHostAndDeniesTelemetry() {
        let text = SoftwareUpdatePresentation.runningExplanation
        XCTAssertTrue(text.contains("downloads.minirun.dev"))
        XCTAssertTrue(text.localizedCaseInsensitiveContains("nothing about this mac"))
    }

    /// Never checked is a fact worth printing. A blank trailing value would be
    /// a bug the reader has to diagnose.
    func testAnUncheckedUpdaterSaysSoRatherThanShowingNothing() {
        XCTAssertEqual(SoftwareUpdatePresentation.lastCheck(nil), "Never checked")
        let now = Date()
        let recent = SoftwareUpdatePresentation.lastCheck(
            now.addingTimeInterval(-3600), now: now)
        XCTAssertTrue(recent.hasPrefix("Checked "))
        XCTAssertGreaterThan(recent.count, "Checked ".count)
    }

    // MARK: - What the shipped bundle declares

    /// Sparkle compares `CFBundleVersion`, so the build number is the thing
    /// that has to increase. The repository's scheme is `YYYYMMDDNN`, which is
    /// monotonic as an integer for as long as dates increase and no more than
    /// 99 builds are stamped in a day.
    func testTheBuildNumberSchemeIsMonotonicAsAnInteger() {
        // `Bundle.main` in a hosted test is the app under test, so this is the
        // number Sparkle would actually compare.
        let stamped = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
        let build = try? XCTUnwrap(stamped)
        XCTAssertNotNil(build)
        if let build {
            XCTAssertEqual(build.count, 10, "the scheme is YYYYMMDDNN")
            XCTAssertNotNil(
                UInt64(build), "a build number Sparkle cannot read as a number is not ordered")
        }

        let scheme = ["2026081402", "2026081403", "2026081500", "2027010100"]
        let numbers = scheme.map { UInt64($0) }
        XCTAssertFalse(numbers.contains(nil))
        XCTAssertEqual(numbers.compactMap { $0 }, numbers.compactMap { $0 }.sorted())
        // Same-day rollover must not tie, and a later date must beat a later
        // build number from an earlier date.
        XCTAssertLessThan(UInt64("2026081402")!, UInt64("2026081403")!)
        XCTAssertLessThan(UInt64("2026081499")!, UInt64("2026081500")!)
    }
}
