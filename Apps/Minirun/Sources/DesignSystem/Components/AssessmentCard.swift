import MinirunKit
import SwiftUI

/// What one drive measured.
///
/// The job here is to make a slow link visible *before* a day is spent copying
/// onto the wrong port, so the two rates lead and everything that qualifies
/// them follows: a caution the assessment itself raised, the sentence that
/// names the link class, and the one note that says a faster port exists.
///
/// It used to be a bordered card with a mono `MetricRow` table inside it. In
/// the product-page language (DESIGN I.37) a measurement is a definition list
/// on the page's own surface: the digits are tabular and right-aligned so one
/// row's magnitude can be compared with the row above it, the unit is a
/// separate column so `GB/s` never shifts sideways under `MB/s`, and the two
/// controls are text links rather than two small bordered buttons wedged into
/// a heading. Model fit and token projections still belong on each model's
/// detail screen and are deliberately absent.
struct AssessmentFacts: View {
    let report: VolumeAssessmentReport
    /// Nil on the read-only surfaces (iOS), where a report can be shown but not
    /// taken.
    var onReassess: (() -> Void)?
    var onForget: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            // The classification already sits in the Link row of the facts;
            // repeating it as a sentence under them read as an echo.
            facts
            notes
            controls
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("drive assessment")
    }

    // MARK: Facts

    /// Both halves of the measurement on one aligned grid. The sequential
    /// figure is the one every projection divides by; the scattered figure is
    /// beside it because a drive whose random rate collapses is a drive whose
    /// expert stream will collapse.
    private var facts: some View {
        MRFactList(labelWidth: 104) {
            MRFact(label: "Link") {
                Text(report.classification.linkClass.label)
                    .font(MRType.prose)
                    .foregroundStyle(linkColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let spot = report.spot {
                MRFact(label: "Sequential") {
                    MRUnitValue(
                        text: MRFormat.throughput(spot.sequentialBytesPerSecond),
                        detail: "\(MRFormat.bytesDecimal(spot.sequentialBytesRead)) read")
                }
                MRFact(label: "Scattered") {
                    MRUnitValue(
                        text: MRFormat.throughput(spot.scatteredBytesPerSecond),
                        detail: "\(MRFormat.grouped(spot.scatteredReadCount)) reads of "
                            + MRFormat.bytesDecimal(UInt64(spot.scatteredReadBytes)))
                }
                MRFact(label: "Duration") {
                    MRUnitValue(
                        text: MRFormat.duration(spot.durationSeconds),
                        detail: "\(MRFormat.bytesDecimal(spot.totalBytesRead)) in total")
                }
            }
            if let free = report.space.dependableBytes {
                MRFact(label: "Free space") {
                    MRUnitValue(
                        text: MRFormat.bytesDecimal(free),
                        detail: "the dependable figure, not the purgeable one")
                }
            }
            // A date is not a quantity, so it does not go through
            // `MRUnitValue`: splitting "2026-08-03 09:06" at its last space
            // would right-align the date and set the clock as its unit. The
            // row's own status line above carries the relative form.
            MRFact("Assessed", MRFormat.timestamp(report.assessedAt))
        }
    }

    private var linkColor: Color {
        switch report.classification.linkClass {
        case .internalNVMe, .usb4OrThunderbolt: return MRColor.ok
        case .ambiguous: return MRColor.secondary
        case .usb3Gen2, .belowUSB3Gen2: return MRColor.caution
        }
    }

    // MARK: Notes
    //
    // Every one of these is actionable: re-measure, move the drive to a faster
    // port, or stop trusting a number a short read has already invalidated. A
    // measurement that is simply slow gets no note at all — the sentence above
    // already says what the link is.

    @ViewBuilder private var notes: some View {
        if let caution = report.caution {
            MRInlineNote(message: caution, title: "This drive was not measured")
        }
        if report.isStale() {
            MRInlineNote(
                message: "That measurement is more than a week old — assess the drive again.")
        }
        if let degraded = StorageAssessmentPresentation.fasterLinkSentence(report) {
            MRInlineNote(message: degraded, title: "A faster port exists")
        }
        if let spot = report.spot, spot.shortReadCount > 0 {
            MRInlineNote(
                message: "\(MRFormat.grouped(spot.shortReadCount)) reads came back short. "
                    + "A drive that returns short reads is a drive that is going away.",
                title: "Short reads",
                tone: .attention,
                systemImage: "exclamationmark.triangle")
        }
        if let spot = report.spot, !spot.cacheBypassed {
            MRInlineNote(
                message: "F_NOCACHE was refused on this filesystem, so these numbers may "
                    + "include the page cache.",
                title: "The cache could not be bypassed")
        }
    }

    // MARK: Controls

    @ViewBuilder private var controls: some View {
        if onReassess != nil || onForget != nil {
            HStack(spacing: MRSpace.s4) {
                if let onReassess {
                    Button("Assess again", action: onReassess).mrTextLink()
                }
                if let onForget {
                    Button("Forget this measurement", action: onForget).mrTextLink(.quiet)
                }
                Spacer(minLength: 0)
            }
        }
    }
}

/// A measured quantity with its unit in its own column.
///
/// `1.85 GB/s` above `240 MB/s` in one string leaves the units at two different
/// x positions and the digits at two more, which is exactly the comparison the
/// two rows exist to make. Splitting at the last space puts the digits in a
/// right-aligned column and the unit in a left-aligned one, so both line up
/// down the list. A value that does not begin with a digit — `free space
/// unknown`, `—` — is not a quantity and is printed whole.
struct MRUnitValue: View {
    let text: String
    var detail: String?
    var numberWidth: CGFloat = 64

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            if let split = Self.split(text) {
                HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                    // Left-aligned with the other values in the list: a
                    // right-aligned number under a left-aligned detail line
                    // read as two columns fighting.
                    Text(split.number)
                        .font(MRType.figure)
                        .foregroundStyle(MRColor.primary)
                    Text(split.unit)
                        .font(MRType.prose)
                        .foregroundStyle(MRColor.secondary)
                    Spacer(minLength: 0)
                }
            } else {
                Text(text)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let detail {
                Text(detail)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(detail.map { "\(text), \($0)" } ?? text)
    }

    /// `("1.85", "GB/s")`, or nil when the string is a sentence rather than a
    /// quantity.
    static func split(_ text: String) -> (number: String, unit: String)? {
        guard let first = text.first, first.isNumber || first == "<" || first == "-" else {
            return nil
        }
        guard let index = text.lastIndex(of: " ") else { return (text, "") }
        return (String(text[..<index]), String(text[text.index(after: index)...]))
    }
}

/// Storage-only copy composed from a drive report. The underlying report also
/// carries per-model projections for the Models screen; none of that content
/// is allowed to leak back into Storage.
enum StorageAssessmentPresentation {
    static func fasterLinkSentence(_ report: VolumeAssessmentReport) -> String? {
        guard let rate = report.classification.measuredBytesPerSecond,
            let multiple = report.classification.fasterTierMultiple,
            let name = report.classification.fasterTierName
        else { return nil }
        return
            "This drive measured \(MRFormat.throughput(rate)). A \(name) connection has "
            + "measured \(MRFormat.throughput(rate * multiple)) — about "
            + String(format: "%.1f", multiple) + "× faster."
    }
}

/// The two strings the assessment composes from a report.
///
/// Separate from the view so the app's own suite can check the arithmetic and
/// the wording without instantiating SwiftUI.
enum AssessmentFormat {

    /// The fit, as a value: what it needs and what is left, or how far short.
    static func fitValue(_ fit: FitCheck) -> String {
        switch fit.verdict {
        case .alreadyHere(let bytes):
            return "\(MRFormat.bytesDecimal(bytes)) here"
        case .fits(let spare):
            return "\(MRFormat.bytesDecimal(fit.requiredBytes)) · "
                + "\(MRFormat.bytesDecimal(UInt64(max(0, spare)))) spare"
        case .short(let by):
            return "\(MRFormat.bytesDecimal(by)) short"
        case .unknown:
            return "free space unknown"
        }
    }

    /// `5:12 – 6:37 / token`. The `≈` is added by ``ValueText``'s projected
    /// provenance, so it appears exactly once.
    static func range(_ projection: ProjectedTokenTime) -> String {
        "\(MRFormat.clock(projection.secondsPerTokenLow)) – "
            + "\(MRFormat.clock(projection.secondsPerTokenHigh)) / token"
    }

    /// The line the model screen shows under its storage section.
    static func onVolumeLine(
        volumeName: String?, projection: ProjectedTokenTime, measuredAt: Date
    ) -> String {
        "on \(volumeName ?? "this volume"): ≈ \(range(projection)) "
            + "(measured \(MRFormat.timestamp(measuredAt)))"
    }
}
