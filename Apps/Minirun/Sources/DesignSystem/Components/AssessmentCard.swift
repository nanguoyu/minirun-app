import MinirunKit
import SwiftUI

/// What one drive measured.
///
/// The card's job is to make a link visible *before* a day is spent copying onto
/// the wrong port. So it leads with the rate — mono, because it was measured —
/// and puts the caution beside the measurement: an operator who reads one line
/// should read the one that says the drive is three times slower than it could
/// be. Model fit and token projections belong on each model's detail screen.
struct AssessmentCard: View {
    let report: VolumeAssessmentReport
    /// Nil on the read-only surfaces (iOS), where a report can be shown but not
    /// taken.
    var onReassess: (() -> Void)?
    var onForget: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            header
            if let caution = report.caution {
                cautionRow(caution, tone: MRColor.caution)
            }
            if report.spot != nil {
                rates
            }
            if let degraded = StorageAssessmentPresentation.fasterLinkSentence(report) {
                cautionRow(degraded, tone: MRColor.caution)
            }
        }
        .padding(MRSpace.s3)
        .mrCard(MRColor.raised)
        .accessibilityElement(children: .contain)
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s3) {
            VStack(alignment: .leading, spacing: 1) {
                Text(report.volumeName ?? report.locationPath)
                    .font(MRType.headline)
                    .foregroundStyle(MRColor.primary)
                Text(
                    "assessed \(MRFormat.relative(report.assessedAt)) · "
                        + MRFormat.timestamp(report.assessedAt)
                )
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
            }
            Spacer(minLength: MRSpace.s3)
            VStack(alignment: .trailing, spacing: 1) {
                Text(report.classification.linkClass.label)
                    .font(MRType.caption)
                    .foregroundStyle(linkColor)
                if report.isStale() {
                    Text("measurement out of date")
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.caution)
                }
            }
            if let onReassess {
                Button("Re-assess", action: onReassess).controlSize(.small)
            }
            if let onForget {
                Button("Forget", action: onForget).controlSize(.small)
            }
        }
    }

    private var linkColor: Color {
        switch report.classification.linkClass {
        case .internalNVMe, .usb4OrThunderbolt: return MRColor.ok
        case .ambiguous: return MRColor.secondary
        case .usb3Gen2, .belowUSB3Gen2: return MRColor.caution
        }
    }

    // MARK: Rates

    /// Both halves of the measurement, on one aligned grid. The sequential
    /// figure is the one every projection divides by; the scattered figure is
    /// beside it because a drive whose random rate collapses is a drive whose
    /// expert stream will collapse.
    @ViewBuilder private var rates: some View {
        if let spot = report.spot {
            VStack(alignment: .leading, spacing: MRSpace.s1) {
                MetricRow(
                    label: "sequential",
                    value: MRFormat.throughput(spot.sequentialBytesPerSecond),
                    detail: "\(MRFormat.bytesDecimal(spot.sequentialBytesRead)) read")
                MetricRow(
                    label: "scattered",
                    value: MRFormat.throughput(spot.scatteredBytesPerSecond),
                    detail: "\(MRFormat.grouped(spot.scatteredReadCount)) reads of "
                        + MRFormat.bytesDecimal(UInt64(spot.scatteredReadBytes)))
                MetricRow(
                    label: "duration",
                    value: MRFormat.duration(spot.durationSeconds),
                    detail: "\(MRFormat.bytesDecimal(spot.totalBytesRead)) in total")
                if let free = report.space.dependableBytes {
                    MetricRow(
                        label: "free space", value: MRFormat.bytesDecimal(free),
                        detail: "the dependable figure, not the purgeable one")
                }
                Text(report.classification.sentence)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if spot.shortReadCount > 0 {
                    Text(
                        "\(MRFormat.grouped(spot.shortReadCount)) reads came back short. "
                            + "A drive that returns short reads is a drive that is going away."
                    )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.refuse)
                    .fixedSize(horizontal: false, vertical: true)
                }
                if !spot.cacheBypassed {
                    Text(
                        "F_NOCACHE was refused on this filesystem, so these numbers may include "
                            + "the page cache."
                    )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.caution)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func cautionRow(_ text: String, tone: Color) -> some View {
        Text(text)
            .font(MRType.caption)
            .foregroundStyle(tone)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(MRSpace.s2)
            .mrCard(MRColor.panel, stroke: tone.opacity(0.4), radius: MRRadius.control)
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

/// The two strings the card composes from a report.
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
