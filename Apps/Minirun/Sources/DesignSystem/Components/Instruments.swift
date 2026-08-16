import MinirunKit
import SwiftUI

// MARK: - C · Budget gauge

/// The footprint filling against the stated budget, with a peak watermark that
/// only ever moves right.
///
/// If the peak passes the stated budget the whole gauge turns `refuse` and
/// stays refused for the rest of the run. `budgetRespected == false` is a
/// permanent fact about a run, not a transient alarm.
struct BudgetGauge: View {
    let footprintBytes: UInt64
    let peakBytes: UInt64
    let declaredBudgetBytes: UInt64
    var latchedBreach: Bool = false

    private var breached: Bool { latchedBreach || peakBytes > declaredBudgetBytes }
    private var spareBytes: UInt64 {
        declaredBudgetBytes > footprintBytes ? declaredBudgetBytes - footprintBytes : 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack {
                Text("Footprint").mrLabel()
                Spacer()
                if breached {
                    StatusChip(text: "budget not held", tone: .refuse)
                }
            }

            GeometryReader { geometry in
                let width = geometry.size.width
                let denominator = Double(max(1, max(declaredBudgetBytes, peakBytes)))
                let fill = CGFloat(Double(footprintBytes) / denominator) * width
                let watermark = CGFloat(Double(peakBytes) / denominator) * width
                let budgetMark = CGFloat(Double(declaredBudgetBytes) / denominator) * width

                ZStack(alignment: .leading) {
                    Rectangle().fill(MRColor.hairline)
                    Rectangle()
                        .fill(breached ? MRColor.refuse : MRColor.tierPinned)
                        .frame(width: max(0, fill))
                    Rectangle()
                        .fill(MRColor.primary)
                        .frame(width: 2)
                        .offset(x: max(0, watermark - 1))
                    Rectangle()
                        .fill(breached ? MRColor.refuse : MRColor.tertiary)
                        .frame(width: 1)
                        .offset(x: max(0, budgetMark - 1))
                }
            }
            .frame(height: 18)
            .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))

            Text(label)
                .font(MRType.micro)
                .foregroundStyle(breached ? MRColor.refuse : MRColor.secondary)
                .animation(nil, value: label)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("footprint against stated budget")
        .accessibilityValue(spoken)
    }

    private var label: String {
        "\(MRFormat.bytesDecimal(footprintBytes)) of "
            + "\(MRFormat.bytesDecimal(declaredBudgetBytes)) · peak "
            + "\(MRFormat.bytesDecimal(peakBytes)) · \(MRFormat.bytesDecimal(spareBytes)) spare"
    }

    private var spoken: String {
        "footprint \(MRFormat.bytesDecimal(footprintBytes)) of "
            + "\(MRFormat.bytesDecimal(declaredBudgetBytes)) stated, peak "
            + "\(MRFormat.bytesDecimal(peakBytes)), \(MRFormat.bytesDecimal(spareBytes)) spare"
            + (breached ? ", the stated budget was exceeded" : "")
    }
}

// MARK: - D · Layer ladder

/// Pure layout arithmetic for the 93-cell ladder. The inspector's supported
/// minimum is 300 points, so a fixed 371-point row is not a valid intrinsic
/// size. This policy shrinks rungs to the proposed width and never asks its
/// parent for the old fixed width.
enum LayerLadderGeometry {
    struct Metrics: Equatable {
        let cellWidth: CGFloat
        let spacing: CGFloat

        func occupiedWidth(cellCount: Int) -> CGFloat {
            guard cellCount > 0 else { return 0 }
            return cellWidth * CGFloat(cellCount)
                + spacing * CGFloat(max(0, cellCount - 1))
        }
    }

    static func metrics(
        availableWidth: CGFloat,
        cellCount: Int,
        preferredCellWidth: CGFloat = 3,
        preferredSpacing: CGFloat = 1
    ) -> Metrics {
        guard cellCount > 0, availableWidth > 0 else {
            return Metrics(cellWidth: 0, spacing: 0)
        }
        let gaps = CGFloat(max(0, cellCount - 1))
        let spacing = min(
            preferredSpacing,
            gaps > 0 ? availableWidth / gaps : preferredSpacing)
        let usable = max(0, availableWidth - spacing * gaps)
        return Metrics(
            cellWidth: min(preferredCellWidth, usable / CGFloat(cellCount)),
            spacing: spacing)
    }
}

/// Screen and VoiceOver use the same one-based description. `LayerCell.index`
/// is an engine index and is intentionally zero-based; it must never leak into
/// product progress copy.
enum LayerLadderAccessibility {
    static func completedCount(in cells: [LayerCell]) -> Int {
        cells.filter { $0.status == .done }.count
    }

    static func visibleSummary(for cells: [LayerCell]) -> String {
        guard !cells.isEmpty else { return "Not reported" }
        let done = completedCount(in: cells)
        if let current = cells.first(where: { $0.status == .computing }) {
            return "\(done) complete · layer \(current.index + 1) of \(cells.count)"
        }
        return "\(done) of \(cells.count) complete"
    }

    static func spokenValue(for cells: [LayerCell]) -> String {
        guard !cells.isEmpty else { return "not reported" }
        let done = completedCount(in: cells)
        var clauses = ["\(done) of \(cells.count) layers complete"]
        if let current = cells.first(where: { $0.status == .computing }) {
            clauses.append("layer \(current.index + 1) of \(cells.count) is computing")
        } else if let staged = cells.first(where: { $0.status == .staged }) {
            clauses.append("layer \(staged.index + 1) of \(cells.count) is staged")
        }
        return clauses.joined(separator: ", ")
    }

    static func tooltip(for cell: LayerCell) -> String {
        var parts = [
            "layer \(cell.index + 1)",
            "stored \(MRFormat.bytesDecimal(cell.storedBytes))",
            "widened \(MRFormat.bytesDecimal(cell.widenedBytes))",
        ]
        if let seconds = cell.wallSeconds { parts.append(String(format: "%.3f s", seconds)) }
        parts.append(cell.wasStaged ? "staged" : "serial")
        return parts.joined(separator: " · ")
    }
}

/// One cell per layer. The cursor advancing 93 cells over six minutes *is* the
/// picture of layers streaming.
///
/// One accessibility element with a progress value, not 93 elements: a
/// screen-reader user does not want to hear ninety-three cells.
struct LayerLadder: View {
    let cells: [LayerCell]
    var cellWidth: CGFloat = 3
    var cellHeight: CGFloat = 26
    var wraps = false

    var body: some View {
        Group {
            if wraps {
                FlowGrid(cells: cells, cellWidth: cellWidth, cellHeight: cellHeight)
            } else {
                GeometryReader { geometry in
                    let metrics = LayerLadderGeometry.metrics(
                        availableWidth: geometry.size.width,
                        cellCount: cells.count,
                        preferredCellWidth: cellWidth)
                    HStack(spacing: metrics.spacing) {
                        ForEach(cells) { cell in
                            rung(cell, width: metrics.cellWidth)
                        }
                    }
                }
                .frame(height: cellHeight)
            }
        }
        .accessibilityRepresentation {
            ProgressView(
                value: Double(LayerLadderAccessibility.completedCount(in: cells)),
                total: Double(max(1, cells.count))
            ) {
                Text("Layer progress")
            }
            .accessibilityValue(LayerLadderAccessibility.spokenValue(for: cells))
        }
    }

    private func rung(_ cell: LayerCell, width: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 1, style: .continuous)
            .fill(color(for: cell.status))
            .frame(width: width, height: cellHeight)
            .help(LayerLadderAccessibility.tooltip(for: cell))
    }

    private func color(for status: LayerCell.Status) -> Color {
        switch status {
        case .pending: return MRColor.hairline
        case .staged: return MRColor.tierPinned
        case .computing: return MRColor.streamDet
        case .done: return MRColor.tertiary
        }
    }

    /// The wrapped form for a narrow screen. Fixed columns rather than a
    /// `LazyVGrid`, because the ladder is a picture of a sequence and a lazy
    /// grid that recycles cells makes the cursor jump.
    private struct FlowGrid: View {
        let cells: [LayerCell]
        let cellWidth: CGFloat
        let cellHeight: CGFloat

        var body: some View {
            let perRow = 32
            let rows = stride(from: 0, to: cells.count, by: perRow).map {
                Array(cells[$0..<min($0 + perRow, cells.count)])
            }
            return VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    HStack(spacing: 2) {
                        ForEach(row) { cell in
                            RoundedRectangle(cornerRadius: 1, style: .continuous)
                                .fill(fill(cell.status))
                                .frame(height: cellHeight * 0.45)
                        }
                    }
                }
            }
        }

        private func fill(_ status: LayerCell.Status) -> Color {
            switch status {
            case .pending: return MRColor.hairline
            case .staged: return MRColor.tierPinned
            case .computing: return MRColor.streamDet
            case .done: return MRColor.tertiary
            }
        }
    }
}

// MARK: - E · Byte-flow ribbon

/// Two stacked lanes of successful model payload reads in runner-owned
/// half-second windows. Page-cache hits are included, so this is process data
/// flow rather than a claim about the physical drive or link.
///
/// Under Reduce Motion it stops scrolling and becomes a twelve-bucket static
/// bar chart, because a ribbon that slides is motion and a bar chart is not.
struct ByteFlowRibbon: View {
    let samples: [FlowSample]
    var laneHeight: CGFloat = 22
    /// The label above the lanes. A caller that has already introduced the
    /// instrument under a section heading passes the short form.
    var title = "Model data flow"
    /// The ⓘ belongs to whichever surface owns the heading. Inside the panel's
    /// one STORAGE section that is the section, not each chart.
    var showsHelp = true

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var motionPolicy: MRAccessibleMotionPolicy {
        MRAccessibleMotionPolicy(reduceMotion: reduceMotion)
    }

    private var peak: Double {
        max(samples.map(\.totalBytesPerSecond).max() ?? 1, 1)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            HStack {
                Text(title).mrLabel()
                if showsHelp {
                    HelpTip(
                        InstrumentHelpCopy.byteFlow,
                        accessibilityLabel: MRAccessibility.helpLabel(
                            subject: "model data flow"))
                }
                Spacer()
                Text(
                    samples.last.map { MRFormat.throughput($0.totalBytesPerSecond) }
                        ?? "Not reported"
                )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .animation(nil, value: samples.count)
            }
            if !samples.isEmpty {
                if motionPolicy.usesStaticRibbon {
                    buckets
                } else {
                    VStack(spacing: 2) {
                        lane(\.deterministicBytesPerSecond, color: MRColor.streamDet)
                        lane(\.expertBytesPerSecond, color: MRColor.streamExp)
                    }
                }
                StreamLegend()
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("model data flow, model layers and routed experts")
        .accessibilityValue(spoken)
        .transaction { transaction in
            if !motionPolicy.animatesGeometry { transaction.animation = nil }
        }
    }

    private func lane(
        _ key: KeyPath<FlowSample, Double>, color: Color
    ) -> some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height
            let count = max(1, samples.count)

            ZStack(alignment: .bottomLeading) {
                Rectangle().fill(MRColor.hairline.opacity(0.35))

                Path { path in
                    path.move(to: CGPoint(x: 0, y: height))
                    for (index, sample) in samples.enumerated() {
                        let x = width * CGFloat(index) / CGFloat(max(1, count - 1))
                        let value = sample[keyPath: key] / max(1, peak)
                        path.addLine(to: CGPoint(x: x, y: height * (1 - CGFloat(value))))
                    }
                    path.addLine(to: CGPoint(x: width, y: height))
                    path.closeSubpath()
                }
                .fill(color.opacity(0.55))

            }
        }
        .frame(height: laneHeight)
        .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
    }

    /// Twelve buckets, refreshed at 0.5 Hz, for Reduce Motion.
    private var buckets: some View {
        let bucketCount = 12
        let size = max(1, samples.count / bucketCount)
        let grouped = stride(from: 0, to: samples.count, by: size).map { start -> Double in
            let slice = samples[start..<min(start + size, samples.count)]
            guard !slice.isEmpty else { return 0 }
            return slice.map(\.totalBytesPerSecond).reduce(0, +) / Double(slice.count)
        }
        return HStack(alignment: .bottom, spacing: 3) {
            ForEach(Array(grouped.enumerated()), id: \.offset) { _, value in
                RoundedRectangle(cornerRadius: 2)
                    .fill(MRColor.streamDet.opacity(0.7))
                    .frame(height: max(2, CGFloat(value / max(1, peak)) * laneHeight * 2))
            }
        }
        .frame(height: laneHeight * 2, alignment: .bottom)
    }

    private var spoken: String {
        guard let last = samples.last else { return "not reported" }
        return "model layers \(MRFormat.throughput(last.deterministicBytesPerSecond)), "
            + "routed experts \(MRFormat.throughput(last.expertBytesPerSecond))"
    }
}

// MARK: - The two streams, named once

/// One legend for both storage instruments.
///
/// The ribbon's lanes and the byte-split line under the readouts describe the
/// same two streams, and they used to name them twice in two vocabularies —
/// "model layers / experts" here and "deterministic / expert" there. There is
/// one vocabulary now, and it is this one; `deterministic` remains the code
/// identifier and stays out of product copy.
struct StreamLegend: View {
    var body: some View {
        HStack(spacing: MRSpace.s3) {
            entry("model layers", MRColor.streamDet)
            entry("experts", MRColor.streamExp)
            Spacer()
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("legend")
        .accessibilityValue("model layers, experts")
    }

    private func entry(_ text: String, _ color: Color) -> some View {
        HStack(spacing: MRSpace.s1) {
            RoundedRectangle(cornerRadius: 1).fill(color).frame(width: 8, height: 3)
            Text(text).font(MRType.micro).foregroundStyle(MRColor.tertiary)
        }
    }
}

// MARK: - F · Overlap meter

/// The single most diagnostic instrument, from the `Stalls` pair. One bar, two
/// segments: phase time not blocked on model data and time spent waiting for it.
struct OverlapMeter: View {
    let stalls: StallAccounting?
    let readAhead: ReadAheadAccounting?
    var title = "Storage overlap"
    var showsHelp = true

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack {
                Text(title).mrLabel()
                if showsHelp {
                    HelpTip(
                        InstrumentHelpCopy.overlap,
                        accessibilityLabel: MRAccessibility.helpLabel(subject: "storage overlap"))
                }
                Spacer()
                Text(overlapText)
                    .font(MRType.metric)
                    .foregroundStyle(MRColor.primary)
                    .animation(nil, value: overlapText)
            }

            if let stalls, stalls.overlapFraction != nil {
                // Waiting is drawn in `caution`, not in the ribbon's model-layer
                // blue. Under one STORAGE heading that blue means "this stream",
                // and reusing it for "this much time was lost" put two meanings
                // on one colour a legend away from each other.
                bar(
                    first: stalls.computeSeconds, firstColor: MRColor.tertiary,
                    second: stalls.computeWaitedOnIOSeconds, secondColor: MRColor.caution)

                Text(
                    "not waiting \(MRFormat.clock(stalls.computeSeconds)) · waiting on model data "
                        + MRFormat.clock(stalls.computeWaitedOnIOSeconds)
                )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Not reported")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
            }

            if let readAhead {
                if readAhead.overlapEfficiency != nil {
                    bar(
                        first: max(0, readAhead.stageReadSeconds - readAhead.prefetchWaitSeconds),
                        firstColor: MRColor.tierPinned,
                        second: readAhead.prefetchWaitSeconds, secondColor: MRColor.caution,
                        height: 6)
                }
                Text(
                    "\(readAhead.overlapEfficiency.map { MRFormat.percent($0) } ?? "—") of read time hidden · "
                        + "staging peaked at \(MRFormat.bytesDecimal(readAhead.peakStagedBytes))"
                )
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("storage overlap meter")
        .accessibilityValue(accessibilityValue)
    }

    private var overlapText: String {
        stalls?.overlapFraction.map { MRFormat.percent($0) } ?? "—"
    }

    private var accessibilityValue: String {
        guard let stalls, let fraction = stalls.overlapFraction else {
            return "not reported"
        }
        return "\(MRFormat.percent(fraction)) of the expert phase was not blocked waiting for model data"
    }

    private func bar(
        first: Double, firstColor: Color, second: Double, secondColor: Color,
        height: CGFloat = 12
    ) -> some View {
        GeometryReader { geometry in
            let total = max(0.0001, first + second)
            let width = geometry.size.width
            HStack(spacing: 0) {
                Rectangle().fill(firstColor)
                    .frame(width: width * CGFloat(max(0, first) / total))
                Rectangle().fill(secondColor)
                    .frame(width: width * CGFloat(max(0, second) / total))
            }
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: 2, style: .continuous))
    }
}

// MARK: - Where the time went

/// One pass kind's wall time, split into the terms the engine measured.
///
/// The residual is a slice like any other and is labelled `Unattributed`: a bar
/// that drew only the named terms would make the parts look like the whole, and
/// on tonight's V4 records the residual is the largest single piece. Terms the
/// engine reports *beside* the pass — a background stager's read time, a
/// container total — are not drawn, because adding overlapped work to a wall
/// time makes the parts exceed it.
struct PhaseSplitBar: View {
    let summary: RunPhaseSummary
    /// Named rows under the bar. The rest of the bar keeps its colour without
    /// keeping a row, so a fifteen-term V4 pass does not become a fifteen-row
    /// table in a narrow inspector.
    var namedTermLimit = 3

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            HStack(alignment: .firstTextBaseline) {
                Text(title)
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
                Spacer(minLength: MRSpace.s2)
                Text(MRFormat.phaseSeconds(perPassSeconds))
                    .font(MRType.metric)
                    .foregroundStyle(MRColor.primary)
                    .animation(nil, value: perPassSeconds)
            }
            bar
            ForEach(rows, id: \.name) { term in
                HStack(spacing: MRSpace.s2) {
                    RoundedRectangle(cornerRadius: 1, style: .continuous)
                        .fill(PhaseTermPalette.color(for: term.name))
                        .frame(width: 8, height: 8)
                        .accessibilityHidden(true)
                    Text(RunPhaseTermName.displayName(for: term.name))
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.secondary)
                        .lineLimit(1)
                    Spacer(minLength: MRSpace.s2)
                    Text(
                        MRFormat.phaseSeconds(term.seconds / passDivisor)
                            + (fraction(term).map { " · \(MRFormat.percent($0, digits: 0))" } ?? "")
                    )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .animation(nil, value: term.seconds)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(accessibilityValue)
    }

    private var slices: [RunPhaseTerm] { summary.orderedTermsWithResidual }

    private var rows: [RunPhaseTerm] {
        let named = slices.filter { $0.name != RunPhaseTermName.unattributed }
            .prefix(namedTermLimit)
        let residual = slices.filter { $0.name == RunPhaseTermName.unattributed }
        return Array(named) + residual
    }

    /// An aggregate states its own pass count, so its bar is read per pass —
    /// the mean decode pass — rather than as a total nobody generated.
    private var passDivisor: Double {
        summary.passCount > 1 ? Double(summary.passCount) : 1
    }

    private var perPassSeconds: Double {
        summary.secondsPerPass ?? summary.passSeconds
    }

    private var title: String {
        switch summary.passKind {
        case .prefill: return "Prefill"
        case .decode:
            return summary.passCount > 1
                ? "Decode · mean of \(summary.passCount) passes" : "Decode"
        }
    }

    private func fraction(_ term: RunPhaseTerm) -> Double? {
        guard summary.passSeconds > 0 else { return nil }
        return term.seconds / summary.passSeconds
    }

    private var bar: some View {
        GeometryReader { geometry in
            let total = max(0.0001, slices.reduce(0) { $0 + max(0, $1.seconds) })
            let width = geometry.size.width
            HStack(spacing: 0) {
                ForEach(slices, id: \.name) { term in
                    Rectangle()
                        .fill(PhaseTermPalette.color(for: term.name))
                        .frame(width: width * CGFloat(max(0, term.seconds) / total))
                }
            }
            .frame(width: width, alignment: .leading)
        }
        .frame(height: 12)
        .clipShape(RoundedRectangle(cornerRadius: 2, style: .continuous))
    }

    private var accessibilityValue: String {
        guard summary.passSeconds > 0 else { return "not reported" }
        let parts = rows.map { term in
            "\(RunPhaseTermName.displayName(for: term.name)) "
                + MRFormat.phaseSecondsSpoken(term.seconds / passDivisor)
        }
        return "\(MRFormat.phaseSecondsSpoken(perPassSeconds)) per pass: "
            + parts.joined(separator: ", ")
    }
}

/// One colour per term, stable across bars and across runs.
///
/// Stability is the whole requirement: the prefill bar and the decode bar are
/// read against each other, and a term that changed colour between them would
/// invite exactly the wrong comparison. Waiting keeps the `caution` the overlap
/// meter already gives it; reads keep the deterministic stream's blue; the
/// residual is the empty-track grey, because it is the part nothing has claimed
/// yet.
enum PhaseTermPalette {

    /// Hue says what kind of work a term is; weight separates the members of
    /// one kind. A V4 pass has sixteen terms and this product has eight colour
    /// tokens, so two terms of the same kind sitting next to each other in a bar
    /// would otherwise merge into one slice.
    private static let assigned: [String: (Color, Double)] = [
        RunPhaseTermName.unattributed: (MRColor.hairline, 1),

        // Reads.
        RunPhaseTermName.deterministicRead: (MRColor.streamDet, 1),
        RunPhaseTermName.outputHeadRead: (MRColor.streamDet, 0.6),

        // Waiting for storage.
        RunPhaseTermName.expertIOWait: (MRColor.caution, 1),
        RunPhaseTermName.stagerWait: (MRColor.caution, 0.55),
        RunPhaseTermName.stagedRead: (MRColor.caution, 0.35),

        // The routed-expert phase's own compute.
        RunPhaseTermName.expertGatherCompute: (MRColor.streamExp, 1),
        RunPhaseTermName.lightningIndexer: (MRColor.streamExp, 0.72),
        RunPhaseTermName.expertOther: (MRColor.streamExp, 0.5),

        // Attention and the output head.
        RunPhaseTermName.attentionBarrier: (MRColor.tierPinned, 1),
        RunPhaseTermName.sparseAttention: (MRColor.tierPinned, 0.72),
        RunPhaseTermName.outputHeadCompute: (MRColor.tierPinned, 0.5),

        // Host round trips that decide rather than compute.
        RunPhaseTermName.activationScaleSync: (MRColor.tierStaged, 1),
        RunPhaseTermName.routingSelect: (MRColor.tierStaged, 0.62),

        // Verification and the work of turning bytes into an operand.
        RunPhaseTermName.tileDigest: (MRColor.verify, 1),
        RunPhaseTermName.tileAdoption: (MRColor.verify, 0.7),
        RunPhaseTermName.finitenessSweep: (MRColor.verify, 0.45),

        // Setup, teardown, and giving pages back.
        RunPhaseTermName.reclaim: (MRColor.tierFloor, 1),
        RunPhaseTermName.layerArtifactSetup: (MRColor.tierFloor, 0.7),
        RunPhaseTermName.expertBackendLifecycle: (MRColor.tierFloor, 0.45),
    ]

    /// A term this build has never heard of still gets a colour, chosen from
    /// the same palette by its name so it is the same colour in every bar.
    private static let fallback: [Color] = [
        MRColor.tierPinned, MRColor.tierHot, MRColor.streamExp, MRColor.verify,
        MRColor.tierFloor, MRColor.tierStaged,
    ]

    static func color(for name: String) -> Color {
        if let assigned = assigned[name] { return assigned.0.opacity(assigned.1) }
        let seed = name.utf8.reduce(0) { ($0 &* 31 &+ Int($1)) & 0xFFFF }
        return fallback[seed % fallback.count]
    }
}

// MARK: - G · Cache strip

struct CacheStripView: View {
    let cache: CacheStrip?

    var body: some View {
        Text(cache?.line ?? "Not reported")
            .font(MRType.micro)
            .foregroundStyle(MRColor.tertiary)
            .fixedSize(horizontal: false, vertical: true)
            .animation(nil, value: cache?.line)
            .accessibilityLabel("expert reuse")
            .accessibilityValue(accessibilityValue)
    }

    private var accessibilityValue: String {
        guard let cache else { return "not reported" }
        return cache.accessibilityLine
    }
}

// MARK: - Verification progress

/// Verification is not a spinner. It is the moment files become authoritative
/// over metadata, and it gets its own progress with its own language.
struct VerificationProgress: View {
    let phase: VerificationPhase

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack {
                Text("File integrity").mrLabel(MRColor.verify)
            }
            ProgressView()
                .tint(MRColor.verify)
            Text(phase.sentence)
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }
}
