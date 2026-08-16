import SwiftUI

/// Projected seconds per token across the budget domain.
///
/// The version this replaces drew an unlabelled 76-point sparkline: no axes, no
/// units, no idea which end was fast. It was decorative, and a decorative chart
/// in a panel whose whole argument is "these numbers are honest" is the worst
/// kind of chart to have.
///
/// What is drawn now:
///
///  * a labelled vertical axis in seconds per token, fast at the top, with the
///    two ends printed;
///  * a horizontal axis in budget, from zero to the device ceiling — the same
///    domain as the slider above it, so the two can be read against each other;
///  * the refusal region hatched, because there is no run to project there;
///  * a marker at the budget currently stated, with its value;
///  * runs that actually happened as filled dots, in a different shape and a
///    different colour from anything projected.
///
/// With no calibration there is **no curve at all**. Not a dashed one, not a
/// faded one: the measured points are plotted on their own and the rest of the
/// plot stays empty, because a line drawn through one measurement and a
/// datasheet is a claim nobody made.
struct ProjectionStrip: View {
    let plan: BudgetPlan
    let terms: WorkloadTerms
    let link: VolumeCalibration?
    var measured: [MeasuredPoint] = []
    var evidence: ProjectionEvidence? = nil
    var height: CGFloat = 132

    @State private var showingProvenance = false

    private var hasConfigurationEvidence: Bool {
        evidence?.matches(plan: plan, terms: terms) == true
    }

    private var verifiedLink: VolumeCalibration? {
        hasConfigurationEvidence ? link : nil
    }

    private var verifiedMeasured: [MeasuredPoint] {
        hasConfigurationEvidence ? measured : []
    }

    private var samples: [(budgetBytes: UInt64, projection: SpeedProjection?)] {
        guard verifiedLink != nil else { return [] }
        return SpeedProjector.curve(plan: plan, terms: terms, link: verifiedLink)
    }

    /// The seconds-per-token axis lives in its own column, never on top of the
    /// plot. The version this replaces drew both labels inside the plot's
    /// `ZStack`, where the top one landed in the hatched refusal corner and was
    /// clipped by the rounded rectangle. A gutter cannot be clipped by the thing
    /// it sits beside.
    private static let axisGutter: CGFloat = 54

    @ViewBuilder var body: some View {
        if verifiedLink == nil, verifiedMeasured.isEmpty {
            notMeasured
        } else {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                headline
                HStack(alignment: .top, spacing: MRSpace.s2) {
                    verticalAxis
                    VStack(alignment: .leading, spacing: MRSpace.s1) {
                        plot
                        axisLabels
                    }
                }
                pinnedTierHonesty
                if let link = verifiedLink {
                    provenance(link: link)
                } else {
                    Text(
                        "Completed runs appear as points. No curve is drawn without a verified "
                            + "link measurement."
                    )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    // MARK: Pieces

    private var notMeasured: some View {
        HStack(alignment: .top, spacing: MRSpace.s3) {
            Image(systemName: "gauge.with.dots.needle.bottom.50percent")
                .foregroundStyle(MRColor.tertiary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(ProjectionPresentation.notMeasuredTitle)
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
                Text(ProjectionPresentation.notMeasuredMessage)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(MRColor.raised, radius: MRRadius.control)
        .accessibilityElement(children: .combine)
    }

    /// A plan can say which layers stay resident, but it cannot turn a historic
    /// experiment on another artifact/device into a measurement for this run.
    /// Until evidence is keyed to the selected model, artifact revision and
    /// device, the product says only that the effect is not measured.
    @ViewBuilder private var pinnedTierHonesty: some View {
        if plan.isRunnable, plan.pinnedLayerCount > 0 {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Text("Pinned-layer speed")
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
                Spacer(minLength: MRSpace.s2)
                Text(ProjectionPresentation.notMeasuredTitle)
                    .font(MRType.declaredSmall)
                    .foregroundStyle(MRColor.tertiary)
            }
            .padding(MRSpace.s2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .mrCard(MRColor.raised, radius: MRRadius.control)
            .accessibilityElement(children: .combine)
            .accessibilityLabel("pinned-layer speed is not measured for this configuration")
        }
    }

    @ViewBuilder private var headline: some View {
        if verifiedLink != nil,
            let projection = SpeedProjector.project(
                plan: plan, terms: terms, link: verifiedLink)
        {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Text(projection.displayString)
                    .font(MRType.metric)
                    .foregroundStyle(MRColor.secondary)
                    .animation(nil, value: projection.mid)
                Spacer()
                if projection.deterministicHidden {
                    StatusChip(text: "staged", tone: .neutral)
                }
            }
        } else if verifiedLink != nil {
            Text("no projection at this budget — it is refused")
                .font(MRType.micro)
                .foregroundStyle(MRColor.refuse)
        } else if verifiedMeasured.isEmpty {
            Text("no measured runs yet")
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
        } else {
            Text(
                "\(verifiedMeasured.count) measured run"
                    + "\(verifiedMeasured.count == 1 ? "" : "s"), no curve")
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
        }
    }

    /// The seconds-per-token range the vertical axis covers.
    ///
    /// Computed over the projected curve *and* the measured points together, so
    /// a run slower than anything projected is drawn inside the plot instead of
    /// off the top of it.
    private var verticalRange: (low: Double, high: Double)? {
        var values = samples.compactMap(\.projection?.mid)
        values.append(contentsOf: verifiedMeasured.map(\.secondsPerToken))
        guard let low = values.min(), let high = values.max() else { return nil }
        // A flat curve still needs a band to be drawn in.
        return high - low < 0.0001 ? (low * 0.9, high * 1.1 + 0.001) : (low, high)
    }

    /// Fast at the top, and both ends printed. A speed chart where up means
    /// slower reads as a performance win every time the budget goes down.
    @ViewBuilder private var verticalAxis: some View {
        VStack(alignment: .trailing, spacing: 0) {
            if let range = verticalRange {
                Text("\(MRFormat.clock(range.low))/tok")
                Spacer(minLength: 0)
                Text("\(MRFormat.clock(range.high))/tok")
            } else {
                Text("s/tok")
                Spacer(minLength: 0)
            }
        }
        .font(MRType.micro)
        .foregroundStyle(MRColor.tertiary)
        .lineLimit(1)
        .padding(.vertical, Self.plotInset - 4)
        .frame(width: Self.axisGutter, height: height, alignment: .trailing)
        .accessibilityHidden(true)
    }

    /// The margin the curve keeps from the plot's own edges, so a dot at either
    /// extreme is a whole dot and not a half one against the border.
    private static let plotInset: CGFloat = 9

    private var plot: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let plotHeight = geometry.size.height
            let inset = Self.plotInset
            let ceiling = Double(max(1, plan.deviceCeilingBytes))
            let range = verticalRange

            let position: (UInt64, Double) -> CGPoint = { budget, seconds in
                let fraction = CGFloat(min(1, Double(budget) / ceiling))
                let x = inset + fraction * max(0, width - inset * 2)
                guard let range, range.high > range.low else {
                    return CGPoint(x: x, y: plotHeight / 2)
                }
                let normalised = (seconds - range.low) / (range.high - range.low)
                return CGPoint(
                    x: x, y: inset + CGFloat(normalised) * max(0, plotHeight - inset * 2))
            }
            let hatchWidth =
                width * CGFloat(min(1, Double(plan.profile.refusalThresholdBytes) / ceiling))
            let budgetX = inset
                + CGFloat(min(1, Double(plan.budgetBytes) / ceiling)) * max(0, width - inset * 2)

            ZStack(alignment: .topLeading) {
                Rectangle().fill(MRColor.panel)

                // Refused: no run, therefore no projection, therefore nothing
                // drawn but hatching — and the hatching says why, whenever the
                // region is wide enough to hold the word.
                ZStack {
                    Hatch(color: MRColor.refuse)
                    if hatchWidth >= 26 {
                        Text("refused")
                            .font(MRType.label)
                            .tracking(0.6)
                            .textCase(.uppercase)
                            .foregroundStyle(MRColor.refuse)
                            .lineLimit(1)
                            .fixedSize()
                            .rotationEffect(.degrees(-90))
                    }
                }
                .frame(width: hatchWidth, height: plotHeight)

                // Where the stated budget sits. Dashed, so it reads as a marker
                // the operator placed and never as data.
                Path { path in
                    path.move(to: CGPoint(x: budgetX, y: 0))
                    path.addLine(to: CGPoint(x: budgetX, y: plotHeight))
                }
                .stroke(
                    MRColor.primary.opacity(0.45),
                    style: StrokeStyle(lineWidth: 1, dash: [3, 3]))

                if !samples.isEmpty {
                    Path { path in
                        var started = false
                        for sample in samples {
                            guard let projection = sample.projection else { continue }
                            let point = position(sample.budgetBytes, projection.mid)
                            if started { path.addLine(to: point) } else {
                                path.move(to: point)
                                started = true
                            }
                        }
                    }
                    .stroke(
                        MRColor.streamDet,
                        style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                }

                if let projection = SpeedProjector.project(
                    plan: plan, terms: terms, link: verifiedLink)
                {
                    Circle()
                        .fill(MRColor.panel)
                        .frame(width: 11, height: 11)
                        .overlay(Circle().strokeBorder(MRColor.streamDet, lineWidth: 2.5))
                        .position(position(plan.budgetBytes, projection.mid))
                }

                // Runs that happened. Filled, ringed in the panel colour so a
                // dot that lands on the curve is still a separate object.
                ForEach(verifiedMeasured) { record in
                    Circle()
                        .fill(MRColor.primary)
                        .frame(width: 7, height: 7)
                        .overlay(Circle().strokeBorder(MRColor.panel, lineWidth: 1.5))
                        .position(position(record.budgetBytes, record.secondsPerToken))
                        .help(
                            "measured \(MRFormat.clock(record.secondsPerToken))/token at "
                                + MRFormat.bytesDecimal(record.budgetBytes))
                }

                if samples.isEmpty {
                    Text(
                        verifiedMeasured.isEmpty
                            ? ProjectionPresentation.notMeasuredTitle : "measured runs only"
                    )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .lineLimit(1)
                    // Centred in what is left after the refusal region, not in
                    // the whole plot: the sentence is about the part of the
                    // domain a run could actually happen in.
                    .frame(width: max(0, width - hatchWidth), height: plotHeight)
                    .offset(x: hatchWidth)
                }
            }
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: MRRadius.control, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: MRRadius.control, style: .continuous)
                .strokeBorder(MRColor.hairline, lineWidth: 1)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("projected seconds per token across the budget range")
        .accessibilityValue(spokenPlot)
    }

    /// Budget, from nothing to the device's ceiling — the same domain as the
    /// slider above, so the two can be read against each other. One line each
    /// and never wrapped: a caption free to grow vertically inside an `HStack`
    /// reports its height at one word per line, and in a panel that is a whole
    /// screen's worth of blank space.
    private var axisLabels: some View {
        HStack(spacing: MRSpace.s2) {
            Text("0")
            Spacer(minLength: 0)
            Text("stated \(MRFormat.bytesDecimal(plan.budgetBytes))")
            Spacer(minLength: 0)
            Text(MRFormat.bytesDecimal(plan.deviceCeilingBytes))
        }
        .font(MRType.micro)
        .foregroundStyle(MRColor.tertiary)
        .lineLimit(1)
        .accessibilityHidden(true)
    }

    private var spokenPlot: String {
        if verifiedLink == nil {
            return verifiedMeasured.isEmpty
                ? "not available; no measured run has happened"
                : "\(verifiedMeasured.count) measured runs plotted; no projected curve"
        }
        return SpeedProjector.project(plan: plan, terms: terms, link: verifiedLink)?.spokenString
            ?? "no projection at this budget; it is refused"
    }

    private func provenance(link: VolumeCalibration) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            Button {
                withAnimation(.easeOut(duration: MRMotion.quick)) { showingProvenance.toggle() }
            } label: {
                HStack(spacing: MRSpace.s1) {
                    Image(systemName: showingProvenance ? "chevron.down" : "chevron.right")
                        .imageScale(.small)
                    Text(link.isStale() ? "≈ from a stale calibration" : "≈ where this comes from")
                        .font(MRType.micro)
                }
                .foregroundStyle(link.isStale() ? MRColor.caution : MRColor.tertiary)
            }
            .buttonStyle(.plain)

            if showingProvenance {
                Text(link.provenanceSentence)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(
                    "The ring is a projection from this link measurement. Completed runs are "
                        + "shown separately as filled points."
                )
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
                .fixedSize(horizontal: false, vertical: true)
                if link.isStale() {
                    Text(
                        "This calibration is more than seven days old. It is labelled stale "
                            + "rather than quietly reused."
                    )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.caution)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

enum ProjectionPresentation {
    static let notMeasuredTitle = "Not measured"
    static let notMeasuredMessage =
        "No matching model, artifact, and device measurement exists, so no speed is shown."
}
