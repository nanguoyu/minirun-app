import MinirunKit
import SwiftUI

enum InstrumentPanelPresentation: Equatable {
    case idle
    case preparing
    case measurements
    case unavailable

    init(snapshot: RunSnapshot) {
        if snapshot.state == .idle {
            self = .idle
        } else if snapshot.acceptance != nil {
            self = .measurements
        } else if snapshot.state.isRunning {
            self = .preparing
        } else {
            self = .unavailable
        }
    }
}

enum InstrumentHelpCopy {
    static let byteFlow =
        "Shows successful model-weight reads completed into Minirun in fixed half-second "
        + "windows. Model layers are shared weights; experts are routed weights. Cached "
        + "reads count too, so this is comparable process data flow, not drive or network speed."

    static let overlap =
        "Shows the share of the routed-expert phase that was not blocked waiting for model "
        + "data. It includes compute and other work, so it is not a disk-utilization measure. "
        + "When read-ahead evidence exists, the second line reports how much staged read time "
        + "finished before demand and the largest staging buffer."

    /// The one heading both storage charts now sit under, so the section can
    /// explain the pair rather than each chart explaining itself beside the
    /// other.
    static let storageSection =
        "Two views of what this run is getting out of storage. Data flow is successful "
        + "model-weight reads completed into Minirun in fixed half-second windows — model "
        + "layers are shared weights, experts are routed weights, and cached reads count, so "
        + "it is process data flow rather than drive or network speed. Overlap is the share "
        + "of the routed-expert phase that was not blocked waiting for that data; it includes "
        + "compute, so it is not a disk-utilization measure. When read-ahead evidence exists, "
        + "the last line reports how much staged read time finished before demand and the "
        + "largest staging buffer."

    /// The chip that used to read `Accounting waiting`.
    static let byteCounts =
        "Minirun checks that the model-layer bytes and the expert bytes a run reports add up "
        + "to exactly the total it reports. That total settles at a pass boundary, so until "
        + "this run finishes its first pass there is nothing to check yet and the chip reads "
        + "pending. Balanced means every reported byte was counted once; mismatch names the "
        + "identity that failed."

    static let phaseSplit =
        "Splits each completed pass's wall time into the parts the model measured while it "
        + "ran — waiting for weights, computing, checking and reclaiming. Prefill is the "
        + "prompt; decode is the mean of the passes that followed it. The last slice is the "
        + "time no measurement claimed, and it is shown at its real size rather than hidden."

    static let expertReuse =
        "Shows whether routed expert weights were reused from memory or loaded from storage. "
        + "When reuse is disabled for the current run, it says Off and reports storage loads "
        + "instead of a misleading zero-percent hit rate."
}

/// It renders one immutable `RunSnapshot` and reaches into nothing. Geometry
/// interpolates; digits never do — an animated number misrepresents when it was
/// measured.
struct InstrumentPanelView: View {
    let snapshot: RunSnapshot

    var body: some View {
        ScrollView {
            sections
                .padding(MRSpace.s4)
        }
        // Same rule as the chat-settings panel: a panel takes the platform's
        // panel background, not a colour of ours that only matches it by luck.
        .scrollContentBackground(.hidden)
        .mrWorkspaceSurface()
        .accessibilityElement(children: .contain)
        .accessibilityLabel("instrument panel")
    }

    /// The panel's contents, outside its scroller.
    ///
    /// Split out because this project verifies a screen by looking at it, and
    /// `ScrollView` plus an `ignoresSafeArea` background renders empty through
    /// `ImageRenderer` — the offscreen review of the section list would
    /// otherwise be a picture of the background colour.
    @ViewBuilder var sections: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            switch InstrumentPanelPresentation(snapshot: snapshot) {
            case .idle:
                // On the phone this is the whole of a full-height sheet.
                // Left-aligned in the top corner of an otherwise empty
                // screen it read as content that had failed to load; the
                // Mac inspector is a narrow column where leading alignment
                // is right.
                #if os(iOS)
                    EmptyStateView(
                        systemImage: "waveform.path.ecg",
                        headline: "No measurements yet",
                        message: "Send a message to measure decode rate, model data flow, memory use, storage overlap, and expert reuse.")
                        .frame(minHeight: 420)
                #else
                    emptyState(
                        title: "No measurements yet",
                        message: "Send a message to measure decode rate, model data flow, memory use, storage overlap, and expert reuse.",
                        systemImage: "waveform.path.ecg")
                #endif
            case .preparing:
                chips
                currentStage
                emptyState(
                    title: "Preparing measurements…",
                    message: "Metrics will appear after the model accepts this run.",
                    systemImage: "hourglass")
            case .measurements:
                chips
                currentStage
                rateAndBytes
                budgetGauge
                phaseSplit
                storage
                cache
                thermalTrail
            case .unavailable:
                chips
                emptyState(
                    title: "No run measurements",
                    message: "The model did not accept this run, so no runtime measurements were recorded.",
                    systemImage: "waveform.slash")
            }
        }
    }

    // MARK: - The compact form
    //
    // On the phone the three readouts that matter live INLINE, pinned above the
    // prompt bar, and the sheet carries only the expanded cockpit.
    //
    // The specification asks for a bottom sheet with a 120 pt detent that never
    // collapses to nothing. A sheet cannot do that and leave the prompt bar and
    // the tab bar reachable — it is anchored to the bottom of the screen and
    // covers both. Inlining the compact form keeps the promise the detent was
    // making (three readouts always on screen, never zero) and keeps Run
    // tappable, which the detent version did not. See DESIGN.md.
    struct CompactStrip: View {
        let snapshot: RunSnapshot
        var onExpand: () -> Void

        var body: some View {
            Button(action: onExpand) {
                HStack(alignment: .center, spacing: MRSpace.s4) {
                    readout("decode", MRFormat.tokenRate(rate))
                    readout("data/token", bytesPerToken)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("footprint").mrLabel()
                        gauge
                    }
                    Spacer(minLength: 0)
                    BalanceDot(status: snapshot.accountingStatus)
                    Image(systemName: "chevron.up")
                        .imageScale(.small)
                        .foregroundStyle(MRColor.tertiary)
                }
                .padding(.horizontal, MRSpace.s4)
                .padding(.vertical, MRSpace.s2)
                .frame(maxWidth: .infinity)
                .background(MRColor.panel)
                .overlay(alignment: .top) {
                    Rectangle().fill(MRColor.hairline).frame(height: 1)
                }
            }
            .buttonStyle(.plain)
            .mrMinimumTouchTarget()
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("instrument panel, compact")
            .accessibilityValue(
                "decode \(MRFormat.tokenRateSpoken(rate)), \(bytesPerToken) per decode token, "
                    + footprintSpoken + ", "
                    + MetricEvidencePresentation.accessibilityValue(
                        for: snapshot.accountingStatus))
            .accessibilityHint("opens the full cockpit")
        }

        private func readout(_ label: String, _ value: String) -> some View {
            VStack(alignment: .leading, spacing: 2) {
                Text(label).mrLabel()
                Text(value)
                    .font(MRType.metric)
                    .foregroundStyle(MRColor.primary)
                    .animation(nil, value: value)
            }
        }

        private var rate: Double? {
            snapshot.tokensPerSecond
        }

        private var bytesPerToken: String {
            snapshot.bytesPerToken.map { MRFormat.bytesDecimal($0) } ?? "—"
        }

        @ViewBuilder private var gauge: some View {
            if let telemetry = snapshot.telemetry {
                GeometryReader { geometry in
                    let fraction = min(
                        1.2,
                        Double(telemetry.footprintBytes)
                            / Double(max(1, telemetry.declaredBudgetBytes)))
                    ZStack(alignment: .leading) {
                        Rectangle().fill(MRColor.hairline)
                        Rectangle()
                            .fill(snapshot.budgetBreached ? MRColor.refuse : MRColor.tierPinned)
                            .frame(width: geometry.size.width * CGFloat(min(1, fraction)))
                    }
                }
                .frame(width: 72, height: 10)
                .clipShape(RoundedRectangle(cornerRadius: 2, style: .continuous))
            } else {
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(MRColor.hairline)
                    .frame(width: 72, height: 10)
            }
        }

        private var footprintSpoken: String {
            guard let telemetry = snapshot.telemetry else { return "not reported" }
            return "footprint \(MRFormat.bytesDecimal(telemetry.footprintBytes)) of "
                + "\(MRFormat.bytesDecimal(telemetry.declaredBudgetBytes)) stated"
        }
    }

    // MARK: H · Status chips

    private var chips: some View {
        HStack(spacing: MRSpace.s2) {
            if let telemetry = snapshot.telemetry {
                StatusChip(
                    text: telemetry.thermalState.rawValue,
                    tone: thermalTone(telemetry.thermalState),
                    systemImage: "thermometer.medium")
                    .accessibilityLabel("thermal state")
                    .accessibilityValue(
                        "\(telemetry.thermalState.rawValue), reported by this run")
                if telemetry.lowPowerMode {
                    StatusChip(text: "low power", tone: .caution, systemImage: "battery.25")
                }
            } else {
                StatusChip(text: stateWord, tone: .neutral)
            }
            Spacer()
            AccountingStatusChip(status: snapshot.accountingStatus)
            HelpTip(
                InstrumentHelpCopy.byteCounts,
                accessibilityLabel: MRAccessibility.helpLabel(subject: "byte counts"))
        }
    }

    private var stateWord: String {
        switch snapshot.state {
        case .idle: return "idle"
        case .validating: return "validating"
        case .running: return "running"
        case .finished: return "finished"
        case .cancelled: return "stopped"
        case .halted: return "halted"
        case .refused: return "refused"
        }
    }

    private func thermalTone(_ state: ThermalStateName) -> StatusChip.Tone {
        switch state {
        case .nominal: return .ok
        case .fair: return .caution
        case .serious, .critical: return .refuse
        case .unknown: return .neutral
        }
    }

    // MARK: Current stage

    /// Stage and layer progress are one instrument. Keeping the grid here
    /// avoids presenting the same progress twice in a narrow inspector.
    @ViewBuilder private var currentStage: some View {
        if snapshot.state.isRunning || snapshot.phase != nil {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                HStack(alignment: .firstTextBaseline) {
                    Text(stageTitle).mrLabel()
                    if let detail = stageDetail {
                        Text(detail)
                            .font(MRType.micro)
                            .foregroundStyle(MRColor.secondary)
                    }
                    Spacer()
                    Text(layerProgress)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.secondary)
                        .animation(nil, value: layerProgress)
                }
                if !snapshot.ladder.isEmpty {
                    #if os(iOS)
                        LayerLadder(cells: snapshot.ladder, wraps: true)
                    #else
                        LayerLadder(cells: snapshot.ladder, cellHeight: 18)
                    #endif
                }
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("current stage")
            .accessibilityValue(
                [stageTitle, stageDetail, layerProgress].compactMap { $0 }.joined(separator: ", "))
        }
    }

    private var generationStage: RunGenerationStage? { snapshot.generationStage }

    private var stageTitle: String {
        switch generationStage {
        case .prefill: return "Prefill"
        case .decode: return "Decode"
        case .terminal:
            switch snapshot.state {
            case .finished: return "Complete"
            case .cancelled: return "Stopped"
            case .halted: return "Halted"
            default: return "Complete"
            }
        case .preparing: return "Preparing"
        case nil:
            guard let name = snapshot.phase?.name else { return "Preparing" }
            if name == "prefill" { return "Prefill" }
            if name.hasPrefix("decode ") { return "Decode" }
            return "Preparing"
        }
    }

    private var stageDetail: String? {
        guard let phase = snapshot.phase else { return nil }
        switch generationStage {
        case .prefill: return "Reading prompt"
        case .decode:
            let components = phase.name.split(separator: " ")
            return components.last.flatMap { Int($0) }.map { "Token \($0)" } ?? "Generating"
        case .preparing, nil:
            return phase.detail.isEmpty ? nil : phase.detail
        case .terminal: return nil
        }
    }

    // MARK: A · Rate · B · Bytes per token

    /// Two readouts, and which two depends on the stage.
    ///
    /// Prompt ingestion takes minutes on a flagship artifact, and the decode
    /// pair has nothing to say for any of it: no decode token has completed, so
    /// both read "—" for the whole prefill. During prefill the same two slots
    /// therefore report what the run has actually published — read throughput,
    /// and how far through the layers this pass is — and they change back the
    /// moment the first decode pass begins.
    private var rateAndBytes: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .top, spacing: MRSpace.s4) {
                if generationStage == .prefill {
                    MetricReadout(
                        label: "Prefill read rate",
                        value: prefillRateText,
                        provenance: prefillRateProvenance,
                        spokenValue: prefillRateSpoken)
                    MetricReadout(
                        label: "Prefill progress",
                        value: prefillProgressText,
                        provenance: prefillRemainingText,
                        spokenValue: prefillProgressSpoken)
                } else {
                    MetricReadout(
                        label: "Decode rate",
                        value: MRFormat.tokenRate(rate),
                        provenance: rateProvenance,
                        spokenValue: MRFormat.tokenRateSpoken(rate))
                    MetricReadout(
                        label: "Decode data / token",
                        value: bytesPerTokenText,
                        spokenValue: bytesPerTokenSpoken)
                }
            }
            if let split = byteSplit {
                Text(split)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .animation(nil, value: split)
            }
            // During prefill the read-rate provenance already prints this same
            // total beside the elapsed clock it was divided by; printing it
            // twice is one fact wearing two labels.
            if let current = currentTokenBytes, generationStage != .prefill {
                Text("\(currentPassLabel) · \(MRFormat.bytesDecimal(current)) read so far")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .animation(nil, value: current)
            }
        }
    }

    private var rate: Double? {
        snapshot.tokensPerSecond
    }

    private var rateProvenance: String? {
        guard let seconds = snapshot.telemetry?.decodeSeconds,
            seconds.isFinite, seconds >= 0
        else { return nil }
        let count = snapshot.decodeTokensCompleted
        return "\(MRFormat.clock(seconds)) decode · \(count) token\(count == 1 ? "" : "s")"
    }

    private var bytesPerTokenText: String {
        snapshot.bytesPerToken.map { MRFormat.bytesDecimal($0) } ?? "—"
    }

    private var bytesPerTokenSpoken: String {
        "\(bytesPerTokenText) per decode token"
    }

    private var byteSplit: String? {
        guard let split = snapshot.byteSplitPerToken else { return nil }
        return "model layers \(MRFormat.bytesDecimal(split.deterministic)) · experts "
            + "\(MRFormat.bytesDecimal(split.expert))"
    }

    // MARK: The prefill pair

    private var prefillRateText: String {
        snapshot.readBytesPerSecond.map { MRFormat.throughput($0) } ?? "—"
    }

    /// The two numbers the rate was divided from, named. A throughput with no
    /// visible numerator or denominator is a claim; this is arithmetic the
    /// reader can check.
    private var prefillRateProvenance: String? {
        guard let telemetry = snapshot.telemetry, telemetry.reportsByteAccounting else {
            return nil
        }
        return "\(MRFormat.bytesDecimal(telemetry.bytes.totalBytesRead)) read · "
            + "\(MRFormat.clock(telemetry.elapsed)) elapsed"
    }

    private var prefillRateSpoken: String {
        guard let rate = snapshot.readBytesPerSecond else { return "not reported" }
        return "\(MRFormat.throughput(rate)) of model data read"
    }

    private var prefillProgressText: String {
        guard let progress = snapshot.passLayerProgress else { return "—" }
        return "\(progress.completed) / \(progress.total)"
    }

    /// The estimate is labelled as one, every time it is shown. When it cannot
    /// be derived yet the line says what the numerator is instead of hiding.
    private var prefillRemainingText: String? {
        if let seconds = snapshot.prefillRemainingSecondsEstimate {
            return "≈ \(MRFormat.clock(seconds)) left, estimated"
        }
        return snapshot.passLayerProgress == nil ? nil : "layers read this pass"
    }

    private var prefillProgressSpoken: String {
        guard let progress = snapshot.passLayerProgress else { return "not reported" }
        var spoken = "\(progress.completed) of \(progress.total) layers read this pass"
        if let seconds = snapshot.prefillRemainingSecondsEstimate {
            spoken += ", an estimated \(MRFormat.clock(seconds)) remaining"
        }
        return spoken
    }

    private var currentTokenBytes: UInt64? {
        snapshot.currentTokenBytes?.totalBytesRead
    }

    private var currentPassLabel: String {
        generationStage == .prefill ? "Current prefill" : "Current token"
    }

    // MARK: C · Budget gauge

    @ViewBuilder private var budgetGauge: some View {
        if let telemetry = snapshot.telemetry {
            BudgetGauge(
                footprintBytes: telemetry.footprintBytes,
                peakBytes: telemetry.peakFootprintBytes,
                declaredBudgetBytes: telemetry.declaredBudgetBytes,
                latchedBreach: snapshot.budgetBreached)
        } else {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                Text("Footprint").mrLabel()
                Text("—")
                    .font(MRType.metric)
                    .foregroundStyle(MRColor.primary)
                Text(footprintPendingText)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("footprint against stated budget")
            .accessibilityValue("not reported")
        }
    }

    private var footprintPendingText: String {
        guard let declared = snapshot.acceptance?.declaredBudgetBytes else {
            return "Not reported"
        }
        return "Budget \(MRFormat.bytesDecimal(declared))"
    }

    private var layerProgress: String {
        LayerLadderAccessibility.visibleSummary(for: snapshot.ladder)
    }

    // MARK: Where the time went

    /// One bar per pass kind, from the decomposition the engine published at
    /// each pass boundary. It appears only once a pass has been attributed:
    /// during the first prefill there is nothing to split yet, and a bar drawn
    /// from a pass in flight would be a picture of an unfinished measurement.
    @ViewBuilder private var phaseSplit: some View {
        let bars = snapshot.phaseBars
        if !bars.isEmpty {
            VStack(alignment: .leading, spacing: MRSpace.s3) {
                HStack(spacing: MRSpace.s1) {
                    Text("Where the time went").mrLabel(MRColor.secondary)
                        .accessibilityAddTraits(.isHeader)
                    HelpTip(
                        InstrumentHelpCopy.phaseSplit,
                        accessibilityLabel: MRAccessibility.helpLabel(
                            subject: "where the time went"))
                    Spacer(minLength: 0)
                }
                ForEach(bars, id: \.passKind) { bar in
                    PhaseSplitBar(summary: bar)
                }
            }
            .padding(MRSpace.s3)
            .frame(maxWidth: .infinity, alignment: .leading)
            .mrCard()
            .accessibilityElement(children: .contain)
            .accessibilityLabel("where the time went")
        }
    }

    // MARK: E · F · Storage — one section, two measurements

    /// Data flow and overlap are two views of one subject: what the run is
    /// getting out of storage. Presented as two independent charts with two
    /// headings and two vocabularies they read as competing instruments, which
    /// is exactly how the review received them. They are one section now — the
    /// ribbon over the overlap bar, one ⓘ, and one legend naming the two
    /// streams — and both measurements are unchanged.
    private var storage: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            HStack(spacing: MRSpace.s1) {
                Text("Storage").mrLabel(MRColor.secondary)
                    .accessibilityAddTraits(.isHeader)
                HelpTip(
                    InstrumentHelpCopy.storageSection,
                    accessibilityLabel: MRAccessibility.helpLabel(subject: "storage"))
                Spacer(minLength: 0)
            }
            if snapshot.flow.isEmpty {
                pendingInstrument(
                    "Data flow", "Waiting for the first half-second window",
                    help: InstrumentHelpCopy.byteFlow)
            } else {
                // The legend sits directly under the lanes it names, and there
                // is only this one: the overlap bar below measures time, not
                // streams, and gets its own sentence instead.
                ByteFlowRibbon(samples: snapshot.flow, title: "Data flow", showsHelp: false)
            }
            if snapshot.stalls?.overlapFraction != nil {
                OverlapMeter(
                    stalls: snapshot.stalls, readAhead: snapshot.readAhead,
                    title: "Overlap", showsHelp: false)
            }
        }
        .padding(MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard()
        .accessibilityElement(children: .contain)
        .accessibilityLabel("storage")
    }

    // MARK: G · Cache strip

    @ViewBuilder private var cache: some View {
        if snapshot.cache != nil {
            VStack(alignment: .leading, spacing: MRSpace.s1) {
                HStack(spacing: MRSpace.s1) {
                    Text("Expert reuse").mrLabel()
                    HelpTip(
                        InstrumentHelpCopy.expertReuse,
                        accessibilityLabel: MRAccessibility.helpLabel(
                            subject: "expert reuse"))
                }
                CacheStripView(cache: snapshot.cache)
            }
        }
    }

    private func pendingInstrument(_ title: String, _ message: String, help: String) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            HStack(spacing: MRSpace.s1) {
                Text(title).mrLabel()
                HelpTip(
                    help,
                    accessibilityLabel: MRAccessibility.helpLabel(subject: title))
            }
            Text(message)
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(title)
        .accessibilityValue("pending: \(message)")
    }

    private func emptyState(title: String, message: String, systemImage: String) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            Image(systemName: systemImage)
                .font(.system(size: 24, weight: .regular))
                .foregroundStyle(MRColor.tertiary)
                .accessibilityHidden(true)
            Text(title)
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)
                .accessibilityAddTraits(.isHeader)
            Text(message)
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, MRSpace.s3)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var thermalTrail: some View {
        if !snapshot.thermalTrail.isEmpty {
            VStack(alignment: .leading, spacing: MRSpace.s1) {
                Text("Thermal trail").mrLabel()
                ForEach(Array(snapshot.thermalTrail.enumerated()), id: \.offset) { _, item in
                    Text(
                        "\(item.state.rawValue) at \(MRFormat.clock(item.atSeconds)) "
                            + "during \(item.phase)"
                    )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                }
            }
        }
    }
}
