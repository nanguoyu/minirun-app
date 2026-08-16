import MinirunKit
import SwiftUI

/// The memory dial — the soul of the product.
///
/// One control that states how much RAM a run may use, with the tier split
/// visible. The thumb may be dragged into the refusal zone and nothing snaps
/// back; what happens instead is a card that names the error and the exact byte
/// deficit, and a Send button that goes away until the operator changes their
/// mind.
///
/// It takes a plan and hands back a number. It deliberately does not know whose
/// budget it is drawing — the same view serves a conversation's stated budget
/// and the default a new conversation would start from, and the `role` is what
/// tells the operator which one they are looking at. A dial that could not say
/// which of the two it was editing would be the settings confusion this screen
/// exists to end.
struct MemoryDialView: View {

    enum Presentation: Equatable {
        /// The controls needed to set a chat's budget.
        case compact
        /// The model/default-budget surface uses roomier spacing. Timing is
        /// reported by completed chats, not by an empty projection card.
        case detailed
    }

    enum Role {
        /// This conversation's stated budget. What a turn will actually run at.
        case conversation
        /// The value a NEW conversation with this model would start from.
        case defaultForNewChats

        var title: String {
            switch self {
            case .conversation: return "This chat's memory budget"
            case .defaultForNewChats: return "Default budget for new chats"
            }
        }

        /// What the role means, kept verbatim but behind the header's ⓘ. The
        /// title already says which budget this is; the rest is for the one
        /// time somebody wonders.
        var help: String {
            switch self {
            case .conversation:
                return "Stated for every turn in this conversation until you restate it. "
                    + "It cannot be changed while a turn is running."
            case .defaultForNewChats:
                return "Copied onto a new conversation when it is created. Changing it never "
                    + "touches a conversation that already exists."
            }
        }
    }

    let plan: BudgetPlan
    let entry: CatalogEntry
    var role: Role = .conversation
    var presentation: Presentation = .detailed
    /// A run screen shows this read-only: the budget stated at start, not a
    /// live-editable value.
    var readOnly = false
    /// Nil hides the read-ahead control entirely — used where the depth is not
    /// this screen's to state.
    var readAheadDepth: Binding<Int?>?
    let onChange: (UInt64) -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var motionPolicy: MRAccessibleMotionPolicy {
        MRAccessibleMotionPolicy(reduceMotion: reduceMotion)
    }

    var body: some View {
        VStack(
            alignment: .leading,
            spacing: presentation == .compact ? MRSpace.s2 : MRSpace.s5
        ) {
            header
            if plan.deviceCannotHostModel { deviceTooSmall }
            presets
            dial
            compactSummary
            if plan.refusal != nil {
                RefusalCard(plan: plan) { target in
                    withAnimation(motionPolicy.tierAnimation) { onChange(target) }
                }
            }
            if readAheadDepth != nil { readAhead }
            if readOnly { readOnlyFooter }
        }
    }

    // MARK: Pieces

    /// A title and an ⓘ, and nothing else.
    ///
    /// What stood here was a status readout for the person who wrote the app: a
    /// chip repeating the model already named by the panel's Model picker, a
    /// chip announcing where the numbers came from, and a chip saying
    /// "runnable" — which is what the absence of a refusal card and the presence
    /// of a Send button already say, louder. None of it survived contact with
    /// the question "what does the operator do differently after reading it?"
    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
            Text(role.title)
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)
            HelpTip(
                role.help,
                accessibilityLabel: MRAccessibility.helpLabel(subject: role.title))
            Spacer(minLength: 0)
        }
    }

    private var deviceTooSmall: some View {
        NamedErrorCard(
            headline: "Minirun will not run \(entry.descriptor.displayName) here.",
            message:
                "This device offers \(MRFormat.bytesDecimal(plan.deviceCeilingBytes)). "
                + "\(entry.descriptor.displayName) needs "
                + "\(MRFormat.bytesDecimal(plan.profile.refusalThresholdBytes)) before anything "
                + "else. Downloading it is still allowed — an artifact can live here and run "
                + "elsewhere.",
            namedError: RunError.budgetBelowMinimum(
                declared: plan.deviceCeilingBytes,
                minimum: plan.profile.refusalThresholdBytes,
                model: plan.model
            ).description,
            tone: .refuse)
    }

    /// Three named positions. What the current choice buys is shown in the
    /// summary below the dial; repeating it inside three narrow buttons was the
    /// source of the clipped `pins 13 laye…` labels in the inspector.
    @ViewBuilder private var presets: some View {
        if presentation == .compact {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                HStack(spacing: MRSpace.s2) {
                    Text("Presets")
                        .mrLabel()
                        .frame(width: 60, alignment: .leading)
                    presetButtons
                }
                presetDeficits
            }
        } else {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                Text("Presets").mrLabel()
                presetButtons
                presetDeficits
            }
        }
    }

    private var presetButtons: some View {
        HStack(spacing: MRSpace.s2) {
            ForEach(BudgetPlan.Preset.allCases) { preset in
                let target = plan.budget(for: preset)
                let deficit = plan.presetDeficit(preset)
                let explanation = plan.explanation(for: preset)
                let isSelected = plan.budgetBytes == target
                Button {
                    withAnimation(motionPolicy.tierAnimation) { onChange(target) }
                } label: {
                    HStack(spacing: MRSpace.s1) {
                        VStack(alignment: .leading, spacing: 1) {
                            // One line, always: a chip that hyphenates "Bal-anced"
                            // to make room for its own checkmark reads as broken.
                            Text(preset.rawValue)
                                .font(MRType.caption)
                                .fontWeight(isSelected ? .semibold : .regular)
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                            Text(MRFormat.bytesDecimal(target))
                                .font(MRType.micro)
                                .foregroundStyle(MRColor.tertiary)
                        }
                        Spacer(minLength: 0)
                        if isSelected {
                            Image(systemName: "checkmark")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(MRColor.tierPinned)
                                .accessibilityHidden(true)
                        }
                    }
                    .padding(.horizontal, MRSpace.s2)
                    .padding(.vertical, MRSpace.s1)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: MRRadius.card, style: .continuous)
                            .fill(isSelected ? MRColor.tierPinned.opacity(0.12) : MRColor.raised)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: MRRadius.card, style: .continuous)
                            .strokeBorder(MRColor.hairline, lineWidth: 1)
                    )
                    .contentShape(
                        RoundedRectangle(cornerRadius: MRRadius.card, style: .continuous)
                    )
                }
                .buttonStyle(.plain)
                .mrMinimumTouchTarget()
                .frame(maxWidth: .infinity)
                .disabled(readOnly || deficit != nil)
                .help(
                    deficit.map {
                        "\(explanation) Over this device's ceiling by "
                            + MRFormat.bytesDecimal($0)
                    } ?? explanation)
                .accessibilityHint(explanation)
                .accessibilityValue(isSelected ? "selected" : "not selected")
            }
        }
    }

    @ViewBuilder private var presetDeficits: some View {
        // A preset over the ceiling is shown disabled with the deficit, never hidden.
        ForEach(BudgetPlan.Preset.allCases) { preset in
            if let deficit = plan.presetDeficit(preset) {
                Text(
                    "\(preset.rawValue) needs \(MRFormat.bytesDecimal(deficit)) more than this "
                        + "device offers."
                )
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
            }
        }
    }

    /// The slider, on the device's domain, because that is the domain a budget
    /// is chosen out of.
    private var dial: some View {
        MemoryDial(plan: plan, onChange: readOnly ? nil : onChange)
    }

    /// The answer to the two questions a person changes the slider to answer.
    /// Values wrap as rows rather than competing for three fixed columns.
    private var compactSummary: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            if plan.strategy == .boundedLayerStreaming {
                HStack(alignment: .top, spacing: MRSpace.s3) {
                    compactMetric("Held resident", plan.residentUnitsLabel)
                    compactMetric(
                        plan.pinnedBytes == 0 ? "Read per token" : "Later-token read",
                        MRFormat.bytesDecimal(plan.bytesReadPerToken))
                }
                Text(
                    "V4 holds the deterministic layers this budget can pay for — and the output "
                        + "head above them — and streams the rest, replacing each layer's old "
                        + "causal cache before advancing. The runner stops rather than crossing "
                        + "this budget."
                )
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                if let next = plan.nextPinSentence {
                    Text(next)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.tertiary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                HStack(alignment: .top, spacing: MRSpace.s3) {
                    compactMetric(
                        "Pinned layers",
                        plan.pinnedLayerCount == 0
                            ? "None" : "\(plan.pinnedLayerCount) of \(plan.profile.layerCount)")
                    compactMetric(
                        plan.pinnedLayerCount == 0 ? "Read per token" : "Later-token read",
                        MRFormat.bytesDecimal(plan.bytesReadPerToken))
                }
                if let reuse = plan.pinReuseSentence {
                    Text(reuse)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.tertiary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let next = plan.nextPinSentence {
                    Text(next)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.tertiary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(MRSpace.s2)
        .mrCard(MRColor.raised, radius: MRRadius.control)
    }

    private func compactMetric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
            Text(value)
                .font(MRType.metric)
                .foregroundStyle(MRColor.primary)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Two named choices matching the runtime's measured 0/1 contract.
    ///
    /// The control this replaces said "unset — the runner's own default", which
    /// named a default without saying what it was, and by spec v0.6.19 was also
    /// wrong about it: the prefetchers default on wherever a budget is stated.
    /// So auto means one layer ahead, while off is the serial troubleshooting
    /// arm. Older saved values above one are preserved and shown with an
    /// explicit recovery choice; they are never projected as depth one.
    @ViewBuilder private var readAhead: some View {
        if let binding = readAheadDepth {
            let setting = ReadAheadSetting(binding.wrappedValue)
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                HStack(spacing: MRSpace.s2) {
                    HStack(spacing: MRSpace.s2) {
                        Text("Read-ahead").mrLabel()
                        HelpTip(
                            setting.explanation,
                            accessibilityLabel: MRAccessibility.helpLabel(subject: "read-ahead"))
                    }
                    if presentation == .compact { Spacer(minLength: 0) }
                    if setting.isProductSupported {
                        Picker(
                            "Read-ahead",
                            selection: Binding(
                                get: { mode(of: setting) },
                                set: { binding.wrappedValue = knob(for: $0) })
                        ) {
                            Text("Auto").tag(Mode.auto)
                            Text("Off").tag(Mode.off)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .mrMinimumTouchTarget()
                        .disabled(readOnly)
                        .frame(maxWidth: presentation == .compact ? 170 : .infinity)
                    }
                }

                if let refusal = plan.readAheadRefusal {
                    VStack(alignment: .leading, spacing: MRSpace.s2) {
                        Label(
                            "Choose a supported read-ahead mode",
                            systemImage: "exclamationmark.circle.fill"
                        )
                        .font(MRType.body)
                        .foregroundStyle(MRColor.caution)
                        Text(refusal.message)
                            .font(MRType.caption)
                            .foregroundStyle(MRColor.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        HStack(spacing: MRSpace.s2) {
                            Button("Use Auto") { binding.wrappedValue = nil }
                                .buttonStyle(.borderedProminent)
                                .mrMinimumTouchTarget()
                            Button("Turn Off") { binding.wrappedValue = 0 }
                                .buttonStyle(.bordered)
                                .mrMinimumTouchTarget()
                        }
                        .controlSize(.small)
                    }
                    .padding(MRSpace.s3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .mrCard(MRColor.raised, stroke: MRColor.caution.opacity(0.5))
                    .disabled(readOnly)
                }

            }
        }
    }

    private enum Mode: Hashable { case auto, off }

    private func mode(of setting: ReadAheadSetting) -> Mode {
        switch setting {
        case .auto, .explicit(1): return .auto
        case .off: return .off
        case .explicit: return .auto
        }
    }

    private func knob(for mode: Mode) -> Int? {
        switch mode {
        case .auto: return nil
        case .off: return 0
        }
    }

    private var readOnlyFooter: some View {
        Text(
            "Stop the turn to restate the budget. Minirun does not change a budget mid-run."
        )
        .font(MRType.caption)
        .foregroundStyle(MRColor.caution)
        .fixedSize(horizontal: false, vertical: true)
    }
}
