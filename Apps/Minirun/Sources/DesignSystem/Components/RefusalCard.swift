import SwiftUI

/// The behaviour this product is named for.
///
/// The thumb may be dragged into the refusal zone; nothing snaps back. What
/// happens instead is this card: the required budget, the exact deficit, and a
/// way out that the operator chooses. No configuration is
/// ever quietly adjusted to make a run possible.
struct RefusalCard: View {
    let plan: BudgetPlan
    var onRaise: ((UInt64) -> Void)?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var motionPolicy: MRAccessibleMotionPolicy {
        MRAccessibleMotionPolicy(reduceMotion: reduceMotion)
    }

    var body: some View {
        guard let refusal = plan.refusal else { return AnyView(EmptyView()) }
        return AnyView(card(refusal))
    }

    private func card(_ refusal: BudgetRefusal) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            Text(headline(refusal))
                .font(MRType.headline)
                .foregroundStyle(MRColor.refuse)

            VStack(alignment: .leading, spacing: MRSpace.s2) {
                ForEach(
                    Array(Self.bodyLines(plan: plan, refusal: refusal).enumerated()), id: \.offset
                ) { _, line in
                    Text(line)
                        .font(MRType.body)
                        .foregroundStyle(MRColor.primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Text(
                "Minirun refuses this configuration rather than quietly using more memory "
                    + "than you stated."
            )
            .font(MRType.caption)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)

            if let target = refusal.suggestedBudgetBytes, let onRaise {
                Button("Raise to \(MRFormat.bytesDecimal(target))") { onRaise(target) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .mrMinimumTouchTarget()
            }
        }
        .padding(MRSpace.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(MRColor.panel, stroke: MRColor.refuse.opacity(0.5))
        .transition(motionPolicy.refusalTransition)
        .transaction { transaction in
            if !motionPolicy.animatesGeometry { transaction.animation = nil }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("configuration refused")
        .accessibilityValue(
            "short by \(MRFormat.bytesDecimal(refusal.deficitBytes))")
    }

    private func headline(_ refusal: BudgetRefusal) -> String {
        switch refusal {
        case .belowArithmeticFloor, .belowRequiredMinimum, .belowOnRecordMinimum:
            return "This budget cannot run \(plan.modelName)."
        case .aboveDeviceCeiling:
            return "This device cannot offer that budget."
        }
    }

    /// Product copy names the decision values a person can act on. The complete
    /// planner identity remains covered by arithmetic tests; reproducing its
    /// internal operands here turned a settings card into a debugger.
    static func bodyLines(plan: BudgetPlan, refusal: BudgetRefusal) -> [String] {
        switch refusal {
        case .belowArithmeticFloor(let required, let stated):
            return [
                "Minimum supported budget: \(MRFormat.bytesDecimal(required)).",
                "You stated \(MRFormat.bytesDecimal(stated)). Short by "
                    + "\(MRFormat.bytesDecimal(refusal.deficitBytes)).",
            ]
        case .belowOnRecordMinimum(let minimum, let stated):
            return [
                "Minimum supported budget: \(MRFormat.bytesDecimal(minimum)).",
                "You stated \(MRFormat.bytesDecimal(stated)). Short by "
                    + "\(MRFormat.bytesDecimal(refusal.deficitBytes)).",
            ]
        case .belowRequiredMinimum(let minimum, let stated):
            return [
                "Minimum supported budget: \(MRFormat.bytesDecimal(minimum)).",
                "You stated \(MRFormat.bytesDecimal(stated)). Short by "
                    + "\(MRFormat.bytesDecimal(refusal.deficitBytes)).",
            ]
        case .aboveDeviceCeiling(let ceiling, let stated):
            return [
                "This device offers \(MRFormat.bytesDecimal(ceiling)) to the process.",
                "You stated \(MRFormat.bytesDecimal(stated)) — "
                    + "\(MRFormat.bytesDecimal(refusal.deficitBytes)) more than exists.",
            ]
        }
    }

}

/// A short current-plan fact for an empty hot-set tier. It deliberately carries
/// no historical efficiency ratios: without evidence for this model, artifact
/// revision and device, those figures are not product measurements.
struct HotSetNote: View {
    let plan: BudgetPlan

    var body: some View {
        if plan.isRunnable, plan.hotSetBytes == 0, plan.profile.expertStoredBytes > 0 {
            HStack(alignment: .top, spacing: MRSpace.s2) {
                Image(systemName: "info.circle")
                    .foregroundStyle(MRColor.tertiary)
                    .imageScale(.small)
                Text(
                    "Expert cache: none at this budget. The current plan keeps layers resident "
                        + "first."
                )
                .font(MRType.caption)
                .foregroundStyle(MRColor.tertiary)
                .fixedSize(horizontal: false, vertical: true)
            }
            .accessibilityElement(children: .combine)
        }
    }
}

/// Every error in this product has the same three-part shape: a plain sentence,
/// the exact numbers, and the named error string. No error is ever rendered as
/// prose alone.
struct NamedErrorCard: View {
    let headline: String
    let message: String
    let namedError: String
    var tone: StatusChip.Tone = .refuse
    var numbers: [(String, String)] = []
    var actions: AnyView?

    @State private var showingRaw = false

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            Text(headline)
                .font(MRType.headline)
                .foregroundStyle(tone.color)
                .fixedSize(horizontal: false, vertical: true)

            Text(message)
                .font(MRType.body)
                .foregroundStyle(MRColor.primary)
                .fixedSize(horizontal: false, vertical: true)

            if !numbers.isEmpty {
                VStack(spacing: MRSpace.s1) {
                    ForEach(numbers, id: \.0) { pair in
                        MetricRow(label: pair.0, value: pair.1)
                    }
                }
            }

            DisclosureGroup(isExpanded: $showingRaw) {
                Text(namedError)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, MRSpace.s1)
            } label: {
                Text("Named error").mrLabel()
            }
            .mrMinimumTouchTarget()

            if let actions { actions }
        }
        .padding(MRSpace.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(MRColor.panel, stroke: tone.color.opacity(0.5))
        .accessibilityElement(children: .contain)
        .accessibilityLabel(headline)
    }
}

/// A quiet, product-specific empty state. The optional SF Symbol communicates
/// the surface without leaving a decorative control-shaped placeholder.
struct EmptyStateView: View {
    var systemImage: String?
    let headline: String
    let message: String
    var actionTitle: String?
    var action: (() -> Void)?

    init(
        systemImage: String? = nil, headline: String, message: String,
        actionTitle: String? = nil, action: (() -> Void)? = nil
    ) {
        self.systemImage = systemImage
        self.headline = headline
        self.message = message
        self.actionTitle = actionTitle
        self.action = action
    }

    var body: some View {
        VStack(spacing: MRSpace.s4) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 38, weight: .regular))
                    .foregroundStyle(MRColor.secondary)
                    .accessibilityHidden(true)
            }

            Text(headline)
                .font(MRType.title)
                .foregroundStyle(MRColor.primary)

            Text(message)
                .font(MRType.body)
                .foregroundStyle(MRColor.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
                .fixedSize(horizontal: false, vertical: true)

            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .mrMinimumTouchTarget()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .contain)
    }
}

/// Before any download to a removable volume on iOS. The checkboxes are
/// acknowledgements, not gates — the second one checks itself when the volume
/// resolves.
struct PreconditionCard: View {
    @Binding var powerAcknowledged: Bool
    let driveIsPresent: Bool
    var driveName: String?

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            Text("This drive needs its own power.")
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)

            Text(
                "iPhone cannot bus-power an NVMe enclosure. Use a powered dock or a hub with "
                    + "pass-through power. The reference setup is an ASM2464 enclosure on a "
                    + "powered dock; the phone charges from the same dock."
            )
            .font(MRType.body)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)

            Toggle(isOn: $powerAcknowledged) {
                Text("The enclosure has external power")
                    .font(MRType.caption)
            }
            .toggleStyle(.switch)

            HStack(spacing: MRSpace.s2) {
                Image(systemName: driveIsPresent ? "checkmark.square" : "square")
                    .foregroundStyle(driveIsPresent ? MRColor.ok : MRColor.tertiary)
                Text(
                    driveIsPresent
                        ? "The drive appears below\(driveName.map { " — \($0)" } ?? "")"
                        : "The drive appears below"
                )
                .font(MRType.caption)
                .foregroundStyle(driveIsPresent ? MRColor.primary : MRColor.tertiary)
            }
            .accessibilityElement(children: .combine)
            .accessibilityValue(driveIsPresent ? "drive detected" : "no drive detected yet")

            Text(
                "The security-scoped grant persists across a replug and a reboot, so you are "
                    + "asked to pick the folder once."
            )
            .font(MRType.micro)
            .foregroundStyle(MRColor.tertiary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(MRSpace.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard()
    }
}
