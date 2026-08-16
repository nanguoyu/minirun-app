import SwiftUI

/// One motion decision for every geometry-changing instrument in this file and
/// the refusal/ribbon views that share the same design language. Reduced Motion
/// is an immediate state change: it never inherits a spring from its caller.
enum MRAccessibleMotionPolicy: Equatable {
    case standard
    case reduced

    init(reduceMotion: Bool) {
        self = reduceMotion ? .reduced : .standard
    }

    var animatesGeometry: Bool { self == .standard }
    var usesStaticRibbon: Bool { self == .reduced }
    var tierAnimation: Animation? { animatesGeometry ? MRMotion.tierSpring : nil }
    var refusalTransition: AnyTransition {
        animatesGeometry ? .opacity.combined(with: .move(edge: .top)) : .identity
    }
}

/// The five tiers, drawn across the device's whole memory domain.
///
/// Tiers fill left to right in priority order. The working floor is grey on
/// purpose: it is arithmetic, not an allowance. Right of the thumb is hatched —
/// memory that exists and was not asked for. Left of the floor is hatched in
/// `refuse` — memory that was asked for and cannot run the model.
struct TierBar: View {
    let plan: BudgetPlan
    var height: CGFloat = 26
    /// Draws the refusal boundary tick even when the current budget clears it.
    var showsThreshold = true

    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var ceiling: Double { Double(max(1, plan.deviceCeilingBytes)) }
    private var motionPolicy: MRAccessibleMotionPolicy {
        MRAccessibleMotionPolicy(reduceMotion: reduceMotion)
    }

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let x: (UInt64) -> CGFloat = { bytes in
                CGFloat(min(1, Double(bytes) / ceiling)) * width
            }

            ZStack(alignment: .leading) {
                // The empty track. Everything is drawn on top of this, so a
                // tier that fails to render leaves a visible hole rather than a
                // plausible-looking bar.
                Rectangle().fill(MRColor.tierFree)

                if plan.isRunnable {
                    HStack(spacing: 0) {
                        ForEach(TierItem.all(plan)) { item in
                            segment(item.color, width: x(item.bytes))
                        }
                        Spacer(minLength: 0)
                    }
                } else {
                    // Refused: the tier bar empties to hatching. Nothing is
                    // allocated because nothing was accepted.
                    Hatch(color: MRColor.refuse)
                        .frame(width: x(plan.budgetBytes))
                }

                // Unrequested memory, right of the thumb.
                Hatch(color: contrast == .increased ? MRColor.tertiary : MRColor.hairline)
                    .frame(width: max(0, width - x(plan.budgetBytes)))
                    .offset(x: x(plan.budgetBytes))

                // The refusal zone, always. A budget stated inside it is
                // refused, whatever the thumb currently says, and hatching the
                // whole region makes that a property of the axis rather than
                // something the operator discovers by dragging into it.
                if showsThreshold {
                    Hatch(color: MRColor.refuse)
                        .frame(width: x(plan.profile.refusalThresholdBytes))
                        .opacity(plan.isRunnable ? 0.5 : 1)
                }

                if showsThreshold {
                    Rectangle()
                        .fill(MRColor.refuse)
                        .frame(width: 2)
                        .offset(x: x(plan.profile.refusalThresholdBytes) - 1)
                        .opacity(plan.isRunnable ? 0.55 : 1)
                }

                Rectangle()
                    .fill(MRColor.primary)
                    .frame(width: 2)
                    .offset(x: max(0, x(plan.budgetBytes) - 1))
            }
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: MRRadius.control, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: MRRadius.control, style: .continuous)
                .strokeBorder(
                    contrast == .increased ? MRColor.tertiary : MRColor.hairline, lineWidth: 1)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("memory tiers")
        .accessibilityValue(spokenTiers)
        .animation(motionPolicy.tierAnimation, value: plan.budgetBytes)
        .transaction { transaction in
            if !motionPolicy.animatesGeometry { transaction.animation = nil }
        }
    }

    private func segment(_ color: Color, width: CGFloat) -> some View {
        Rectangle().fill(color).frame(width: max(0, width))
    }

    private var spokenTiers: String {
        guard plan.isRunnable else {
            return "refused: \(MRFormat.bytesDecimal(plan.budgetBytes)) stated, below the "
                + "\(MRFormat.bytesDecimal(plan.profile.refusalThresholdBytes)) this model needs"
        }
        if plan.strategy == .boundedLayerStreaming {
            var parts = [
                "bounded V4 execution floor " + MRFormat.bytesDecimal(plan.floorBytes)
            ]
            if plan.pinnedLayerCount > 0 {
                parts.append(
                    "pinned \(plan.pinnedLayerCount) layers, "
                        + MRFormat.bytesDecimal(plan.pinnedBytes))
            }
            parts.append("budget headroom " + MRFormat.bytesDecimal(plan.freeBytes))
            return parts.joined(separator: ", ")
        }
        var parts = ["working floor \(MRFormat.bytesDecimal(plan.floorBytes))"]
        if plan.stagedBytes > 0 {
            parts.append("staged \(MRFormat.bytesDecimal(plan.stagedBytes))")
        }
        if plan.pinnedLayerCount > 0 {
            parts.append(
                "pinned \(plan.pinnedLayerCount) layers, "
                    + MRFormat.bytesDecimal(plan.pinnedBytes))
        }
        if plan.hotSetBytes > 0 {
            parts.append("hot set \(MRFormat.bytesDecimal(plan.hotSetBytes))")
        }
        parts.append("free \(MRFormat.bytesDecimal(plan.freeBytes))")
        return parts.joined(separator: ", ")
    }
}

/// What the five tiers are, in one place, so the bar and the legend can never
/// disagree about a colour or a name.
///
/// The pinned tier is the one that was missing, and its absence was the bug the
/// owner reported: with no tier for held layers, every byte above the staged
/// pair went to a hot set that the ladder cannot reach and then to free, so
/// dragging the budget **up** grew the empty part of the bar. The ladder's own
/// order is floor, then whatever streams behind compute, then layers held
/// resident at 1.000 bytes saved per resident byte, then the globals bundle at
/// 0.499, then an expert hot set at 0.230 — so on this artifact every byte of
/// surplus below 116,129,117,440 B buys pinned layers, and free grows only past
/// full residency.
struct TierItem: Identifiable {
    let name: String
    let color: Color
    let bytes: UInt64
    let note: String?

    var id: String { name }

    static func all(_ plan: BudgetPlan) -> [TierItem] {
        if plan.strategy == .boundedLayerStreaming {
            // The pinned tier is here for the same reason it is in K3's list,
            // and it was missing for the same reason: V4's dial did not exist
            // when this branch was written, so every byte above the execution
            // envelope went to headroom and raising the budget grew the part
            // of the bar that means "allocated to nothing". Since the V4 dial
            // merged, surplus buys resident deterministic layers at ~0.970
            // bytes saved per resident byte, and headroom grows only once all
            // of them are held.
            return [
                TierItem(
                    name: "Bounded execution", color: MRColor.tierFloor,
                    bytes: plan.floorBytes,
                    note: "layer state replaced in place"),
                TierItem(
                    name: "Pinned", color: MRColor.tierPinned,
                    bytes: plan.pinnedBytes,
                    note: plan.pinnedLayerCount > 0
                        ? "\(plan.pinnedLayerCount) of \(plan.profile.layerCount) layers"
                        : "nothing fits yet"),
                TierItem(
                    name: "Headroom", color: MRColor.tierFree,
                    bytes: plan.freeBytes,
                    note: "hard ceiling remains enforced"),
            ]
        }
        return [
            TierItem(
                name: "Working floor", color: MRColor.tierFloor, bytes: plan.floorBytes,
                note: "immovable"),
            TierItem(
                name: "Staged", color: MRColor.tierStaged, bytes: plan.stagedBytes,
                note: plan.readAheadDepth >= 1 ? "read-ahead on" : "read-ahead off"),
            TierItem(
                name: "Pinned", color: MRColor.tierPinned, bytes: plan.pinnedBytes,
                note: plan.pinnedLayerCount > 0
                    ? "\(plan.pinnedLayerCount) of \(plan.profile.layerCount) layers"
                    : "nothing fits yet"),
            TierItem(
                name: "Hot set", color: MRColor.tierHot, bytes: plan.hotSetBytes,
                note: plan.hotSetBytes > 0 ? nil : "ranks last"),
            TierItem(name: "Free", color: MRColor.tierFree, bytes: plan.freeBytes, note: nil),
        ]
    }
}

/// The tier legend: a grid, on purpose.
///
/// The version this replaces was an `HStack` of four columns, and at inspector
/// width it had nowhere to put "working floor" except through the middle of the
/// word — the panel shipped reading `WORK-` / `ING FLOOR`. A two-by-two grid at
/// the widths this panel actually gets, with every label held at its natural
/// size, has no such choice to make. The columns align because `Grid` aligns
/// them, not because four independent stacks happened to agree.
struct TierLegend: View {
    let plan: BudgetPlan
    /// Two columns of two by default. Four in a row only where there is room,
    /// which the caller knows and this view does not.
    var columns: Int = 2

    var body: some View {
        let items = TierItem.all(plan)
        let rows = stride(from: 0, to: items.count, by: max(1, columns)).map { start in
            Array(items[start..<min(start + max(1, columns), items.count)])
        }
        Grid(alignment: .leading, horizontalSpacing: MRSpace.s4, verticalSpacing: MRSpace.s3) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                GridRow {
                    ForEach(row) { item in entry(item) }
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("tier legend")
    }

    private func entry(_ item: TierItem) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: MRSpace.s1) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(item.color)
                    .frame(width: 9, height: 9)
                Text(item.name)
                    .mrLabel()
                    // The two rules that stop a label from breaking mid-word.
                    // Both are needed: one line, and never narrower than the
                    // word.
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
            }
            Text(MRFormat.bytesDecimal(item.bytes))
                .font(MRType.micro)
                .foregroundStyle(MRColor.primary)
            Text(item.note ?? " ")
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(item.name) \(MRFormat.bytesDecimal(item.bytes))")
    }
}

/// The tiers at the scale of the budget that was chosen.
///
/// The slider and this bar answer two different questions and used to be one
/// control, which is why the panel was unreadable: a 6.5 GB budget drawn across
/// a 32 GiB device axis puts every tier inside the first fifth of the bar, and
/// the staged pair — the thing the whole dial exists to show funding — is a few
/// pixels wide. The slider keeps the device domain, because choosing a budget is
/// a choice against the ceiling. This bar is normalised to the chosen budget,
/// because *how the budget is spent* is a question about the budget and about
/// nothing else.
///
/// Every segment carries its own percentage, so the bar can be read without
/// measuring it against the legend.
struct BudgetCompositionBar: View {
    let plan: BudgetPlan
    var height: CGFloat = 30

    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var total: Double { Double(max(1, plan.budgetBytes)) }
    private var motionPolicy: MRAccessibleMotionPolicy {
        MRAccessibleMotionPolicy(reduceMotion: reduceMotion)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .firstTextBaseline) {
                Text("How the budget is spent").mrLabel(MRColor.secondary)
                Spacer()
                Text("100 % = \(MRFormat.bytesDecimal(plan.budgetBytes))")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
            }
            bar
            if plan.isRunnable {
                Text(sentence)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var bar: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            ZStack(alignment: .leading) {
                Rectangle().fill(MRColor.tierFree)
                if plan.isRunnable {
                    HStack(spacing: 0) {
                        ForEach(TierItem.all(plan)) { item in
                            segment(item, width: CGFloat(Double(item.bytes) / total) * width)
                        }
                    }
                } else {
                    // Refused: nothing was allocated, so there is nothing to
                    // apportion. The bar says that rather than drawing a
                    // plausible-looking split of a budget nobody accepted.
                    Hatch(color: MRColor.refuse)
                    Text("refused — nothing is allocated")
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.refuse)
                        .padding(.horizontal, MRSpace.s2)
                }
            }
        }
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: MRRadius.control, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: MRRadius.control, style: .continuous)
                .strokeBorder(
                    contrast == .increased ? MRColor.tertiary : MRColor.hairline, lineWidth: 1)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("budget composition")
        .accessibilityValue(spoken)
        .animation(motionPolicy.tierAnimation, value: plan.budgetBytes)
        .transaction { transaction in
            if !motionPolicy.animatesGeometry { transaction.animation = nil }
        }
    }

    /// A percentage is printed inside a segment only when it fits. A number
    /// clipped to "1" is worse than no number.
    private func segment(_ item: TierItem, width: CGFloat) -> some View {
        ZStack {
            Rectangle().fill(item.color)
            if width >= 44 {
                Text(percentage(item.bytes))
                    .font(MRType.micro)
                    .foregroundStyle(item.name == "Free" ? MRColor.secondary : MRColor.abyss)
            }
        }
        .frame(width: max(0, width))
    }

    private func percentage(_ bytes: UInt64) -> String {
        let share = Double(bytes) / total * 100
        return share >= 10 ? "\(Int(share.rounded())) %" : String(format: "%.1f %%", share)
    }

    /// Three clauses, in the order the bytes are spent: what the staged pair is
    /// doing, what the pins bought, and what the next increase would buy.
    ///
    /// The last clause is the answer to the question the bar exists to answer —
    /// "what does dragging this up get me?" — and it comes from the plan's own
    /// `nextPinBudgetBytes`, which is the smallest budget that pins strictly
    /// *more*, not this budget plus the next unit's size. The two readings
    /// disagree and only the first is true.
    private var sentence: String {
        if plan.strategy == .boundedLayerStreaming {
            return "V4 streams model weights and replaces one layer's causal state at a time. "
                + "The remaining \(MRFormat.bytesDecimal(plan.freeBytes)) is budget headroom, "
                + "not a promise to retain model layers."
        }
        let staged =
            plan.stagedTierIsFunded
            ? "The staged pair is funded, so the model-layer read hides behind compute"
            : (plan.readAheadDepth >= 1
                ? "The staged pair is not fully funded at this budget, so the read stays serial"
                : "Read-ahead is off, so the model-layer read is serial with compute")

        let pinned: String
        if plan.pinnedLayerCount > 0 {
            pinned = "\(plan.pinnedLayerCount) of \(plan.profile.layerCount) layers are held in "
                + "RAM, so a token reads \(MRFormat.bytesDecimal(plan.bytesSavedPerToken)) fewer "
                + "bytes"
        } else {
            pinned = "Nothing is pinned yet: every layer streams, every token"
        }

        let next = plan.nextPinSentence ?? "Every unit this ladder has is already resident."
        return "\(staged). \(pinned). \(next)"
    }

    private var spoken: String {
        guard plan.isRunnable else { return "refused; nothing is allocated" }
        return TierItem.all(plan)
            .map { "\($0.name) \(percentage($0.bytes))" }
            .joined(separator: ", ")
    }
}

/// The memory dial: the tier bar, a thumb that may be dragged into the refusal
/// zone, and nothing that snaps back.
struct MemoryDial: View {
    let plan: BudgetPlan
    /// Nil makes the whole control read-only, which is what a run screen shows:
    /// the budget stated at start, not a live-editable value.
    let onChange: ((UInt64) -> Void)?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var dragOrigin: UInt64?

    private var isEditable: Bool { onChange != nil }
    private var step: UInt64 { max(1, plan.deviceCeilingBytes / 200) }
    private var motionPolicy: MRAccessibleMotionPolicy {
        MRAccessibleMotionPolicy(reduceMotion: reduceMotion)
    }
    private var interactionHeight: CGFloat {
        #if os(iOS)
            MRAccessibility.interactiveDimension(34, onIOS: true)
        #else
            MRAccessibility.interactiveDimension(34, onIOS: false)
        #endif
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            header
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    TierBar(plan: plan, height: 30)
                    thumb(in: geometry.size.width)
                }
                .contentShape(Rectangle())
                .gesture(isEditable ? drag(width: geometry.size.width) : nil)
            }
            .frame(height: interactionHeight)
            .animation(motionPolicy.tierAnimation, value: plan.budgetBytes)
            .transaction { transaction in
                if !motionPolicy.animatesGeometry { transaction.animation = nil }
            }

            // The axis this control is drawn against, named. The slider's job
            // is choosing a budget out of what the device offers, so its domain
            // is the device's; the composition of that budget is a second bar
            // at a scale where it can be read.
            HStack(alignment: .top, spacing: MRSpace.s3) {
                Text("0")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                Spacer(minLength: MRSpace.s2)
                VStack(alignment: .trailing, spacing: 1) {
                    Text("Device limit").mrLabel()
                    Text(MRFormat.bytesDecimal(plan.deviceCeilingBytes))
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.tertiary)
                }
            }
            Text(
                "Minimum to run "
                    + MRFormat.bytesDecimal(plan.profile.refusalThresholdBytes)
            )
            .font(MRType.micro)
            .foregroundStyle(
                plan.isRunnable ? MRColor.tertiary : MRColor.refuse.opacity(0.8))
            .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .contain)
        .accessibilityRepresentation {
            if isEditable {
                Slider(
                    value: Binding(
                        get: { Double(plan.budgetBytes) },
                        set: { onChange?(UInt64(max(0, $0))) }),
                    in: 0...Double(max(1, plan.deviceCeilingBytes)),
                    step: Double(step)
                ) {
                    Text("memory budget")
                } minimumValueLabel: {
                    Text("0")
                } maximumValueLabel: {
                    Text(MRFormat.bytesDecimal(plan.deviceCeilingBytes))
                }
                .accessibilityValue(spokenValue)
            } else {
                Text("memory budget \(MRFormat.bytesDecimal(plan.budgetBytes)), stated at start")
            }
        }
        #if os(macOS)
            .focusable(isEditable)
            // Focus stays — arrow keys still nudge the budget — but the system
            // focus ring does not. AppKit draws that ring around the whole
            // focusable region, and this region is the entire MEMORY BUDGET
            // block: touching the slider outlined the header, the bar, both
            // axis labels and the minimum sentence in blue. The dial already
            // says where it is with its own thumb.
            .focusEffectDisabled()
            .onKeyPress(.leftArrow) { nudge(-1) }
            .onKeyPress(.rightArrow) { nudge(1) }
        #endif
    }

    private var spokenValue: String {
        if let refusal = plan.refusal {
            return "\(MRFormat.bytesDecimal(plan.budgetBytes)), refused, short by "
                + "\(MRFormat.bytesDecimal(refusal.deficitBytes))"
        }
        if plan.strategy == .boundedLayerStreaming {
            return "\(MRFormat.bytesDecimal(plan.budgetBytes)), bounded V4 execution floor "
                + "\(MRFormat.bytesDecimal(plan.floorBytes)), headroom "
                + MRFormat.bytesDecimal(plan.freeBytes)
        }
        return "\(MRFormat.bytesDecimal(plan.budgetBytes)), "
            + "floor \(MRFormat.bytesDecimal(plan.floorBytes)), "
            + "staged \(MRFormat.bytesDecimal(plan.stagedBytes)), "
            + "\(plan.pinnedLayerCount) layers pinned at "
            + "\(MRFormat.bytesDecimal(plan.pinnedBytes)), "
            + "hot set \(MRFormat.bytesDecimal(plan.hotSetBytes)), "
            + "free \(MRFormat.bytesDecimal(plan.freeBytes))"
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Memory budget").mrLabel(MRColor.secondary)
            Spacer()
            Text(MRFormat.bytesDecimal(plan.budgetBytes))
                .font(MRType.metric)
                .foregroundStyle(plan.isRunnable ? MRColor.primary : MRColor.refuse)
                .animation(nil, value: plan.budgetBytes)
            if !isEditable {
                Text("— stated at start")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
            }
        }
    }

    private func thumb(in width: CGFloat) -> some View {
        let fraction = min(1, Double(plan.budgetBytes) / Double(max(1, plan.deviceCeilingBytes)))
        return RoundedRectangle(cornerRadius: 3, style: .continuous)
            .fill(plan.isRunnable ? MRColor.primary : MRColor.refuse)
            .frame(width: 6, height: 40)
            .offset(x: max(0, min(width - 6, CGFloat(fraction) * width - 3)))
            .shadow(radius: 0)
            .allowsHitTesting(false)
    }

    private func drag(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard width > 0 else { return }
                let fraction = min(1, max(0, value.location.x / width))
                // No snapping, no clamping to the floor: the thumb may be
                // dragged into the refusal zone and stays there.
                onChange?(UInt64(fraction * Double(plan.deviceCeilingBytes)))
            }
    }

    private func nudge(_ direction: Int) -> KeyPress.Result {
        guard isEditable else { return .ignored }
        let delta = Int64(step) * Int64(direction)
        let next = Int64(bitPattern: plan.budgetBytes) + delta
        onChange?(UInt64(max(0, min(Int64(bitPattern: plan.deviceCeilingBytes), next))))
        return .handled
    }
}
