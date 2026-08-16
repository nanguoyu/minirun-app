import SwiftUI

#if canImport(AppKit)
    import AppKit
#endif
#if canImport(UIKit)
    import UIKit
#endif

// MARK: - Hatching

/// 45° hatching, used for exactly two things: unrequested memory to the right
/// of the dial's thumb, and the refusal zone to its left.
///
/// Under Increase Contrast it stops being a pattern and becomes a flat wash, so
/// the boundary between "asked for" and "not asked for" survives a display that
/// eats thin lines.
struct Hatch: View {
    var color: Color
    var spacing: CGFloat = 5
    var lineWidth: CGFloat = 1.5

    @Environment(\.colorSchemeContrast) private var contrast

    var body: some View {
        GeometryReader { geometry in
            if contrast == .increased {
                color.opacity(0.30)
            } else {
                Path { path in
                    let width = geometry.size.width
                    let height = geometry.size.height
                    var x = -height
                    while x < width {
                        path.move(to: CGPoint(x: x, y: height))
                        path.addLine(to: CGPoint(x: x + height, y: 0))
                        x += spacing
                    }
                }
                .stroke(color.opacity(0.55), lineWidth: lineWidth)
            }
        }
        .clipped()
        .allowsHitTesting(false)
    }
}

// MARK: - Numbers with provenance

/// A quantity, rendered in the type that says whether it has been checked.
///
/// This is invariant 3 of the product made into a view: mono means the number
/// came from files, italic means it came from metadata and nobody has looked at
/// the bytes yet. No screen may print a size without going through here.
struct ValueText: View {
    enum Provenance {
        /// Checked against files. Mono, tabular.
        case measured
        /// From an index or a manifest. Italic, never mono.
        case declared
        /// A projection. Mono size, secondary colour, `≈` prefix.
        case projected
    }

    let text: String
    var provenance: Provenance = .measured
    var font: Font?
    var color: Color?

    var body: some View {
        Group {
            switch provenance {
            case .measured:
                Text(text)
                    .font(font ?? MRType.metric)
                    .foregroundStyle(color ?? MRColor.primary)
            case .declared:
                Text(text)
                    .font(font ?? MRType.declared)
                    .foregroundStyle(color ?? MRColor.secondary)
            case .projected:
                Text("≈ \(text)")
                    .font(font ?? MRType.metric)
                    .foregroundStyle(color ?? MRColor.secondary)
            }
        }
        .accessibilityLabel(accessibilityText)
    }

    private var accessibilityText: String {
        switch provenance {
        case .measured: return text
        case .declared: return "\(text), declared, not yet checked against files"
        case .projected: return "approximately \(text), a projection"
        }
    }
}

// MARK: - Readouts

/// A big number with a label, a unit and a note about where it came from.
struct MetricReadout: View {
    let label: String
    let value: String
    var unit: String?
    var provenance: String?
    var valueColor: Color = MRColor.primary
    var spokenValue: String?

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            Text(label).mrLabel()
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Text(value)
                    .font(MRType.readout)
                    .foregroundStyle(valueColor)
                    // Telemetry numerals never animate their value: an animated
                    // number misrepresents when it was measured.
                    .animation(nil, value: value)
                if let unit {
                    Text(unit)
                        .font(MRType.caption)
                        .foregroundStyle(MRColor.tertiary)
                }
            }
            if let provenance {
                Text(provenance)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(label)
        .accessibilityValue(spokenValue ?? "\(value) \(unit ?? "")")
    }
}

/// One line of a detail table: name on the left, value on the right.
struct MetricRow: View {
    let label: String
    let value: String
    var provenance: ValueText.Provenance = .measured
    var valueColor: Color?
    var detail: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s3) {
            // A label wraps rather than truncating: at the inspector's narrow width
            // this row was rendering "increased memory li…" beside "1.56 T…",
            // which is two facts turned into two ellipses. A value that does
            // not fit loses its middle instead, so its ends — the magnitude and
            // the unit, or the volume and the leaf directory — survive.
            //
            // `lineLimit` and never `fixedSize`. A `Text` free to grow
            // vertically inside an `HStack` reports its height at the narrowest
            // width the row could ever offer — one word per line — and that
            // number becomes the minimum height of the column, of the split
            // view and of the window. The window cannot honour it, SwiftUI
            // centres the oversized content, and every column loses its top and
            // bottom. That is the bug that once hid the entire sidebar, and
            // this row reintroduced it for one build.
            Text(label)
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
                .lineLimit(2)
            Spacer(minLength: MRSpace.s2)
            VStack(alignment: .trailing, spacing: 1) {
                ValueText(text: value, provenance: provenance, color: valueColor)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let detail {
                    Text(detail)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.tertiary)
                        .multilineTextAlignment(.trailing)
                        .lineLimit(2)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Numeric entry

/// A whole number, typed.
///
/// Digits only while typing, and the value is committed clamped into `range`
/// when the field loses focus or Return is pressed — a half-typed "1" on the
/// way to "16" must not become the stated value, and an out-of-range entry must
/// not become one either. Focusing the field selects what is in it, so typing
/// replaces the number instead of appending to it.
///
/// The clamp here is a text-entry rule, not the product's budget rule: this is
/// a control whose own range is a runner capability, so a number outside it was
/// never expressible rather than refused after the fact.
struct IntegerField: View {
    @Binding var value: Int
    let range: ClosedRange<Int>
    var accessibilityLabel: String
    var width: CGFloat = 72

    @State private var text: String = ""
    @FocusState private var focused: Bool

    var body: some View {
        TextField("", text: $text)
            .textFieldStyle(.plain)
            .font(MRType.metric)
            .multilineTextAlignment(.trailing)
            .padding(.horizontal, MRSpace.s2)
            .padding(.vertical, MRSpace.s1)
            .frame(width: width)
            .frame(minHeight: MRAccessibility.interactiveDimension(24, onIOS: isIOS))
            .mrCard(MRColor.raised, radius: MRRadius.control)
            .focused($focused)
            #if os(iOS)
                .keyboardType(.numberPad)
            #endif
            .onSubmit { commit() }
            .onAppear { text = String(value) }
            .onChange(of: value) { _, newValue in
                if !focused { text = String(newValue) }
            }
            .onChange(of: text) { _, newText in
                let digits = newText.filter(\.isNumber)
                if digits != newText { text = digits }
            }
            .onChange(of: focused) { _, isFocused in
                if isFocused {
                    selectAll()
                } else {
                    commit()
                }
            }
            .accessibilityLabel(accessibilityLabel)
            .accessibilityValue("\(value)")
    }

    private var isIOS: Bool {
        #if os(iOS)
            return true
        #else
            return false
        #endif
    }

    /// An empty or unparsable field restores the last committed value rather
    /// than inventing a zero.
    static func committed(_ text: String, in range: ClosedRange<Int>, current: Int) -> Int {
        guard let typed = Int(text) else { return current }
        return min(range.upperBound, max(range.lowerBound, typed))
    }

    private func commit() {
        let committed = Self.committed(text, in: range, current: value)
        value = committed
        text = String(committed)
    }

    private func selectAll() {
        #if os(macOS)
            DispatchQueue.main.async {
                NSApp?.sendAction(#selector(NSText.selectAll(_:)), to: nil, from: nil)
            }
        #elseif canImport(UIKit)
            DispatchQueue.main.async {
                UIApplication.shared.sendAction(
                    #selector(UIResponder.selectAll(_:)), to: nil, from: nil, for: nil)
            }
        #endif
    }
}

// MARK: - Chips

struct StatusChip: View {
    enum Tone: Equatable { case neutral, ok, caution, refuse, verify

        var color: Color {
            switch self {
            case .neutral: return MRColor.tertiary
            case .ok: return MRColor.ok
            case .caution: return MRColor.caution
            case .refuse: return MRColor.refuse
            case .verify: return MRColor.verify
            }
        }
    }

    let text: String
    var tone: Tone = .neutral
    var systemImage: String?

    var body: some View {
        HStack(spacing: MRSpace.s1) {
            if let systemImage {
                Image(systemName: systemImage).imageScale(.small)
            }
            Text(text)
                .font(MRType.micro)
        }
        .padding(.horizontal, MRSpace.s2)
        .padding(.vertical, MRSpace.s1)
        .foregroundStyle(tone.color)
        .background(
            Capsule().fill(tone.color.opacity(0.12))
        )
        .overlay(Capsule().strokeBorder(tone.color.opacity(0.35), lineWidth: 1))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(text)
    }
}

/// Product-wide accessibility semantics that can be checked without rendering
/// a platform view hierarchy.
enum MRAccessibility {
    /// Apple's minimum comfortable touch target. Visual glyphs stay compact;
    /// only the invisible interactive frame grows on iOS.
    static let minimumIOSTouchTarget: CGFloat = 44

    static func interactiveDimension(_ proposed: CGFloat, onIOS: Bool) -> CGFloat {
        onIOS ? max(minimumIOSTouchTarget, proposed) : proposed
    }

    static func helpLabel(subject: String) -> String {
        let subject = subject.trimmingCharacters(in: .whitespacesAndNewlines)
        return subject.isEmpty ? "More information" : "About \(subject)"
    }

    /// Legacy call sites still receive a contextual label instead of the old
    /// indistinguishable “more about this”. New call sites should name their
    /// short subject explicitly.
    static func inferredHelpLabel(from explanation: String) -> String {
        let trimmed = explanation.trimmingCharacters(in: .whitespacesAndNewlines)
        let clause = String(trimmed.prefix { !".?!".contains($0) })
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return clause.isEmpty ? "More information" : "More information: \(clause)"
    }
}

extension View {
    /// Preserve compact Mac controls while making the same control comfortably
    /// tappable on iPhone and iPad.
    @ViewBuilder
    func mrMinimumTouchTarget() -> some View {
        #if os(iOS)
            self
                .frame(
                    minWidth: MRAccessibility.minimumIOSTouchTarget,
                    minHeight: MRAccessibility.minimumIOSTouchTarget)
                .contentShape(Rectangle())
        #else
            self
        #endif
    }
}

/// The product's accounting claim, as an 8 pt dot.
///
/// Missing evidence is neutral, never green. Green means the identities the
/// accepted run was required to report are present and balanced; red names the
/// failed identity.
struct BalanceDot: View {
    let status: MetricEvidenceStatus

    init(status: MetricEvidenceStatus) {
        self.status = status
    }

    /// Other product surfaces already have a genuine boolean result. Keep
    /// their call sites concise without treating an absent instrument sample
    /// as `true`.
    init(balanced: Bool, failingIdentity: String? = nil) {
        status = balanced
            ? .valid
            : .failed(failingIdentity ?? "An accounting identity does not hold.")
    }

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 8, height: 8)
            .help(helpText)
            .accessibilityLabel("byte accounting")
            .accessibilityValue(accessibilityValue)
    }

    private var color: Color {
        MetricEvidencePresentation.tone(for: status).color
    }

    private var helpText: String {
        switch status {
        case .notReported: return "Byte counts pending — they settle at a pass boundary."
        case .valid: return "The reported byte counts balance."
        case .failed(let identity): return identity
        }
    }

    private var accessibilityValue: String {
        MetricEvidencePresentation.accessibilityValue(for: status)
    }
}

/// One mapping from evidence to product language and tone. Missing evidence is
/// neutral by construction; only `.valid` can produce the green success tone.
enum MetricEvidencePresentation {
    static func tone(for status: MetricEvidenceStatus) -> StatusChip.Tone {
        switch status {
        case .notReported: return .neutral
        case .valid: return .ok
        case .failed: return .refuse
        }
    }

    static func accessibilityValue(for status: MetricEvidenceStatus) -> String {
        switch status {
        case .notReported: return "not reported"
        case .valid: return "balanced"
        case .failed(let identity): return "not balanced: \(identity)"
        }
    }
}

/// The same tri-state accounting result as `BalanceDot`, but with enough text
/// to be understood without hover. The full instrument panel uses this form;
/// the dot remains available only for genuinely space-constrained surfaces.
struct AccountingStatusChip: View {
    let status: MetricEvidenceStatus

    var body: some View {
        StatusChip(text: title, tone: tone, systemImage: systemImage)
            .help(detail)
            .accessibilityValue(MetricEvidencePresentation.accessibilityValue(for: status))
            .accessibilityHint(detail)
    }

    /// `Accounting waiting` was the ledger talking to itself: it named an
    /// internal bookkeeping state rather than the thing on screen. What is
    /// waiting is the byte counts, and they are waiting because they settle at
    /// a pass boundary — which is what the ⓘ beside this chip now says in full.
    var title: String {
        switch status {
        case .notReported: return "Byte counts pending"
        case .valid: return "Byte counts balanced"
        case .failed: return "Byte counts mismatch"
        }
    }

    private var tone: StatusChip.Tone {
        MetricEvidencePresentation.tone(for: status)
    }

    private var systemImage: String {
        switch status {
        case .notReported: return "clock"
        case .valid: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    private var detail: String {
        switch status {
        case .notReported:
            return "Byte counts settle at a pass boundary. This run has not completed one yet."
        case .valid:
            return "Every byte reported by this run is counted exactly once."
        case .failed(let identity):
            return "The reported byte counts do not add up: \(identity)"
        }
    }
}

// MARK: - Digests

/// Mono, truncating in the middle, copyable. Reproducibility is a first-class
/// user-visible output here, not a debug affordance.
struct DigestLabel: View {
    let digest: String
    var label: String?
    var showsCopyButton = true

    @State private var copied = false

    var body: some View {
        HStack(spacing: MRSpace.s2) {
            if let label {
                Text(label).mrLabel()
            }
            Text(MRFormat.shortDigest(digest))
                .font(MRType.micro)
                .foregroundStyle(MRColor.secondary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
            if showsCopyButton {
                Button {
                    copy(digest)
                } label: {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        .imageScale(.small)
                }
                .buttonStyle(.plain)
                .mrMinimumTouchTarget()
                .foregroundStyle(MRColor.tertiary)
                .help("Copy the full digest")
                .accessibilityLabel("Copy the full digest")
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label ?? "digest")
        .accessibilityValue(spelled(digest))
    }

    private func copy(_ text: String) {
        #if canImport(AppKit)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        #elseif canImport(UIKit)
            UIPasteboard.general.string = text
        #endif
        copied = true
        Task {
            try? await Task.sleep(for: .seconds(2))
            copied = false
        }
    }

    /// VoiceOver reads a 64-character hex string as a word salad; the first
    /// eight characters, spelled, are what a person actually compares.
    private func spelled(_ hex: String) -> String {
        "starts \(hex.prefix(8).map(String.init).joined(separator: " ")), "
            + "\(hex.count) hexadecimal characters"
    }
}

// MARK: - Layout

/// A row of chips that wraps instead of cramming.
///
/// `HStack` has exactly two answers when its children do not fit: squeeze them
/// until the words hyphenate mid-syllable, or clip. Both shipped in this app —
/// "WORK-ING" in the dial's legend was the visible one. A wrapping layout has a
/// third answer, which is the correct one: put the chip that does not fit on the
/// next line at its natural size.
struct FlowRow: Layout {
    var spacing: CGFloat = MRSpace.s2
    var lineSpacing: CGFloat = MRSpace.s2

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        let rows = arrange(subviews, in: width)
        let height = rows.reduce(0) { $0 + $1.height } + lineSpacing * CGFloat(max(0, rows.count - 1))
        let widest = rows.map(\.width).max() ?? 0
        return CGSize(width: min(width, max(widest, 0)), height: height)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        var y = bounds.minY
        for row in arrange(subviews, in: bounds.width) {
            var x = bounds.minX
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(
                    at: CGPoint(x: x, y: y + (row.height - size.height) / 2),
                    proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + lineSpacing
        }
    }

    private struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    private func arrange(_ subviews: Subviews, in width: CGFloat) -> [Row] {
        var rows: [Row] = []
        var row = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let next = row.indices.isEmpty ? size.width : row.width + spacing + size.width
            if !row.indices.isEmpty && next > width {
                rows.append(row)
                row = Row()
                row.indices = [index]
                row.width = size.width
                row.height = size.height
            } else {
                row.indices.append(index)
                row.width = next
                row.height = max(row.height, size.height)
            }
        }
        if !row.indices.isEmpty { rows.append(row) }
        return rows
    }
}

// MARK: - Explanations that do not shout

/// An ⓘ that holds a paragraph.
///
/// This product has a lot to explain and every sentence of it used to be
/// printed on the screen, which turned the settings panel into a manual nobody
/// read. The rule now: a section gets at most one short caption, and everything
/// else that is worth saying goes behind one of these. The text is still there,
/// still verbatim, still one tap away — it is simply not in the way.
struct HelpTip: View {
    let text: String
    let accessibilityLabel: String
    @State private var showing = false

    init(_ text: String, accessibilityLabel: String? = nil) {
        self.text = text
        self.accessibilityLabel = accessibilityLabel
            ?? MRAccessibility.inferredHelpLabel(from: text)
    }

    var body: some View {
        Button {
            showing.toggle()
        } label: {
            Image(systemName: "info.circle")
                .imageScale(.small)
                .foregroundStyle(MRColor.tertiary)
        }
        .buttonStyle(.plain)
        .mrMinimumTouchTarget()
        .help(text)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint(text)
        .popover(isPresented: $showing, arrowEdge: .bottom) {
            Text(text)
                .font(MRType.caption)
                .foregroundStyle(MRColor.primary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(MRSpace.s4)
                .frame(width: 300)
        }
    }
}

// MARK: - Section furniture

struct SectionHeader: View {
    let title: String
    var trailing: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).mrLabel(MRColor.secondary)
            Spacer()
            if let trailing {
                Text(trailing)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
            }
        }
        .accessibilityAddTraits(.isHeader)
    }
}

/// A card that carries a whole instrument or a whole message. One surface
/// treatment in this product: a fill and a hairline.
struct Panel<Content: View>: View {
    var title: String?
    var trailing: AnyView?
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            if title != nil || trailing != nil {
                HStack(alignment: .firstTextBaseline) {
                    if let title {
                        Text(title)
                            .mrLabel(MRColor.secondary)
                            .accessibilityAddTraits(.isHeader)
                    }
                    Spacer()
                    if let trailing { trailing }
                }
            }
            content
        }
        .padding(MRSpace.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard()
    }
}
