import SwiftUI

// =============================================================================
// MARK: - The product-page language
// =============================================================================
//
// Hierarchy, not cards.
//
// The screens in this app were built as instrument panels: every block is a
// bordered card with an ALL-CAPS label, every one of them carries the same
// weight, and the reader's eye has nowhere to land. That is the right shape for
// a running model, where six independent instruments are all equally live. It
// is the wrong shape for the page that introduces a model, which is read the
// way a product page is read — what this is, what state it is in, one action;
// then the one thing that is happening; then the facts, quietly.
//
// So this file has no card in it. One surface, hairline separators, and border,
// fill and shadow spent only where something genuinely needs lifting off the
// page. Colour carries state and nothing else: a dot is blue while something
// moves, green when it is done, grey when there is nothing here, amber when the
// operator has to do something. Sentence case throughout; monospace only for
// paths, hashes and true code.
//
// These components are new rather than a rewrite of `Panel`/`StatusChip`,
// because the panel screens still want the old language and the two would fight
// over every default. Model detail is the first page in this one; the phase-2
// list is in DESIGN.md.

// MARK: - State as colour

/// What a status dot on a product page is allowed to mean.
///
/// Four states, because a person reading a page glances at a dot and reads a
/// sentence; a taxonomy of nine is a legend, and a legend is what this replaced.
enum MRPageStatusTone: Equatable, Sendable, CaseIterable {
    /// Something is happening right now. Accent.
    case moving
    /// Done, and usable. Green.
    case ready
    /// Nothing is here, and nothing is wrong. Grey.
    case idle
    /// The operator has to do something before this moves. Amber.
    case attention

    var color: Color {
        switch self {
        case .moving: return MRColor.accent
        case .ready: return MRColor.ok
        case .idle: return MRColor.tertiary
        case .attention: return MRColor.caution
        }
    }

    /// The halo behind the dot. Grey states get none: a dot that means
    /// "nothing here" must not glow.
    var halo: Color? {
        switch self {
        case .moving: return MRColor.accentSoft
        case .ready: return MRColor.okSoft
        case .attention: return MRColor.cautionSoft
        case .idle: return nil
        }
    }

    /// How VoiceOver says the colour, since it cannot see it.
    var spokenState: String {
        switch self {
        case .moving: return "in progress"
        case .ready: return "ready"
        case .idle: return "not available"
        case .attention: return "needs attention"
        }
    }

    /// The panel language's chip tones, mapped onto this page's four.
    ///
    /// `verify` and `refuse` both land on `attention`: a spot-check that has to
    /// be finished and a device that cannot run the model are, to the eye
    /// scanning a page, the same message — *this one is not going to work until
    /// something changes*. The sentence beside the dot says which.
    static func from(_ tone: StatusChip.Tone) -> MRPageStatusTone {
        switch tone {
        case .neutral: return .idle
        case .ok: return .ready
        case .caution, .refuse, .verify: return .attention
        }
    }
}

/// Eight points of colour, with a three-point halo when the state is live.
struct MRStatusDot: View {
    let tone: MRPageStatusTone
    var diameter: CGFloat = 8

    var body: some View {
        Circle()
            .fill(tone.color)
            .frame(width: diameter, height: diameter)
            .background {
                if let halo = tone.halo {
                    Circle()
                        .fill(halo)
                        .frame(width: diameter + 6, height: diameter + 6)
                }
            }
            .accessibilityHidden(true)
    }
}

/// A dot and one sentence. The status of the whole page, in a line.
struct MRStatusLine: View {
    let tone: MRPageStatusTone
    let sentence: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
            MRStatusDot(tone: tone)
                .alignmentGuide(.firstTextBaseline) { _ in 7 }
            Text(sentence)
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("status, \(tone.spokenState)")
        .accessibilityValue(sentence)
    }
}

// MARK: - Controls

/// The page's primary action: ink fill, surface text, no gradient.
struct MRFilledButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(MRType.control)
            .foregroundStyle(MRColor.panel)
            .padding(.horizontal, MRSpace.s3)
            .frame(minHeight: MRPageControl.height)
            .background(
                RoundedRectangle(cornerRadius: MRRadius.action, style: .continuous)
                    .fill(MRColor.primary)
            )
            .opacity(isEnabled ? (configuration.isPressed ? 0.82 : 1) : 0.4)
            .contentShape(Rectangle())
    }
}

/// Everything beside the primary action: a hairline, and the page's ink.
struct MROutlineButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(MRType.control)
            .foregroundStyle(MRColor.primary)
            .padding(.horizontal, MRSpace.s3)
            .frame(minHeight: MRPageControl.height)
            .background(
                RoundedRectangle(cornerRadius: MRRadius.action, style: .continuous)
                    .fill(MRColor.raised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: MRRadius.action, style: .continuous)
                    .strokeBorder(MRColor.hairline, lineWidth: 1)
            )
            .opacity(isEnabled ? (configuration.isPressed ? 0.7 : 1) : 0.4)
            .contentShape(Rectangle())
    }
}

/// A text link. No border, no background — the page's quiet third tier.
struct MRTextLinkStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var role: Role = .accent

    enum Role { case accent, quiet, destructive }

    private var color: Color {
        switch role {
        case .accent: return MRColor.accent
        case .quiet: return MRColor.secondary
        case .destructive: return MRColor.refuse
        }
    }

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(MRType.control)
            .foregroundStyle(color)
            .frame(minHeight: MRPageControl.linkHeight)
            .opacity(isEnabled ? (configuration.isPressed ? 0.6 : 1) : 0.4)
            .contentShape(Rectangle())
    }
}

enum MRPageControl {
    /// A button on this page. The phone's minimum comfortable target on iOS;
    /// a Mac-sized control on macOS.
    static var height: CGFloat {
        #if os(iOS)
            return MRAccessibility.minimumIOSTouchTarget
        #else
            return 30
        #endif
    }

    /// A text link's tappable band. Still 44 on the phone even though the text
    /// is 13 points tall.
    static var linkHeight: CGFloat {
        #if os(iOS)
            return MRAccessibility.minimumIOSTouchTarget
        #else
            return 22
        #endif
    }
}

/// Where a control is drawn, which decides who owns its shape.
///
/// This is the product's one navigation-bar layout rule, and it is here rather
/// than on any screen because every screen was getting it wrong in the same
/// way. iOS 26 draws each navigation-bar item inside its own glass capsule and
/// measures that capsule from the item's content. A page control brings a
/// filled shape, a hairline border, horizontal padding and a 44-point minimum
/// of its own, so a bar item wearing one asks for a second control inside the
/// system's: the capsule swells past the bar's trailing inset and is cut off by
/// the screen edge — which is what Storage's **Rescan** did, with half the
/// capsule off the right of an iPhone. A hand-set `frame(minWidth:minHeight:)`
/// on a bar item's glyph does the same thing for the same reason.
///
/// A bar item is a label and nothing else. The bar already supplies the target,
/// the padding, the tint and the material. macOS has no such bar in this
/// product — it draws its own 52-point column headers — so there the same
/// control keeps the page's shape.
enum MRControlPlacement {
    /// Inside the app's own layout, on a product page or column header.
    case page
    /// Inside the platform's navigation bar.
    case navigationBar
}

extension View {
    func mrFilledAction() -> some View { buttonStyle(MRFilledButtonStyle()) }
    func mrOutlineAction() -> some View { buttonStyle(MROutlineButtonStyle()) }

    /// The same control, drawn by whoever owns the surface it sits on.
    @ViewBuilder
    func mrOutlineAction(_ placement: MRControlPlacement) -> some View {
        switch placement {
        case .page:
            mrOutlineAction()
        case .navigationBar:
            #if os(iOS)
                self
            #else
                mrOutlineAction()
            #endif
        }
    }

    func mrTextLink(_ role: MRTextLinkStyle.Role = .accent) -> some View {
        buttonStyle(MRTextLinkStyle(role: role))
    }
}

// MARK: - Page furniture

/// The one separator this language has.
struct MRHairline: View {
    @Environment(\.mrContrast) private var contrast

    var body: some View {
        Rectangle()
            .fill(contrast.hairline)
            .frame(height: 1)
            .accessibilityHidden(true)
    }
}

/// A section of a product page: a hairline, a 13 pt semibold heading in
/// sentence case, and its content. No border, no fill, no ALL-CAPS.
struct MRPageSection<Content: View>: View {
    var title: String?
    var showsSeparator = true
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            if showsSeparator { MRHairline().padding(.bottom, MRSpace.s1) }
            if let title {
                Text(title)
                    .font(MRType.sectionHeading)
                    .foregroundStyle(MRColor.primary)
                    .accessibilityAddTraits(.isHeader)
            }
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, showsSeparator ? MRSpace.s5 : 0)
    }
}

/// The publisher's own mark, in a 64-point squircle.
///
/// A tile and not a bare image: the logos are drawn for light and dark grounds
/// alike and several of them are square, so the page gives them one shape to
/// sit in rather than four different silhouettes at the top of four pages.
struct MRProductTile<Content: View>: View {
    var size: CGFloat = 64
    @ViewBuilder var content: Content

    var body: some View {
        content
            .frame(width: size, height: size)
            .background(
                RoundedRectangle(cornerRadius: MRRadius.tile, style: .continuous)
                    .fill(MRColor.raised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: MRRadius.tile, style: .continuous)
                    .strokeBorder(MRColor.hairline, lineWidth: 1)
            )
            .accessibilityHidden(true)
    }
}

/// The top of a product page: mark, name, one line of what it is, one status
/// sentence with a dot, and the actions on the trailing edge.
///
/// On a phone the actions drop under the identity rather than squeezing the
/// name into a vertical alphabet.
struct MRProductHeader<Logo: View, Actions: View>: View {
    let title: String
    var subtitle: String?
    var status: (tone: MRPageStatusTone, sentence: String)?
    @ViewBuilder var logo: Logo
    @ViewBuilder var actions: Actions

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: MRSpace.s4) {
                logo
                identity
                Spacer(minLength: MRSpace.s4)
                actionRow
                    .fixedSize(horizontal: true, vertical: false)
            }
            VStack(alignment: .leading, spacing: MRSpace.s4) {
                HStack(alignment: .top, spacing: MRSpace.s3) {
                    logo
                    identity
                    Spacer(minLength: 0)
                }
                actionRow
            }
        }
    }

    private var identity: some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            Text(title)
                .font(MRType.pageTitle)
                .tracking(-0.5)
                .foregroundStyle(MRColor.primary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)
            if let subtitle {
                Text(subtitle)
                    .font(MRType.pageSubtitle)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let status {
                MRStatusLine(tone: status.tone, sentence: status.sentence)
                    .padding(.top, MRSpace.s1)
            }
        }
    }

    private var actionRow: some View {
        HStack(spacing: MRSpace.s2) { actions }
    }
}

// MARK: - The focal block

/// The one block on the page with weight: what a transfer has moved, out of
/// what, how fast, where to, and the two links that lead away from it.
///
/// Every part is optional because the same block draws a running transfer and a
/// cancelled one. A stopped transfer has no rate and no estimate — it has a
/// sentence about what is on the drive — and this component cannot print one
/// for it, because there is nowhere to put it.
struct MRTransferBlock<Links: View>: View {
    /// "47.6 GB" — what is on the drive now.
    var headline: String?
    /// "of 517 GB" — the quieter half.
    var headlineDetail: String?
    /// "35 MB/s · file 491 of 624 · about 3.7 hours left"
    var meta: String?
    /// Nil draws no bar at all: an empty track is a claim too.
    var fraction: Double?
    /// A sentence in place of, or under, the number.
    var sentence: String?
    /// The destination, in mono with a folder glyph.
    var path: String?
    var accessibilitySummary: String?
    @ViewBuilder var links: Links

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            if headline != nil || meta != nil {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .firstTextBaseline, spacing: MRSpace.s4) {
                        bigNumber
                        Spacer(minLength: MRSpace.s3)
                        metaText
                    }
                    VStack(alignment: .leading, spacing: MRSpace.s1) {
                        bigNumber
                        metaText
                    }
                }
            }
            if let fraction {
                MRProgressBar(fraction: fraction)
            }
            if let sentence {
                Text(sentence)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .firstTextBaseline, spacing: MRSpace.s4) {
                    pathLabel
                    Spacer(minLength: MRSpace.s3)
                    linkRow
                }
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    pathLabel
                    linkRow
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("transfer")
        .accessibilityValue(accessibilitySummary ?? spoken)
    }

    private var spoken: String {
        [headline, headlineDetail, meta, sentence]
            .compactMap { $0 }
            .joined(separator: ", ")
    }

    @ViewBuilder private var bigNumber: some View {
        if let headline {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Text(headline)
                    .font(MRType.bigFigure)
                    .foregroundStyle(MRColor.primary)
                    // A number never animates its own value: an interpolated
                    // figure misrepresents when it was measured.
                    .animation(nil, value: headline)
                if let headlineDetail {
                    Text(headlineDetail)
                        .font(MRType.bigFigureDetail)
                        .foregroundStyle(MRColor.secondary)
                }
            }
        }
    }

    @ViewBuilder private var metaText: some View {
        if let meta {
            Text(meta)
                .font(MRType.figure)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var pathLabel: some View {
        if let path {
            MRPathLabel(path: path)
        }
    }

    private var linkRow: some View {
        HStack(spacing: MRSpace.s4) { links }
    }
}

/// Five points of track, and the accent over it. Not `ProgressView`: the system
/// bar is 4 points of the system's own tint and cannot be either.
struct MRProgressBar: View {
    let fraction: Double
    var height: CGFloat = 5

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(MRColor.hairline)
                Capsule()
                    .fill(MRColor.accent)
                    .frame(width: max(0, min(1, fraction)) * geometry.size.width)
            }
        }
        .frame(height: height)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("progress")
        .accessibilityValue(MRFormat.percent(max(0, min(1, fraction)), digits: 0))
    }
}

/// A filesystem path: a folder glyph and SF Mono, selectable, losing its middle
/// rather than its ends when it does not fit.
struct MRPathLabel: View {
    let path: String
    var systemImage = "folder"

    var body: some View {
        HStack(spacing: MRSpace.s2) {
            Image(systemName: systemImage)
                .imageScale(.small)
                .foregroundStyle(MRColor.tertiary)
                .accessibilityHidden(true)
            Text(path)
                .font(MRType.path)
                .foregroundStyle(MRColor.secondary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("location")
        .accessibilityValue(path)
    }
}

// MARK: - Facts

/// Two columns when the page is wide enough to read them as two, one when it
/// is not.
///
/// `ViewThatFits` cannot make this decision: it asks each candidate for its
/// ideal width, and a column of wrapping sentences answers with the width of
/// its longest sentence on one line — so the two-column candidate never "fits"
/// and a 760-point Mac window got the phone's layout. This asks the container
/// how wide it is, which is the actual question.
struct MRColumns: Layout {
    var spacing: CGFloat = MRSpace.s7
    /// Below this the columns stack. Two 300-point columns and a gutter is
    /// the narrowest arrangement that still reads as a definition list beside
    /// a summary rather than as two cramped ones.
    var minimumWideWidth: CGFloat = 660
    var trailingWidth: CGFloat = 280
    /// The gap between the two when they are stacked. A section wants air; a
    /// list row wants the two halves to stay one row's worth of thing.
    var stackedSpacing: CGFloat = MRSpace.s5

    func sizeThatFits(
        proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) -> CGSize {
        guard subviews.count == 2 else {
            return CGSize(width: proposal.width ?? 0, height: 0)
        }
        let width = proposal.width ?? minimumWideWidth
        if isWide(width) {
            let leading = subviews[0].sizeThatFits(
                ProposedViewSize(width: leadingWidth(in: width), height: nil))
            let trailing = subviews[1].sizeThatFits(
                ProposedViewSize(width: trailingWidth, height: nil))
            return CGSize(width: width, height: max(leading.height, trailing.height))
        }
        let leading = subviews[0].sizeThatFits(ProposedViewSize(width: width, height: nil))
        let trailing = subviews[1].sizeThatFits(ProposedViewSize(width: width, height: nil))
        return CGSize(width: width, height: leading.height + stackedSpacing + trailing.height)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        guard subviews.count == 2 else { return }
        if isWide(bounds.width) {
            let leading = leadingWidth(in: bounds.width)
            subviews[0].place(
                at: CGPoint(x: bounds.minX, y: bounds.minY), anchor: .topLeading,
                proposal: ProposedViewSize(width: leading, height: nil))
            subviews[1].place(
                at: CGPoint(x: bounds.minX + leading + spacing, y: bounds.minY),
                anchor: .topLeading,
                proposal: ProposedViewSize(width: trailingWidth, height: nil))
            return
        }
        let leadingHeight = subviews[0]
            .sizeThatFits(ProposedViewSize(width: bounds.width, height: nil)).height
        subviews[0].place(
            at: CGPoint(x: bounds.minX, y: bounds.minY), anchor: .topLeading,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
        subviews[1].place(
            at: CGPoint(x: bounds.minX, y: bounds.minY + leadingHeight + stackedSpacing),
            anchor: .topLeading,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
    }

    private func isWide(_ width: CGFloat) -> Bool { width >= minimumWideWidth }

    private func leadingWidth(in width: CGFloat) -> CGFloat {
        max(0, width - spacing - trailingWidth)
    }
}

/// A borderless definition list. Name on the left in the page's third ink,
/// value on the right in its first.
struct MRFactList<Content: View>: View {
    var labelWidth: CGFloat = 108
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            content
        }
        .environment(\.mrFactLabelWidth, labelWidth)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// One line of a definition list.
struct MRFact<Value: View>: View {
    let label: String
    @ViewBuilder var value: Value

    @Environment(\.mrFactLabelWidth) private var labelWidth

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s3) {
            Text(label)
                .font(MRType.prose)
                .foregroundStyle(MRColor.tertiary)
                .frame(width: labelWidth, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            value
                .frame(maxWidth: .infinity, alignment: .leading)
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
    }
}

extension MRFact where Value == MRFactValue {
    /// The common case: a label and a string, in the face its provenance
    /// earns. Measured is upright and tabular; declared is italic, exactly as
    /// invariant 3 requires, and neither is mono.
    init(_ label: String, _ text: String, provenance: ValueText.Provenance = .measured) {
        self.init(label: label) { MRFactValue(text: text, provenance: provenance) }
    }
}

struct MRFactValue: View {
    let text: String
    var provenance: ValueText.Provenance = .measured

    var body: some View {
        switch provenance {
        case .measured:
            Text(text)
                .font(MRType.figure)
                .foregroundStyle(MRColor.primary)
                .fixedSize(horizontal: false, vertical: true)
        case .declared:
            Text(text)
                .font(MRType.declaredSmall)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityLabel("\(text), declared, not yet checked against files")
        case .projected:
            Text("≈ \(text)")
                .font(MRType.figure)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityLabel("approximately \(text), a projection")
        }
    }
}

private struct MRFactLabelWidthKey: EnvironmentKey {
    static let defaultValue: CGFloat = 108
}

extension EnvironmentValues {
    /// The width of a definition list's name column, set once by the list so
    /// every value in it starts at the same x.
    var mrFactLabelWidth: CGFloat {
        get { self[MRFactLabelWidthKey.self] }
        set { self[MRFactLabelWidthKey.self] = newValue }
    }
}

/// One readiness item: a dot, a name, and the sentence that says what the dot
/// means. This is what replaced two grey monospaced pills.
struct MRReadinessItem: View {
    let tone: MRPageStatusTone
    let title: String
    let sentence: String

    var body: some View {
        HStack(alignment: .top, spacing: MRSpace.s2) {
            MRStatusDot(tone: tone)
                .padding(.top, 5)
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(MRType.prose.weight(.medium))
                    .foregroundStyle(MRColor.primary)
                Text(sentence)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title), \(tone.spokenState)")
        .accessibilityValue(sentence)
    }
}

/// A dot and one short phrase, on one line.
///
/// The compact form of ``MRReadinessItem``, for a readiness fact that lives in
/// a definition list where the label column already says which fact it is. Two
/// of these are what replaced a third of the model page's width: the readiness
/// column spent it on two wrapped sentences, and pushed the copies on the drive
/// — the part of the page an operator can act on — below the fold.
struct MRReadinessLine: View {
    let tone: MRPageStatusTone
    /// Says the state in words, because the dot's colour is not in the
    /// accessibility tree and must not be the only thing carrying it.
    let phrase: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
            MRStatusDot(tone: tone, diameter: 7)
                .alignmentGuide(.firstTextBaseline) { _ in 6 }
            Text(phrase)
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }
}

// MARK: - Notes

/// A note the operator can act on: a tinted ground, an icon, and an
/// instruction.
///
/// The rule that goes with it — and it is the point of the component — is that
/// a note is drawn **only when there is something to do**. The orange
/// monospaced sentence this replaced appeared under a running transfer to
/// explain why a button the operator could not see was disabled.
struct MRInlineNote: View {
    let message: String
    var title: String?
    var tone: MRPageStatusTone = .attention
    var systemImage: String?

    private var ground: Color {
        switch tone {
        case .attention: return MRColor.cautionSoft
        case .moving, .ready: return MRColor.accentSoft
        case .idle: return MRColor.raised
        }
    }

    private var glyph: String {
        if let systemImage { return systemImage }
        switch tone {
        case .attention: return "exclamationmark.triangle"
        case .ready: return "checkmark.circle"
        case .moving, .idle: return "info.circle"
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: MRSpace.s2) {
            Image(systemName: glyph)
                .imageScale(.medium)
                .foregroundStyle(tone.color)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                if let title {
                    Text(title)
                        .font(MRType.prose.weight(.medium))
                        .foregroundStyle(MRColor.primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Text(message)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: MRRadius.card, style: .continuous).fill(ground)
        )
        .accessibilityElement(children: .combine)
    }
}

/// A row in a quiet list — an earlier transfer, a local copy — with its own
/// trailing actions. Rows, hairlines, and nothing else.
struct MRQuietRow<Leading: View, Trailing: View>: View {
    @ViewBuilder var leading: Leading
    @ViewBuilder var trailing: Trailing

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: MRSpace.s4) {
                leading
                Spacer(minLength: MRSpace.s3)
                HStack(spacing: MRSpace.s3) { trailing }
            }
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                leading
                HStack(spacing: MRSpace.s3) { trailing }
            }
        }
        .padding(.vertical, MRSpace.s2)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Lists
//
// A list in this language is rows and hairlines. It is not a stack of cards,
// and it is not a grid of pills: every row says one thing about one subject —
// what it is, where it stands, how big it is — and the eye runs down the
// trailing column comparing states, which is the only reason a list exists.

/// A dot and a short sentence at the end of a list row.
///
/// `MRStatusLine` is the page-wide form: it opens at the leading edge and wraps
/// as prose under a heading. A row's status lives in the trailing column, where
/// it has to line up with the status above it, so this one right-aligns its
/// sentence and takes the smaller dot that a 13-point row can carry.
struct MRRowStatus: View {
    let tone: MRPageStatusTone
    let sentence: String
    var alignment: TextAlignment = .trailing

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
            MRStatusDot(tone: tone, diameter: 7)
                .alignmentGuide(.firstTextBaseline) { _ in 6 }
            Text(sentence)
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
                .multilineTextAlignment(alignment)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("status, \(tone.spokenState)")
        .accessibilityValue(sentence)
    }
}

/// One row of a product list: an identity on the leading edge, and the quiet
/// trailing column that says where this one stands.
///
/// The split is `MRColumns` and not `ViewThatFits`, for the reason `MRColumns`
/// itself documents: a wrapping sentence answers the "ideal width" question
/// with the width of its longest line, so a `ViewThatFits` row full of
/// sentences takes the stacked layout in a 900-point window. Asking the
/// container how wide it is is the actual question.
struct MRListRow<Leading: View, Trailing: View>: View {
    /// Below this the trailing column moves under the identity.
    var minimumWideWidth: CGFloat = 520
    var trailingWidth: CGFloat = 230
    @ViewBuilder var leading: Leading
    @ViewBuilder var trailing: Trailing

    var body: some View {
        MRColumns(
            spacing: MRSpace.s4,
            minimumWideWidth: minimumWideWidth,
            trailingWidth: trailingWidth,
            stackedSpacing: MRSpace.s2
        ) {
            leading
            VStack(alignment: .trailing, spacing: MRSpace.s1) { trailing }
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(.vertical, MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
    }
}

/// The identity half of a list row: the publisher's own mark, the name, and
/// the one line that says what the thing is.
struct MRRowIdentity<Mark: View>: View {
    let title: String
    var subtitle: String?
    /// A third line, in the page's third ink — what this build can do with the
    /// model, or what the last run cost. Absent far more often than present.
    var note: String?
    var noteColor: Color = MRColor.tertiary
    @ViewBuilder var mark: Mark

    var body: some View {
        HStack(alignment: .top, spacing: MRSpace.s3) {
            mark
            VStack(alignment: .leading, spacing: MRSpace.s1) {
                Text(title)
                    .font(MRType.headline)
                    .foregroundStyle(MRColor.primary)
                    .fixedSize(horizontal: false, vertical: true)
                if let subtitle {
                    Text(subtitle)
                        .font(MRType.prose)
                        .foregroundStyle(MRColor.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let note {
                    Text(note)
                        .font(MRType.prose)
                        .foregroundStyle(noteColor)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
    }
}

/// The chevron a row that opens something wears. Nothing else in this language
/// draws one: a row without a destination must not look like a row with one.
struct MRDisclosureChevron: View {
    var body: some View {
        Image(systemName: "chevron.right")
            .font(.caption.weight(.semibold))
            .foregroundStyle(MRColor.tertiary)
            .accessibilityHidden(true)
    }
}

/// A quantity in a list column: upright and tabular when it was counted,
/// italic when it is still the repository's own claim.
///
/// Invariant 3 in the one place a list is tempted to break it. A column of
/// sizes lines up either way, because the italic face takes tabular figures
/// too — the distinction the invariant needs is upright versus italic, not
/// aligned versus ragged.
struct MRRowQuantity: View {
    let text: String
    var provenance: ValueText.Provenance = .measured

    var body: some View {
        switch provenance {
        case .measured:
            Text(text)
                .font(MRType.figure)
                .foregroundStyle(MRColor.primary)
        case .declared:
            Text(text)
                .font(MRType.declaredSmall.monospacedDigit())
                .foregroundStyle(MRColor.secondary)
                .accessibilityLabel("\(text), declared, not yet checked against files")
        case .projected:
            Text("≈ \(text)")
                .font(MRType.figure)
                .foregroundStyle(MRColor.secondary)
                .accessibilityLabel("approximately \(text), a projection")
        }
    }
}
