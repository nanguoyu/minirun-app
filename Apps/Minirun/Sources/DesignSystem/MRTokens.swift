import SwiftUI

#if canImport(AppKit)
    import AppKit
#endif
#if canImport(UIKit)
    import UIKit
#endif

// The design tokens from the product specification, verbatim where the
// specification gives a value. Dark-first; light is derived, not an
// afterthought. No view in this app may reference a hex literal — if a colour
// is needed and is not here, it is a token that has not been designed yet.

// MARK: - Colour

/// A colour that resolves differently in light and dark, built once at the
/// token and never at a call site.
private func dyn(_ dark: UInt32, _ light: UInt32) -> Color {
    #if canImport(AppKit)
        return Color(
            nsColor: NSColor(name: nil) { appearance in
                let isDark =
                    appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
                    || appearance.bestMatch(from: [
                        .accessibilityHighContrastAqua, .accessibilityHighContrastDarkAqua,
                    ]) == .accessibilityHighContrastDarkAqua
                return NSColor(rgb: isDark ? dark : light)
            })
    #elseif canImport(UIKit)
        return Color(
            uiColor: UIColor { traits in
                UIColor(rgb: traits.userInterfaceStyle == .dark ? dark : light)
            })
    #else
        return Color(rgb: dark)
    #endif
}

public enum MRColor {
    // Surfaces                       dark        light
    public static let abyss = dyn(0x0B0D10, 0xF7F8FA)  // app background
    public static let panel = dyn(0x14171C, 0xFFFFFF)  // cards, instrument panel
    public static let raised = dyn(0x1C2027, 0xFFFFFF)  // controls on panel
    public static let hairline = dyn(0x2A2F38, 0xDCE0E6)  // 1px separators, empty track

    // Text
    //
    // Every one of these, and every state colour below that is ever set as a
    // foreground, clears 4.5:1 against `abyss`, `panel` and `raised` in BOTH
    // appearances. `MRTokenContrastTests` proves it rather than asserting it;
    // the light column used to sit at 2.96:1 for `tertiary`, which is most of
    // the small print in the product.
    public static let primary = dyn(0xE8EBEF, 0x101317)
    public static let secondary = dyn(0x9BA3AF, 0x5A626D)
    public static let tertiary = dyn(0x8A929E, 0x6B727C)

    // Memory tiers — the dial's own palette, used nowhere else.
    public static let tierFloor = dyn(0x8B93A1, 0x6E7681)  // working floor: immovable, grey
    // Staged and pinned were one colour while the dial had one tier for both,
    // and the dial was wrong about that: a staged layer is a transient buffer
    // that exists for one pair, and a pinned layer is resident for the life of
    // the run. They cost different bytes, they are decided by different rules,
    // and drawing them in the same blue made a bar that could not show the
    // difference between renting and owning.
    public static let tierStaged = dyn(0x7ED0E8, 0x1B6D80)  // transient read-ahead pair
    public static let tierPinned = dyn(0x5B8DEF, 0x2A62C9)  // layers held resident
    public static let tierHot = dyn(0x37B98A, 0x1E8F66)  // expert hot set
    public static let tierFree = dyn(0x2A2F38, 0xE4E8ED)  // unallocated headroom

    // Streams — the ribbon, the layer ladder, and the projection curve, which
    // is drawn as a line and must therefore survive the same contrast floor.
    public static let streamDet = dyn(0x4FA8FF, 0x1A64C4)
    public static let streamExp = dyn(0x37B98A, 0x1E8F66)

    // State. These are set as foregrounds far more often than as fills, so the
    // light column is the darker one — a mid-green caption on white is the
    // classic 3:1 that reads fine to the person who picked it and to nobody
    // else.
    public static let ok = dyn(0x37B98A, 0x137553)
    public static let caution = dyn(0xE8A33D, 0x8A5A00)  // thermal .fair, low power
    public static let refuse = dyn(0xF2686C, 0xC62A2F)  // refusal, .serious/.critical
    public static let verify = dyn(0xA78BFA, 0x6B3ED8)  // integrity verification

    // MARK: The product accent
    //
    // One indigo, and it means exactly one thing: *this is moving, or this is
    // something to press*. The transfer bar, the status dot of a running job,
    // and every text link on a product page.
    //
    // It is not `tierPinned`. The dial's five tier colours are a legend for a
    // bar chart of memory, and the download card borrowed the pinned blue for
    // its progress tint because it was the nearest blue in the file — which
    // made a transfer look like a memory tier. The dial keeps its palette; a
    // product page uses this.
    public static let accent = dyn(0x6D8FF0, 0x2F5BD7)

    /// The accent as a wash: the halo around a live status dot, a selected
    /// row, an inline note that is informational rather than a warning.
    public static var accentSoft: Color { accent.opacity(0.14) }
    /// A warning note's ground. `caution` remains the foreground on it.
    public static var cautionSoft: Color { caution.opacity(0.12) }
    /// A finished state's halo.
    public static var okSoft: Color { ok.opacity(0.16) }
}

// MARK: - Type
//
// SF Pro for prose, SF Mono for every number.
//
// The specification gives fixed point sizes. They are reproduced here through
// the matching Dynamic Type text styles rather than `Font.system(size:)`,
// because a fixed size does not scale and the product ships on a phone. Each
// token resolves to the specification's size at the default content size and
// grows from there; the spec size is in the comment so the mapping is checkable
// rather than asserted.

public enum MRType {
    /// 22 pt semibold.
    public static let title = Font.system(.title2, design: .default, weight: .semibold)
    /// 17 pt semibold.
    public static let headline = Font.system(.headline, design: .default, weight: .semibold)
    /// 15 pt regular.
    public static let body = Font.system(.subheadline, design: .default, weight: .regular)
    /// 13 pt regular.
    public static let caption = Font.system(.footnote, design: .default, weight: .regular)
    /// 11 pt medium; always paired with `.mrLabel()` for tracking and case.
    public static let label = Font.system(.caption2, design: .default, weight: .medium)

    // Telemetry — always monospaced, always tabular.
    /// 28 pt medium mono.
    public static let readout = Font.system(.title, design: .monospaced, weight: .medium)
        .monospacedDigit()
    /// 15 pt regular mono.
    public static let metric = Font.system(.subheadline, design: .monospaced, weight: .regular)
        .monospacedDigit()
    /// 11 pt regular mono.
    public static let micro = Font.system(.caption2, design: .monospaced, weight: .regular)
        .monospacedDigit()

    /// Small explanatory prose: a status line, a section caption, a sentence
    /// under a control.
    ///
    /// `micro` is a telemetry token — mono, tabular — and prose set in it reads
    /// as a code listing. That is the house voice on the Mac, where the product
    /// is an instrument panel; on the phone it made General, Models and Storage
    /// look like terminal output. Numbers keep `metric`/`readout`/`micro` on
    /// both platforms.
    public static var smallProse: Font {
        #if os(iOS)
            Font.caption
        #else
            micro
        #endif
    }

    /// A quantity that has NOT been checked against files. Italic SF Pro, never
    /// mono: the type itself tells you whether a number has been verified.
    public static let declared = Font.system(.subheadline, design: .default, weight: .regular)
        .italic()
    /// The small form of the same.
    public static let declaredSmall = Font.system(.footnote, design: .default, weight: .regular)
        .italic()

    // MARK: - Product page
    //
    // The instrument-panel voice above is right for a running model and wrong
    // for the page that introduces one. A product page is read the way a
    // product page is read — a name, a sentence, a number — and the panel's
    // mono-everything turned that page into a terminal listing.
    //
    // The rule here: SF Pro throughout, with `.monospacedDigit()` wherever
    // figures have to line up, and SF Mono reserved for paths, hashes and true
    // code. Invariant 3 — the type says whether a number was checked — still
    // holds, because the distinction it needs is upright versus italic:
    // `declared` is still italic and still never mono.

    /// 28 pt semibold, the model's own name. The specification asks for 26;
    /// `.title` is the text style that resolves nearest to it and, unlike a
    /// fixed 26, grows with Dynamic Type. Pair with `.tracking(-0.5)`.
    public static let pageTitle = Font.system(.title, design: .default, weight: .semibold)
    /// 15 pt, the one line under the name: what it is, how big, whose licence.
    public static let pageSubtitle = Font.system(.subheadline, design: .default, weight: .regular)
    /// 13 pt semibold. A section's name, in sentence case — the ALL-CAPS
    /// tracked micro-label is not used on a product page.
    public static let sectionHeading = Font.system(.footnote, design: .default, weight: .semibold)
    /// 13 pt. The page's body voice: a status sentence, a definition, a note.
    public static let prose = Font.system(.footnote, design: .default, weight: .regular)
    /// 13 pt with tabular figures. Every quantity in a list that has to align
    /// with the quantity above it.
    public static let figure = Font.system(.footnote, design: .default, weight: .regular)
        .monospacedDigit()
    /// 34 pt semibold tabular. The one big number a block is allowed.
    public static let bigFigure = Font.system(.largeTitle, design: .default, weight: .semibold)
        .monospacedDigit()
    /// 15 pt, the quieter half of a big number — "of 517 GB".
    public static let bigFigureDetail = Font.system(
        .subheadline, design: .default, weight: .regular
    )
    .monospacedDigit()
    /// 13 pt medium. A button, a text link.
    public static let control = Font.system(.footnote, design: .default, weight: .medium)
    /// 12 pt mono. A filesystem path, a revision, a named error — and nothing
    /// else.
    public static let path = Font.system(.caption, design: .monospaced, weight: .regular)
}

public enum MRSpace {
    public static let s0: CGFloat = 2
    public static let s1: CGFloat = 4
    public static let s2: CGFloat = 8
    public static let s3: CGFloat = 12
    public static let s4: CGFloat = 16
    public static let s5: CGFloat = 24
    public static let s6: CGFloat = 32
    public static let s7: CGFloat = 48
}

/// The composer row's one shared height.
///
/// The message field and the Send/Stop button beside it are one row, and a row
/// whose two controls measure themselves independently is the row that shipped:
/// a 40-point field with a 22-point button pinned to its bottom edge. Both
/// controls take this height, so a single-line composer is one horizontal band
/// and the button stays with the field's last line as it grows.
public enum MRComposer {
    public static var controlHeight: CGFloat {
        #if os(iOS)
            return MRAccessibility.minimumIOSTouchTarget
        #else
            return 34
        #endif
    }
}

public enum MRRadius {
    public static let control: CGFloat = 6
    public static let card: CGFloat = 10
    public static let panel: CGFloat = 14
    public static let pill: CGFloat = 999
    /// A button on a product page: 8, not 6 — it sits beside 26 pt type.
    public static let action: CGFloat = 8
    /// The publisher tile at the top of a product page. 16 on 64 points is the
    /// squircle every app icon on the platform already wears.
    public static let tile: CGFloat = 16
}

public enum MRMotion {
    public static let quick: Double = 0.12  // hover, chip state
    public static let standard: Double = 0.20  // panel/sheet transitions
    public static let deliberate: Double = 0.32  // dial tier resize
    public static let tierSpring = Animation.spring(response: 0.32, dampingFraction: 0.86)
}

// MARK: - Shared view modifiers

extension View {
    /// The uppercase, tracked micro-label the specification asks for.
    public func mrLabel(_ color: Color = MRColor.tertiary) -> some View {
        self
            .font(MRType.label)
            .tracking(0.6)
            .textCase(.uppercase)
            .foregroundStyle(color)
    }

    /// A panel card: the one surface treatment in the product. No gradient, no
    /// shadow, no glow — a hairline and a fill.
    public func mrCard(
        _ fill: Color = MRColor.panel,
        stroke: Color = MRColor.hairline,
        radius: CGFloat = MRRadius.card
    ) -> some View {
        self
            .background(
                RoundedRectangle(cornerRadius: radius, style: .continuous).fill(fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(stroke, lineWidth: 1)
            )
    }

    /// The opaque workspace surface. A sidebar is allowed to borrow the
    /// desktop through macOS material; the transcript, Settings detail and
    /// inspector are reading surfaces and must remain calm and legible over
    /// every wallpaper.
    @ViewBuilder
    public func mrWorkspaceSurface() -> some View {
        self.background(MRColor.abyss.ignoresSafeArea())
    }

    /// A product page: one surface, edge to edge.
    ///
    /// The screens built as instrument panels stack cards on `abyss`. A page
    /// that introduces a product is the opposite arrangement — a single sheet
    /// divided by hairlines, with weight spent only on the one block that
    /// needs lifting — so it takes `panel` for the whole column rather than for
    /// six rectangles drawn on top of it.
    @ViewBuilder
    public func mrProductPage() -> some View {
        self.background(MRColor.panel.ignoresSafeArea())
    }

    /// The one translucent navigation surface in the macOS product. Keeping
    /// this separate from `mrWorkspaceSurface()` preserves the native sidebar
    /// relationship without washing the entire window with the wallpaper.
    @ViewBuilder
    public func mrSidebarSurface() -> some View {
        #if os(macOS)
            self.background(MRSidebarMaterial().ignoresSafeArea())
        #else
            self.background(MRColor.abyss.ignoresSafeArea())
        #endif
    }

    /// One background story for iOS Settings.
    ///
    /// A `List` or `Form` inside a navigation stack paints the system's grouped
    /// background — pure black in dark mode — while every other page in this
    /// product paints `abyss`. The Settings root was the one screen wearing the
    /// system's colour, and the seam was visible the moment a section was
    /// pushed. These two modifiers put the page on `abyss` and the grouped
    /// cards on `raised`, which is the same relationship the rest of the app
    /// already draws with `mrCard`.
    @ViewBuilder
    public func mrSettingsList() -> some View {
        #if os(iOS)
            self
                .scrollContentBackground(.hidden)
                .background(MRColor.abyss.ignoresSafeArea())
        #else
            self
        #endif
    }

    /// The card a grouped Settings row sits on. macOS draws its own panels.
    @ViewBuilder
    public func mrSettingsRow() -> some View {
        #if os(iOS)
            self.listRowBackground(MRColor.raised)
        #else
            self
        #endif
    }

    /// Navigation titles belong to the phone's navigation bar. macOS owns its
    /// column headers explicitly so a pushed detail can never place text over
    /// the Settings sidebar or summon a separated window-toolbar group.
    @ViewBuilder
    public func mrPhoneNavigationTitle(_ title: String) -> some View {
        #if os(iOS)
            self.navigationTitle(title)
        #else
            self
        #endif
    }
}

#if os(macOS)
    private struct MRSidebarMaterial: NSViewRepresentable {
        func makeNSView(context: Context) -> NSVisualEffectView {
            let view = NSVisualEffectView()
            view.material = .sidebar
            view.blendingMode = .behindWindow
            view.state = .followsWindowActiveState
            return view
        }

        func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
    }
#endif

// MARK: - Contrast-aware hairline
//
// Increase Contrast: hairline becomes `tertiary`, hatching becomes solid at 30%.
// Both rules live here so no component re-decides them.

public struct MRContrast {
    public let increased: Bool
    public init(increased: Bool) { self.increased = increased }

    public var hairline: Color { increased ? MRColor.tertiary : MRColor.hairline }
    /// Hatching is drawn as a pattern normally, and as a flat 30 % wash when
    /// the operator has asked for more contrast.
    public var hatchIsSolid: Bool { increased }
    public var solidHatchOpacity: Double { 0.30 }
}

extension EnvironmentValues {
    public var mrContrast: MRContrast {
        MRContrast(increased: colorSchemeContrast == .increased)
    }
}

// MARK: - Colour construction helpers

#if canImport(AppKit)
    extension NSColor {
        fileprivate convenience init(rgb: UInt32) {
            self.init(
                srgbRed: Double((rgb >> 16) & 0xFF) / 255,
                green: Double((rgb >> 8) & 0xFF) / 255,
                blue: Double(rgb & 0xFF) / 255,
                alpha: 1)
        }
    }
#endif

#if canImport(UIKit)
    extension UIColor {
        fileprivate convenience init(rgb: UInt32) {
            self.init(
                red: Double((rgb >> 16) & 0xFF) / 255,
                green: Double((rgb >> 8) & 0xFF) / 255,
                blue: Double(rgb & 0xFF) / 255,
                alpha: 1)
        }
    }
#endif

extension Color {
    fileprivate init(rgb: UInt32) {
        self.init(
            .sRGB,
            red: Double((rgb >> 16) & 0xFF) / 255,
            green: Double((rgb >> 8) & 0xFF) / 255,
            blue: Double(rgb & 0xFF) / 255,
            opacity: 1)
    }
}
