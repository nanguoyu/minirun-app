import SwiftUI

/// A product About page: what Minirun is for, which build is open, and where
/// its terms and its borrowed artwork come from. Engineering diagnostics belong
/// in development tools, not in the ordinary settings hierarchy somebody uses
/// to understand the product.
///
/// In the product-page language (DESIGN I.37) this is the same shape as the
/// model page: a mark, a name, one line of what it is, then facts. The three
/// 60-point tinted link rows are gone — a website, a privacy policy and a set
/// of terms are not three products, and dressing them as rows with coloured
/// icons gave them the weight of one.
struct AboutView: View {
    var body: some View {
        ScrollView {
            AboutPage()
        }
        .frame(maxWidth: .infinity)
        .mrProductPage()
        .mrPhoneNavigationTitle("About")
    }
}

/// The About page, without the scroll view around it — the same split the model
/// page made, so `ImageRenderer` can draw it offscreen for review.
struct AboutPage: View {
    @Environment(\.openURL) private var openURL

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            MRProductHeader(
                title: AboutPresentation.productName,
                subtitle: AboutPresentation.purpose
            ) {
                logo
            } actions: {
                EmptyView()
            }

            Text(AboutPresentation.privacyNote)
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, MRSpace.s3)

            build
            links
            icons
        }
        .padding(pagePadding)
        .frame(maxWidth: 720, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    private var logo: some View {
        MRProductTile {
            Image("MinirunLogo")
                .renderingMode(.original)
                .resizable()
                .scaledToFit()
                .frame(width: 64, height: 64)
                .clipShape(RoundedRectangle(cornerRadius: MRRadius.tile, style: .continuous))
        }
    }

    // MARK: - This build

    /// Two figures, not one composed sentence. A marketing version and a build
    /// number answer different questions — *which release is this* and *which
    /// exact binary* — and a reader comparing one against a release note should
    /// not have to pick it out of a parenthesis.
    private var build: some View {
        MRPageSection(title: AboutPresentation.buildTitle) {
            MRFactList(labelWidth: 96) {
                MRFact(AboutPresentation.versionLabel, Self.infoString("CFBundleShortVersionString"))
                MRFact(AboutPresentation.buildLabel, Self.infoString("CFBundleVersion"))
            }
        }
    }

    /// The bundle's own value, or a dash. A missing key is a build-settings
    /// fault and says so by being visibly absent rather than by inventing a
    /// version number.
    static func infoString(_ key: String) -> String {
        (Bundle.main.object(forInfoDictionaryKey: key) as? String) ?? "—"
    }

    // MARK: - Links

    private var links: some View {
        MRPageSection(title: AboutPresentation.linksTitle) {
            MRFactList(labelWidth: 96) {
                linkFact(
                    label: AboutPresentation.websiteTitle,
                    title: AboutPresentation.websiteLabel,
                    destination: AboutPresentation.websiteURL)
                linkFact(
                    label: AboutPresentation.privacyTitle,
                    title: AboutPresentation.privacyLabel,
                    destination: AboutPresentation.privacyURL)
                linkFact(
                    label: AboutPresentation.termsTitle,
                    title: AboutPresentation.termsLabel,
                    destination: AboutPresentation.termsURL)
            }
        }
    }

    /// A `Button` and not a `Link`: `Link` is AppKit-backed on macOS and draws
    /// as an unrenderable placeholder in `ImageRenderer`, which is where this
    /// page is reviewed. `openURL` is what `Link` calls anyway.
    private func linkFact(label: String, title: String, destination: URL) -> some View {
        MRFact(label: label) {
            Button(title) { openURL(destination) }
                .mrTextLink()
                .accessibilityLabel("\(label), \(title)")
                .accessibilityHint("Opens in your browser")
        }
    }

    // MARK: - Icons

    /// The publisher marks are somebody else's work, under a licence with an
    /// attribution clause, so the attribution is prose on a page a person can
    /// find — not a notice file nothing in the product ever opens.
    private var icons: some View {
        MRPageSection(title: AboutPresentation.iconsTitle) {
            Text(AboutPresentation.iconsNotice)
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Button(AboutPresentation.iconsSourceLabel) {
                openURL(AboutPresentation.iconsSourceURL)
            }
            .mrTextLink()
            .accessibilityHint("Opens in your browser")
        }
    }
}

enum AboutPresentation {
    static let productName = "Minirun"
    static let purpose = "Run large models on this device, streaming their weights from disk."
    static let privacyNote =
        "Conversations stay in Minirun's app data. Model files stay in folders you choose."
    static let buildTitle = "This build"
    static let versionLabel = "Version"
    static let buildLabel = "Build"
    static let linksTitle = "Links"
    static let websiteTitle = "Website"
    static let privacyTitle = "Privacy Policy"
    static let termsTitle = "Terms of Service"
    static let websiteLabel = "minirun.dev"
    static let websiteURL = URL(string: "https://minirun.dev")!
    static let privacyLabel = "minirun.dev/privacy"
    static let privacyURL = URL(string: "https://minirun.dev/privacy")!
    static let termsLabel = "minirun.dev/terms"
    static let termsURL = URL(string: "https://minirun.dev/terms")!

    static let iconsTitle = "Icons"
    /// Verbatim in substance with `Assets.xcassets/LobeIconsNotice.dataset` and
    /// `THIRD_PARTY_NOTICES.md`: the marks are MIT-licensed artwork from Lobe
    /// Icons, and the trademarks in them belong to the publishers they name.
    static let iconsNotice =
        "The publisher marks beside each model come from Lobe Icons, used under the MIT "
        + "licence. Publisher names and logos remain trademarks of their owners and appear "
        + "here only to identify their models."
    static let iconsSourceLabel = "github.com/lobehub/lobe-icons"
    static let iconsSourceURL = URL(string: "https://github.com/lobehub/lobe-icons")!
}
