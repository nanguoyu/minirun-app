import SwiftUI

/// A product About page: what Minirun is for and which version is open.
/// Engineering diagnostics belong in development tools, not in the ordinary
/// settings hierarchy somebody uses to understand the product.
struct AboutView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MRSpace.s6) {
                masthead
                links
            }
                .padding(pagePadding)
                .frame(maxWidth: 640, alignment: .leading)
                .frame(maxWidth: .infinity)
        }
        .frame(maxWidth: .infinity)
        .mrWorkspaceSurface()
        .mrPhoneNavigationTitle("About")
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    /// Product identity and purpose belong together. The horizontal form keeps
    /// the Mac page grounded in the Settings grid; the compact form keeps the
    /// same hierarchy without squeezing the copy on iPhone.
    private var masthead: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: MRSpace.s4) {
                logo
                productCopy(alignment: .leading, textAlignment: .leading)
            }
            VStack(spacing: MRSpace.s3) {
                logo
                productCopy(alignment: .center, textAlignment: .center)
            }
        }
    }

    private var logo: some View {
        Image("MinirunLogo")
            .renderingMode(.original)
            .resizable()
            .scaledToFit()
            .frame(width: 96, height: 96)
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .strokeBorder(MRColor.hairline, lineWidth: 1)
            }
            .accessibilityHidden(true)
    }

    private func productCopy(
        alignment: HorizontalAlignment,
        textAlignment: TextAlignment
    ) -> some View {
        VStack(alignment: alignment, spacing: MRSpace.s2) {
            Text(AboutPresentation.productName)
                .font(MRType.title)
                .foregroundStyle(MRColor.primary)
            Text(versionLine)
                .font(MRType.caption)
                .foregroundStyle(MRColor.tertiary)
                .textSelection(.enabled)
            Text(AboutPresentation.purpose)
                .font(MRType.body)
                .foregroundStyle(MRColor.secondary)
                .multilineTextAlignment(textAlignment)
                .fixedSize(horizontal: false, vertical: true)
            Text(AboutPresentation.privacyNote)
                .font(MRType.caption)
                .foregroundStyle(MRColor.tertiary)
                .multilineTextAlignment(textAlignment)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var links: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            SectionHeader(title: AboutPresentation.linksTitle)
            VStack(spacing: 0) {
                linkRow(
                    title: AboutPresentation.websiteTitle,
                    label: AboutPresentation.websiteLabel,
                    destination: AboutPresentation.websiteURL,
                    systemImage: "globe",
                    fill: MRColor.tierPinned)
                linkDivider
                linkRow(
                    title: AboutPresentation.privacyTitle,
                    label: AboutPresentation.privacyLabel,
                    destination: AboutPresentation.privacyURL,
                    systemImage: "hand.raised.fill",
                    fill: MRColor.tierHot)
                linkDivider
                linkRow(
                    title: AboutPresentation.termsTitle,
                    label: AboutPresentation.termsLabel,
                    destination: AboutPresentation.termsURL,
                    systemImage: "doc.text.fill",
                    fill: MRColor.verify)
            }
            .mrCard()
        }
    }

    private var linkDivider: some View {
        Divider()
            .overlay(MRColor.hairline)
            .padding(.leading, 64)
    }

    private func linkRow(
        title: String,
        label: String,
        destination: URL,
        systemImage: String,
        fill: Color
    ) -> some View {
        Link(destination: destination) {
            HStack(spacing: MRSpace.s3) {
                ZStack {
                    RoundedRectangle(cornerRadius: MRRadius.control, style: .continuous)
                        .fill(fill)
                    Image(systemName: systemImage)
                        .font(.system(.body, weight: .semibold))
                        .foregroundStyle(.white)
                }
                .frame(width: 40, height: 40)
                .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: MRSpace.s0) {
                    Text(title)
                        .font(MRType.body)
                        .foregroundStyle(MRColor.primary)
                    Text(label)
                        .font(MRType.caption)
                        .foregroundStyle(MRColor.tertiary)
                }
                Spacer(minLength: MRSpace.s3)
                Image(systemName: "arrow.up.right")
                    .imageScale(.small)
                    .foregroundStyle(MRColor.secondary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, MRSpace.s3)
            .padding(.vertical, MRSpace.s2)
            .frame(maxWidth: .infinity, minHeight: 60, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(title), \(label)")
        .accessibilityHint("Opens in your browser")
    }

    private var versionLine: String {
        let version =
            (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String)
            ?? "—"
        return AboutPresentation.versionLine(version)
    }

}

enum AboutPresentation {
    static let productName = "Minirun"
    static let purpose = "Run large models on this device, streaming their weights from disk."
    static let privacyNote =
        "Conversations stay in Minirun's app data. Model files stay in folders you choose."
    static let linksTitle = "Links"
    static let websiteTitle = "Website"
    static let privacyTitle = "Privacy Policy"
    static let termsTitle = "Terms of Service"
    static func versionLine(_ shortVersion: String) -> String { "Version \(shortVersion)" }
    static let websiteLabel = "minirun.dev"
    static let websiteURL = URL(string: "https://minirun.dev")!
    static let privacyLabel = "minirun.dev/privacy"
    static let privacyURL = URL(string: "https://minirun.dev/privacy")!
    static let termsLabel = "minirun.dev/terms"
    static let termsURL = URL(string: "https://minirun.dev/terms")!
}
