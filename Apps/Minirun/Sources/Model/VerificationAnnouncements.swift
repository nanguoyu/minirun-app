import Foundation
import UserNotifications

#if os(iOS)
    import UIKit
#elseif os(macOS)
    import AppKit
#endif

/// What a finished verification has to say for itself.
struct VerificationAnnouncement: Equatable, Sendable {
    let rootPath: String
    let title: String
    let body: String
    let isFailure: Bool
}

/// Tells the operator a verification ended while they were somewhere else.
///
/// A full K3 pass is half an hour. Nobody watches a progress bar for half an
/// hour, which is the entire reason the pass now survives being backgrounded —
/// and a result nobody is told about is not much better than a result that
/// never arrived.
@MainActor
protocol VerificationAnnouncing: AnyObject {
    /// Ask for permission to post. Called when a verification starts, and not
    /// at launch: at launch there is nothing to be notified about yet.
    func prepare()
    func announce(_ announcement: VerificationAnnouncement)
}

/// Whether the operator is looking at this app right now.
enum VerificationAnnouncementPolicy {
    @MainActor
    static func applicationIsFrontmost() -> Bool {
        #if os(iOS)
            return UIApplication.shared.applicationState == .active
        #elseif os(macOS)
            // Deliberately `isActive` rather than "a window exists": a Minirun
            // window buried behind Xcode is exactly the case the notification
            // is for.
            return NSApplication.shared.isActive
        #else
            return true
        #endif
    }
}

/// The real thing, on both platforms, through one code path.
///
/// macOS gets the same notification for the same reason iOS does — the window
/// is often not the front window — and the only platform difference is how
/// "frontmost" is spelled, which lives in the policy above.
@MainActor
final class UserNotificationAnnouncer: NSObject, VerificationAnnouncing {
    static let categoryIdentifier = "minirun.verification"
    /// Read by the delegate below, and by nothing else. A notification payload
    /// is untrusted input in general; this one carries a constant.
    static let sectionKey = "minirun.settingsSection"

    /// Where a tap lands. Wired by `AppModel.live`.
    var onOpenModels: (@MainActor () -> Void)?

    private var didRequestAuthorization = false

    /// Start receiving taps. Called at launch, because a tap may be what
    /// launched the app.
    func activate() {
        UNUserNotificationCenter.current().delegate = self
    }

    func prepare() {
        guard !didRequestAuthorization else { return }
        didRequestAuthorization = true
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) {
            _, _ in
            // A refusal is the operator's answer and needs no screen of its
            // own: the row still shows the result when they come back.
        }
    }

    func announce(_ announcement: VerificationAnnouncement) {
        let content = UNMutableNotificationContent()
        content.title = announcement.title
        content.body = announcement.body
        content.sound = .default
        content.categoryIdentifier = Self.categoryIdentifier
        content.userInfo = [Self.sectionKey: AppModel.SettingsSection.models.rawValue]
        // No trigger: post it now. The work it describes has already happened.
        UNUserNotificationCenter.current().add(
            UNNotificationRequest(
                identifier: "minirun.verification.\(UUID().uuidString)",
                content: content, trigger: nil))
    }
}

extension UserNotificationAnnouncer: UNUserNotificationCenterDelegate {
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        // Nothing is posted while the app is frontmost, so reaching this means
        // the operator arrived between the post and its delivery. The row in
        // front of them is the better answer.
        []
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        let section = response.notification.request.content.userInfo[Self.sectionKey] as? String
        guard section == AppModel.SettingsSection.models.rawValue else { return }
        await MainActor.run { self.onOpenModels?() }
    }
}
