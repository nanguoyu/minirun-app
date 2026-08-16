import Foundation

#if os(iOS)
    import BackgroundTasks
    import UIKit
#endif

/// The platform capability a long verification needs while the app is not the
/// screen the operator is looking at.
///
/// A full K3 pass is about half an hour of hashing over USB. Pulling down
/// Notification Center used to end it: iOS suspends the app within seconds of
/// it going inactive, and before checkpointing existed the whole pass was lost.
/// This is the seam through which the app asks the system to keep that work
/// running, and through which the system tells the app its time is up.
///
/// It is deliberately a protocol with a no-op implementation. macOS does not
/// suspend a foreground app's work, and the tests must be able to drive
/// begin/expire/end without a device.
@MainActor
protocol VerificationBackgroundWork: AnyObject {
    /// Ask the system to keep the started work running in the background.
    ///
    /// Returns whether the system accepted. A refusal is not a reason to stop
    /// verifying — it is a reason to expect a suspension, which the checkpoint
    /// already survives.
    @discardableResult
    func begin(
        title: String, subtitle: String, totalBytes: UInt64,
        onExpiration: @escaping @MainActor () -> Void
    ) -> Bool
    func update(subtitle: String, completedBytes: UInt64, totalBytes: UInt64)
    func end(success: Bool)
    /// True between an accepted `begin` and its `end` or expiration.
    var isRunning: Bool { get }
    /// Why the last `begin` was refused, when it was. Shown, not logged.
    var refusal: String? { get }
}

/// macOS, previews, and any platform that does not suspend the app.
@MainActor
final class UnsuspendedVerificationWork: VerificationBackgroundWork {
    private(set) var isRunning = false
    let refusal: String? = nil

    init() {}

    @discardableResult
    func begin(
        title: String, subtitle: String, totalBytes: UInt64,
        onExpiration: @escaping @MainActor () -> Void
    ) -> Bool {
        isRunning = true
        return true
    }

    func update(subtitle: String, completedBytes: UInt64, totalBytes: UInt64) {}

    func end(success: Bool) { isRunning = false }
}

#if os(iOS)

    /// iOS 26+ continued processing.
    ///
    /// `BGContinuedProcessingTask` is the API for exactly this situation: work
    /// the user started in the foreground, which the system continues in the
    /// background with its own progress UI on the Dynamic Island and Lock
    /// Screen, which the user can cancel from that UI, and which the system may
    /// expire when conditions change. Expiration is not a failure — the pass
    /// checkpoints every completed file, so it becomes a pause.
    ///
    /// The identifier follows the shape the framework header requires: the
    /// bundle id, semantic context, and a unique final component, registered
    /// once through the matching `.*` wildcard. `Info-iOS.plist` lists that
    /// wildcard under `BGTaskSchedulerPermittedIdentifiers`; no
    /// `UIBackgroundModes` entry is involved, because a continued-processing
    /// task is a foreground-initiated assertion rather than a background launch.
    @available(iOS 26.0, *)
    @MainActor
    final class ContinuedProcessingVerificationWork: VerificationBackgroundWork {
        /// Must remain the bundle id plus semantic context. The submitted
        /// identifier appends one unique component below it.
        static let identifierPrefix = "wang.wangdongdong.minirun.verification"
        static var permittedIdentifier: String { identifierPrefix + ".*" }

        private static var registeredHandler = false
        private static weak var current: ContinuedProcessingVerificationWork?

        private var task: BGContinuedProcessingTask?
        private var submittedIdentifier: String?
        private var onExpiration: (@MainActor () -> Void)?
        private var totalUnits: Int64 = 0
        private var completedUnits: Int64 = 0
        private var announcedPercent = -1
        private var latestSubtitle = ""
        private(set) var isRunning = false
        private(set) var refusal: String?

        init() {}

        @discardableResult
        func begin(
            title: String, subtitle: String, totalBytes: UInt64,
            onExpiration: @escaping @MainActor () -> Void
        ) -> Bool {
            guard !isRunning else { return true }
            refusal = nil
            registerLaunchHandlerIfNeeded()
            Self.current = self

            let identifier =
                Self.identifierPrefix + "." + UUID().uuidString.prefix(8).lowercased()
            let request = BGContinuedProcessingTaskRequest(
                identifier: identifier, title: title, subtitle: subtitle)
            // `.fail` rather than `.queue`. A queued request runs when the
            // system has room, which may be long after this verification has
            // already been suspended or finished; a request that cannot start
            // now is better reported now, since the checkpoint — not the
            // scheduler — is what makes the work recoverable either way.
            request.strategy = .fail
            request.requiredResources = []

            totalUnits = Int64(clamping: totalBytes)
            completedUnits = 0
            latestSubtitle = subtitle
            announcedPercent = -1
            do {
                try BGTaskScheduler.shared.submit(request)
            } catch {
                refusal = Self.sentence(for: error)
                Self.current = nil
                return false
            }
            submittedIdentifier = identifier
            self.onExpiration = onExpiration
            isRunning = true
            return true
        }

        func update(subtitle: String, completedBytes: UInt64, totalBytes: UInt64) {
            totalUnits = Int64(clamping: totalBytes)
            completedUnits = min(Int64(clamping: completedBytes), totalUnits)
            latestSubtitle = subtitle
            guard let task else { return }
            task.progress.totalUnitCount = totalUnits
            task.progress.completedUnitCount = completedUnits
            let percent =
                totalUnits > 0 ? Int(Double(completedUnits) / Double(totalUnits) * 100) : 0
            // The system panel is not a log. Retitle on whole percentage points
            // rather than once per file.
            guard percent != announcedPercent else { return }
            announcedPercent = percent
            task.updateTitle(task.title, subtitle: subtitle)
        }

        func end(success: Bool) {
            guard isRunning else { return }
            if let task {
                task.setTaskCompleted(success: success)
            } else if let submittedIdentifier {
                BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: submittedIdentifier)
            }
            clear()
        }

        // MARK: - System callbacks

        private func registerLaunchHandlerIfNeeded() {
            // Registering the same identifier twice terminates the app, and a
            // continued-processing handler is explicitly exempt from the
            // register-before-launch-finishes rule, so this is done once here
            // rather than at startup for a feature most launches never use.
            guard !Self.registeredHandler else { return }
            Self.registeredHandler = BGTaskScheduler.shared.register(
                forTaskWithIdentifier: Self.permittedIdentifier, using: .main
            ) { task in
                MainActor.assumeIsolated {
                    guard let continuation = task as? BGContinuedProcessingTask else {
                        task.setTaskCompleted(success: false)
                        return
                    }
                    guard let owner = Self.current, owner.isRunning else {
                        // Nothing is verifying any more. Hand the time back
                        // instead of holding an assertion nobody is using.
                        continuation.setTaskCompleted(success: true)
                        return
                    }
                    owner.adopt(continuation)
                }
            }
        }

        private func adopt(_ task: BGContinuedProcessingTask) {
            self.task = task
            task.progress.totalUnitCount = max(totalUnits, 1)
            task.progress.completedUnitCount = completedUnits
            task.expirationHandler = { [weak self] in
                Task { @MainActor in self?.expire() }
            }
        }

        private func expire() {
            guard isRunning else { return }
            let handler = onExpiration
            onExpiration = nil
            // Ask the verifier to stop first: its checkpoint holds every file
            // it has finished, so the only work lost is the file in flight.
            handler?()
            task?.setTaskCompleted(success: false)
            clear()
        }

        private func clear() {
            task = nil
            submittedIdentifier = nil
            onExpiration = nil
            isRunning = false
            if Self.current === self { Self.current = nil }
        }

        private static func sentence(for error: Error) -> String {
            let code = (error as NSError).code
            guard (error as NSError).domain == BGTaskScheduler.errorDomain else {
                return String(describing: error)
            }
            switch BGTaskScheduler.Error.Code(rawValue: code) {
            case .unavailable:
                return "the system will not run background work for this app right now"
            case .notPermitted:
                return "this build is not permitted to run background verification"
            case .tooManyPendingTaskRequests:
                return "another background task request is still pending"
            default:
                return "the system could not start this verification in the background now"
            }
        }
    }

    /// iOS before continued processing existed: the ordinary background task
    /// assertion. It buys the roughly thirty seconds the system grants a
    /// backgrounded app, which is not enough to finish a terabyte but is enough
    /// to stop cleanly on a file boundary and keep the checkpoint.
    @MainActor
    final class GraceWindowVerificationWork: VerificationBackgroundWork {
        private var identifier: UIBackgroundTaskIdentifier = .invalid
        private var onExpiration: (@MainActor () -> Void)?
        private(set) var isRunning = false
        private(set) var refusal: String?

        init() {}

        @discardableResult
        func begin(
            title: String, subtitle: String, totalBytes: UInt64,
            onExpiration: @escaping @MainActor () -> Void
        ) -> Bool {
            guard !isRunning else { return true }
            self.onExpiration = onExpiration
            identifier = UIApplication.shared.beginBackgroundTask(withName: title) {
                [weak self] in
                MainActor.assumeIsolated { self?.expire() }
            }
            guard identifier != .invalid else {
                refusal = "the system declined to extend this app's background time"
                self.onExpiration = nil
                return false
            }
            refusal = nil
            isRunning = true
            return true
        }

        func update(subtitle: String, completedBytes: UInt64, totalBytes: UInt64) {}

        func end(success: Bool) {
            guard isRunning else { return }
            UIApplication.shared.endBackgroundTask(identifier)
            identifier = .invalid
            onExpiration = nil
            isRunning = false
        }

        private func expire() {
            guard isRunning else { return }
            let handler = onExpiration
            onExpiration = nil
            handler?()
            UIApplication.shared.endBackgroundTask(identifier)
            identifier = .invalid
            isRunning = false
        }
    }

#endif

enum VerificationBackgroundWorkFactory {
    /// The best continuation this OS offers. Never nil: a platform that cannot
    /// continue the work still gets an object, so the caller has one code path.
    @MainActor
    static func make() -> any VerificationBackgroundWork {
        #if os(iOS)
            if #available(iOS 26.0, *) {
                return ContinuedProcessingVerificationWork()
            }
            return GraceWindowVerificationWork()
        #else
            return UnsuspendedVerificationWork()
        #endif
    }
}
