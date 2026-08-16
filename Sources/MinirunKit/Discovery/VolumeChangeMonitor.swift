import Foundation

#if canImport(AppKit)
    import AppKit
#endif

/// Calls back when a volume appears or disappears.
///
/// Installed-ness is a fact about a directory, and on a machine with removable
/// storage that fact changes without the app doing anything. Polling would
/// answer it late; the workspace already publishes the two events that matter,
/// so this is a subscription rather than a timer.
///
/// macOS has `NSWorkspace.didMountNotification`. iOS has no equivalent because
/// it has no user-mountable volumes in the same sense — an external drive
/// reaches an app through the document picker, and the app learns about it by
/// being handed a URL. So on iOS this type starts, observes nothing, and stops:
/// a monitor that is honest about hearing nothing is better than one that
/// invents a polling loop nobody asked for. Callers refresh on foreground and
/// on the operator's own pull instead, which they do anyway.
public final class VolumeChangeMonitor: @unchecked Sendable {
    /// Called on the main queue. The URL is the volume that changed where the
    /// system named one.
    public typealias Handler = @Sendable (URL?) -> Void

    private let lock = NSLock()
    private var tokens: [NSObjectProtocol] = []

    public init() {}

    /// True where this platform actually publishes mount events.
    public static var isSupported: Bool {
        #if canImport(AppKit)
            return true
        #else
            return false
        #endif
    }

    public func start(_ handler: @escaping Handler) {
        stop()
        #if canImport(AppKit)
            let center = NSWorkspace.shared.notificationCenter
            let names: [Notification.Name] = [
                NSWorkspace.didMountNotification,
                NSWorkspace.didUnmountNotification,
                // Fires *before* the unmount completes, which is the only point
                // at which an app still holding a scope can drop it in time.
                NSWorkspace.willUnmountNotification,
                NSWorkspace.didRenameVolumeNotification,
            ]
            let observers = names.map { name in
                center.addObserver(forName: name, object: nil, queue: .main) { note in
                    handler(note.userInfo?[NSWorkspace.volumeURLUserInfoKey] as? URL)
                }
            }
            lock.lock()
            tokens = observers
            lock.unlock()
        #endif
    }

    public func stop() {
        lock.lock()
        let observers = tokens
        tokens = []
        lock.unlock()
        #if canImport(AppKit)
            for token in observers { NSWorkspace.shared.notificationCenter.removeObserver(token) }
        #else
            _ = observers
        #endif
    }

    deinit { stop() }
}
