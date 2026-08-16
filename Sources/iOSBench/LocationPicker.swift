import Foundation
import StorageCore

#if canImport(UIKit)
    import UIKit
    import UniformTypeIdentifiers
#endif

/// Why a pick did not produce a usable location.
public enum LocationPickError: Error, CustomStringConvertible, LocalizedError, Equatable {
    case cancelled
    case notADirectory(path: String)
    /// The picker returned nothing, which is not a documented outcome.
    case noURLReturned
    case pickerUnavailable

    public var description: String {
        switch self {
        case .cancelled:
            return "the picker was cancelled"
        case .notADirectory(let path):
            return "'\(path)' is not a directory"
        case .noURLReturned:
            return "the document picker returned no URL"
        case .pickerUnavailable:
            return "no document picker on this platform"
        }
    }

    public var errorDescription: String? { description }
}

/// What happens to a URL after the user picks it — deliberately free of UIKit.
///
/// Everything interesting about "the user chose a folder" is here: it is a
/// directory, it gets bookmarked so the next launch does not ask again, and
/// the volume and artifact facts get read while the security scope is still
/// held. Keeping it out of the `canImport(UIKit)` gate is what makes the
/// on-device path testable on the development Mac, which spec §4.3's "verify,
/// do not assume" only makes affordable if most of it can be verified without
/// a device.
public struct PickedLocationHandler {
    /// Default key for the one location this harness remembers.
    public static let modelDirectoryKey = "modelDirectory"

    private let store: SecurityScopedLocationStore
    private let key: String

    public init(store: SecurityScopedLocationStore, key: String = PickedLocationHandler.modelDirectoryKey) {
        self.store = store
        self.key = key
    }

    /// Validate, bookmark, and describe a freshly picked directory.
    ///
    /// The bookmark is created *inside* the access bracket. On iOS that
    /// ordering is required, not stylistic: bookmarking a security-scoped URL
    /// outside its bracket fails, and the failure is a thrown error at pick
    /// time rather than a silent one at next launch.
    public func accept(_ url: URL) throws -> ModelLocationReport {
        try store.withAccess(to: url) { scoped in
            var isDirectory: ObjCBool = false
            let exists = FileManager.default.fileExists(
                atPath: scoped.path, isDirectory: &isDirectory)
            guard exists, isDirectory.boolValue else {
                throw LocationPickError.notADirectory(path: scoped.path)
            }
            try store.remember(scoped, forKey: key)
            return ModelLocationResolver.describe(directory: scoped)
        }
    }

    /// Describe the remembered location without asking the user again.
    ///
    /// Returns nil when nothing has been picked yet. A bookmark that no longer
    /// resolves — the usual cause being a detached volume — throws rather than
    /// returning nil, because "never picked" and "picked but the drive is
    /// gone" call for different words on screen (spec §5.8).
    public func describeRemembered() throws -> ModelLocationReport? {
        guard store.hasBookmark(forKey: key) else { return nil }
        return try store.withAccess(toKey: key) { url in
            ModelLocationResolver.describe(directory: url)
        }
    }

    public func forget() {
        store.forget(key: key)
    }
}

#if canImport(UIKit)

    /// Presents `UIDocumentPickerViewController` and turns its delegate
    /// callbacks into a single `async` result.
    ///
    /// The coordinator keeps a strong reference to itself for exactly as long
    /// as the picker is on screen. `UIDocumentPickerViewController` holds its
    /// delegate weakly, and a coordinator that is only owned by the `async`
    /// frame is deallocated the moment the continuation suspends — the picker
    /// then reports to nobody and the `await` never returns. The
    /// self-reference is released in ``finish(_:)``, which every exit path
    /// goes through.
    @MainActor
    public final class DirectoryPickerCoordinator: NSObject, UIDocumentPickerDelegate {
        private var continuation: CheckedContinuation<URL, Error>?
        private var selfReference: DirectoryPickerCoordinator?

        public override init() { super.init() }

        /// Present a folder picker from `presenter` and return the chosen URL.
        ///
        /// `asCopy: false` is the whole point: a copy would land in the app
        /// container, which for a multi-gigabyte artifact is both impossible
        /// and the opposite of what M8.5 is measuring. The consequence is that
        /// the returned URL is security-scoped, and every later read of it has
        /// to happen inside a bracket — see ``PickedLocationHandler``.
        public func pickDirectory(from presenter: UIViewController) async throws -> URL {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                self.selfReference = self

                let picker = UIDocumentPickerViewController(
                    forOpeningContentTypes: [.folder], asCopy: false)
                picker.delegate = self
                picker.allowsMultipleSelection = false
                presenter.present(picker, animated: true)
            }
        }

        private func finish(_ result: Result<URL, Error>) {
            let pending = continuation
            continuation = nil
            selfReference = nil
            pending?.resume(with: result)
        }

        public func documentPicker(
            _ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]
        ) {
            guard let url = urls.first else {
                finish(.failure(LocationPickError.noURLReturned))
                return
            }
            finish(.success(url))
        }

        public func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            finish(.failure(LocationPickError.cancelled))
        }
    }

#endif
