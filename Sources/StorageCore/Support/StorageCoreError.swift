import Foundation

/// Errors surfaced by StorageCore.
///
/// Spec 5.8 and 19.2 require explicit failure: an unreadable path, a short
/// read, or an out-of-range argument must be reported, never worked around
/// silently.
/// `Equatable` because every payload is a plain value and callers hold these in
/// state that is compared. The product app's download state machine carries one
/// in `failed(_:)`; a state machine whose failure case cannot be compared is a
/// state machine SwiftUI cannot diff, and the alternative — storing the
/// rendered `description` instead — would keep the sentence and throw away the
/// operation, path and errno behind it.
public enum StorageCoreError: Error, Equatable, CustomStringConvertible, LocalizedError {
    /// A command-line or configuration value is outside its allowed range.
    case invalidArgument(String)
    /// A POSIX call failed. Carries the operation, the path, and `errno`.
    case posix(operation: String, path: String, code: Int32)
    /// A read or write returned fewer bytes than requested without an error.
    case shortTransfer(operation: String, path: String, offset: UInt64, expected: Int, actual: Int)
    /// Environment or volume information could not be read.
    case environmentUnavailable(String)

    public var description: String {
        switch self {
        case .invalidArgument(let detail):
            return "invalid argument: \(detail)"
        case .posix(let operation, let path, let code):
            return "\(operation) failed for '\(path)': \(String(cString: strerror(code))) (errno \(code))"
        case .shortTransfer(let operation, let path, let offset, let expected, let actual):
            return "\(operation) of '\(path)' at offset \(offset) returned \(actual) of \(expected) bytes"
        case .environmentUnavailable(let detail):
            return "environment unavailable: \(detail)"
        }
    }

    public var errorDescription: String? { description }
}
