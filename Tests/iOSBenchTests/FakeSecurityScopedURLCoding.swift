import Foundation
import XCTest

@testable import iOSBench

/// A ``SecurityScopedURLCoding`` whose every branch can be commanded.
///
/// The point is not to avoid touching the filesystem — the real coding is
/// tested against real directories in the same suite. It is that the failure
/// modes spec §4.3 wants verified (a refused access grant, a resolution that
/// throws because the volume is detached) cannot be produced on a Mac on
/// demand, and an untested error path in the bracket is exactly the kind of
/// leak that only shows up as an un-ejectable drive on the phone.
final class FakeSecurityScopedURLCoding: SecurityScopedURLCoding {
    enum Failure: Error, Equatable {
        case cannotBookmark
        case cannotResolve
    }

    /// `bookmark data -> url` for whatever has been bookmarked so far.
    private var urls: [Data: URL] = [:]
    private var nextID = 0

    // Commands.
    var failBookmarkCreation = false
    var failResolution = false
    var refuseAccess = false
    /// URLs to report stale on the next resolve.
    var staleURLs: Set<URL> = []

    // Observations.
    private(set) var startCount = 0
    private(set) var stopCount = 0
    private(set) var bookmarkCount = 0
    /// Every start/stop in order, so a test can assert nesting and not just
    /// totals.
    private(set) var accessLog: [String] = []

    var isBalanced: Bool { startCount == stopCount }

    func bookmarkData(for url: URL) throws -> Data {
        if failBookmarkCreation { throw Failure.cannotBookmark }
        bookmarkCount += 1
        nextID += 1
        let data = Data("bookmark-\(nextID)-\(url.path)".utf8)
        urls[data] = url
        return data
    }

    func resolve(_ data: Data) throws -> ResolvedBookmark {
        if failResolution { throw Failure.cannotResolve }
        guard let url = urls[data] else { throw Failure.cannotResolve }
        return ResolvedBookmark(url: url, isStale: staleURLs.contains(url))
    }

    func startAccessing(_ url: URL) -> Bool {
        if refuseAccess {
            accessLog.append("refused \(url.lastPathComponent)")
            return false
        }
        startCount += 1
        accessLog.append("start \(url.lastPathComponent)")
        return true
    }

    func stopAccessing(_ url: URL) {
        stopCount += 1
        accessLog.append("stop \(url.lastPathComponent)")
    }
}

/// A directory that deletes itself when the test ends.
struct TemporaryDirectory {
    let url: URL

    init(_ testCase: XCTestCase, name: String = "iosbench") {
        let created = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("\(name)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: created, withIntermediateDirectories: true)
        url = created
        testCase.addTeardownBlock {
            try? FileManager.default.removeItem(at: created)
        }
    }

    @discardableResult
    func write(_ bytes: Int, to relativePath: String) -> URL {
        let target = url.appendingPathComponent(relativePath)
        try? FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        _ = FileManager.default.createFile(
            atPath: target.path, contents: Data(repeating: 0x5A, count: bytes))
        return target
    }

    func writeManifest(_ object: [String: Any]) {
        let data = try! JSONSerialization.data(withJSONObject: object)
        try! data.write(to: url.appendingPathComponent("manifest.json"))
    }
}
