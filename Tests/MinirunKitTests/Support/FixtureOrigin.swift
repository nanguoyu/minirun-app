import Foundation
import MinirunKit

/// A stand-in for `huggingface.co`, shaped like the recorded fixtures.
///
/// It answers the five things this package asks a real origin for — the model
/// list, the tree, pinned repository-info siblings, `index.json`, and a ranged
/// `resolve` — and it reproduces the
/// two behaviours that were measured on the live API and that a naive fake
/// would get wrong:
///
/// - `resolve` answers **302**, carrying `x-linked-size`, `x-linked-etag` and
///   `x-repo-commit`. The digest lives on that response and nowhere else.
/// - the redirect target answers **206** with an `ETag` that is *not* the
///   SHA-256 — the live CDN's is a Xet hash of different bytes. Any code that
///   reaches for `ETag` as a digest fails here, loudly, instead of passing
///   forever in production.
final class FixtureOrigin: @unchecked Sendable {
    private let lock = NSLock()
    private var contents: [String: Data] = [:]
    private var revision: String
    /// What `x-repo-commit` claims, which a test can move under a transfer.
    private var servedRevision: String?
    /// Paths whose bytes are served corrupted, to drive a digest failure.
    private var corrupted: Set<String> = []
    /// Statuses to answer with once each, oldest first, before serving bytes.
    private var injectedFailures: [String: [Int]] = [:]
    private var rangesRefused = false
    private var requestLog: [String] = []
    /// Seconds to dawdle before answering. A localhost transfer of three
    /// megabytes is over in milliseconds, which leaves no window in which to
    /// observe a pause; this makes the window real rather than making the test
    /// race for it.
    private var responseDelay: TimeInterval = 0

    let repoID: String
    private(set) var server: LocalHTTPServer!

    private init(repoID: String, revision: String, files: [String: Data]) {
        self.repoID = repoID
        self.revision = revision
        self.contents = files
    }

    /// Two steps, because the handler needs the origin and the origin needs the
    /// server: the socket is opened only once `self` exists to answer on it.
    static func start(
        repoID: String = "nanguoyu/Fixture-minirun", revision: String, files: [String: Data]
    ) throws -> FixtureOrigin {
        let origin = FixtureOrigin(repoID: repoID, revision: revision, files: files)
        origin.server = try LocalHTTPServer { [weak origin] request in
            origin?.respond(to: request)
                ?? LocalHTTPServer.Response(status: 500, body: Data("origin is gone\n".utf8))
        }
        return origin
    }

    var baseURL: URL { server.baseURL }
    var repo: HuggingFaceRepoRef { HuggingFaceRepoRef(repoID: repoID, revision: revision) }

    // MARK: - Knobs

    func corrupt(_ path: String) {
        lock.lock()
        corrupted.insert(path)
        lock.unlock()
    }

    func heal(_ path: String) {
        lock.lock()
        corrupted.remove(path)
        lock.unlock()
    }

    func serveRevision(_ revision: String?) {
        lock.lock()
        servedRevision = revision
        lock.unlock()
    }

    func delayResponses(by seconds: TimeInterval) {
        lock.lock()
        responseDelay = seconds
        lock.unlock()
    }

    func refuseRanges(_ refuse: Bool) {
        lock.lock()
        rangesRefused = refuse
        lock.unlock()
    }

    /// Answer the next `statuses.count` requests for `path` with these statuses
    /// before serving it normally.
    func failNext(_ path: String, with statuses: [Int]) {
        lock.lock()
        injectedFailures[path, default: []].append(contentsOf: statuses)
        lock.unlock()
    }

    var requests: [String] {
        lock.lock()
        defer { lock.unlock() }
        return requestLog
    }

    func stop() { server.stop() }

    // MARK: - Facts a test asserts against

    func bytes(of path: String) -> Data? {
        lock.lock()
        defer { lock.unlock() }
        return contents[path]
    }

    var paths: [String] {
        lock.lock()
        defer { lock.unlock() }
        return contents.keys.sorted()
    }

    // MARK: - Routing

    private func respond(to request: LocalHTTPServer.Request) -> LocalHTTPServer.Response {
        lock.lock()
        requestLog.append(request.path)
        let files = contents
        let revision = self.revision
        let served = servedRevision ?? revision
        let corrupted = self.corrupted
        let refuseRanges = rangesRefused
        var pending = injectedFailures[request.path] ?? []
        var injected: Int?
        if !pending.isEmpty {
            injected = pending.removeFirst()
            injectedFailures[request.path] = pending
        }
        let delay = responseDelay
        lock.unlock()

        if delay > 0 { Thread.sleep(forTimeInterval: delay) }

        if let injected {
            return LocalHTTPServer.Response(status: injected, body: Data("injected\n".utf8))
        }

        let path = request.path

        if path == "/api/models" {
            return .json(
                [
                    ["id": repoID, "sha": revision],
                    ["id": "nanguoyu/unrelated-mlx", "sha": String(repeating: "b", count: 40)],
                ])
        }

        if path == "/api/models/\(repoID)/tree/\(revision)" {
            return treePage(files: files, page: Int(request.query["page"] ?? "1") ?? 1)
        }

        if path == "/api/models/\(repoID)/revision/\(revision)" {
            return .json([
                "id": repoID,
                "sha": revision,
                "siblings": files.keys.sorted().map { ["rfilename": $0] },
            ])
        }

        if path.hasPrefix("/\(repoID)/resolve/\(revision)/") {
            let file = String(path.dropFirst("/\(repoID)/resolve/\(revision)/".count))
            guard let body = files[file] else {
                return LocalHTTPServer.Response(status: 404, body: Data("no such file\n".utf8))
            }
            // The origin's 302 is where the digest lives.
            return LocalHTTPServer.Response(
                status: 302,
                headers: [
                    "location": "/cdn/\(file)",
                    "x-linked-size": "\(body.count)",
                    "x-linked-etag": "\"\(FileDigestComputer.sha256(of: body))\"",
                    "x-repo-commit": served,
                    "accept-ranges": "bytes",
                ],
                body: Data("redirecting\n".utf8))
        }

        if path.hasPrefix("/cdn/") {
            let file = String(path.dropFirst("/cdn/".count))
            guard var body = files[file] else {
                return LocalHTTPServer.Response(status: 404, body: Data("no such file\n".utf8))
            }
            let honest = body
            if corrupted.contains(file) {
                body = Data(body.map { $0 ^ 0xFF })
            }
            // The CDN's ETag is a hash of *something else*. Nothing may use it.
            let headers = [
                "etag": "\"\(FileDigestComputer.sha256(of: Data(honest.reversed())))\"",
                "accept-ranges": "bytes",
                "x-repo-commit": served,
            ]
            guard let range = request.range, !refuseRanges else {
                return LocalHTTPServer.Response(status: 200, headers: headers, body: body)
            }
            let last = min(range.last ?? UInt64(body.count - 1), UInt64(body.count - 1))
            guard range.first <= last else {
                return LocalHTTPServer.Response(status: 416, headers: headers)
            }
            let slice = body.subdata(in: Int(range.first)..<Int(last) + 1)
            var partial = headers
            partial["content-range"] = "bytes \(range.first)-\(last)/\(body.count)"
            return LocalHTTPServer.Response(status: 206, headers: partial, body: slice)
        }

        return LocalHTTPServer.Response(status: 404, body: Data("no route\n".utf8))
    }

    /// The tree, split over two pages so the `Link: …; rel="next"` walk is
    /// exercised. The live API paginates at 50 entries and a K3 tree needs 12
    /// pages, so a client that reads one page and stops under-counts by an
    /// order of magnitude while looking entirely healthy.
    private func treePage(files: [String: Data], page: Int) -> LocalHTTPServer.Response {
        let sorted = files.keys.sorted()
        let half = (sorted.count + 1) / 2
        let slice = page == 1 ? Array(sorted.prefix(half)) : Array(sorted.dropFirst(half))
        let entries: [[String: Any]] = slice.map { path in
            let body = files[path]!
            // `.json` and `.txt` are stored as ordinary git blobs, as H3 stores
            // 16 of its payload files; everything else is LFS-backed.
            if path.hasSuffix(".json") || path.hasSuffix(".txt") {
                return [
                    "type": "file", "path": path, "size": body.count,
                    "oid": FileDigestComputer.gitBlobSHA1(of: body),
                ]
            }
            return [
                "type": "file", "path": path, "size": body.count,
                "oid": String(repeating: "0", count: 40),
                "lfs": ["oid": FileDigestComputer.sha256(of: body), "size": body.count],
            ]
        }
        var headers: [String: String] = [:]
        if page == 1 {
            headers["link"] =
                "<\(baseURL.absoluteString)/api/models/\(repoID)/tree/\(revision)?recursive=1&expand=1&page=2>; rel=\"next\""
        }
        return .json(entries + [["type": "directory", "path": "ignored-directory"]], headers: headers)
    }
}
