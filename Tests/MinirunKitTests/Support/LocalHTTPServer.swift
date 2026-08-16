import Darwin
import Foundation

/// A single-purpose HTTP/1.1 server on 127.0.0.1, for driving the *real*
/// ``URLSessionTransport`` in tests.
///
/// A fake transport can prove the download manager's logic, and one is used for
/// that. It cannot prove the transport itself: that redirects are followed with
/// the `Range` header intact, that the origin's `x-linked-etag` survives the hop
/// to the CDN, that a `206` is parsed, that header names arrive in whatever case
/// the server chose. Those are exactly the places where a wrong assumption
/// produces a green check over wrong bytes, so they get a real socket.
///
/// Deliberately minimal: one request per connection, `Connection: close`, no
/// keep-alive, no chunked encoding. Ordinary fixtures are a few megabytes; the
/// streaming benchmark repeats one small buffer without allocating its whole
/// response.
final class LocalHTTPServer: @unchecked Sendable {

    struct Request {
        var method: String
        var path: String
        var query: [String: String]
        var headers: [String: String]
        /// Parsed `Range: bytes=first-last`. `last` is nil for an open range.
        var range: (first: UInt64, last: UInt64?)?
    }

    struct Response {
        struct RepeatedBody {
            let chunk: Data
            let repetitions: Int

            var byteCount: Int {
                let (count, overflow) = chunk.count.multipliedReportingOverflow(
                    by: repetitions)
                precondition(!overflow, "test response length must be representable")
                return count
            }
        }

        var status: Int
        var headers: [String: String]
        var body: Data
        var repeatedBody: RepeatedBody?

        init(status: Int, headers: [String: String] = [:], body: Data = Data()) {
            self.status = status
            self.headers = headers
            self.body = body
            repeatedBody = nil
        }

        static func repeating(
            status: Int, headers: [String: String] = [:],
            chunk: Data, repetitions: Int
        ) -> Response {
            precondition(!chunk.isEmpty, "streamed test response needs a body chunk")
            precondition(repetitions >= 0, "streamed repetition count cannot be negative")
            var response = Response(status: status, headers: headers)
            response.repeatedBody = RepeatedBody(
                chunk: chunk, repetitions: repetitions)
            return response
        }

        var bodyByteCount: Int {
            repeatedBody?.byteCount ?? body.count
        }

        static func json(_ object: Any, headers: [String: String] = [:]) -> Response {
            let data = (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
            var all = headers
            all["content-type"] = "application/json"
            return Response(status: 200, headers: all, body: data)
        }
    }

    private let listenerFD: Int32
    private let handler: @Sendable (Request) -> Response
    /// Concurrent, and that is not a performance choice. The accept loop never
    /// returns, so on a serial queue it would occupy the only worker and no
    /// connection would ever be served — every request would simply time out.
    private let queue = DispatchQueue(
        label: "minirun.test.http", qos: .userInitiated, attributes: .concurrent)
    private let lock = NSLock()
    private var stopped = false
    private(set) var port: UInt16 = 0

    var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

    init(handler: @escaping @Sendable (Request) -> Response) throws {
        self.handler = handler

        listenerFD = socket(AF_INET, SOCK_STREAM, 0)
        guard listenerFD >= 0 else { throw Failure("socket() failed: \(errno)") }
        var reuse: Int32 = 1
        setsockopt(
            listenerFD, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0  // an ephemeral port, so parallel tests never collide
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(listenerFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else {
            close(listenerFD)
            throw Failure("bind() failed: \(errno)")
        }
        guard listen(listenerFD, 16) == 0 else {
            close(listenerFD)
            throw Failure("listen() failed: \(errno)")
        }

        var bound_address = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &bound_address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(listenerFD, $0, &length)
            }
        }
        port = UInt16(bigEndian: bound_address.sin_port)

        queue.async { [weak self] in self?.acceptLoop() }
    }

    func stop() {
        lock.lock()
        let alreadyStopped = stopped
        stopped = true
        lock.unlock()
        if !alreadyStopped { close(listenerFD) }
    }

    deinit { stop() }

    private var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }

    private func acceptLoop() {
        while !isStopped {
            let connection = accept(listenerFD, nil, nil)
            if connection < 0 {
                if isStopped { return }
                continue
            }
            let fd = connection
            // Writing to a socket the client already closed raises SIGPIPE,
            // which kills the whole test process rather than failing one call.
            var noSignal: Int32 = 1
            setsockopt(
                fd, SOL_SOCKET, SO_NOSIGPIPE, &noSignal, socklen_t(MemoryLayout<Int32>.size))
            queue.async { [weak self] in
                self?.serve(fd)
                // Half-close and let the client read the tail before the
                // socket goes away; a bare `close` with data still queued can
                // send an RST and truncate a megabyte-long body.
                shutdown(fd, SHUT_WR)
                var drain = [UInt8](repeating: 0, count: 1024)
                while recv(fd, &drain, drain.count, 0) > 0 {}
                close(fd)
            }
        }
    }

    private func serve(_ fd: Int32) {
        guard let head = readHead(fd) else { return }
        let request = LocalHTTPServer.parse(head)
        let response = handler(request)

        var out = "HTTP/1.1 \(response.status) \(LocalHTTPServer.reason(response.status))\r\n"
        var headers = response.headers
        if headers.keys.allSatisfy({ $0.lowercased() != "content-length" }) {
            headers["content-length"] = "\(response.bodyByteCount)"
        }
        headers["connection"] = "close"
        for (name, value) in headers.sorted(by: { $0.key < $1.key }) {
            out += "\(name): \(value)\r\n"
        }
        out += "\r\n"

        guard write(Data(out.utf8), to: fd) else { return }
        if let repeated = response.repeatedBody {
            for _ in 0..<repeated.repetitions {
                guard write(repeated.chunk, to: fd) else { return }
            }
        } else {
            // A HEAD-like 302 still gets its body written; the body is the tiny
            // human-readable note the real origin sends, and writing it keeps
            // content-length honest.
            _ = write(response.body, to: fd)
        }
    }

    private func write(_ data: Data, to fd: Int32) -> Bool {
        data.withUnsafeBytes { buffer in
            var sent = 0
            while sent < buffer.count {
                let written = send(
                    fd, buffer.baseAddress!.advanced(by: sent), buffer.count - sent, 0)
                if written > 0 {
                    sent += written
                    continue
                }
                // A short write is normal on a socket; only a real error ends
                // the response, and a truncated response is exactly the sort of
                // flake that would look like a downloader bug.
                if written < 0 && (errno == EINTR || errno == EAGAIN) { continue }
                return false
            }
            return true
        }
    }

    /// Read up to the end of the headers. Bodies are never sent to this server.
    private func readHead(_ fd: Int32) -> String? {
        var accumulated = Data()
        var chunk = [UInt8](repeating: 0, count: 4096)
        while accumulated.range(of: Data("\r\n\r\n".utf8)) == nil {
            let read = recv(fd, &chunk, chunk.count, 0)
            if read <= 0 { break }
            accumulated.append(contentsOf: chunk[0..<read])
            if accumulated.count > 64 * 1024 { break }
        }
        guard !accumulated.isEmpty else { return nil }
        return String(decoding: accumulated, as: UTF8.self)
    }

    static func parse(_ head: String) -> Request {
        let lines = head.components(separatedBy: "\r\n")
        let parts = (lines.first ?? "").split(separator: " ")
        let method = parts.count > 0 ? String(parts[0]) : "GET"
        let target = parts.count > 1 ? String(parts[1]) : "/"

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[line.startIndex..<colon].lowercased()] =
                line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }

        let components = URLComponents(string: "http://127.0.0.1" + target)
        var query: [String: String] = [:]
        for item in components?.queryItems ?? [] { query[item.name] = item.value ?? "" }

        var range: (first: UInt64, last: UInt64?)?
        if let raw = headers["range"], raw.hasPrefix("bytes=") {
            let spec = raw.dropFirst("bytes=".count)
            let bounds = spec.split(separator: "-", omittingEmptySubsequences: false)
            if bounds.count == 2, let first = UInt64(bounds[0]) {
                range = (first, UInt64(bounds[1]))
            }
        }

        return Request(
            method: method, path: components?.percentEncodedPath ?? target, query: query,
            headers: headers, range: range)
    }

    static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 206: return "Partial Content"
        case 302: return "Found"
        case 404: return "Not Found"
        case 416: return "Range Not Satisfiable"
        case 500: return "Internal Server Error"
        case 503: return "Service Unavailable"
        default: return "Status"
        }
    }

    struct Failure: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
}
