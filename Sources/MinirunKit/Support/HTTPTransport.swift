import Darwin
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// The Hugging Face origin. A constant rather than a literal sprinkled through
/// the code, because every test that must not touch the network works by
/// pointing this somewhere else.
public enum HuggingFaceEndpoint {
    public static let `default` = URL(string: "https://huggingface.co")!
}

/// One HTTP request, reduced to the four things this package ever varies.
///
/// A value type and not a `URLRequest` on purpose: the catalog and the download
/// manager share this seam, so a fake transport drives both, and neither can
/// reach a `URLSession` behaviour that a fake cannot reproduce.
public struct HTTPRequest: Sendable, Equatable {
    public var url: URL
    public var headers: [String: String]
    /// Inclusive on both ends, as `Range: bytes=first-last` is.
    public var byteRange: ClosedRange<UInt64>?
    public var timeout: TimeInterval

    public init(
        url: URL, headers: [String: String] = [:],
        byteRange: ClosedRange<UInt64>? = nil, timeout: TimeInterval = 60
    ) {
        self.url = url
        self.headers = headers
        self.byteRange = byteRange
        self.timeout = timeout
    }

    /// The wire form of `byteRange`, or nil when the whole object was asked for.
    public var rangeHeaderValue: String? {
        byteRange.map { "bytes=\($0.lowerBound)-\($0.upperBound)" }
    }
}

/// One HTTP response, with the header names already lowercased.
///
/// Lowercasing is not tidying. HTTP header names are case-insensitive, the CDN
/// and the origin disagree about capitalisation on the very headers this
/// package reads, and a digest check that silently found no `x-linked-etag`
/// because it looked for `X-Linked-Etag` would fall through to "no digest
/// available" — a green path over unverified bytes.
public struct HTTPResponse: Sendable, Equatable {
    public let statusCode: Int
    public let headers: [String: String]
    public let body: Data?

    public init(statusCode: Int, headers: [String: String], body: Data?) {
        self.statusCode = statusCode
        self.headers = headers.reduce(into: [:]) { $0[$1.key.lowercased()] = $1.value }
        self.body = body
    }

    public func header(_ name: String) -> String? { headers[name.lowercased()] }

    /// `Content-Range: bytes 0-1023/4096` → `(0, 1023, 4096)`.
    public var contentRange: (start: UInt64, end: UInt64, total: UInt64)? {
        guard let raw = header("content-range") else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("bytes ") else { return nil }
        let spec = trimmed.dropFirst("bytes ".count)
        let halves = spec.split(separator: "/", maxSplits: 1, omittingEmptySubsequences: false)
        guard halves.count == 2 else { return nil }
        let bounds = halves[0].split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        guard bounds.count == 2,
            let start = UInt64(bounds[0]), let end = UInt64(bounds[1]),
            let total = UInt64(halves[1])
        else { return nil }
        return (start, end, total)
    }

    /// `x-linked-size` — the size of the LFS object behind a pointer, which is
    /// the size the file will have on disk. `content-length` on the origin's
    /// 302 is the size of the redirect body, not of the file.
    public var linkedSizeBytes: UInt64? { header("x-linked-size").flatMap(UInt64.init) }

    /// `x-linked-etag`, unquoted — the SHA-256 of the LFS object.
    ///
    /// Deliberately not `etag`. The CDN's `ETag` on the 200/206 is the Xet
    /// content hash, a different function of different bytes; measured on
    /// `layer01/layer01-w1.mxfp4tile` it reads `4aed0742…` where the SHA-256 is
    /// `ad7cd23e…`. Comparing a downloaded file's SHA-256 against it fails
    /// forever; comparing the *CDN's* hash against itself passes forever, which
    /// is the failure mode worth naming: a green check over wrong bytes.
    public var linkedSHA256: String? {
        guard let raw = header("x-linked-etag") else { return nil }
        return HTTPResponse.unquote(raw).lowercased()
    }

    /// `x-repo-commit` — the commit the origin actually served. Compared against
    /// the pinned revision on every response.
    public var repoCommit: String? { header("x-repo-commit").map(HTTPResponse.unquote) }

    public enum NextPage: Sendable, Equatable {
        case absent
        case url(URL)
        case malformed(String)
    }

    /// Parse `Link: <…>; rel="next"` without conflating an absent continuation
    /// with a continuation the response declared but encoded unsafely.
    public var nextPage: NextPage {
        guard let link = header("link") else { return .absent }
        for part in link.split(separator: ",") {
            let segments = part.split(separator: ";")
            guard segments.count >= 2 else { continue }
            let relations = segments.dropFirst().map {
                $0.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "\"", with: "")
            }
            guard relations.contains("rel=next") else { continue }
            let target = segments[0].trimmingCharacters(in: .whitespaces)
            guard target.hasPrefix("<"), target.hasSuffix(">") else {
                return .malformed(String(part))
            }
            let raw = String(target.dropFirst().dropLast())
            guard let url = URL(string: raw), url.scheme != nil, url.host != nil else {
                return .malformed(raw)
            }
            return .url(url)
        }
        return .absent
    }

    /// Compatibility projection for callers that do not consume pagination.
    public var nextPageURL: URL? {
        if case .url(let url) = nextPage { return url }
        return nil
    }

    static func unquote(_ value: String) -> String {
        var text = value.trimmingCharacters(in: .whitespaces)
        if text.hasPrefix("W/") { text = String(text.dropFirst(2)) }
        if text.hasPrefix("\""), text.hasSuffix("\""), text.count >= 2 {
            text = String(text.dropFirst().dropLast())
        }
        return text
    }
}

/// The one place this package talks to the network.
public protocol HTTPTransport: Sendable {
    /// Fetch a response into memory. For metadata only — an artifact file goes
    /// through ``download(_:into:)``, which never holds a whole
    /// file.
    func data(for request: HTTPRequest) async throws -> HTTPResponse

    /// Stream an artifact response into a caller-owned, already-open sink.
    ///
    /// The transport receives neither a path nor a descriptor. It can admit a
    /// response and offer body chunks, but it cannot reopen, replace, seek or
    /// close the artifact leaf. The sink also refuses body bytes until the
    /// response has been bound to the exact request.
    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
}

public enum HTTPTransportError: Error, Equatable, CustomStringConvertible {
    case notHTTP(url: String)
    case cannotOpenFile(path: String, reason: String)
    case writeFailed(path: String, reason: String)
    case invalidSink(reason: String)
    /// The final response was observed but cannot supply the requested bytes.
    /// Carrying the response preserves provenance/error priority without
    /// allowing any of its rejected body into the sink.
    case responseRejected(response: HTTPResponse, reason: String)
    case responseBodyLength(expected: UInt64, received: UInt64)
    /// Rejected bytes could not be removed from the borrowed leaf. The manager
    /// must discard/taint that leaf and must not resume from its visible size.
    case discardFailed(reason: String)
    case cancelled

    public var description: String {
        switch self {
        case .notHTTP(let url): return "'\(url)' did not answer with an HTTP response"
        case .cannotOpenFile(let path, let reason): return "cannot open '\(path)': \(reason)"
        case .writeFailed(let path, let reason): return "write to '\(path)' failed: \(reason)"
        case .invalidSink(let reason): return "download sink is unavailable: \(reason)"
        case .responseRejected(let response, let reason):
            return "HTTP \(response.statusCode) response cannot supply this download: \(reason)"
        case .responseBodyLength(let expected, let received):
            return "response body is \(received) bytes; exactly \(expected) were required"
        case .discardFailed(let reason):
            return "rejected response bytes could not be discarded: \(reason)"
        case .cancelled: return "cancelled"
        }
    }
}

/// A capability to write one HTTP body into one already-open artifact leaf.
///
/// `DownloadManager` constructs this only after opening the part file through
/// its device/inode-bound directory walk. The descriptor is borrowed: this
/// type never closes it, and the transport never sees it. Writes use explicit
/// offsets, are bounded to one response, and are refused until the response's
/// status, range and length have been admitted.
public final class HTTPDownloadSink: @unchecked Sendable {
    private enum State: Equatable {
        case awaitingResponse
        case admitted
        case sealed
        case aborted
    }

    /// The response facts that authorize the bytes offered under one body.
    /// Shape validation alone is insufficient: an untrusted transport could
    /// otherwise admit bytes under one repository commit and return a second,
    /// identically sized response for the manager to inspect.
    private struct AdmissionIdentity: Equatable {
        let statusCode: Int
        let contentRange: String?
        let contentLength: String?
        let contentEncoding: String?
        let repoCommit: String?
        let linkedSHA256: String?
        let linkedSize: String?

        init(_ response: HTTPResponse) {
            statusCode = response.statusCode
            contentRange = Self.normalized(response.header("content-range"))
            contentLength = Self.normalized(response.header("content-length"))
            contentEncoding = Self.normalized(response.header("content-encoding"))?.lowercased()
            repoCommit = Self.normalized(response.header("x-repo-commit"))
            linkedSHA256 = Self.normalized(response.header("x-linked-etag"))
            linkedSize = Self.normalized(response.header("x-linked-size"))
        }

        private static func normalized(_ value: String?) -> String? {
            value?.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private let descriptor: Int32
    private let request: HTTPRequest
    private let expectedObjectLength: UInt64
    private let writeOffset: UInt64
    private let expectedBodyLength: UInt64
    private let onBytes: @Sendable (UInt64) -> Void
    private let cancellationRequested: @Sendable () -> Bool
    private let truncateOperation: @Sendable (Int32, off_t) -> Int32
    private let lock = NSLock()
    private var state: State = .awaitingResponse
    private var written: UInt64 = 0
    private var received: UInt64 = 0
    private var wasted: UInt64 = 0
    private var discardedWholeResponse = false
    private var terminalFailure: HTTPTransportError?
    private var admittedIdentity: AdmissionIdentity?

    /// Internal because only the owner of an already-authorized descriptor can
    /// mint this capability. External transports consume it through the public
    /// response/body methods below; they cannot manufacture filesystem access.
    init(
        borrowing descriptor: Int32, request: HTTPRequest,
        expectedObjectLength: UInt64,
        onBytes: @escaping @Sendable (UInt64) -> Void,
        cancellationRequested: @escaping @Sendable () -> Bool = { false },
        truncateOperation: @escaping @Sendable (Int32, off_t) -> Int32 = {
            Darwin.ftruncate($0, $1)
        }
    ) throws {
        guard descriptor >= 0 else {
            throw HTTPTransportError.invalidSink(reason: "the borrowed descriptor is invalid")
        }
        let offset: UInt64
        let bodyLength: UInt64
        if let range = request.byteRange {
            offset = range.lowerBound
            let (distance, underflow) = range.upperBound.subtractingReportingOverflow(
                range.lowerBound)
            let (length, overflow) = distance.addingReportingOverflow(1)
            guard !underflow, !overflow, range.upperBound < expectedObjectLength else {
                throw HTTPTransportError.invalidSink(
                    reason: "the requested byte range is outside the expected object")
            }
            bodyLength = length
        } else {
            offset = 0
            bodyLength = expectedObjectLength
        }
        let (endOffset, overflow) = offset.addingReportingOverflow(bodyLength)
        guard !overflow, endOffset <= UInt64(Int64.max) else {
            throw HTTPTransportError.invalidSink(
                reason: "the response offset is not representable by the filesystem API")
        }

        var metadata = stat()
        guard Darwin.fstat(descriptor, &metadata) == 0 else {
            throw HTTPTransportError.invalidSink(reason: String(cString: strerror(errno)))
        }
        guard metadata.st_mode & S_IFMT == S_IFREG, metadata.st_size >= 0 else {
            throw HTTPTransportError.invalidSink(reason: "the borrowed leaf is not a regular file")
        }
        guard UInt64(metadata.st_size) == offset else {
            throw HTTPTransportError.invalidSink(
                reason:
                    "the borrowed leaf is \(metadata.st_size) bytes; the response starts at \(offset)")
        }

        self.descriptor = descriptor
        self.request = request
        self.expectedObjectLength = expectedObjectLength
        self.writeOffset = offset
        self.expectedBodyLength = bodyLength
        self.onBytes = onBytes
        self.cancellationRequested = cancellationRequested
        self.truncateOperation = truncateOperation
    }

    /// Admit the final (post-redirect) response before its first body byte.
    /// Repeating the same admission is harmless; changing it is refused.
    public func admit(_ response: HTTPResponse) throws {
        lock.lock()
        defer { lock.unlock() }
        if let terminalFailure { throw terminalFailure }
        try refuseIfCancelledLocked()
        switch state {
        case .awaitingResponse:
            break
        case .admitted:
            do {
                try validate(response)
                try requireOriginalAdmission(response)
                return
            } catch let failure as HTTPTransportError {
                throw discardCurrentResponseLocked(andFailWith: failure)
            }
        case .sealed:
            do {
                try validate(response)
                try requireOriginalAdmission(response)
                return
            } catch let failure as HTTPTransportError {
                throw discardCurrentResponseLocked(andFailWith: failure)
            }
        case .aborted:
            throw failLocked(.invalidSink(reason: "the response was aborted"))
        }

        do {
            try validate(response)
            admittedIdentity = AdmissionIdentity(response)
            state = .admitted
        } catch let failure as HTTPTransportError {
            throw discardCurrentResponseLocked(andFailWith: failure)
        }
    }

    /// Write one body fragment at the next checked offset.
    public func write(_ data: Data) throws {
        var credited: UInt64 = 0
        let receivedNow = UInt64(data.count)
        var thrown: Error?

        lock.lock()
        // A delegate callback means these bytes have already arrived from the
        // wire. Account them before inspecting terminal state: cancellation
        // and a retained, already-aborted capability must not make late
        // network traffic disappear.
        received = Self.saturatingAdd(received, receivedNow)
        if let terminalFailure {
            wasted = Self.saturatingAdd(wasted, receivedNow)
            thrown = terminalFailure
        } else {
            do {
                switch state {
                case .admitted:
                    break
                case .sealed:
                    // Extra body data invalidates the response even when the
                    // exact advertised length was sealed first. Roll back now,
                    // while the descriptor is still borrowed by this sink.
                    throw discardCurrentResponseLocked(
                        andFailWith: .responseBodyLength(
                            expected: expectedBodyLength, received: received))
                case .awaitingResponse:
                    throw HTTPTransportError.invalidSink(
                        reason: "body bytes arrived before a response was admitted")
                case .aborted:
                    throw HTTPTransportError.invalidSink(
                        reason: "body bytes arrived after the response was aborted")
                }
                try refuseIfCancelledLocked()
                let remaining = expectedBodyLength - written
                guard receivedNow <= remaining else {
                    // A protocol-invalid response is not a resumable prefix.
                    // Roll back only this request; bytes from earlier, sealed
                    // ranges remain the resume authority.
                    let (received, overflow) = written.addingReportingOverflow(receivedNow)
                    let failure = HTTPTransportError.responseBodyLength(
                        expected: expectedBodyLength, received: overflow ? UInt64.max : received)
                    throw discardCurrentResponseLocked(andFailWith: failure)
                }

                try data.withUnsafeBytes { raw in
                    guard let base = raw.baseAddress else { return }
                    var cursor = 0
                    while cursor < raw.count {
                        try refuseIfCancelledLocked()
                        let logicalOffset = writeOffset + written + UInt64(cursor)
                        let result = Darwin.pwrite(
                            descriptor, base.advanced(by: cursor), raw.count - cursor,
                            off_t(logicalOffset))
                        if result > 0 {
                            cursor += result
                            credited += UInt64(result)
                            continue
                        }
                        if result < 0, errno == EINTR { continue }
                        let reason =
                            result == 0 ? "pwrite returned zero" : String(cString: strerror(errno))
                        throw HTTPTransportError.writeFailed(
                            path: "<borrowed artifact leaf>", reason: reason)
                    }
                }
                written += credited
            } catch let failure as HTTPTransportError {
                if credited > 0 { written += credited }
                if !discardedWholeResponse, receivedNow > credited {
                    wasted = Self.saturatingAdd(wasted, receivedNow - credited)
                }
                thrown = failLocked(failure)
            } catch {
                if credited > 0 { written += credited }
                if !discardedWholeResponse, receivedNow > credited {
                    wasted = Self.saturatingAdd(wasted, receivedNow - credited)
                }
                let failure = HTTPTransportError.writeFailed(
                    path: "<borrowed artifact leaf>", reason: String(describing: error))
                thrown = failLocked(failure)
            }
        }
        lock.unlock()

        if receivedNow > 0 { onBytes(receivedNow) }
        if let thrown { throw thrown }
    }

    /// Close the response capability without closing the borrowed descriptor.
    /// The file must now be exactly the offset plus the admitted body.
    public func seal() throws {
        lock.lock()
        defer { lock.unlock() }
        if let terminalFailure { throw terminalFailure }
        try refuseIfCancelledLocked()
        if state == .sealed {
            try verifyExactFileSizeLocked()
            return
        }
        guard state == .admitted else {
            throw failLocked(.invalidSink(reason: "no successful response was admitted"))
        }
        guard written == expectedBodyLength else {
            throw failLocked(
                .responseBodyLength(expected: expectedBodyLength, received: written))
        }
        try verifyExactFileSizeLocked()
        state = .sealed
    }

    private func verifyExactFileSizeLocked() throws {
        var metadata = stat()
        guard Darwin.fstat(descriptor, &metadata) == 0, metadata.st_size >= 0 else {
            throw failLocked(.invalidSink(reason: String(cString: strerror(errno))))
        }
        let expectedSize = writeOffset + expectedBodyLength
        guard UInt64(metadata.st_size) == expectedSize else {
            throw failLocked(
                .invalidSink(
                    reason:
                        "the borrowed leaf is \(metadata.st_size) bytes after a \(expectedSize)-byte seal"))
        }
    }

    /// Revoke a retained capability before its owner closes the descriptor.
    /// Partial bytes remain resumable; cancellation policy decides later
    /// whether to unlink them.
    public func abort() {
        lock.lock()
        // Sealing proves the response length; it does not extend the lifetime
        // of this borrowed capability. The owner may close and later reuse the
        // descriptor number as soon as it returns, so even a sealed sink must
        // become inert before that point.
        state = .aborted
        lock.unlock()
    }

    var isSealed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return state == .sealed
    }

    /// Network bytes that this response made unusable. The manager reads this
    /// once after revoking the capability and adds it to the job's waste
    /// counter. Keeping the value in the sink makes rollback, cancellation and
    /// partial-write accounting agree on one serialized truth.
    var discardedBodyBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return wasted
    }

    /// Manager-only fail-closed cleanup for a transport that returns after
    /// violating the sink protocol. Cancellation intentionally uses `abort()`
    /// instead so its successfully landed prefix remains resumable.
    @discardableResult
    func discardCurrentResponse(
        because failure: HTTPTransportError
    ) -> HTTPTransportError {
        lock.lock()
        defer { lock.unlock() }
        return discardCurrentResponseLocked(andFailWith: failure)
    }

    private func validate(_ response: HTTPResponse) throws {
        if let range = request.byteRange {
            guard response.statusCode == 206 else {
                throw HTTPTransportError.responseRejected(
                    response: response,
                    reason: "a byte-range request requires 206 Partial Content")
            }
            guard let contentRange = response.contentRange,
                contentRange.start == range.lowerBound,
                contentRange.end == range.upperBound,
                contentRange.total == expectedObjectLength,
                contentRange.start <= contentRange.end,
                contentRange.end < contentRange.total
            else {
                throw HTTPTransportError.responseRejected(
                    response: response,
                    reason: "Content-Range does not exactly match the requested object range")
            }
        } else {
            guard response.statusCode == 200 else {
                throw HTTPTransportError.responseRejected(
                    response: response,
                    reason: "a complete-object request requires 200 OK")
            }
        }

        guard let rawLength = response.header("content-length")?.trimmingCharacters(
            in: .whitespacesAndNewlines),
            !rawLength.isEmpty, rawLength.utf8.allSatisfy({ (48...57).contains($0) }),
            let contentLength = UInt64(rawLength), contentLength == expectedBodyLength
        else {
            throw HTTPTransportError.responseRejected(
                response: response,
                reason: "Content-Length must equal \(expectedBodyLength)")
        }
        if let encoding = response.header("content-encoding")?.trimmingCharacters(
            in: .whitespacesAndNewlines),
            !encoding.isEmpty, encoding.lowercased() != "identity"
        {
            throw HTTPTransportError.responseRejected(
                response: response,
                reason: "Content-Encoding '\(encoding)' is not byte-preserving identity")
        }
    }

    private func requireOriginalAdmission(_ response: HTTPResponse) throws {
        guard let admittedIdentity,
            AdmissionIdentity(response) == admittedIdentity
        else {
            throw HTTPTransportError.responseRejected(
                response: response,
                reason: "response authority changed after body admission")
        }
    }

    private func refuseIfCancelledLocked() throws {
        if cancellationRequested() {
            throw failLocked(.cancelled)
        }
    }

    /// A response that fails admission cannot contribute a resumable prefix.
    /// Truncating back to this request's offset preserves ranges that a prior
    /// request already sealed while discarding every byte supplied under the
    /// rejected response. This also closes a custom-transport trick where it
    /// admits one response, writes it, then returns a different response.
    @discardableResult
    private func discardCurrentResponseLocked(
        andFailWith failure: HTTPTransportError
    ) -> HTTPTransportError {
        discardedWholeResponse = true
        wasted = received
        var truncated: Int32
        repeat {
            truncated = truncateOperation(descriptor, off_t(writeOffset))
        } while truncated != 0 && errno == EINTR
        guard truncated == 0 else {
            let cleanupFailure = HTTPTransportError.discardFailed(
                reason: String(cString: strerror(errno)))
            // A failed rollback is more important than an earlier protocol
            // error: the current range may still be present on disk.
            terminalFailure = cleanupFailure
            state = .aborted
            return cleanupFailure
        }
        written = 0
        // This method is the manager's integrity boundary. Once rollback has
        // succeeded, the reason for rejecting the returned response becomes
        // the terminal result even if a custom transport swallowed an earlier
        // sink failure.
        terminalFailure = failure
        state = .aborted
        return failure
    }

    @discardableResult
    private func failLocked(_ failure: HTTPTransportError) -> HTTPTransportError {
        if terminalFailure == nil { terminalFailure = failure }
        state = .aborted
        return terminalFailure ?? failure
    }

    private static func saturatingAdd(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? UInt64.max : sum
    }
}

/// `URLSession`, with the redirect chain preserved.
///
/// The redirect is the whole reason this is not three lines. `resolve/<rev>/…`
/// answers `302` and puts `x-linked-size`, `x-linked-etag` and `x-repo-commit`
/// on *that* response; `URLSession` follows the redirect and hands back the
/// CDN's headers, where none of the three exist. So the delegate records the
/// redirect's headers and they are overlaid onto the final response — origin
/// wins for exactly those three names, and for nothing else.
public struct URLSessionTransport: HTTPTransport {
    /// Header names taken from the origin's redirect rather than from the CDN's
    /// final answer.
    static let originHeaderNames = ["x-linked-size", "x-linked-etag", "x-repo-commit"]

    private let configuration: URLSessionConfiguration

    public init(configuration: URLSessionConfiguration = .default) {
        self.configuration = configuration
    }

    public func data(for request: HTTPRequest) async throws -> HTTPResponse {
        let collector = ResponseCollector(sink: nil, request: request)
        let session = URLSession(
            configuration: configuration, delegate: collector, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await run(request, on: session, collector: collector)
    }

    public func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        let collector = ResponseCollector(sink: sink, request: request)
        let session = URLSession(
            configuration: configuration, delegate: collector, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await run(request, on: session, collector: collector)
    }

    private func run(
        _ request: HTTPRequest, on session: URLSession, collector: ResponseCollector
    ) async throws -> HTTPResponse {
        var urlRequest = URLRequest(url: request.url, timeoutInterval: request.timeout)
        for (name, value) in request.headers { urlRequest.setValue(value, forHTTPHeaderField: name) }
        // Artifact digests describe transferred bytes. Transparent content
        // coding would make Content-Length and the bytes written name
        // different representations, so downloads require identity.
        urlRequest.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        if let range = request.rangeHeaderValue {
            urlRequest.setValue(range, forHTTPHeaderField: "Range")
        }

        let task = session.dataTask(with: urlRequest)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                collector.attach(continuation)
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }
}

/// The network authority to which an explicit `Authorization` value belongs.
///
/// Default ports are normalized so `https://host` and `https://host:443` are
/// one authority. Anything other than HTTP(S) fails closed: a redirect cannot
/// carry a credential to an origin whose effective port is undefined here.
struct HTTPRedirectOrigin: Sendable, Equatable {
    let scheme: String
    let host: String
    let port: Int

    init?(_ url: URL) {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
            let scheme = components.scheme?.lowercased(),
            scheme == "http" || scheme == "https",
            let host = components.host?.lowercased(), !host.isEmpty
        else { return nil }
        let port = components.port ?? (scheme == "https" ? 443 : 80)
        self.init(scheme: scheme, host: host, port: port)
    }

    private init(scheme: String, host: String, port: Int) {
        self.scheme = scheme
        self.host = host
        self.port = port
    }
}

/// Binds one caller-supplied authorization value to the request's origin.
/// URLSession's synthesized redirect request is not treated as authority: the
/// delegate explicitly restores the value for the same origin and explicitly
/// removes it everywhere else.
private struct RedirectAuthorizationPolicy: Sendable {
    let origin: HTTPRedirectOrigin?
    let value: String

    init?(_ request: HTTPRequest) {
        guard let value = request.headers.first(where: {
            $0.key.caseInsensitiveCompare("Authorization") == .orderedSame
        })?.value,
            !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        origin = HTTPRedirectOrigin(request.url)
        self.value = value
    }

    func apply(to request: inout URLRequest) {
        if let origin, let url = request.url, HTTPRedirectOrigin(url) == origin {
            request.setValue(value, forHTTPHeaderField: "Authorization")
        } else {
            request.setValue(nil, forHTTPHeaderField: "Authorization")
        }
    }

    func isSameOrigin(_ url: URL) -> Bool {
        guard let origin else { return false }
        return HTTPRedirectOrigin(url) == origin
    }
}

/// Turns the delegate callbacks into one `HTTPResponse`.
final class ResponseCollector: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let sink: HTTPDownloadSink?
    private let redirectAuthorizationPolicy: RedirectAuthorizationPolicy?
    private var authorizationChainStayedAtOrigin = true
    private var continuation: CheckedContinuation<HTTPResponse, Error>?
    private var terminalResult: Result<HTTPResponse, Error>?
    private var finished = false
    private var status: Int?
    private var headers: [String: String] = [:]
    private var originHeaders: [String: String] = [:]
    private var capturedOriginRedirect = false
    private var buffer = Data()

    init(sink: HTTPDownloadSink?, request: HTTPRequest? = nil) {
        self.sink = sink
        redirectAuthorizationPolicy = request.flatMap(RedirectAuthorizationPolicy.init)
    }

    func attach(_ continuation: CheckedContinuation<HTTPResponse, Error>) {
        lock.lock()
        let ready = terminalResult
        terminalResult = nil
        let wasFinished = finished
        let alreadyAttached = self.continuation != nil
        if ready == nil, !wasFinished, !alreadyAttached {
            self.continuation = continuation
        }
        lock.unlock()

        if let ready {
            Self.resume(continuation, with: ready)
        } else if wasFinished || alreadyAttached {
            continuation.resume(
                throwing: HTTPTransportError.invalidSink(
                    reason: "the response continuation was attached more than once"))
        }
    }

    private func finish(with result: Result<HTTPResponse, Error>) {
        lock.lock()
        guard !finished else {
            lock.unlock()
            return
        }
        finished = true
        let pending = continuation
        continuation = nil
        if pending == nil { terminalResult = result }
        lock.unlock()
        guard let pending else { return }
        Self.resume(pending, with: result)
    }

    private static func resume(
        _ continuation: CheckedContinuation<HTTPResponse, Error>,
        with result: Result<HTTPResponse, Error>
    ) {
        switch result {
        case .success(let response): continuation.resume(returning: response)
        case .failure(let error): continuation.resume(throwing: error)
        }
    }

    // MARK: URLSessionTaskDelegate

    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        lock.lock()
        // Snapshot one authority response, including the absence of a header.
        // Filling gaps from a later CDN redirect would let that hop invent the
        // repository commit or linked digest that the resolve origin omitted.
        if !capturedOriginRedirect {
            originHeaders = ResponseCollector.lowercased(response.allHeaderFields)
            capturedOriginRedirect = true
        }
        if let policy = redirectAuthorizationPolicy,
            request.url.map(policy.isSameOrigin) != true
        {
            authorizationChainStayedAtOrigin = false
        }
        let mayCarryAuthorization = authorizationChainStayedAtOrigin
        lock.unlock()

        // `Range` is re-applied explicitly rather than assumed to survive the
        // hop. A redirect that silently drops it turns a 64 MiB chunk request
        // into a five-gigabyte one, which does not fail — it succeeds, slowly,
        // and writes the whole file where a chunk was expected.
        var followed = request
        if followed.value(forHTTPHeaderField: "Range") == nil,
            let original = task.originalRequest?.value(forHTTPHeaderField: "Range")
        {
            followed.setValue(original, forHTTPHeaderField: "Range")
        }
        // Authorization is an origin capability, not a redirect-chain
        // capability. Reapply it only when scheme, host and effective port all
        // match the initial request; otherwise remove it even if URLSession's
        // proposed request retained it. Artifact redirects still proceed to
        // signed CDN URLs, with Range and identity encoding intact.
        if mayCarryAuthorization {
            redirectAuthorizationPolicy?.apply(to: &followed)
        } else {
            followed.setValue(nil, forHTTPHeaderField: "Authorization")
        }
        followed.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        completionHandler(followed)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?)
    {
        if let error {
            if (error as? URLError)?.code == .cancelled {
                finish(with: .failure(HTTPTransportError.cancelled))
            } else {
                finish(with: .failure(error))
            }
            return
        }
        lock.lock()
        let status = self.status
        let merged = authoritativeHeadersLocked()
        // A download reports no body except when the response was not the
        // object, in which case the error page is what the caller needs.
        let body = (sink == nil || !(status == 200 || status == 206)) ? buffer : nil
        lock.unlock()

        guard let status else {
            finish(with: .failure(HTTPTransportError.notHTTP(url: task.originalRequest?.url?.absoluteString ?? "?")))
            return
        }
        if let sink, status == 200 || status == 206 {
            do {
                try sink.seal()
            } catch {
                finish(with: .failure(error))
                return
            }
        }
        finish(with: .success(HTTPResponse(statusCode: status, headers: merged, body: body)))
    }

    // MARK: URLSessionDataDelegate

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            finish(with: .failure(HTTPTransportError.notHTTP(url: response.url?.absoluteString ?? "?")))
            return
        }
        lock.lock()
        status = http.statusCode
        headers = ResponseCollector.lowercased(http.allHeaderFields)
        let responseHeaders = authoritativeHeadersLocked()
        lock.unlock()

        if let sink, (200...299).contains(http.statusCode) {
            do {
                try sink.admit(
                    HTTPResponse(
                        statusCode: http.statusCode, headers: responseHeaders, body: nil))
            } catch {
                completionHandler(.cancel)
                dataTask.cancel()
                finish(with: .failure(error))
                return
            }
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        let status = self.status ?? 0
        lock.unlock()

        // Only a body that *is* the object goes into the file. A 503's error
        // page is still a body, and appending it to a part file would leave
        // nine bytes of HTML in the middle of a tile — after which the resume
        // offset is past them and no size check ever notices.
        if let sink, status == 200 || status == 206 {
            do {
                try sink.write(data)
                return
            } catch {
                dataTask.cancel()
                finish(with: .failure(error))
                return
            }
        }
        lock.lock()
        if sink == nil {
            // `data(for:)` is the explicit in-memory metadata API. Its caller
            // needs the complete JSON or pointer body.
            buffer.append(data)
        } else {
            // A download keeps an error body only for diagnostics. Cap it
            // exactly: one unusually large delegate fragment must not leap
            // past the limit.
            let remaining = max(0, 64 * 1024 - buffer.count)
            if remaining > 0 {
                buffer.append(contentsOf: data.prefix(remaining))
            }
        }
        lock.unlock()
    }

    static func lowercased(_ fields: [AnyHashable: Any]) -> [String: String] {
        var out: [String: String] = [:]
        for (key, value) in fields {
            guard let name = key as? String else { continue }
            out[name.lowercased()] = String(describing: value)
        }
        return out
    }

    /// The first redirect is the resolve origin's complete authority snapshot.
    /// If it omitted an authority header, remove any value supplied by a later
    /// redirect or final CDN response rather than filling the gap.
    private func authoritativeHeadersLocked() -> [String: String] {
        var merged = headers
        guard capturedOriginRedirect else { return merged }
        for name in URLSessionTransport.originHeaderNames {
            merged.removeValue(forKey: name)
            if let value = originHeaders[name] { merged[name] = value }
        }
        return merged
    }
}
