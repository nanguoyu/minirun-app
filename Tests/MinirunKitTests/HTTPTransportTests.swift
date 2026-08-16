import Darwin
import Foundation
import XCTest

@testable import MinirunKit

/// The transport itself, over a real socket. A fake transport cannot prove any
/// of this — these are the behaviours a wrong assumption hides.
final class HTTPTransportTests: XCTestCase {

    private var origin: FixtureOrigin!
    private let revision = String(repeating: "7", count: 40)
    private let streamingMemoryCeilingBytes: UInt64 = 64 << 20

    override func setUpWithError() throws {
        try super.setUpWithError()
        origin = try FixtureOrigin.start(revision: revision, files: FixtureArtifact.files())
    }

    override func tearDown() {
        origin?.stop()
        origin = nil
        super.tearDown()
    }

    /// The digest lives on the origin's 302. `URLSession` follows the redirect
    /// and hands back the CDN's headers, where it does not exist — so if this
    /// fails, every download silently becomes unverifiable.
    func testTheOriginsDigestSurvivesTheRedirectToTheCDN() async throws {
        let transport = URLSessionTransport()
        let path = "global/global-embed.bin"
        let body = try XCTUnwrap(origin.bytes(of: path))
        let url = origin.repo.resolveURL(path: path, endpoint: origin.baseURL)

        let response = try await transport.data(for: HTTPRequest(url: url))
        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(response.linkedSHA256, FileDigestComputer.sha256(of: body))
        XCTAssertEqual(response.linkedSizeBytes, UInt64(body.count))
        XCTAssertEqual(response.repoCommit, revision)
        XCTAssertEqual(response.body, body)
    }

    func testMetadataBearerIsRemovedBeforeACrossOriginRedirect() async throws {
        let bearer = "Bearer metadata-secret"
        let destinationProbe = HeaderProbe()
        let destinationBody = Data("destination".utf8)
        let destination = try LocalHTTPServer { request in
            destinationProbe.record(request.headers)
            return LocalHTTPServer.Response(status: 200, body: destinationBody)
        }
        defer { destination.stop() }

        let sourceProbe = HeaderProbe()
        let source = try LocalHTTPServer { request in
            sourceProbe.record(request.headers)
            return LocalHTTPServer.Response(
                status: 302,
                headers: [
                    "location": destination.baseURL.appendingPathComponent("metadata").absoluteString
                ])
        }
        defer { source.stop() }

        let response = try await URLSessionTransport().data(
            for: HTTPRequest(
                url: source.baseURL.appendingPathComponent("metadata"),
                headers: ["Authorization": bearer]))

        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(response.body, destinationBody)
        XCTAssertEqual(sourceProbe.headers["authorization"], bearer)
        XCTAssertEqual(destinationProbe.requestCount, 1, "the redirect should still be usable")
        XCTAssertNil(
            destinationProbe.headers["authorization"],
            "a different effective port is a different origin and must never receive the bearer")
    }

    func testMetadataBearerIsPreservedAcrossASameOriginRedirect() async throws {
        let bearer = "Bearer same-origin-secret"
        let probe = HeaderProbe()
        let finalBody = Data("same origin".utf8)
        let server = try LocalHTTPServer { request in
            probe.record(request.headers)
            if request.path == "/start" {
                return LocalHTTPServer.Response(
                    status: 302, headers: ["location": "/finish"])
            }
            return LocalHTTPServer.Response(status: 200, body: finalBody)
        }
        defer { server.stop() }

        let response = try await URLSessionTransport().data(
            for: HTTPRequest(
                url: server.baseURL.appendingPathComponent("start"),
                headers: ["authorization": bearer]))

        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(response.body, finalBody)
        XCTAssertEqual(probe.allHeaders.count, 2)
        XCTAssertEqual(probe.allHeaders[0]["authorization"], bearer)
        XCTAssertEqual(probe.allHeaders[1]["authorization"], bearer)
    }

    func testBearerNeverReturnsAfterAChainLeavesItsOrigin() async throws {
        let bearer = "Bearer do-not-bounce"
        let originProbe = HeaderProbe()
        let foreignProbe = HeaderProbe()
        let finalBody = Data("returned without bearer".utf8)

        final class OriginBox: @unchecked Sendable { var url: URL! }
        let originBox = OriginBox()
        let foreign = try LocalHTTPServer { request in
            foreignProbe.record(request.headers)
            return LocalHTTPServer.Response(
                status: 302,
                headers: [
                    "location": originBox.url.appendingPathComponent("finish").absoluteString
                ])
        }
        defer { foreign.stop() }
        let origin = try LocalHTTPServer { request in
            originProbe.record(request.headers)
            if request.path == "/start" {
                return LocalHTTPServer.Response(
                    status: 302,
                    headers: [
                        "location": foreign.baseURL.appendingPathComponent("middle").absoluteString
                    ])
            }
            return LocalHTTPServer.Response(status: 200, body: finalBody)
        }
        defer { origin.stop() }
        originBox.url = origin.baseURL

        let response = try await URLSessionTransport().data(
            for: HTTPRequest(
                url: origin.baseURL.appendingPathComponent("start"),
                headers: ["Authorization": bearer]))

        XCTAssertEqual(response.body, finalBody)
        XCTAssertEqual(originProbe.allHeaders.count, 2)
        XCTAssertEqual(originProbe.allHeaders[0]["authorization"], bearer)
        XCTAssertNil(foreignProbe.headers["authorization"])
        XCTAssertNil(originProbe.allHeaders[1]["authorization"])
    }

    func testRedirectOriginUsesSchemeHostAndEffectivePort() throws {
        XCTAssertEqual(
            HTTPRedirectOrigin(try XCTUnwrap(URL(string: "https://EXAMPLE.com/path"))),
            HTTPRedirectOrigin(try XCTUnwrap(URL(string: "https://example.com:443/other"))))
        XCTAssertNotEqual(
            HTTPRedirectOrigin(try XCTUnwrap(URL(string: "http://example.com"))),
            HTTPRedirectOrigin(try XCTUnwrap(URL(string: "https://example.com"))))
        XCTAssertNotEqual(
            HTTPRedirectOrigin(try XCTUnwrap(URL(string: "https://example.com"))),
            HTTPRedirectOrigin(try XCTUnwrap(URL(string: "https://other.example.com"))))
        XCTAssertNotEqual(
            HTTPRedirectOrigin(try XCTUnwrap(URL(string: "https://example.com"))),
            HTTPRedirectOrigin(try XCTUnwrap(URL(string: "https://example.com:444"))))
    }

    /// And the CDN's own `ETag` is never mistaken for it. The fixture serves an
    /// `ETag` that is a hash of *different* bytes, exactly as the live CDN does.
    func testTheCDNETagIsNotOfferedAsADigest() async throws {
        let path = "global/global-embed.bin"
        let body = try XCTUnwrap(origin.bytes(of: path))
        let response = try await URLSessionTransport().data(
            for: HTTPRequest(url: origin.baseURL.appendingPathComponent("cdn/\(path)")))
        let etag = try XCTUnwrap(response.header("etag")).replacingOccurrences(of: "\"", with: "")
        XCTAssertNotEqual(etag, FileDigestComputer.sha256(of: body))
        XCTAssertNil(response.linkedSHA256)
    }

    func testARangedRequestGets206AndTheRangeSurvivesTheRedirect() async throws {
        let path = FixtureArtifact.largeFilePath
        let body = try XCTUnwrap(origin.bytes(of: path))
        let url = origin.repo.resolveURL(path: path, endpoint: origin.baseURL)

        let response = try await URLSessionTransport().data(
            for: HTTPRequest(url: url, byteRange: 16...47))
        XCTAssertEqual(response.statusCode, 206, "the redirect must not drop the Range header")
        XCTAssertEqual(response.contentRange?.start, 16)
        XCTAssertEqual(response.contentRange?.end, 47)
        XCTAssertEqual(response.contentRange?.total, UInt64(body.count))
        XCTAssertEqual(response.body, body.subdata(in: 16..<48))
    }

    func testDownloadAppendsAndCountsEveryByteItWrote() async throws {
        let path = FixtureArtifact.largeFilePath
        let body = try XCTUnwrap(origin.bytes(of: path))
        let url = origin.repo.resolveURL(path: path, endpoint: origin.baseURL)
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-transport-\(UUID().uuidString).part")
        defer { try? FileManager.default.removeItem(at: file) }
        XCTAssertTrue(FileManager.default.createFile(atPath: file.path, contents: nil))
        let descriptor = Darwin.open(file.path, O_RDWR | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { _ = Darwin.close(descriptor) }

        let counted = Counter()
        let firstRequest = HTTPRequest(url: url, byteRange: 0...999)
        let firstSink = try HTTPDownloadSink(
            borrowing: descriptor, request: firstRequest,
            expectedObjectLength: UInt64(body.count), onBytes: { counted.add($0) })
        _ = try await URLSessionTransport().download(firstRequest, into: firstSink)
        XCTAssertEqual(try Data(contentsOf: file), body.prefix(1000))
        XCTAssertEqual(counted.total, 1000)
        XCTAssertNotEqual(
            Darwin.fcntl(descriptor, F_GETFD), -1,
            "the transport must not close its borrowed descriptor")

        // Appending continues where the file ends, which is what resume relies
        // on: the file's length is the offset, not a remembered number.
        let secondRequest = HTTPRequest(url: url, byteRange: 1000...1999)
        let secondSink = try HTTPDownloadSink(
            borrowing: descriptor, request: secondRequest,
            expectedObjectLength: UInt64(body.count), onBytes: { counted.add($0) })
        _ = try await URLSessionTransport().download(secondRequest, into: secondSink)
        XCTAssertEqual(try Data(contentsOf: file), body.prefix(2000))
        XCTAssertEqual(counted.total, 2000)
    }

    func testRangeResponseMustBeExactBeforeTheFirstBodyByte() async throws {
        let cases: [(name: String, response: LocalHTTPServer.Response)] = [
            (
                "range ignored with 200",
                .init(status: 200, body: Data(repeating: 1, count: 8))
            ),
            (
                "wrong content range",
                .init(
                    status: 206,
                    headers: ["content-range": "bytes 1-8/16"],
                    body: Data(repeating: 2, count: 8))
            ),
            (
                "overlong body declared up front",
                .init(
                    status: 206,
                    headers: ["content-range": "bytes 0-7/16"],
                    body: Data(repeating: 3, count: 9))
            ),
            (
                "encoded representation",
                .init(
                    status: 206,
                    headers: [
                        "content-range": "bytes 0-7/16",
                        "content-encoding": "gzip",
                    ],
                    body: Data(repeating: 4, count: 8))
            ),
        ]

        for item in cases {
            let server = try LocalHTTPServer { _ in item.response }
            defer { server.stop() }
            let request = HTTPRequest(url: server.baseURL.appendingPathComponent("object"), byteRange: 0...7)
            let fixture = try SinkFixture(request: request, objectLength: 16)
            do {
                _ = try await URLSessionTransport().download(request, into: fixture.sink)
                XCTFail("\(item.name) must be refused")
            } catch {
                guard case .responseRejected? = error as? HTTPTransportError else {
                    XCTFail("\(item.name) produced \(error)")
                    continue
                }
            }
            XCTAssertEqual(
                try Data(contentsOf: fixture.url), Data(),
                "\(item.name) wrote before response admission")
        }
    }

    func testFullObjectAcceptsOnly200() async throws {
        let bytes = FixtureBytes.make(4096, seed: 71)
        for status in [200, 206] {
            let server = try LocalHTTPServer { _ in
                LocalHTTPServer.Response(
                    status: status,
                    headers: status == 206 ? ["content-range": "bytes 0-4095/4096"] : [:],
                    body: bytes)
            }
            defer { server.stop() }
            let request = HTTPRequest(url: server.baseURL.appendingPathComponent("object"))
            let fixture = try SinkFixture(request: request, objectLength: UInt64(bytes.count))
            if status == 200 {
                _ = try await URLSessionTransport().download(request, into: fixture.sink)
                XCTAssertEqual(try Data(contentsOf: fixture.url), bytes)
            } else {
                do {
                    _ = try await URLSessionTransport().download(request, into: fixture.sink)
                    XCTFail("a complete-object request must reject 206")
                } catch {
                    guard case .responseRejected? = error as? HTTPTransportError else {
                        return XCTFail("expected responseRejected, got \(error)")
                    }
                }
                XCTAssertEqual(try Data(contentsOf: fixture.url), Data())
            }
        }
    }

    func testShortBodyCannotSealButLeavesOnlyItsResumablePrefix() async throws {
        let body = Data(repeating: 0xA5, count: 8)
        let server = try LocalHTTPServer { _ in
            LocalHTTPServer.Response(
                status: 206,
                headers: [
                    "content-range": "bytes 0-15/16",
                    // Deliberately larger than the bytes the socket sends.
                    "content-length": "16",
                ], body: body)
        }
        defer { server.stop() }
        let request = HTTPRequest(url: server.baseURL.appendingPathComponent("short"), byteRange: 0...15)
        let fixture = try SinkFixture(request: request, objectLength: 16)

        do {
            _ = try await URLSessionTransport().download(request, into: fixture.sink)
            XCTFail("a short response must not seal")
        } catch {
            // URLSession may surface the premature EOF itself; if it hands the
            // response through, the sink's exact-length seal surfaces it.
        }
        XCTAssertLessThanOrEqual(try Data(contentsOf: fixture.url).count, body.count)
        XCTAssertFalse(fixture.sink.isSealed)
    }

    func testDownloadRequestsIdentityEncodingAndAbortRevokesEvenASealedSink() async throws {
        let probe = HeaderProbe()
        let bytes = Data(repeating: 0x5A, count: 32)
        let server = try LocalHTTPServer { request in
            probe.record(request.headers)
            return LocalHTTPServer.Response(
                status: 206,
                headers: ["content-range": "bytes 0-31/32"], body: bytes)
        }
        defer { server.stop() }
        let request = HTTPRequest(url: server.baseURL.appendingPathComponent("identity"), byteRange: 0...31)
        let fixture = try SinkFixture(request: request, objectLength: 32)
        _ = try await URLSessionTransport().download(request, into: fixture.sink)
        XCTAssertEqual(probe.headers["accept-encoding"], "identity")

        let retained = try SinkFixture(request: request, objectLength: 32)
        let response = HTTPResponse(
            statusCode: 206,
            headers: ["content-range": "bytes 0-31/32", "content-length": "32"], body: nil)
        try retained.sink.admit(response)
        retained.sink.abort()
        XCTAssertThrowsError(try retained.sink.write(bytes))
        XCTAssertEqual(try Data(contentsOf: retained.url), Data())

        let sealed = try SinkFixture(request: request, objectLength: 32)
        try sealed.sink.admit(response)
        try sealed.sink.write(bytes)
        try sealed.sink.seal()
        sealed.sink.abort()
        XCTAssertThrowsError(try sealed.sink.write(Data()))
        XCTAssertThrowsError(try sealed.sink.admit(response))
        XCTAssertEqual(try Data(contentsOf: sealed.url), bytes)
        XCTAssertNotEqual(
            Darwin.fcntl(sealed.descriptor, F_GETFD), -1,
            "revoking a sealed sink must not close its owner's descriptor")
    }

    func testOverlongOrChangedResponseRollsBackOnlyItsCurrentRange() throws {
        let sealedPrefix = Data([0xA0, 0xA1, 0xA2, 0xA3])
        let request = HTTPRequest(
            url: URL(string: "https://example.invalid/object")!, byteRange: 4...7)

        do {
            let network = Counter()
            let fixture = try SinkFixture(
                request: request, objectLength: 8, initial: sealedPrefix,
                onBytes: { network.add($0) })
            let admitted = HTTPResponse(
                statusCode: 206,
                headers: [
                    "content-range": "bytes 4-7/8",
                    "content-length": "4",
                ], body: nil)
            try fixture.sink.admit(admitted)
            try fixture.sink.write(Data([1, 2]))
            XCTAssertThrowsError(try fixture.sink.write(Data([3, 4, 5]))) { error in
                guard case .responseBodyLength? = error as? HTTPTransportError else {
                    return XCTFail("expected responseBodyLength, got \(error)")
                }
            }
            XCTAssertEqual(try Data(contentsOf: fixture.url), sealedPrefix)
            XCTAssertFalse(fixture.sink.isSealed)
            XCTAssertEqual(network.total, 5, "all offered body bytes came off the wire")
            XCTAssertEqual(
                fixture.sink.discardedBodyBytes, 5,
                "every byte from an overlong response is unusable")
        }

        do {
            let fixture = try SinkFixture(
                request: request, objectLength: 8, initial: sealedPrefix)
            let admitted = HTTPResponse(
                statusCode: 206,
                headers: [
                    "content-range": "bytes 4-7/8",
                    "content-length": "4",
                ], body: nil)
            try fixture.sink.admit(admitted)
            try fixture.sink.write(Data([1, 2, 3, 4]))
            try fixture.sink.seal()

            let returned = HTTPResponse(
                statusCode: 503, headers: ["content-length": "0"], body: nil)
            XCTAssertThrowsError(try fixture.sink.admit(returned)) { error in
                guard case .responseRejected? = error as? HTTPTransportError else {
                    return XCTFail("expected responseRejected, got \(error)")
                }
            }
            XCTAssertEqual(
                try Data(contentsOf: fixture.url), sealedPrefix,
                "a transport cannot admit one response and return another")
            XCTAssertEqual(fixture.sink.discardedBodyBytes, 4)
        }

        do {
            let fixture = try SinkFixture(
                request: request, objectLength: 8, initial: sealedPrefix)
            let admitted = HTTPResponse(
                statusCode: 206,
                headers: [
                    "content-range": "bytes 4-7/8",
                    "content-length": "4",
                    "x-repo-commit": "admitted-revision",
                    "x-linked-size": "8",
                ], body: nil)
            try fixture.sink.admit(admitted)
            try fixture.sink.write(Data([1, 2, 3, 4]))
            try fixture.sink.seal()

            let returned = HTTPResponse(
                statusCode: 206,
                headers: [
                    "content-range": "bytes 4-7/8",
                    "content-length": "4",
                    "x-repo-commit": "returned-revision",
                    "x-linked-size": "8",
                ], body: nil)
            XCTAssertThrowsError(try fixture.sink.admit(returned)) { error in
                guard case .responseRejected(_, let reason)? = error as? HTTPTransportError,
                    reason.contains("authority changed")
                else {
                    return XCTFail("expected response authority refusal, got \(error)")
                }
            }
            XCTAssertEqual(try Data(contentsOf: fixture.url), sealedPrefix)
            XCTAssertEqual(fixture.sink.discardedBodyBytes, 4)
        }

        do {
            let network = Counter()
            let fixture = try SinkFixture(
                request: request, objectLength: 8, initial: sealedPrefix,
                onBytes: { network.add($0) })
            let response = HTTPResponse(
                statusCode: 206,
                headers: [
                    "content-range": "bytes 4-7/8",
                    "content-length": "4",
                ], body: nil)
            try fixture.sink.admit(response)
            try fixture.sink.write(Data([1, 2, 3, 4]))
            try fixture.sink.seal()

            XCTAssertThrowsError(try fixture.sink.write(Data([5]))) { error in
                guard case .responseBodyLength? = error as? HTTPTransportError else {
                    return XCTFail("expected sealed-response overrun, got \(error)")
                }
            }
            XCTAssertEqual(try Data(contentsOf: fixture.url), sealedPrefix)
            XCTAssertEqual(network.total, 5)
            XCTAssertEqual(fixture.sink.discardedBodyBytes, 5)
        }
    }

    func testRollbackRetriesEINTRAndSurfacesAPersistentCleanupFailure() throws {
        let prefix = Data([0xA0, 0xA1, 0xA2, 0xA3])
        let request = HTTPRequest(
            url: URL(string: "https://example.invalid/rollback")!, byteRange: 4...7)
        let response = HTTPResponse(
            statusCode: 206,
            headers: [
                "content-range": "bytes 4-7/8",
                "content-length": "4",
            ], body: nil)

        do {
            let probe = TruncateProbe(mode: .failOnceWithEINTR)
            let fixture = try SinkFixture(
                request: request, objectLength: 8, initial: prefix,
                truncateOperation: { probe.truncate($0, to: $1) })
            try fixture.sink.admit(response)
            try fixture.sink.write(Data([1, 2]))
            XCTAssertThrowsError(try fixture.sink.write(Data([3, 4, 5]))) { error in
                guard case .responseBodyLength? = error as? HTTPTransportError else {
                    return XCTFail("expected original length error, got \(error)")
                }
            }
            XCTAssertEqual(probe.attemptCount, 2)
            XCTAssertEqual(try Data(contentsOf: fixture.url), prefix)
        }

        do {
            let probe = TruncateProbe(mode: .failAlwaysWithEIO)
            let fixture = try SinkFixture(
                request: request, objectLength: 8, initial: prefix,
                truncateOperation: { probe.truncate($0, to: $1) })
            try fixture.sink.admit(response)
            try fixture.sink.write(Data([1, 2]))
            XCTAssertThrowsError(try fixture.sink.write(Data([3, 4, 5]))) { error in
                guard case .discardFailed? = error as? HTTPTransportError else {
                    return XCTFail("expected discardFailed, got \(error)")
                }
            }
            XCTAssertEqual(probe.attemptCount, 1)
            XCTAssertEqual(
                try Data(contentsOf: fixture.url), prefix + Data([1, 2]),
                "a failed rollback must report that its current range remains")
            XCTAssertEqual(fixture.sink.discardedBodyBytes, 5)
        }
    }

    func testFirstRedirectIsACompleteAuthoritySnapshotAcrossMultipleHops() async throws {
        let bytes = Data([1, 2, 3, 4, 5, 6, 7, 8])
        let finalProbe = HeaderProbe()
        let final = try LocalHTTPServer { request in
            finalProbe.record(request.headers)
            return LocalHTTPServer.Response(
                status: 206,
                headers: [
                    "content-range": "bytes 0-7/8",
                    "x-repo-commit": "final-hop-must-not-fill-origin-gap",
                    "x-linked-size": "800",
                ], body: bytes)
        }
        defer { final.stop() }
        let middleProbe = HeaderProbe()
        let middle = try LocalHTTPServer { request in
            middleProbe.record(request.headers)
            return LocalHTTPServer.Response(
                status: 302,
                headers: [
                    "location": final.baseURL.appendingPathComponent("object").absoluteString,
                    "x-repo-commit": "middle-hop-must-not-fill-origin-gap",
                    "x-linked-size": "80",
                ], body: Data("middle".utf8))
        }
        defer { middle.stop() }
        let first = try LocalHTTPServer { _ in
            LocalHTTPServer.Response(
                status: 302,
                headers: [
                    "location": middle.baseURL.appendingPathComponent("object").absoluteString,
                    "x-linked-size": "8",
                    "x-linked-etag": "\"origin-digest\"",
                    // Deliberately no x-repo-commit. Its absence is authority.
                ], body: Data("origin".utf8))
        }
        defer { first.stop() }

        let request = HTTPRequest(
            url: first.baseURL.appendingPathComponent("resolve"),
            headers: ["Authorization": "Bearer artifact-secret"], byteRange: 0...7)
        let fixture = try SinkFixture(request: request, objectLength: 8)
        let response = try await URLSessionTransport().download(
            request, into: fixture.sink)

        XCTAssertNil(middleProbe.headers["authorization"])
        XCTAssertNil(finalProbe.headers["authorization"])
        XCTAssertEqual(middleProbe.headers["range"], "bytes=0-7")
        XCTAssertEqual(finalProbe.headers["range"], "bytes=0-7")
        XCTAssertNil(response.repoCommit)
        XCTAssertEqual(response.linkedSizeBytes, 8)
        XCTAssertEqual(response.linkedSHA256, "origin-digest")
        XCTAssertEqual(try Data(contentsOf: fixture.url), bytes)
    }

    func testPreCancelledCompletionBeforeAttachStillResumesTheContinuation() async throws {
        let collector = ResponseCollector(sink: nil)
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }
        let task = session.dataTask(
            with: URL(string: "https://example.invalid/pre-cancelled")!)

        // Reproduce the ordering possible when a task is already cancelled:
        // URLSession completes before `run` reaches collector.attach().
        collector.urlSession(
            session, task: task, didCompleteWithError: URLError(.cancelled))
        do {
            _ = try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<HTTPResponse, Error>) in
                collector.attach(continuation)
            }
            XCTFail("the pre-cancelled operation must throw")
        } catch {
            XCTAssertEqual(error as? HTTPTransportError, .cancelled)
        }
    }

    func testAbortPreventsWritesAfterTheOwnerReusesTheDescriptorNumber() throws {
        let firstURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-fd-owner-\(UUID().uuidString)")
        let replacementURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-fd-reused-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: firstURL)
            try? FileManager.default.removeItem(at: replacementURL)
        }
        try Data().write(to: firstURL)
        let sentinel = Data("replacement must remain unchanged".utf8)
        try sentinel.write(to: replacementURL)
        let owned = Darwin.open(firstURL.path, O_RDWR | O_CLOEXEC)
        let replacement = Darwin.open(replacementURL.path, O_RDWR | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(owned, 0)
        XCTAssertGreaterThanOrEqual(replacement, 0)
        guard owned >= 0, replacement >= 0 else { return }

        let request = HTTPRequest(
            url: URL(string: "https://example.invalid/reuse")!, byteRange: 0...3)
        let sink = try HTTPDownloadSink(
            borrowing: owned, request: request, expectedObjectLength: 4,
            onBytes: { _ in })
        let response = HTTPResponse(
            statusCode: 206,
            headers: [
                "content-range": "bytes 0-3/4",
                "content-length": "4",
            ], body: nil)
        try sink.admit(response)
        sink.abort()

        // dup2 atomically performs owner-close plus descriptor-number reuse,
        // leaving no scheduling window in which an unrelated thread can take
        // the number. A retained sink must still be inert.
        XCTAssertEqual(Darwin.dup2(replacement, owned), owned)
        _ = Darwin.close(replacement)
        XCTAssertThrowsError(try sink.write(Data([9, 9, 9, 9])))
        _ = Darwin.close(owned)
        XCTAssertEqual(try Data(contentsOf: replacementURL), sentinel)
    }

    func testSinkRequiresTheExactExistingOffsetAndAnInObjectRange() throws {
        XCTAssertThrowsError(
            try SinkFixture(
                request: HTTPRequest(
                    url: URL(string: "https://example.invalid/offset")!,
                    byteRange: 4...7),
                objectLength: 8, initial: Data(repeating: 0, count: 3))) { error in
                    guard case .invalidSink? = error as? HTTPTransportError else {
                        return XCTFail("expected invalidSink, got \(error)")
                    }
                }
        XCTAssertThrowsError(
            try SinkFixture(
                request: HTTPRequest(
                    url: URL(string: "https://example.invalid/range")!,
                    byteRange: 4...8),
                objectLength: 8, initial: Data(repeating: 0, count: 4))) { error in
                    guard case .invalidSink? = error as? HTTPTransportError else {
                        return XCTFail("expected invalidSink, got \(error)")
                    }
                }
    }

    func testCancellationRevokesTheSinkEvenForEmptyOrRetainedWrites() throws {
        let cancellation = CancellationFlag()
        let network = Counter()
        let request = HTTPRequest(
            url: URL(string: "https://example.invalid/cancel")!, byteRange: 0...3)
        let fixture = try SinkFixture(
            request: request, objectLength: 4,
            onBytes: { network.add($0) },
            cancellationRequested: { cancellation.isCancelled })
        let response = HTTPResponse(
            statusCode: 206,
            headers: [
                "content-range": "bytes 0-3/4",
                "content-length": "4",
            ], body: nil)
        try fixture.sink.admit(response)
        cancellation.cancel()

        for bytes in [Data(), Data([1, 2, 3, 4])] {
            XCTAssertThrowsError(try fixture.sink.write(bytes)) { error in
                XCTAssertEqual(error as? HTTPTransportError, .cancelled)
            }
        }
        XCTAssertEqual(try Data(contentsOf: fixture.url), Data())
        XCTAssertEqual(network.total, 4, "late delegate bytes remain network traffic")
        XCTAssertEqual(fixture.sink.discardedBodyBytes, 4)
        XCTAssertNotEqual(
            Darwin.fcntl(fixture.descriptor, F_GETFD), -1,
            "revoking the sink must not close its owner's descriptor")

        let abortedNetwork = Counter()
        let aborted = try SinkFixture(
            request: request, objectLength: 4,
            onBytes: { abortedNetwork.add($0) })
        try aborted.sink.admit(response)
        aborted.sink.abort()
        XCTAssertThrowsError(try aborted.sink.write(Data([5, 6])))
        XCTAssertEqual(abortedNetwork.total, 2)
        XCTAssertEqual(
            aborted.sink.discardedBodyBytes, 2,
            "bytes offered after explicit revocation are unusable traffic")
    }

    func testAnExactBodyThatTheTransportDidNotSealIsDiscarded() throws {
        let request = HTTPRequest(
            url: URL(string: "https://example.invalid/unsealed")!, byteRange: 0...3)
        let fixture = try SinkFixture(request: request, objectLength: 4)
        let response = HTTPResponse(
            statusCode: 206,
            headers: [
                "content-range": "bytes 0-3/4",
                "content-length": "4",
            ], body: nil)
        try fixture.sink.admit(response)
        try fixture.sink.write(Data([1, 2, 3, 4]))

        XCTAssertFalse(fixture.sink.isSealed)
        let failure = fixture.sink.discardCurrentResponse(
            because: .invalidSink(reason: "transport returned before sealing"))
        guard case .invalidSink = failure else {
            return XCTFail("expected invalidSink, got \(failure)")
        }
        XCTAssertEqual(try Data(contentsOf: fixture.url), Data())
    }

    func testDownloadErrorBodyIsExactlyBoundedWithoutTruncatingMetadataResponses() async throws {
        let body = FixtureBytes.make(256 << 10, seed: 73)
        let server = try LocalHTTPServer { _ in
            LocalHTTPServer.Response(status: 503, body: body)
        }
        defer { server.stop() }
        let url = server.baseURL.appendingPathComponent("large-error")

        let metadata = try await URLSessionTransport().data(for: HTTPRequest(url: url))
        XCTAssertEqual(metadata.body, body)

        let request = HTTPRequest(url: url, byteRange: 0...7)
        let fixture = try SinkFixture(request: request, objectLength: 8)
        let download = try await URLSessionTransport().download(
            request, into: fixture.sink)
        XCTAssertEqual(download.statusCode, 503)
        XCTAssertEqual(download.body?.count, 64 << 10)
        XCTAssertEqual(download.body, Data(body.prefix(64 << 10)))
        XCTAssertEqual(try Data(contentsOf: fixture.url), Data())
    }

    /// A real loopback socket drives URLSession and the production delegate;
    /// the server repeats one 64 KiB buffer so neither side owns a whole-body
    /// fixture. This is both the reproducible local throughput benchmark and a
    /// regression gate against reintroducing `Data` accumulation.
    func testLocalSocketStreamingHasABoundedResidentWorkingSet() async throws {
        #if os(macOS)
            let bodyBytes = 128 << 20
            let chunkBytes = 64 << 10
            let chunk = FixtureBytes.make(chunkBytes, seed: 72)
            let repetitions = bodyBytes / chunkBytes
            let server = try LocalHTTPServer { _ in
                .repeating(
                    status: 206,
                    headers: [
                        "content-range": "bytes 0-\(bodyBytes - 1)/\(bodyBytes)"
                    ],
                    chunk: chunk, repetitions: repetitions)
            }
            defer { server.stop() }
            let request = HTTPRequest(
                url: server.baseURL.appendingPathComponent("stream"),
                byteRange: 0...UInt64(bodyBytes - 1))
            let fixture = try SinkFixture(
                request: request, objectLength: UInt64(bodyBytes))

            let baseline = try physicalFootprintBytes()
            let completion = CompletionFlag()
            let started = ContinuousClock.now
            let task = Task {
                defer { completion.finish() }
                return try await URLSessionTransport().download(
                    request, into: fixture.sink)
            }
            var peak = baseline
            let deadline = ContinuousClock.now + .seconds(30)
            while !completion.isFinished {
                if ContinuousClock.now >= deadline {
                    task.cancel()
                    _ = try? await task.value
                    return XCTFail("loopback streaming benchmark exceeded 30 seconds")
                }
                peak = max(peak, try physicalFootprintBytes())
                try await Task.sleep(nanoseconds: 2_000_000)
            }
            let response = try await task.value
            peak = max(peak, try physicalFootprintBytes())
            let elapsed = ContinuousClock.now - started
            let seconds = Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            let rate = seconds > 0 ? Double(bodyBytes) / seconds : 0
            let growth = peak > baseline ? peak - baseline : 0
            let secondsText = String(format: "%.6f", seconds)
            let rateText = String(format: "%.0f", rate)
            var metadata = stat()
            #if DEBUG
                let buildConfiguration = "debug"
            #else
                let buildConfiguration = "release"
            #endif

            XCTAssertEqual(response.statusCode, 206)
            XCTAssertEqual(Darwin.fstat(fixture.descriptor, &metadata), 0)
            XCTAssertEqual(metadata.st_size, off_t(bodyBytes))
            XCTAssertLessThanOrEqual(
                growth, streamingMemoryCeilingBytes,
                "streaming \(bodyBytes) bytes grew footprint by \(growth) bytes")
            print(
                "MINIRUN_HTTP_FD_SINK_BENCH bytes=\(bodyBytes) "
                    + "chunk_bytes=\(chunkBytes) seconds=\(secondsText) "
                    + "bytes_per_second=\(rateText) "
                    + "peak_resident_growth_bytes=\(growth) "
                    + "transport=loopback-http build=\(buildConfiguration) cache=page-cache "
                    + "thermal=uncontrolled os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
        #else
            throw XCTSkip("resident-memory sampling is a macOS regression gate")
        #endif
    }

    func testHeaderNamesAreLowercasedWhateverTheServerChose() {
        let response = HTTPResponse(
            statusCode: 200, headers: ["X-Linked-ETag": "\"ABC\"", "Content-Range": "bytes 0-1/2"],
            body: nil)
        XCTAssertEqual(response.linkedSHA256, "abc")
        XCTAssertEqual(response.contentRange?.total, 2)
    }

    func testTheLinkHeaderYieldsTheNextPage() {
        let response = HTTPResponse(
            statusCode: 200,
            headers: [
                "link":
                    "<https://example.com/first>; rel=\"prev\", <https://example.com/second>; rel=\"next\""
            ],
            body: nil)
        XCTAssertEqual(response.nextPageURL?.absoluteString, "https://example.com/second")

        XCTAssertNil(
            HTTPResponse(statusCode: 200, headers: [:], body: nil).nextPageURL)
        XCTAssertNil(
            HTTPResponse(
                statusCode: 200, headers: ["link": "<https://example.com/x>; rel=\"prev\""],
                body: nil
            ).nextPageURL)
        if case .malformed = HTTPResponse(
            statusCode: 200, headers: ["link": "not-a-target; rel=\"next\""], body: nil
        ).nextPage {
            // expected
        } else {
            XCTFail("a declared malformed next page must not look like the final page")
        }
    }

    func testAMissingFileIs404RatherThanAnEmptyFile() async throws {
        let url = origin.repo.resolveURL(path: "nope/absent.bin", endpoint: origin.baseURL)
        let response = try await URLSessionTransport().data(for: HTTPRequest(url: url))
        XCTAssertEqual(response.statusCode, 404)
    }

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var value: UInt64 = 0
        func add(_ bytes: UInt64) {
            lock.lock()
            value &+= bytes
            lock.unlock()
        }
        var total: UInt64 {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    private final class HeaderProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [[String: String]] = []
        func record(_ headers: [String: String]) {
            lock.lock()
            values.append(headers)
            lock.unlock()
        }
        var headers: [String: String] {
            lock.lock()
            defer { lock.unlock() }
            return values.last ?? [:]
        }
        var allHeaders: [[String: String]] {
            lock.lock()
            defer { lock.unlock() }
            return values
        }
        var requestCount: Int {
            lock.lock()
            defer { lock.unlock() }
            return values.count
        }
    }

    private final class CancellationFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var cancelled = false

        var isCancelled: Bool {
            lock.lock()
            defer { lock.unlock() }
            return cancelled
        }

        func cancel() {
            lock.lock()
            cancelled = true
            lock.unlock()
        }
    }

    private final class CompletionFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var finished = false

        var isFinished: Bool {
            lock.lock()
            defer { lock.unlock() }
            return finished
        }

        func finish() {
            lock.lock()
            finished = true
            lock.unlock()
        }
    }

    private final class TruncateProbe: @unchecked Sendable {
        enum Mode {
            case failOnceWithEINTR
            case failAlwaysWithEIO
        }

        private let mode: Mode
        private let lock = NSLock()
        private var attempts = 0

        init(mode: Mode) { self.mode = mode }

        func truncate(_ descriptor: Int32, to offset: off_t) -> Int32 {
            lock.lock()
            attempts += 1
            let currentAttempt = attempts
            lock.unlock()
            switch mode {
            case .failOnceWithEINTR where currentAttempt == 1:
                errno = EINTR
                return -1
            case .failAlwaysWithEIO:
                errno = EIO
                return -1
            default:
                return Darwin.ftruncate(descriptor, offset)
            }
        }

        var attemptCount: Int {
            lock.lock()
            defer { lock.unlock() }
            return attempts
        }
    }

    private final class SinkFixture {
        let url: URL
        let descriptor: Int32
        let sink: HTTPDownloadSink

        init(
            request: HTTPRequest, objectLength: UInt64, initial: Data = Data(),
            onBytes: @escaping @Sendable (UInt64) -> Void = { _ in },
            cancellationRequested: @escaping @Sendable () -> Bool = { false },
            truncateOperation: @escaping @Sendable (Int32, off_t) -> Int32 = {
                Darwin.ftruncate($0, $1)
            }
        ) throws {
            url = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-http-sink-\(UUID().uuidString).part")
            try initial.write(to: url)
            descriptor = Darwin.open(url.path, O_RDWR | O_CLOEXEC)
            guard descriptor >= 0 else {
                throw HTTPTransportError.cannotOpenFile(
                    path: url.path, reason: String(cString: strerror(errno)))
            }
            do {
                sink = try HTTPDownloadSink(
                    borrowing: descriptor, request: request,
                    expectedObjectLength: objectLength, onBytes: onBytes,
                    cancellationRequested: cancellationRequested,
                    truncateOperation: truncateOperation)
            } catch {
                _ = Darwin.close(descriptor)
                try? FileManager.default.removeItem(at: url)
                throw error
            }
        }

        deinit {
            sink.abort()
            _ = Darwin.close(descriptor)
            try? FileManager.default.removeItem(at: url)
        }
    }

    #if os(macOS)
        private func physicalFootprintBytes() throws -> UInt64 {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(
                MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
            let status = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(
                    to: integer_t.self, capacity: Int(count)
                ) { rebound in
                    task_info(
                        mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
                }
            }
            guard status == KERN_SUCCESS else {
                throw NSError(
                    domain: NSMachErrorDomain, code: Int(status),
                    userInfo: [
                        NSLocalizedDescriptionKey: "task_info(TASK_VM_INFO) failed"
                    ])
            }
            return UInt64(info.phys_footprint)
        }
    #endif
}
