import CoreFoundation
import Foundation

/// Walks a repository's file tree over the Hugging Face API.
///
/// The tree is the truth this package works from. `index.json` is a claim the
/// publisher wrote; the tree is what the server will actually serve, it names
/// every file including the ones no index mentions, and it is the only complete
/// digest table that exists — `lfs.oid` for LFS-backed blobs, `oid` for the
/// plain git blobs an index would not think to list.
public struct HuggingFaceTreeClient: Sendable {
    private let transport: any HTTPTransport
    private let endpoint: URL
    private let token: (@Sendable () -> String?)?

    public init(
        transport: any HTTPTransport, endpoint: URL = HuggingFaceEndpoint.default,
        token: (@Sendable () -> String?)? = nil
    ) {
        self.transport = transport
        self.endpoint = endpoint
        self.token = token
    }

    /// Every file in the repository at the pinned revision, sorted by path.
    ///
    /// Pagination is by the `Link: …; rel="next"` header — the first page holds
    /// 50 entries and a K3 tree holds 470, so a client that reads one page and
    /// stops under-counts by an order of magnitude while looking perfectly
    /// healthy. A missing `Link` header is not proof that a page was final, so
    /// the candidate path set is reconciled with the independently returned
    /// `siblings` set from the pinned repository-info endpoint before it is
    /// exposed to callers.
    public func files(in repo: HuggingFaceRepoRef) async throws -> [RepoFile] {
        (try await completeTree(in: repo)).files
    }

    /// A digest-bearing tree whose path set has an independent completeness
    /// authority. Internal callers that persist verification evidence retain
    /// the authority identity as well as the digest-plan identity.
    func completeTree(in repo: HuggingFaceRepoRef) async throws -> HuggingFaceCompleteTree {
        guard HuggingFaceRepoRef.isCanonicalRepositoryID(repo.repoID) else {
            throw CatalogError.malformedResponse(
                "'\(repo.repoID)' is not a canonical Hugging Face repository ID")
        }
        guard repo.isPinnedCommit else {
            throw CatalogError.unpinnedRevision(repoID: repo.repoID, given: repo.revision)
        }
        let firstPage = repo.treeURL(endpoint: endpoint)
        guard Self.httpOrigin(of: firstPage) != nil else {
            throw CatalogError.malformedResponse(
                "the tree endpoint is not an absolute HTTP(S) URL: \(firstPage.absoluteString)")
        }
        var next: URL? = firstPage
        var seen = Set<URL>()
        var seenPaths = Set<String>()
        var files: [RepoFile] = []

        while let url = next {
            guard Self.sameOrigin(url, firstPage) else {
                throw CatalogError.malformedResponse(
                    "the tree API returned a cross-origin next page: \(url.absoluteString)")
            }
            guard seen.insert(url).inserted else {
                throw CatalogError.malformedResponse(
                    "the tree API paginated back to \(url.absoluteString)")
            }
            let response = try await transport.data(for: request(url))
            guard response.statusCode == 200 else {
                throw CatalogError.transport(status: response.statusCode, url: url.absoluteString)
            }
            guard let body = response.body else {
                throw CatalogError.malformedResponse("empty tree page from \(url.absoluteString)")
            }
            let following: URL?
            switch response.nextPage {
            case .absent:
                following = nil
            case .url(let url):
                following = url
            case .malformed(let value):
                throw CatalogError.malformedResponse(
                    "the tree API returned a malformed next page: \(value)")
            }
            if let following, !Self.sameOrigin(following, firstPage) {
                // Check the target before constructing its request. In
                // particular, the caller's bearer token must never be copied
                // onto a Link target chosen by another origin.
                throw CatalogError.malformedResponse(
                    "the tree API returned a cross-origin next page: \(following.absoluteString)")
            }
            for file in try HuggingFaceTreeClient.decodePage(body) {
                guard seenPaths.insert(file.path).inserted else {
                    throw CatalogError.malformedResponse(
                        "the tree lists '\(file.path)' more than once")
                }
                files.append(file)
            }
            next = following
        }
        let sorted = files.sorted { $0.path < $1.path }
        let authority = try await repositoryFileAuthority(in: repo)
        let candidatePaths = Set(sorted.map(\.path))
        guard candidatePaths == authority.paths else {
            let missing = authority.paths.subtracting(candidatePaths).sorted()
            let unexpected = candidatePaths.subtracting(authority.paths).sorted()
            var faults: [String] = []
            if !missing.isEmpty {
                faults.append(
                    "tree omitted \(missing.count) repository-info path(s), first '\(missing[0])'")
            }
            if !unexpected.isEmpty {
                faults.append(
                    "tree added \(unexpected.count) path(s), first '\(unexpected[0])'")
            }
            throw CatalogError.malformedResponse(
                "the paginated tree for \(repo.repoID) at \(repo.revision) is not complete: "
                    + faults.joined(separator: "; "))
        }
        return HuggingFaceCompleteTree(
            files: sorted,
            completenessAuthorityIdentity: RepositoryFileSetIdentity.compute(authority.paths))
    }

    /// Fetch one small file at the pinned revision. Returns nil on 404, because
    /// "there is no index.json" is an answer and not a failure.
    public func fetchSmallFile(_ path: String, in repo: HuggingFaceRepoRef) async throws -> Data? {
        guard HuggingFaceRepoRef.isCanonicalRepositoryID(repo.repoID) else {
            throw CatalogError.malformedResponse(
                "'\(repo.repoID)' is not a canonical Hugging Face repository ID")
        }
        guard repo.isPinnedCommit else {
            throw CatalogError.unpinnedRevision(repoID: repo.repoID, given: repo.revision)
        }
        guard Self.isSafeRepositoryPath(path) else {
            throw CatalogError.malformedResponse(
                "'\(path)' is not a canonical repository-relative path")
        }
        let url = repo.resolveURL(path: path, endpoint: endpoint)
        let response = try await transport.data(for: request(url))
        if response.statusCode == 404 { return nil }
        guard response.statusCode == 200 else {
            throw CatalogError.transport(status: response.statusCode, url: url.absoluteString)
        }
        if let served = response.repoCommit, served != repo.revision {
            throw CatalogError.malformedResponse(
                "asked for \(repo.revision) and was served \(served) for \(path)")
        }
        return response.body
    }

    private func request(_ url: URL) -> HTTPRequest {
        var headers = ["accept": "application/json"]
        if let token = token?() { headers["authorization"] = "Bearer \(token)" }
        return HTTPRequest(url: url, headers: headers)
    }

    /// File-name authority from the pinned `model_info` response.
    ///
    /// This is intentionally a different endpoint and response shape from the
    /// paginated tree. It does not provide the digests we need, but Hugging
    /// Face documents `siblings` as the files constituting the model. Exact
    /// path-set agreement makes a stripped final-page `Link` fail closed.
    private func repositoryFileAuthority(
        in repo: HuggingFaceRepoRef
    ) async throws -> HuggingFaceRepositoryFileAuthority {
        var components = URLComponents(
            url: HuggingFaceRepoRef.url(
                endpoint,
                segments: ["api", "models", repo.repoID, "revision", repo.revision]),
            resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "expand", value: "sha"),
            URLQueryItem(name: "expand", value: "siblings"),
        ]
        guard let url = components.url else {
            throw CatalogError.malformedResponse(
                "could not build repository-info URL for \(repo.repoID)")
        }
        let response = try await transport.data(for: request(url))
        guard response.statusCode == 200 else {
            throw CatalogError.transport(status: response.statusCode, url: url.absoluteString)
        }
        guard let body = response.body,
            let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        else {
            throw CatalogError.malformedResponse(
                "repository info for \(repo.repoID) is not an object")
        }
        guard let servedRevision = object["sha"] as? String,
            servedRevision == repo.revision
        else {
            let served = object["sha"] as? String ?? "<no sha>"
            throw CatalogError.malformedResponse(
                "repository info for \(repo.repoID) served \(served), expected \(repo.revision)")
        }
        guard let siblings = object["siblings"] as? [[String: Any]] else {
            throw CatalogError.malformedResponse(
                "repository info for \(repo.repoID) carries no siblings file authority")
        }
        var paths = Set<String>()
        for sibling in siblings {
            guard let path = sibling["rfilename"] as? String,
                Self.isSafeRepositoryPath(path)
            else {
                throw CatalogError.malformedResponse(
                    "repository info for \(repo.repoID) carries an invalid sibling path")
            }
            guard paths.insert(path).inserted else {
                throw CatalogError.malformedResponse(
                    "repository info for \(repo.repoID) names '\(path)' more than once")
            }
        }
        return HuggingFaceRepositoryFileAuthority(paths: paths)
    }

    /// Decode one page of `/api/models/…/tree/…?recursive=1&expand=1`.
    ///
    /// Directories are dropped: they carry no bytes and no digest, and a
    /// "file" this package cannot digest is one it refuses to plan for.
    public static func decodePage(_ data: Data) throws -> [RepoFile] {
        guard let entries = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw CatalogError.malformedResponse("tree page is not an array of objects")
        }
        var files: [RepoFile] = []
        var seenPaths = Set<String>()
        for entry in entries {
            let type = entry["type"] as? String ?? "file"
            guard type == "file" else { continue }
            guard let path = entry["path"] as? String else {
                throw CatalogError.malformedResponse("a tree entry has no 'path'")
            }
            guard isSafeRepositoryPath(path) else {
                throw CatalogError.malformedResponse(
                    "tree path '\(path)' is not a canonical repository-relative path")
            }
            guard seenPaths.insert(path).inserted else {
                throw CatalogError.malformedResponse("the tree lists '\(path)' more than once")
            }
            let digest: FileDigest
            let size: UInt64
            if let rawLFS = entry["lfs"], !(rawLFS is NSNull) {
                guard let lfs = rawLFS as? [String: Any] else {
                    throw CatalogError.malformedResponse("'\(path)' carries an invalid lfs object")
                }
                guard let oid = lfs["oid"] as? String, isHexDigest(oid, digits: 64) else {
                    throw CatalogError.malformedResponse("'\(path)' carries an invalid lfs sha256")
                }
                digest = .sha256(hex: oid.lowercased())
                // The LFS object's size, not the pointer's. `size` at the top
                // level of an expanded entry is already the object size, but
                // `lfs.size` is the authoritative one when both are present.
                let rawSize = lfs.keys.contains("size") ? lfs["size"] : entry["size"]
                size = try decodeSize(rawSize, path: path)
            } else if let oid = entry["oid"] as? String {
                // Not LFS-backed: a plain git blob, whose id is a SHA-1 over a
                // header and the content. H3 publishes 16 payload files this
                // way. Treating every file as SHA-256 would leave them
                // unverifiable and — worse — silently unverified.
                guard isHexDigest(oid, digits: 40) else {
                    throw CatalogError.malformedResponse("'\(path)' carries an invalid git blob oid")
                }
                digest = .gitBlobSHA1(hex: oid.lowercased())
                size = try decodeSize(entry["size"], path: path)
            } else {
                throw CatalogError.malformedResponse("'\(path)' carries neither an oid nor an lfs oid")
            }
            files.append(
                RepoFile(
                    path: path, sizeBytes: size, digest: digest,
                    isPayload: RepoFileClassification.isPayload(path: path)))
        }
        return files
    }

    /// The repository path contract shared by planning and repair: a nonempty,
    /// `/`-separated relative path with no component that normalization could
    /// discard or move above the artifact root.
    static func isSafeRepositoryPath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.utf8.contains(0) else { return false }
        guard !path.hasPrefix("/"), !path.hasPrefix("\\") else { return false }

        // A drive-qualified path is absolute on Windows. Minirun currently
        // targets Apple platforms, but rejecting it here keeps the persisted
        // plan portable and makes the repo-relative contract literal.
        let prefix = Array(path.utf8.prefix(3))
        if prefix.count == 3,
            ((65...90).contains(prefix[0]) || (97...122).contains(prefix[0])),
            prefix[1] == 58, prefix[2] == 47 || prefix[2] == 92
        {
            return false
        }

        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        return !components.isEmpty
            && components.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    private static func decodeSize(_ value: Any?, path: String) throws -> UInt64 {
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            let size = UInt64(number.stringValue)
        else {
            throw CatalogError.malformedResponse("'\(path)' carries no valid integer size")
        }
        return size
    }

    private static func isHexDigest(_ value: String, digits: Int) -> Bool {
        let bytes = value.utf8
        guard bytes.count == digits else { return false }
        return bytes.allSatisfy {
            (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
        }
    }

    private struct HTTPOrigin: Equatable {
        let scheme: String
        let host: String
        let port: Int
    }

    private static func httpOrigin(of url: URL) -> HTTPOrigin? {
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
            let host = url.host?.lowercased(), !host.isEmpty
        else { return nil }
        let port = url.port ?? (scheme == "https" ? 443 : 80)
        return HTTPOrigin(scheme: scheme, host: host, port: port)
    }

    private static func sameOrigin(_ lhs: URL, _ rhs: URL) -> Bool {
        guard let left = httpOrigin(of: lhs), let right = httpOrigin(of: rhs) else { return false }
        return left == right
    }
}

struct HuggingFaceCompleteTree: Sendable, Equatable {
    let files: [RepoFile]
    let completenessAuthorityIdentity: String
}

private struct HuggingFaceRepositoryFileAuthority: Sendable {
    let paths: Set<String>
}

/// Stable identity of the independent repository-info path set.
///
/// This has a separate domain tag from the digest-plan identity: agreement of
/// the two values would be meaningless because one proves membership while
/// the other binds sizes and digests.
enum RepositoryFileSetIdentity {
    static func compute<S: Sequence>(_ paths: S) -> String where S.Element == String {
        var canonical = Data("minirun-hugging-face-siblings-v1".utf8)
        canonical.append(0x0A)
        for path in paths.sorted() {
            canonical.append(Data(String(path.utf8.count).utf8))
            canonical.append(0x3A)
            canonical.append(Data(path.utf8))
            canonical.append(0x0A)
        }
        return FileDigestComputer.sha256(of: canonical)
    }
}
