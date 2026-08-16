import Foundation

/// Somewhere a catalog can come from.
public protocol ModelCatalogSource: Sendable {
    func fetch() async throws -> ModelCatalogSnapshot
}

/// One repository returned by Hugging Face's indexed model listing.
///
/// This is discovery evidence, not artifact evidence. It deliberately carries
/// no byte counts, file counts, layout claim or runner promise: those facts are
/// available only after the pinned repository tree has been walked and
/// reconciled with the independent repository-info response.
public struct PublishedModelReference: Sendable, Equatable, Hashable, Identifiable {
    public let id: ModelID
    public let displayName: String
    public let repository: HuggingFaceRepoRef

    public init(id: ModelID, displayName: String, repository: HuggingFaceRepoRef) {
        self.id = id
        self.displayName = displayName
        self.repository = repository
    }
}

/// The two-stage Hugging Face surface used by the product browser.
///
/// Listing is one small indexed request (plus authorized pagination). Resolving
/// one selection performs the expensive complete-tree reconciliation required
/// before details, verification or download planning may trust it.
public protocol PublishedModelSource: ModelCatalogSource {
    func listPublishedModels() async throws -> [PublishedModelReference]
    func resolvePublishedModel(_ reference: PublishedModelReference) async throws
        -> ModelDescriptor
}

/// Builds a catalog by walking the published repositories.
///
/// The numbers a descriptor carries — payload bytes, file counts, largest file
/// — are *derived from the tree*, not read out of anybody's metadata. That is
/// the whole point: the bundled snapshot records what the repos held on
/// 2026-08-11, and a live fetch that disagrees is reported as a divergence
/// rather than quietly believed or quietly ignored.
public struct HuggingFaceCatalogSource: PublishedModelSource {
    /// Repository identities whose format this build can inspect.
    ///
    /// The list response still pins every refresh to one immutable commit and
    /// the verifier still proves that commit's complete tree. Recognition is
    /// deliberately not tied to one compiled commit, however: the owner can
    /// repair or extend a minirun repository without publishing a new App.
    /// Executability remains a later, stricter check against the verified
    /// index schema, tokenizer identity and rooted files.
    public static let recognizedRepos: [String: ModelID] = {
        var recognized: [String: ModelID] = [:]
        for descriptor in ModelCatalog.bundled.models {
            guard let repo = descriptor.source.repo,
                descriptor.architecture != .unrecognized,
                descriptor.layout != .unrecognized
            else { continue }
            precondition(
                recognized.updateValue(descriptor.id, forKey: repo.repoID) == nil,
                "the bundled catalog names one repository more than once")
        }
        return recognized
    }()

    private let author: String
    private let endpoint: URL
    private let transport: any HTTPTransport
    private let token: (@Sendable () -> String?)?

    public init(
        author: String = "nanguoyu",
        endpoint: URL = HuggingFaceEndpoint.default,
        transport: any HTTPTransport = URLSessionTransport(),
        token: (@Sendable () -> String?)? = nil
    ) {
        self.author = author
        self.endpoint = endpoint
        self.transport = transport
        self.token = token
    }

    public func fetch() async throws -> ModelCatalogSnapshot {
        let listed = try await listPublishedModels()
        var models: [ModelDescriptor] = []
        for reference in listed {
            models.append(try await resolvePublishedModel(reference))
        }

        return ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live, models: models.sorted { $0.id.rawValue < $1.id.rawValue })
    }

    /// Return the complete owned `*-minirun` repository list without walking
    /// any repository tree. The list still pins every row to a 40-hex commit;
    /// callers must resolve a row before presenting sizes or planning bytes.
    public func listPublishedModels() async throws -> [PublishedModelReference] {
        let listed = try await listRepositories()
        return try listed.map { repoID, revision in
            let repository = HuggingFaceRepoRef(repoID: repoID, revision: revision)
            guard repository.isPinnedCommit else {
                throw CatalogError.unpinnedRevision(repoID: repoID, given: revision)
            }
            return Self.reference(for: repository)
        }
        .sorted { lhs, rhs in
            lhs.displayName.localizedStandardCompare(rhs.displayName) == .orderedAscending
        }
    }

    /// Resolve exactly one listed repository into a descriptor whose counts
    /// and sizes came from a complete, independently reconciled pinned tree.
    public func resolvePublishedModel(_ reference: PublishedModelReference) async throws
        -> ModelDescriptor
    {
        let repository = reference.repository
        guard isCandidateRepositoryID(repository.repoID) else {
            throw CatalogError.malformedResponse(
                "\(repository.repoID) is not an owned *-minirun repository")
        }
        guard repository.isPinnedCommit else {
            throw CatalogError.unpinnedRevision(
                repoID: repository.repoID, given: repository.revision)
        }
        let canonical = Self.reference(for: repository)
        guard canonical == reference else {
            throw CatalogError.malformedResponse(
                "the published model reference for \(repository.repoID) changed identity")
        }
        let client = HuggingFaceTreeClient(
            transport: transport, endpoint: endpoint, token: token)
        let files = try await client.files(in: repository)
        guard !files.isEmpty else {
            throw CatalogError.malformedResponse(
                "\(repository.repoID) at \(repository.revision) lists no files")
        }
        if let id = Self.recognizedRepos[repository.repoID],
            let previous = ModelCatalog.bundled.descriptor(id)
        {
            return Self.descriptor(
                id: previous.id, layout: previous.layout,
                architecture: previous.architecture, repo: repository, files: files,
                carryingOver: previous)
        }
        return Self.unrecognizedDescriptor(repo: repository, files: files)
    }

    private static func reference(for repository: HuggingFaceRepoRef)
        -> PublishedModelReference
    {
        if let id = recognizedRepos[repository.repoID],
            let previous = ModelCatalog.bundled.descriptor(id)
        {
            return PublishedModelReference(
                id: id, displayName: previous.displayName, repository: repository)
        }
        let repositoryName = repository.repoID.split(separator: "/").last.map(String.init)
            ?? repository.repoID
        return PublishedModelReference(
            id: .discoveredHuggingFaceRepository(repository.repoID),
            displayName: String(repositoryName.dropLast("-minirun".count)),
            repository: repository)
    }

    /// `GET /api/models?author=…&expand=sha`, reduced to repo id → pinned commit.
    ///
    /// The Hub's indexed listing always returns `id`; `expand=sha` asks for the
    /// one additional field discovery needs and avoids `full=true`, which also
    /// returns tags and every sibling. A catalogue built without `sha` would
    /// have nothing immutable to pin and is refused.
    private func listRepositories() async throws -> [String: String] {
        var components = URLComponents(
            url: endpoint.appendingPathComponent("api/models"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "author", value: author),
            URLQueryItem(name: "expand", value: "sha"),
        ]
        guard let url = components.url else {
            throw CatalogError.malformedResponse("could not build the model list URL")
        }
        var headers = ["accept": "application/json"]
        if let token = token?() { headers["authorization"] = "Bearer \(token)" }
        var out: [String: String] = [:]
        var next: URL? = url
        var seenPages = Set<String>()
        while let pageURL = next {
            guard seenPages.count < 1_024 else {
                throw CatalogError.malformedResponse(
                    "the model list exceeded 1024 pagination pages")
            }
            guard let identity = authorizedListPageIdentity(pageURL, firstPage: url) else {
                throw CatalogError.malformedResponse(
                    "the model list returned an unauthorized next page: \(pageURL.absoluteString)")
            }
            guard seenPages.insert(identity).inserted else {
                throw CatalogError.malformedResponse(
                    "the model list paginated back to \(pageURL.absoluteString)")
            }
            let response = try await transport.data(
                for: HTTPRequest(url: pageURL, headers: headers))
            guard response.statusCode == 200 else {
                throw CatalogError.transport(
                    status: response.statusCode, url: pageURL.absoluteString)
            }
            guard let body = response.body,
                let entries = try? JSONSerialization.jsonObject(with: body) as? [[String: Any]]
            else {
                throw CatalogError.malformedResponse(
                    "the model list page is not an array of objects")
            }
            for entry in entries {
                guard let id = entry["id"] as? String else { continue }
                // The author publishes more than the minirun repos. The author
                // and suffix checks keep that unrelated set out, while
                // deliberately retaining new minirun repositories that this
                // build has never seen before.
                guard isCandidateRepositoryID(id) else { continue }
                guard let sha = entry["sha"] as? String else {
                    throw CatalogError.unpinnedRevision(
                        repoID: id, given: "<no sha in the listing>")
                }
                if let existing = out[id] {
                    guard existing == sha else {
                        throw CatalogError.malformedResponse(
                            "the model list names \(id) more than once with different commits")
                    }
                    continue
                }
                out[id] = sha
            }
            let following: URL?
            switch response.nextPage {
            case .absent:
                following = nil
            case .url(let url):
                following = url
            case .malformed(let value):
                throw CatalogError.malformedResponse(
                    "the model list returned a malformed next page: \(value)")
            }
            if let following,
                authorizedListPageIdentity(following, firstPage: url) == nil
            {
                // Validate before constructing a request so the bearer token
                // cannot follow an origin or query chosen by a response.
                throw CatalogError.malformedResponse(
                    "the model list returned an unauthorized next page: \(following.absoluteString)")
            }
            next = following
        }
        return out
    }

    /// Canonical identity for one authorized model-list page. Pagination may
    /// add cursor fields, but it cannot leave the endpoint-derived path or
    /// change/duplicate the author/expansion authority carried by the first request.
    private func authorizedListPageIdentity(_ candidate: URL, firstPage: URL) -> String? {
        guard let candidateComponents = URLComponents(
                url: candidate, resolvingAgainstBaseURL: false),
            let firstComponents = URLComponents(url: firstPage, resolvingAgainstBaseURL: false),
            let candidateScheme = candidateComponents.scheme?.lowercased(),
            let firstScheme = firstComponents.scheme?.lowercased(),
            candidateScheme == "http" || candidateScheme == "https",
            candidateScheme == firstScheme,
            let candidateHost = candidateComponents.host?.lowercased(),
            let firstHost = firstComponents.host?.lowercased(),
            candidateHost == firstHost,
            let candidatePort = effectivePort(candidateComponents),
            let firstPort = effectivePort(firstComponents),
            candidatePort == firstPort,
            candidateComponents.user == nil, candidateComponents.password == nil,
            candidateComponents.fragment == nil,
            candidateComponents.percentEncodedPath == firstComponents.percentEncodedPath
        else { return nil }

        let queryItems = candidateComponents.queryItems ?? []
        let authors = queryItems.filter { $0.name == "author" }
        let expansions = queryItems.filter { $0.name == "expand" }
        guard authors.count == 1, authors[0].value == author,
            expansions.count == 1, expansions[0].value == "sha"
        else { return nil }

        // Hugging Face listing continuations have used cursor-based and
        // page/limit forms. Permit pagination controls only. Forwarding the
        // bearer token to a same-origin URL whose response-chosen query changes
        // expansion or authentication semantics would still be an authority
        // escalation.
        let allowedNames = Set(["author", "expand", "cursor", "p", "limit"])
        guard queryItems.allSatisfy({ allowedNames.contains($0.name) }) else { return nil }
        for name in ["cursor", "p", "limit"] {
            guard queryItems.filter({ $0.name == name }).count <= 1 else { return nil }
        }
        if let cursor = queryItems.first(where: { $0.name == "cursor" }) {
            guard let value = cursor.value, !value.isEmpty else { return nil }
        }
        var normalizedNumericValues: [String: String] = [:]
        for name in ["p", "limit"] {
            if let item = queryItems.first(where: { $0.name == name }) {
                guard let value = item.value, !value.isEmpty,
                    value.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }),
                    let numericValue = UInt64(value)
                else { return nil }
                normalizedNumericValues[name] = String(numericValue)
            }
        }

        // JSON gives an unambiguous canonical representation of decoded query
        // names/values; concatenating `&` and `=` can alias distinct cursors.
        let normalizedPairs = queryItems
            .map { [$0.name, normalizedNumericValues[$0.name] ?? $0.value ?? ""] }
            .sorted {
                if $0[0] == $1[0] { return $0[1] < $1[1] }
                return $0[0] < $1[0]
            }
        guard let normalizedData = try? JSONSerialization.data(
                withJSONObject: normalizedPairs, options: [.sortedKeys]),
            let normalizedQuery = String(data: normalizedData, encoding: .utf8)
        else { return nil }
        return "\(candidateScheme)://\(candidateHost):\(candidatePort)"
            + "\(candidateComponents.percentEncodedPath)?\(normalizedQuery)"
    }

    private func effectivePort(_ components: URLComponents) -> Int? {
        if let port = components.port { return port }
        switch components.scheme?.lowercased() {
        case "http": return 80
        case "https": return 443
        default: return nil
        }
    }

    /// Exactly `author/name-minirun`, with no path component that a URL or
    /// filesystem could normalise away. The API query is not treated as proof
    /// that every returned object really belongs to the requested author.
    private func isCandidateRepositoryID(_ repoID: String) -> Bool {
        guard HuggingFaceRepoRef.isCanonicalRepositoryID(repoID) else { return false }
        let components = repoID.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2, components[0] == Substring(author) else { return false }
        let name = components[1]
        return name.hasSuffix("-minirun") && name.count > "-minirun".count
    }

    /// Assemble a descriptor from tree-derived numbers, keeping the
    /// operator-facing text and the measured minimum budget from the bundled
    /// copy. Bytes are measurable from here; a minimum budget is not — it is a
    /// record of a run that happened, and the network does not know about those.
    static func descriptor(
        id: ModelID, layout: ArtifactLayout, architecture: ModelArchitecture,
        repo: HuggingFaceRepoRef, files: [RepoFile], carryingOver previous: ModelDescriptor?,
        displayName: String? = nil,
        notes: [String]? = nil
    ) -> ModelDescriptor {
        let payload = files.filter(\.isPayload)
        let metadata = files.filter { !$0.isPayload }
        return ModelDescriptor(
            id: id,
            displayName: displayName ?? previous?.displayName ?? repo.repoID,
            architecture: architecture,
            layout: layout,
            source: .huggingFaceRepo(repo),
            payloadBytes: saturatingByteSum(payload),
            metadataBytes: saturatingByteSum(metadata),
            payloadFileCount: payload.count,
            metadataFileCount: metadata.count,
            largestFileBytes: files.map(\.sizeBytes).max() ?? 0,
            minimumBudgetBytes: previous?.minimumBudgetBytes,
            runner: previous?.runner ?? .none,
            licenseName: previous?.licenseName ?? "unstated",
            licenseAcknowledgementRequired: previous?.licenseAcknowledgementRequired ?? true,
            notes: notes ?? previous?.notes ?? [])
    }

    private static func unrecognizedDescriptor(
        repo: HuggingFaceRepoRef, files: [RepoFile]
    ) -> ModelDescriptor {
        let repositoryName = repo.repoID.split(separator: "/").last.map(String.init) ?? repo.repoID
        let displayName = String(repositoryName.dropLast("-minirun".count))
        return descriptor(
            id: .discoveredHuggingFaceRepository(repo.repoID),
            layout: .unrecognized,
            architecture: .unrecognized,
            repo: repo,
            files: files,
            carryingOver: nil,
            displayName: displayName,
            notes: [
                "Discovered on Hugging Face; this build does not recognize its architecture or artifact layout."
            ])
    }

    private static func saturatingByteSum(_ files: [RepoFile]) -> UInt64 {
        var total: UInt64 = 0
        for file in files {
            let (sum, overflow) = total.addingReportingOverflow(file.sizeBytes)
            if overflow { return .max }
            total = sum
        }
        return total
    }
}
