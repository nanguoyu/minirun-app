import Foundation

/// A successful live refresh and whether it also became durable for a later
/// offline launch. Cache persistence is secondary: callers must not discard a
/// fresh live list merely because the local cache could not be updated.
public struct CatalogRefreshResult: Sendable, Equatable {
    public let snapshot: ModelCatalogSnapshot
    public let cacheWarning: String?

    public init(snapshot: ModelCatalogSnapshot, cacheWarning: String?) {
        self.snapshot = snapshot
        self.cacheWarning = cacheWarning
    }
}

/// The catalog, with a cache in front of it and a bundled copy behind it.
public actor ModelCatalog {
    private let live: any ModelCatalogSource
    private let bundledSnapshot: ModelCatalogSnapshot
    private let cache: (any CatalogCache)?
    private var memo: ModelCatalogSnapshot?

    public init(
        live: any ModelCatalogSource = HuggingFaceCatalogSource(),
        bundled: ModelCatalogSnapshot = ModelCatalog.bundled,
        cache: (any CatalogCache)? = FileCatalogCache.default
    ) {
        self.live = live
        self.bundledSnapshot = bundled
        self.cache = cache
    }

    /// Never throws.
    ///
    /// A model list is not worth failing a launch over: the bundled copy is
    /// always there and always honest about being the bundled copy, so a device
    /// on a plane shows a catalog labelled `bundled` rather than an error
    /// dialog. Callers that *asked* for the network want ``refresh()``, which
    /// does throw.
    public func snapshot(_ policy: RefreshPolicy = .preferCachedWithin(3600)) async
        -> ModelCatalogSnapshot
    {
        switch policy {
        case .bundledOnly:
            return bundledSnapshot
        case .preferCachedWithin(let maximumAge):
            if let memo, Date().timeIntervalSince(memo.generatedAt) <= maximumAge { return memo }
            if let cached = try? cache?.load(),
                Date().timeIntervalSince(cached.generatedAt) <= maximumAge
            {
                memo = cached.labelled(.cached)
                return memo!
            }
            fallthrough
        case .forceLive:
            if let fresh = try? await refresh() { return fresh }
            if let cached = try? cache?.load() { return cached.labelled(.cached) }
            return bundledSnapshot
        }
    }

    /// Throws. Used when the operator asked for a refresh and is owed the
    /// reason it did not happen.
    @discardableResult
    public func refresh() async throws -> ModelCatalogSnapshot {
        (try await refreshResult()).snapshot
    }

    /// Fetch a live snapshot and separately report whether the offline cache
    /// accepted it. A cache write is not allowed to turn usable current data
    /// into a refresh failure, but it is also not silently swallowed.
    @discardableResult
    public func refreshResult() async throws -> CatalogRefreshResult {
        let fresh = try await live.fetch()
        memo = fresh
        guard let cache else {
            return CatalogRefreshResult(snapshot: fresh, cacheWarning: nil)
        }
        do {
            try cache.save(fresh)
            return CatalogRefreshResult(snapshot: fresh, cacheWarning: nil)
        } catch {
            return CatalogRefreshResult(snapshot: fresh, cacheWarning: String(describing: error))
        }
    }

    public func descriptor(_ id: ModelID) async throws -> ModelDescriptor {
        if let found = await snapshot().descriptor(id) { return found }
        throw CatalogError.unknownModel(id)
    }

    /// The fast, indexed remote-model list used by a storefront or picker.
    ///
    /// A source that supports two-stage discovery performs no tree walks here.
    /// The fallback preserves source compatibility for custom catalog sources,
    /// but production's Hugging Face source always takes the lightweight path.
    public func publishedModels() async throws -> [PublishedModelReference] {
        if let source = live as? any PublishedModelSource {
            return try await source.listPublishedModels()
        }
        let fetched = try await live.fetch()
        return fetched.models.compactMap(Self.publishedReference)
            .sorted {
                $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending
            }
    }

    /// Resolve and durably merge one selected remote model. Other cached
    /// descriptors are retained so an offline, verified local model does not
    /// disappear merely because the user inspected a different repository.
    @discardableResult
    public func resolvePublishedModel(_ reference: PublishedModelReference) async throws
        -> CatalogRefreshResult
    {
        let descriptor: ModelDescriptor
        if let source = live as? any PublishedModelSource {
            descriptor = try await source.resolvePublishedModel(reference)
        } else {
            let fetched = try await live.fetch()
            guard let found = fetched.models.first(where: {
                $0.id == reference.id && $0.source.repo == reference.repository
            }) else {
                throw CatalogError.unknownModel(reference.id)
            }
            descriptor = found
        }
        guard descriptor.id == reference.id,
            descriptor.source.repo == reference.repository
        else {
            throw CatalogError.malformedResponse(
                "resolved descriptor does not match \(reference.repository.repoID) "
                    + "at \(reference.repository.revision)")
        }

        let existing = memo ?? (try? cache?.load()) ?? bundledSnapshot
        let repositoryID = reference.repository.repoID
        var models = existing.models.filter { model in
            guard model.id != descriptor.id else { return false }
            return model.source.repo?.repoID != repositoryID
        }
        models.append(descriptor)
        let merged = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live,
            models: models.sorted { $0.id.rawValue < $1.id.rawValue })
        memo = merged
        guard let cache else {
            return CatalogRefreshResult(snapshot: merged, cacheWarning: nil)
        }
        do {
            try cache.save(merged)
            return CatalogRefreshResult(snapshot: merged, cacheWarning: nil)
        } catch {
            return CatalogRefreshResult(
                snapshot: merged, cacheWarning: String(describing: error))
        }
    }

    private nonisolated static func publishedReference(
        _ descriptor: ModelDescriptor
    ) -> PublishedModelReference? {
        guard let repository = descriptor.source.repo else { return nil }
        return PublishedModelReference(
            id: descriptor.id, displayName: descriptor.displayName, repository: repository)
    }

    /// Divergence of the current snapshot from the bundled one, which is the
    /// only comparison worth making automatically: the bundled numbers are what
    /// this build's sizing, budgeting and free-space refusals were written
    /// against.
    public func divergenceFromBundled(_ policy: RefreshPolicy = .forceLive) async
        -> [CatalogDivergence]
    {
        await snapshot(policy).divergence(from: bundledSnapshot)
    }

    // MARK: - The copy compiled into this build

    /// The snapshot shipped in the bundle.
    ///
    /// Loaded once and held: it is read on every fallback path, including ones
    /// that run while the network is timing out, and a resource read that can
    /// fail is not something a fallback should depend on twice.
    public nonisolated static var bundled: ModelCatalogSnapshot { BundledCatalog.snapshot }
}

enum BundledCatalog {
    static let snapshot: ModelCatalogSnapshot = load()

    private static func load() -> ModelCatalogSnapshot {
        guard let url = resourceURL() else {
            fatalError(
                "MinirunKit's bundled model-catalog.json is missing from the resource bundle; "
                    + "the target's `resources:` declaration and the file must agree.")
        }
        do {
            let data = try Data(contentsOf: url)
            return try MinirunKitJSON.decoder().decode(ModelCatalogSnapshot.self, from: data)
        } catch {
            fatalError("MinirunKit's bundled model-catalog.json could not be read: \(error)")
        }
    }

    private static func resourceURL() -> URL? {
        // `.copy` of a single file lands it at the bundle root; the
        // subdirectory form is checked too so the declaration can change to
        // `.copy("Resources")` without this becoming a mystery.
        Bundle.module.url(forResource: "model-catalog", withExtension: "json")
            ?? Bundle.module.url(
                forResource: "model-catalog", withExtension: "json", subdirectory: "Resources")
    }
}
