import Foundation

/// A per-token record of which experts a layer routed to.
///
/// Spec §11.3 requires every cache policy to be evaluated "on the same routing
/// trace and memory budget", which only means something if the trace is an
/// artefact rather than a function call: two policies compared on two different
/// random draws are not compared at all. So a trace is a file, with a seed and
/// a manifest, and the engine replays it.
///
/// Two forms, and the difference is deliberate:
///
/// * **pre-selected** — `selections` carries the ids. Synthetic locality
///   (uniform, zipf) lives here, because the point is to control locality
///   exactly, and running a router on inputs reverse-engineered to produce a
///   given zipf exponent would control nothing.
/// * **hidden-state surrogate** — `hiddenSeed` carries a seed instead, and the
///   engine routes with the layer's **real** router weights and noaux bias.
///   The ids are then a measured property of the checkpoint rather than an
///   assumption about it.
public struct RoutingTrace {

    public struct Generator: Codable {
        public let method: String
        public let distribution: String
        public let seed: UInt64
        public let zipfExponent: Double?
        /// Rank -> expert id. Without it a zipf trace would always make expert
        /// 0 the hottest, and "expert 0" is also the first expert in the
        /// container, so locality and layout would be confounded.
        public let rankPermutationSeed: UInt64?

        enum CodingKeys: String, CodingKey {
            case method, distribution, seed
            case zipfExponent = "zipf_exponent"
            case rankPermutationSeed = "rank_permutation_seed"
        }
    }

    public let name: String
    public let numExperts: Int
    public let expertsPerToken: Int
    public let tokens: Int
    public let generator: Generator
    /// Layer -> token -> expert ids. Empty for a hidden-state trace.
    public let selections: [Int: [[Int]]]
    /// Present when the engine must run the real router itself.
    public let hiddenSeed: UInt64?
    public let path: String?

    public var isPreSelected: Bool { !selections.isEmpty }

    public func ids(layer: Int, token: Int) -> [Int]? {
        guard let perToken = selections[layer], token < perToken.count else { return nil }
        return perToken[token]
    }

    /// How often each expert appears, for the layer. The pinned-hot-set policy
    /// is built from exactly this and nothing else.
    public func frequencies(layer: Int) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for row in selections[layer] ?? [] {
            for expert in row { counts[expert, default: 0] += 1 }
        }
        return counts
    }

    /// The `count` most frequently routed experts, most frequent first. Ties
    /// break on the expert id so the set is reproducible.
    public func hottestExperts(layer: Int, count: Int) -> [Int] {
        let counts = frequencies(layer: layer)
        return counts
            .sorted { ($0.value, -$0.key) > ($1.value, -$1.key) }
            .prefix(count)
            .map(\.key)
    }

    /// Locality summary, so a table row can say what the trace actually was
    /// rather than what it was asked to be.
    public struct Locality: Codable {
        public let layer: Int
        public let distinctExperts: Int
        public let selections: Int
        /// Share of all selections taken by the single hottest expert.
        public let topOneShare: Double
        /// Share taken by the hottest 5% of experts.
        public let topFivePercentShare: Double
        /// Shannon entropy of the selection distribution, in bits. `log2(E)`
        /// for a perfectly uniform trace over E experts.
        public let entropyBits: Double
        /// Fraction of a token's experts that the previous token also used.
        /// This is the number a per-token cache can actually exploit.
        public let consecutiveTokenOverlap: Double
    }

    public func locality(layer: Int) -> Locality? {
        guard let rows = selections[layer], !rows.isEmpty else { return nil }
        let counts = frequencies(layer: layer)
        let total = Double(rows.reduce(0) { $0 + $1.count })
        let sorted = counts.values.sorted(by: >)
        let topFive = max(1, Int((Double(numExperts) * 0.05).rounded(.up)))
        let entropy = counts.values.reduce(0.0) { partial, count in
            let p = Double(count) / total
            return partial - p * log2(p)
        }
        var overlap = 0.0
        for (previous, current) in zip(rows, rows.dropFirst()) {
            let shared = Set(previous).intersection(current).count
            overlap += Double(shared) / Double(current.count)
        }
        return Locality(
            layer: layer,
            distinctExperts: counts.count,
            selections: Int(total),
            topOneShare: Double(sorted.first ?? 0) / total,
            topFivePercentShare: Double(sorted.prefix(topFive).reduce(0, +)) / total,
            entropyBits: entropy,
            consecutiveTokenOverlap: rows.count > 1 ? overlap / Double(rows.count - 1) : 0)
    }

    // MARK: - Loading

    private struct Document: Decodable {
        let name: String
        let numExperts: Int
        let expertsPerToken: Int
        let tokens: Int
        let generator: Generator
        let layers: [String: [[Int]]]?
        let hiddenSeed: UInt64?

        enum CodingKeys: String, CodingKey {
            case name, generator, layers, tokens
            case numExperts = "num_experts"
            case expertsPerToken = "experts_per_token"
            case hiddenSeed = "hidden_seed"
        }
    }

    public init(path: String) throws {
        let document = try JSONDecoder().decode(
            Document.self, from: try Data(contentsOf: URL(fileURLWithPath: path)))

        var selections: [Int: [[Int]]] = [:]
        for (key, rows) in document.layers ?? [:] {
            guard let layer = Int(key) else {
                throw TinyK3Error.configuration("routing trace layer key '\(key)' is not an int")
            }
            guard rows.count == document.tokens else {
                throw TinyK3Error.configuration(
                    "layer \(layer) has \(rows.count) rows, header says \(document.tokens) tokens")
            }
            for (token, row) in rows.enumerated() {
                guard row.count == document.expertsPerToken else {
                    throw TinyK3Error.configuration(
                        "layer \(layer) token \(token) selects \(row.count) experts, "
                            + "header says \(document.expertsPerToken)")
                }
                guard row.allSatisfy({ $0 >= 0 && $0 < document.numExperts }) else {
                    throw TinyK3Error.configuration(
                        "layer \(layer) token \(token) selects an expert outside "
                            + "0..<\(document.numExperts)")
                }
                guard Set(row).count == row.count else {
                    throw TinyK3Error.configuration(
                        "layer \(layer) token \(token) selects the same expert twice; "
                            + "top-k is a set")
                }
            }
            selections[layer] = rows
        }
        guard !selections.isEmpty || document.hiddenSeed != nil else {
            throw TinyK3Error.configuration(
                "routing trace has neither pre-selected ids nor a hidden-state seed")
        }

        self.name = document.name
        self.numExperts = document.numExperts
        self.expertsPerToken = document.expertsPerToken
        self.tokens = document.tokens
        self.generator = document.generator
        self.selections = selections
        self.hiddenSeed = document.hiddenSeed
        self.path = path
    }

    public init(
        name: String,
        numExperts: Int,
        expertsPerToken: Int,
        tokens: Int,
        generator: Generator,
        selections: [Int: [[Int]]],
        hiddenSeed: UInt64? = nil
    ) {
        self.name = name
        self.numExperts = numExperts
        self.expertsPerToken = expertsPerToken
        self.tokens = tokens
        self.generator = generator
        self.selections = selections
        self.hiddenSeed = hiddenSeed
        self.path = nil
    }
}
