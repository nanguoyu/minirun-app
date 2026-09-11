import Foundation

/// The address side of DeepSeek V4.1 Flash's Engram: which 24 rows a token reads.
///
/// Per token and per Engram block, three n-grams (orders 2, 3 and 4) are each
/// hashed by eight heads, giving 24 row indices into that block's
/// 384-million-row table. The whole computation is integer, so agreement with
/// the reference means *equal*, not close, and this type is checked that way
/// against indices dumped from `NgramHashState` itself.
///
/// ## The equation
///
/// For a token at absolute position `p`, with `c[i]` the compressed id at
/// position `i`:
///
/// ```text
/// t[s]    = c[p - s], or the pad class when p < s or any of c[p-s...p] is dead
/// r[1]    = t[0] * m[1] ^ t[1] * m[1]         (the 2-gram)
/// r[k]    = r[k-1] ^ t[k] * m[k]              (the (k+1)-gram)
/// row     = r[k] % prime[k-1][head] + offset[(k-1) * 8 + head]
/// ```
///
/// Three properties of that are easy to get wrong and are asserted here:
///
/// - **The multiplier is indexed by lookback, not by n-gram order.** A 4-gram
///   reuses the same three products the 3-gram used, XORed with one more.
/// - **`blocked` is sticky.** Once a lookback runs off the start of the
///   sequence or hits a dead token, every longer lookback is padded too, so an
///   n-gram never spans a masked span.
/// - **Nothing overflows.** The reference bounds its multipliers so that
///   `compressedID * multiplier` stays inside Int64; ``init(map:)`` re-derives
///   that bound from the shipped constants and refuses a map that breaks it,
///   because a wrapped product in Swift is a different row, silently.
///
/// ## Prefill and decode are the same code
///
/// The reference keeps the compressed ids of the whole context in a cache and
/// gathers `p`, `p-1`, `p-2`, `p-3` out of it, which is what makes a prompt
/// processed in one call and the same prompt processed one token at a time
/// produce identical indices. Only four positions are ever read, so this keeps
/// four — see ``recent`` — and carries `position` across calls.
public struct DeepSeekV41EngramHasher {
    /// A masked position: an image span in the multimodal path. Text-only
    /// callers never produce one.
    public static let dead: Int64 = -1

    /// `[layer][column]` row indices for one token.
    public typealias TokenRows = [[Int]]

    private let compressed: [Int32]
    private let padClass: Int64
    private let layerCount: Int
    private let columnCount: Int
    private let maxNgramSize: Int
    private let multipliers: [[Int64]]
    /// `[layer][column]`, the primes and offsets already flattened in draw
    /// order so the inner loop is one array read rather than two divisions.
    private let columnPrimes: [[Int64]]
    private let columnOffsets: [[Int]]
    private let rowCounts: [Int]

    /// The last `maxNgramSize` compressed ids, indexed by `position % size`.
    ///
    /// Four is exactly enough and not a heuristic: the reference gathers
    /// `max(p - s, 0)` for `s` in `0..<4`. For `p >= 4` every one of those is
    /// in `p-3...p`; for `p < 4` the clamp can only reach position 0, which is
    /// still one of the last four positions written.
    private var recent: [Int64]
    /// The absolute position of the next token. Prefill sets it to the prompt
    /// length; decode carries it forward one token at a time.
    public private(set) var position: Int

    public init(map: DeepSeekV41EngramTokenMap.Map) throws {
        try self.init(
            compressedIDs: map.compressedIDs,
            classCount: map.classCount)
    }

    public init(compressedIDs: [Int32], classCount: Int) throws {
        let constants = DeepSeekV41EngramConstants.self
        guard !compressedIDs.isEmpty else {
            throw DeepSeekV4Error.configuration(
                "the Engram hasher needs a non-empty compressed token map")
        }
        guard classCount == constants.compressedVocabularySize else {
            throw DeepSeekV4Error.configuration(
                "the compressed token map has \(classCount) classes; the shipped "
                    + "Engram multipliers are derived from "
                    + "\(constants.compressedVocabularySize)")
        }
        guard constants.padTokenID >= 0, constants.padTokenID < compressedIDs.count
        else {
            throw DeepSeekV4Error.configuration(
                "the Engram pad token id \(constants.padTokenID) is outside the "
                    + "\(compressedIDs.count)-token map")
        }
        let highest = compressedIDs.max().map(Int64.init) ?? 0
        guard highest < Int64(classCount) else {
            throw DeepSeekV4Error.configuration(
                "the compressed token map addresses class \(highest) but declares "
                    + "\(classCount)")
        }
        let layers = constants.layerIDs.count
        guard constants.multipliers.count == layers,
            constants.primes.count == layers,
            constants.bucketOffsets.count == layers,
            constants.rowCounts.count == layers,
            constants.multipliers.allSatisfy({ $0.count == constants.maxNgramSize }),
            constants.primes.allSatisfy({
                $0.count == constants.maxNgramSize - 1
                    && $0.allSatisfy { $0.count == constants.headCount }
            }),
            constants.bucketOffsets.allSatisfy({ $0.count == constants.hashColumnCount })
        else {
            throw DeepSeekV4Error.configuration(
                "the shipped Engram constants are not the declared shape")
        }
        for multiplier in constants.multipliers.joined() {
            guard multiplier > 0, multiplier % 2 == 1,
                !multiplier.multipliedReportingOverflow(by: highest).overflow
            else {
                throw DeepSeekV4Error.configuration(
                    "Engram multiplier \(multiplier) is not an odd Int64 that keeps "
                        + "class \(highest) inside Int64")
            }
        }

        self.compressed = compressedIDs
        self.padClass = Int64(compressedIDs[constants.padTokenID])
        self.layerCount = layers
        self.columnCount = constants.hashColumnCount
        self.maxNgramSize = constants.maxNgramSize
        self.multipliers = constants.multipliers
        self.columnPrimes = constants.primes.map { layer in
            layer.joined().map(Int64.init)
        }
        self.columnOffsets = constants.bucketOffsets
        self.rowCounts = constants.rowCounts
        self.recent = [Int64](repeating: 0, count: constants.maxNgramSize)
        self.position = 0
    }

    /// Forget the context. The next token hashes as position 0 again.
    public mutating func reset() {
        position = 0
        for index in recent.indices { recent[index] = 0 }
    }

    /// The row indices for the next `tokenIDs`, `[token][layer][column]`.
    ///
    /// One call for the whole prompt and one call per token produce the same
    /// indices; that equality is the point of carrying ``position`` and
    /// ``recent`` rather than rebuilding from a token list.
    ///
    /// `alive` marks the positions that take part in an n-gram. Text-only
    /// callers pass nil, which is not the same as passing all-true only in that
    /// it skips the allocation — the reference's `token_mask` is likewise
    /// absent rather than true for a text batch.
    public mutating func rows(
        forTokens tokenIDs: [Int],
        alive: [Bool]? = nil
    ) throws -> [TokenRows] {
        if let alive, alive.count != tokenIDs.count {
            throw DeepSeekV4Error.configuration(
                "the Engram liveness mask has \(alive.count) entries for "
                    + "\(tokenIDs.count) tokens")
        }
        var result = [TokenRows]()
        result.reserveCapacity(tokenIDs.count)
        var lookback = [Int64](repeating: 0, count: maxNgramSize)

        for (index, tokenID) in tokenIDs.enumerated() {
            guard tokenID >= 0, tokenID < compressed.count else {
                throw DeepSeekV4Error.configuration(
                    "token id \(tokenID) is outside the \(compressed.count)-token "
                        + "Engram compressed map")
            }
            let isAlive = alive?[index] ?? true
            let current = isAlive ? Int64(compressed[tokenID]) : Self.dead
            let absolute = position
            recent[absolute % maxNgramSize] = current

            var blocked = false
            for shift in 0..<maxNgramSize {
                let source = recent[Swift.max(absolute - shift, 0) % maxNgramSize]
                blocked = blocked || absolute < shift || source == Self.dead
                lookback[shift] = blocked ? padClass : source
            }

            var perLayer = TokenRows()
            perLayer.reserveCapacity(layerCount)
            for layer in 0..<layerCount {
                let layerMultipliers = multipliers[layer]
                let primes = columnPrimes[layer]
                let offsets = columnOffsets[layer]
                var columns = [Int](repeating: 0, count: columnCount)
                // Plain `*`, not `&*`: `init` proved no product here can
                // overflow, and a trap on a broken proof is better than a
                // wrapped product, which is just a different row.
                var rolling = lookback[0] * layerMultipliers[0]
                for order in 1..<maxNgramSize {
                    rolling ^= lookback[order] * layerMultipliers[order]
                    let base = (order - 1) * DeepSeekV41EngramConstants.headCount
                    for head in 0..<DeepSeekV41EngramConstants.headCount {
                        let column = base + head
                        columns[column] =
                            Int(rolling % primes[column]) + offsets[column]
                    }
                }
                perLayer.append(columns)
            }
            try validate(perLayer, token: index)
            result.append(perLayer)
            position = absolute + 1
        }
        return result
    }

    /// Every index must land inside its own layer's table.
    ///
    /// Cheap — 48 comparisons per token — and the failure it catches is a read
    /// of 4 KiB from somewhere else in a 98 GB file, which produces plausible
    /// numbers rather than a crash.
    private func validate(_ perLayer: TokenRows, token: Int) throws {
        for (layer, columns) in perLayer.enumerated() {
            for (column, index) in columns.enumerated() {
                guard index >= 0, index < rowCounts[layer] else {
                    throw DeepSeekV4Error.configuration(
                        "Engram row \(index) for token \(token), block "
                            + "\(DeepSeekV41EngramConstants.layerIDs[layer]), "
                            + "column \(column) is outside the table's "
                            + "\(rowCounts[layer]) rows")
                }
            }
        }
    }
}
