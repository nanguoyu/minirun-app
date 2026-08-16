import Foundation

/// The checkpoint's tiktoken vocabulary: decode always, encode when it can.
///
/// `tiktoken.model` is the standard tiktoken BPE file — one `base64(bytes) rank`
/// pair per line, 163,584 of them — followed by 256 reserved special-token ids
/// that exist only in the tokenizer wrapper and have no byte string. 163,584 +
/// 256 = 163,840, the model's vocabulary size, which this type checks.
///
/// ## Decode is required, encode is not
///
/// M6's exit criterion is a deterministic token, and a token id nobody can read
/// is a poor deliverable — so `decode` is mandatory and is what turns a greedy
/// id into text. `encode` is the optional half: the committed trace bundles
/// carry the exact input ids, so a port can be gated end-to-end without ever
/// running the BPE. It is implemented here anyway, and validated against those
/// committed ids, so the fallback is a measured choice rather than an
/// assumption.
public final class TikTokenVocabulary {

    /// `bytes -> rank`, the mergeable ranks.
    private let ranks: [Data: Int]
    /// `rank -> bytes`, for decode.
    private let tokens: [Data]
    /// Ids at or above this are the wrapper's reserved special tokens; they
    /// have no byte string in `tiktoken.model`.
    public let baseTokenCount: Int
    public let specialTokenCount: Int

    public var vocabularySize: Int { baseTokenCount + specialTokenCount }

    /// `TikTokenTokenizer.num_reserved_special_tokens`.
    public static let reservedSpecialTokens = 256

    /// The tokenizer's own `pat_str`, joined from
    /// `tokenization_kimi.py:54-66`.
    ///
    /// The character-class intersections (`[\p{Lu}...&&[^\p{Han}]]`) are Unicode
    /// set operations. tiktoken evaluates them through Rust's `regex`; ICU —
    /// which is what `NSRegularExpression` is — supports the same `&&` syntax,
    /// and both engines are leftmost-first over the alternation, so the split
    /// agrees. That agreement is asserted against the committed prompt ids
    /// rather than argued.
    public static let pattern = [
        #"[\p{Han}]+"#,
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#,
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#,
        #"\p{N}{1,3}"#,
        #" ?[^\s\p{L}\p{N}]+[\r\n]*"#,
        #"\s*[\r\n]+"#,
        #"\s+(?!\S)"#,
        #"\s+"#,
    ].joined(separator: "|")

    private let splitter: NSRegularExpression?

    public convenience init(contentsOf path: String) throws {
        try self.init(data: Data(contentsOf: URL(fileURLWithPath: path)))
    }

    /// Parse a verified in-memory tokenizer without reopening its pathname.
    ///
    /// Product code uses this after a bounded rooted-descriptor read. The file
    /// convenience above remains for fixtures and command-line callers.
    public init(data: Data) throws {
        guard let text = String(data: data, encoding: .utf8) else {
            throw TinyK3Error.tokenizer("tiktoken.model is not valid UTF-8")
        }
        var ranks: [Data: Int] = [:]
        var tokens: [Data] = []
        ranks.reserveCapacity(200_000)

        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let parts = line.split(separator: " ", maxSplits: 1)
            guard parts.count == 2, let rank = Int(parts[1]),
                let bytes = Data(base64Encoded: String(parts[0]))
            else {
                throw TinyK3Error.tokenizer("malformed tiktoken.model line: '\(line)'")
            }
            guard rank == tokens.count else {
                throw TinyK3Error.tokenizer(
                    "tiktoken.model ranks are not contiguous: expected \(tokens.count), got \(rank)")
            }
            ranks[bytes] = rank
            tokens.append(bytes)
        }
        guard !tokens.isEmpty else { throw TinyK3Error.tokenizer("tiktoken.model is empty") }

        self.ranks = ranks
        self.tokens = tokens
        self.baseTokenCount = tokens.count
        self.specialTokenCount = Self.reservedSpecialTokens
        self.splitter = try? NSRegularExpression(pattern: Self.pattern, options: [])
    }

    /// Whether the BPE splitter compiled on this platform. When it did not,
    /// ``encode(_:)`` throws and the caller falls back to fixture-provided ids.
    public var canEncode: Bool { splitter != nil }

    // MARK: - Decode

    /// The raw bytes of one token, or `nil` for a reserved special id.
    public func bytes(of id: Int) -> Data? {
        guard id >= 0, id < baseTokenCount else { return nil }
        return tokens[id]
    }

    /// Ids to text.
    ///
    /// Byte strings are concatenated before decoding, because a BPE token can
    /// end mid-codepoint — decoding token by token would produce replacement
    /// characters wherever a multi-byte character straddles a token boundary,
    /// which is common in the Chinese prompts.
    public func decode(_ ids: [Int]) -> String {
        var buffer = Data()
        var out = ""
        func flush() {
            if !buffer.isEmpty {
                out += String(decoding: buffer, as: UTF8.self)
                buffer.removeAll(keepingCapacity: true)
            }
        }
        for id in ids {
            if let bytes = bytes(of: id) {
                buffer.append(bytes)
            } else {
                flush()
                out += specialTokenName(id)
            }
        }
        flush()
        return out
    }

    /// Reserved special ids have no byte string; they are rendered by name so a
    /// decoded string never silently drops a token.
    public func specialTokenName(_ id: Int) -> String {
        guard id >= baseTokenCount, id < vocabularySize else { return "<|invalid_token_\(id)|>" }
        if let token = K3SpecialToken(rawValue: id - baseTokenCount) {
            return token.name
        }
        return "<|reserved_token_\(id)|>"
    }

    // MARK: - Encode

    /// The marker every K3 control token is written with, e.g. `<|open|>`.
    ///
    /// The reserved ids 163,584–163,839 have no byte string in `tiktoken.model`,
    /// so no sequence of merges can ever produce one. Any occurrence of this
    /// marker in text handed to ``encode(_:)`` is therefore a request the
    /// encoder cannot serve.
    static let controlTokenOpen = "<|"
    static let controlTokenClose = "|>"

    /// The first `<|…|>` marker in `text`, or `nil`.
    ///
    /// Deliberately literal rather than a regular expression: this is a refusal
    /// check, not a parser, and it must not depend on the same
    /// `NSRegularExpression` availability that ``canEncode`` already guards.
    static func firstControlTokenMarker(in text: String) -> String? {
        guard let open = text.range(of: controlTokenOpen) else { return nil }
        guard let close = text.range(of: controlTokenClose, range: open.upperBound..<text.endIndex)
        else { return nil }
        return String(text[open.lowerBound..<close.upperBound])
    }

    /// Text to ids, `add_special_tokens=False`.
    ///
    /// Two stages, as tiktoken does them: split on the tokenizer's own regex,
    /// then byte-pair merge each piece independently against the mergeable
    /// ranks.
    ///
    /// ## Control tokens are refused, not approximated
    ///
    /// This encoder implements the mergeable-rank half of the tokenizer and
    /// nothing else. The special-token layer — `<|open|>`, `<|end_of_msg|>` and
    /// the rest of ids 163,584–163,839 — lives in the Python wrapper's
    /// `allowed_special` machinery, which is not ported. Fed chat-framed text,
    /// the merge loop happily encodes `<|open|>` as its seven literal bytes and
    /// returns a longer id list that looks perfectly ordinary: M9C measured 57
    /// ids where 24 are correct, with nothing raised. That is the
    /// silent-wrong-answer shape spec §5 forbids, so per spec §17.4 this throws
    /// until the special-token layer is actually implemented.
    ///
    /// The cost of the refusal is that text legitimately containing `<|…|>` can
    /// no longer be encoded. That is the intended trade: a caller who means the
    /// literal characters is far rarer than one who means the control token, and
    /// only the first case has a workaround (encode the surrounding pieces).
    public func encode(_ text: String) throws -> [Int] {
        guard let splitter else {
            throw TinyK3Error.tokenizer(
                "the tokenizer's split pattern did not compile on this platform; "
                    + "use the fixture-provided input ids")
        }
        if let marker = Self.firstControlTokenMarker(in: text) {
            throw TinyK3Error.tokenizer(
                "text contains the control-token marker '\(marker)', and this encoder implements "
                    + "only the mergeable ranks — ids \(baseTokenCount)–\(vocabularySize - 1) have "
                    + "no byte string and cannot be produced by merging. Encoding it would "
                    + "silently return the marker's literal bytes instead. Build the id list with "
                    + "the reference tokenizer, or splice the control id in around encoded pieces.")
        }
        let ns = text as NSString
        var ids: [Int] = []
        var consumed = 0

        for match in splitter.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) {
            // A gap means the pattern did not cover the input, which would
            // silently drop characters. tiktoken's pattern set covers every
            // character, so a gap is a porting failure and is reported.
            guard match.range.location == consumed else {
                throw TinyK3Error.tokenizer(
                    "split pattern skipped characters at offset \(consumed); the port of pat_str "
                        + "does not match tiktoken's")
            }
            consumed = match.range.location + match.range.length
            let piece = ns.substring(with: match.range)
            ids += try bytePairEncode(Data(piece.utf8))
        }
        guard consumed == ns.length else {
            throw TinyK3Error.tokenizer(
                "split pattern covered \(consumed) of \(ns.length) UTF-16 units")
        }
        return ids
    }

    /// tiktoken's `byte_pair_encode`: repeatedly merge the adjacent pair with
    /// the lowest rank.
    ///
    /// Quadratic in the piece length, which is the right trade here — the
    /// regex above splits text into pieces of a few bytes, and a linear
    /// implementation would add a heap for no measurable gain at this scale.
    func bytePairEncode(_ piece: Data) throws -> [Int] {
        if let rank = ranks[piece] { return [rank] }
        guard piece.count > 1 else {
            throw TinyK3Error.tokenizer(
                "byte 0x\(String(piece.first ?? 0, radix: 16)) is not in the vocabulary")
        }

        var parts: [Data] = piece.map { Data([$0]) }
        while parts.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(parts.count - 1) {
                if let rank = ranks[parts[index] + parts[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            parts[bestIndex] = parts[bestIndex] + parts[bestIndex + 1]
            parts.remove(at: bestIndex + 1)
        }

        return try parts.map { part in
            guard let rank = ranks[part] else {
                throw TinyK3Error.tokenizer("merged piece \(part as NSData) is not in the vocabulary")
            }
            return rank
        }
    }
}
