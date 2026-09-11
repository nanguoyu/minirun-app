import Foundation

/// The compressed token ids Engram hashes n-grams over.
///
/// Engram does not hash token ids. It hashes a smaller id space in which every
/// token whose text *normalizes* alike collapses together, so `" The"`, `"the"`
/// and `"THE"` address the same rows: 99,092 classes for the checkpoint's
/// 129,280 tokens. The map is not stored in the checkpoint. DeepSeek's
/// `inference/engram.py` rebuilds it at load time from the tokenizer alone,
/// and `DeepSeekV41EngramConstants.compressedVocabularySize` — which every hash
/// multiplier is derived from — is that map's class count. A map that differs
/// anywhere therefore does not merely mis-address a few rows; it rehashes the
/// whole 384-million-row table into rows that mean nothing.
///
/// ## The eight steps, and why they are spelled out
///
/// The reference builds one `tokenizers` normalizer sequence:
///
/// ```text
/// NFKC -> NFD -> StripAccents -> Lowercase
///      -> Replace([ \t\r\n]+, " ") -> Replace(^ $, U+E000)
///      -> Strip -> Replace(U+E000, " ")
/// ```
///
/// Three of those steps do something other than their name suggests, and each
/// is a place a plausible Swift version would silently differ:
///
/// - **`StripAccents` removes every Mark, not only the accents.** The Rust
///   implementation filters on general category `M`, so a Devanagari vowel sign
///   (`Mc`) and an enclosing circle (`Me`) go the same way a combining acute
///   (`Mn`) does. Measured against the pinned tokenizer, not assumed.
/// - **`Lowercase` is per scalar, with no contextual rules.** It maps each
///   character through Unicode's unconditional lowercase mapping, so a Greek
///   capital sigma at the end of a word becomes `σ`, never the final form `ς`.
///   `String.lowercased()` on the *whole* string applies the final-sigma rule
///   and would disagree; applied one scalar at a time it cannot, because the
///   rule needs a preceding cased letter to fire.
/// - **The sentinel dance exists for exactly one token.** A token that is a
///   single space would be emptied by `Strip` and merged with unrelated tokens,
///   so it is parked in a private-use scalar across the strip and put back.
///
/// The order matters as much as the steps. `NFD` before `StripAccents` is what
/// turns `é` into a strippable mark; `StripAccents` before `Lowercase` is why
/// `İ` becomes `i` rather than `i` followed by a combining dot. And nothing
/// recomposes afterwards, so Hangul stays decomposed into jamo.
///
/// Every one of these is checked against a fixture produced by running the
/// reference itself over all 129,280 tokens; see
/// `DeepSeekV41EngramTokenMapTests`.
public enum DeepSeekV41EngramTokenMap {
    /// A private-use scalar, so a token that is exactly one space survives
    /// `Strip` instead of collapsing to the empty string.
    static let sentinel: Unicode.Scalar = "\u{E000}"
    /// The replacement character a lossy UTF-8 decode leaves behind. Its
    /// presence, not its count, is what marks a partial-UTF-8 byte token.
    ///
    /// Tested against the *scalars* and never with `String.contains(_: Character)`:
    /// `contains` compares grapheme clusters, and a replacement character
    /// followed by a combining mark is one cluster that is not equal to a bare
    /// `U+FFFD`. Twenty of the checkpoint's 1,484 partial-UTF-8 tokens are
    /// exactly that shape, and a `Character` test silently normalizes them as
    /// text instead of keying them by their raw form — eleven compressed
    /// classes short, and every hash multiplier derived from the wrong count.
    static let replacement: Unicode.Scalar = "\u{FFFD}"

    /// Whether a decoded token carries a lossy-decode marker.
    static func isPartialUTF8(_ text: String) -> Bool {
        text.unicodeScalars.contains(replacement)
    }

    /// One compressed id per token id, plus the number of distinct classes.
    public struct Map: Sendable {
        /// `compressedIDs[tokenID]`, for every id the vocabulary declares.
        public let compressedIDs: [Int32]
        /// Distinct classes; must equal
        /// ``DeepSeekV41EngramConstants/compressedVocabularySize``.
        public let classCount: Int

        public subscript(tokenID: Int) -> Int32 {
            compressedIDs[tokenID]
        }
    }

    /// Build the map from a verified `tokenizer.json`.
    ///
    /// Ids are handed out in first-seen order over ascending token ids, which
    /// is why this walks the vocabulary in order and never sorts: the class of
    /// token 5,000 depends on which classes tokens 0…4,999 already created.
    ///
    /// `expectedClassCount` defaults to the released count and is checked, for
    /// the reason the reference asserts the same thing: a map with a different
    /// class count rehashes everything downstream of it.
    public static func build(
        vocabulary: DeepSeekV4Vocabulary,
        expectedClassCount: Int? = DeepSeekV41EngramConstants.compressedVocabularySize
    ) throws -> Map {
        let count = vocabulary.vocabularySize
        guard count > 0 else {
            throw DeepSeekV4Error.configuration(
                "the Engram token map needs a non-empty vocabulary")
        }
        // Keyed by UTF-8 bytes, never by `String`. Swift's `String` equality is
        // canonical equivalence, so `"Ì"` and `"I\u{0300}"` are one key — and
        // the partial-UTF-8 tokens are keyed by their *raw* byte-level form,
        // which is unnormalized and full of Latin-1 letters that decompose.
        // Python's dict compares code points, so a `[String: Int32]` here
        // merges classes the reference keeps apart: measured, 99,090 instead of
        // 99,092, which would have rehashed the whole table.
        var classes: [[UInt8]: Int32] = [:]
        classes.reserveCapacity(count)
        var compressed = [Int32](repeating: 0, count: count)
        var lowercaseCache: [Unicode.Scalar: [Unicode.Scalar]] = [:]

        for tokenID in 0..<count {
            let key = Array(
                try self.key(
                    forTokenID: tokenID, vocabulary: vocabulary,
                    lowercaseCache: &lowercaseCache
                ).utf8)
            if let existing = classes[key] {
                compressed[tokenID] = existing
            } else {
                let assigned = Int32(classes.count)
                classes[key] = assigned
                compressed[tokenID] = assigned
            }
        }

        if let expectedClassCount, classes.count != expectedClassCount {
            throw DeepSeekV4Error.configuration(
                "the Engram token map produced \(classes.count) compressed classes; "
                    + "the released Engram hash geometry is derived from "
                    + "\(expectedClassCount)")
        }
        return Map(compressedIDs: compressed, classCount: classes.count)
    }

    /// The key one token contributes: its normalized text, or its raw form.
    static func key(
        forTokenID tokenID: Int,
        vocabulary: DeepSeekV4Vocabulary,
        lowercaseCache: inout [Unicode.Scalar: [Unicode.Scalar]]
    ) throws -> String {
        let text = vocabulary.decode([tokenID])
        if isPartialUTF8(text) {
            // A partial UTF-8 byte token: there is no text to normalize, so the
            // reference keys it by the vocabulary's own spelling of the token.
            guard let raw = vocabulary.rawToken(for: tokenID) else {
                throw DeepSeekV4Error.configuration(
                    "token id \(tokenID) has no raw form to key the Engram map by")
            }
            return raw
        }
        let normalized = normalizedKey(text, lowercaseCache: &lowercaseCache)
        return normalized.isEmpty ? text : normalized
    }

    /// The eight-step normalizer, in order.
    public static func normalizedKey(_ text: String) -> String {
        var cache: [Unicode.Scalar: [Unicode.Scalar]] = [:]
        return normalizedKey(text, lowercaseCache: &cache)
    }

    static func normalizedKey(
        _ text: String,
        lowercaseCache: inout [Unicode.Scalar: [Unicode.Scalar]]
    ) -> String {
        // 1. NFKC, then 2. NFD. Not a no-op pair: NFKC folds the compatibility
        // forms (ligatures, fullwidth, circled digits) and composes, and NFD
        // then splits the canonical compositions back apart so step 3 can see
        // the marks. Nothing recomposes afterwards.
        let decomposed = text
            .precomposedStringWithCompatibilityMapping
            .decomposedStringWithCanonicalMapping

        var scalars = [Unicode.Scalar]()
        scalars.reserveCapacity(decomposed.unicodeScalars.count)
        for scalar in decomposed.unicodeScalars {
            // 3. StripAccents: every Mark category, not only nonspacing.
            if isMark(scalar) { continue }
            // 4. Lowercase, one scalar at a time and therefore contextless.
            if let mapped = lowercaseCache[scalar] {
                scalars.append(contentsOf: mapped)
            } else {
                let mapped = Array(String(scalar).lowercased().unicodeScalars)
                lowercaseCache[scalar] = mapped
                scalars.append(contentsOf: mapped)
            }
        }

        // 5. Collapse runs of the four ASCII spacing characters onto one space.
        // Only those four: every other whitespace scalar is left where it is,
        // and NFKC has already turned NBSP and the ideographic space into
        // ordinary spaces so they take part.
        var collapsed = [Unicode.Scalar]()
        collapsed.reserveCapacity(scalars.count)
        var inRun = false
        for scalar in scalars {
            if isCollapsibleSpace(scalar) {
                if !inRun { collapsed.append(" ") }
                inRun = true
            } else {
                collapsed.append(scalar)
                inRun = false
            }
        }

        // 6. A string that is exactly one space becomes the sentinel, so that
        // 7. Strip cannot empty it, and 8. puts the space back.
        if collapsed.count == 1, collapsed[0] == " " {
            collapsed = [sentinel]
        }
        var start = 0
        var end = collapsed.count
        while start < end, collapsed[start].properties.isWhitespace { start += 1 }
        while end > start, collapsed[end - 1].properties.isWhitespace { end -= 1 }
        var view = String.UnicodeScalarView()
        for index in start..<end {
            view.append(collapsed[index] == sentinel ? " " : collapsed[index])
        }
        return String(view)
    }

    /// General category `M`: `Mn`, `Mc` and `Me` alike.
    static func isMark(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .nonspacingMark, .spacingMark, .enclosingMark: return true
        default: return false
        }
    }

    /// The four characters the reference's `[ \t\r\n]+` matches, and no others.
    static func isCollapsibleSpace(_ scalar: Unicode.Scalar) -> Bool {
        scalar == " " || scalar == "\t" || scalar == "\r" || scalar == "\n"
    }
}
