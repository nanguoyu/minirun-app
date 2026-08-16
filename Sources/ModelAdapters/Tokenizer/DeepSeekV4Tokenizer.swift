import CoreFoundation
import Foundation

/// A checked reader for DeepSeek V4's Hugging Face byte-level BPE vocabulary.
///
/// The product loads this value from the completely verified artifact.  It
/// deliberately implements the tokenizer schema carried by that artifact
/// instead of assuming that a repository name implies one fixed vocabulary.
/// A future repository update is accepted when it still describes the same
/// supported byte-level BPE contract; an incompatible tokenizer fails by name.
public final class DeepSeekV4Vocabulary: @unchecked Sendable {
    public enum VocabularyError: Error, LocalizedError, Equatable {
        case invalid(String)
        case unsupportedText(String)

        public var errorDescription: String? {
            switch self {
            case .invalid(let message), .unsupportedText(let message): return message
            }
        }
    }

    private struct Pair: Hashable {
        let left: String
        let right: String
    }

    private struct Node {
        var token: String
        var previous: Int?
        var next: Int?
        var active = true
        var version = 0
    }

    private struct Candidate {
        let rank: Int
        let left: Int
        let right: Int
        let leftVersion: Int
        let rightVersion: Int
    }

    private struct MinHeap {
        var storage: [Candidate] = []

        mutating func push(_ value: Candidate) {
            storage.append(value)
            var child = storage.count - 1
            while child > 0 {
                let parent = (child - 1) / 2
                guard Self.precedes(storage[child], storage[parent]) else { break }
                storage.swapAt(child, parent)
                child = parent
            }
        }

        mutating func pop() -> Candidate? {
            guard !storage.isEmpty else { return nil }
            if storage.count == 1 { return storage.removeLast() }
            let result = storage[0]
            storage[0] = storage.removeLast()
            var parent = 0
            while true {
                let left = parent * 2 + 1
                guard left < storage.count else { break }
                let right = left + 1
                let child = right < storage.count && Self.precedes(storage[right], storage[left])
                    ? right : left
                guard Self.precedes(storage[child], storage[parent]) else { break }
                storage.swapAt(parent, child)
                parent = child
            }
            return result
        }

        /// Merge rank is authoritative.  Repeated instances of one merge use
        /// the leftmost occurrence, matching the reference BPE pass.
        private static func precedes(_ lhs: Candidate, _ rhs: Candidate) -> Bool {
            if lhs.rank != rhs.rank { return lhs.rank < rhs.rank }
            return lhs.left < rhs.left
        }
    }

    private let tokenIDs: [String: Int]
    private let baseTokens: [String]
    private let addedTokensByID: [Int: String]
    private let addedTokenContents: [String]
    private let mergeRanks: [Pair: Int]
    private let splitters: [NSRegularExpression]
    private let byteScalars: [Unicode.Scalar]
    private let scalarBytes: [Unicode.Scalar: UInt8]

    public let baseTokenCount: Int
    public let vocabularySize: Int

    public convenience init(contentsOf path: String) throws {
        try self.init(data: Data(contentsOf: URL(fileURLWithPath: path)))
    }

    /// Parses a verified in-memory `tokenizer.json` without reopening a path.
    public init(data: Data) throws {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw VocabularyError.invalid("tokenizer.json is not a JSON object") }
        guard root["version"] as? String == "1.0",
            let model = root["model"] as? [String: Any],
            model["type"] as? String == "BPE",
            model["dropout"] is NSNull,
            model["unk_token"] is NSNull,
            Self.isAbsentString(model["continuing_subword_prefix"]),
            Self.isAbsentString(model["end_of_word_suffix"]),
            model["fuse_unk"] as? Bool == false,
            model["byte_fallback"] as? Bool == false
        else {
            throw VocabularyError.invalid(
                "the verified tokenizer is not the supported no-dropout byte-level BPE schema")
        }

        guard let rawVocabulary = model["vocab"] as? [String: Any], !rawVocabulary.isEmpty
        else { throw VocabularyError.invalid("the BPE vocabulary is empty or malformed") }
        var tokenIDs: [String: Int] = [:]
        var tokensByID: [Int: String] = [:]
        for (token, rawID) in rawVocabulary {
            guard let id = Self.integer(rawID), id >= 0,
                tokenIDs.updateValue(id, forKey: token) == nil,
                tokensByID.updateValue(token, forKey: id) == nil
            else { throw VocabularyError.invalid("the BPE vocabulary has duplicate or invalid ids") }
        }
        let baseCount = tokensByID.count
        guard Set(tokensByID.keys) == Set(0..<baseCount) else {
            throw VocabularyError.invalid("the BPE vocabulary ids are not contiguous from zero")
        }

        guard let rawAdded = root["added_tokens"] as? [[String: Any]] else {
            throw VocabularyError.invalid("tokenizer.json has no added-token table")
        }
        var addedByID: [Int: String] = [:]
        var addedByContent: [String: Int] = [:]
        for entry in rawAdded {
            guard let id = Self.integer(entry["id"]), id >= 0,
                let content = entry["content"] as? String, !content.isEmpty,
                entry["single_word"] as? Bool == false,
                entry["lstrip"] as? Bool == false,
                entry["rstrip"] as? Bool == false,
                entry["normalized"] is Bool,
                entry["special"] is Bool
            else { throw VocabularyError.invalid("the added-token table is malformed") }
            if let base = tokensByID[id], base != content {
                throw VocabularyError.invalid(
                    "added token id \(id) disagrees with the BPE vocabulary")
            }
            guard addedByID.updateValue(content, forKey: id) == nil,
                addedByContent.updateValue(id, forKey: content) == nil
            else { throw VocabularyError.invalid("the added-token table contains duplicates") }
        }
        let maximumID = max(tokensByID.keys.max() ?? -1, addedByID.keys.max() ?? -1)
        guard maximumID >= 0,
            Set(tokensByID.keys).union(addedByID.keys) == Set(0...maximumID)
        else { throw VocabularyError.invalid("the complete tokenizer ids are not contiguous") }

        guard let rawMerges = model["merges"] as? [String] else {
            throw VocabularyError.invalid("the BPE merge table is malformed")
        }
        var mergeRanks: [Pair: Int] = [:]
        mergeRanks.reserveCapacity(rawMerges.count)
        for (rank, value) in rawMerges.enumerated() {
            guard let separator = value.firstIndex(of: " "), separator != value.startIndex,
                separator < value.index(before: value.endIndex)
            else { throw VocabularyError.invalid("BPE merge \(rank) is malformed") }
            let pair = Pair(
                left: String(value[..<separator]),
                right: String(value[value.index(after: separator)...]))
            guard mergeRanks.updateValue(rank, forKey: pair) == nil,
                tokenIDs[pair.left + pair.right] != nil
            else {
                throw VocabularyError.invalid(
                    "BPE merge \(rank) is duplicated or produces no vocabulary token")
            }
        }

        let splitPatterns = try Self.splitPatterns(from: root)
        let splitters: [NSRegularExpression]
        do {
            splitters = try splitPatterns.map { try NSRegularExpression(pattern: $0) }
        } catch {
            throw VocabularyError.invalid("a byte-level pre-tokenizer regex did not compile: \(error)")
        }

        let byteScalars = Self.makeByteScalars()
        var scalarBytes: [Unicode.Scalar: UInt8] = [:]
        for (byte, scalar) in byteScalars.enumerated() {
            scalarBytes[scalar] = UInt8(byte)
        }
        let addedIDs = Set(addedByID.keys)
        for id in 0..<baseCount where !addedIDs.contains(id) {
            guard let token = tokensByID[id],
                token.unicodeScalars.allSatisfy({ scalarBytes[$0] != nil })
            else {
                throw VocabularyError.invalid(
                    "BPE token id \(id) cannot be decoded by the declared byte-level alphabet")
            }
        }

        self.tokenIDs = tokenIDs.merging(addedByContent) { base, added in
            precondition(base == added)
            return base
        }
        self.baseTokens = (0..<baseCount).map { tokensByID[$0]! }
        self.addedTokensByID = addedByID
        // Longest first makes diagnostics deterministic when one control is a
        // prefix of another.  Encoding ordinary message content refuses every
        // added token; structural controls are inserted through `id(for:)`.
        self.addedTokenContents = addedByContent.keys.sorted {
            if $0.count != $1.count { return $0.count > $1.count }
            return $0 < $1
        }
        self.mergeRanks = mergeRanks
        self.splitters = splitters
        self.byteScalars = byteScalars
        self.scalarBytes = scalarBytes
        self.baseTokenCount = baseCount
        self.vocabularySize = maximumID + 1
    }

    /// The exact id of a structural token declared by this tokenizer.
    public func id(for content: String) throws -> Int {
        guard let id = tokenIDs[content], addedTokensByID[id] == content else {
            throw VocabularyError.invalid(
                "the verified tokenizer does not declare structural token '\(content)'")
        }
        return id
    }

    /// Encodes ordinary text only.  Added-token markup is refused so message
    /// content cannot smuggle role, thinking, task or EOS controls into the
    /// typed chat template.
    public func encode(_ text: String) throws -> [Int] {
        if let marker = addedTokenContents.first(where: { text.contains($0) }) {
            throw VocabularyError.unsupportedText(
                "message text contains tokenizer control '\(marker)'; controls may only be inserted by the typed V4 chat template")
        }
        var pieces = [text]
        for splitter in splitters {
            pieces = pieces.flatMap { Self.isolatedPieces(in: $0, using: splitter) }
        }

        var result: [Int] = []
        for piece in pieces where !piece.isEmpty {
            let byteLevel = String(piece.utf8.map { Character(byteScalars[Int($0)]) })
            result.append(contentsOf: try bytePairEncode(byteLevel))
        }
        return result
    }

    /// Decodes ids to user-visible text.  Byte fragments are joined before
    /// UTF-8 decoding because a token boundary may split one scalar.
    public func decode(_ ids: [Int]) -> String {
        var bytes = Data()
        var output = ""
        func flush() {
            guard !bytes.isEmpty else { return }
            output += String(decoding: bytes, as: UTF8.self)
            bytes.removeAll(keepingCapacity: true)
        }

        for id in ids {
            if let added = addedTokensByID[id] {
                flush()
                output += added
                continue
            }
            guard id >= 0, id < baseTokens.count else {
                flush()
                output += "<|invalid_token_\(id)|>"
                continue
            }
            for scalar in baseTokens[id].unicodeScalars {
                if let byte = scalarBytes[scalar] {
                    bytes.append(byte)
                } else {
                    flush()
                    output += "<|invalid_byte_token_\(id)|>"
                    break
                }
            }
        }
        flush()
        return output
    }

    private func bytePairEncode(_ value: String) throws -> [Int] {
        if let id = tokenIDs[value] { return [id] }
        let initial = value.unicodeScalars.map(String.init)
        guard !initial.isEmpty else { return [] }
        var nodes = initial.enumerated().map { index, token in
            Node(
                token: token,
                previous: index == 0 ? nil : index - 1,
                next: index + 1 == initial.count ? nil : index + 1)
        }
        var heap = MinHeap()

        func candidate(left: Int, right: Int) -> Candidate? {
            guard nodes[left].active, nodes[right].active,
                nodes[left].next == right,
                let rank = mergeRanks[Pair(left: nodes[left].token, right: nodes[right].token)]
            else { return nil }
            return Candidate(
                rank: rank, left: left, right: right,
                leftVersion: nodes[left].version, rightVersion: nodes[right].version)
        }
        if nodes.count > 1 {
            for left in 0..<(nodes.count - 1) {
                if let value = candidate(left: left, right: left + 1) { heap.push(value) }
            }
        }

        while let best = heap.pop() {
            guard nodes[best.left].active, nodes[best.right].active,
                nodes[best.left].version == best.leftVersion,
                nodes[best.right].version == best.rightVersion,
                nodes[best.left].next == best.right,
                mergeRanks[Pair(
                    left: nodes[best.left].token, right: nodes[best.right].token)] == best.rank
            else { continue }

            let previous = nodes[best.left].previous
            let following = nodes[best.right].next
            nodes[best.left].token += nodes[best.right].token
            nodes[best.left].next = following
            nodes[best.left].version += 1
            nodes[best.right].active = false
            nodes[best.right].version += 1
            if let following { nodes[following].previous = best.left }

            if let previous, let value = candidate(left: previous, right: best.left) {
                heap.push(value)
            }
            if let following, let value = candidate(left: best.left, right: following) {
                heap.push(value)
            }
        }

        var ids: [Int] = []
        var cursor: Int? = 0
        while let index = cursor {
            guard nodes[index].active, let id = tokenIDs[nodes[index].token] else {
                throw VocabularyError.invalid(
                    "the BPE merge result is absent from the verified vocabulary")
            }
            ids.append(id)
            cursor = nodes[index].next
        }
        return ids
    }

    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        return Int(number.stringValue)
    }

    private static func isAbsentString(_ value: Any?) -> Bool {
        value is NSNull || value as? String == ""
    }

    private static func splitPatterns(from root: [String: Any]) throws -> [String] {
        guard let pre = root["pre_tokenizer"] as? [String: Any],
            pre["type"] as? String == "Sequence",
            let items = pre["pretokenizers"] as? [[String: Any]], items.count == 4
        else { throw VocabularyError.invalid("the byte-level pre-tokenizer sequence is missing") }
        var patterns: [String] = []
        for index in 0..<3 {
            let item = items[index]
            guard item["type"] as? String == "Split",
                item["behavior"] as? String == "Isolated",
                item["invert"] as? Bool == false,
                let pattern = item["pattern"] as? [String: Any],
                let regex = pattern["Regex"] as? String
            else { throw VocabularyError.invalid("pre-tokenizer split \(index) is unsupported") }
            patterns.append(regex)
        }
        let byteLevel = items[3]
        guard byteLevel["type"] as? String == "ByteLevel",
            byteLevel["add_prefix_space"] as? Bool == false,
            byteLevel["use_regex"] as? Bool == false
        else { throw VocabularyError.invalid("the final byte-level pre-tokenizer is unsupported") }
        guard let decoder = root["decoder"] as? [String: Any],
            decoder["type"] as? String == "ByteLevel"
        else { throw VocabularyError.invalid("the tokenizer has no byte-level decoder") }
        return patterns
    }

    private static func isolatedPieces(
        in text: String, using expression: NSRegularExpression
    ) -> [String] {
        guard !text.isEmpty else { return [] }
        let source = text as NSString
        let matches = expression.matches(
            in: text, range: NSRange(location: 0, length: source.length))
        guard !matches.isEmpty else { return [text] }
        var result: [String] = []
        var cursor = 0
        for match in matches {
            if match.range.location > cursor {
                result.append(source.substring(with: NSRange(
                    location: cursor, length: match.range.location - cursor)))
            }
            if match.range.length > 0 { result.append(source.substring(with: match.range)) }
            cursor = match.range.location + match.range.length
        }
        if cursor < source.length {
            result.append(source.substring(from: cursor))
        }
        return result
    }

    /// GPT-2/ByteLevel's reversible byte-to-Unicode alphabet.
    private static func makeByteScalars() -> [Unicode.Scalar] {
        var direct = Array(33...126) + Array(161...172) + Array(174...255)
        var codepoints = direct
        var extra = 0
        for byte in 0...255 where !direct.contains(byte) {
            direct.append(byte)
            codepoints.append(256 + extra)
            extra += 1
        }
        var result = Array(repeating: Unicode.Scalar(0)!, count: 256)
        for (byte, codepoint) in zip(direct, codepoints) {
            result[byte] = Unicode.Scalar(codepoint)!
        }
        return result
    }
}

/// The deliberately narrow, official DeepSeek V4 text-chat controls.
public enum DeepSeekV4ChatControl: String, CaseIterable, Sendable {
    case beginningOfSentence = "<｜begin▁of▁sentence｜>"
    case endOfSentence = "<｜end▁of▁sentence｜>"
    case user = "<｜User｜>"
    case assistant = "<｜Assistant｜>"
    case thinkingEnd = "</think>"
}

public enum DeepSeekV4ChatRole: String, Sendable {
    case system
    case user
    case assistant
}

public struct DeepSeekV4ChatMessage: Equatable, Sendable {
    public let role: DeepSeekV4ChatRole
    public let content: String

    public init(role: DeepSeekV4ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

private enum DeepSeekV4ChatSegment {
    case text(String)
    case control(DeepSeekV4ChatControl)
}

/// DeepSeek V4's official text-only chat renderer with thinking disabled.
///
/// Tools, task tokens, multimodal blocks and thinking-mode content are absent
/// on purpose.  They have additional framing and parsing rules in the official
/// encoder and cannot be inferred from this safe text-chat subset.
public struct DeepSeekV4NoThinkingChatPrompt: Equatable, Sendable {
    public let messages: [DeepSeekV4ChatMessage]
    public let addAssistantGenerationPrefix: Bool

    public init(
        messages: [DeepSeekV4ChatMessage],
        addAssistantGenerationPrefix: Bool = true
    ) {
        self.messages = messages
        self.addAssistantGenerationPrefix = addAssistantGenerationPrefix
    }

    public var rendered: String {
        segments.map {
            switch $0 {
            case .text(let value): return value
            case .control(let value): return value.rawValue
            }
        }.joined()
    }

    public func encode(using vocabulary: DeepSeekV4Vocabulary) throws -> [Int] {
        var ids: [Int] = []
        for segment in segments {
            switch segment {
            case .text(let value): ids.append(contentsOf: try vocabulary.encode(value))
            case .control(let value): ids.append(try vocabulary.id(for: value.rawValue))
            }
        }
        return ids
    }

    private var segments: [DeepSeekV4ChatSegment] {
        var result: [DeepSeekV4ChatSegment] = [.control(.beginningOfSentence)]
        for (index, message) in messages.enumerated() {
            switch message.role {
            case .system:
                if !message.content.isEmpty { result.append(.text(message.content)) }
            case .user:
                result.append(.control(.user))
                if !message.content.isEmpty { result.append(.text(message.content)) }
            case .assistant:
                if !message.content.isEmpty { result.append(.text(message.content)) }
                result.append(.control(.endOfSentence))
            }

            let nextRole = index + 1 < messages.count ? messages[index + 1].role : nil
            if message.role == .user,
                nextRole == .assistant || (nextRole == nil && addAssistantGenerationPrefix)
            {
                result.append(.control(.assistant))
                result.append(.control(.thinkingEnd))
            }
        }
        return result
    }
}
