import Foundation
import XCTest

/// The recorded Hugging Face responses under `Fixtures/hf`.
///
/// Recorded from the live API on 2026-08-11 with `curl`, verbatim: the tree
/// pages are the exact bodies each page returned, `index.json` is the exact
/// document each repository publishes, and `resolve-206.json` holds the header
/// blocks of a real `302` and the real `206` it redirected to, plus the first
/// sixteen bytes of the object. Nothing here was hand-written to match the
/// code, which is the point — these fixtures are what the code is measured
/// against.
enum RecordedFixtures {
    struct Repo {
        let folderName: String
        let repoID: String
        let revision: String
        /// What the repository's own `index.json` declares.
        let declaredFiles: Int
        let declaredBytes: UInt64
        /// What the payload rule finds in the tree.
        let expectedPayloadFiles: Int
        let expectedMetadataFiles: Int
        let expectedPayloadBytes: UInt64
        let expectedMetadataBytes: UInt64
        let expectedLargestFileBytes: UInt64
        /// Payload files stored as plain git blobs rather than LFS objects.
        let expectedNonLFSPayloadFiles: Int
    }

    static let kimiK3 = Repo(
        folderName: "Kimi-K3-minirun",
        repoID: "nanguoyu/Kimi-K3-minirun",
        revision: "4715a509773c958e26a31f5e07dd2897f20e0400",
        declaredFiles: 372, declaredBytes: 1_559_976_181_760,
        expectedPayloadFiles: 372, expectedMetadataFiles: 99,
        expectedPayloadBytes: 1_559_976_181_760, expectedMetadataBytes: 19_614_826,
        expectedLargestFileBytes: 5_240_799_232, expectedNonLFSPayloadFiles: 0)

    static let deepseekV4Flash = Repo(
        folderName: "DeepSeek-V4-Flash-0731-minirun",
        repoID: "nanguoyu/DeepSeek-V4-Flash-0731-minirun",
        revision: "37bfd0311d681018c5bb74a24625c0522c1f0883",
        declaredFiles: 577, declaredBytes: 166_893_192_184,
        expectedPayloadFiles: 577, expectedMetadataFiles: 51,
        expectedPayloadBytes: 166_893_192_184, expectedMetadataBytes: 527_260,
        expectedLargestFileBytes: 1_140_867_072, expectedNonLFSPayloadFiles: 0)

    static let minimaxH3 = Repo(
        folderName: "MiniMax-H3-minirun",
        repoID: "nanguoyu/MiniMax-H3-minirun",
        revision: "230643b613250a57eb06454d2725eda060d2c64b",
        declaredFiles: 778, declaredBytes: 63_969_279_450,
        expectedPayloadFiles: 778, expectedMetadataFiles: 118,
        expectedPayloadBytes: 63_969_279_450, expectedMetadataBytes: 782_716,
        expectedLargestFileBytes: 5_061_033_024, expectedNonLFSPayloadFiles: 16)

    static let all = [kimiK3, deepseekV4Flash, minimaxH3]

    static var root: URL {
        guard let url = Bundle.module.url(forResource: "hf", withExtension: nil) else {
            fatalError("Fixtures/hf is missing from MinirunKitTests' resource bundle")
        }
        return url
    }

    static func folder(_ repo: Repo) -> URL {
        root.appendingPathComponent(repo.folderName, isDirectory: true)
    }

    /// Every recorded tree page for a repository, in the order they were served.
    static func treePages(_ repo: Repo) throws -> [Data] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: folder(repo), includingPropertiesForKeys: nil)
        return
            try contents
            .filter { $0.lastPathComponent.hasPrefix("tree-page-") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { try Data(contentsOf: $0) }
    }

    static func indexJSON(_ repo: Repo) throws -> Data {
        try Data(contentsOf: folder(repo).appendingPathComponent("index.json"))
    }

    /// The recorded `302` → `206` pair for one LFS-backed payload file.
    struct ResolvePair {
        let path: String
        let treeEntry: [String: Any]
        let redirect: (status: Int, headers: [String: String])
        let content: (status: Int, headers: [String: String])
    }

    static func resolvePair(_ repo: Repo) throws -> ResolvePair {
        let data = try Data(contentsOf: folder(repo).appendingPathComponent("resolve-206.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let path = root["path"] as? String,
            let entry = root["treeEntry"] as? [String: Any],
            let responses = root["responses"] as? [[String: Any]],
            responses.count >= 2
        else {
            throw NSError(
                domain: "RecordedFixtures", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "resolve-206.json is not the shape recorded"])
        }
        func decode(_ object: [String: Any]) -> (status: Int, headers: [String: String]) {
            (object["status"] as? Int ?? 0, object["headers"] as? [String: String] ?? [:])
        }
        return ResolvePair(
            path: path, treeEntry: entry,
            redirect: decode(responses[0]), content: decode(responses[1]))
    }
}
