import Darwin
import Foundation
import ModelAdapters
import MinirunRunners
import StorageCore
import XCTest

@testable import MinirunKit
@testable import MinirunApp

@MainActor
final class DeepSeekV4ProductRuntimeTests: XCTestCase {
    func testVerifiedTokenizerUsesDynamicPublicationAndOfficialNoThinkingFraming() throws {
        let fixture = try makeArtifact(
            sourceRevision: String(repeating: "a", count: 40),
            publicationRevision: String(repeating: "b", count: 40))
        defer { try? FileManager.default.removeItem(at: fixture.location) }

        let tokenizer = try DeepSeekV4ProductTokenizer(artifact: fixture.reference)
        let ids = try tokenizer.encode(turns: [PromptTurn(role: .user, text: "A")])

        XCTAssertEqual(ids, [256, 258, 65, 259, 260])
        XCTAssertEqual(tokenizer.decode(tokenIDs: [65, 257, 66]), "A")
        XCTAssertEqual(tokenizer.verifiedIdentity.model, .deepseekV4Flash)
        XCTAssertEqual(
            tokenizer.verifiedIdentity.artifactRevision,
            String(repeating: "b", count: 40))
        XCTAssertEqual(
            tokenizer.verifiedIdentity.sourceRevision,
            String(repeating: "a", count: 40))
        XCTAssertTrue(tokenizer.appliesChatTemplate)
        XCTAssertTrue(tokenizer.provenance.contains(String(repeating: "b", count: 12)))
        XCTAssertTrue(tokenizer.provenance.contains(String(repeating: "a", count: 12)))
    }

    func testCompatibilityMovesWithConsistentImmutableSourceRevisions() throws {
        let first = ArtifactIndexIdentity.parse(try indexData(
            sourceRevision: String(repeating: "1", count: 40)))
        let later = ArtifactIndexIdentity.parse(try indexData(
            sourceRevision: String(repeating: "2", count: 40)))
        let mismatched = ArtifactIndexIdentity.parse(try indexData(
            sourceRevision: String(repeating: "3", count: 40),
            tokenizerSourceRevision: String(repeating: "4", count: 40)))
        let mutable = ArtifactIndexIdentity.parse(try indexData(
            sourceRevision: "main", tokenizerSourceRevision: "main"))

        XCTAssertTrue(DeepSeekV4ProductArtifact.accepts(first))
        XCTAssertTrue(DeepSeekV4ProductArtifact.accepts(later))
        XCTAssertFalse(DeepSeekV4ProductArtifact.accepts(mismatched))
        XCTAssertFalse(DeepSeekV4ProductArtifact.accepts(mutable))
    }

    func testTokenizerRefusesBytesChangedAfterCompleteVerification() throws {
        let fixture = try makeArtifact(
            sourceRevision: String(repeating: "c", count: 40),
            publicationRevision: String(repeating: "d", count: 40))
        defer { try? FileManager.default.removeItem(at: fixture.location) }

        var changed = try Data(contentsOf: fixture.tokenizerURL)
        changed[changed.startIndex] ^= 0x01
        try changed.write(to: fixture.tokenizerURL, options: .atomic)

        XCTAssertThrowsError(
            try DeepSeekV4ProductTokenizer(artifact: fixture.reference)
        ) { error in
            XCTAssertTrue(
                String(describing: error).contains("tokenizer")
                    || String(describing: error).contains("verified"),
                "unexpected refusal: \(error)")
        }
    }

    func testShippingRegistryComposesTheCompleteV4ProviderOnArm64() throws {
        #if arch(arm64)
            let runtime = try XCTUnwrap(
                ModelRuntimeRegistry.product.runtime(for: .deepseekV4Flash))
            XCTAssertEqual(
                ModelRuntimeRegistry.product.registeredModelIDs,
                [.kimiK3, .deepseekV4Flash])
            XCTAssertEqual(
                runtime.capabilities.minimumBudgetBytes,
                DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
            XCTAssertEqual(
                runtime.capabilities.maximumNewTokens,
                DeepSeekV4ProductMemoryBudget.maximumNewTokens)
            XCTAssertFalse(runtime.capabilities.supportsPinPlan)

            let fixture = try makeArtifact(
                sourceRevision: String(repeating: "5", count: 40),
                publicationRevision: String(repeating: "6", count: 40))
            defer { try? FileManager.default.removeItem(at: fixture.location) }
            let conversation = Conversation(
                settings: ConversationSettings(
                    model: .deepseekV4Flash,
                    memoryBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes))
            let prepared = try runtime.prepare(
                artifact: fixture.reference, for: conversation)
            XCTAssertTrue(prepared.tokenizer is DeepSeekV4ProductTokenizer)
            XCTAssertTrue(prepared.runner is DeepSeekV4DecodeRunner)
            XCTAssertFalse(prepared.runner.capabilities.supportsPinPlan)
            XCTAssertEqual(
                prepared.progressLayerCount, 43,
                "the instrument ladder must use the verified V4 config, not the one-entry "
                    + "unknown planner census")
            XCTAssertNil(
                prepared.artifactCensus,
                "layer progress must not manufacture unmeasured V4 residency geometry")
        #else
            XCTAssertNil(ModelRuntimeRegistry.product.runtime(for: .deepseekV4Flash))
            XCTAssertNil(ModelRuntimeRegistry.product.runtime(for: .kimiK3))
            XCTAssertEqual(ModelRuntimeRegistry.product.registeredModelIDs, [])
        #endif
    }

    func testRunnerScaleFollowsTheStatedBudgetAndNeverStatesBelowTheFloor() {
        let floor = DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
        let ceiling = DeepSeekV4ProductMemoryBudget.maximumNewTokens

        XCTAssertEqual(
            DeepSeekV4ProductRuntimeProvider.runnerScale(statedBudgetBytes: floor), .product,
            "the Floor preset is the gated product path a chat that never touched the dial gets")
        XCTAssertEqual(
            DeepSeekV4ProductRuntimeProvider.runnerScale(statedBudgetBytes: floor - 1), .product,
            "a persisted sub-floor budget must meet the dial's refusal, not a runner that was "
                + "handed the number as its own minimum")
        XCTAssertEqual(
            DeepSeekV4ProductRuntimeProvider.runnerScale(statedBudgetBytes: 0), .product)
        XCTAssertEqual(
            DeepSeekV4ProductRuntimeProvider.runnerScale(statedBudgetBytes: floor + 1),
            .stated(minimumBudgetBytes: floor + 1, maximumNewTokens: ceiling))
        XCTAssertEqual(
            DeepSeekV4ProductRuntimeProvider.runnerScale(statedBudgetBytes: 8_000_000_000),
            .stated(minimumBudgetBytes: 8_000_000_000, maximumNewTokens: ceiling))
        XCTAssertEqual(
            DeepSeekV4ProductRuntimeProvider.runnerScale(statedBudgetBytes: 20_000_000_000),
            .stated(minimumBudgetBytes: 20_000_000_000, maximumNewTokens: ceiling))
    }

    func testPreparedRunnerCarriesTheConversationBudgetWhileCapabilitiesStayProduct() throws {
        #if arch(arm64)
            let runtime = try XCTUnwrap(
                ModelRuntimeRegistry.product.runtime(for: .deepseekV4Flash))
            let fixture = try makeArtifact(
                sourceRevision: String(repeating: "7", count: 40),
                publicationRevision: String(repeating: "8", count: 40))
            defer { try? FileManager.default.removeItem(at: fixture.location) }

            let floor = DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
            let ceiling = DeepSeekV4ProductMemoryBudget.maximumNewTokens

            func preparedScale(budgetBytes: UInt64) throws -> DeepSeekV4DecodeRunner.Scale {
                let conversation = Conversation(
                    settings: ConversationSettings(
                        model: .deepseekV4Flash, memoryBudgetBytes: budgetBytes))
                let prepared = try runtime.prepare(
                    artifact: fixture.reference, for: conversation)
                return try XCTUnwrap(prepared.runner as? DeepSeekV4DecodeRunner).scale
            }

            XCTAssertEqual(try preparedScale(budgetBytes: floor), .product)
            XCTAssertEqual(
                try preparedScale(budgetBytes: 8_000_000_000),
                .stated(minimumBudgetBytes: 8_000_000_000, maximumNewTokens: ceiling),
                "the dial's 8 GB must reach the runner; before this it did not and 2 GB and "
                    + "8 GB ran identically")
            XCTAssertEqual(
                try preparedScale(budgetBytes: 20_000_000_000),
                .stated(minimumBudgetBytes: 20_000_000_000, maximumNewTokens: ceiling))
            XCTAssertEqual(
                try preparedScale(budgetBytes: floor / 2), .product,
                "a sub-floor persisted budget keeps the product path and the existing refusal")

            XCTAssertEqual(
                runtime.capabilities.minimumBudgetBytes, floor,
                "what the model requires is a product-scale fact, not one conversation's dial")
            XCTAssertEqual(runtime.capabilities.maximumNewTokens, ceiling)
        #else
            throw XCTSkip("the V4 product runtime is registered on Apple silicon only")
        #endif
    }

    private struct Fixture {
        let location: URL
        let tokenizerURL: URL
        let reference: ArtifactReference
    }

    private func makeArtifact(
        sourceRevision: String, publicationRevision: String
    ) throws -> Fixture {
        let location = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunV4ProductTokenizer-\(UUID().uuidString)")
        let artifact = location.appendingPathComponent("artifact", isDirectory: true)
        let unit = artifact.appendingPathComponent("global00", isDirectory: true)
        try FileManager.default.createDirectory(at: unit, withIntermediateDirectories: true)

        let tokenizerData = try jsonData(tokenizerObject())
        let tokenizerURL = artifact.appendingPathComponent("tokenizer.json")
        try tokenizerData.write(to: tokenizerURL)

        let configurationData = try testConfigurationData()
        let configurationURL = artifact.appendingPathComponent("config.json")
        try configurationData.write(to: configurationURL)

        let licenceData = Data("synthetic DeepSeek V4 test licence".utf8)
        let licenceURL = artifact.appendingPathComponent("LICENSE")
        try licenceData.write(to: licenceURL)

        let payloadData = Data([0x42])
        let payloadURL = unit.appendingPathComponent("payload.bin")
        try payloadData.write(to: payloadURL)

        let index = try indexData(
            sourceRevision: sourceRevision,
            configurationBytes: configurationData.count,
            configurationSHA256: FileDigestComputer.sha256(of: configurationData),
            tokenizerBytes: tokenizerData.count,
            tokenizerSHA256: FileDigestComputer.sha256(of: tokenizerData))
        let indexURL = artifact.appendingPathComponent("index.json")
        try index.write(to: indexURL)

        let verifiedFiles = try [
            verifiedFile(path: "global00/payload.bin", url: payloadURL, isPayload: true),
            verifiedFile(path: "config.json", url: configurationURL, isPayload: false),
            verifiedFile(path: "tokenizer.json", url: tokenizerURL, isPayload: false),
            verifiedFile(path: "LICENSE", url: licenceURL, isPayload: false),
            verifiedFile(path: "index.json", url: indexURL, isPayload: false),
        ]
        let repositoryFiles = verifiedFiles.map {
            RepoFile(
                path: $0.path, sizeBytes: $0.expectedSizeBytes,
                digest: $0.digest, isPayload: $0.isPayload)
        }
        let planIdentity = ArtifactDigestPlanIdentity.compute(repositoryFiles)
        let total = try verifiedFiles.reduce(UInt64(0)) { partial, file in
            let sum = partial.addingReportingOverflow(file.expectedSizeBytes)
            guard !sum.overflow else { throw TestError.overflow }
            return sum.partialValue
        }
        let metadata = total - UInt64(payloadData.count)
        let root = try ArtifactVerificationRoot.open(
            relativeComponents: ["artifact"], beneath: location)
        let evidence = ArtifactVerificationEvidence(
            model: .deepseekV4Flash,
            repository: HuggingFaceRepoRef(
                repoID: DeepSeekV4ProductArtifact.publicationRepository,
                revision: publicationRevision),
            treeIdentity: planIdentity,
            completenessAuthorityIdentity: planIdentity,
            selectedPlanIdentity: planIdentity,
            treeFileCount: verifiedFiles.count,
            treePayloadFileCount: 1,
            treeMetadataFileCount: verifiedFiles.count - 1,
            treeBytes: total,
            treePayloadBytes: UInt64(payloadData.count),
            treeMetadataBytes: metadata,
            selectedBytes: total,
            index: ArtifactIndexIdentity.parse(index),
            root: root.artifactIdentity,
            files: verifiedFiles)
        let authority = try ArtifactRuntimeAuthority(root: root, evidence: evidence)
        return Fixture(
            location: location, tokenizerURL: tokenizerURL,
            reference: ArtifactReference(root: artifact, runtimeAuthority: authority))
    }

    private func indexData(
        sourceRevision: String,
        tokenizerSourceRevision: String? = nil,
        configurationBytes: Int = 1,
        configurationSHA256: String = String(repeating: "a", count: 64),
        tokenizerBytes: Int = 1,
        tokenizerSHA256: String = String(repeating: "b", count: 64)
    ) throws -> Data {
        try jsonData([
            "relationship": "byte-preserving repack",
            "source_repo": DeepSeekV4ProductArtifact.sourceRepository,
            "source_revision": sourceRevision,
            "units": [["unit": "global00", "files": 1, "bytes": 1]],
            "total_bytes": 1,
            "configuration": [
                "file": "config.json", "bytes": configurationBytes,
                "sha256": configurationSHA256,
                "source_repo": DeepSeekV4ProductArtifact.sourceRepository,
                "source_revision": sourceRevision,
            ],
            "tokenizer": [
                "file": "tokenizer.json", "bytes": tokenizerBytes,
                "sha256": tokenizerSHA256,
                "source_repo": DeepSeekV4ProductArtifact.sourceRepository,
                "source_revision": tokenizerSourceRevision ?? sourceRevision,
                "license": "LICENSE",
            ],
        ])
    }

    private func testConfigurationData() throws -> Data {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent(
                "../../../Tests/Fixtures/deepseek-v4/current-official-config.json")
            .standardizedFileURL
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        object["vocab_size"] = 261
        object["bos_token_id"] = 256
        object["eos_token_id"] = 257
        return try jsonData(object)
    }

    private func tokenizerObject() -> [String: Any] {
        let alphabet = byteAlphabet()
        var vocabulary: [String: Int] = [:]
        for byte in 0..<256 { vocabulary[String(alphabet[byte])] = byte }

        let controls = DeepSeekV4ChatControl.allCases
        let added = controls.enumerated().map { offset, control -> [String: Any] in
            [
                "id": 256 + offset,
                "content": control.rawValue,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": true,
                "special": false,
            ]
        }
        return [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": added,
            "normalizer": ["type": "Sequence", "normalizers": []],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    split(#"\p{N}{1,3}"#),
                    split(#"[一-龥぀-ゟ゠-ヿ]+"#),
                    split(
                        ##"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##),
                    [
                        "type": "ByteLevel", "add_prefix_space": false,
                        "trim_offsets": true, "use_regex": false,
                    ],
                ],
            ],
            "post_processor": [
                "type": "ByteLevel", "add_prefix_space": true,
                "trim_offsets": false, "use_regex": true,
            ],
            "decoder": [
                "type": "ByteLevel", "add_prefix_space": true,
                "trim_offsets": true, "use_regex": true,
            ],
            "model": [
                "type": "BPE", "dropout": NSNull(), "unk_token": NSNull(),
                "continuing_subword_prefix": "", "end_of_word_suffix": "",
                "fuse_unk": false, "byte_fallback": false,
                "vocab": vocabulary, "merges": [],
            ],
        ]
    }

    private func split(_ regex: String) -> [String: Any] {
        [
            "type": "Split", "pattern": ["Regex": regex],
            "behavior": "Isolated", "invert": false,
        ]
    }

    private func byteAlphabet() -> [Unicode.Scalar] {
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

    private func jsonData(_ object: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func verifiedFile(
        path: String, url: URL, isPayload: Bool
    ) throws -> ArtifactVerifiedFile {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        defer { _ = Darwin.close(descriptor) }
        let filesystem = try ArtifactFilesystemEvidence.identity(
            of: descriptor, path: path, expectedDirectory: false)
        return ArtifactVerifiedFile(
            path: path,
            expectedSizeBytes: filesystem.sizeBytes,
            digest: .sha256(hex: try FileDigestComputer.sha256(ofFileAt: url)),
            isPayload: isPayload,
            filesystem: filesystem)
    }

    private enum TestError: Error { case overflow }
}
