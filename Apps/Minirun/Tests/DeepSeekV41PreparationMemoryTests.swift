import Darwin
import Foundation
import ModelAdapters
import StorageCore
import XCTest

@testable import MinirunApp
@testable import MinirunKit

/// **What a V4.1 chat allocates before it has read a weight, measured on the
/// platform that is killed for it.**
///
/// ## The run this exists for
///
/// On 2026-09-11 three chats on the owner's iPhone 16 Pro (8 GB) were killed by
/// Jetsam — `vm-pageshortage`, 5,234 / 5,084 / ~5,100 MB resident — and the
/// flight recorder's trace for the third was a start line and *nothing else*:
///
/// ```text
/// scale product · declared budget 1.900 GB · max new tokens 64 · prompt tokens 11
/// entry footprint 0.111 GB · resident 0.213 GB · device available 6.331 GB
/// pin plan none -- every block streams
/// NO END LINE
/// ```
///
/// 0.111 GB when the run was handed over, about 5.1 GB when the process died,
/// and not one sample in between. The same run on a Mac adds 0.73 GB at its
/// peak.
///
/// It was not the file-access path, the Engram page reads, the expert pool or
/// the MLX allocator. It was **one loop in `DeepSeekV41Model.init`**: a rotary
/// table per backbone block, sized by the checkpoint's declared
/// `max_position_embeddings` rather than by the chat. DeepSeek V4.1 Flash
/// declares 1,048,576 positions at `rope_head_dim = 64` over forty blocks —
/// 40 x 1,048,576 x 32 x (4 + 4) B = **10,737,418,240 B, exactly 10.0 GiB** of
/// float32 cosines and sines — built inside `factory.prepare`, before the
/// budget floor is taken and before the first payload byte is read. 268 MB a
/// block; the phone died between the eighteenth and the nineteenth.
///
/// A Mac never noticed, because the tables were allocated *before*
/// `prepareForExecution()` and so were charged to the floor the budget is
/// measured from rather than to the run. That floor was recorded as "10.0 GiB"
/// and attributed to the 517 GB verification pass. It was these tables.
///
/// ## Why the assertions are the ones below
///
/// The cause is arithmetic and a ceiling, so most of this is arithmetic and a
/// ceiling — plus one real `phys_footprint` measurement of the allocation
/// itself, taken in the iOS test host, because a bound that has only ever been
/// computed is not a bound that has been observed.
///
/// The configuration is read through an ``ArtifactRuntimeAuthority`` rather
/// than from a path, because that is the opener the app uses on iOS: every
/// component `O_NOFOLLOW`, the recorded identity rechecked, the descriptor
/// handed over. It is also the first candidate the investigation ruled out, and
/// ruling it out is worth keeping.
final class DeepSeekV41PreparationMemoryTests: XCTestCase {

    /// One backbone block's tables at the published geometry:
    /// 1,048,576 positions x 32 pairs x (4 B cosine + 4 B sine).
    private static let publishedBytesPerBlock: UInt64 = 268_435_456

    // MARK: The published configuration, as the phone reads it

    /// `Tests/Fixtures/deepseek-v41/` holds the real published pair —
    /// `inference-config.json` (the authority, ADR 0020) and `config.json` (the
    /// cross-check) — for the checkpoint that ships. Not the mini checkpoint:
    /// the numbers that killed the phone are this one's.
    private static var publishedConfigurationDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("../../../Tests/Fixtures/deepseek-v41")
            .standardizedFileURL
    }

    private var scratch = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        scratch = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-v41-prepare-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    /// The two configuration documents, read the way
    /// `DeepSeekV41ArtifactWorkloadFactory` reads them: beneath a rooted
    /// runtime authority, by repository-relative name, never by path.
    private func publishedConfig() throws -> DeepSeekV41Config {
        let root = scratch.appendingPathComponent("artifact", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var files: [ArtifactVerifiedFile] = []
        for name in ["config.json", "inference-config.json"] {
            let destination = root.appendingPathComponent(name, isDirectory: false)
            try FileManager.default.copyItem(
                at: Self.publishedConfigurationDirectory.appendingPathComponent(name),
                to: destination)
            files.append(try verifiedFile(path: name, url: destination, isPayload: false))
        }
        // Complete evidence describes at least one payload file, because a
        // container with no payload is not an artifact. This stands in for the
        // 517 GB of units a configuration read does not touch, and keeps the
        // evidence the shape `ArtifactRuntimeAuthority` insists on.
        let unit = root.appendingPathComponent("global00-stand-in.bin", isDirectory: false)
        try Data("not a unit".utf8).write(to: unit)
        files.append(
            try verifiedFile(path: "global00-stand-in.bin", url: unit, isPayload: true))

        let rooted = try ArtifactVerificationRoot.open(
            relativeComponents: ["artifact"], beneath: scratch)
        let identity = ArtifactDigestPlanIdentity.compute(
            files.map {
                RepoFile(
                    path: $0.path, sizeBytes: $0.expectedSizeBytes, digest: $0.digest,
                    isPayload: $0.isPayload)
            })
        let payloadBytes = files.filter(\.isPayload)
            .reduce(UInt64(0)) { $0 + $1.expectedSizeBytes }
        let metadataBytes = files.filter { !$0.isPayload }
            .reduce(UInt64(0)) { $0 + $1.expectedSizeBytes }
        let evidence = ArtifactVerificationEvidence(
            model: .deepseekV41Flash,
            repository: HuggingFaceRepoRef(
                repoID: "nanguoyu/DeepSeek-V4.1-Flash-minirun",
                revision: String(repeating: "a", count: 40)),
            treeIdentity: identity,
            completenessAuthorityIdentity: identity,
            selectedPlanIdentity: identity,
            treeFileCount: files.count,
            treePayloadFileCount: files.filter(\.isPayload).count,
            treeMetadataFileCount: files.filter { !$0.isPayload }.count,
            treeBytes: payloadBytes + metadataBytes,
            treePayloadBytes: payloadBytes,
            treeMetadataBytes: metadataBytes,
            selectedBytes: payloadBytes + metadataBytes,
            index: ArtifactIndexIdentity.parse(Data("{}".utf8)),
            root: rooted.artifactIdentity,
            files: files)
        let authority = try ArtifactRuntimeAuthority(root: rooted, evidence: evidence)

        // `openFile(_:).readAll(maximumBytes:)`, exactly as the factory does —
        // and the same shape of read that candidate (a) of the investigation
        // asked about. Two configuration documents, a megabyte apiece at most.
        let huggingFace = try authority.openFile("config.json").readAll(maximumBytes: 1 << 20)
        let inference = try authority.openFile("inference-config.json")
            .readAll(maximumBytes: 1 << 20)
        return try DeepSeekV41Config(
            inferenceJSON: inference, huggingFaceJSON: huggingFace)
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
            path: path, expectedSizeBytes: filesystem.sizeBytes,
            digest: .sha256(hex: try FileDigestComputer.sha256(ofFileAt: url)),
            isPayload: isPayload,
            filesystem: filesystem)
    }

    private func footprintBytes() throws -> UInt64 {
        try XCTUnwrap(ProcessFootprint.current()).footprintBytes
    }

    // MARK: The geometry, and what it would have cost

    /// The numbers the kill report has to be explained by, stated as arithmetic
    /// over the published configuration rather than as constants written here.
    func testThePublishedCheckpointDeclaresAMillionPositionsAndFortyBlocks() throws {
        let config = try publishedConfig()
        XCTAssertEqual(config.numberOfLayers, 40)
        XCTAssertEqual(config.ropeHeadDimension, 64)
        XCTAssertEqual(config.maximumPositionCount, 1_048_576)

        let ceiling = DeepSeekV41Model.rotaryTableBytes(
            positionCount: config.maximumPositionCount, config: config)
        // Exactly 10.0 GiB, and exactly what `DeepSeekV41Model.init` allocated
        // inside `factory.prepare` before 2026-09-11.
        XCTAssertEqual(ceiling, 10_737_418_240)
        XCTAssertEqual(ceiling, 10 << 30)
        XCTAssertEqual(
            ceiling / UInt64(config.numberOfLayers), Self.publishedBytesPerBlock)

        // The kills bracket: with 0.111-0.213 GB already held when the run was
        // handed over, eighteen blocks is under five gigabytes and nineteen is
        // over, and the three reports were 5,084 / ~5,100 / 5,234 MB.
        XCTAssertLessThan(18 * Self.publishedBytesPerBlock, 5_000_000_000)
        XCTAssertGreaterThan(19 * Self.publishedBytesPerBlock, 5_000_000_000)
    }

    /// The chat that was killed asked for eleven prompt tokens and 64 new ones.
    /// Seventy-four positions, not 1,048,576.
    func testTheTablesAChatNeedsAreTheChatsPositionsAndNotTheCheckpointsCeiling() throws {
        let config = try publishedConfig()

        // The product's own widest chat: 512 prompt tokens and 64 new ones.
        let widest = 512 + 64 - 1
        let widestBytes = DeepSeekV41Model.rotaryTableBytes(
            positionCount: widest, config: config)
        XCTAssertEqual(widestBytes, 5_888_000)
        XCTAssertLessThan(widestBytes, DeepSeekV41Model.rotaryTableBudgetBytes)

        // And the chat the trace above belongs to: 74 positions, 758 kB.
        XCTAssertEqual(
            DeepSeekV41Model.rotaryTableBytes(positionCount: 11 + 64 - 1, config: config),
            757_760)

        // A model asked for the checkpoint's ceiling is refused by name, before
        // it allocates, rather than killed while it does. The refusal itself is
        // `DeepSeekV41ModelTests`' subject; what is asserted here is that the
        // stated envelope is the side of the line these two numbers fall on.
        XCTAssertGreaterThan(
            DeepSeekV41Model.rotaryTableBytes(
                positionCount: config.maximumPositionCount, config: config),
            DeepSeekV41Model.rotaryTableBudgetBytes)
    }

    // MARK: The measurement

    /// `phys_footprint` across the allocation itself, in the iOS test host.
    ///
    /// This is the one assertion here that is an observation rather than a
    /// product: building every rotary table a product chat needs must not move
    /// the number Jetsam judges this process by more than a stated bound. The
    /// tables for a 576-position chat are 5,888,000 B of payload; 64 MB is
    /// generous room for the allocator's own rounding and for whatever else the
    /// test host does between the two readings, and it is 182 times below the
    /// 10.0 GiB the same loop used to take.
    func testBuildingAChatsRotaryTablesBarelyMovesThePhysFootprint() throws {
        let config = try publishedConfig()
        let positions = 512 + 64 - 1

        // Warm the reading: the first `task_info` in a process, and the first
        // `pow`/`cos` through libm, are not what this measures.
        _ = try DeepSeekV41RotaryTable.forBlock(0, config: config, positionCount: 1)
        let before = try footprintBytes()

        var tables: [DeepSeekV41RotaryTable] = []
        tables.reserveCapacity(config.numberOfLayers)
        for block in 0..<config.numberOfLayers {
            tables.append(
                try DeepSeekV41RotaryTable.forBlock(
                    block, config: config, positionCount: positions))
        }
        let after = try footprintBytes()

        XCTAssertEqual(tables.count, 40)
        XCTAssertEqual(tables[0].cosines.count, positions * 32)
        XCTAssertEqual(tables[39].sines.count, positions * 32)

        let grew = after > before ? after - before : 0
        XCTAssertLessThan(
            grew, 64 << 20,
            "preparing a \(positions)-position chat grew phys_footprint by \(grew) B "
                + "against tables of "
                + "\(DeepSeekV41Model.rotaryTableBytes(positionCount: positions, config: config)) B")
    }
}
