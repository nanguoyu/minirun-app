import Foundation
import XCTest

@testable import MinirunRunners

/// Does the opt-in logits dump write what a deviation analysis can read back,
/// and does it write nothing at all when nobody asked for it?
///
/// The dump exists to answer one question the run JSON cannot —
/// `docs/experiments/2026-08-17-v4-batched-heads-logits.md`, "how far apart are
/// the two paths' logits" — and that answer is only worth as much as the
/// round-trip beneath it. A float32 vector that loses its last place on the way
/// to disk would make every deviation in that record its own artefact, so the
/// gate here is bit equality of the raw words, not a tolerance.
final class DeepSeekV4LogitsDumpTests: XCTestCase {

    private func scratch() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("v4-logits-dump-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    /// Values chosen so a lossy round-trip cannot hide: both signs, several
    /// binades, a subnormal, and two neighbours one ulp apart.
    private func fixture() -> [Float] {
        var values: [Float] = [
            0, -0, 1, -1, .leastNonzeroMagnitude, -.leastNormalMagnitude,
            .greatestFiniteMagnitude, 3.141_592_6, -2.718_281_8,
        ]
        values.append(Float(bitPattern: values[7].bitPattern &+ 1))
        for index in 0..<64 {
            values.append(Float(index % 9 == 0 ? -1 : 1) * Float(index) * 1.000_000_1e-3)
        }
        return values
    }

    // MARK: The round trip

    func testEveryPositionRoundTripsBitForBit() throws {
        let directory = try scratch()
        let dump = try XCTUnwrap(
            DeepSeekV4LogitsDump(
                directoryPath: directory.path,
                environment: ["MINIRUN_V4_BATCH_HEADS": "1", "PATH": "/ignored"]))
        let logits = fixture()

        for position in 0..<3 {
            dump.record(
                position: position, logits: logits, tokenID: 671 + position,
                isPrefillToken: position == 0)
        }

        for position in 0..<3 {
            let raw = try Data(
                contentsOf: directory.appendingPathComponent(
                    DeepSeekV4LogitsDump.vectorName(position: position)))
            XCTAssertEqual(raw.count, logits.count * 4, "position \(position)")

            var readBack = [Float]()
            for index in 0..<logits.count {
                let word = (0..<4).reversed().reduce(UInt32(0)) { partial, byte in
                    (partial << 8) | UInt32(raw[index * 4 + byte])
                }
                readBack.append(Float(bitPattern: word))
            }
            // Bit patterns, not values: `-0 == 0` and `nan != nan` would both
            // let a broken writer through an `XCTAssertEqual` on `[Float]`.
            XCTAssertEqual(
                readBack.map(\.bitPattern), logits.map(\.bitPattern),
                "position \(position) did not round-trip")

            let sidecar = try JSONDecoder().decode(
                DeepSeekV4LogitsDump.Sidecar.self,
                from: Data(
                    contentsOf: directory.appendingPathComponent(
                        DeepSeekV4LogitsDump.sidecarName(position: position))))
            XCTAssertEqual(sidecar.position, position)
            XCTAssertEqual(sidecar.tokenID, 671 + position)
            XCTAssertEqual(sidecar.isPrefillToken, position == 0)
            XCTAssertEqual(sidecar.vocabSize, logits.count)
            XCTAssertEqual(sidecar.dtype, "float32")
            XCTAssertEqual(sidecar.byteOrder, "little")
            XCTAssertEqual(sidecar.byteCount, raw.count)
            XCTAssertEqual(sidecar.file, "logits-\(position).bin")
            XCTAssertEqual(sidecar.sha256.count, 64)
            // Only the run's own knobs, and all of them: a dump whose sidecar
            // did not say which arm produced it would make two directories
            // indistinguishable.
            XCTAssertEqual(sidecar.knobs, ["MINIRUN_V4_BATCH_HEADS": "1"])
        }
    }

    /// The digest in the sidecar is the one the run summary reports, so the
    /// last position's vector can be tied to the arm that claims to have
    /// produced it.
    func testTheSidecarDigestIsTheRunSummaryDigest() throws {
        let directory = try scratch()
        let dump = try XCTUnwrap(
            DeepSeekV4LogitsDump(directoryPath: directory.path, environment: [:]))
        let logits = fixture()
        dump.record(position: 0, logits: logits, tokenID: 7, isPrefillToken: true)

        let raw = try Data(
            contentsOf: directory.appendingPathComponent(
                DeepSeekV4LogitsDump.vectorName(position: 0)))
        let sidecar = try JSONDecoder().decode(
            DeepSeekV4LogitsDump.Sidecar.self,
            from: Data(
                contentsOf: directory.appendingPathComponent(
                    DeepSeekV4LogitsDump.sidecarName(position: 0))))
        XCTAssertEqual(sidecar.sha256, DeepSeekV4RunEngine.logitsDigest(logits))
        XCTAssertEqual(sidecar.byteCount, raw.count)
    }

    // MARK: Off is off

    /// With the variable unset there is no sink, and therefore no file.
    ///
    /// Asserted on the resolver rather than on a directory listing because
    /// "nothing was written" is only meaningful if nothing *could* have been:
    /// the engine holds `nil` and the call site is a branch.
    func testNothingIsWrittenWithoutTheVariable() throws {
        XCTAssertNil(DeepSeekV4LogitsDump.make(environment: [:]))
        XCTAssertNil(
            DeepSeekV4LogitsDump.make(environment: ["MINIRUN_V4_DUMP_LOGITS": ""]))
        XCTAssertNil(
            DeepSeekV4LogitsDump.make(environment: ["MINIRUN_V4_DUMP_LOGITS": "   "]))
        XCTAssertNil(
            DeepSeekV4LogitsDump.make(environment: ["MINIRUN_V4_DUMP_ROUTING": "/tmp/x"]))

        // And the directory a run would have used stays empty, because the run
        // never made one: an unset variable does not create the path it does
        // not name.
        let directory = try scratch()
        XCTAssertNil(DeepSeekV4LogitsDump.make(environment: [:]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    func testTheVariableNamesTheDirectoryItCreates() throws {
        let directory = try scratch().appendingPathComponent("nested", isDirectory: true)
        let dump = try XCTUnwrap(
            DeepSeekV4LogitsDump.make(
                environment: [DeepSeekV4LogitsDump.environmentKey: " \(directory.path) "]))
        dump.record(position: 4, logits: [1, 2, 3], tokenID: 9, isPrefillToken: false)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("logits-4.bin").path))
        XCTAssertEqual(
            try Data(contentsOf: directory.appendingPathComponent("logits-4.bin")).count, 12)
    }
}
