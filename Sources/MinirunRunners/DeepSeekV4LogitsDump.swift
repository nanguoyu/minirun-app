import Foundation
import StorageCore

/// An opt-in dump of the full final-logits vector at every generated position.
///
/// The V4 run record carries `logitsSHA256` — one digest, of the *last* pass's
/// logits — and a digest can only ever say "the same" or "different". When
/// `docs/experiments/2026-08-17-v4-dispatch-cuts.md` §5 measured the batched
/// head loop on the real artifact, the digest moved and the completion changed
/// from 40 ids to 38, and the record had to say in as many words that "the
/// maximum relative difference between the two arms' logits is *not* measurable
/// from these outputs". Deciding whether a non-bit-identical path is acceptable
/// needs the size of the difference, not only its existence. This is the
/// smallest thing that produces it: the K3 arm has written its logits since
/// `m9c-logits-run<tag>.bin` (`docs/experiments/2026-08-10-m9d-layer-oracle.md`)
/// and the V4 arm has not.
///
/// It is off unless `MINIRUN_V4_DUMP_LOGITS` names a directory, and when it is
/// off every call site is one load of a resolved-once static and one branch.
/// Nothing about the arithmetic, the schedule, or the number of forcing points
/// changes: the vector written here was already computed, already on the host,
/// and already checked for finiteness and length by the caller before a token
/// was chosen from it.
///
/// The format is one pair of files per generated position, in the named
/// directory:
///
/// ```text
/// logits-<position>.bin    vocabSize float32 words, little-endian, no header
/// logits-<position>.json   what that file is and what run produced it
/// ```
///
/// `<position>` is zero-based over the generated ids — position 0 is the token
/// prefill produced, and every later position is one decode pass — so the *n*th
/// vector and the *n*th entry of `generatedTokenIDs` describe the same step.
/// The `.bin` is raw so that any reader can `numpy.fromfile(…, "<f4")` it; the
/// sidecar carries everything needed to interpret and attribute it, including
/// the `MINIRUN_*` environment the run saw, because two dumps are only
/// comparable if what differed between them is written down.
final class DeepSeekV4LogitsDump: @unchecked Sendable {
    /// Names the directory the dump is written to. Unset ⇒ no dump at all.
    static let environmentKey = "MINIRUN_V4_DUMP_LOGITS"

    /// The process-wide sink, resolved once from the environment.
    ///
    /// A resolved-once static and not a run parameter, for the reason
    /// ``DeepSeekV4RoutingTrace/shared`` gives: this is a diagnostic that must
    /// not become a knob the product path can be configured into. A run that
    /// wants it says so in its environment, and every other run pays a load and
    /// a branch per generated token.
    static let shared: DeepSeekV4LogitsDump? = make()

    static func make(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> DeepSeekV4LogitsDump? {
        guard let path = environment[environmentKey]?
            .trimmingCharacters(in: .whitespaces), !path.isEmpty
        else { return nil }
        return DeepSeekV4LogitsDump(directoryPath: path, environment: environment)
    }

    /// Every `MINIRUN_*` variable the process was given, which is the part of
    /// the environment that can change what the arithmetic produces.
    ///
    /// Recorded per position rather than once per run so that a single pair of
    /// files is self-describing: a dump directory that has been moved, renamed,
    /// or half-copied still says which knobs produced the vector beside it.
    private let knobs: [String: String]
    private let directory: URL
    private let lock = NSLock()

    init?(directoryPath: String, environment: [String: String]) {
        let directory = URL(fileURLWithPath: directoryPath, isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
        } catch {
            // A dump that cannot be written is not a reason to fail a run that
            // was not asked to produce one. It is a reason to say so once.
            FileHandle.standardError.write(
                Data(
                    ("\(Self.environmentKey)=\(directoryPath) is not writable: "
                        + "\(error.localizedDescription)\n").utf8))
            return nil
        }
        self.directory = directory
        knobs = environment.filter { $0.key.hasPrefix("MINIRUN_") }
    }

    /// Write one position's vector and its sidecar.
    ///
    /// Failures are reported and not thrown: this is an instrument attached to
    /// a measurement arm, and a full disk must not turn a 100-second run into a
    /// refusal that looks like a model failure.
    func record(position: Int, logits: [Float], tokenID: Int, isPrefillToken: Bool) {
        let raw = Self.rawBytes(logits)
        let sidecar = Sidecar(
            file: Self.vectorName(position: position),
            position: position,
            tokenID: tokenID,
            isPrefillToken: isPrefillToken,
            vocabSize: logits.count,
            dtype: "float32",
            byteOrder: "little",
            byteCount: raw.count,
            // The same digest convention the run summary's `logitsSHA256` uses,
            // so the last position's sidecar and the run JSON must agree — which
            // is what says the dump is of the run it claims to be of.
            sha256: SHA256.hexString(SHA256.hash(raw)),
            knobs: knobs)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        lock.lock()
        defer { lock.unlock() }
        do {
            try raw.write(
                to: directory.appendingPathComponent(sidecar.file), options: .atomic)
            try encoder.encode(sidecar).write(
                to: directory.appendingPathComponent(
                    Self.sidecarName(position: position)), options: .atomic)
        } catch {
            FileHandle.standardError.write(
                Data(
                    ("logits dump failed at position \(position): "
                        + "\(error.localizedDescription)\n").utf8))
        }
    }

    static func vectorName(position: Int) -> String { "logits-\(position).bin" }
    static func sidecarName(position: Int) -> String { "logits-\(position).json" }

    /// The bytes a logits vector is both written as and digested over.
    ///
    /// One function so that the file on disk and the run summary's
    /// `logitsSHA256` are of the *same* bytes by construction rather than by
    /// two transcriptions of "float32, little-endian" agreeing. The run
    /// engine's digest calls this too.
    static func rawBytes(_ logits: [Float]) -> Data {
        var raw = Data(capacity: logits.count * MemoryLayout<UInt32>.size)
        for value in logits {
            withUnsafeBytes(of: value.bitPattern.littleEndian) { raw.append(contentsOf: $0) }
        }
        return raw
    }

    struct Sidecar: Codable, Equatable {
        let file: String
        let position: Int
        let tokenID: Int
        let isPrefillToken: Bool
        let vocabSize: Int
        let dtype: String
        let byteOrder: String
        let byteCount: Int
        let sha256: String
        let knobs: [String: String]
    }
}
