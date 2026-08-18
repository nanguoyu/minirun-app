import Foundation

/// An opt-in dump of the routed expert ids every MoE layer selects.
///
/// ADR 0017's estimate for draft-and-verify rests on one unmeasured quantity:
/// the **union ratio** `u = |E_t ∪ E_{t+1}| / |E_t|` of the experts two
/// adjacent positions route to, per layer. A K-position pass reads that union
/// once instead of reading each position's set separately, so `u` is what
/// decides whether a wider pass costs more bytes than it saves. The record
/// says the measurement needs "no DSpark code at all — a 40-token arm already
/// selects 43 × 6 experts per pass"; this is the smallest way to get those
/// numbers out of an arm that is otherwise unchanged.
///
/// It is off unless `MINIRUN_V4_DUMP_ROUTING` names a writable path, and when
/// it is off every call site is one load and one branch. Nothing about the
/// arithmetic, the schedule, or the number of forcing points changes: the ids
/// recorded here were already computed, already on the host, and already used
/// to name tiles for the pager.
///
/// The format is one line per layer per call:
///
/// ```text
/// <sequenceNumber> <layer> <rows> <id>,<id>,…
/// ```
///
/// `rows` is the number of positions the call routed — 1 for a decode pass,
/// the prompt length for prefill — and the ids are row-major over
/// `rows × expertsPerToken`. A decode pass is therefore the maximal run of
/// consecutive `rows == 1` lines whose layer numbers ascend from the first
/// layer, which is what the analysis script reconstructs.
public final class DeepSeekV4RoutingTrace: @unchecked Sendable {
    /// Names the file the dump is written to. Unset ⇒ no dump at all.
    public static let environmentKey = "MINIRUN_V4_DUMP_ROUTING"

    /// The process-wide sink, resolved once from the environment.
    ///
    /// A singleton and not a threaded parameter on purpose: the routing
    /// decision is made three call frames below anything that holds a run's
    /// telemetry, and threading a diagnostic through the three block families
    /// would touch exactly the arithmetic this measurement must not disturb.
    public static let shared: DeepSeekV4RoutingTrace? = make()

    private static func make(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> DeepSeekV4RoutingTrace? {
        guard let path = environment[environmentKey]?
            .trimmingCharacters(in: .whitespaces), !path.isEmpty
        else { return nil }
        return DeepSeekV4RoutingTrace(path: path)
    }

    private let lock = NSLock()
    private let handle: FileHandle?
    private var sequenceNumber = 0

    init?(path: String) {
        let manager = FileManager.default
        manager.createFile(atPath: path, contents: nil)
        guard let handle = FileHandle(forWritingAtPath: path) else { return nil }
        self.handle = handle
    }

    /// Record one layer's routing decision. Cheap enough to sit on the decode
    /// path: a string built from ids the caller already holds, and one write.
    public func record(layer: Int, ids: [[Int]]) {
        let rows = ids.map { row in row.map(String.init).joined(separator: ",") }
            .joined(separator: ";")
        // Numbering and writing under one lock: the sequence number is only
        // useful if it agrees with the file's own order, and two decode threads
        // are not a shape this ever has to support anyway.
        lock.lock()
        sequenceNumber += 1
        let line = "\(sequenceNumber) \(layer) \(ids.count) \(rows)\n"
        if let data = line.data(using: .utf8) {
            try? handle?.write(contentsOf: data)
        }
        lock.unlock()
    }
}
