import Foundation
import ModelAdapters

/// The V4.1 arm's forward-trace sink: ``DeepSeekV41TraceWriter`` behind an
/// environment variable, the way ``DeepSeekV41LogitsDump`` is the logits sink.
///
/// `MINIRUN_V41_DUMP_TRACE=<dir>` turns it on; `MINIRUN_V41_DUMP_TRACE_PASSES`
/// says how many passes carry tensors (default 3: the prefill and the first two
/// decode passes). Routing, index selections and Engram rows are written for
/// every pass whatever that number is, because they are what a comparison with
/// the reference's `routing.json` and `engram.json` needs and they are small.
///
/// Only the **first** generation in the process is traced. The product gate
/// runs a completion arm and then a stop arm in one process; a second writer
/// into the same directory would interleave two runs' passes, and the
/// completion arm is the one the reference is compared with.
enum DeepSeekV41TraceDump {
    static let environmentKey = "MINIRUN_V41_DUMP_TRACE"
    static let passesKey = "MINIRUN_V41_DUMP_TRACE_PASSES"
    static let defaultTensorPasses = 3

    private static let lock = NSLock()
    nonisolated(unsafe) private static var claimed = false

    /// A writer for this generation, or nil when tracing is off or a previous
    /// generation in this process already took it.
    static func claim(
        metadata: [String: String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> DeepSeekV41TraceWriter? {
        guard let path = environment[environmentKey]?
            .trimmingCharacters(in: .whitespaces), !path.isEmpty
        else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard !claimed else { return nil }
        claimed = true
        let passes = environment[passesKey].flatMap { Int($0) } ?? defaultTensorPasses
        var stamped = metadata
        for (key, value) in environment where key.hasPrefix("MINIRUN_") {
            stamped["env." + key] = value
        }
        do {
            return try make(path: path, passes: passes, metadata: stamped)
        } catch {
            FileHandle.standardError.write(
                Data(
                    ("\(environmentKey)=\(path) is not writable: "
                        + "\(error.localizedDescription)\n").utf8))
            return nil
        }
    }

    static func make(
        path: String, passes: Int, metadata: [String: String]
    ) throws -> DeepSeekV41TraceWriter {
        try DeepSeekV41TraceWriter(
            directory: URL(fileURLWithPath: path, isDirectory: true),
            tensorPasses: passes,
            producer: "minirun-swift-mlx",
            metadata: metadata)
    }
}
