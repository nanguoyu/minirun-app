import Foundation

/// The V4.1 arm's per-position logits sink.
///
/// The same instrument ``DeepSeekV4LogitsDump`` is, behind its own environment
/// variable, because the two runners are separate arms and a run that dumped
/// both models' vectors into one directory would produce files nobody could
/// attribute. The *format* is deliberately identical — raw little-endian float32
/// beside a JSON sidecar — so the comparison a phase-3 gate makes between a
/// Minirun run and a reference run is one `numpy.fromfile` on either side.
enum DeepSeekV41LogitsDump {
    static let environmentKey = "MINIRUN_V41_DUMP_LOGITS"

    /// Resolved once from the environment, and nil unless it names a directory.
    static let shared: DeepSeekV4LogitsDump? = make()

    static func make(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> DeepSeekV4LogitsDump? {
        guard let path = environment[environmentKey]?
            .trimmingCharacters(in: .whitespaces), !path.isEmpty
        else { return nil }
        return DeepSeekV4LogitsDump(directoryPath: path, environment: environment)
    }
}
