import Foundation
import MinirunKit

/// Which models this build can actually decode.
///
/// A model whose artifact is published but whose runner is not in this build is
/// **absent from the map**, not represented by a stub that fails at `start`.
/// The catalog's `RunnerAvailability.none` and a `nil` here are the same fact,
/// and the app is expected to read it before it offers a Run button.
public enum MinirunRunnerRegistry {

    public static func runner(for model: ModelID) -> (any RunnerFacade)? { all[model] }

    /// One entry today. Every other published model is absent rather than
    /// stubbed: a facade this build cannot test would be an untested promise
    /// rather than a runner.
    public static var all: [ModelID: any RunnerFacade] {
        [.kimiK3: K3DecodeRunner()]
    }
}
