import Foundation
import MLX

/// Which of V4's pure-arithmetic subgraphs are handed to `MLX.compile`, and the
/// escape hatch that takes them all back.
///
/// `docs/experiments/2026-08-17-v4-mlx-dispatch.md` measured a decode pass at
/// ~135,000 MLX primitives and `mlx::core::detail::compile` entered exactly zero
/// times. Two of the families it censused are pure functions of their inputs
/// with no host round trip inside them, which is the precondition `MLX.compile`
/// needs, and both were measured compiled on the fixture:
///
/// | subgraph | uncompiled | compiled | bits |
/// | --- | ---: | ---: | --- |
/// | `splitSinkhorn` at 20 iterations | 0.7200 ms | 0.3904 ms | 0 of 16 words differ |
/// | `roundAndDequantize` | 0.1180 ms | 0.0431 ms | bit-identical over the tie sweep |
///
/// **Compilation is a fusion, and a fused kernel may cancel what separate
/// kernels cannot** — `roundToFiniteE4M3`'s subnormal branch is literally
/// `(m + c) - c`. That is why each family has its own gate
/// (``DeepSeekV4CompiledSubgraphTests``, `DeepSeekV4FP8RoundingTests`) and why
/// this type exists at all rather than a bare `compile { … }` at each call
/// site: the shipped default is a decision the 40-token digest settles, and it
/// has to be reversible from the environment without a rebuild.
public enum DeepSeekV4Compile {

    /// `MINIRUN_V4_COMPILE=0` restores the uncompiled path for *every* family
    /// below, which is the condition every measurement taken before this change
    /// was taken under.
    public static let escapeHatchVariable = "MINIRUN_V4_COMPILE"

    /// `MINIRUN_V4_COMPILE_FP8=0|1` overrides the FP8 chain alone, so an arm can
    /// separate the two families without rebuilding.
    public static let fp8Variable = "MINIRUN_V4_COMPILE_FP8"

    /// The Sinkhorn loop ships compiled: it is 16 floats of arithmetic behind
    /// 188 primitives, its compiled form is bit-identical on the fixture, and
    /// the 40-token arm reproduced the published digest with it on.
    public static let sinkhornCompiledByDefault = true

    /// The FP8 post-scale chain ships compiled on the same evidence: the tie
    /// sweep is green on this mlx pin and the 40-token arm reproduced the
    /// published digest with it on. **Flip this to `false` — or set
    /// `MINIRUN_V4_COMPILE_FP8=0` — the moment an arm's digest moves**; every
    /// distinct activation shape the decode path presents is its own generated
    /// kernel, and only a run over all of them can say the whole set is safe.
    public static let fp8ChainCompiledByDefault = true

    /// Whether ``DeepSeekV4HyperConnections/splitSinkhorn(mixes:scale:base:multiplicity:iterations:epsilon:phaseAccounting:diagnostics:compiling:)``
    /// compiles its tail, resolved once from the process environment.
    public static let compilesSinkhorn = resolvesSinkhorn(
        in: ProcessInfo.processInfo.environment)

    /// Whether ``DeepSeekV4FP8ActivationReference/quantizeDequantize(_:blockSize:phaseAccounting:diagnostics:compiling:)``
    /// compiles its post-scale chain, resolved once from the process
    /// environment.
    public static let compilesFP8Chain = resolvesFP8Chain(
        in: ProcessInfo.processInfo.environment)

    /// The escape hatch, as a function of an environment rather than of *the*
    /// environment, so a test can state it.
    static func escapeHatchAllowsCompilation(in environment: [String: String]) -> Bool {
        environment[escapeHatchVariable] != "0"
    }

    static func resolvesSinkhorn(in environment: [String: String]) -> Bool {
        escapeHatchAllowsCompilation(in: environment) && sinkhornCompiledByDefault
    }

    static func resolvesFP8Chain(in environment: [String: String]) -> Bool {
        guard escapeHatchAllowsCompilation(in: environment) else { return false }
        switch environment[fp8Variable] {
        case "1": return true
        case "0": return false
        default: return fp8ChainCompiledByDefault
        }
    }
}

/// The compiled closures themselves, built once and kept.
///
/// `MLX.compile` caches its trace against the *identity of the closure it was
/// given*, so a `compile { … }` written at a call site is a fresh compilation
/// on every call and strictly slower than not compiling at all. Everything here
/// is therefore a stored closure, and the Sinkhorn's — whose graph depends on
/// three Swift constants that are not MLX inputs — is keyed by those constants.
enum DeepSeekV4CompiledSubgraphs {

    /// Everything after the FP8 block scale is chosen. One shape per call site;
    /// mlx re-traces per shape behind this single closure.
    static let fp8RoundAndDequantize: @Sendable (MLXArray, MLXArray) -> MLXArray = compile {
        blocked, scale in
        DeepSeekV4FP8ActivationReference.roundAndDequantize(blocked: blocked, scale: scale)
    }

    private struct SinkhornShape: Hashable {
        let multiplicity: Int
        let iterations: Int
        let epsilon: UInt32
    }

    private static let lock = NSLock()
    private static var sinkhornCache = [SinkhornShape: @Sendable ([MLXArray]) -> [MLXArray]]()

    /// `(mixes, scale, base) -> (pre, post, combination)`, compiled once per
    /// `(multiplicity, iterations, epsilon)` the process ever asks for. A decode
    /// run asks for exactly one.
    static func sinkhornTail(
        multiplicity: Int, iterations: Int, epsilon: Float
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        let shape = SinkhornShape(
            multiplicity: multiplicity, iterations: iterations, epsilon: epsilon.bitPattern)
        lock.lock()
        defer { lock.unlock() }
        if let existing = sinkhornCache[shape] { return existing }
        let compiled = compile { (arrays: [MLXArray]) -> [MLXArray] in
            DeepSeekV4HyperConnections.sinkhornTail(
                mixes: arrays[0], scale: arrays[1], base: arrays[2],
                multiplicity: multiplicity, iterations: iterations, epsilon: epsilon)
        }
        sinkhornCache[shape] = compiled
        return compiled
    }
}
