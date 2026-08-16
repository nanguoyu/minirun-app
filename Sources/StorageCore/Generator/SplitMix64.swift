import Foundation

/// SplitMix64, used both as a sequential stream and as a counter-based
/// function.
///
/// The counter-based form is the load-bearing property for this project. A
/// benchmark that reads block 90,000 of a file must be able to say what those
/// bytes should contain without generating the 89,999 blocks before it, so the
/// generator cannot carry hidden state between blocks. SplitMix64's state
/// advance is a plain addition, which means the n-th output of a stream is a
/// closed-form function of the seed and n — `value(seed:index:)` below — and
/// verification of any offset costs the same as verification of the first.
public struct SplitMix64: RandomNumberGenerator {
    /// Fractional part of the golden ratio, the standard SplitMix64 increment.
    public static let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15

    public private(set) var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state = state &+ Self.gamma
        return Self.mix(state)
    }

    /// SplitMix64 finalizer (MurmurHash3-style avalanche).
    @inline(__always)
    public static func mix(_ input: UInt64) -> UInt64 {
        var z = input
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// The `index`-th output (zero-based) of a stream seeded with `seed`,
    /// computed directly. Equivalent to calling `next()` `index + 1` times on
    /// `SplitMix64(seed: seed)`, and asserted as such by the tests.
    @inline(__always)
    public static func value(seed: UInt64, index: UInt64) -> UInt64 {
        mix(seed &+ (index &+ 1) &* gamma)
    }

    /// Derives an independent stream seed from a base seed and a label. Used so
    /// that offset selection and file content can share one user-visible
    /// `--seed` without the two streams correlating.
    public static func derive(seed: UInt64, label: UInt64) -> UInt64 {
        mix(seed ^ mix(label &+ 0xD1B5_4A32_D192_ED03))
    }
}
