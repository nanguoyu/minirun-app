/// Whether a V4 execution runs its guard-only finiteness sweeps.
///
/// Several V4 operators open with a sweep that exists only to produce a better
/// error message: `isFinite(x).all().item(Bool.self)` over an activation or a
/// parameter table, or a host pull of a small table followed by
/// `allSatisfy(\.isFinite)`. Each one is a full host sync — it forces whatever
/// lazy graph produced the value before any of the operator's own arithmetic is
/// expressed — and the 2026-08-15 attributed decode pass measured 553 of them
/// costing 0.39 s of a 6.6 s pass, 5.9% of the token.
///
/// A sweep is diagnostic, not structural: no result depends on it, and nothing
/// downstream is safe only because a sweep ran. What makes a corrupted result
/// fail closed is the checks that read the *outcome* rather than the inputs —
/// ``DeepSeekV4GlobalArtifact/logits(for:chunkRows:cancellationCheck:boundaryReclaim:)``
/// refuses a logit window containing a non-finite value, `greedyToken` refuses
/// non-finite logits before a token is chosen, and the output head's own
/// parameters are swept once on admission in `readHeadParameters` rather than
/// on every token. A run with the sweeps off therefore still refuses to emit a
/// token computed from a non-finite value; it finds out once at the end of the
/// pass instead of 553 times inside it.
///
/// The default at every declaration site is ``validating``: byte-identical to
/// the behavior before this switch existed, including every thrown message. A
/// caller states ``off`` when it wants the product's arithmetic and is content
/// with the downstream checks — the runner does exactly that, for both of its
/// scales. Oracle and guard tests keep the default and are unaffected.
public struct DeepSeekV4Diagnostics: Sendable, Equatable {
    /// Run the guard-only finiteness sweeps.
    public var validateFiniteness: Bool

    public init(validateFiniteness: Bool) {
        self.validateFiniteness = validateFiniteness
    }

    /// Every guard-only sweep runs. The default at every declaration site.
    public static let validating = DeepSeekV4Diagnostics(validateFiniteness: true)

    /// No guard-only sweep runs. Every guard's throw path is still present and
    /// still reachable when a caller asks for ``validating``; what this value
    /// removes is the host sync and the check, nothing else.
    public static let off = DeepSeekV4Diagnostics(validateFiniteness: false)
}
