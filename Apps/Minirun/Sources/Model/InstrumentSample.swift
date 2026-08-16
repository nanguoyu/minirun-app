import Foundation
import MinirunKit

// =============================================================================
// MARK: - Synthetic preview/test instrumentation
// =============================================================================
//
// Product runners report expert/cache/read-ahead evidence in
// `RunTelemetry.instrumentation` and publish model payload flow through the
// runner-owned fixed-window stream on `RunSession`. The Debug preview runner
// predates those product bindings and still publishes its synthetic
// ladder/flow values through this second stream:
//
//   D · layer ladder      — per-layer status, stored/widened bytes, wall seconds
//   E · byte-flow ribbon  — the deterministic/expert split at sample rate
//   F · overlap meter     — the `Stalls` pair and the read-ahead wait/stage pair
//   G · cache strip       — hits, misses, evictions, rejected admissions, pinned
//
// This is not execution authority and the live K3 product path does not depend
// on it. Missing samples stay missing; the panel never turns an absent preview
// counter into a measured zero.

/// One synthetic preview/test sample outside the canonical product telemetry.
struct InstrumentSample: Sendable, Equatable {
    var at: Date
    /// Each field is optional independently. A runner can publish the evidence
    /// it has without manufacturing zeroes for instruments it cannot report.
    var ladder: [LayerCell]?
    var flow: FlowSample?
    var stalls: StallAccounting?
    var cache: CacheStrip?
    var readAhead: ReadAheadAccounting?

    init(
        at: Date, ladder: [LayerCell]? = nil, flow: FlowSample? = nil,
        stalls: StallAccounting? = nil, cache: CacheStrip? = nil,
        readAhead: ReadAheadAccounting? = nil
    ) {
        self.at = at
        self.ladder = ladder
        self.flow = flow
        self.stalls = stalls
        self.cache = cache
        self.readAhead = readAhead
    }
}

/// A synthetic runner that can also feed preview-only instrument values.
protocol InstrumentedRunner: RunnerFacade {
    /// Live for the duration of the run identified by `handle`. Finishes when
    /// the run does.
    func instrumentation(for handle: RunHandleID) -> AsyncStream<InstrumentSample>
}
