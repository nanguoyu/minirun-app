import Foundation

/// The layers whose expert choice a decode pass knows before it starts.
///
/// V4's first `num_hash_layers` layers replace the learned router's *choice*
/// (not its weights) with a `[vocabulary, expertsPerToken]` integer table:
/// `tid2eid`. So for those layers, and only those, the six experts a token will
/// select are a table lookup on the token id — available at the top of the pass,
/// before the embedding is read, and certainly before the layer runs. The spec
/// has flagged this since v0.6.2 ("arbitrarily prefetchable", §15 expansion
/// track) and the 2026-08-14 audit's P1-8 quantified it; nothing collected it,
/// because the backend that would have held the reads was destroyed at each
/// layer's own boundary.
///
/// At the published geometry this is **3 of 43 layers, 3 x 3 x 6 = 54 tiles =
/// 240,651,648 B per token, 6.98% of the 3,449,340,288 B a decode pass reads.**
/// The dividend is exactly computable and it is small; the reason to collect it
/// is that it is free once the backends outlive their layers, not that it is
/// large.
///
/// The table is read through ``DeepSeekV4LayerArtifact/loadTokenExpertMap``,
/// which retains the first successful load for the artifact's lifetime, so
/// building this plan re-reads nothing after the first pass that touched those
/// layers — and the prefill always has.
extension DeepSeekV4ModelArtifact {

    /// Every hash-routed layer paired with the token table that decides its
    /// experts, in layer order.
    ///
    /// Returns an empty array for a model with no hash layers, which is then
    /// simply a run with no cross-layer dividend to collect. A layer whose
    /// table cannot be read is omitted rather than failing the run: this is a
    /// scheduling hint, and the layer will read exactly the same bytes on
    /// demand. Nothing here changes what is computed.
    public func hashRoutedExpertPlan(
        cancellationCheck: () throws -> Void = {}
    ) -> [DeepSeekV4RunExpertBackends.HashRoutedLayer] {
        var planned: [DeepSeekV4RunExpertBackends.HashRoutedLayer] = []
        for layer in layers where layer.plan.usesHashRouting {
            guard let expertsPerToken = layer.plan.hashRoutedExpertsPerToken,
                let expertCount = layer.plan.hashRoutedExpertCount
            else { continue }
            guard let map = try? layer.artifact.loadTokenExpertMap(
                tensor: "layers.\(layer.plan.layer).ffn.gate.tid2eid",
                vocabularySize: plan.config.vocabularySize,
                expertsPerToken: expertsPerToken,
                expertCount: expertCount,
                cancellationCheck: cancellationCheck)
            else { continue }
            planned.append(
                DeepSeekV4RunExpertBackends.HashRoutedLayer(
                    layer: layer.plan.layer, map: map))
        }
        return planned
    }

    /// One ``DeepSeekV4RunExpertBackends/LayerUnit`` per admitted layer.
    public var routedExpertUnits: [DeepSeekV4RunExpertBackends.LayerUnit] {
        layers.map {
            DeepSeekV4RunExpertBackends.LayerUnit(
                layer: $0.plan.layer, containers: $0.expertUnit.containers)
        }
    }

    /// One expert residency for this artifact, wired to the same read and phase
    /// accounting the per-layer bracket reports through.
    ///
    /// Built here rather than at the call site so the two accounting objects
    /// stay module-internal: a run-scoped residency must charge its expert
    /// bytes to the same counter, and its backend construction to the same
    /// `expertBackendLifecycle` bracket, or the lifetime change would appear
    /// as a phase term vanishing rather than as a phase term moving.
    public func makeRunScopedExpertBackends(
        configuration: PagedRoutedExpertBackend.Configuration
    ) throws -> DeepSeekV4RunExpertBackends {
        try DeepSeekV4RunExpertBackends(
            units: routedExpertUnits,
            configuration: configuration,
            successfulReadObserver: { [readAccounting] bytes in
                readAccounting.recordExpert(bytes, didOverflow: false)
            },
            phaseObserver: phaseAccounting,
            buildObserver: { [phaseAccounting] nanoseconds in
                phaseAccounting.recordExpertBackendBuild(nanoseconds: nanoseconds)
            },
            retireObserver: { [phaseAccounting] nanoseconds in
                phaseAccounting.recordExpertBackendShutdown(nanoseconds: nanoseconds)
            })
    }
}

extension DeepSeekV4LayerPlan {
    /// The routing geometry a hash-routed layer's token table must match.
    ///
    /// Nil for a learned-routing layer, which has no table: its ids come from
    /// the layer's own hidden state and cannot be known before it runs. That is
    /// the whole asymmetry this file exists to exploit, so it is expressed as
    /// an absence rather than a default.
    var hashRoutedExpertsPerToken: Int? {
        switch self {
        case .hashWindow(let geometry): geometry.expertsPerToken
        case .compressedIndexed(let geometry):
            geometry.routingKind == .tokenHash ? geometry.expertsPerToken : nil
        case .compressedLearned: nil
        }
    }

    var hashRoutedExpertCount: Int? {
        switch self {
        case .hashWindow(let geometry): geometry.expertCount
        case .compressedIndexed(let geometry):
            geometry.routingKind == .tokenHash ? geometry.expertCount : nil
        case .compressedLearned: nil
        }
    }
}
