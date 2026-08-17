import Foundation
import MinirunKit
import StorageCore

/// Where the wall time of one DeepSeek V4 pass went.
///
/// ## Why this exists
///
/// K3 has had an expert-phase decomposition since v0.6.18
/// (``ExpertPhaseMetrics``). V4 had none at all: the audit of 2026-08-14 could
/// observe that a decode pass takes ~53 s and moves ~11.2 GB, and could list
/// the structural candidates that might own the difference, but it could not
/// say which of them does. Every per-item saving that audit names is therefore
/// an estimate. This type is the instrument that has to exist before any of
/// them becomes a measurement.
///
/// ## The accounting identity
///
/// One pass is bracketed by the engine, and the terms below are measured
/// inside it. They are disjoint by construction — no two of them can be on the
/// stack at the same time:
///
/// ```text
/// passSeconds == deterministicReadSeconds + tileDigestSeconds
///              + expertPhaseSeconds
///              + outputHeadReadSeconds + outputHeadComputeSeconds
///              + reclaimSeconds
///              + expertBackendLifecycleSeconds + layerArtifactSetupSeconds
///              + tileAdoptionSeconds
///              + activationScaleSyncSeconds + finitenessSweepSeconds
///              + routingSelectSeconds + sparseAttentionSeconds
///              + lightningIndexerSeconds
///              + unattributedSeconds
/// ```
///
/// - ``deterministicReadSeconds`` is the synchronous `pread` loop of
///   ``DeepSeekV4LayerArtifact`` — `loadBlockFP8`, `loadFloatingTensor`, and
///   `loadTokenExpertMap`. It is the one place the decode thread blocks on a
///   deterministic read, and V4 has no stager behind it.
/// - ``tileDigestSeconds`` is `QuantizedTileContainer.verifyTileDigest` inside
///   `loadBlockFP8`. It brackets digests that actually ran: under
///   ``TileDigestPolicy/trustHeldAuthority(_:)`` a load records nothing here
///   and increments ``tileDigestSkippedUnderAuthorityCount`` instead, so the
///   term states the work done rather than the work configured.
/// - ``expertPhaseSeconds`` is the routed-expert gather entry point, split
///   again by ``expertIOWaitSeconds`` and ``expertGatherComputeSeconds`` on the
///   K3 rule: `ioWait` brackets only `acquire`, `gatherCompute` brackets only
///   the `eval`s that force the gather graph.
/// - ``outputHeadReadSeconds`` and ``outputHeadComputeSeconds`` are the
///   per-token `lm_head` walk in `DeepSeekV4GlobalArtifact.logits`: the row
///   window's read-and-widen, and the matmul/`eval`/`asArray` that consumes it.
/// - ``reclaimSeconds`` is `DeepSeekV4TransientMemory.reclaim()` — the
///   `MLX.Memory.clearCache()` plus `malloc_zone_pressure_relief` pair that V4
///   runs at every evaluated layer boundary and after every logit window.
///
/// ## The residual's own brackets, added 2026-08-15
///
/// The first instrumented stated-scale gate left 4.4–4.6 s of a 6.4–7.1 s
/// decode pass — 66% — outside every term above
/// (`docs/experiments/2026-08-15-v4-stated-gate-8gb.md`). These eight terms
/// carve named pieces off that remainder. They are the audit's candidates
/// (`docs/2026-08-14-decode-performance-audit.md` §2) turned into
/// measurements, and like the six above no two can be on the stack together:
///
/// - ``expertBackendLifecycleSeconds`` is the `PagedRoutedExpertBackend`
///   construction and `shutdown()` that `withLayerBackend` performs once per
///   layer per pass — the `BufferPool` allocation and the per-container
///   `TileReader` thread spawn, and the teardown that joins them. It is
///   deliberately outside ``expertPhaseSeconds``, which brackets only the
///   gather, so the two remain disjoint (audit P0-3).
/// - ``layerArtifactSetupSeconds`` is one layer's artifact-execution setup —
///   the plain-tensor loads, `PlainWeights` assembly, token→expert table and
///   projection-closure construction — **less** the deterministic reads and
///   tile digests that happen inside it. Those two already have their own
///   terms, so this bracket subtracts them rather than counting them twice.
/// - ``tileAdoptionSeconds`` is `BlockFP8Weights.adopting` inside
///   `loadBlockFP8` — the byte scan of the whole packed matrix and the CPU
///   expansion of its scale grid that turn a read tile into an operand. It is
///   neither the read nor the digest, both of which are bracketed around it
///   (audit P2). The bracket is deliberately unchanged by the 2026-08-15
///   adoption fast path: it still measures whatever adoption still does, so a
///   later arm reads the saving off the same term rather than off a term that
///   moved. ``packedFinitenessMemoHitCount`` says how many of those adoptions
///   skipped the scan entirely.
/// - ``activationScaleSyncSeconds`` is `exactPowerOfTwoScales` in the FP8 and
///   FP4 activation references: `eval()` + `asArray` to the host, the scale
///   loop, and the re-upload. The audit estimated 400–500 of these per token
///   statically; ``activationScaleSyncCount`` states the measured number
///   (audit P1-7).
/// - ``finitenessSweepSeconds`` is the guard-only host syncs — the ones whose
///   sole purpose is validation and whose removal would change no computed
///   value: the `isFinite(...).all().item()` sweeps in compressed attention's
///   `validate`, in the MoE FFN's correction-bias guard, and at the entry of
///   both activation references, plus the hyper-connection head collapse's
///   host pull of its function/base tables. Load-bearing host pulls whose
///   values feed the arithmetic — `splitSinkhorn`'s three scale floats, the
///   routing scores — are *not* here; they belong to the term that consumes
///   them or to the residual.
/// - ``routingSelectSeconds`` is `DeepSeekV4Router.select`: the sqrt-softplus
///   score `eval` + `asArray`, the host top-k sort over every expert, and the
///   gathered-weight re-upload (audit P2).
/// - ``sparseAttentionSeconds`` is `DeepSeekV4SparseAttention.forward`, the
///   64-head loop that re-expresses the same KV gather per head (audit P1-6).
///   **It builds a lazy graph and forces nothing**, so this term states the
///   cost of expressing those ~2,752 micro-graphs per token, not of running
///   them; their arithmetic lands wherever the next sync forces it.
/// - ``lightningIndexerSeconds`` is `DeepSeekV4LightningIndexer.select` — the
///   score `eval`/`asArray` and the host top-k. Unlike the sparse-attention
///   bracket this one *does* contain a sync, so it also absorbs whatever
///   deferred graph that sync forces.
///
/// ``unattributedSeconds`` is the remainder, and it is *derived* rather than
/// stored so it can never disagree with the terms. It is not small and is not
/// meant to look small: attention and MLP arithmetic, the rotary and
/// hyper-connection expressions, `blockFP8MM` and its evals, per-layer cache
/// rebuilds, and the global artifact's own non-head reads
/// (`loadHeadParameters`, `embeddingRows`) all land there. Naming a bucket is
/// what makes it possible to move work out of the residual later; pretending
/// the sum is complete is not.
public struct DeepSeekV4PhaseMetrics: Codable, Sendable, Equatable {

    // MARK: Work

    /// `DeepSeekV4LayerArtifact` read calls, not `pread` calls. One matrix or
    /// one plain tensor is one count however many 4 MiB chunks it took.
    public var deterministicReadCount: Int = 0
    /// Digests actually computed. A load that skipped one under held
    /// verification authority is not counted here and contributes no time.
    public var tileDigestCount: Int = 0
    /// Tile loads served under ``TileDigestPolicy/trustHeldAuthority(_:)``,
    /// which recomputed no digest. Stated so a run's record says how much of
    /// its work ran under trust rather than leaving a fallen digest total to be
    /// read as a speedup.
    public var tileDigestSkippedUnderAuthorityCount: Int = 0
    /// Loads the memory dial's pinned tier served without reading anything.
    ///
    /// Counted for the same reason ``tileDigestSkippedUnderAuthorityCount`` is:
    /// a fallen `deterministicReadCount` on its own reads as work that stopped
    /// happening, and this says where it went instead. The bytes are in
    /// ``DeepSeekV4ReadAccountingSnapshot/pinnedServedBytes``; a served load
    /// contributes no time to any bracket, which is the whole point of it.
    public var pinnedServedCount: Int = 0
    /// `RoutedExpertBackend.gather` entries.
    public var expertGatherCallCount: Int = 0
    /// Tile leases taken. One per tile the routing decision actually selected.
    public var expertAcquireCount: Int = 0
    /// `logitChunkRows` windows walked over the head.
    public var outputHeadWindowCount: Int = 0
    public var reclaimCount: Int = 0
    /// Expert backends built. One per layer per pass under the current
    /// per-layer lifecycle; the matching `shutdown()` is charged to the same
    /// seconds term but is not counted again.
    public var expertBackendLifecycleCount: Int = 0
    /// Layer artifact-execution setups entered. One per layer per pass.
    public var layerArtifactSetupCount: Int = 0
    /// Block-FP8 tiles adopted. A pinned-tier hit adopts nothing and is
    /// counted in ``pinnedServedCount`` instead.
    public var tileAdoptionCount: Int = 0
    /// Adoptions that took the packed matrix's E4M3 finiteness as already
    /// established, rather than re-deriving it, because this run had scanned
    /// these exact bytes and holds the authority that binds the file to that
    /// scan. Stated for the reason ``tileDigestSkippedUnderAuthorityCount`` is:
    /// a fallen ``tileAdoptionSeconds`` should be readable as "the scan stopped
    /// repeating" rather than as an unexplained speedup. Under
    /// ``TileDigestPolicy/verifyEveryLoad`` this is always zero.
    public var packedFinitenessMemoHitCount: Int = 0
    /// `exactPowerOfTwoScales` calls. The audit's static estimate was 400–500
    /// per token; this is the number that actually ran.
    public var activationScaleSyncCount: Int = 0
    /// Guard-only host sweeps. One per bracketed guard, not per array inside
    /// a guard that sweeps several.
    public var finitenessSweepCount: Int = 0
    /// `DeepSeekV4Router.select` entries.
    public var routingSelectCount: Int = 0
    /// `DeepSeekV4SparseAttention.forward` entries.
    public var sparseAttentionCount: Int = 0
    /// `DeepSeekV4LightningIndexer.select` entries.
    public var lightningIndexerCount: Int = 0

    // MARK: The pass

    /// Wall time of the pass these terms were measured inside, stated by
    /// whoever brackets the pass. Nil on a cumulative run-scoped snapshot,
    /// which has no pass of its own — see ``withPassSeconds(_:)``.
    public var passSeconds: Double?

    // MARK: The split

    public var deterministicReadSeconds: Double = 0
    public var tileDigestSeconds: Double = 0
    /// Wall time the decode thread spent inside the routed-expert gather.
    /// Contains ``expertIOWaitSeconds`` and ``expertGatherComputeSeconds``.
    public var expertPhaseSeconds: Double = 0
    public var outputHeadReadSeconds: Double = 0
    public var outputHeadComputeSeconds: Double = 0
    public var reclaimSeconds: Double = 0

    // MARK: The residual's brackets

    /// Building and tearing down one layer's `PagedRoutedExpertBackend`.
    /// Outside ``expertPhaseSeconds``, which is the gather only.
    public var expertBackendLifecycleSeconds: Double = 0
    /// One layer's artifact-execution setup, less the deterministic reads and
    /// tile digests inside it — those are already named above.
    public var layerArtifactSetupSeconds: Double = 0
    /// `BlockFP8Weights.adopting`: the packed byte scan and scale expansion.
    public var tileAdoptionSeconds: Double = 0
    /// FP8/FP4 `exactPowerOfTwoScales` host round trips.
    public var activationScaleSyncSeconds: Double = 0
    /// Guard-only host syncs that validate and compute nothing.
    public var finitenessSweepSeconds: Double = 0
    /// `DeepSeekV4Router.select`: score sync, host sort, re-upload.
    public var routingSelectSeconds: Double = 0
    /// Expressing the 64-head sparse attention graph. Forces no arithmetic.
    public var sparseAttentionSeconds: Double = 0
    /// `DeepSeekV4LightningIndexer.select`: score sync and host top-k.
    public var lightningIndexerSeconds: Double = 0

    // MARK: Inside the expert phase

    /// Of ``expertPhaseSeconds``, blocked in `acquire` waiting for tile bytes.
    public var expertIOWaitSeconds: Double = 0
    /// Of ``expertPhaseSeconds``, blocked in `eval` forcing the gather graph.
    public var expertGatherComputeSeconds: Double = 0

    /// Sticky. Saturated nanosecond totals cannot prove the identity above.
    public var timingAccountingOverflowed: Bool = false
    /// Sticky. Saturated event counts cannot prove a per-call rate either.
    public var eventAccountingOverflowed: Bool = false

    // MARK: The census, added 2026-08-16

    /// How often the pass stopped, by named site — nil unless
    /// `MINIRUN_V4_EVAL_CENSUS` armed it.
    ///
    /// Strictly additive to everything above: it contributes to no seconds
    /// term, participates in no accounting identity, and its absence is what
    /// every record written before 2026-08-16 means. The terms above say where
    /// the seconds went; this says how many times the GPU pipeline was drained
    /// to produce them, which the seconds cannot show — a sync whose own
    /// bracket is 0.1 ms still pushes the work pending behind it into whatever
    /// bracket is open, usually the residual.
    public var evalCensus: DeepSeekV4EvalCensus?

    public init() {}

    // MARK: Derived

    /// The fourteen disjoint terms. Never stored, so it cannot drift.
    public var attributedSeconds: Double {
        deterministicReadSeconds + tileDigestSeconds + expertPhaseSeconds
            + outputHeadReadSeconds + outputHeadComputeSeconds + reclaimSeconds
            + expertBackendLifecycleSeconds + layerArtifactSetupSeconds
            + tileAdoptionSeconds
            + activationScaleSyncSeconds + finitenessSweepSeconds
            + routingSelectSeconds + sparseAttentionSeconds
            + lightningIndexerSeconds
    }

    /// Of the pass, the terms added when the residual got its own brackets.
    /// Stated separately so a record can be compared against one written
    /// before they existed, where all seven are zero by definition.
    public var residualBracketSeconds: Double {
        expertBackendLifecycleSeconds + layerArtifactSetupSeconds
            + tileAdoptionSeconds
            + activationScaleSyncSeconds + finitenessSweepSeconds
            + routingSelectSeconds + sparseAttentionSeconds
            + lightningIndexerSeconds
    }

    /// Everything the pass spent blocked on a host round trip that exists to
    /// validate or to decide, rather than to produce an activation.
    public var hostSyncSeconds: Double {
        activationScaleSyncSeconds + finitenessSweepSeconds
            + routingSelectSeconds + lightningIndexerSeconds
    }

    /// The pass minus everything named. Nil when no pass wall was stated,
    /// because a residual against an unknown total is not a number.
    public var unattributedSeconds: Double? {
        passSeconds.map { $0 - attributedSeconds }
    }

    /// The expert phase minus its two measured halves: expression building,
    /// tile adoption/copy, `TilePlan` construction, slot bookkeeping.
    public var expertUnattributedSeconds: Double {
        expertPhaseSeconds - expertIOWaitSeconds - expertGatherComputeSeconds
    }

    /// Everything the pass spent waiting on storage, deterministic and routed.
    public var storageWaitSeconds: Double {
        deterministicReadSeconds + expertIOWaitSeconds + outputHeadReadSeconds
    }

    public func fraction(of seconds: Double) -> Double? {
        guard let passSeconds, passSeconds > 0 else { return nil }
        return seconds / passSeconds
    }

    /// Every term is non-negative, the expert sub-split is inside its phase,
    /// and the named terms fit inside the pass. A microsecond of tolerance
    /// absorbs the clock reads themselves.
    public var isAccountingBalanced: Bool {
        guard !timingAccountingOverflowed else { return false }
        let terms = [
            deterministicReadSeconds, tileDigestSeconds, expertPhaseSeconds,
            outputHeadReadSeconds, outputHeadComputeSeconds, reclaimSeconds,
            expertBackendLifecycleSeconds, layerArtifactSetupSeconds,
            tileAdoptionSeconds,
            activationScaleSyncSeconds, finitenessSweepSeconds,
            routingSelectSeconds, sparseAttentionSeconds,
            lightningIndexerSeconds,
            expertIOWaitSeconds, expertGatherComputeSeconds,
        ]
        guard terms.allSatisfy({ $0 >= 0 }) else { return false }
        guard expertUnattributedSeconds >= -1e-6 else { return false }
        guard let unattributedSeconds else { return true }
        return unattributedSeconds >= -1e-6
    }

    /// Restate these terms against a pass wall the caller measured.
    ///
    /// The accounting object counts cumulatively over a run; a pass is a
    /// difference of two of its snapshots, and only the caller that bracketed
    /// the pass knows how long it was.
    public func withPassSeconds(_ seconds: Double) -> DeepSeekV4PhaseMetrics {
        var copy = self
        copy.passSeconds = seconds
        return copy
    }

    /// The terms completed after an earlier cumulative boundary.
    ///
    /// Every stored term is monotonic. A term moving backwards is not a
    /// negative amount of work; it is evidence that the two snapshots cannot be
    /// compared, so subtraction fails closed exactly as `ByteAccounting`'s
    /// does. A saturated total at either end also fails: a difference of two
    /// clamped numbers is not a duration. The result carries no
    /// ``passSeconds`` — the caller states that.
    public func subtracting(
        _ earlier: DeepSeekV4PhaseMetrics
    ) -> DeepSeekV4PhaseMetrics? {
        guard !timingAccountingOverflowed, !earlier.timingAccountingOverflowed,
            !eventAccountingOverflowed, !earlier.eventAccountingOverflowed
        else { return nil }

        var result = DeepSeekV4PhaseMetrics()
        let seconds: [WritableKeyPath<DeepSeekV4PhaseMetrics, Double>] = [
            \.deterministicReadSeconds, \.tileDigestSeconds, \.expertPhaseSeconds,
            \.outputHeadReadSeconds, \.outputHeadComputeSeconds, \.reclaimSeconds,
            \.expertBackendLifecycleSeconds, \.layerArtifactSetupSeconds,
            \.tileAdoptionSeconds,
            \.activationScaleSyncSeconds, \.finitenessSweepSeconds,
            \.routingSelectSeconds, \.sparseAttentionSeconds,
            \.lightningIndexerSeconds,
            \.expertIOWaitSeconds, \.expertGatherComputeSeconds,
        ]
        for path in seconds {
            let current = self[keyPath: path]
            let previous = earlier[keyPath: path]
            guard current >= previous else { return nil }
            result[keyPath: path] = current - previous
        }
        let counts: [WritableKeyPath<DeepSeekV4PhaseMetrics, Int>] = [
            \.deterministicReadCount, \.tileDigestCount,
            \.tileDigestSkippedUnderAuthorityCount, \.pinnedServedCount,
            \.expertGatherCallCount,
            \.expertAcquireCount, \.outputHeadWindowCount, \.reclaimCount,
            \.expertBackendLifecycleCount, \.layerArtifactSetupCount,
            \.tileAdoptionCount, \.packedFinitenessMemoHitCount,
            \.activationScaleSyncCount, \.finitenessSweepCount,
            \.routingSelectCount, \.sparseAttentionCount,
            \.lightningIndexerCount,
        ]
        for path in counts {
            let current = self[keyPath: path]
            let previous = earlier[keyPath: path]
            guard current >= previous else { return nil }
            result[keyPath: path] = current - previous
        }
        // The census differences the same way the counts do, and fails the
        // whole subtraction the same way: a pass whose census cannot be
        // differenced is a pass whose accounting cannot be trusted either, and
        // returning the seconds without the syncs would quietly hand back a
        // half-measured pass. A run with the census off has nil at both ends
        // and stays nil, which is not a failure.
        switch (evalCensus, earlier.evalCensus) {
        case (nil, nil):
            result.evalCensus = nil
        case (let current?, let previous?):
            guard let difference = current.subtracting(previous) else { return nil }
            result.evalCensus = difference
        default:
            return nil
        }
        return result.isAccountingBalanced ? result : nil
    }

    /// These terms as the model-agnostic ones a product can render.
    ///
    /// The projection is deliberately flat and lossy in one direction only: it
    /// keeps every second of the identity above, and drops the parameters and
    /// sticky overflow flags a screen has no use for. ``expertPhaseSeconds`` is
    /// published as its three disjoint pieces rather than as the container,
    /// because a bar that drew the phase *and* its halves would count the same
    /// seconds twice.
    public var runPhaseTerms: [RunPhaseTerm] {
        [
            RunPhaseTerm(
                name: RunPhaseTermName.deterministicRead,
                seconds: deterministicReadSeconds, count: deterministicReadCount),
            RunPhaseTerm(
                name: RunPhaseTermName.tileDigest,
                seconds: tileDigestSeconds, count: tileDigestCount),
            RunPhaseTerm(
                name: RunPhaseTermName.expertIOWait,
                seconds: expertIOWaitSeconds, count: expertAcquireCount),
            RunPhaseTerm(
                name: RunPhaseTermName.expertGatherCompute,
                seconds: expertGatherComputeSeconds, count: expertGatherCallCount),
            RunPhaseTerm(
                name: RunPhaseTermName.expertOther, seconds: max(0, expertUnattributedSeconds)),
            RunPhaseTerm(
                name: RunPhaseTermName.outputHeadRead,
                seconds: outputHeadReadSeconds, count: outputHeadWindowCount),
            RunPhaseTerm(
                name: RunPhaseTermName.outputHeadCompute, seconds: outputHeadComputeSeconds),
            RunPhaseTerm(
                name: RunPhaseTermName.reclaim, seconds: reclaimSeconds, count: reclaimCount),
            RunPhaseTerm(
                name: RunPhaseTermName.expertBackendLifecycle,
                seconds: expertBackendLifecycleSeconds, count: expertBackendLifecycleCount),
            RunPhaseTerm(
                name: RunPhaseTermName.layerArtifactSetup,
                seconds: layerArtifactSetupSeconds, count: layerArtifactSetupCount),
            RunPhaseTerm(
                name: RunPhaseTermName.tileAdoption,
                seconds: tileAdoptionSeconds, count: tileAdoptionCount),
            RunPhaseTerm(
                name: RunPhaseTermName.activationScaleSync,
                seconds: activationScaleSyncSeconds, count: activationScaleSyncCount),
            RunPhaseTerm(
                name: RunPhaseTermName.finitenessSweep,
                seconds: finitenessSweepSeconds, count: finitenessSweepCount),
            RunPhaseTerm(
                name: RunPhaseTermName.routingSelect,
                seconds: routingSelectSeconds, count: routingSelectCount),
            RunPhaseTerm(
                name: RunPhaseTermName.sparseAttention,
                seconds: sparseAttentionSeconds, count: sparseAttentionCount),
            RunPhaseTerm(
                name: RunPhaseTermName.lightningIndexer,
                seconds: lightningIndexerSeconds, count: lightningIndexerCount),
        ]
    }

    /// This pass, as the event the product stream carries. Nil when no pass
    /// wall was stated — a decomposition against an unknown total is not one.
    public func runPhaseSummary(passKind: RunPassKind, passIndex: Int) -> RunPhaseSummary? {
        guard let passSeconds else { return nil }
        return RunPhaseSummary(
            passKind: passKind, passIndex: passIndex, passSeconds: passSeconds,
            terms: runPhaseTerms)
    }

    /// One line for a run's log: the pass, its named terms, and the residual.
    public var summaryLine: String {
        String(
            format: "[v4 phase] %@ pass = %.1f s det read + %.1f s digest + %.1f s expert "
                + "(%.1f io / %.1f gather) + %.1f s head read + %.1f s head compute "
                + "+ %.1f s reclaim + %.1f s backend + %.1f s layer setup "
                + "+ %.1f s adoption + %.1f s scale sync + %.1f s finiteness + %.1f s routing "
                + "+ %.1f s sparse attn + %.1f s indexer + %@ other; "
                + "%d det reads, %d pinned, %d digests, "
                + "%d loads trusted, %d gathers, %d acquires, %d head windows, "
                + "%d reclaims, %d backends, %d layer setups, %d adoptions "
                + "(%d finiteness memo hits), "
                + "%d scale syncs, "
                + "%d sweeps, %d routes, %d sparse attns, %d indexer selects",
            passSeconds.map { String(format: "%.1f s", $0) } ?? "unstated",
            deterministicReadSeconds, tileDigestSeconds, expertPhaseSeconds,
            expertIOWaitSeconds, expertGatherComputeSeconds,
            outputHeadReadSeconds, outputHeadComputeSeconds, reclaimSeconds,
            expertBackendLifecycleSeconds, layerArtifactSetupSeconds,
            tileAdoptionSeconds,
            activationScaleSyncSeconds, finitenessSweepSeconds,
            routingSelectSeconds, sparseAttentionSeconds, lightningIndexerSeconds,
            unattributedSeconds.map { String(format: "%.1f s", $0) } ?? "unknown",
            deterministicReadCount, pinnedServedCount, tileDigestCount,
            tileDigestSkippedUnderAuthorityCount, expertGatherCallCount,
            expertAcquireCount, outputHeadWindowCount, reclaimCount,
            expertBackendLifecycleCount, layerArtifactSetupCount,
            tileAdoptionCount, packedFinitenessMemoHitCount,
            activationScaleSyncCount, finitenessSweepCount, routingSelectCount,
            sparseAttentionCount, lightningIndexerCount)
    }

    /// The census line, when the run armed one. Separate from
    /// ``summaryLine`` because it answers a different question and a record
    /// written without the census must read exactly as it did before.
    public var evalCensusLine: String? { evalCensus?.summaryLine }

    enum CodingKeys: String, CodingKey {
        case deterministicReadCount, tileDigestCount
        case tileDigestSkippedUnderAuthorityCount
        case pinnedServedCount
        case expertGatherCallCount
        case expertAcquireCount, outputHeadWindowCount, reclaimCount
        case expertBackendLifecycleCount, layerArtifactSetupCount, tileAdoptionCount
        case packedFinitenessMemoHitCount
        case activationScaleSyncCount, finitenessSweepCount
        case routingSelectCount, sparseAttentionCount, lightningIndexerCount
        case passSeconds
        case deterministicReadSeconds, tileDigestSeconds, expertPhaseSeconds
        case outputHeadReadSeconds, outputHeadComputeSeconds, reclaimSeconds
        case expertBackendLifecycleSeconds, layerArtifactSetupSeconds
        case tileAdoptionSeconds
        case activationScaleSyncSeconds, finitenessSweepSeconds
        case routingSelectSeconds, sparseAttentionSeconds, lightningIndexerSeconds
        case expertIOWaitSeconds, expertGatherComputeSeconds
        case timingAccountingOverflowed, eventAccountingOverflowed
        case evalCensus
        case attributedSeconds, unattributedSeconds, expertUnattributedSeconds
        case storageWaitSeconds, residualBracketSeconds, hostSyncSeconds
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        deterministicReadCount = try container.decode(Int.self, forKey: .deterministicReadCount)
        tileDigestCount = try container.decode(Int.self, forKey: .tileDigestCount)
        // Records written before the policy existed ran every digest, and a
        // missing key there means zero trusted loads rather than unknown.
        tileDigestSkippedUnderAuthorityCount = try container.decodeIfPresent(
            Int.self, forKey: .tileDigestSkippedUnderAuthorityCount) ?? 0
        // Absent from every record written before the memory dial existed, and
        // zero pins is exactly what those runs measured.
        pinnedServedCount = try container.decodeIfPresent(
            Int.self, forKey: .pinnedServedCount) ?? 0
        expertGatherCallCount = try container.decode(Int.self, forKey: .expertGatherCallCount)
        expertAcquireCount = try container.decode(Int.self, forKey: .expertAcquireCount)
        outputHeadWindowCount = try container.decode(Int.self, forKey: .outputHeadWindowCount)
        reclaimCount = try container.decode(Int.self, forKey: .reclaimCount)
        // Absent from every record written before the residual got its own
        // brackets. Zero is what those runs measured for these terms — they
        // are all inside the residual there — so a missing key is not unknown.
        expertBackendLifecycleCount = try container.decodeIfPresent(
            Int.self, forKey: .expertBackendLifecycleCount) ?? 0
        layerArtifactSetupCount = try container.decodeIfPresent(
            Int.self, forKey: .layerArtifactSetupCount) ?? 0
        tileAdoptionCount = try container.decodeIfPresent(
            Int.self, forKey: .tileAdoptionCount) ?? 0
        packedFinitenessMemoHitCount = try container.decodeIfPresent(
            Int.self, forKey: .packedFinitenessMemoHitCount) ?? 0
        activationScaleSyncCount = try container.decodeIfPresent(
            Int.self, forKey: .activationScaleSyncCount) ?? 0
        finitenessSweepCount = try container.decodeIfPresent(
            Int.self, forKey: .finitenessSweepCount) ?? 0
        routingSelectCount = try container.decodeIfPresent(
            Int.self, forKey: .routingSelectCount) ?? 0
        sparseAttentionCount = try container.decodeIfPresent(
            Int.self, forKey: .sparseAttentionCount) ?? 0
        lightningIndexerCount = try container.decodeIfPresent(
            Int.self, forKey: .lightningIndexerCount) ?? 0
        passSeconds = try container.decodeIfPresent(Double.self, forKey: .passSeconds)
        deterministicReadSeconds = try container.decode(
            Double.self, forKey: .deterministicReadSeconds)
        tileDigestSeconds = try container.decode(Double.self, forKey: .tileDigestSeconds)
        expertPhaseSeconds = try container.decode(Double.self, forKey: .expertPhaseSeconds)
        outputHeadReadSeconds = try container.decode(Double.self, forKey: .outputHeadReadSeconds)
        outputHeadComputeSeconds = try container.decode(
            Double.self, forKey: .outputHeadComputeSeconds)
        reclaimSeconds = try container.decode(Double.self, forKey: .reclaimSeconds)
        expertBackendLifecycleSeconds = try container.decodeIfPresent(
            Double.self, forKey: .expertBackendLifecycleSeconds) ?? 0
        layerArtifactSetupSeconds = try container.decodeIfPresent(
            Double.self, forKey: .layerArtifactSetupSeconds) ?? 0
        tileAdoptionSeconds = try container.decodeIfPresent(
            Double.self, forKey: .tileAdoptionSeconds) ?? 0
        activationScaleSyncSeconds = try container.decodeIfPresent(
            Double.self, forKey: .activationScaleSyncSeconds) ?? 0
        finitenessSweepSeconds = try container.decodeIfPresent(
            Double.self, forKey: .finitenessSweepSeconds) ?? 0
        routingSelectSeconds = try container.decodeIfPresent(
            Double.self, forKey: .routingSelectSeconds) ?? 0
        sparseAttentionSeconds = try container.decodeIfPresent(
            Double.self, forKey: .sparseAttentionSeconds) ?? 0
        lightningIndexerSeconds = try container.decodeIfPresent(
            Double.self, forKey: .lightningIndexerSeconds) ?? 0
        expertIOWaitSeconds = try container.decode(Double.self, forKey: .expertIOWaitSeconds)
        expertGatherComputeSeconds = try container.decode(
            Double.self, forKey: .expertGatherComputeSeconds)
        timingAccountingOverflowed = try container.decodeIfPresent(
            Bool.self, forKey: .timingAccountingOverflowed) ?? false
        eventAccountingOverflowed = try container.decodeIfPresent(
            Bool.self, forKey: .eventAccountingOverflowed) ?? false
        // Absent means the census was not armed for that run — which is not
        // "zero syncs", so it decodes to nil rather than to an empty table.
        evalCensus = try container.decodeIfPresent(
            DeepSeekV4EvalCensus.self, forKey: .evalCensus)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(deterministicReadCount, forKey: .deterministicReadCount)
        try container.encode(tileDigestCount, forKey: .tileDigestCount)
        try container.encode(
            tileDigestSkippedUnderAuthorityCount,
            forKey: .tileDigestSkippedUnderAuthorityCount)
        try container.encode(pinnedServedCount, forKey: .pinnedServedCount)
        try container.encode(expertGatherCallCount, forKey: .expertGatherCallCount)
        try container.encode(expertAcquireCount, forKey: .expertAcquireCount)
        try container.encode(outputHeadWindowCount, forKey: .outputHeadWindowCount)
        try container.encode(reclaimCount, forKey: .reclaimCount)
        try container.encode(
            expertBackendLifecycleCount, forKey: .expertBackendLifecycleCount)
        try container.encode(layerArtifactSetupCount, forKey: .layerArtifactSetupCount)
        try container.encode(tileAdoptionCount, forKey: .tileAdoptionCount)
        try container.encode(
            packedFinitenessMemoHitCount, forKey: .packedFinitenessMemoHitCount)
        try container.encode(activationScaleSyncCount, forKey: .activationScaleSyncCount)
        try container.encode(finitenessSweepCount, forKey: .finitenessSweepCount)
        try container.encode(routingSelectCount, forKey: .routingSelectCount)
        try container.encode(sparseAttentionCount, forKey: .sparseAttentionCount)
        try container.encode(lightningIndexerCount, forKey: .lightningIndexerCount)
        try container.encodeIfPresent(passSeconds, forKey: .passSeconds)
        try container.encode(deterministicReadSeconds, forKey: .deterministicReadSeconds)
        try container.encode(tileDigestSeconds, forKey: .tileDigestSeconds)
        try container.encode(expertPhaseSeconds, forKey: .expertPhaseSeconds)
        try container.encode(outputHeadReadSeconds, forKey: .outputHeadReadSeconds)
        try container.encode(outputHeadComputeSeconds, forKey: .outputHeadComputeSeconds)
        try container.encode(reclaimSeconds, forKey: .reclaimSeconds)
        try container.encode(
            expertBackendLifecycleSeconds, forKey: .expertBackendLifecycleSeconds)
        try container.encode(layerArtifactSetupSeconds, forKey: .layerArtifactSetupSeconds)
        try container.encode(tileAdoptionSeconds, forKey: .tileAdoptionSeconds)
        try container.encode(activationScaleSyncSeconds, forKey: .activationScaleSyncSeconds)
        try container.encode(finitenessSweepSeconds, forKey: .finitenessSweepSeconds)
        try container.encode(routingSelectSeconds, forKey: .routingSelectSeconds)
        try container.encode(sparseAttentionSeconds, forKey: .sparseAttentionSeconds)
        try container.encode(lightningIndexerSeconds, forKey: .lightningIndexerSeconds)
        try container.encode(expertIOWaitSeconds, forKey: .expertIOWaitSeconds)
        try container.encode(expertGatherComputeSeconds, forKey: .expertGatherComputeSeconds)
        try container.encode(timingAccountingOverflowed, forKey: .timingAccountingOverflowed)
        try container.encode(eventAccountingOverflowed, forKey: .eventAccountingOverflowed)
        // Omitted entirely when the census was not armed, which is what every
        // record written before it existed already says.
        try container.encodeIfPresent(evalCensus, forKey: .evalCensus)
        // Derived, written so a reader of the JSON does not have to redo the
        // arithmetic — and never read back, so the two cannot disagree.
        try container.encode(attributedSeconds, forKey: .attributedSeconds)
        try container.encodeIfPresent(unattributedSeconds, forKey: .unattributedSeconds)
        try container.encode(expertUnattributedSeconds, forKey: .expertUnattributedSeconds)
        try container.encode(storageWaitSeconds, forKey: .storageWaitSeconds)
        try container.encode(residualBracketSeconds, forKey: .residualBracketSeconds)
        try container.encode(hostSyncSeconds, forKey: .hostSyncSeconds)
    }
}

/// Bracket exactly one call and hand the elapsed nanoseconds to `record`.
///
/// The bracket is this function's own scope, so it cannot accidentally extend
/// over the statements that follow the measured call — which is the mistake a
/// bare `defer` at the call site makes. A call that throws still spent its
/// time and is still recorded: a total that silently drops the failing case is
/// the flattering kind of wrong. With no recorder, this is `body()`.
@inline(__always)
func measuringPhase<Result>(
    _ record: ((UInt64) -> Void)?,
    _ body: () throws -> Result
) rethrows -> Result {
    guard let record else { return try body() }
    let start = MonotonicClock.now()
    defer { record(MonotonicClock.now() &- start) }
    return try body()
}

/// Cumulative, run-scoped phase brackets for one V4 execution.
///
/// Every recording site is on the decode thread, but the object is locked all
/// the same: the engine samples it at pass boundaries, and a snapshot taken
/// while a `pread` loop is adding to it must not tear. A pass is the difference
/// of two snapshots (``DeepSeekV4PhaseMetrics/subtracting(_:)``), which is what
/// keeps prefill and each decode pass separate without a second instrument.
///
/// The brackets are one monotonic clock read each, taken per call and never per
/// 4 MiB chunk or per band. Instrumentation that changes the thing it measures
/// is not an instrument.
public final class DeepSeekV4PhaseAccounting: @unchecked Sendable {
    private struct Bracket {
        var nanoseconds = UInt64Accounting.SaturatingSum()
        var count = 0
        var countOverflowed = false

        mutating func add(_ increment: UInt64) {
            nanoseconds.add(increment)
            let next = count.addingReportingOverflow(1)
            count = next.overflow ? .max : next.partialValue
            countOverflowed = countOverflowed || next.overflow
        }

        var seconds: Double { Double(nanoseconds.value) / 1e9 }
    }

    private let lock = NSLock()
    private var deterministicRead = Bracket()
    private var tileDigest = Bracket()
    /// A count with no bracket, because a skipped digest has no duration to
    /// measure. Charging it a zero-length bracket would put it in a timing
    /// total where it does not belong.
    private var tileDigestSkippedUnderAuthority = 0
    private var tileDigestSkippedOverflowed = false
    /// Another count with no bracket, for the same reason: a load the pinned
    /// tier served spent no time reading, and a zero-length bracket would put
    /// it inside `deterministicReadSeconds` where it would look like a read
    /// that took no time rather than a read that did not happen.
    private var pinnedServed = 0
    private var pinnedServedOverflowed = false
    private var expertPhase = Bracket()
    private var expertIOWait = Bracket()
    private var expertGatherCompute = Bracket()
    private var outputHeadRead = Bracket()
    private var outputHeadCompute = Bracket()
    private var reclaim = Bracket()
    /// Construction and teardown are two brackets and one term: a backend that
    /// was built and then torn down is one lifecycle, so only the build is
    /// counted as an event while both contribute time.
    private var expertBackendBuild = Bracket()
    private var expertBackendShutdown = Bracket()
    private var layerArtifactSetup = Bracket()
    private var tileAdoption = Bracket()
    /// A count with no bracket of its own: the adoption it belongs to is
    /// already bracketed, and this says why that bracket came out short.
    private var packedFinitenessMemoHits = 0
    private var packedFinitenessMemoOverflowed = false
    private var activationScaleSync = Bracket()
    private var finitenessSweep = Bracket()
    private var routingSelect = Bracket()
    private var sparseAttention = Bracket()
    private var lightningIndexer = Bracket()

    /// How often the pass stopped, beside where its seconds went.
    ///
    /// It rides this object rather than being threaded separately because this
    /// object is already at every site that could force a sync — that is what
    /// made bracketing them possible in the first place. Off unless
    /// `MINIRUN_V4_EVAL_CENSUS` armed it, and then it adds one branch per
    /// recording site.
    public let evalCensus: DeepSeekV4EvalCensusCounter

    public convenience init() {
        self.init(evalCensus: DeepSeekV4EvalCensusCounter())
    }

    /// Stated explicitly so a test can arm the census without reaching into
    /// the process environment.
    public init(evalCensus: DeepSeekV4EvalCensusCounter) {
        self.evalCensus = evalCensus
    }

    /// One **blocking** host sync at `site`. Named `recordEval` to sit beside
    /// the other `record…` calls; it charges no seconds to anything.
    @inline(__always)
    public func recordEval(_ site: DeepSeekV4EvalSite) {
        evalCensus.record(site)
    }

    /// One **deferred** submission at `site` — an `MLX.asyncEval` that bounded
    /// the graph without waiting for the GPU.
    @inline(__always)
    public func recordAsyncEval(_ site: DeepSeekV4EvalSite) {
        evalCensus.recordAsync(site)
    }

    public var snapshot: DeepSeekV4PhaseMetrics {
        lock.lock()
        defer { lock.unlock() }
        var metrics = DeepSeekV4PhaseMetrics()
        metrics.deterministicReadCount = deterministicRead.count
        metrics.tileDigestCount = tileDigest.count
        metrics.tileDigestSkippedUnderAuthorityCount = tileDigestSkippedUnderAuthority
        metrics.pinnedServedCount = pinnedServed
        metrics.expertGatherCallCount = expertPhase.count
        metrics.expertAcquireCount = expertIOWait.count
        metrics.outputHeadWindowCount = outputHeadRead.count
        metrics.reclaimCount = reclaim.count
        metrics.expertBackendLifecycleCount = expertBackendBuild.count
        metrics.layerArtifactSetupCount = layerArtifactSetup.count
        metrics.tileAdoptionCount = tileAdoption.count
        metrics.packedFinitenessMemoHitCount = packedFinitenessMemoHits
        metrics.activationScaleSyncCount = activationScaleSync.count
        metrics.finitenessSweepCount = finitenessSweep.count
        metrics.routingSelectCount = routingSelect.count
        metrics.sparseAttentionCount = sparseAttention.count
        metrics.lightningIndexerCount = lightningIndexer.count
        metrics.expertBackendLifecycleSeconds =
            expertBackendBuild.seconds + expertBackendShutdown.seconds
        metrics.layerArtifactSetupSeconds = layerArtifactSetup.seconds
        metrics.tileAdoptionSeconds = tileAdoption.seconds
        metrics.activationScaleSyncSeconds = activationScaleSync.seconds
        metrics.finitenessSweepSeconds = finitenessSweep.seconds
        metrics.routingSelectSeconds = routingSelect.seconds
        metrics.sparseAttentionSeconds = sparseAttention.seconds
        metrics.lightningIndexerSeconds = lightningIndexer.seconds
        metrics.deterministicReadSeconds = deterministicRead.seconds
        metrics.tileDigestSeconds = tileDigest.seconds
        metrics.expertPhaseSeconds = expertPhase.seconds
        metrics.outputHeadReadSeconds = outputHeadRead.seconds
        metrics.outputHeadComputeSeconds = outputHeadCompute.seconds
        metrics.reclaimSeconds = reclaim.seconds
        metrics.expertIOWaitSeconds = expertIOWait.seconds
        metrics.expertGatherComputeSeconds = expertGatherCompute.seconds
        for bracket in [
            deterministicRead, tileDigest, expertPhase, expertIOWait,
            expertGatherCompute, outputHeadRead, outputHeadCompute, reclaim,
            expertBackendBuild, expertBackendShutdown, layerArtifactSetup,
            tileAdoption,
            activationScaleSync, finitenessSweep, routingSelect,
            sparseAttention, lightningIndexer,
        ] {
            metrics.timingAccountingOverflowed =
                metrics.timingAccountingOverflowed || bracket.nanoseconds.didOverflow
            metrics.eventAccountingOverflowed =
                metrics.eventAccountingOverflowed || bracket.countOverflowed
        }
        metrics.eventAccountingOverflowed =
            metrics.eventAccountingOverflowed || tileDigestSkippedOverflowed
                || pinnedServedOverflowed || packedFinitenessMemoOverflowed
        // Nil when the census is off. Deliberately not folded into
        // `eventAccountingOverflowed`: a saturated census says the sync counts
        // are unusable, not that the seconds are.
        metrics.evalCensus = evalCensus.snapshot
        return metrics
    }

    /// One load served from the memory dial's pinned tier.
    func recordPinnedServed() {
        lock.lock()
        let next = pinnedServed.addingReportingOverflow(1)
        pinnedServed = next.overflow ? .max : next.partialValue
        pinnedServedOverflowed = pinnedServedOverflowed || next.overflow
        lock.unlock()
    }

    func recordDeterministicRead(nanoseconds: UInt64) {
        lock.lock()
        deterministicRead.add(nanoseconds)
        lock.unlock()
    }

    func recordTileDigest(nanoseconds: UInt64) {
        lock.lock()
        tileDigest.add(nanoseconds)
        lock.unlock()
    }

    func recordTileDigestSkippedUnderAuthority() {
        lock.lock()
        let next = tileDigestSkippedUnderAuthority.addingReportingOverflow(1)
        tileDigestSkippedUnderAuthority = next.overflow ? .max : next.partialValue
        tileDigestSkippedOverflowed = tileDigestSkippedOverflowed || next.overflow
        lock.unlock()
    }

    func recordOutputHeadRead(nanoseconds: UInt64) {
        lock.lock()
        outputHeadRead.add(nanoseconds)
        lock.unlock()
    }

    func recordOutputHeadCompute(nanoseconds: UInt64) {
        lock.lock()
        outputHeadCompute.add(nanoseconds)
        lock.unlock()
    }

    func recordReclaim(nanoseconds: UInt64) {
        lock.lock()
        reclaim.add(nanoseconds)
        lock.unlock()
    }

    func recordExpertBackendBuild(nanoseconds: UInt64) {
        lock.lock()
        expertBackendBuild.add(nanoseconds)
        lock.unlock()
    }

    /// Charged to the same term as the build and counted as no new lifecycle:
    /// one backend built and torn down is one event, not two.
    func recordExpertBackendShutdown(nanoseconds: UInt64) {
        lock.lock()
        expertBackendShutdown.add(nanoseconds)
        lock.unlock()
    }

    /// One adoption that did not re-derive the packed matrix's finiteness.
    func recordPackedFinitenessMemoHit() {
        lock.lock()
        let next = packedFinitenessMemoHits.addingReportingOverflow(1)
        packedFinitenessMemoHits = next.overflow ? .max : next.partialValue
        packedFinitenessMemoOverflowed = packedFinitenessMemoOverflowed || next.overflow
        lock.unlock()
    }

    func recordTileAdoption(nanoseconds: UInt64) {
        lock.lock()
        tileAdoption.add(nanoseconds)
        lock.unlock()
    }

    func recordActivationScaleSync(nanoseconds: UInt64) {
        lock.lock()
        activationScaleSync.add(nanoseconds)
        lock.unlock()
    }

    func recordFinitenessSweep(nanoseconds: UInt64) {
        lock.lock()
        finitenessSweep.add(nanoseconds)
        lock.unlock()
    }

    func recordRoutingSelect(nanoseconds: UInt64) {
        lock.lock()
        routingSelect.add(nanoseconds)
        lock.unlock()
    }

    func recordSparseAttention(nanoseconds: UInt64) {
        lock.lock()
        sparseAttention.add(nanoseconds)
        lock.unlock()
    }

    func recordLightningIndexer(nanoseconds: UInt64) {
        lock.lock()
        lightningIndexer.add(nanoseconds)
        lock.unlock()
    }

    /// The deterministic-read and tile-digest totals so far.
    ///
    /// ``measuringLayerArtifactSetup(_:_:)`` reads this at both ends of its
    /// bracket so the reads that happen *inside* the setup can be subtracted
    /// from it. Those reads already own two terms of their own; charging them
    /// again here would make the sum larger than the pass.
    var nestedReadNanoseconds: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return deterministicRead.nanoseconds.value &+ tileDigest.nanoseconds.value
    }

    /// One layer's artifact setup, less the reads and digests inside it.
    ///
    /// A saturating subtraction rather than a signed one: the excluded total
    /// cannot exceed the bracket's own wall time unless a counter saturated,
    /// and a saturated counter already fails ``DeepSeekV4PhaseMetrics/
    /// isAccountingBalanced``. Clamping keeps every stored term non-negative
    /// so the identity can still be asserted.
    func recordLayerArtifactSetup(nanoseconds: UInt64, excluding excluded: UInt64) {
        lock.lock()
        layerArtifactSetup.add(nanoseconds > excluded ? nanoseconds &- excluded : 0)
        lock.unlock()
    }
}

/// Bracket one layer's artifact-execution setup, excluding the deterministic
/// reads and tile digests it performs.
///
/// The nested terms are read from the accounting itself rather than passed in,
/// so the exclusion cannot drift from what those brackets actually recorded.
/// Every recording site inside the body is on this same thread; a background
/// expert read does not touch either of the two excluded brackets.
@inline(__always)
func measuringLayerArtifactSetup<Result>(
    _ accounting: DeepSeekV4PhaseAccounting?,
    _ body: () throws -> Result
) rethrows -> Result {
    guard let accounting else { return try body() }
    let nestedAtStart = accounting.nestedReadNanoseconds
    let start = MonotonicClock.now()
    defer {
        accounting.recordLayerArtifactSetup(
            nanoseconds: MonotonicClock.now() &- start,
            excluding: accounting.nestedReadNanoseconds &- nestedAtStart)
    }
    return try body()
}

extension DeepSeekV4PhaseAccounting: RoutedExpertPhaseObserver {
    public func recordExpertPhase(nanoseconds: UInt64) {
        lock.lock()
        expertPhase.add(nanoseconds)
        lock.unlock()
    }

    public func recordExpertIOWait(nanoseconds: UInt64) {
        lock.lock()
        expertIOWait.add(nanoseconds)
        lock.unlock()
    }

    public func recordExpertGatherCompute(nanoseconds: UInt64) {
        lock.lock()
        expertGatherCompute.add(nanoseconds)
        lock.unlock()
    }

    public func recordExpertGatherEval() {
        recordEval(.expertGather)
    }
}
