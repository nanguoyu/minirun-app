import Foundation
import MLX
import MLXBridge

/// What one V4.1 block needs to run, in the form the operators take it.
///
/// A block is assembled by a *source* — a fixture in a test, an opened unit in
/// the runner — and the model never learns which. That is the same seam V4 has
/// between ``DeepSeekV4ModelArtifact`` and its layer readers, and it exists
/// here for the same reason: the arithmetic was established against fixtures in
/// phases 1 and 2, and phase 3 must be able to run exactly that arithmetic over
/// 517 GB of container without a second implementation of it.
public struct DeepSeekV41BlockWeights {
    /// The six mHC tensors and the two norms.
    public let hyper: DeepSeekV41DenseBlock.HyperWeights
    /// Attention, including whichever of the compressor and indexer this
    /// block's mode publishes.
    public let attention: DeepSeekV41AttentionWeights
    /// `ffn.gate.weight`, `[experts, hidden]` — read as float32 by the router.
    public let gateWeight: MLXArray
    /// `ffn.gate.bias`, `[experts]`.
    public let gateBias: MLXArray
    public let sharedGate: BlockFP8Weights
    public let sharedDown: BlockFP8Weights
    public let sharedUp: BlockFP8Weights
    /// The routed experts, a half-tile at a time.
    public let experts: DeepSeekV41RoutedExpertSource
    /// Present on `engram_layer_ids` and nowhere else.
    public let engram: Engram?

    /// One Engram module's three tensors and its page reader.
    public struct Engram {
        public let keyValue: BlockFP8Weights
        public let queryWeight: MLXArray
        public let keyWeight: MLXArray
        public let rowProvider: DeepSeekV41EngramRowProvider

        public init(
            keyValue: BlockFP8Weights, queryWeight: MLXArray, keyWeight: MLXArray,
            rowProvider: DeepSeekV41EngramRowProvider
        ) {
            self.keyValue = keyValue
            self.queryWeight = queryWeight
            self.keyWeight = keyWeight
            self.rowProvider = rowProvider
        }
    }

    public init(
        hyper: DeepSeekV41DenseBlock.HyperWeights,
        attention: DeepSeekV41AttentionWeights,
        gateWeight: MLXArray,
        gateBias: MLXArray,
        sharedGate: BlockFP8Weights,
        sharedDown: BlockFP8Weights,
        sharedUp: BlockFP8Weights,
        experts: DeepSeekV41RoutedExpertSource,
        engram: Engram? = nil
    ) {
        self.hyper = hyper
        self.attention = attention
        self.gateWeight = gateWeight
        self.gateBias = gateBias
        self.sharedGate = sharedGate
        self.sharedDown = sharedDown
        self.sharedUp = sharedUp
        self.experts = experts
        self.engram = engram
    }
}

/// Where a V4.1 forward gets its weights.
///
/// ``withBlock(_:_:)`` is scoped rather than a getter because that is the whole
/// of the storage-native discipline: a source that pins a block returns the
/// same arrays every call, and a source that streams one reads it, hands it
/// over, and releases it when the body returns. The model expresses no opinion.
public protocol DeepSeekV41WeightSource: AnyObject {
    /// `[tokens, hidden]` bfloat16 embedding rows, in the order given.
    func embeddingRows(_ tokens: [Int]) throws -> MLXArray
    /// `norm.weight` from `global00`.
    func finalNormWeight() throws -> MLXArray
    /// `[tokens, vocabulary]` float32 logits from the normalized hidden state.
    /// The head is 1.32 GB, so how it is read — resident, or in row windows —
    /// is the source's decision and not the model's.
    func logits(normalized: MLXArray) throws -> MLXArray
    /// Run `body` with block `index` available. What happens to those weights
    /// afterwards is the source's business.
    func withBlock<R>(
        _ index: Int, _ body: (DeepSeekV41BlockWeights) throws -> R
    ) throws -> R
    /// Issue the Engram page reads for a token's rows before block 0 runs.
    ///
    /// `rows` is `[engramBlockOrdinal][token][column]` — the hasher's own
    /// output transposed to the order the tables are addressed in. The default
    /// does nothing, which is right for a source whose rows are already in
    /// memory.
    func prefetchEngramRows(_ rows: [[[Int]]]) throws
}

extension DeepSeekV41WeightSource {
    public func prefetchEngramRows(_ rows: [[[Int]]]) throws {}
}

/// Where a run's Engram row addresses come from.
///
/// The addresses are integers and carry n-gram state across the prefill/decode
/// split, so this is a stateful object rather than a function: one is built per
/// generation and advanced exactly once per token.
///
/// It is a seam rather than a direct use of ``DeepSeekV41EngramHasher`` because
/// that type is built on the **published** constants — the multipliers from
/// `np.random.default_rng(10007 * layer_id)` and the 24 primes per block drawn
/// against a 16,000,000-entry hash space — which are properties of the shipped
/// checkpoint and not of a configuration. A model built at another geometry
/// (the mini checkpoint the end-to-end fixture runs at) has its own primes, and
/// a model that reached for the shipped ones there would be hashing with a
/// different table than the reference it is being compared to.
public protocol DeepSeekV41EngramIndexSource: AnyObject {
    /// `[token][engramBlock][column]` for the next tokens, advancing the
    /// n-gram state by exactly `tokenIDs.count` positions.
    func rows(forTokens tokenIDs: [Int]) throws -> [[[Int]]]
}

/// The published hasher as an index source.
///
/// Construction is where the configuration and the shipped constants are held
/// against each other: a checkpoint whose compressed vocabulary is not the one
/// the multipliers were drawn from would rehash the whole table, and that is a
/// refusal here rather than 384 million rows of plausible noise.
public final class DeepSeekV41PublishedEngramIndexSource: DeepSeekV41EngramIndexSource {
    var hasher: DeepSeekV41EngramHasher

    public init(map: DeepSeekV41EngramTokenMap.Map, config: DeepSeekV41Config) throws {
        guard config.engramCompressedVocabularySize
            == DeepSeekV41EngramConstants.compressedVocabularySize
        else {
            throw DeepSeekV41Error.engram(
                "this checkpoint declares \(config.engramCompressedVocabularySize) compressed "
                    + "Engram classes; the shipped multipliers are drawn from "
                    + "\(DeepSeekV41EngramConstants.compressedVocabularySize)")
        }
        guard config.engramLayerIDs == DeepSeekV41EngramConstants.layerIDs else {
            throw DeepSeekV41Error.engram(
                "this checkpoint has Engram at \(config.engramLayerIDs); the shipped "
                    + "constants are for \(DeepSeekV41EngramConstants.layerIDs)")
        }
        guard config.engramRowCounts.map({ Int($0) })
            == DeepSeekV41EngramConstants.rowCounts
        else {
            throw DeepSeekV41Error.engram(
                "this checkpoint's Engram tables have \(config.engramRowCounts) rows; the "
                    + "shipped primes sum to \(DeepSeekV41EngramConstants.rowCounts)")
        }
        self.hasher = try DeepSeekV41EngramHasher(map: map)
    }

    public func rows(forTokens tokenIDs: [Int]) throws -> [[[Int]]] {
        try hasher.rows(forTokens: tokenIDs)
    }
}

/// The per-block state a V4.1 run carries between passes.
///
/// Every entry is a cache the reference carries in a module attribute: the
/// sliding-window ring, the main-KV latents and index keys of the four source
/// blocks, the compressor's partial group, and the shared runtime.
///
/// ## The shared runtime is carried, not rebuilt
///
/// `SharedAttentionRuntime` is a **module-level global** in the reference —
/// `shared_attn = SharedAttentionRuntime()` beside the class, with the comment
/// "nothing needs resetting between forwards" — so what a block published in
/// one pass is what a block that publishes nothing reads in the next. At the
/// inference revision before ``DeepSeekV41UpstreamInference/revision`` that was
/// load-bearing: a ratio-2 encoder source published `index_k` only when its
/// group completed, so on every other decode step blocks 2 through 19 scored
/// against block 20's keys from the previous token. Upstream `8b3cec70` made
/// every owner publish its own keys on every step, so no published block reads
/// another pass's keys any more; the runtime is still carried, because that is
/// the reference's structure and a per-pass rebuild would be a second model of
/// it (`docs/experiments/2026-11-10-v41-upstream-october.md`).
public struct DeepSeekV41GenerationState {
    var caches: [DeepSeekV41AttentionCaches]
    let shared: DeepSeekV41SharedAttentionRuntime
    let engramIndices: (any DeepSeekV41EngramIndexSource)?
    var nextPosition: Int
    var positionLimit: Int
    /// Every token id a pass has consumed, in position order. A session
    /// snapshot is bound to exactly these ids (``DeepSeekV41SessionState``).
    var consumedTokenIDs: [Int]
    /// How many of those ids the position-0 prefill took as one block. Every
    /// later id went one pass each, so this number and the ids are the whole
    /// shape of the computation (ADR 0023), and a snapshot records it.
    var firstBlockTokenCount: Int

    init(
        caches: [DeepSeekV41AttentionCaches],
        shared: DeepSeekV41SharedAttentionRuntime,
        engramIndices: (any DeepSeekV41EngramIndexSource)?,
        nextPosition: Int,
        positionLimit: Int,
        consumedTokenIDs: [Int] = [],
        firstBlockTokenCount: Int = 0
    ) {
        self.caches = caches
        self.shared = shared
        self.engramIndices = engramIndices
        self.nextPosition = nextPosition
        self.positionLimit = positionLimit
        self.consumedTokenIDs = consumedTokenIDs
        self.firstBlockTokenCount = firstBlockTokenCount
    }

    /// Absolute position the next decode step consumes.
    public var position: Int { nextPosition }
    /// The run-scoped ceiling chosen before prefill.
    public var decodePositionLimit: Int { positionLimit }
    public var blockCount: Int { caches.count }
    /// The ids the passes so far consumed, position by position.
    public var tokenIDs: [Int] { consumedTokenIDs }
    /// The length of the block the position-0 prefill took; zero before it.
    public var firstBlockTokens: Int { firstBlockTokenCount }

    /// Drop every retained cache. A released state is terminal.
    public mutating func release() {
        caches.removeAll(keepingCapacity: false)
        consumedTokenIDs.removeAll(keepingCapacity: false)
        firstBlockTokenCount = 0
        nextPosition = 0
    }
}

/// Where a pass hands out the last token's residual stream, block by block.
///
/// Nothing in a run reads this and the product never installs one. It exists
/// because a whole-model disagreement reports one number — the logits differ —
/// and that number cannot say *which* of forty blocks started it. With this,
/// the same comparison a test makes against the reference's final logits can be
/// made against every block boundary, which turns a bug report into a
/// bisection. The reference has the same seam
/// (`v41_dense_reference.BLOCK_STREAM_TRACE`) and records the same tensor.
public protocol DeepSeekV41ForwardTrace: AnyObject {
    /// `stream` is `[multiplicity, hidden]` for the pass's **last** token,
    /// as it stands *before* `block` runs; `block == blockCount` is the state
    /// after the last block, before the final norm.
    func recordStream(block: Int, stream: MLXArray)
    /// The attention branch's own output for the pass's last token,
    /// `[hidden]` — before the mHC expansion folds it back into the stream.
    func recordAttention(block: Int, output: MLXArray)
    /// The MoE branch's own output for the pass's last token, `[hidden]`.
    func recordFeedForward(block: Int, output: MLXArray)

    // The whole-pass instruments. Every token of the pass rather than the last
    // one, because a prefill disagreement can start at any of its positions and
    // the reference's hooks see every one of them
    // (`Tools/v41_flash/streaming_reference.py --trace-dir`). The default for
    // each is to do nothing, so a trace that wants only the last token's
    // streams is unchanged.

    /// A pass begins: `tokenIDs` at absolute positions `startPosition...`.
    func beginPass(startPosition: Int, tokenIDs: [Int])
    /// `stream` is `[tokens, multiplicity, hidden]` and `preMix`
    /// `[tokens, multiplicity]`, as they stand *before* `block` runs and before
    /// any Engram on that block: what block `block - 1` returned. `block ==
    /// blockCount` is the state the final collapse reads.
    func recordStreamState(block: Int, stream: MLXArray, preMix: MLXArray)
    /// The attention branch's output, `[tokens, hidden]`, before `hc_post`.
    func recordAttentionBranch(block: Int, output: MLXArray)
    /// The MoE branch's output, `[tokens, hidden]`, before `hc_post`.
    func recordFeedForwardBranch(block: Int, output: MLXArray)
    /// Every token's routing decision at `block`, as the router made it.
    func recordRouting(block: Int, selection: DeepSeekV41Router.Selection)
    /// The compressed positions a block that *selects* chose, per token: the
    /// Top-K list exactly as handed to sparse attention (offset applied,
    /// unreachable entries `-1`), and the candidate mask where this block is
    /// the candidate source.
    func recordIndexSelection(
        block: Int, indices: [[Int]], offset: Int, candidateMask: [[Bool]]?)
    /// Engram at `block`: the table rows each token addressed, and the
    /// expanded stream it returned, `[tokens, multiplicity, hidden]`.
    func recordEngram(block: Int, rows: [[Int]], output: MLXArray)
    /// The pass has produced its logits.
    func endPass(greedyTokenID: Int)
}

extension DeepSeekV41ForwardTrace {
    public func recordStream(block: Int, stream: MLXArray) {}
    public func recordAttention(block: Int, output: MLXArray) {}
    public func recordFeedForward(block: Int, output: MLXArray) {}
    public func beginPass(startPosition: Int, tokenIDs: [Int]) {}
    public func recordStreamState(block: Int, stream: MLXArray, preMix: MLXArray) {}
    public func recordAttentionBranch(block: Int, output: MLXArray) {}
    public func recordFeedForwardBranch(block: Int, output: MLXArray) {}
    public func recordRouting(block: Int, selection: DeepSeekV41Router.Selection) {}
    public func recordIndexSelection(
        block: Int, indices: [[Int]], offset: Int, candidateMask: [[Bool]]?
    ) {}
    public func recordEngram(block: Int, rows: [[Int]], output: MLXArray) {}
    public func endPass(greedyTokenID: Int) {}
}

/// One pass's logits and the token greedy decoding picked.
public struct DeepSeekV41Step: Sendable, Equatable {
    public let logits: [Float]
    public let greedyTokenID: Int
    public let completedBlocks: Int

    public init(logits: [Float], greedyTokenID: Int, completedBlocks: Int) {
        self.logits = logits
        self.greedyTokenID = greedyTokenID
        self.completedBlocks = completedBlocks
    }
}

/// DeepSeek V4.1 Flash's text path: `Transformer.forward` over forty blocks.
///
/// ## What the reference does, and what this does
///
/// ```python
/// h = self.embed(tokens)                       # [b, s, dim]
/// h = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)  # the four residual streams
/// pre_mix = make_identity_pre_mix(h)           # block 0 reads copy 0
/// shared = SharedAttentionRuntime()
/// for layer in self.layers:
///     h, pre_mix = layer(h, start_pos, pre_mix, shared, ...)
/// h = self.norm(layer.hc_pre(h, pre_mix))
/// return self.head(h)
/// ```
///
/// with Engram called on the *expanded* stream before blocks 1 and 14, which is
/// where ``DeepSeekV41DenseBlock/threading(stream:preMix:hyper:hyperGeometry:attention:moe:engram:phaseAccounting:diagnostics:)``
/// puts it.
///
/// ### Prefill is exact
///
/// All forty blocks run over every prompt token. DeepSeek's serving path runs
/// only the encoder over the prompt and then replays the last `window_size`
/// tokens through the decoder, and says in its own report that this is not
/// mathematically equivalent. A gate against a reference run cannot be taken
/// against an approximation, so the exact path is what ships and bounded replay
/// is not implemented at all — an absent optimisation being better than a
/// silent one.
///
/// ### The Engram reads are issued before block 0
///
/// The 48 row addresses a token needs depend only on token ids, so they are
/// known before any compute starts. ``forward(tokens:state:cancellationCheck:onBlockCompleted:)``
/// hashes first, hands the addresses to the source's
/// ``DeepSeekV41WeightSource/prefetchEngramRows(_:)``, and only then runs
/// block 0. Engram sits at blocks 1 and 14, so block 1's rows have one block of
/// compute to hide behind and block 14's have thirteen.
public final class DeepSeekV41Model {
    public let config: DeepSeekV41Config
    /// Which of the four CSA2 jobs each block has, derived once.
    public let modes: [DeepSeekV41AttentionMode]
    private let source: any DeepSeekV41WeightSource
    let geometries: [DeepSeekV41AttentionGeometry]
    private let tables: [DeepSeekV41RotaryTable]
    private let hyperGeometry: DeepSeekV41DenseBlock.HyperGeometry
    private let expertActivation: DeepSeekV41ExpertActivation
    /// Whether the routed gather's operands are evaluated as they are built.
    /// See ``DeepSeekV41RoutedExperts/perExpertOutputs(_:expertIDs:routingWeights:swiGLULimit:source:activation:boundsLiveOperands:diagnostics:phaseAccounting:stream:)``.
    private let boundsLiveOperands: Bool
    private let diagnostics: DeepSeekV4Diagnostics
    private let phaseAccounting: DeepSeekV4PhaseAccounting?
    let makeEngramIndices: (() throws -> any DeepSeekV41EngramIndexSource)?
    /// Nil in every run. See ``DeepSeekV41ForwardTrace``.
    public var trace: (any DeepSeekV41ForwardTrace)?
    /// Backbone blocks only. The three DSpark blocks are published and named,
    /// and the text path does not run them (ADR 0017 found V4's drafter did
    /// not pay at this arm length; V4.1's is deferred for the same reason).
    public var blockCount: Int { config.numberOfLayers }

    /// The positions this model answers for: a prefill, a restored session
    /// and every decode step must stay inside `0..<positionCount`.
    ///
    /// It costs nothing to state the checkpoint's own ceiling here. The rotary
    /// rows are computed for the positions a pass touches
    /// (``DeepSeekV41RotaryTable/isOnDemand``), so no table is built for the
    /// positions a session has not reached; what remains refused is a count past
    /// `max_position_embeddings`.
    public let positionCount: Int

    /// What *held* per-block rotary tables would cost, at a geometry and a
    /// position count. The model no longer builds them; this is the arithmetic
    /// behind ``rotaryTableBudgetBytes`` and the record it cites.
    ///
    /// One table per backbone block holds `positionCount x ropeHeadDimension/2`
    /// float32 cosines and the same again in sines.
    public static func rotaryTableBytes(
        positionCount: Int, config: DeepSeekV41Config
    ) -> UInt64 {
        let perBlock = UInt64(max(0, positionCount))
            .multipliedReportingOverflow(by: UInt64(max(0, config.ropeHeadDimension) / 2))
            .partialValue
            .multipliedReportingOverflow(by: 8)  // float32 cosine + float32 sine
            .partialValue
        return perBlock
            .multipliedReportingOverflow(by: UInt64(max(0, config.numberOfLayers)))
            .partialValue
    }

    /// The envelope the eager rotary tables were allowed, kept as the record
    /// of why they went. Nothing is refused against it any more: the model
    /// computes rotary rows per pass (see ``positionCount``).
    ///
    /// ## Why there is a ceiling at all
    ///
    /// `max_position_embeddings` is a property of the *checkpoint*, not a
    /// working set: DeepSeek V4.1 Flash declares **1,048,576**. Forty blocks of
    /// float32 cosines and sines over a million positions at
    /// `rope_head_dim = 64` is 40 x 1,048,576 x 32 x 8 =
    /// **10,737,418,240 B — exactly 10.0 GiB**, and ``init`` builds them before
    /// a single weight byte is read.
    ///
    /// On a Mac that disappeared into the floor the budget is measured from,
    /// which is why the phase 3 arms never saw it and why the run record's
    /// "entry footprint" was 10.0 GiB. On the owner's iPhone 16 Pro it was the
    /// whole story: three chats on 2026-09-11 were killed by Jetsam at
    /// 5,234 / 5,084 / ~5,100 MB resident — about nineteen blocks into this
    /// loop — before the flight recorder could write one sample.
    ///
    /// 256 MiB is 26,214 positions at the published geometry: forty-five times
    /// the iPhone product's 512-token prompt plus 64-token reply and
    /// twenty-five times the Mac's 768 plus 256, well above any
    /// stated harness arm, and forty times below the allocation that killed the
    /// phone. A caller that needs more says so by needing more positions, and
    /// gets a refusal it can read instead of a `SIGKILL` it cannot.
    public static let rotaryTableBudgetBytes: UInt64 = 256 << 20

    /// - Parameters:
    ///   - engramIndices: builds the row-address source one generation uses.
    ///     Required when the configuration has Engram blocks — absent is a
    ///     refusal there rather than a model that quietly skips them.
    public init(
        config: DeepSeekV41Config,
        source: any DeepSeekV41WeightSource,
        engramIndices: (() throws -> any DeepSeekV41EngramIndexSource)? = nil,
        positionCount: Int,
        expertActivation: DeepSeekV41ExpertActivation = .referenceFP8,
        boundsLiveOperands: Bool = false,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws {
        guard positionCount > 0, positionCount <= config.maximumPositionCount else {
            throw DeepSeekV41Error.configuration(
                "position count \(positionCount) is outside 1...\(config.maximumPositionCount)")
        }
        if !config.engramLayerIDs.isEmpty, engramIndices == nil {
            throw DeepSeekV41Error.engram(
                "this configuration has Engram at blocks \(config.engramLayerIDs) and no "
                    + "row-address source was supplied")
        }
        self.config = config
        self.source = source
        self.positionCount = positionCount
        self.makeEngramIndices = engramIndices
        self.expertActivation = expertActivation
        self.boundsLiveOperands = boundsLiveOperands
        self.phaseAccounting = phaseAccounting
        self.diagnostics = diagnostics
        self.hyperGeometry = DeepSeekV41DenseBlock.HyperGeometry(config: config)
        var geometries = [DeepSeekV41AttentionGeometry]()
        var tables = [DeepSeekV41RotaryTable]()
        var modes = [DeepSeekV41AttentionMode]()
        geometries.reserveCapacity(config.numberOfLayers)
        for block in 0..<config.numberOfLayers {
            let geometry = try DeepSeekV41AttentionGeometry(block: block, config: config)
            geometries.append(geometry)
            modes.append(geometry.mode)
            tables.append(
                try DeepSeekV41RotaryTable.forBlock(
                    block, config: config, positionCount: positionCount, onDemand: true))
        }
        self.geometries = geometries
        self.tables = tables
        self.modes = modes
    }

    // MARK: - Generation

    /// Prefill the whole prompt and return the first token together with the
    /// state every later decode step advances.
    public func prefill(
        tokenIDs: [Int],
        decodePositionLimit: Int,
        cancellationCheck: () throws -> Void = {},
        onBlockCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> (step: DeepSeekV41Step, state: DeepSeekV41GenerationState) {
        guard !tokenIDs.isEmpty else {
            throw DeepSeekV41Error.configuration("prefill requires at least one token")
        }
        // Against `positionCount`, the positions this model answers for: a
        // limit past them is a run that would refuse at a rotation somewhere in
        // the middle of a pass rather than here.
        guard decodePositionLimit >= tokenIDs.count,
            decodePositionLimit <= positionCount
        else {
            throw DeepSeekV41Error.configuration(
                "decode position limit \(decodePositionLimit) cannot hold a "
                    + "\(tokenIDs.count)-token prompt inside the \(positionCount) "
                    + "positions this model was built for")
        }
        try validate(tokenIDs)
        var caches = [DeepSeekV41AttentionCaches]()
        caches.reserveCapacity(config.numberOfLayers)
        for block in 0..<config.numberOfLayers {
            caches.append(try DeepSeekV41AttentionCaches(geometry: geometries[block]))
        }
        var state = DeepSeekV41GenerationState(
            caches: caches,
            shared: DeepSeekV41SharedAttentionRuntime(),
            engramIndices: try makeEngramIndices?(),
            nextPosition: 0,
            positionLimit: decodePositionLimit)
        let step = try forward(
            tokens: tokenIDs, state: &state,
            cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
        state.firstBlockTokenCount = tokenIDs.count
        return (step, state)
    }

    /// Advance one token through every block cache.
    ///
    /// `state` is `inout` because a V4.1 pass replaces the four main-KV caches
    /// and forty window rings in place: holding a second complete generation so
    /// the step could be retried would double the largest state a run carries,
    /// which is the same reasoning behind V4's consuming session decode.
    public func decode(
        tokenID: Int,
        state: inout DeepSeekV41GenerationState,
        cancellationCheck: () throws -> Void = {},
        onBlockCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> DeepSeekV41Step {
        try validate([tokenID])
        guard state.nextPosition > 0, state.caches.count == config.numberOfLayers else {
            throw DeepSeekV41Error.configuration("decode requires a prefilled state")
        }
        guard state.nextPosition < state.positionLimit else {
            throw DeepSeekV41Error.configuration(
                "decode position \(state.nextPosition) reaches this run's limit "
                    + "\(state.positionLimit)")
        }
        return try forward(
            tokens: [tokenID], state: &state,
            cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
    }

    // MARK: - One pass

    private func forward(
        tokens: [Int],
        state: inout DeepSeekV41GenerationState,
        cancellationCheck: () throws -> Void,
        onBlockCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> DeepSeekV41Step {
        try cancellationCheck()
        let startPosition = state.nextPosition
        guard startPosition + tokens.count <= state.positionLimit else {
            throw DeepSeekV41Error.configuration(
                "a \(tokens.count)-token pass at position \(startPosition) exceeds this "
                    + "run's limit \(state.positionLimit)")
        }

        // The Engram addresses first, and the reads issued, before block 0.
        // `rows[engramOrdinal][token][column]`; the hasher advances its n-gram
        // state exactly once per token whether this is a 512-token prefill or a
        // one-token decode, which is what makes the two produce the same rows.
        var engramRows: [[[Int]]] = []
        if !config.engramLayerIDs.isEmpty {
            guard let indices = state.engramIndices else {
                throw DeepSeekV41Error.engram(
                    "this generation carries no Engram row-address source")
            }
            let hashed = try indices.rows(forTokens: tokens)
            guard hashed.count == tokens.count,
                hashed.allSatisfy({ $0.count == config.engramLayerIDs.count })
            else {
                throw DeepSeekV41Error.engram(
                    "the row-address source returned rows for \(hashed.count) tokens and "
                        + "\(hashed.first?.count ?? 0) blocks; this pass has \(tokens.count) "
                        + "tokens and \(config.engramLayerIDs.count) Engram blocks")
            }
            engramRows = (0..<config.engramLayerIDs.count).map { ordinal in
                hashed.map { $0[ordinal] }
            }
            try source.prefetchEngramRows(engramRows)
        }

        var stream = try DeepSeekV41Embedding.expandToResidualStreams(
            try source.embeddingRows(tokens),
            multiplicity: config.hyperConnectionMultiplicity)
        var preMix = try DeepSeekV41HyperConnections.identityPreMix(
            tokens: tokens.count, multiplicity: config.hyperConnectionMultiplicity)
        let shared = state.shared
        trace?.beginPass(startPosition: startPosition, tokenIDs: tokens)

        for block in 0..<config.numberOfLayers {
            try cancellationCheck()
            trace?.recordStream(block: block, stream: stream[tokens.count - 1])
            trace?.recordStreamState(block: block, stream: stream, preMix: preMix)
            let engramOrdinal = config.engramLayerIDs.firstIndex(of: block)
            let caches = state.caches[block]
            let result = try autoreleasepool {
                try source.withBlock(block) { weights in
                    try runBlock(
                        block, weights: weights, stream: stream, preMix: preMix,
                        caches: caches, shared: shared, startPosition: startPosition,
                        engramRows: engramOrdinal.map { engramRows[$0] },
                        cancellationCheck: cancellationCheck)
                }
            }
            stream = result.stream
            preMix = result.preMix
            state.caches[block] = result.caches
            onBlockCompleted(block + 1, config.numberOfLayers)
        }

        try cancellationCheck()
        trace?.recordStream(block: config.numberOfLayers, stream: stream[tokens.count - 1])
        trace?.recordStreamState(block: config.numberOfLayers, stream: stream, preMix: preMix)
        let normalized = try DeepSeekV41Head.normalized(
            stream: stream, preMix: preMix,
            finalNormWeight: try source.finalNormWeight(),
            normEpsilon: config.rmsNormEpsilon)
        guard normalized.ndim == 2, normalized.shape[0] == tokens.count,
            normalized.shape[1] == config.hiddenSize
        else {
            throw DeepSeekV41Error.configuration(
                "the final hidden state does not match the pass and model geometry")
        }
        let last = normalized[(tokens.count - 1)..., 0...]
        MLX.asyncEval([last])
        let logits = try source.logits(normalized: last)
        guard logits.ndim == 2, logits.shape[0] == 1,
            logits.shape[1] == config.vocabularySize
        else {
            throw DeepSeekV41Error.configuration(
                "the head returned \(logits.shape) for one token of a "
                    + "\(config.vocabularySize)-entry vocabulary")
        }
        let values = logits.asType(.float32).asArray(Float.self)
        state.nextPosition = startPosition + tokens.count
        state.consumedTokenIDs.append(contentsOf: tokens)
        let greedy = try Self.greedyToken(in: values)
        trace?.endPass(greedyTokenID: greedy)
        return DeepSeekV41Step(
            logits: values,
            greedyTokenID: greedy,
            completedBlocks: config.numberOfLayers)
    }

    private func runBlock(
        _ block: Int,
        weights: DeepSeekV41BlockWeights,
        stream: MLXArray,
        preMix: MLXArray,
        caches: DeepSeekV41AttentionCaches,
        shared: DeepSeekV41SharedAttentionRuntime,
        startPosition: Int,
        engramRows: [[Int]]?,
        cancellationCheck: () throws -> Void
    ) throws -> DeepSeekV41DenseBlock.Result {
        let engramWeights = weights.engram
        if (engramRows == nil) != (engramWeights == nil) {
            throw DeepSeekV41Error.engram(
                "block \(block) has Engram rows \(engramRows == nil ? "absent" : "present") "
                    + "and Engram weights \(engramWeights == nil ? "absent" : "present")")
        }
        let engram: ((MLXArray) throws -> MLXArray)?
        let trace = self.trace
        if let engramRows, let engramWeights {
            engram = { [config, diagnostics, phaseAccounting] hidden in
                let output = try DeepSeekV41Engram.apply(
                    hidden,
                    rowIndices: engramRows,
                    rowProvider: engramWeights.rowProvider,
                    wkv: engramWeights.keyValue,
                    queryWeight: engramWeights.queryWeight,
                    keyWeight: engramWeights.keyWeight,
                    dimension: config.hiddenSize,
                    multiplicity: config.hyperConnectionMultiplicity,
                    epsilon: config.rmsNormEpsilon,
                    // From the checkpoint, never the shipped constant: the row
                    // width is a property of the table being read.
                    headDimension: config.engramHeadDimension,
                    phaseAccounting: phaseAccounting,
                    diagnostics: diagnostics)
                trace?.recordEngram(block: block, rows: engramRows, output: output)
                return output
            }
        } else {
            engram = nil
        }

        let expertsPerToken = try config.expertsPerToken(block: block)
        let routedExpertCount = try config.routedExpertCount(block: block)
        // Only a block that runs its own indexer selects; a Reuse block reads
        // what its source published and is not a second observation of it.
        let selects = modes[block] == .full || modes[block] == .reindex
        let isCandidateSource = geometries[block].isCandidateSource
        return try DeepSeekV41DenseBlock.forward(
            stream: stream,
            preMix: preMix,
            hyper: weights.hyper,
            hyperGeometry: hyperGeometry,
            attentionGeometry: geometries[block],
            attentionWeights: weights.attention,
            caches: caches,
            shared: shared,
            table: tables[block],
            startPosition: startPosition,
            observeAttention: trace.map { [geometries] trace in
                { output in
                    let tokens = output.shape[0]
                    trace.recordAttention(block: block, output: output[tokens - 1])
                    trace.recordAttentionBranch(block: block, output: output)
                    if selects, let indices = shared.topKIndices {
                        // The window entries come first in the list sparse
                        // attention reads: the whole chunk at prefill, the ring
                        // at decode.
                        let offset =
                            startPosition == 0 ? tokens : geometries[block].windowSize
                        trace.recordIndexSelection(
                            block: block, indices: indices, offset: offset,
                            candidateMask: isCandidateSource ? shared.candidateMask : nil)
                    }
                }
            },
            moe: {
                [config, expertActivation, boundsLiveOperands, diagnostics, phaseAccounting]
                hidden in
                try cancellationCheck()
                // The whole routed sublayer is one phase bracket, and it has to
                // be: `expertIOWaitSeconds` is measured *inside* it, and the
                // phase identity states the wait as a share of the phase. A run
                // that recorded the wait and not the phase reports a negative
                // remainder, `isAccountingBalanced` refuses it, and every pass
                // comes back with no decomposition at all -- which is how this
                // bracket came to be added.
                return try measuringPhase(
                    phaseAccounting,
                    excludingGPUBoundaryFrom: phaseAccounting?
                        .recordExpertPhase(nanoseconds:)
                ) { () throws -> MLXArray in
                // Route first, then hand the *whole block's* routed set to the
                // source before evaluating any of it. `DeepSeekV41MoE.forward`
                // would do the same two steps; splitting them is what lets the
                // read-ahead see every token's experts at once rather than one
                // token's at a time, which is the seam
                // `docs/experiments/2026-09-11-v41-phase2-moe.md`'s second open
                // item asks for. A prefill of N tokens names up to 6N experts
                // here; a decode step names six.
                let selection = try measuringPhase(
                    phaseAccounting,
                    excludingGPUBoundaryFrom: phaseAccounting?
                        .recordExpertRoute(nanoseconds:)
                ) {
                    // Its own bracket inside the phase, not V4's pass-level
                    // `routingSelectSeconds`: the same seconds under two names
                    // is what `isAccountingBalanced` refuses.
                    try DeepSeekV41Router.route(
                        hidden: hidden,
                        gateWeight: weights.gateWeight,
                        bias: weights.gateBias,
                        expertCount: routedExpertCount,
                        expertsPerToken: expertsPerToken,
                        normalizeSelectedWeights: config.normalizedTopK,
                        routingScale: config.routingScale,
                        phaseAccounting: phaseAccounting)
                }
                trace?.recordRouting(block: block, selection: selection)
                try weights.experts.prefetch(
                    Array(Set(selection.ids.flatMap { $0 })).sorted())
                let combined = try DeepSeekV41MoE.evaluate(
                    hidden,
                    selection: selection,
                    swiGLULimit: config.swiGLULimit,
                    source: weights.experts,
                    sharedGate: weights.sharedGate,
                    sharedDown: weights.sharedDown,
                    sharedUp: weights.sharedUp,
                    activation: expertActivation,
                    boundsLiveOperands: boundsLiveOperands,
                    diagnostics: diagnostics,
                    phaseAccounting: phaseAccounting
                ).combined
                trace?.recordFeedForward(
                    block: block, output: combined[combined.shape[0] - 1])
                trace?.recordFeedForwardBranch(block: block, output: combined)
                return combined
                }
            },
            engram: engram,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
    }

    // MARK: - Helpers

    private func validate(_ tokens: [Int]) throws {
        for token in tokens where token < 0 || token >= config.vocabularySize {
            throw DeepSeekV41Error.configuration(
                "token id \(token) is outside the \(config.vocabularySize)-entry vocabulary")
        }
    }

    /// The greedy pick, refusing a non-finite vector rather than choosing from
    /// one — the same fail-closed rule V4's head applies.
    public static func greedyToken(in logits: [Float]) throws -> Int {
        guard !logits.isEmpty else {
            throw DeepSeekV41Error.configuration("cannot choose a token from no logits")
        }
        var best = 0
        var bestValue = -Float.infinity
        for (index, value) in logits.enumerated() {
            guard value.isFinite else {
                throw DeepSeekV41Error.configuration(
                    "logit \(index) is not finite; refusing to choose a token")
            }
            if value > bestValue {
                bestValue = value
                best = index
            }
        }
        return best
    }
}
