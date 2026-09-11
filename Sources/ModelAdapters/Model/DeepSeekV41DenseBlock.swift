import MLX

/// One V4.1 backbone block, minus the MoE and minus Engram.
///
/// `Block.forward` in the reference is eleven lines, and every one of them is
/// about *where* the hyper-connection coefficients come from:
///
/// ```python
/// residual = x
/// attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, ...)
/// x = self.hc_pre(x, pre_mix)          # the caller's
/// x = self.attn_norm(x)
/// x = self.attn(x, start_pos)
/// x = self.hc_post(x, residual, attn_post, attn_comb)
/// residual = x
/// ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, ...)
/// x = self.hc_pre(x, attn_pre)         # this block's attention's
/// x = self.ffn_norm(x)
/// x = self.ffn(x, image_mask)
/// x = self.hc_post(x, residual, ffn_post, ffn_comb)
/// return x, ffn_pre                    # the next block's
/// ```
///
/// So the block is a threading of one value — `pre_mix` — through the whole
/// stack, and this type is that threading with the two sublayers left as hooks.
///
/// ## Why the hooks
///
/// The MoE (384 routed experts, paired FP4 tiles) and the Engram module
/// (48 random 4 KiB page reads a token from two 100 GB tables) are being built
/// beside this, and both are storage-shaped in ways the dense path is not. A
/// block that owned them could not be tested without them. So:
///
/// - ``moe`` is `[tokens, hidden] -> [tokens, hidden]`, called where
///   `self.ffn(x, image_mask)` is;
/// - ``engram`` is `[tokens, multiplicity, hidden] -> [tokens, multiplicity, hidden]`
///   and optional, called where `Transformer.forward` calls
///   `layer.engram(h, ...)` — which is **before** the block, not inside it, and
///   on the expanded stream rather than on a contracted one. Blocks 1 and 14
///   have one; the other thirty-eight pass `nil`.
///
/// A test injects an identity or a small matmul for either. Phase 3 injects the
/// real ones. Nothing in this file knows which it got.
///
/// ## Phase accounting
///
/// ``DeepSeekV4PhaseAccounting`` is passed straight through and **no V4.1 term
/// was added**. The four operators here that book time — the FP8 activation
/// scale search, the finiteness sweeps, sparse attention and the indexer's score
/// pull — are the same four sites V4 already names, doing the same thing at a
/// different block size. A term that meant something different under the same
/// name would be worse than no term; when the V4.1 decode loop exists and
/// measures itself (ADR 0020 says there is no run yet to measure), whatever it
/// finds unattributed is what earns a new one.
public enum DeepSeekV41DenseBlock {
    /// The mHC tensors of one block. All six are float32 in the checkpoint —
    /// the reference builds them under `with set_dtype(torch.float32)` — and
    /// `hc_head_*` is not among them, because V4.1 publishes none.
    public struct HyperWeights {
        public let attentionFunction: MLXArray
        public let attentionBase: MLXArray
        public let attentionScale: MLXArray
        public let feedForwardFunction: MLXArray
        public let feedForwardBase: MLXArray
        public let feedForwardScale: MLXArray
        public let attentionNorm: MLXArray
        public let feedForwardNorm: MLXArray

        public init(
            attentionFunction: MLXArray, attentionBase: MLXArray, attentionScale: MLXArray,
            feedForwardFunction: MLXArray, feedForwardBase: MLXArray,
            feedForwardScale: MLXArray, attentionNorm: MLXArray, feedForwardNorm: MLXArray
        ) {
            self.attentionFunction = attentionFunction
            self.attentionBase = attentionBase
            self.attentionScale = attentionScale
            self.feedForwardFunction = feedForwardFunction
            self.feedForwardBase = feedForwardBase
            self.feedForwardScale = feedForwardScale
            self.attentionNorm = attentionNorm
            self.feedForwardNorm = feedForwardNorm
        }

        /// Read the six mHC tensors and the two norms from one unit.
        public init(
            artifact: DeepSeekV41BlockArtifact,
            cancellationCheck: () throws -> Void = {}
        ) throws {
            func vector(_ vector: DeepSeekV41BlockVector) throws -> MLXArray {
                try artifact.loadVector(vector, cancellationCheck: cancellationCheck)
            }
            self.init(
                attentionFunction: try vector(.hyperAttentionFunction),
                attentionBase: try vector(.hyperAttentionBase),
                attentionScale: try vector(.hyperAttentionScale),
                feedForwardFunction: try vector(.hyperFeedForwardFunction),
                feedForwardBase: try vector(.hyperFeedForwardBase),
                feedForwardScale: try vector(.hyperFeedForwardScale),
                attentionNorm: try vector(.attentionNorm),
                feedForwardNorm: try vector(.feedForwardNorm))
        }
    }

    /// The mHC geometry a block needs, all of it from the configuration.
    public struct HyperGeometry: Sendable, Equatable {
        public let multiplicity: Int
        public let sinkhornIterations: Int
        public let epsilon: Float
        public let normEpsilon: Float

        public init(config: DeepSeekV41Config) {
            self.multiplicity = config.hyperConnectionMultiplicity
            self.sinkhornIterations = config.hyperConnectionSinkhornIterations
            self.epsilon = config.hyperConnectionEpsilon
            self.normEpsilon = config.rmsNormEpsilon
        }

        public init(
            multiplicity: Int, sinkhornIterations: Int, epsilon: Float, normEpsilon: Float
        ) {
            self.multiplicity = multiplicity
            self.sinkhornIterations = sinkhornIterations
            self.epsilon = epsilon
            self.normEpsilon = normEpsilon
        }
    }

    public struct Result {
        /// `[tokens, multiplicity, hidden]`, the residual copies after the block.
        public let stream: MLXArray
        /// `ffn_pre`: the mix the **next** block's attention contracts with.
        public let preMix: MLXArray
        public let caches: DeepSeekV41AttentionCaches
    }

    /// The mHC threading with **all three** sublayers injected.
    ///
    /// ``forward(stream:preMix:hyper:hyperGeometry:attentionGeometry:attentionWeights:caches:shared:table:startPosition:moe:engram:phaseAccounting:diagnostics:)``
    /// is this with the real attention supplied, and is what a run calls. This
    /// form exists so the threading can be held to the reference on its own,
    /// with sublayers a fixture can state in one line each: the thing being
    /// checked is *which coefficients reach which sublayer*, and a stub makes
    /// that visible where a 512-dimensional attention would bury it.
    public static func threading(
        stream: MLXArray,
        preMix: MLXArray,
        hyper: HyperWeights,
        hyperGeometry: HyperGeometry,
        attention: (MLXArray) throws -> MLXArray,
        moe: (MLXArray) throws -> MLXArray,
        engram: ((MLXArray) throws -> MLXArray)? = nil,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> (stream: MLXArray, preMix: MLXArray) {
        guard stream.ndim == 3, stream.shape[1] == hyperGeometry.multiplicity else {
            throw DeepSeekV41Error.configuration(
                "the residual stream must be [tokens, \(hyperGeometry.multiplicity), hidden]; "
                    + "got \(stream.shape)")
        }
        // `Transformer.forward` runs Engram on the *expanded* stream, before the
        // block, and only on the two blocks that have one.
        var current = stream
        if let engram { current = try engram(stream) }

        var residual = current
        let attentionCoefficients = try DeepSeekV41HyperConnections.mixes(
            stream: current,
            function: hyper.attentionFunction,
            scale: hyper.attentionScale,
            base: hyper.attentionBase,
            multiplicity: hyperGeometry.multiplicity,
            iterations: hyperGeometry.sinkhornIterations,
            epsilon: hyperGeometry.epsilon,
            normEpsilon: hyperGeometry.normEpsilon,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let attentionInput = try DeepSeekV41RMSNorm.apply(
            try DeepSeekV41HyperConnections.contract(stream: current, preMix: preMix),
            weight: hyper.attentionNorm, epsilon: hyperGeometry.normEpsilon)
        current = try DeepSeekV41HyperConnections.expand(
            branch: try attention(attentionInput), residual: residual,
            coefficients: attentionCoefficients)

        residual = current
        let feedForwardCoefficients = try DeepSeekV41HyperConnections.mixes(
            stream: current,
            function: hyper.feedForwardFunction,
            scale: hyper.feedForwardScale,
            base: hyper.feedForwardBase,
            multiplicity: hyperGeometry.multiplicity,
            iterations: hyperGeometry.sinkhornIterations,
            epsilon: hyperGeometry.epsilon,
            normEpsilon: hyperGeometry.normEpsilon,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        // The one-block shift: the FFN contracts with what *this block's
        // attention* produced, not with what it produced itself.
        let feedForwardInput = try DeepSeekV41RMSNorm.apply(
            try DeepSeekV41HyperConnections.contract(
                stream: current, preMix: attentionCoefficients.pre),
            weight: hyper.feedForwardNorm, epsilon: hyperGeometry.normEpsilon)
        current = try DeepSeekV41HyperConnections.expand(
            branch: try moe(feedForwardInput), residual: residual,
            coefficients: feedForwardCoefficients)
        return (current, feedForwardCoefficients.pre)
    }

    /// The block, for prefill (`startPosition == 0`, many tokens) and for decode
    /// (`startPosition > 0`, exactly one token with the caches carried).
    ///
    /// The two are the same code. Nothing here branches on the phase: the phase
    /// reaches the window ring, the compressor's partial group and the reachable
    /// counts, and each of those three states what it does about it where it
    /// does it.
    public static func forward(
        stream: MLXArray,
        preMix: MLXArray,
        hyper: HyperWeights,
        hyperGeometry: HyperGeometry,
        attentionGeometry: DeepSeekV41AttentionGeometry,
        attentionWeights: DeepSeekV41AttentionWeights,
        caches: DeepSeekV41AttentionCaches,
        shared: DeepSeekV41SharedAttentionRuntime,
        table: DeepSeekV41RotaryTable,
        startPosition: Int,
        observeAttention: ((MLXArray) -> Void)? = nil,
        moe: (MLXArray) throws -> MLXArray,
        engram: ((MLXArray) throws -> MLXArray)? = nil,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> Result {
        var attentionCaches = caches
        let threaded = try threading(
            stream: stream,
            preMix: preMix,
            hyper: hyper,
            hyperGeometry: hyperGeometry,
            attention: { input in
                let attended = try DeepSeekV41AttentionForward.forward(
                    hidden: input,
                    geometry: attentionGeometry,
                    weights: attentionWeights,
                    caches: attentionCaches,
                    shared: shared,
                    table: table,
                    startPosition: startPosition,
                    phaseAccounting: phaseAccounting,
                    diagnostics: diagnostics)
                attentionCaches = attended.caches
                observeAttention?(attended.output[attended.output.shape[0] - 1])
                return attended.output
            },
            moe: moe,
            engram: engram,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return Result(
            stream: threaded.stream,
            preMix: threaded.preMix,
            caches: attentionCaches)
    }
}
