import MLX

/// `Transformer.forward`'s tail: collapse, final norm, logits.
///
/// ```python
/// h = layer.hc_pre(h, pre_mix)      # the LAST block's, under the mix it returned
/// logits = self.head(self.norm(h))
/// ```
///
/// ## The head is bfloat16, not FP8
///
/// `docs/experiments/2026-09-10-v41-flash-container.md` measures
/// `embed.weight` and `head.weight` together at 2,647,654,400 bytes, **BF16** —
/// two `[129280, 5120]` tables at two bytes an element. `ParallelHead` holds the
/// same weight as float32 and computes `F.linear(x.float(), weight)`, so the
/// logits come out in float32 with a single rounding: the bfloat16 checkpoint
/// value widened exactly, and one float32 reduction per vocabulary row.
///
/// So there is no activation quantization on this path and no
/// ``DeepSeekV41FP8Linear`` call. A runner that quantized here would be adding a
/// rounding the reference does not have.
///
/// ## Why the head is streamed in row windows
///
/// 1.32 GB is above what a Balanced-class budget will pin beside forty dense
/// blocks, and ``DeepSeekV41GlobalArtifact/loadRows(_:first:count:cancellationCheck:)``
/// refuses a read wider than its own window for exactly that reason. So the
/// logits are accumulated one window at a time, the way V4's output head is —
/// each window contributes a contiguous span of the 129,280 logits and nothing
/// larger than one window is ever resident.
public enum DeepSeekV41Head {
    /// Collapse the residual copies and apply the final norm.
    ///
    /// The collapse is the *last block's* `hc_pre` under the mix that block
    /// returned. V4.1 publishes no `hc_head_*`, so there is nothing else it
    /// could be — see ``DeepSeekV41HyperConnections/collapse(stream:preMix:)``.
    public static func normalized(
        stream: MLXArray,
        preMix: MLXArray,
        finalNormWeight: MLXArray,
        normEpsilon: Float
    ) throws -> MLXArray {
        let collapsed = try DeepSeekV41HyperConnections.collapse(
            stream: stream, preMix: preMix)
        return try DeepSeekV41RMSNorm.apply(
            collapsed, weight: finalNormWeight, epsilon: normEpsilon)
    }

    /// `F.linear(x.float(), head.float())` over one resident head matrix.
    ///
    /// The float32 cast on *both* operands is the reference's, and it is where
    /// the logits stop being bfloat16.
    public static func logits(
        normalized: MLXArray, headWeight: MLXArray
    ) throws -> MLXArray {
        guard normalized.ndim == 2, headWeight.ndim == 2,
            headWeight.shape[1] == normalized.shape[1]
        else {
            throw DeepSeekV41Error.configuration(
                "head weight \(headWeight.shape) does not match a [tokens, hidden] "
                    + "\(normalized.shape)")
        }
        return k3Linear(normalized.asType(.float32), headWeight.asType(.float32))
    }

    /// The same logits, accumulated one row window at a time.
    ///
    /// `windowRows` is the number of vocabulary rows held at once. The result is
    /// `[tokens, vocabulary]` float32; the windows are concatenated in row
    /// order, so this is the same tensor the resident form produces and the
    /// test beside it asserts that rather than assuming it.
    public static func streamedLogits(
        normalized: MLXArray,
        artifact: DeepSeekV41GlobalArtifact,
        vocabularySize: Int,
        windowRows: Int,
        boundsLiveWindows: Bool = false,
        cancellationCheck: () throws -> Void = {},
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> MLXArray {
        try streamedLogits(
            normalized: normalized,
            vocabularySize: vocabularySize,
            windowRows: windowRows,
            boundsLiveWindows: boundsLiveWindows,
            cancellationCheck: cancellationCheck,
            phaseAccounting: phaseAccounting
        ) { first, count in
            try artifact.loadRows(
                .head, first: first, count: count, cancellationCheck: cancellationCheck)
        }
    }

    /// The same accumulation over an arbitrary row source.
    ///
    /// `rows(first, count)` returns `[count, hidden]`. The artifact-backed form
    /// above is this with
    /// ``DeepSeekV41GlobalArtifact/loadRows(_:first:count:cancellationCheck:)``
    /// supplied; a test supplies a resident matrix, so the *accumulation* can be
    /// held to the resident form's bits on a machine with no 1.32 GB container
    /// on it.
    public static func streamedLogits(
        normalized: MLXArray,
        vocabularySize: Int,
        windowRows: Int,
        boundsLiveWindows: Bool = false,
        cancellationCheck: () throws -> Void = {},
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        rows: (Int, Int) throws -> MLXArray
    ) throws -> MLXArray {
        guard vocabularySize > 0, windowRows > 0 else {
            throw DeepSeekV41Error.configuration(
                "vocabulary size and head window must be positive; got "
                    + "\(vocabularySize) and \(windowRows)")
        }
        let widened = normalized.asType(.float32)
        var spans = [MLXArray]()
        var first = 0
        while first < vocabularySize {
            try cancellationCheck()
            let count = Swift.min(windowRows, vocabularySize - first)
            let window = try measuringPhase(
                phaseAccounting,
                excludingGPUBoundaryFrom: phaseAccounting?.recordOutputHeadRead(nanoseconds:)
            ) {
                try rows(first, count)
            }
            spans.append(
                measuringPhase(
                    phaseAccounting,
                    excludingGPUBoundaryFrom: phaseAccounting?
                        .recordOutputHeadCompute(nanoseconds:)
                ) {
                    k3Linear(widened, window.asType(.float32))
                })
            // Windowing the *read* is not windowing the *residency*. MLX is
            // lazy, so a span that is only appended keeps its own bfloat16
            // window alive until something evaluates it — and nothing does
            // until the caller asks for the logits, by which point all
            // `vocabularySize / windowRows` windows are resident and the walk
            // is holding the whole 1.32 GB table it was walked to avoid.
            //
            // Evaluating the span here is what makes the window a window: the
            // product is 4 KB, the window it consumed is freed, and the peak
            // is one window rather than the table. It forces exactly the
            // computation the caller's `eval` would force, in the same order,
            // so the logits are bit-identical — `DeepSeekV41HeadTests` holds
            // both modes to the same bits.
            if boundsLiveWindows, let span = spans.last { MLX.eval(span) }
            first += count
        }
        return concatenated(spans, axis: -1)
    }
}
