import Foundation

/// The revision of DeepSeek's own inference code this runner implements.
///
/// Two revisions describe a V4.1 run and they are not the same thing. The
/// *weights* come from the source revision the container was converted from,
/// which the container's `index.json` carries and the runner checks. The
/// *arithmetic* comes from `inference/model.py` and `inference/kernel.py`, and
/// DeepSeek can change those without touching a single weight: on 2026-10-12
/// it did exactly that (commits `8b3cec70` and `4dd75a8b`), and the head that
/// carries both is the revision below.
///
/// This is the one place the runner names it. The run record carries it beside
/// the source revision (`inferenceRevision`), so a digest is always attributable
/// to the code it is meant to reproduce. ADR 0022 says what happens when
/// upstream moves it again.
public enum DeepSeekV41UpstreamInference {
    public static let repository = "deepseek-ai/DeepSeek-V4.1-Flash"

    /// `deepseek-ai/DeepSeek-V4.1-Flash` at 2026-10-16. Over the previous pin
    /// (`df42c109f1defefcbfcedbe7d905718a12266e40`) its `inference/model.py`
    /// differs in two behaviours, both implemented here:
    ///
    /// - an index-key owner publishes its own key cache on every step, not only
    ///   on a step that completes a compression group (`8b3cec70`, "Restore
    ///   each indexer's K cache on incomplete compression steps");
    /// - the attention output is quantized and dequantized through FP8
    ///   (`act_quant`, block 32, power-of-two scales) before the grouped `wo_a`
    ///   einsum (`4dd75a8b`).
    ///
    /// No safetensors file differs between the two revisions.
    public static let revision = "314113f8a7dadb388215396ba5fbee2c610a04d6"

    /// SHA-256 of `inference/model.py` at ``revision``.
    public static let modelSourceSHA256 =
        "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
}
