import Foundation
import StorageCore

/// The Mac's archived first-token logits, and the comparison M10 owes against
/// them.
///
/// ## Why this runs on the phone
///
/// The obvious arrangement is to write the device's logits to a file, copy it to
/// the Mac, and compare there. That arrangement failed on contact with reality:
/// the device's developer disk image wedged, and with it went every way of
/// moving a file off the phone. A comparison that cannot be performed is not
/// evidence.
///
/// Carrying the 655 KB reference *into* the app inverts the dependency. The run
/// reports the comparison as numbers in its own result, so the milestone's
/// criterion (b) survives having no file transport, a flat battery, or a
/// disconnected cable.
///
/// ## What is compared, and what is deliberately not
///
/// Spec §10.4 (v0.6.12) is explicit that cross-hardware logits are **not**
/// expected to be bit-identical, and that a flagship token cannot be bit-gated
/// across implementations at all: at 896-expert top-16 over 92 layers, ~19% of
/// routing decisions sit below the 1e-3 margin the tiny model was calibrated on,
/// so two correct implementations may route differently. A different GPU is such
/// an implementation.
///
/// So the metric is the **output-scale relative error**,
/// `max|ours − ref| / max|ref|`, never elementwise — the same quantity M9D used
/// against the server oracle, and the one §10.4 mandates at these reduction
/// lengths. The reference tolerance is `max(1e-5, 4·√K·eps_f32)` at `K = 7168`,
/// the hidden width the final projection reduces over.
///
/// Bit-identity is still *reported*, because "the two platforms agreed exactly"
/// would be a remarkable fact worth having. It is simply not the gate.
public enum MacReferenceLogits {

    /// SHA-256 of the archived vector, from
    /// `docs/experiments/data/2026-08-10-m10-k3-iphone/mac-reference-logits.sha256`.
    ///
    /// Checked every load. A reference that silently became a different run's
    /// logits would move criterion (b)'s answer with nothing to say why, and
    /// this file is copied by hand into the target's resources.
    public static let expectedSHA256 =
        "afb7ac5e2c63b5e17713636053e65aacde7e8bfa4377b9dd16507bc66b3c3a0d"

    /// The Mac run these came from (M9C, reproduced M10).
    public static let greedyTokenId = 3372
    public static let top1Top2Gap = 0.8024177551269531

    /// `tolerance(K) = max(1e-5, 4·√K·eps_f32)` — spec §10.4, reduction-length
    /// dependent. `K` is the hidden width, 7168.
    public static func tolerance(reductionLength K: Int) -> Double {
        max(1e-5, 4 * Double(K).squareRoot() * Double(Float.ulpOfOne))
    }

    public struct Comparison: Encodable, Sendable {
        /// `max|ours − ref| / max|ref|`. Never elementwise (§10.4).
        public let relative: Double
        public let tolerance: Double
        /// `tolerance / relative` — how much room the check had. M9D's tightest
        /// against the server oracle was 3.52×.
        public let margin: Double
        public let withinTolerance: Bool
        public let maxAbsoluteDifference: Double
        public let maxReferenceMagnitude: Double
        /// Reported, not gated: cross-GPU bit-identity is not expected.
        public let bitIdentical: Bool
        public let differingElements: Int
        public let elements: Int
        public let referenceSHA256: String
        public let referenceGreedyTokenId: Int
        public let deviceGreedyTokenId: Int
        public let greedyTokensAgree: Bool
        public let referenceTop1Top2Gap: Double
        public let deviceTop1Top2Gap: Double
    }

    /// Load the reference, verifying its digest.
    public static func load() throws -> [Float] {
        guard let url = Bundle.module.url(
            forResource: "mac-reference-logits", withExtension: "bin")
        else {
            throw ScenarioError.failed("mac-reference-logits.bin is missing from the app bundle")
        }
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data).map { String(format: "%02x", $0) }.joined()
        guard digest == expectedSHA256 else {
            throw ScenarioError.failed(
                "the bundled Mac reference hashes to \(digest), expected \(expectedSHA256)")
        }
        guard data.count % 4 == 0 else {
            throw ScenarioError.failed("the Mac reference is \(data.count) B, not a float32 vector")
        }
        var values = [Float](repeating: 0, count: data.count / 4)
        for index in values.indices {
            let start = index * 4
            let bits = UInt32(data[start]) | UInt32(data[start + 1]) << 8
                | UInt32(data[start + 2]) << 16 | UInt32(data[start + 3]) << 24
            values[index] = Float(bitPattern: bits)
        }
        return values
    }

    /// Compare a device run's logits against the Mac's.
    public static func compare(
        device: [Float], deviceGreedy: Int, deviceGap: Double, reductionLength: Int
    ) throws -> Comparison {
        let reference = try load()
        return try compareVectors(
            device: device,
            reference: reference,
            deviceGreedy: deviceGreedy,
            deviceGap: deviceGap,
            reductionLength: reductionLength)
    }

    /// The arithmetic behind ``compare(device:deviceGreedy:deviceGap:reductionLength:)``.
    ///
    /// Kept internal so the correctness suite can exercise malformed vectors
    /// without replacing the digest-pinned bundled reference. A non-finite
    /// logit is a failed comparison, not a value that IEEE comparison rules are
    /// allowed to make disappear from a maximum.
    static func compareVectors(
        device: [Float],
        reference: [Float],
        deviceGreedy: Int,
        deviceGap: Double,
        reductionLength: Int
    ) throws -> Comparison {
        guard reference.count == device.count else {
            throw ScenarioError.failed(
                "the Mac reference has \(reference.count) logits, this run produced "
                    + "\(device.count)")
        }
        guard reductionLength > 0 else {
            throw ScenarioError.failed(
                "the logits comparison reduction length must be positive, got "
                    + "\(reductionLength)")
        }
        guard deviceGap.isFinite else {
            throw ScenarioError.failed("the device top-1/top-2 gap is not finite")
        }
        if let index = reference.firstIndex(where: { !$0.isFinite }) {
            throw ScenarioError.failed(
                "the Mac reference contains a non-finite logit at index \(index)")
        }
        if let index = device.firstIndex(where: { !$0.isFinite }) {
            throw ScenarioError.failed(
                "the device output contains a non-finite logit at index \(index)")
        }

        var maxAbsoluteDifference = 0.0
        var maxReferenceMagnitude = 0.0
        var differing = 0
        for (ours, theirs) in zip(device, reference) {
            if ours.bitPattern != theirs.bitPattern { differing += 1 }
            let difference = abs(Double(ours) - Double(theirs))
            if difference > maxAbsoluteDifference { maxAbsoluteDifference = difference }
            let magnitude = abs(Double(theirs))
            if magnitude > maxReferenceMagnitude { maxReferenceMagnitude = magnitude }
        }
        // The output-scale denominator, not a per-element one: dividing by each
        // element would make a logit near zero dominate a comparison that is
        // about the vector's scale.
        guard maxReferenceMagnitude > 0 else {
            throw ScenarioError.failed(
                "the Mac reference has zero magnitude, so output-scale relative error "
                    + "is undefined")
        }
        let relative = maxAbsoluteDifference / maxReferenceMagnitude
        let allowed = tolerance(reductionLength: reductionLength)

        return Comparison(
            relative: relative,
            tolerance: allowed,
            margin: relative > 0 ? allowed / relative : .infinity,
            withinTolerance: relative <= allowed,
            maxAbsoluteDifference: maxAbsoluteDifference,
            maxReferenceMagnitude: maxReferenceMagnitude,
            bitIdentical: differing == 0,
            differingElements: differing,
            elements: device.count,
            referenceSHA256: expectedSHA256,
            referenceGreedyTokenId: greedyTokenId,
            deviceGreedyTokenId: deviceGreedy,
            greedyTokensAgree: deviceGreedy == greedyTokenId,
            referenceTop1Top2Gap: top1Top2Gap,
            deviceTop1Top2Gap: deviceGap)
    }
}
