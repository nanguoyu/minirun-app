import Foundation

/// Logical storage retained by DeepSeek V4's incremental attention state.
///
/// This is deliberately separate from process-footprint policy. The arrays
/// below are an architecture fact derived from the verified config and the
/// request's absolute position limit; allocator overhead and transient layer
/// execution are measured product terms owned by the runner. Keeping the two
/// terms separate prevents a measured process peak from being mislabeled as
/// exact cache geometry.
public enum DeepSeekV4GenerationMemoryBudget {
    public static let stateElementBytes: UInt64 = 4

    /// Exact logical bytes of every float32 array retained between generation
    /// passes for a batch-one request.
    ///
    /// V4 emits its first output from prefill, so `maximumNewTokens - 1`
    /// additional positions are reserved. Ratio-zero layers keep one circular
    /// window; ratio-four layers additionally keep main/index compressed
    /// caches and two overlapping compressor states; ratio-128 layers keep one
    /// compressed cache and one non-overlapping compressor state.
    public static func logicalStateBytes(
        config: DeepSeekV4Config,
        promptTokenCount: Int,
        maximumNewTokens: Int
    ) throws -> UInt64 {
        let positionLimit = try generationPositionLimit(
            config: config,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens)
        var total = CheckedByteTotal()
        for layer in 0..<config.numberOfLayers {
            try total.add(
                logicalLayerStateBytes(
                    config: config, positionLimit: positionLimit, layer: layer),
                name: "V4 layer states")
        }
        return total.value
    }

    /// Exact logical bytes retained by the largest single layer state.
    ///
    /// The product decoder replaces one layer state at a time. During that
    /// replacement both the old and new state for exactly one layer may be
    /// live, so this value is an explicit term in its admitted memory plan.
    public static func maximumLayerStateBytes(
        config: DeepSeekV4Config,
        promptTokenCount: Int,
        maximumNewTokens: Int
    ) throws -> UInt64 {
        let positionLimit = try generationPositionLimit(
            config: config,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens)
        guard config.numberOfLayers > 0 else {
            throw DeepSeekV4Error.configuration(
                "generation memory requires at least one model layer")
        }
        var largest: UInt64 = 0
        for layer in 0..<config.numberOfLayers {
            largest = max(
                largest,
                try logicalLayerStateBytes(
                    config: config, positionLimit: positionLimit, layer: layer))
        }
        return largest
    }

    static func logicalLayerStateBytes(
        config: DeepSeekV4Config,
        positionLimit: Int,
        layer: Int
    ) throws -> UInt64 {
        guard positionLimit > 0, positionLimit <= config.maximumPositionCount else {
            throw DeepSeekV4Error.configuration(
                "V4 layer state position limit is outside the verified config")
        }
        var total = CheckedByteTotal()
        try total.add(
            bytes(
                factors: [config.slidingWindow, config.attentionHeadDimension],
                name: "V4 circular attention window"),
            name: "V4 circular attention window")
        let ratio = try config.compressionRatio(layer: layer)
        switch ratio {
        case 0:
            break
        case 4:
            let compressedCapacity = positionLimit / ratio
            try total.add(
                bytes: [compressedCapacity, config.attentionHeadDimension],
                name: "ratio-four compressed KV cache")
            try total.add(
                bytes: [compressedCapacity, config.indexHeadDimension],
                name: "ratio-four index cache")

            // Two arrays (values and scores), two overlapping ratio-sized
            // windows, and a projected width of 2 * head dimension.
            try total.add(
                bytes: [2, ratio, 2, 2, config.attentionHeadDimension],
                name: "ratio-four main compressor state")
            try total.add(
                bytes: [2, ratio, 2, 2, config.indexHeadDimension],
                name: "ratio-four index compressor state")
        case 128:
            let compressedCapacity = positionLimit / ratio
            try total.add(
                bytes: [compressedCapacity, config.attentionHeadDimension],
                name: "ratio-128 compressed KV cache")
            // Two non-overlapping arrays: values and scores.
            try total.add(
                bytes: [2, ratio, config.attentionHeadDimension],
                name: "ratio-128 compressor state")
        default:
            throw DeepSeekV4Error.unsupportedArchitecture(
                "layer \(layer) uses unsupported compression ratio \(ratio)")
        }
        return total.value
    }

    private static func generationPositionLimit(
        config: DeepSeekV4Config,
        promptTokenCount: Int,
        maximumNewTokens: Int
    ) throws -> Int {
        guard promptTokenCount > 0 else {
            throw DeepSeekV4Error.configuration(
                "generation memory requires at least one prompt token")
        }
        guard maximumNewTokens > 0 else {
            throw DeepSeekV4Error.configuration(
                "generation memory requires at least one requested output token")
        }
        let extraPositions = maximumNewTokens.addingReportingOverflow(-1)
        let positionLimit = promptTokenCount.addingReportingOverflow(
            extraPositions.partialValue)
        guard !extraPositions.overflow, !positionLimit.overflow,
            positionLimit.partialValue <= config.maximumPositionCount
        else {
            throw DeepSeekV4Error.configuration(
                "prompt and response position limit is not representable by the verified config")
        }
        return positionLimit.partialValue
    }

    fileprivate static func bytes(factors: [Int], name: String) throws -> UInt64 {
        var product = stateElementBytes
        for factor in factors {
            guard let value = UInt64(exactly: factor) else {
                throw DeepSeekV4Error.configuration(
                    "\(name) has a negative or unrepresentable dimension")
            }
            let next = product.multipliedReportingOverflow(by: value)
            guard !next.overflow else {
                throw DeepSeekV4Error.configuration("\(name) exceeds UInt64.max")
            }
            product = next.partialValue
        }
        return product
    }
}

private struct CheckedByteTotal {
    private(set) var value: UInt64 = 0

    mutating func add(_ bytes: UInt64, name: String) throws {
        let next = value.addingReportingOverflow(bytes)
        guard !next.overflow else {
            throw DeepSeekV4Error.configuration("\(name) total exceeds UInt64.max")
        }
        value = next.partialValue
    }

    mutating func add(bytes factors: [Int], name: String) throws {
        try add(
            DeepSeekV4GenerationMemoryBudget.bytes(factors: factors, name: name),
            name: name)
    }
}
