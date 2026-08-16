import Foundation
import StorageCore

/// The evaluated K3 state retained between token passes.
///
/// This is not a guess based on process memory. It is the exact float32 shape
/// of the state the operators return:
///
/// - every KDA layer retains three short-convolution windows plus one
///   `[heads, valueDim, keyDim]` recurrent matrix;
/// - every MLA layer retains one normalized low-rank latent and one shared
///   `k_rot` slice per cached position.
///
/// The distinction matters at the product boundary. The first pass creates
/// this state incrementally, while the second pass starts with all of it alive
/// and then opens the widest layer again. A one-token peak therefore cannot be
/// used as the lower bound for a multi-token chat.
public struct K3GenerationStateMemory: Sendable, Equatable {
    public let kdaBytes: UInt64
    public let mlaBytesPerCachedPosition: UInt64

    public init(config: TinyK3Config) throws {
        guard config.numHiddenLayers > 0 else {
            throw TinyK3Error.configuration("the K3 layer count is not positive")
        }
        let expected = Set(1...config.numHiddenLayers)
        guard config.kdaLayers.isDisjoint(with: config.fullAttentionLayers),
            config.kdaLayers.union(config.fullAttentionLayers) == expected
        else {
            throw TinyK3Error.configuration(
                "the KDA and full-attention layer lists do not partition the model")
        }

        let kdaLayers = try Self.nonnegative(config.kdaLayers.count, "KDA layer count")
        let mlaLayers = try Self.nonnegative(config.fullAttentionLayers.count, "MLA layer count")
        let kdaHeads = try Self.positive(config.kdaNumHeads, "KDA head count")
        let kdaHeadDim = try Self.positive(config.kdaHeadDim, "KDA head dimension")
        let kernel = try Self.positive(
            config.kdaShortConvKernelSize, "KDA convolution width")
        let latentWidth = try Self.positive(config.kvLoraRank, "MLA latent width")
        let sharedKeyWidth = try Self.positive(
            config.qkRopeHeadDim, "MLA shared k_rot width")

        let recurrent = try Self.product(
            [kdaLayers, kdaHeads, kdaHeadDim, kdaHeadDim, 4],
            "KDA recurrent state")
        let convolutions = try Self.product(
            [kdaLayers, 3, kdaHeads, kdaHeadDim, kernel, 4],
            "KDA convolution state")
        guard let kda = UInt64Accounting.checkedSum([recurrent, convolutions]) else {
            throw TinyK3Error.configuration("KDA rolling state exceeds UInt64.max")
        }
        kdaBytes = kda

        guard let retainedWidth = UInt64Accounting.checkedSum(
            [latentWidth, sharedKeyWidth])
        else {
            throw TinyK3Error.configuration("MLA retained width exceeds UInt64.max")
        }
        mlaBytesPerCachedPosition = try Self.product(
            [mlaLayers, retainedWidth, 4], "MLA latent cache per position")
    }

    public func bytes(cachedPositions: Int) throws -> UInt64 {
        let positions = try Self.nonnegative(cachedPositions, "cached position count")
        let mla = mlaBytesPerCachedPosition.multipliedReportingOverflow(by: positions)
        guard !mla.overflow,
            let total = UInt64Accounting.checkedSum([kdaBytes, mla.partialValue])
        else {
            throw TinyK3Error.configuration("K3 generation state exceeds UInt64.max")
        }
        return total
    }

    private static func nonnegative(_ value: Int, _ name: String) throws -> UInt64 {
        guard value >= 0 else {
            throw TinyK3Error.configuration("\(name) is negative")
        }
        return UInt64(value)
    }

    private static func positive(_ value: Int, _ name: String) throws -> UInt64 {
        guard value > 0 else {
            throw TinyK3Error.configuration("\(name) is not positive")
        }
        return UInt64(value)
    }

    private static func product(_ values: [UInt64], _ name: String) throws -> UInt64 {
        var product: UInt64 = 1
        for value in values {
            let result = product.multipliedReportingOverflow(by: value)
            guard !result.overflow else {
                throw TinyK3Error.configuration("\(name) exceeds UInt64.max")
            }
            product = result.partialValue
        }
        return product
    }
}
