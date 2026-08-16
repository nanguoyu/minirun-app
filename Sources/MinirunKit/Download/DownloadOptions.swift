import Foundation

/// How a transfer behaves. Every knob is refused when it is out of range —
/// never clamped, never rounded to something reasonable.
///
/// Clamping is how a run ends up measured under settings nobody chose. A caller
/// that asks for a 3 MiB chunk and silently gets 4 MiB has a benchmark whose
/// stated configuration is a lie, so ``validated()`` throws and names the knob,
/// the value and the range instead.
public struct DownloadOptions: Sendable, Equatable, Codable {
    /// Bytes per ranged request. 1 MiB … 1 GiB, a whole number of MiB.
    public var chunkBytes: UInt64 = 64 << 20
    /// Files transferred at once. 1 … 16.
    public var maximumConcurrentFiles: Int = 4
    /// Chunks in flight within one file. See ``validated()`` — this build
    /// accepts only 1.
    public var maximumConcurrentChunksPerFile: Int = 1
    /// Digest each file as soon as it completes, rather than at the end. Costs
    /// a second read of the file; buys knowing which file went wrong while the
    /// job can still refetch it cheaply.
    public var verifyAfterEachFile: Bool = true
    public var allowMetered: Bool = false
    public var requireExternalPower: Bool = false
    /// Free space to leave behind after the transfer. Not a safety margin for
    /// the transfer — a margin for the machine, which needs room to keep
    /// working while a terabyte lands on it.
    public var freeSpaceHeadroomBytes: UInt64 = 8 << 30
    public var retry: RetryPolicy = RetryPolicy()

    public init() {}

    public static let chunkRange: ClosedRange<UInt64> = (1 << 20)...(1 << 30)
    public static let concurrentFileRange: ClosedRange<Int> = 1...16

    /// Throws ``DownloadError/invalidOption(name:value:allowed:)`` naming the
    /// knob, the value and the allowed range.
    public func validated() throws -> DownloadOptions {
        guard Self.chunkRange.contains(chunkBytes) else {
            throw DownloadError.invalidOption(
                name: "chunkBytes", value: "\(chunkBytes)",
                allowed: "\(Self.chunkRange.lowerBound) … \(Self.chunkRange.upperBound)")
        }
        guard chunkBytes % (1 << 20) == 0 else {
            throw DownloadError.invalidOption(
                name: "chunkBytes", value: "\(chunkBytes)", allowed: "a whole number of MiB")
        }
        guard Self.concurrentFileRange.contains(maximumConcurrentFiles) else {
            throw DownloadError.invalidOption(
                name: "maximumConcurrentFiles", value: "\(maximumConcurrentFiles)",
                allowed: "1 … 16")
        }
        // Refused rather than ignored. Parallel chunks within one file need
        // sparse writes into a preallocated part file, and that breaks the
        // resume contract this manager is built on: the part file's *length*
        // is what a resume trusts, over any recorded state (spec §5.10). A
        // sparse part file's length says nothing about which bytes are real.
        guard maximumConcurrentChunksPerFile == 1 else {
            throw DownloadError.invalidOption(
                name: "maximumConcurrentChunksPerFile", value: "\(maximumConcurrentChunksPerFile)",
                allowed:
                    "1 — parallel chunks within one file would make the part file's length stop meaning 'bytes safely on disk', which is what resume reads")
        }
        _ = try retry.validated()
        return self
    }
}

public struct RetryPolicy: Sendable, Equatable, Codable {
    public var maximumAttempts: Int = 5
    public var initialBackoff: TimeInterval = 1
    public var backoffMultiplier: Double = 2
    public var maximumBackoff: TimeInterval = 60

    public init() {}

    public func validated() throws -> RetryPolicy {
        guard (1...10).contains(maximumAttempts) else {
            throw DownloadError.invalidOption(
                name: "retry.maximumAttempts", value: "\(maximumAttempts)", allowed: "1 … 10")
        }
        guard initialBackoff.isFinite, initialBackoff >= 0 else {
            throw DownloadError.invalidOption(
                name: "retry.initialBackoff", value: "\(initialBackoff)",
                allowed: "a finite value >= 0")
        }
        guard backoffMultiplier.isFinite, backoffMultiplier >= 1 else {
            throw DownloadError.invalidOption(
                name: "retry.backoffMultiplier", value: "\(backoffMultiplier)",
                allowed: "a finite value >= 1")
        }
        guard maximumBackoff.isFinite, maximumBackoff >= initialBackoff else {
            throw DownloadError.invalidOption(
                name: "retry.maximumBackoff", value: "\(maximumBackoff)",
                allowed: "a finite value >= retry.initialBackoff (\(initialBackoff))")
        }
        return self
    }

    /// Seconds to wait before attempt `attempt` (1-based; attempt 1 waits none).
    public func backoff(beforeAttempt attempt: Int) -> TimeInterval {
        guard attempt > 1 else { return 0 }
        let raw = initialBackoff * pow(backoffMultiplier, Double(attempt - 2))
        guard raw.isFinite else { return maximumBackoff }
        return min(raw, maximumBackoff)
    }
}
