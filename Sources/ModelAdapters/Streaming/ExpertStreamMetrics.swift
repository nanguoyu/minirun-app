import Foundation
import StorageCore

/// One run of ``ExpertStreamEngine``, in the shape spec §18.3 asks for.
///
/// Everything a table in the experiment record needs, and nothing that has to
/// be recomputed by hand afterwards. In particular the byte counts are split by
/// **class** — deterministic versus expert — because the ratio between them is
/// the number M9's per-token budget is made of, and a single "bytes read"
/// figure would hide it.
public struct ExpertStreamMetrics: Codable {

    public struct Run: Codable {
        public var schemaVersion = 1
        public var timestamp: String
        public var gitRevision: String
        public var device: DeviceInfo
        public var os: OSInfo
        public var configuration: String
        public var thermalState: String
    }

    public struct Workload: Codable {
        public var tokens: Int
        public var layers: [Int]
        public var expertsPerToken: Int
        public var expertsPerLayer: Int
        public var trace: String
        public var traceDistribution: String
        public var traceSeed: UInt64
        public var routedByRealRouter: Bool
        public var granularity: String
        public var readsPerExpert: String
        public var computedDeterministicRoles: [String]
        public var streamedOnlyDeterministicRoles: [String]
        /// Which routed-expert program ran: `per-expert` (one dispatch per
        /// expert per row band, what M8 measured) or `batched`.
        public var gatherProgram: String
        /// Experts covered by one `gatherQuantizedMM` under `batched`, after
        /// clamping to the top-k.
        public var expertsPerDispatch: Int
        /// `gatherQuantizedMM` calls issued on the routed-expert path over the
        /// whole run. Counted rather than derived from the configuration: the
        /// point of batching is this number, so a derived one would only
        /// restate the intent back.
        public var expertGatherDispatches: Int
        /// The same per token, which is what compares across runs of different
        /// length.
        public var expertGatherDispatchesPerToken: Double
    }

    /// Byte and request accounting for one stream class.
    public struct StreamIO: Codable {
        public var bytesRequested: UInt64
        public var bytesRead: UInt64
        /// Sticky. Saturated totals cannot prove a request/read identity.
        public var byteAccountingOverflowed: Bool
        public var readCount: Int
        public var shortReadCount: Int
        public var errorCount: Int
        public var firstError: String?
        public var latency: LatencyStatistics?
        /// Bytes the cache saved: requested minus read. Zero by construction
        /// for a stream with no cache.
        public var bytesServedFromCache: UInt64 {
            bytesRequested > bytesRead ? bytesRequested - bytesRead : 0
        }
        public var isByteAccountingBalanced: Bool {
            !byteAccountingOverflowed && bytesRequested == bytesRead
        }
        public var bytesPerToken: Double
        public var throughputMBPerSecond: Double
    }

    public struct Cache: Codable {
        public var policy: String
        public var budgetBytes: Int
        public var budgetExpertsEquivalent: Double
        public var slotCount: Int
        public var slotBytes: Int
        public var pinnedRegionCount: Int
        public var pinnedExpertCount: Int
        public var hits: Int
        public var misses: Int
        public var hitRate: Double
        public var evictions: Int
        public var admissions: Int
        public var rejectedAdmissions: Int
    }

    public struct Memory: Codable {
        public var expertPoolBudgetBytes: Int
        public var expertPoolPeakBytes: Int
        public var expertPoolPeakSlots: Int
        public var deterministicPoolBudgetBytes: Int
        public var deterministicPoolPeakBytes: Int
        public var totalPagerBudgetBytes: Int
        /// Advisory, and NOT additive with the pool bytes: MLX counts adopted
        /// external buffers as its own (spec §18.1, measured in M5).
        public var mlxPeakBytes: Int
        public var mlxCacheLimitBytes: Int
        public var processPeakResidentBytes: UInt64?
        public var processResidentBytes: UInt64?
        public var poolFaults: [String]
        public var budgetRespected: Bool
        /// Slot census at the end of the run, all classes summed. Every field
        /// but `free` must be zero, or something leaked.
        public var finalCensus: SlotCensus
    }

    /// The pair spec §18.1 insists on: only both together attribute a stall.
    public struct Stalls: Codable {
        /// Wall time the decode loop sat inside `acquire` waiting for bytes.
        public var computeWaitedOnIOSeconds: Double
        /// Wall time a reader's dispatcher sat waiting for a free slot.
        public var readerWaitedOnSlotSeconds: Double
        public var computeSeconds: Double
        public var wallSeconds: Double
        /// Share of the run the loop was blocked on storage.
        public var ioWaitFraction: Double
        /// `1 - ioWaitFraction`, i.e. how much of the read time the loop
        /// managed to hide behind compute and behind other reads.
        public var overlapFraction: Double
    }

    /// One slice of the run, for showing throughput did not drift.
    public struct Window: Codable {
        public var label: String
        public var firstToken: Int
        public var tokenCount: Int
        public var seconds: Double
        public var tokensPerSecond: Double
        public var bytesRead: UInt64
        public var byteAccountingOverflowed: Bool
        public var throughputMBPerSecond: Double
        public var thermalState: String
    }

    public var run: Run
    public var workload: Workload
    public var deterministic: StreamIO
    public var expert: StreamIO
    public var combinedThroughputMBPerSecond: Double
    /// Deterministic bytes divided by expert bytes, per token. The measured
    /// version of the ratio spec §15.5 projects from the GGUF checkpoint.
    public var deterministicToExpertByteRatio: Double
    /// Sticky. Saturated nanosecond totals are reported but not interpreted.
    public var timingAccountingOverflowed: Bool
    public var cache: Cache
    public var memory: Memory
    public var stalls: Stalls
    public var timing: [Window]
    public var tokensPerSecond: Double
    public var secondsPerToken: Double
    public var wallSeconds: Double
    public var localityByLayer: [RoutingTrace.Locality]
    /// Locality of the experts the run really fetched. Equal to
    /// `localityByLayer` for a pre-selected trace; for a hidden-state trace it
    /// is the measured behaviour of the checkpoint's own router.
    public var realizedLocality: [RoutingTrace.Locality]
    public var notes: [String]

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    public func write(to path: String) throws {
        try encoded().write(to: URL(fileURLWithPath: path))
    }
}

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
