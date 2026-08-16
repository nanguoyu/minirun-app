import Foundation

/// How offsets are chosen across the file.
public enum AccessPattern: String, Codable, CaseIterable, Sendable {
    /// Uniformly distributed, read-size-aligned blocks.
    case random
    /// Each worker walks its own contiguous slice of the file in order.
    case sequential
}

/// Which I/O interface is under test.
///
/// Spec M1 lists "pread worker pool versus Dispatch I/O" as an experiment, so
/// these are two implementations of the same workload rather than two
/// benchmarks — the offset sequence and the accounting are identical, and only
/// the mechanism differs.
public enum IOMode: String, Codable, CaseIterable, Sendable {
    /// `queue_depth` threads, each issuing a blocking `pread`.
    case pread
    /// Random-access `DispatchIO` channels with `queue_depth` reads in flight.
    case dispatchio
}

/// How many `DispatchIO` channels the dispatchio engine opens.
///
/// Dispatch I/O reads far slower than a `pread` pool as concurrency rises, with
/// peak RSS an order of magnitude higher. Two explanations fit that — a single
/// channel serializing the reads, or per-request cost inside the Dispatch I/O
/// path — and one topology cannot separate them. This option separates them:
/// switching between `one` and `per-slot` changes the channel count and nothing
/// else.
///
/// It has already answered the question once, on an M1 Pro over a warm cache:
/// `per-slot` overlaps `one` at every queue depth, so channel serialization is
/// *not* the mechanism. Kept because that measurement wants repeating against
/// cold external storage before anything is concluded from it. See
/// `docs/experiments/2026-08-07-dispatchio-channel-topology.md`.
public enum DispatchIOChannelMode: String, Codable, CaseIterable, Sendable {
    /// One channel shared by every in-flight read.
    case one
    /// One channel per in-flight slot, each on its own duplicated descriptor.
    case perSlot = "per-slot"
}

/// Cache policy applied to the descriptor under test.
public enum CacheMode: String, Codable, Sendable {
    /// Ordinary buffered reads through the unified page cache.
    case pageCache = "page_cache"
    /// `F_NOCACHE`: reads bypass the page cache for this descriptor.
    case noCache = "f_nocache"

    /// Honest description of what the mode does and does not guarantee. It ends
    /// up in the JSON because spec §17.4 requires a result to state its cache
    /// state, and because "no cache" overpromises.
    public var note: String {
        switch self {
        case .pageCache:
            return
                "Reads go through the unified page cache. A file that fits in RAM will be "
                + "served from memory after the first pass, so repeated runs measure memory "
                + "bandwidth rather than storage."
        case .noCache:
            return
                "F_NOCACHE was set on the descriptor, so reads bypass the page cache. This is "
                + "not the same as a cold cache: a genuinely cold measurement additionally "
                + "requires a freshly written file or a remounted volume, and the drive's own "
                + "cache is still in play."
        }
    }
}

/// Everything that defines a benchmark run (WP-003).
///
/// Every field here is reproduced verbatim in the result's `workload` object;
/// spec §17.4 requires the configuration to travel with the measurement, and
/// the cheapest way to guarantee that is for the JSON to be generated from this
/// struct rather than re-listed by hand.
public struct BenchmarkConfiguration: Equatable {
    public var path: String
    public var readSizeBytes: Int
    public var queueDepth: Int
    public var durationSeconds: Double
    public var warmupSeconds: Double
    public var pattern: AccessPattern
    public var mode: IOMode
    /// Channel topology for `mode == .dispatchio`. Ignored by the pread engine,
    /// but still echoed into the result so a sweep's parameters read uniformly.
    public var dispatchIOChannels: DispatchIOChannelMode
    public var disableCache: Bool
    /// Must match the seed the file was generated with for `verify` to mean
    /// anything. Also seeds offset selection, via an independent derived stream.
    public var seed: UInt64
    public var verify: Bool
    /// Verify one read in every N. Full verification of every read would make
    /// the CPU, not the storage, the thing being measured.
    public var verifySampleInterval: Int

    public static let defaultVerifySampleInterval = 64
    public static let maximumReadSizeBytes = 1 << 30
    public static let maximumQueueDepth = 1024
    public static let maximumDurationSeconds = 24.0 * 60 * 60

    public init(
        path: String,
        readSizeBytes: Int,
        queueDepth: Int,
        durationSeconds: Double,
        warmupSeconds: Double = 0,
        pattern: AccessPattern = .random,
        mode: IOMode = .pread,
        dispatchIOChannels: DispatchIOChannelMode = .one,
        disableCache: Bool = false,
        seed: UInt64 = 42,
        verify: Bool = false,
        verifySampleInterval: Int = BenchmarkConfiguration.defaultVerifySampleInterval
    ) {
        self.path = path
        self.readSizeBytes = readSizeBytes
        self.queueDepth = queueDepth
        self.durationSeconds = durationSeconds
        self.warmupSeconds = warmupSeconds
        self.pattern = pattern
        self.mode = mode
        self.dispatchIOChannels = dispatchIOChannels
        self.disableCache = disableCache
        self.seed = seed
        self.verify = verify
        self.verifySampleInterval = verifySampleInterval
    }

    public var cacheMode: CacheMode { disableCache ? .noCache : .pageCache }

    /// Checks the arguments alone, without touching the filesystem.
    public func validate() throws {
        guard !path.isEmpty else {
            throw StorageCoreError.invalidArgument("--path is required")
        }
        guard readSizeBytes > 0 else {
            throw StorageCoreError.invalidArgument("--read-size must be greater than zero")
        }
        guard readSizeBytes <= Self.maximumReadSizeBytes else {
            throw StorageCoreError.invalidArgument(
                "--read-size of \(ByteSize.format(UInt64(readSizeBytes))) exceeds the "
                    + "\(ByteSize.format(UInt64(Self.maximumReadSizeBytes))) limit")
        }
        guard queueDepth >= 1 else {
            throw StorageCoreError.invalidArgument("--queue-depth must be at least 1")
        }
        guard queueDepth <= Self.maximumQueueDepth else {
            throw StorageCoreError.invalidArgument(
                "--queue-depth of \(queueDepth) exceeds the limit of \(Self.maximumQueueDepth)")
        }
        guard durationSeconds > 0, durationSeconds.isFinite else {
            throw StorageCoreError.invalidArgument("--duration must be greater than zero")
        }
        guard durationSeconds <= Self.maximumDurationSeconds else {
            throw StorageCoreError.invalidArgument(
                "--duration of \(durationSeconds) s exceeds the 24 hour limit")
        }
        guard warmupSeconds >= 0, warmupSeconds.isFinite else {
            throw StorageCoreError.invalidArgument("--warmup must not be negative")
        }
        guard verifySampleInterval >= 1 else {
            throw StorageCoreError.invalidArgument(
                "--verify-sample-interval must be at least 1")
        }
    }

    /// Checks the arguments against the file actually being read.
    public func validate(fileSizeBytes: UInt64) throws {
        try validate()
        guard fileSizeBytes >= UInt64(readSizeBytes) else {
            throw StorageCoreError.invalidArgument(
                "file is \(ByteSize.format(fileSizeBytes)) but --read-size is "
                    + "\(ByteSize.format(UInt64(readSizeBytes))): no complete block fits. "
                    + "Generate a larger file or reduce --read-size.")
        }
    }
}
