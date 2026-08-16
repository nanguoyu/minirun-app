import Foundation

/// Detects the configuration in which a run reports memory bandwidth while
/// looking like a storage measurement.
///
/// This is the most likely way to get a wrong answer out of this tool, and it
/// is silent: nothing fails, the numbers are simply about the page cache
/// instead of the drive. On a 32 GiB machine a 256 MiB file read for five
/// seconds reports tens of gigabytes per second.
///
/// `F_NOCACHE` does not rescue it. The flag stops reads *populating* the cache;
/// it does not evict pages an earlier run already left resident, and a file
/// small enough to sit in memory generally is resident by the second run. So
/// the test is on size alone, and the cache mode only changes the advice.
public enum CacheConfounding {
    /// A file smaller than this multiple of physical memory is treated as at
    /// risk. Two rather than one, because "fits in memory" is not a clean
    /// threshold — the cache holds a large fraction of a file somewhat bigger
    /// than RAM, and the resulting mix is harder to interpret than either
    /// extreme.
    public static let memoryMultiplier: UInt64 = 2

    public static func isAtRisk(fileSizeBytes: UInt64, physicalMemoryBytes: UInt64) -> Bool {
        guard physicalMemoryBytes > 0 else { return false }
        let (threshold, overflow) = physicalMemoryBytes.multipliedReportingOverflow(
            by: memoryMultiplier)
        return fileSizeBytes < (overflow ? UInt64.max : threshold)
    }

    /// One line for stderr. Nil when the run is not at risk.
    public static func warning(
        fileSizeBytes: UInt64,
        physicalMemoryBytes: UInt64,
        cacheMode: CacheMode
    ) -> String? {
        guard isAtRisk(fileSizeBytes: fileSizeBytes, physicalMemoryBytes: physicalMemoryBytes)
        else { return nil }

        let sizes =
            "\(ByteSize.format(fileSizeBytes)) file, \(ByteSize.format(physicalMemoryBytes)) of memory"
        switch cacheMode {
        case .pageCache:
            return
                "warning: \(sizes) — this file fits in the page cache, so the result may "
                + "describe memory bandwidth rather than storage. Use a larger file, a freshly "
                + "generated one, or a just-mounted volume."
        case .noCache:
            return
                "warning: \(sizes) — F_NOCACHE stops these reads populating the cache but does "
                + "not evict pages an earlier run left resident, so the result may still "
                + "describe memory bandwidth. Regenerate the file or use a larger one."
        }
    }
}
