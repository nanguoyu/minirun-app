import Darwin
import Foundation

/// Process memory footprint, read from the kernel's task info.
///
/// Spec §18.1 lists peak process memory as a core observable, and the whole
/// point of a storage-native runtime is that the footprint stays bounded while
/// bytes stream through. Reading it costs one `task_info` call, so a benchmark
/// can afford to sample it.
public struct MemoryUsage: Codable, Equatable {
    /// Current resident set size.
    public let residentBytes: UInt64
    /// High-water resident set size for the lifetime of the process. This is
    /// the number a bounded-memory claim has to stand on.
    public let peakResidentBytes: UInt64

    public static func current() -> MemoryUsage? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }

        return MemoryUsage(
            residentBytes: UInt64(info.resident_size),
            peakResidentBytes: UInt64(info.resident_size_max)
        )
    }
}
