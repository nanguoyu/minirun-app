import Darwin
import Foundation

/// The memory numbers iOS actually judges a process by.
///
/// ``MemoryUsage`` reports resident size, which is the right number on the Mac
/// and the wrong one on the phone. iOS's jetsam does not look at RSS: it looks
/// at **phys_footprint**, which counts a task's compressed pages, its IOKit and
/// GPU allocations, and its purgeable memory, and excludes the clean file-backed
/// pages RSS includes. The two disagree in both directions, so a bounded-memory
/// claim made on a device has to be made in the units the killer uses.
///
/// Spec §4.4 asks for exactly one more thing — that the runtime treat the
/// budget as advisory and moving rather than fixed — and names
/// `os_proc_available_memory()` as the source. That call is iOS-only and
/// returns what this process may still allocate before it is at risk, which is
/// the complement of the footprint rather than a restatement of it.
///
/// This lives beside ``MemoryUsage`` rather than inside it because the existing
/// type is embedded in benchmark JSON that has already been published to
/// `docs/experiments/data/`; adding fields there would change files that
/// previous milestones' claims rest on.
public struct ProcessFootprint: Codable, Equatable, Sendable {

    /// `phys_footprint` from `task_vm_info`: the number iOS's memory limit is
    /// enforced against. Present on macOS too, where it is merely informative.
    public let footprintBytes: UInt64

    /// The high-water mark of ``footprintBytes`` over the process lifetime.
    public let peakFootprintBytes: UInt64

    /// What `os_proc_available_memory()` says is still available to this
    /// process. Nil on macOS, where the call does not exist, and nil on iOS if
    /// it returns 0 — which it does when the process is not memory-limited, and
    /// reporting that as "0 bytes left" would invert its meaning.
    public let availableBytes: UInt64?

    public init(footprintBytes: UInt64, peakFootprintBytes: UInt64, availableBytes: UInt64?) {
        self.footprintBytes = footprintBytes
        self.peakFootprintBytes = peakFootprintBytes
        self.availableBytes = availableBytes
    }

    public static func current() -> ProcessFootprint? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)

        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        guard status == KERN_SUCCESS else { return nil }

        // `phys_footprint_peak` sits past the end of the older, shorter
        // `TASK_VM_INFO` revision, so it is only meaningful when the kernel
        // filled in a struct long enough to contain it. Reading it
        // unconditionally would report whatever happened to be in the tail.
        let peakIsPresent =
            count >= mach_msg_type_number_t(
                MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let footprint = UInt64(info.phys_footprint)
        let peak = peakIsPresent ? UInt64(info.ledger_phys_footprint_peak) : footprint

        return ProcessFootprint(
            footprintBytes: footprint,
            // A peak the kernel reports below the current value is not a peak;
            // prefer the reading we know is live.
            peakFootprintBytes: max(peak, footprint),
            availableBytes: Self.availableMemory()
        )
    }

    /// `os_proc_available_memory()`, where it exists.
    public static func availableMemory() -> UInt64? {
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
            let available = os_proc_available_memory()
            return available > 0 ? UInt64(available) : nil
        #else
            return nil
        #endif
    }
}
