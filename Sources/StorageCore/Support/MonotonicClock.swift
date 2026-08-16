import Darwin
import Foundation

/// Monotonic nanosecond timestamps for latency accounting.
///
/// `CLOCK_UPTIME_RAW` is the raw mach timebase already expressed in
/// nanoseconds: it never steps backwards when the wall clock is adjusted, and
/// unlike `CLOCK_MONOTONIC` it does not advance while the machine sleeps —
/// a laptop lid closed mid-run should not be charged to a read. It needs no
/// timebase conversion, which keeps the per-read timing overhead to a single
/// call.
public enum MonotonicClock {
    @inline(__always)
    public static func now() -> UInt64 {
        clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    @inline(__always)
    public static func seconds(since start: UInt64) -> Double {
        Double(now() &- start) / 1_000_000_000
    }

    @inline(__always)
    public static func seconds(from start: UInt64, to end: UInt64) -> Double {
        Double(end &- start) / 1_000_000_000
    }

    @inline(__always)
    public static func nanoseconds(_ seconds: Double) -> UInt64 {
        UInt64((seconds * 1_000_000_000).rounded())
    }
}
