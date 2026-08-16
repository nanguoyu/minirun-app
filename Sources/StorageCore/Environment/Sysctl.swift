import Darwin
import Foundation

/// Thin `sysctlbyname` wrapper.
///
/// Every accessor returns an optional and never traps: the keys used here are
/// not uniformly present across Apple silicon generations and Intel Macs
/// (`hw.perflevel*` in particular exists only on asymmetric CPUs), and a
/// missing key must be reported as missing rather than guessed at.
public enum Sysctl {
    /// String-valued sysctl, e.g. `hw.model`, `machdep.cpu.brand_string`.
    public static func string(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        let bytes = buffer.prefix { $0 != 0 }
        let value = String(decoding: bytes, as: UTF8.self)
        return value.isEmpty ? nil : value
    }

    /// Integer-valued sysctl, widened to `UInt64` regardless of the kernel's
    /// 32- or 64-bit representation for that key.
    public static func integer(_ name: String) -> UInt64? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0 else { return nil }
        switch size {
        case MemoryLayout<UInt32>.size:
            var value: UInt32 = 0
            var length = MemoryLayout<UInt32>.size
            guard sysctlbyname(name, &value, &length, nil, 0) == 0 else { return nil }
            return UInt64(value)
        case MemoryLayout<UInt64>.size:
            var value: UInt64 = 0
            var length = MemoryLayout<UInt64>.size
            guard sysctlbyname(name, &value, &length, nil, 0) == 0 else { return nil }
            return value
        default:
            return nil
        }
    }

    /// Integer-valued sysctl narrowed to `Int`, dropping values that do not fit.
    public static func int(_ name: String) -> Int? {
        guard let value = integer(name), value <= UInt64(Int.max) else { return nil }
        return Int(value)
    }
}
