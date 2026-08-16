/// Checked arithmetic for counters whose value is part of a correctness claim.
///
/// A wrapping counter can return to a plausible small value and make an
/// accounting identity look balanced. These helpers either return no sum, or
/// saturate while carrying a sticky overflow bit that callers can report.
public enum UInt64Accounting {
    public struct SaturatingSum: Sendable, Equatable {
        public private(set) var value: UInt64
        public private(set) var didOverflow: Bool

        public init(value: UInt64 = 0, didOverflow: Bool = false) {
            self.value = value
            self.didOverflow = didOverflow
            if didOverflow { self.value = .max }
        }

        public mutating func add(_ increment: UInt64) {
            guard !didOverflow else { return }
            let result = value.addingReportingOverflow(increment)
            if result.overflow {
                value = .max
                didOverflow = true
            } else {
                value = result.partialValue
            }
        }
    }

    public static func checkedSum<S: Sequence>(_ values: S) -> UInt64?
    where S.Element == UInt64 {
        var total: UInt64 = 0
        for value in values {
            let result = total.addingReportingOverflow(value)
            guard !result.overflow else { return nil }
            total = result.partialValue
        }
        return total
    }

    public static func saturatingSum<S: Sequence>(_ values: S) -> SaturatingSum
    where S.Element == UInt64 {
        var total = SaturatingSum()
        for value in values { total.add(value) }
        return total
    }
}
