import Foundation

/// Whether layer *i+1*'s stored bytes may be read while layer *i* is resident
/// and computing, inside a stated budget.
///
/// ## Why this arithmetic lives here and not only in the adapter
///
/// `ModelAdapters`' `DeterministicReadAheadSchedule` is the *runtime* stager's
/// copy of this decision, and it stays where it is: it sits in a file that links
/// MLX, and the memory dial has to run in a UI process that links neither MLX
/// nor Metal. ``MemoryDialPlanner`` calls this one so that the plan a screen
/// draws is computed by the same formula the run will apply, on a target the app
/// can link.
///
/// Two copies of a formula drift. This one cannot, because
/// `ModelAdaptersTests/ReadAheadScheduleAgreementTests` walks the whole flagship
/// layer table at several budgets and fails if the adapter's decision and this
/// one ever differ in any field. That test is the reason this file is a
/// duplicate rather than a fork.
public enum ReadAheadPairSchedule {

    /// What one layer's deterministic stream costs, in both forms it exists in.
    public struct LayerBytes: Sendable, Equatable, Codable {
        /// Bytes as written: BF16 for nearly everything, F32 for a few vectors.
        /// This is what staging allocates and reads.
        public let storedBytes: UInt64
        /// Bytes after the weight source widens every tensor to float32. This is
        /// what the layer occupies while it is the resident one.
        public let residentBytes: UInt64

        public init(storedBytes: UInt64, residentBytes: UInt64) {
            self.storedBytes = storedBytes
            self.residentBytes = residentBytes
        }
    }

    /// Whether one `(resident, staged)` pair fits, and the arithmetic that said
    /// so. Recorded rather than recomputed, so a skip in a run's JSON — or in a
    /// dial's warning line — can be read back without the layer table.
    public struct Decision: Sendable, Equatable, Codable {
        public let residentLayer: Int
        public let stagedLayer: Int
        public let residentBytes: UInt64
        public let stagedBytes: UInt64
        /// Everything else the budget must also cover — expert pool, MLX cache,
        /// working set. Stated by the caller; zero means "this budget describes
        /// the two deterministic terms alone".
        public let reserveBytes: UInt64
        public let budgetBytes: UInt64

        public init(
            residentLayer: Int, stagedLayer: Int, residentBytes: UInt64, stagedBytes: UInt64,
            reserveBytes: UInt64, budgetBytes: UInt64
        ) {
            self.residentLayer = residentLayer
            self.stagedLayer = stagedLayer
            self.residentBytes = residentBytes
            self.stagedBytes = stagedBytes
            self.reserveBytes = reserveBytes
            self.budgetBytes = budgetBytes
        }

        /// `resident + staged + reserve`, saturating rather than trapping: a
        /// nonsensical layer table must produce a refusal, not a crash.
        public var requiredBytes: UInt64 {
            PlanningUInt64Arithmetic.sum([residentBytes, stagedBytes, reserveBytes]).value
        }

        public var isAllowed: Bool {
            let sum = residentBytes.addingReportingOverflow(stagedBytes)
            guard !sum.overflow else { return false }
            let total = sum.partialValue.addingReportingOverflow(reserveBytes)
            guard !total.overflow else { return false }
            return total.partialValue <= budgetBytes
        }

        /// Budget minus requirement. Negative on a skip, which is the number
        /// worth reading: it says how far over the pair was.
        public var headroomBytes: Int64 {
            let required = PlanningUInt64Arithmetic.sum(
                [residentBytes, stagedBytes, reserveBytes])
            // Once the requirement itself exceeds UInt64 there is no exact
            // signed deficit to report. `Int64.min` is the conservative
            // sentinel; importantly, a budget of UInt64.max must not look like
            // it missed an overflowing requirement by zero bytes.
            guard !required.overflow else { return .min }
            if required.value > budgetBytes {
                let deficit = required.value - budgetBytes
                guard deficit <= UInt64(Int64.max) else { return .min }
                return -Int64(deficit)
            }
            return Int64(clamping: budgetBytes - required.value)
        }
    }

    /// Does staging `next` while `resident` is resident fit inside `budget`?
    public static func decide(
        residentLayer: Int,
        resident: LayerBytes,
        stagedLayer: Int,
        next: LayerBytes,
        budgetBytes: UInt64,
        reserveBytes: UInt64 = 0
    ) -> Decision {
        Decision(
            residentLayer: residentLayer,
            stagedLayer: stagedLayer,
            residentBytes: resident.residentBytes,
            stagedBytes: next.storedBytes,
            reserveBytes: reserveBytes,
            budgetBytes: budgetBytes)
    }

    /// Every consecutive pair of a layer table, in order. `layers.count - 1`
    /// decisions; an empty or single-layer table has none.
    public static func plan(
        _ layers: [LayerBytes], budgetBytes: UInt64, reserveBytes: UInt64 = 0
    ) -> [Decision] {
        guard layers.count > 1 else { return [] }
        return (0..<(layers.count - 1)).map { index in
            decide(
                residentLayer: index, resident: layers[index],
                stagedLayer: index + 1, next: layers[index + 1],
                budgetBytes: budgetBytes, reserveBytes: reserveBytes)
        }
    }
}
