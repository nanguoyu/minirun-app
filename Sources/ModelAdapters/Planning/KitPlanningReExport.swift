/// The memory dial's arithmetic moved to `MinirunKit`, and this file is why
/// nothing else in the adapter had to change.
///
/// ## Why it moved
///
/// `MemoryDialPlanner` is pure arithmetic over `UInt64` — no MLX, no Metal, no
/// clock, no filesystem — and two very different processes need it: the decode
/// loop, which turns a `PinPlan` into a pinned tier, and the Minirun app's
/// memory dial, which draws the plan a budget implies *before* anything is run.
/// The app links `MinirunKit` and `StorageCore` and deliberately links no MLX,
/// so a planner living in `ModelAdapters` was unreachable from the screen whose
/// whole subject it is. The app grew a second, simpler copy of the composition
/// arithmetic instead, and that copy was wrong: it had no pinned tier at all, so
/// raising the budget appeared to buy free space rather than resident layers.
///
/// One planner, in the target both halves can link, is the fix. `MinirunKit`
/// depends on `StorageCore` and `iOSBench` only, so `ModelAdapters -> MinirunKit`
/// adds no cycle and does not reverse spec §7.1's arrow: the adapter is the top
/// of that arrow and is allowed to depend downward.
///
/// ## Why a re-export rather than an import per file
///
/// `ArtifactWeightSource.swift` names `PinPlan` and `K3DecodeScenario.swift`
/// names both `PinPlan` and `MemoryDialPlanner`, and both files are frozen this
/// week. Re-exporting the module here makes the moved types visible to every
/// file that already imports `ModelAdapters`, so the relocation is a change to
/// the package graph and to no consumer's source.
@_exported import MinirunKit
