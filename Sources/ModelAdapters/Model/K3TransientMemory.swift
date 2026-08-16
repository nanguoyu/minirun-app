import Darwin
import MLX

/// Releases K3 working allocations only after an evaluated layer has stopped
/// owning its weights.
///
/// MLX's cache and Darwin malloc are distinct pools. Clearing only one can
/// leave the other carrying a previous layer's high-water pages into the next
/// layer or token. This operation deliberately has one shared call site in
/// ``K3Model`` so the product runner and device scenario cannot drift.
enum K3TransientMemory {
    static func reclaim() {
        MLX.Memory.clearCache()
        _ = malloc_zone_pressure_relief(nil, 0)
    }
}
