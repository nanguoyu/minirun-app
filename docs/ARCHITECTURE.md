# Architecture

Minirun keeps storage, model adaptation, and product execution in one public
monorepo while their interfaces still evolve together.

```text
StorageCore <- MLXBridge <- ModelAdapters <- BenchScenarios <- MinirunRunners
StorageCore <- MinirunKit <----------------------------------------^
StorageCore <- MinirunKit <- Apps/Minirun
```

- `StorageCore` owns bounded reads, containers, scheduling, and environment
  reporting. It depends only on Foundation and Darwin.
- `MLXBridge` is the thin MLX boundary above storage.
- `ModelAdapters` owns model-specific layouts and operators.
- `MinirunKit` owns catalog, download, verification, storage lifecycle, memory
  planning, and the runner protocol. It links no MLX or Metal.
- `MinirunRunners` binds verified artifacts to real K3 and V4 execution.
- `Apps/Minirun` is one SwiftUI product source tree for macOS and iOS.

The runtime never infers executability from a model name alone. Catalog
metadata, a completely verified local tree, tokenizer evidence, rooted storage
authority, architecture support, and a registered runner must all agree before
chat is enabled.

Full model weights are external artifacts. The source repository contains only
code, metadata, public app assets, and synthetic or regenerable test evidence.
