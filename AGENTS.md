# Contributor operating rules

Read `README.md`, `CONTRIBUTING.md` and `docs/ARCHITECTURE.md` before a
nontrivial change.

- Correctness precedes optimization.
- Memory use is a stated budget, never an emergent full-model fallback.
- Keep the dependency direction:
  `StorageCore <- ModelAdapters <- BenchScenarios <- MinirunRunners` and
  `StorageCore <- MinirunKit <- Apps/Minirun`.
- `StorageCore` links Foundation and Darwin only. MLX enters from above it.
- Add tests with behavioral changes. Add reproducible benchmarks and record the
  tradeoff for I/O, memory, scheduling, or kernel changes.
- Do not commit model weights, credentials, signing configuration, captured
  conversations, private paths, or generated Xcode user state.
- Sign every commit under DCO 1.1 with `git commit -s`.

Run package tests through `xcodebuild`, not `swift test`, and use generic iOS
Simulator destinations for routine iOS compile checks. Visual changes require
visual review in light and dark appearances on both affected platforms.
