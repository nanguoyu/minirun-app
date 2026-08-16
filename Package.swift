// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "minirun",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "StorageCore", targets: ["StorageCore"]),
        .library(name: "MLXBridge", targets: ["MLXBridge"]),
        .library(name: "ModelAdapters", targets: ["ModelAdapters"]),
        .library(name: "iOSBench", targets: ["iOSBench"]),
        .library(name: "BenchScenarios", targets: ["BenchScenarios"]),
        .library(name: "MinirunKit", targets: ["MinirunKit"]),
        .library(name: "MinirunRunners", targets: ["MinirunRunners"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.4"),
    ],
    targets: [
        .target(name: "StorageCore", plugins: ["EmbedGitRevision"]),
        .testTarget(name: "StorageCoreTests", dependencies: ["StorageCore"]),
        .target(
            name: "MLXBridge",
            dependencies: [
                "StorageCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MinirunKit",
            dependencies: ["StorageCore", "iOSBench"],
            resources: [.copy("Resources/model-catalog.json")]
        ),
        .testTarget(
            name: "MinirunKitTests",
            dependencies: ["MinirunKit", "StorageCore"],
            resources: [.copy("Fixtures/hf")]
        ),
        .target(name: "iOSBench", dependencies: ["StorageCore"]),
        .testTarget(
            name: "iOSBenchTests",
            dependencies: ["iOSBench", "StorageCore"]
        ),
        .target(
            name: "ModelAdapters",
            dependencies: ["MLXBridge", "MinirunKit", "StorageCore"]
        ),
        .target(
            name: "BenchScenarios",
            dependencies: ["ModelAdapters", "iOSBench", "StorageCore"],
            resources: [.copy("Resources/k3-flagship-config.json")]
        ),
        .target(
            name: "MinirunRunners",
            dependencies: [
                "MinirunKit",
                "BenchScenarios",
                "ModelAdapters",
                "StorageCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MinirunRunnersTests",
            dependencies: ["MinirunRunners", "ModelAdapters", "StorageCore"]
        ),
        .plugin(name: "EmbedGitRevision", capability: .buildTool()),
    ]
)
