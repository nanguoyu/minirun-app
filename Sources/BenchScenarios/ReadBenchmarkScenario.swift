import Foundation
import MLX
import StorageCore

/// A bounded read benchmark against real container files, wherever they are.
///
/// One scenario serves both storage questions this phase asks — what internal
/// NAND does, and what the external NVMe does through the phone's USB3 port —
/// because they differ only in which directory the operator points at. Two
/// scenarios would have been two chances for the internal and external numbers
/// to be produced by subtly different code, and the comparison between them is
/// the whole point.
///
/// It reads **real artifact files, read-only**, rather than a generated one.
/// That costs the content verification `minirun bench --verify` can do, and buys
/// the thing that matters more here: the external volume is not ours to write
/// to, and a benchmark that needed to write would not be runnable against it at
/// all.
public struct ReadBenchmarkScenario: HarnessScenario {

    /// The read shapes this phase asks for. Sequential is the headline
    /// bandwidth number; `wholeExpert` is what the pager actually issues, so it
    /// is the one a decode projection can be made from.
    public enum Shape: String, Codable, Sendable, CaseIterable {
        case sequential
        case wholeExpert = "whole-expert"

        var pattern: AccessPattern {
            switch self {
            case .sequential: return .sequential
            case .wholeExpert: return .random
            }
        }

        /// 884,736 B is one Qwen expert — the request size `TileReader` issues
        /// on every gather, measured in M8.5 phase A. 8 MiB for the sequential
        /// arm, which is asking about bandwidth rather than about request cost.
        var readSizeBytes: Int {
            switch self {
            case .sequential: return 8 << 20
            case .wholeExpert: return 884_736
            }
        }
    }

    public let id: String
    public let title: String
    public let summary: String

    /// Resolved when the scenario runs, not when it is constructed: on the
    /// phone the directory comes from a document picker that has not been shown
    /// yet, and on an external volume it may have gone away in between.
    private let resolveDirectory: @Sendable () throws -> URL
    private let shapes: [Shape]
    private let secondsPerShape: Double
    private let queueDepth: Int
    /// Run a small GPU workload alongside the reads.
    ///
    /// This is spec §6.3's open question #1 in its cheapest honest form: does
    /// sustained streaming from an external volume interfere with concurrent
    /// Metal work, or the reverse? A read benchmark on an idle GPU cannot
    /// answer it, and the real runtime never has an idle GPU.
    ///
    /// Deliberately trivial and deliberately *small*: a few matmuls on
    /// megabyte-scale arrays, sized to keep the GPU busy without competing for
    /// the memory the pager is trying to bound. It is a disturbance, not a
    /// second benchmark, and it reports only whether it kept running.
    private let concurrentMetalLoad: Bool
    /// Held for the duration of the run and released after. On iOS a
    /// security-scoped URL stops being readable the moment this is dropped, and
    /// a read that fails for that reason must not be mistaken for a drive
    /// disconnect.
    private let accessBracket: (@Sendable () -> Void)?

    public init(
        id: String,
        title: String,
        summary: String,
        shapes: [Shape] = Shape.allCases,
        secondsPerShape: Double = 60,
        queueDepth: Int = 4,
        concurrentMetalLoad: Bool = false,
        resolveDirectory: @escaping @Sendable () throws -> URL,
        accessBracket: (@Sendable () -> Void)? = nil
    ) {
        self.id = id
        self.title = title
        self.summary = summary
        self.shapes = shapes
        self.secondsPerShape = secondsPerShape
        self.queueDepth = queueDepth
        self.concurrentMetalLoad = concurrentMetalLoad
        self.resolveDirectory = resolveDirectory
        self.accessBracket = accessBracket
    }

    public func run(_ context: ScenarioContext) throws -> ScenarioResult {
        let directory = try resolveDirectory()
        defer { accessBracket?() }

        let files = try Self.readableFiles(in: directory)
        guard !files.isEmpty else {
            throw ScenarioError.notReady(
                "no readable regular files of usable size in \(directory.path)")
        }
        context.log("reading from \(directory.path)")
        context.log("\(files.count) candidate file(s); largest \(ByteSize.format(files[0].size))")

        let volume = StorageInfo.describing(path: directory.path)
        if let name = volume.volumeName {
            context.log(
                "volume: \(name)"
                    + (volume.filesystemType.map { " (\($0))" } ?? "")
                    + (volume.volumeIsRemovable == true ? ", removable" : "")
                    + (volume.volumeIsReadOnly == true ? ", read-only" : ""))
        }

        var runs: [ShapeRun] = []
        let started = Date()

        var agitator: MetalAgitator?
        if concurrentMetalLoad {
            let running = MetalAgitator()
            running.start()
            agitator = running
            context.log("Metal workload running alongside the reads (open question #1)")
        }

        for (index, shape) in shapes.enumerated() {
            try context.checkCancellation()
            // A different file per shape where there is one, so the second
            // shape cannot be served from the page cache the first one warmed.
            // `F_NOCACHE` is set as well; belt and braces, because spec §5.7
            // treats a warm-cache measurement as no measurement at all.
            let file = files[index % files.count]
            context.log(
                "\(shape.rawValue): \(ByteSize.format(UInt64(shape.readSizeBytes))) reads, "
                    + "depth \(queueDepth), \(Int(secondsPerShape)) s, on \(file.url.lastPathComponent)")

            let configuration = BenchmarkConfiguration(
                path: file.url.path,
                readSizeBytes: shape.readSizeBytes,
                queueDepth: queueDepth,
                durationSeconds: secondsPerShape,
                warmupSeconds: 1,
                pattern: shape.pattern,
                mode: .pread,
                disableCache: true,
                seed: 42,
                verify: false)

            let before = HarnessTelemetry.sample(declaredBudgetBytes: 0)
            context.report(
                ScenarioProgress(
                    phase: shape.rawValue,
                    fraction: Double(index) / Double(shapes.count),
                    detail: file.url.lastPathComponent,
                    telemetry: before))

            let result: BenchmarkResult
            do {
                result = try StorageBenchmark.run(configuration: configuration) { warning in
                    context.log("warning: \(warning)")
                }
            } catch {
                // Spec §12.4/§19.4: the affected source and the reason are the
                // report, and a surfaced failure here is a result rather than a
                // crash. Re-thrown so the harness marks the run failed, but
                // logged first so the reason survives.
                context.log("READ FAILED on \(file.url.path): \(error)")
                throw error
            }

            let after = HarnessTelemetry.sample(declaredBudgetBytes: 0)
            let throughput = result.io.throughputMbPerSecond
            context.log(
                "\(shape.rawValue): \(String(format: "%.3f", throughput / 1000)) GB/s, "
                    + "\(result.io.readCount) reads, \(result.io.shortReadCount) short, "
                    + "\(result.io.errorCount) errors, thermal "
                    + "\(result.runtime.thermalStateStart) -> \(result.runtime.thermalStateEnd)")
            if result.io.shortReadCount > 0 || result.io.errorCount > 0 {
                context.log(
                    "NOTE: short reads or errors are the §12.4 path doing its job — "
                        + "recorded, not swallowed")
            }
            context.report(
                ScenarioProgress(
                    phase: shape.rawValue,
                    fraction: Double(index + 1) / Double(shapes.count),
                    detail: String(format: "%.3f GB/s", throughput / 1000),
                    bytesRead: result.io.bytesRead,
                    // `throughput` is the result's decimal MB/s; the progress
                    // field is bytes per second, and conflating the two is how a
                    // report ends up a million times wrong.
                    bytesPerSecond: throughput * 1e6,
                    telemetry: after))

            runs.append(ShapeRun(shape: shape, file: file.url.lastPathComponent, result: result))
        }

        var metalIterations: Int?
        if let agitator {
            let outcome = agitator.stop()
            metalIterations = outcome.iterations
            context.log(
                "Metal workload completed \(outcome.iterations) matmuls during the run"
                    + (outcome.failure.map { " — but reported: \($0)" } ?? ""))
            if outcome.iterations == 0 {
                context.log(
                    "WARNING: the Metal workload did no work, so this run does NOT answer "
                        + "the concurrency question")
            }
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let payload = Payload(
            scenario: id,
            revision: BuildInfo.gitRevision,
            directory: directory.path,
            volume: volume,
            elapsedSeconds: Date().timeIntervalSince(started),
            metalIterations: metalIterations,
            runs: runs)

        let best = runs.map { $0.result.io.throughputMbPerSecond }.max() ?? 0
        let shortReads = runs.reduce(0) { $0 + $1.result.io.shortReadCount }
        let errors = runs.reduce(0) { $0 + $1.result.io.errorCount }
        return ScenarioResult(
            headline: String(format: "peak %.3f GB/s", best / 1000)
                + (shortReads + errors > 0 ? ", \(shortReads) short / \(errors) errors" : ""),
            lines: runs.map {
                String(
                    format: "%@: %.3f GB/s over %@", $0.shape.rawValue,
                    $0.result.io.throughputMbPerSecond / 1000,
                    ByteSize.format($0.result.io.bytesRead))
            },
            json: try? encoder.encode(payload),
            jsonFilename: "\(id).json")
    }

    struct ShapeRun: Encodable {
        let shape: Shape
        let file: String
        let result: BenchmarkResult
    }

    struct Payload: Encodable {
        let scenario: String
        let revision: String
        let directory: String
        let volume: StorageInfo
        let elapsedSeconds: Double
        /// Nil when no Metal workload was asked for; 0 would mean it was asked
        /// for and did nothing, which is a different and much worse thing.
        let metalIterations: Int?
        let runs: [ShapeRun]
    }

    /// A small, continuous GPU workload for the duration of a read benchmark.
    ///
    /// Runs on its own thread and stops when told. It counts its own iterations
    /// so the record can say the GPU really was busy — a "concurrent Metal
    /// workload" that silently failed to start would turn a null result into a
    /// false one.
    final class MetalAgitator: @unchecked Sendable {
        private let stopFlag = ManagedAtomicFlag()
        private let lock = NSLock()
        private var iterations = 0
        private var failure: String?
        private var thread: Thread?

        func start() {
            let thread = Thread { [self] in
                // 512x512 float32 is 1 MiB per operand. Big enough that the
                // GPU is doing real work, small enough that this cannot be
                // mistaken for the memory pressure the pager is bounding.
                let a = MLXRandom.uniform(low: 0, high: 1, [512, 512])
                let b = MLXRandom.uniform(low: 0, high: 1, [512, 512])
                while !stopFlag.isSet {
                    let c = matmul(a, b)
                    c.eval()
                    lock.lock()
                    iterations += 1
                    lock.unlock()
                }
            }
            thread.name = "minirun.metal-agitator"
            thread.start()
            self.thread = thread
        }

        func stop() -> (iterations: Int, failure: String?) {
            stopFlag.set()
            // The loop checks the flag once per matmul, so this returns within
            // one iteration rather than needing a join.
            Thread.sleep(forTimeInterval: 0.2)
            lock.lock()
            defer { lock.unlock() }
            return (iterations, failure)
        }
    }

    struct Candidate {
        let url: URL
        let size: UInt64
    }

    /// Regular files big enough for a read of the largest shape, largest first.
    ///
    /// Sorted rather than "first found" so two runs against the same directory
    /// choose the same files, which is what makes internal and external
    /// comparable.
    ///
    /// Recurses, bounded by `maximumDepth`, because the two artifact layouts
    /// this project produces do not agree about nesting: the Qwen containers
    /// are 144 files in one flat directory, while the flagship K3 artifact
    /// groups its blobs under a per-unit subdirectory (spec §9.2). A scan that
    /// only looked at the top level would find nothing on the external volume
    /// and report "no readable files" for a drive that is full of them —
    /// a false negative that reads exactly like a mount failure.
    ///
    /// Depth is bounded rather than unlimited so that pointing this at a volume
    /// root cannot turn into an unbounded walk of someone's whole disk.
    static func readableFiles(
        in directory: URL, minimumBytes: UInt64 = 64 << 20, maximumDepth: Int = 3
    ) throws -> [Candidate] {
        let manager = FileManager.default
        var found: [Candidate] = []
        var directories: [(URL, Int)] = [(directory, 0)]

        while let (current, depth) = directories.popLast() {
            let names: [URL]
            do {
                names = try manager.contentsOfDirectory(
                    at: current,
                    includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey, .isDirectoryKey],
                    options: [.skipsHiddenFiles])
            } catch {
                // An unreadable subdirectory is not a reason to abandon the
                // scan; on a removable volume it may simply have gone away.
                continue
            }
            for url in names {
                let values = try? url.resourceValues(
                    forKeys: [.fileSizeKey, .isRegularFileKey, .isDirectoryKey])
                if values?.isDirectory == true {
                    if depth < maximumDepth { directories.append((url, depth + 1)) }
                    continue
                }
                guard values?.isRegularFile == true, let size = values?.fileSize, size > 0 else {
                    continue
                }
                let bytes = UInt64(size)
                if bytes >= minimumBytes { found.append(Candidate(url: url, size: bytes)) }
            }
        }
        return found.sorted { ($0.size, $0.url.path) > ($1.size, $1.url.path) }
    }
}
