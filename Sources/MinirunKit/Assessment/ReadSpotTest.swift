import Darwin
import Foundation
import StorageCore

/// One file the spot test will read, and how big it is.
///
/// Targets are always files that are *already on the volume*. Nothing in this
/// module creates a file, and that is a rule rather than an implementation
/// detail: the volume this instrument matters most on holds the published
/// artifact and is mounted read-only by standing instruction, so a benchmark
/// that needs to write is a benchmark that cannot be run where the answer is
/// needed. Reading the artifact's own bytes also measures the shape of read the
/// pager will actually issue, which a freshly written scratch file would not.
public struct SpotTestTarget: Sendable, Equatable, Codable {
    public let path: String
    public let sizeBytes: UInt64

    public init(path: String, sizeBytes: UInt64) {
        self.path = path
        self.sizeBytes = sizeBytes
    }
}

/// Why a volume could not be measured.
///
/// Every case is reported to the operator verbatim. None of them is recoverable
/// by trying something else — in particular, none of them is recoverable by
/// writing a test file, which this module never does.
public enum ReadSpotRefusal: Error, Equatable, CustomStringConvertible {
    /// The location is not there, or the volume is not mounted.
    case locationUnreadable(path: String, reason: String)
    /// The location holds no file big enough to measure against.
    case noTargets(path: String, largestFoundBytes: UInt64, minimumBytes: UInt64)
    case openFailed(path: String, code: Int32)
    case readFailed(path: String, code: Int32)
    case cancelled

    public var description: String {
        switch self {
        case .locationUnreadable(let path, let reason):
            return "'\(path)' could not be read: \(reason)"
        case .noTargets(let path, let largest, let minimum):
            return
                "nothing under '\(path)' is large enough to measure a link with — the biggest file "
                + "found is \(ByteSize.format(largest)) and the test needs at least "
                + "\(ByteSize.format(minimum)). Nothing was written to the volume to make one."
        case .openFailed(let path, let code):
            return "could not open '\(path)' for reading: \(String(cString: strerror(code)))"
        case .readFailed(let path, let code):
            return "a read of '\(path)' failed: \(String(cString: strerror(code)))"
        case .cancelled:
            return "the spot test was stopped before it finished"
        }
    }
}

/// Which half of the test is running, and how far it has got.
public struct ReadSpotProgress: Sendable, Equatable {
    public enum Phase: String, Sendable, Codable {
        case choosingTargets
        case sequential
        case scattered
    }

    public let phase: Phase
    public let bytesRead: UInt64
    public let plannedBytes: UInt64

    public init(phase: Phase, bytesRead: UInt64, plannedBytes: UInt64) {
        self.phase = phase
        self.bytesRead = bytesRead
        self.plannedBytes = plannedBytes
    }

    public var fraction: Double {
        guard plannedBytes > 0 else { return 0 }
        return min(1, Double(bytesRead) / Double(plannedBytes))
    }
}

/// What a spot test measured. Two rates, because they answer different
/// questions.
public struct ReadSpotResult: Codable, Sendable, Equatable {
    public let locationPath: String
    /// The files that were read, largest first. Recorded so a second run can be
    /// compared against the first without wondering what it read.
    public let targets: [SpotTestTarget]
    public let sequentialBytesRead: UInt64
    /// Time inside the sequential read loop. Opens and closes are outside it:
    /// they are constant, tiny, and not part of what a pager pays per byte.
    public let sequentialSeconds: Double
    public let scatteredBytesRead: UInt64
    public let scatteredSeconds: Double
    public let scatteredReadCount: Int
    public let scatteredReadBytes: Int
    /// Reads that came back short. Never zero silently: a removable drive going
    /// away shows up here first (spec §4.5).
    public let shortReadCount: Int
    /// `F_NOCACHE` was accepted on every descriptor. When false the numbers may
    /// be measuring the page cache and say so.
    public let cacheBypassed: Bool
    public let startedAt: Date
    public let durationSeconds: Double

    public init(
        locationPath: String, targets: [SpotTestTarget],
        sequentialBytesRead: UInt64, sequentialSeconds: Double,
        scatteredBytesRead: UInt64, scatteredSeconds: Double,
        scatteredReadCount: Int, scatteredReadBytes: Int,
        shortReadCount: Int, cacheBypassed: Bool, startedAt: Date, durationSeconds: Double
    ) {
        self.locationPath = locationPath
        self.targets = targets
        self.sequentialBytesRead = sequentialBytesRead
        self.sequentialSeconds = sequentialSeconds
        self.scatteredBytesRead = scatteredBytesRead
        self.scatteredSeconds = scatteredSeconds
        self.scatteredReadCount = scatteredReadCount
        self.scatteredReadBytes = scatteredReadBytes
        self.shortReadCount = shortReadCount
        self.cacheBypassed = cacheBypassed
        self.startedAt = startedAt
        self.durationSeconds = durationSeconds
    }

    public var totalBytesRead: UInt64 { sequentialBytesRead &+ scatteredBytesRead }

    public var sequentialBytesPerSecond: Double {
        guard sequentialSeconds > 0 else { return 0 }
        return Double(sequentialBytesRead) / sequentialSeconds
    }

    public var scatteredBytesPerSecond: Double {
        guard scatteredSeconds > 0 else { return 0 }
        return Double(scatteredBytesRead) / scatteredSeconds
    }

    /// The rate a link classification and a projection are computed from.
    ///
    /// The sequential figure, and deliberately not the scattered one: the
    /// deterministic half of a token is a long run of large reads, it is the
    /// larger half of the bytes, and it is the term the record's 0.95 versus
    /// 2.84 GB/s comparison was measured on. The scattered figure is reported
    /// beside it because a drive whose random rate collapses is a drive whose
    /// expert stream will collapse, and that is worth seeing — but it is not the
    /// denominator.
    public var referenceBytesPerSecond: Double { sequentialBytesPerSecond }
}

/// A bounded, operator-triggered read benchmark against files already on a
/// volume.
///
/// It exists because a measured rate cannot lie about the link. The same
/// enclosure on this project's own bench enumerated at USB4 40 Gb/s and at USB3
/// 10 Gb/s on different days, and the difference was 3.4× on the token — a
/// difference nobody saw until the deterministic term was divided into the bytes
/// (`docs/experiments/2026-08-11-readahead-mac.md`). Two attributions were made
/// and both were wrong before anyone looked at the bus. This is the instrument
/// that would have looked.
///
/// It never runs by itself. A screen appearing is not consent to read two
/// gigabytes off somebody's drive, exactly as a screen appearing is not consent
/// to digest 1.56 TB (``ArtifactVerification``).
public struct ReadSpotTest: Sendable {

    public struct Configuration: Sendable, Equatable, Codable {
        /// Bytes of sequential reading. ~2 GB by default: long enough that a
        /// 0.95 GB/s link and a 2.84 GB/s link are hundreds of milliseconds
        /// apart rather than tens, short enough that the operator gets an
        /// answer in a couple of seconds.
        public var sequentialBytes: UInt64
        /// Chunk size for the sequential half. 4 MiB is the shape the
        /// deterministic stream reads in.
        public var sequentialReadBytes: Int
        /// How many scattered reads. A few hundred, so the number is a rate and
        /// not a sample of one seek.
        public var scatteredReadCount: Int
        /// Size of one scattered read — whole-expert-shaped, not 4 KiB. A
        /// routed expert is a multi-megabyte contiguous run, and a 4 KiB random
        /// benchmark would measure an IOPS wall this product never hits.
        public var scatteredReadBytes: Int
        /// A file smaller than this is not worth measuring against: the
        /// sequential half would spend its time in `open` rather than in
        /// `pread`.
        public var minimumTargetBytes: UInt64
        public var maximumTargets: Int
        /// Fixed, so two runs read the same offsets and are comparable.
        public var seed: UInt64

        public init(
            sequentialBytes: UInt64, sequentialReadBytes: Int,
            scatteredReadCount: Int, scatteredReadBytes: Int,
            minimumTargetBytes: UInt64, maximumTargets: Int, seed: UInt64
        ) {
            self.sequentialBytes = sequentialBytes
            self.sequentialReadBytes = sequentialReadBytes
            self.scatteredReadCount = scatteredReadCount
            self.scatteredReadBytes = scatteredReadBytes
            self.minimumTargetBytes = minimumTargetBytes
            self.maximumTargets = maximumTargets
            self.seed = seed
        }

        /// What the app's Assess button runs: ~2 GB sequential plus 256 × 4 MiB
        /// scattered, so ~3.07 GB in total.
        public static let standard = Configuration(
            sequentialBytes: 2_000_000_000,
            sequentialReadBytes: 4 << 20,
            scatteredReadCount: 256,
            scatteredReadBytes: 4 << 20,
            minimumTargetBytes: 64 << 20,
            maximumTargets: 16,
            seed: 0x5370_6F74_5465_7374)

        /// The same instrument at a quarter of the reading, for a test arm that
        /// has to finish inside a suite. ~512 MB in total.
        public static let reduced = Configuration(
            sequentialBytes: 384_000_000,
            sequentialReadBytes: 4 << 20,
            scatteredReadCount: 32,
            scatteredReadBytes: 4 << 20,
            minimumTargetBytes: 16 << 20,
            maximumTargets: 8,
            seed: 0x5370_6F74_5465_7374)

        /// Bytes the test intends to read. Shown before it starts, because an
        /// operator is entitled to know the size of what they just authorised.
        public var plannedBytes: UInt64 {
            sequentialBytes &+ UInt64(scatteredReadCount) &* UInt64(scatteredReadBytes)
        }
    }

    public let configuration: Configuration

    public init(configuration: Configuration = .standard) {
        self.configuration = configuration
    }

    // MARK: - Choosing what to read

    /// The largest files under `root`, largest first.
    ///
    /// Largest first and not a random sample, for two reasons. A large file is
    /// where a sequential run of any length can be taken without wrapping, and
    /// on a minirun artifact the large files *are* the payload — the tile
    /// containers a run reads — so the measurement is taken on the bytes the
    /// product will actually stream. Ties break on path so the choice is
    /// reproducible across runs and across machines.
    ///
    /// Metadata only: `stat` per entry, nothing opened.
    public func targets(under root: URL, fileManager: FileManager = .default) throws
        -> [SpotTestTarget]
    {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory) else {
            throw ReadSpotRefusal.locationUnreadable(
                path: root.path, reason: "the path does not exist")
        }
        guard isDirectory.boolValue else {
            guard
                let size = (try? root.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
                    .map(UInt64.init)
            else {
                throw ReadSpotRefusal.locationUnreadable(
                    path: root.path, reason: "its size could not be read")
            }
            return [SpotTestTarget(path: root.path, sizeBytes: size)]
        }
        guard
            let enumerator = fileManager.enumerator(
                at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
                options: [.skipsPackageDescendants, .skipsHiddenFiles])
        else {
            throw ReadSpotRefusal.locationUnreadable(
                path: root.path, reason: "the directory could not be enumerated")
        }

        var found: [SpotTestTarget] = []
        var largest: UInt64 = 0
        for case let url as URL in enumerator {
            let name = url.lastPathComponent
            // The same bookkeeping the locator skips. `.cache` is the
            // downloader's part-file store: reading a half-written part would
            // measure the same drive and describe a file nobody will stream.
            if ArtifactLocator.ignoredDirectories.contains(name) {
                enumerator.skipDescendants()
                continue
            }
            if ArtifactLocator.ignoredFiles.contains(name) { continue }
            guard let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                values.isRegularFile == true, let size = values.fileSize.map(UInt64.init)
            else { continue }
            largest = max(largest, size)
            guard size >= configuration.minimumTargetBytes else { continue }
            found.append(SpotTestTarget(path: url.path, sizeBytes: size))
        }

        guard !found.isEmpty else {
            throw ReadSpotRefusal.noTargets(
                path: root.path, largestFoundBytes: largest,
                minimumBytes: configuration.minimumTargetBytes)
        }
        return
            found
            .sorted {
                $0.sizeBytes == $1.sizeBytes ? $0.path < $1.path : $0.sizeBytes > $1.sizeBytes
            }
            .prefix(configuration.maximumTargets)
            .map { $0 }
    }

    // MARK: - Running

    /// Read, measure, and report. Synchronous on purpose — the caller decides
    /// which thread pays for it, and the app pays for it off the main actor.
    ///
    /// `isCancelled` is consulted between reads, so a stop lands within one
    /// chunk rather than at the end of the run.
    public func run(
        on root: URL,
        fileManager: FileManager = .default,
        isCancelled: () -> Bool = { false },
        progress: (ReadSpotProgress) -> Void = { _ in }
    ) throws -> ReadSpotResult {
        let startedAt = Date()
        let runStart = MonotonicClock.now()
        progress(ReadSpotProgress(phase: .choosingTargets, bytesRead: 0, plannedBytes: configuration.plannedBytes))

        let targets = try targets(under: root, fileManager: fileManager)
        if isCancelled() { throw ReadSpotRefusal.cancelled }

        let bufferBytes = max(configuration.sequentialReadBytes, configuration.scatteredReadBytes)
        let buffer = try AlignedBuffer(count: bufferBytes)
        var cacheBypassed = true
        var shortReads = 0
        var bytesSoFar: UInt64 = 0

        // MARK: Sequential

        var sequentialBytes: UInt64 = 0
        var sequentialNanos: UInt64 = 0
        var remaining = configuration.sequentialBytes

        for target in targets where remaining > 0 {
            let descriptor = try open(target.path, bypassCache: &cacheBypassed)
            defer { close(descriptor) }
            var offset: UInt64 = 0
            while remaining > 0, offset < target.sizeBytes {
                if isCancelled() { throw ReadSpotRefusal.cancelled }
                let want = Int(min(UInt64(configuration.sequentialReadBytes), remaining))
                let start = MonotonicClock.now()
                let got = pread(descriptor, buffer.pointer, want, off_t(offset))
                sequentialNanos &+= MonotonicClock.now() &- start
                if got < 0 {
                    if errno == EINTR { continue }
                    throw ReadSpotRefusal.readFailed(path: target.path, code: errno)
                }
                if got == 0 { break }
                if got < want { shortReads += 1 }
                sequentialBytes &+= UInt64(got)
                bytesSoFar &+= UInt64(got)
                offset &+= UInt64(got)
                remaining &-= min(remaining, UInt64(got))
                progress(
                    ReadSpotProgress(
                        phase: .sequential, bytesRead: bytesSoFar,
                        plannedBytes: configuration.plannedBytes))
            }
        }

        // MARK: Scattered
        //
        // Every descriptor is opened once and the reads are issued across all of
        // them in a fixed pseudo-random order. Opening per read would time the
        // open; issuing them file by file would turn a scatter into several
        // short sequential runs, which is the thing this half exists not to
        // measure.

        var descriptors: [Int32] = []
        defer { for descriptor in descriptors { close(descriptor) } }
        var eligible: [SpotTestTarget] = []
        for target in targets where target.sizeBytes > UInt64(configuration.scatteredReadBytes) {
            descriptors.append(try open(target.path, bypassCache: &cacheBypassed))
            eligible.append(target)
        }

        var scatteredBytes: UInt64 = 0
        var scatteredNanos: UInt64 = 0
        var completedScattered = 0
        if !eligible.isEmpty {
            var random = SplitMix64(seed: configuration.seed)
            for _ in 0..<configuration.scatteredReadCount {
                if isCancelled() { throw ReadSpotRefusal.cancelled }
                let index = Int(random.next() % UInt64(eligible.count))
                let target = eligible[index]
                let span = target.sizeBytes - UInt64(configuration.scatteredReadBytes)
                // Page-aligned, because an unaligned pread on a raw device path
                // is a different measurement and the pager issues aligned reads.
                let offset = (random.next() % (span + 1)) & ~UInt64(0xFFF)
                let start = MonotonicClock.now()
                let got = pread(
                    descriptors[index], buffer.pointer, configuration.scatteredReadBytes,
                    off_t(offset))
                scatteredNanos &+= MonotonicClock.now() &- start
                if got < 0 {
                    if errno == EINTR { continue }
                    throw ReadSpotRefusal.readFailed(path: target.path, code: errno)
                }
                if got < configuration.scatteredReadBytes { shortReads += 1 }
                scatteredBytes &+= UInt64(got)
                bytesSoFar &+= UInt64(got)
                completedScattered += 1
                progress(
                    ReadSpotProgress(
                        phase: .scattered, bytesRead: bytesSoFar,
                        plannedBytes: configuration.plannedBytes))
            }
        }

        return ReadSpotResult(
            locationPath: root.path,
            targets: targets,
            sequentialBytesRead: sequentialBytes,
            sequentialSeconds: Double(sequentialNanos) / 1_000_000_000,
            scatteredBytesRead: scatteredBytes,
            scatteredSeconds: Double(scatteredNanos) / 1_000_000_000,
            scatteredReadCount: completedScattered,
            scatteredReadBytes: configuration.scatteredReadBytes,
            shortReadCount: shortReads,
            cacheBypassed: cacheBypassed,
            startedAt: startedAt,
            durationSeconds: MonotonicClock.seconds(since: runStart))
    }

    /// `O_RDONLY`, and `F_NOCACHE` so the reads do not populate the page cache.
    ///
    /// A failing `F_NOCACHE` is recorded rather than thrown: some filesystems
    /// refuse it, and a measurement that says "this may have been the page
    /// cache" is more useful than no measurement (spec §5.7, and
    /// ``CacheConfounding`` for why the flag is not a complete answer).
    private func open(_ path: String, bypassCache: inout Bool) throws -> Int32 {
        let descriptor = Darwin.open(path, O_RDONLY)
        guard descriptor >= 0 else {
            throw ReadSpotRefusal.openFailed(path: path, code: errno)
        }
        if fcntl(descriptor, F_NOCACHE, 1) == -1 { bypassCache = false }
        return descriptor
    }
}
