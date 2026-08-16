import Foundation
import StorageCore

/// Reads a volume, measures it, and says whether it is good enough.
///
/// The order matters and is the whole design: **measure first, classify from the
/// measurement, and project only from what was measured**. A classification that
/// started from a bus name would inherit the bus name's optimism, which is
/// exactly how a 0.95 GB/s session got attributed to a concurrent build and then
/// to a Debug configuration before anyone divided bytes by seconds
/// (`docs/experiments/2026-08-11-readahead-mac.md`).
///
/// Nothing here writes to the volume. Not a scratch file, not a temporary
/// directory, not on a volume that would allow it: the instrument has to work on
/// the read-only volume that holds the published artifact, and an instrument
/// with two paths would be measuring two different things on two different
/// machines. Where there is nothing to read, the report carries a caution and no
/// numbers.
public struct VolumeAssessor: Sendable {

    public let spotTest: ReadSpotTest
    public let marginBytes: UInt64
    private let hardwareProbe: any HardwareLinkProbing
    private let fileManager: FileManager

    public init(
        spotTest: ReadSpotTest = ReadSpotTest(),
        marginBytes: UInt64 = FitCheck.defaultMarginBytes,
        hardwareProbe: any HardwareLinkProbing = IORegistryHardwareProbe(),
        fileManager: FileManager = .default
    ) {
        self.spotTest = spotTest
        self.marginBytes = marginBytes
        self.hardwareProbe = hardwareProbe
        self.fileManager = fileManager
    }

    /// Which directory the spot test reads from.
    ///
    /// The largest artifact already on the location, when there is one: its
    /// files are the bytes a run will actually stream, and measuring them needs
    /// no new bytes to exist. The location root otherwise, which is what makes
    /// an empty registered folder produce a caution rather than a temp file.
    public func readRoot(for location: URL, scan: LocationScan?) -> URL {
        guard let artifact = scan?.artifacts.max(by: { $0.bytesOnDisk < $1.bytesOnDisk }),
            artifact.bytesOnDisk > 0
        else { return location }
        return artifact.url
    }

    /// Assess one location.
    ///
    /// Never throws. A refused spot test becomes a caution on the report,
    /// because "this volume could not be measured, and here is the sentence
    /// saying why" is a finding an operator can act on, and a thrown error at
    /// this level would take the fit checks down with it.
    public func assess(
        location: URL,
        scan: LocationScan?,
        catalog: ModelCatalogSnapshot,
        workloads: [ModelID: TokenWorkload],
        now: Date = Date(),
        isCancelled: () -> Bool = { false },
        progress: (ReadSpotProgress) -> Void = { _ in }
    ) -> VolumeAssessmentReport {
        let info = StorageInfo.describing(path: location.path)
        let space = FreeSpace(info)

        var spot: ReadSpotResult?
        var caution: String?
        do {
            spot = try spotTest.run(
                on: readRoot(for: location, scan: scan), fileManager: fileManager,
                isCancelled: isCancelled, progress: progress)
        } catch let refusal as ReadSpotRefusal {
            caution = refusal.description
        } catch {
            caution = "\(error)"
        }

        let rate = spot?.referenceBytesPerSecond
        let classification = LinkClassifier.classify(
            bytesPerSecond: rate, hardware: hardwareProbe.facts(forMountPath: location.path))

        // Bytes already on this location, per model, counted by discovery. A
        // model that is here is measured, not estimated.
        var onDisk: [ModelID: UInt64] = [:]
        for artifact in scan?.artifacts ?? [] {
            guard let model = artifact.model else { continue }
            onDisk[model] = max(onDisk[model] ?? 0, artifact.bytesOnDisk)
        }

        let verdicts =
            catalog.models
            .map { descriptor -> ModelVerdict in
                let fit = FitCheck.check(
                    model: descriptor.id, displayName: descriptor.displayName,
                    catalogBytes: descriptor.totalBytes, bytesOnDisk: onDisk[descriptor.id],
                    availableBytes: space.dependableBytes, marginBytes: marginBytes)
                let projection = workloads[descriptor.id].flatMap {
                    TokenTimeProjector.project(workload: $0, bytesPerSecond: rate)
                }
                return ModelVerdict(fit: fit, projection: projection)
            }
            .sorted { $0.fit.displayName < $1.fit.displayName }

        return VolumeAssessmentReport(
            locationPath: location.path,
            volumeName: info.volumeName ?? scan?.displayName,
            volumeMountPath: info.volumeMountPath,
            space: space,
            assessedAt: now,
            spot: spot,
            caution: caution,
            classification: classification,
            verdicts: verdicts)
    }
}
