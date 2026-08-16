import Foundation
import StorageCore

/// Number formatting for the sentences this module writes.
///
/// The app's `MRFormat` is the product's formatter and this is not a second one:
/// the two rules reproduced here — decimal scale for artifact sizes and link
/// rates, `m:ss` for a token time — are the app's, spelled the same way, because
/// a verdict line composed in the kit must read identically to a number the
/// screen prints beside it. `AssessmentPhrasingTests` pins the agreement.
enum AssessmentPhrasing {

    /// Decimal — GB/TB, 1000-based. Stated budgets, artifact sizes, link rates.
    static func bytesDecimal(_ bytes: UInt64) -> String {
        let units = ["B", "kB", "MB", "GB", "TB", "PB"]
        var value = Double(bytes)
        var index = 0
        while value >= 1000, index < units.count - 1 {
            value /= 1000
            index += 1
        }
        if index == 0 { return "\(bytes) B" }
        let digits = value >= 100 ? 0 : (value >= 10 ? 1 : 2)
        return String(format: "%.\(digits)f %@", value, units[index])
    }

    static func throughput(_ bytesPerSecond: Double) -> String {
        guard bytesPerSecond > 0, bytesPerSecond.isFinite else { return "—" }
        if bytesPerSecond >= 1e9 { return String(format: "%.2f GB/s", bytesPerSecond / 1e9) }
        return String(format: "%.0f MB/s", bytesPerSecond / 1e6)
    }

    /// `m:ss` under an hour, `h:mm:ss` above it.
    static func clock(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "—" }
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 { return String(format: "%d:%02d:%02d", hours, minutes, secs) }
        return String(format: "%d:%02d", minutes, secs)
    }
}

/// One model's answer for one volume: does it fit, and how slow would a token be.
public struct ModelVerdict: Codable, Sendable, Equatable, Identifiable {
    public var id: String { fit.model.rawValue }
    public let fit: FitCheck
    /// Nil whenever no rate was measured. The line then says so instead of
    /// showing a number.
    public let projection: ProjectedTokenTime?

    public init(fit: FitCheck, projection: ProjectedTokenTime?) {
        self.fit = fit
        self.projection = projection
    }

    /// The sentence the report card shows, verbatim.
    public var line: String {
        var text = "\(fit.displayName): "
        switch fit.verdict {
        case .alreadyHere(let bytes):
            text += "\(AssessmentPhrasing.bytesDecimal(bytes)) already here"
        case .fits(let spare):
            text +=
                "\(AssessmentPhrasing.bytesDecimal(fit.requiredBytes)) fits, "
                + "\(AssessmentPhrasing.bytesDecimal(UInt64(max(0, spare)))) to spare"
        case .short(let by):
            text +=
                "\(AssessmentPhrasing.bytesDecimal(fit.requiredBytes)) does not fit — "
                + "\(AssessmentPhrasing.bytesDecimal(by)) short of the free space plus "
                + "\(AssessmentPhrasing.bytesDecimal(fit.marginBytes)) margin"
        case .unknown(let reason):
            text += "fit unknown — \(reason)"
        }
        if let projection {
            text +=
                ". ≈ \(AssessmentPhrasing.clock(projection.secondsPerTokenLow)) – "
                + "\(AssessmentPhrasing.clock(projection.secondsPerTokenHigh)) / token "
                + "at \(AssessmentPhrasing.throughput(projection.bytesPerSecond))"
        } else {
            text += ". No rate was measured here, so no token time is projected"
        }
        return text + "."
    }
}

/// Everything one assessment of one location established.
///
/// One value, Codable, so the whole of it can be persisted and read back. It is
/// dated because a storage assessment goes out of date the moment a drive is
/// replugged into a different port — which is precisely the failure this module
/// exists to catch, and would be an absurd thing for it to hide behind a number
/// with no date on it.
public struct VolumeAssessmentReport: Codable, Sendable, Equatable, Identifiable {
    public var id: String { locationPath }
    /// The registered location that was assessed.
    public let locationPath: String
    public let volumeName: String?
    public let volumeMountPath: String?
    public let space: FreeSpace
    public let assessedAt: Date
    /// Nil when no test was run. Then ``caution`` says why, and every projection
    /// is nil.
    public let spot: ReadSpotResult?
    /// Why there is no spot test, shown verbatim. Nil when there is one.
    public let caution: String?
    public let classification: LinkClassification
    public let verdicts: [ModelVerdict]

    public init(
        locationPath: String, volumeName: String?, volumeMountPath: String?, space: FreeSpace,
        assessedAt: Date, spot: ReadSpotResult?, caution: String?,
        classification: LinkClassification, verdicts: [ModelVerdict]
    ) {
        self.locationPath = locationPath
        self.volumeName = volumeName
        self.volumeMountPath = volumeMountPath
        self.space = space
        self.assessedAt = assessedAt
        self.spot = spot
        self.caution = caution
        self.classification = classification
        self.verdicts = verdicts
    }

    /// Seven days, the same window a link calibration goes stale in. A report
    /// past it is labelled stale and never quietly reused.
    public static let stalenessSeconds: TimeInterval = 7 * 86400

    public func isStale(now: Date = Date()) -> Bool {
        now.timeIntervalSince(assessedAt) > Self.stalenessSeconds
    }

    public var measuredBytesPerSecond: Double? { spot?.referenceBytesPerSecond }

    /// The one line a caution banner shows when the link is a degraded one.
    ///
    /// Only ever composed when the table has an entry for a faster tier, so the
    /// multiple is a ratio of two measured numbers and not an extrapolation.
    public var degradedLinkSentence: String? {
        guard classification.isDegraded, let rate = classification.measuredBytesPerSecond,
            let multiple = classification.fasterTierMultiple,
            let name = classification.fasterTierName
        else { return nil }
        return
            "This volume is reading at \(AssessmentPhrasing.throughput(rate)). On this project's "
            + "own bench the same class of enclosure reached "
            + "\(AssessmentPhrasing.throughput(rate * multiple)) over a \(name) — about "
            + String(format: "%.1f", multiple) + "× faster. A token here costs that difference."
    }

    public var headline: String {
        guard let spot else {
            return caution ?? "This location was not measured."
        }
        return
            "\(AssessmentPhrasing.throughput(spot.sequentialBytesPerSecond)) sequential, "
            + "\(AssessmentPhrasing.throughput(spot.scatteredBytesPerSecond)) scattered, "
            + "over \(AssessmentPhrasing.bytesDecimal(spot.totalBytesRead)) read."
    }
}

/// Where the last report for each location lives between launches.
public protocol AssessmentLedger: Sendable {
    func record(_ report: VolumeAssessmentReport)
    func report(forLocationPath path: String) -> VolumeAssessmentReport?
    func forget(locationPath: String)
}

public struct UserDefaultsAssessmentLedger: AssessmentLedger {
    private final class DefaultsBox: @unchecked Sendable {
        let defaults: UserDefaults
        init(_ defaults: UserDefaults) { self.defaults = defaults }
    }

    private let box: DefaultsBox
    private let prefix: String

    public init(defaults: UserDefaults = .standard, prefix: String = "minirun.kit.assessment.") {
        self.box = DefaultsBox(defaults)
        self.prefix = prefix
    }

    public func record(_ report: VolumeAssessmentReport) {
        guard let data = try? MinirunKitJSON.encoder(pretty: false).encode(report) else { return }
        box.defaults.set(data, forKey: prefix + report.locationPath)
    }

    public func report(forLocationPath path: String) -> VolumeAssessmentReport? {
        guard let data = box.defaults.data(forKey: prefix + path) else { return nil }
        return try? MinirunKitJSON.decoder().decode(VolumeAssessmentReport.self, from: data)
    }

    public func forget(locationPath: String) {
        box.defaults.removeObject(forKey: prefix + locationPath)
    }
}

public final class InMemoryAssessmentLedger: AssessmentLedger, @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String: VolumeAssessmentReport] = [:]

    public init() {}

    public func record(_ report: VolumeAssessmentReport) {
        lock.lock()
        defer { lock.unlock() }
        storage[report.locationPath] = report
    }

    public func report(forLocationPath path: String) -> VolumeAssessmentReport? {
        lock.lock()
        defer { lock.unlock() }
        return storage[path]
    }

    public func forget(locationPath: String) {
        lock.lock()
        defer { lock.unlock() }
        storage.removeValue(forKey: locationPath)
    }
}
