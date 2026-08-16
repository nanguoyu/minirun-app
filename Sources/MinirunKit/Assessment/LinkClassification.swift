import Foundation

#if os(macOS)
    import IOKit
#endif

/// What kind of link a measured rate is consistent with.
///
/// Consistent with, and not "is": a rate is evidence about a link, and the two
/// are not the same statement. The middle case exists for that reason and is
/// stated out loud rather than rounded to whichever neighbour is closer.
public enum LinkClass: String, Codable, Sendable, CaseIterable {
    /// Slower than a USB3 Gen2 payload ceiling. A hub, a Gen1 port, a spinning
    /// disk, or a network mount.
    case belowUSB3Gen2
    /// ~0.95 GB/s, the USB3 Gen2 (10 Gb/s) payload ceiling.
    case usb3Gen2
    /// Between the two ceilings, and therefore not attributable to either.
    case ambiguous
    /// ≥ 2.3 GB/s: a USB4 tunnel or Thunderbolt.
    case usb4OrThunderbolt
    /// Faster than any external link this project has measured — an internal
    /// Apple SSD.
    case internalNVMe

    public var label: String {
        switch self {
        case .belowUSB3Gen2: return "slower than USB3 Gen2"
        case .usb3Gen2: return "USB3 Gen2 class"
        case .ambiguous: return "between the known ceilings"
        case .usb4OrThunderbolt: return "USB4 / Thunderbolt class"
        case .internalNVMe: return "internal NVMe class"
        }
    }

    /// A link this product has a faster, measured alternative for.
    public var isDegraded: Bool {
        self == .usb3Gen2 || self == .belowUSB3Gen2
    }
}

/// Where a classification's confidence comes from.
///
/// It is a field and not a footnote because the two sources answer different
/// questions and neither is complete. A measurement gives a rate and cannot name
/// a bus. The IO registry names an interconnect family and cannot say which
/// generation negotiated. Saying which one spoke is the difference between a
/// finding and a guess.
public enum LinkProvenance: String, Codable, Sendable {
    /// A rate was measured; no hardware fact was available.
    case measurement
    /// The IO registry answered; nothing was measured.
    case hardware
    /// Both, and they are reported side by side rather than merged.
    case both
    /// Neither. The classification is `ambiguous` and says nothing else.
    case undetermined
}

/// What the IO registry says about the volume's interconnect.
///
/// Deliberately thin. `Physical Interconnect` is a family — `USB`,
/// `PCI-Express`, `Thunderbolt` — and never a generation or a rate, so nothing
/// here can be turned into a speed. That is the honest limit of what the
/// registry knows, and inventing "USB4" from `USB` is exactly the fabrication
/// this type exists to make impossible.
public struct HardwareLinkFacts: Codable, Sendable, Equatable {
    public let interconnect: String
    public let location: String?
    public let bsdName: String?

    public init(interconnect: String, location: String?, bsdName: String?) {
        self.interconnect = interconnect
        self.location = location
        self.bsdName = bsdName
    }

    public var isInternal: Bool { location?.caseInsensitiveCompare("Internal") == .orderedSame }
}

/// Best-effort hardware corroboration.
public protocol HardwareLinkProbing: Sendable {
    /// Nil whenever the answer is not known — including every case where the
    /// platform, the sandbox or the filesystem declines to say. Never a guess.
    func facts(forMountPath path: String) -> HardwareLinkFacts?
}

/// The probe for platforms and builds that cannot ask. Always nil, which is the
/// truthful answer.
public struct UndeterminedHardwareProbe: HardwareLinkProbing {
    public init() {}
    public func facts(forMountPath path: String) -> HardwareLinkFacts? { nil }
}

/// Reads `Protocol Characteristics` off the IO registry entry behind a mount.
///
/// macOS only, and nil everywhere else. On iOS there is nothing to ask and no
/// external volume to ask about, so the classification there rests on the
/// measurement alone — which is the arrangement this module was designed for
/// anyway.
public struct IORegistryHardwareProbe: HardwareLinkProbing {
    public init() {}

    public func facts(forMountPath path: String) -> HardwareLinkFacts? {
        #if os(macOS)
            var buffer = statfs()
            guard statfs(path, &buffer) == 0 else { return nil }
            let device = withUnsafeBytes(of: &buffer.f_mntfromname) { raw -> String in
                String(cString: raw.baseAddress!.assumingMemoryBound(to: CChar.self))
            }
            let prefix = "/dev/"
            guard device.hasPrefix(prefix) else { return nil }
            let bsdName = String(device.dropFirst(prefix.count))

            guard let matching = IOBSDNameMatching(kIOMainPortDefault, 0, bsdName) else {
                return nil
            }
            let service = IOServiceGetMatchingService(kIOMainPortDefault, matching)
            guard service != 0 else { return nil }
            defer { IOObjectRelease(service) }

            guard
                let property = IORegistryEntrySearchCFProperty(
                    service, kIOServicePlane, "Protocol Characteristics" as CFString,
                    kCFAllocatorDefault,
                    IOOptionBits(kIORegistryIterateRecursively | kIORegistryIterateParents)),
                let characteristics = property as? [String: Any],
                let interconnect = characteristics["Physical Interconnect"] as? String
            else { return nil }

            return HardwareLinkFacts(
                interconnect: interconnect,
                location: characteristics["Physical Interconnect Location"] as? String,
                bsdName: bsdName)
        #else
            return nil
        #endif
    }
}

/// One row of the ceiling table: a class, the rate this project measured for it,
/// and where that number is written down.
public struct LinkReference: Sendable, Equatable, Codable {
    public let linkClass: LinkClass
    public let bytesPerSecond: Double
    public let name: String
    /// The record the number comes from. Every entry has one; a row without a
    /// record does not belong in the table.
    public let source: String

    public init(linkClass: LinkClass, bytesPerSecond: Double, name: String, source: String) {
        self.linkClass = linkClass
        self.bytesPerSecond = bytesPerSecond
        self.name = name
        self.source = source
    }
}

/// A measured rate, what it is consistent with, and how confident that is.
public struct LinkClassification: Codable, Sendable, Equatable {
    public let linkClass: LinkClass
    public let measuredBytesPerSecond: Double?
    public let provenance: LinkProvenance
    public let hardware: HardwareLinkFacts?
    /// How much faster the next tier up has measured on this project's own
    /// bench, and what it is called. Nil unless the table has an entry —
    /// a multiple invented for a link nobody measured would be the same mistake
    /// as a projection without a calibration.
    public let fasterTierMultiple: Double?
    public let fasterTierName: String?

    public init(
        linkClass: LinkClass, measuredBytesPerSecond: Double?, provenance: LinkProvenance,
        hardware: HardwareLinkFacts?, fasterTierMultiple: Double?, fasterTierName: String?
    ) {
        self.linkClass = linkClass
        self.measuredBytesPerSecond = measuredBytesPerSecond
        self.provenance = provenance
        self.hardware = hardware
        self.fasterTierMultiple = fasterTierMultiple
        self.fasterTierName = fasterTierName
    }

    public var isDegraded: Bool { linkClass.isDegraded && measuredBytesPerSecond != nil }

    /// The sentence a report card shows. Never asserts a bus name the evidence
    /// does not carry.
    public var sentence: String {
        switch provenance {
        case .undetermined:
            return
                "Nothing was measured and the system did not say what this volume is attached to, "
                + "so its link is undetermined."
        case .hardware:
            guard let hardware else { return "The link is undetermined." }
            return
                "The system reports a \(hardware.interconnect) interconnect"
                + (hardware.location.map { " (\($0.lowercased()))" } ?? "")
                + ", which names a family and not a speed. Nothing has been measured here, so no "
                + "rate is claimed."
        case .measurement, .both:
            let rate = measuredBytesPerSecond ?? 0
            var text =
                String(format: "%.2f GB/s measured", rate / 1e9)
                + " — \(linkClass.label)."
            if provenance == .both, let hardware {
                text +=
                    " The system reports a \(hardware.interconnect) interconnect"
                    + (hardware.location.map { " (\($0.lowercased()))" } ?? "") + "."
            }
            return text
        }
    }
}

/// The ceiling table, and the rule that reads it.
public enum LinkClassifier {

    /// Every rate here was measured by this project, on this project's rigs, and
    /// every row names the record it came from. Nothing is a datasheet figure.
    ///
    /// | class | measured | record |
    /// | --- | --- | --- |
    /// | USB3 Gen2 | 0.95 GB/s | `2026-08-11-readahead-mac.md` |
    /// | USB4 tunnel | 2.84 GB/s | `2026-08-11-readahead-mac.md` |
    /// | internal Apple SSD | 6.9 GB/s | `2026-08-08-mac-internal-storage-curves.md` |
    /// | iPhone USB-C port | 0.923 GB/s | `2026-08-10-m10-k3-iphone.md` |
    ///
    /// The iPhone row is in the table as a *known ceiling* and not as a class:
    /// it lands inside the USB3 Gen2 band, which is the truthful thing for it to
    /// do, and the phone's port is not something a Mac's classification should
    /// ever be compared against.
    public static let table: [LinkReference] = [
        LinkReference(
            linkClass: .usb3Gen2, bytesPerSecond: 0.95e9, name: "USB3 Gen2, 10 Gb/s",
            source: "docs/experiments/2026-08-11-readahead-mac.md"),
        LinkReference(
            linkClass: .usb4OrThunderbolt, bytesPerSecond: 2.84e9, name: "USB4 tunnel, 40 Gb/s",
            source: "docs/experiments/2026-08-11-readahead-mac.md"),
        LinkReference(
            linkClass: .internalNVMe, bytesPerSecond: 6.9e9, name: "internal Apple SSD",
            source: "docs/experiments/2026-08-08-mac-internal-storage-curves.md"),
    ]

    /// The iPhone's own USB-C port, measured in the pager's request shape. Used
    /// in copy on iOS, never as a Mac's comparison.
    public static let iPhonePortCeiling = LinkReference(
        linkClass: .usb3Gen2, bytesPerSecond: 0.923e9, name: "iPhone USB-C port",
        source: "docs/experiments/2026-08-10-m10-k3-iphone.md")

    /// Upper edge of the USB3 Gen2 band. Above this a rate is no longer
    /// attributable to a 10 Gb/s link.
    public static let usb3Gen2CeilingBytesPerSecond = 1.15e9
    /// Below this, the link is slower than a USB3 Gen2 payload ceiling by a
    /// margin no thermal drift explains.
    public static let usb3Gen2FloorBytesPerSecond = 0.60e9
    /// At or above this, the rate is only reachable over USB4/Thunderbolt.
    public static let usb4FloorBytesPerSecond = 2.30e9
    /// At or above this, no external link this project has measured reaches it.
    public static let internalFloorBytesPerSecond = 4.00e9

    public static func linkClass(forBytesPerSecond rate: Double) -> LinkClass {
        if rate >= internalFloorBytesPerSecond { return .internalNVMe }
        if rate >= usb4FloorBytesPerSecond { return .usb4OrThunderbolt }
        if rate > usb3Gen2CeilingBytesPerSecond { return .ambiguous }
        if rate >= usb3Gen2FloorBytesPerSecond { return .usb3Gen2 }
        return .belowUSB3Gen2
    }

    /// Classify a measured rate, corroborated by hardware where the platform
    /// will say.
    ///
    /// A nil or non-positive rate never produces a class: the result is
    /// `ambiguous`, with whatever the hardware said and nothing more. This is
    /// the same rule the projection layer keeps — no measurement, no claim.
    public static func classify(
        bytesPerSecond: Double?, hardware: HardwareLinkFacts? = nil
    ) -> LinkClassification {
        guard let rate = bytesPerSecond, rate > 0, rate.isFinite else {
            return LinkClassification(
                linkClass: .ambiguous, measuredBytesPerSecond: nil,
                provenance: hardware == nil ? .undetermined : .hardware,
                hardware: hardware, fasterTierMultiple: nil, fasterTierName: nil)
        }
        let classified = linkClass(forBytesPerSecond: rate)
        var multiple: Double?
        var name: String?
        if classified.isDegraded,
            let faster = table.first(where: { $0.linkClass == .usb4OrThunderbolt })
        {
            multiple = faster.bytesPerSecond / rate
            name = faster.name
        }
        return LinkClassification(
            linkClass: classified, measuredBytesPerSecond: rate,
            provenance: hardware == nil ? .measurement : .both,
            hardware: hardware, fasterTierMultiple: multiple, fasterTierName: name)
    }
}
