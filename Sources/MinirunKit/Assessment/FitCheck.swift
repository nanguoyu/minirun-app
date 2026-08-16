import Foundation

/// Whether one model's bytes fit on one volume.
public enum FitVerdict: Sendable, Equatable, Codable {
    /// An artifact for this model is already here, and the bytes were counted
    /// from the directory rather than from the catalog.
    case alreadyHere(bytesOnDisk: UInt64)
    case fits(spareBytes: Int64)
    case short(byBytes: UInt64)
    /// The volume did not report its free space. Not "fine" — a copy that starts
    /// on an unprobeable volume finds out how much room there was by running out
    /// of it.
    case unknown(reason: String)
}

/// Model bytes plus a working margin against a volume's dependable free space.
///
/// The margin is not decoration. `volumeAvailableCapacity` is what the caller
/// gets to spend; a transfer that lands exactly on it leaves a volume with no
/// room for the filesystem's own bookkeeping, and an artifact that only fits
/// into purgeable space does not fit.
public struct FitCheck: Codable, Sendable, Equatable, Identifiable {
    public var id: String { model.rawValue }
    public let model: ModelID
    public let displayName: String
    /// What the model needs. The catalog's total, unless an artifact is here, in
    /// which case it is what the directory actually holds.
    public let requiredBytes: UInt64
    /// True when `requiredBytes` was counted from files rather than read from
    /// the catalog. Drives the measured/declared typography.
    public let requiredIsMeasured: Bool
    public let marginBytes: UInt64
    public let availableBytes: Int64?
    public let verdict: FitVerdict

    public init(
        model: ModelID, displayName: String, requiredBytes: UInt64, requiredIsMeasured: Bool,
        marginBytes: UInt64, availableBytes: Int64?, verdict: FitVerdict
    ) {
        self.model = model
        self.displayName = displayName
        self.requiredBytes = requiredBytes
        self.requiredIsMeasured = requiredIsMeasured
        self.marginBytes = marginBytes
        self.availableBytes = availableBytes
        self.verdict = verdict
    }

    /// The same 8 GiB the downloader keeps back
    /// (``DownloadOptions/freeSpaceHeadroomBytes``). One number, so a fit check
    /// on a settings screen cannot promise what a download then refuses.
    public static let defaultMarginBytes: UInt64 = 8 << 30

    public var fits: Bool {
        switch verdict {
        case .alreadyHere, .fits: return true
        case .short, .unknown: return false
        }
    }

    /// Bytes still needed, or nil when it fits.
    public var shortfallBytes: UInt64? {
        if case .short(let by) = verdict { return by }
        return nil
    }

    /// Decide one model against one volume.
    ///
    /// `bytesOnDisk` is what discovery counted for an artifact of this model
    /// *on this volume*, when there is one. Present-and-counted wins over the
    /// catalog's expectation — files win over metadata, which is the same rule
    /// discovery keeps.
    public static func check(
        model: ModelID, displayName: String, catalogBytes: UInt64, bytesOnDisk: UInt64?,
        availableBytes: Int64?, marginBytes: UInt64 = FitCheck.defaultMarginBytes
    ) -> FitCheck {
        if let bytesOnDisk {
            return FitCheck(
                model: model, displayName: displayName, requiredBytes: bytesOnDisk,
                requiredIsMeasured: true, marginBytes: marginBytes,
                availableBytes: availableBytes, verdict: .alreadyHere(bytesOnDisk: bytesOnDisk))
        }
        guard let availableBytes else {
            return FitCheck(
                model: model, displayName: displayName, requiredBytes: catalogBytes,
                requiredIsMeasured: false, marginBytes: marginBytes, availableBytes: nil,
                verdict: .unknown(
                    reason: "the volume reported no available capacity, so its free space is "
                        + "not known"))
        }
        guard availableBytes >= 0 else {
            return FitCheck(
                model: model, displayName: displayName, requiredBytes: catalogBytes,
                requiredIsMeasured: false, marginBytes: marginBytes,
                availableBytes: availableBytes,
                verdict: .unknown(
                    reason: "the volume reported a negative available capacity"))
        }
        let (need, overflow) = catalogBytes.addingReportingOverflow(marginBytes)
        guard !overflow else {
            return FitCheck(
                model: model, displayName: displayName, requiredBytes: catalogBytes,
                requiredIsMeasured: false, marginBytes: marginBytes,
                availableBytes: availableBytes,
                verdict: .unknown(
                    reason: "the model bytes plus the required margin overflow UInt64"))
        }
        let available = UInt64(availableBytes)
        if available >= need {
            return FitCheck(
                model: model, displayName: displayName, requiredBytes: catalogBytes,
                requiredIsMeasured: false, marginBytes: marginBytes,
                availableBytes: availableBytes,
                // `availableBytes` is at most Int64.max, so a successful
                // unsigned comparison proves the spare is Int64-representable.
                verdict: .fits(spareBytes: Int64(available - need)))
        }
        return FitCheck(
            model: model, displayName: displayName, requiredBytes: catalogBytes,
            requiredIsMeasured: false, marginBytes: marginBytes, availableBytes: availableBytes,
            verdict: .short(byBytes: need - available))
    }
}
