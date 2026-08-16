import CoreFoundation
import Foundation

/// One file in a repository, as the repository itself describes it.
public struct RepoFile: Codable, Sendable, Equatable, Hashable {
    /// Repo-relative, `/`-separated, exactly as the tree spells it.
    public let path: String
    public let sizeBytes: UInt64
    public let digest: FileDigest
    /// Per the classification rule in ``RepoFileClassification``.
    public let isPayload: Bool

    public init(path: String, sizeBytes: UInt64, digest: FileDigest, isPayload: Bool) {
        self.path = path
        self.sizeBytes = sizeBytes
        self.digest = digest
        self.isPayload = isPayload
    }

    public var lastComponent: String {
        path.split(separator: "/").last.map(String.init) ?? path
    }
}

/// Which files a run reads, and which merely describe them.
///
/// The rule is one sentence and it is exact:
///
/// > A file is **metadata** iff it sits at the repository root, or its last
/// > path component is `manifest.json`. Everything else is **payload**.
///
/// It is stated as a rule rather than a heuristic because it is checkable:
/// applied to the three published `*-minirun` repositories it reproduces each
/// repository's own `index.json` counts and byte totals exactly — K3 372
/// payload files, V4-Flash 577, H3 778, with the byte sums matching
/// `index.bytes` to the byte. `MinirunKitTests` asserts that against recorded
/// trees, so a future repo that breaks the rule breaks a test rather than
/// quietly mis-sizing a 1.5 TB download.
public enum RepoFileClassification {
    public static func isPayload(path: String) -> Bool {
        let components = path.split(separator: "/")
        if components.count <= 1 { return false }  // repository root
        if components.last == "manifest.json" { return false }
        return true
    }
}

/// What a repository's own `index.json` claims about itself.
///
/// A claim and not a fact. Every published `index.json` in these repos counts
/// only payload — and none of the three carries a complete per-file digest
/// table, whatever the shape of its `sha256` fields. The complete table is the
/// tree.
public struct IndexClaim: Codable, Sendable, Equatable {
    public let declaredFileCount: Int
    public let declaredBytes: UInt64
    public let sourceRepo: String?
    public let sourceRevision: String?

    public init(
        declaredFileCount: Int, declaredBytes: UInt64,
        sourceRepo: String?, sourceRevision: String?
    ) {
        self.declaredFileCount = declaredFileCount
        self.declaredBytes = declaredBytes
        self.sourceRepo = sourceRepo
        self.sourceRevision = sourceRevision
    }

    /// Reads the four fields this package uses out of an `index.json` body,
    /// tolerating everything else.
    ///
    /// Two shapes exist across the three published repos and there is no schema
    /// to appeal to, so both are read:
    ///
    /// - K3 states the totals at the top level: `files` and `bytes`.
    /// - V4-Flash and H3 state `total_bytes` at the top level and break the
    ///   file count down over a `units` array, each with its own `files` and
    ///   `bytes`.
    ///
    /// Recorded on 2026-08-11, the second form sums to 577 files /
    /// 166,893,192,184 bytes for V4-Flash and 778 / 63,969,279,450 for H3 —
    /// which is exactly what the payload rule finds in their trees.
    /// `source_repo` and `source_revision` are read only where they exist at
    /// the top level; H3 carries a `sources` map of several upstreams instead,
    /// and collapsing that to one string would be inventing a provenance.
    public static func parse(_ data: Data) -> IndexClaim? {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        func number(_ keys: [String]) -> UInt64? {
            for key in keys {
                guard let raw = root[key] else { continue }
                guard let value = Self.unsignedInteger(raw) else { return nil }
                return value
            }
            return nil
        }
        let units = root["units"] as? [[String: Any]]
        let unitFiles: UInt64?
        if let units {
            var total: UInt64 = 0
            var valid = true
            for unit in units {
                guard let raw = unit["files"] else { continue }
                guard let files = Self.unsignedInteger(raw) else {
                    valid = false
                    break
                }
                let (sum, overflow) = total.addingReportingOverflow(files)
                guard !overflow else {
                    valid = false
                    break
                }
                total = sum
            }
            unitFiles = valid ? total : nil
        } else {
            unitFiles = nil
        }

        let hasDirectFileCount = root["files"] != nil || root["file_count"] != nil
        let directFileCount = number(["files", "file_count"])
        guard !hasDirectFileCount || directFileCount != nil else { return nil }
        guard let files = directFileCount ?? unitFiles,
            files <= UInt64(Int.max),
            let bytes = number(["bytes", "total_bytes"])
        else { return nil }

        return IndexClaim(
            declaredFileCount: Int(files), declaredBytes: bytes,
            sourceRepo: root["source_repo"] as? String,
            sourceRevision: root["source_revision"] as? String)
    }

    private static func unsignedInteger(_ value: Any) -> UInt64? {
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        return UInt64(number.stringValue)
    }
}

/// What the tree says, next to what `index.json` says, and the difference.
///
/// Spec §5.10 made concrete: where files and metadata disagree, the files win,
/// and the disagreement is *recorded* rather than resolved by believing the
/// metadata. Nothing in this type changes a plan — it only makes the
/// discrepancy visible in the plan, in the job state, and in whatever the
/// operator is shown.
public struct IndexReconciliation: Codable, Sendable, Equatable {
    public let treeFileCount: Int
    public let treeBytes: UInt64
    public let payloadFileCount: Int
    public let payloadBytes: UInt64
    /// `declared - payload`. Positive means the index promised files the tree
    /// does not have.
    public let fileCountDelta: Int
    public let byteDelta: Int64
    public let note: String?

    public var agrees: Bool { fileCountDelta == 0 && byteDelta == 0 }

    public init(
        treeFileCount: Int, treeBytes: UInt64, payloadFileCount: Int, payloadBytes: UInt64,
        fileCountDelta: Int, byteDelta: Int64, note: String?
    ) {
        self.treeFileCount = treeFileCount
        self.treeBytes = treeBytes
        self.payloadFileCount = payloadFileCount
        self.payloadBytes = payloadBytes
        self.fileCountDelta = fileCountDelta
        self.byteDelta = byteDelta
        self.note = note
    }

    public static func between(files: [RepoFile], claim: IndexClaim?) -> IndexReconciliation {
        let payload = files.filter(\.isPayload)
        let payloadBytes = Self.saturatingByteSum(payload)
        let treeBytes = Self.saturatingByteSum(files)
        guard let claim else {
            return IndexReconciliation(
                treeFileCount: files.count, treeBytes: treeBytes,
                payloadFileCount: payload.count, payloadBytes: payloadBytes,
                fileCountDelta: 0, byteDelta: 0,
                note: "no index.json was published at the repository root; the tree is the only record")
        }
        let fileDelta = Self.saturatingDifference(claim.declaredFileCount, payload.count)
        let byteDelta = Self.saturatingByteDifference(claim.declaredBytes, payloadBytes)
        let note: String? =
            (fileDelta == 0 && byteDelta == 0)
            ? nil
            : "index.json declares \(claim.declaredFileCount) payload files totalling \(claim.declaredBytes) bytes; the tree holds \(payload.count) totalling \(payloadBytes). The tree is what will be fetched."
        return IndexReconciliation(
            treeFileCount: files.count, treeBytes: treeBytes,
            payloadFileCount: payload.count, payloadBytes: payloadBytes,
            fileCountDelta: fileDelta, byteDelta: byteDelta, note: note)
    }

    private static func saturatingByteSum(_ files: [RepoFile]) -> UInt64 {
        var total: UInt64 = 0
        for file in files {
            let (sum, overflow) = total.addingReportingOverflow(file.sizeBytes)
            if overflow { return .max }
            total = sum
        }
        return total
    }

    private static func saturatingDifference(_ lhs: Int, _ rhs: Int) -> Int {
        let (difference, overflow) = lhs.subtractingReportingOverflow(rhs)
        guard !overflow else { return rhs >= 0 ? .min : .max }
        return difference
    }

    private static func saturatingByteDifference(_ lhs: UInt64, _ rhs: UInt64) -> Int64 {
        if lhs >= rhs {
            let difference = lhs - rhs
            return difference > UInt64(Int64.max) ? .max : Int64(difference)
        }
        let difference = rhs - lhs
        return difference > UInt64(Int64.max) ? .min : -Int64(difference)
    }
}

/// Every file that will be fetched, at one pinned commit.
public struct DownloadPlan: Codable, Sendable, Equatable {
    public let model: ModelID
    /// Pinned. Every URL a job builds carries this revision, and every response
    /// is checked against it.
    public let repo: HuggingFaceRepoRef
    public let files: [RepoFile]
    public let index: IndexClaim?
    public let reconciliation: IndexReconciliation

    public init(
        model: ModelID, repo: HuggingFaceRepoRef, files: [RepoFile],
        index: IndexClaim?, reconciliation: IndexReconciliation
    ) {
        self.model = model
        self.repo = repo
        self.files = files
        self.index = index
        self.reconciliation = reconciliation
    }

    public var payloadBytes: UInt64 {
        Self.checkedByteSum(files.lazy.filter(\.isPayload)) ?? .max
    }
    public var metadataBytes: UInt64 {
        Self.checkedByteSum(files.lazy.filter { !$0.isPayload }) ?? .max
    }
    /// Saturates for an externally constructed invalid plan. The download
    /// manager rejects that plan with a named error; this accessor remains
    /// total so merely inspecting untrusted decoded state cannot crash.
    public var totalBytes: UInt64 { Self.checkedByteSum(files) ?? .max }
    public var largestFileBytes: UInt64 { files.map(\.sizeBytes).max() ?? 0 }

    /// `payload + metadata == total`, which holds by construction and is
    /// asserted anyway: the two accessors filter on a flag, and a flag that
    /// ever becomes a tri-state would break the identity silently.
    public var bucketsAccountForEveryByte: Bool {
        guard let payload = Self.checkedByteSum(files.lazy.filter(\.isPayload)),
            let metadata = Self.checkedByteSum(files.lazy.filter { !$0.isPayload }),
            let total = Self.checkedByteSum(files)
        else { return false }
        let (bucketTotal, overflow) = payload.addingReportingOverflow(metadata)
        return !overflow && bucketTotal == total
    }

    public func file(at path: String) -> RepoFile? {
        files.first { $0.path == path }
    }

    private static func checkedByteSum<S: Sequence>(_ files: S) -> UInt64?
    where S.Element == RepoFile {
        var total: UInt64 = 0
        for file in files {
            let (sum, overflow) = total.addingReportingOverflow(file.sizeBytes)
            guard !overflow else { return nil }
            total = sum
        }
        return total
    }
}
