import Foundation
import StorageCore

extension ModelLocationReport {
    /// One line per fact, for the on-device screen and for a log line.
    ///
    /// Sizes go through ``StorageCore/ByteSize/format(_:)`` so the phone and
    /// `minirun env` say GiB the same way; a report that reads differently in
    /// two places is a report someone will eventually mis-transcribe into an
    /// experiment note (spec §17.4).
    public var summaryLines: [String] {
        var lines: [String] = []
        lines.append("path: \(directoryPath)")

        if let name = storage.volumeName {
            var volume = "volume: \(name)"
            if let type = storage.filesystemType { volume += " (\(type))" }
            if storage.volumeIsRemovable == true { volume += ", removable" }
            lines.append(volume)
        }

        if let free = freeBytes {
            var space = "free: \(ByteSize.format(UInt64(max(0, free))))"
            if let total = storage.volumeTotalBytes {
                space += " of \(ByteSize.format(UInt64(max(0, total))))"
            }
            lines.append(space)
        } else {
            lines.append("free: unavailable")
        }

        switch layout {
        case nil:
            lines.append("artifact: none (no manifest.json here)")
        case .unrecognized:
            lines.append("artifact: manifest.json present but not understood")
        case .some(let kind):
            lines.append("artifact: \(kind.rawValue), \(files.count) files named")
            lines.append("on disk: \(ByteSize.format(presentBytes))")
            if isComplete {
                lines.append("status: complete")
            } else {
                let missing = missingFiles
                let head = missing.prefix(3).joined(separator: ", ")
                lines.append(
                    "status: \(missing.count) file(s) missing"
                        + (head.isEmpty ? "" : " — \(head)\(missing.count > 3 ? ", …" : "")"))
            }
            if let complete = builderReportsComplete {
                lines.append("builder flag: complete=\(complete)")
            }
        }

        if let problem = manifestProblem {
            lines.append("note: \(problem)")
        }
        return lines
    }
}
