import Darwin
import Foundation

/// Errors specific to the engram row-page container.
public enum EngramContainerError: Error, CustomStringConvertible, LocalizedError {
    case badMagic(found: String)
    case unsupportedVersion(UInt32)
    case malformedHeader(String)
    case geometryUnsupported(String)
    case truncated(path: String, expectedBytes: UInt64, actualBytes: UInt64)
    case rowOutOfRange(row: UInt64, rowBase: UInt64, rows: UInt64)
    case pageOutOfRange(page: UInt64, pageCount: UInt64)

    public var description: String {
        switch self {
        case .badMagic(let found):
            return "not a minirun engram part: magic was '\(found)'"
        case .unsupportedVersion(let version):
            return
                "engram part version \(version) is not supported (this build reads "
                + "\(EngramRowPageContainer.readableVersions.map(String.init).joined(separator: " and ")))"
        case .malformedHeader(let detail):
            return "engram part header is malformed: \(detail)"
        case .geometryUnsupported(let detail):
            return "engram part geometry is not supported: \(detail)"
        case .truncated(let path, let expected, let actual):
            return
                "engram part '\(path)' is truncated: header describes \(expected) bytes, "
                + "file is \(actual)"
        case .rowOutOfRange(let row, let base, let rows):
            return "row \(row) is not in this part [\(base), \(base + rows))"
        case .pageOutOfRange(let page, let count):
            return "page \(page) requested from a part of \(count) pages"
        }
    }

    public var errorDescription: String? { description }
}

/// A table of hundreds of millions of quantized rows, read one row at a time.
///
/// ## Why this is not a tile container
///
/// `QuantizedTileContainer` describes matrices a kernel multiplies by: one
/// scale grid per tile, and a per-tile digest table in a 16 KiB header. The
/// DeepSeek V4.1 `engram.embed` tables are neither. They are 384 million rows
/// of 256 FP8 values with eight E8M0 exponents each, read at *random* — 24 rows
/// per token per Engram layer — and a per-tile digest table for 25.6 million
/// tiles does not fit in any header.
///
/// The checkpoint's own layout is also wrong for that access pattern: the
/// values and their scales are two separate runs 98 GB apart, so reading one
/// row verbatim is two reads a page apart — 96 IOPS per token across the two
/// layers instead of 48.
///
/// ## The layout
///
/// A **page** is 4096 bytes and holds whole rows only:
///
/// ```text
/// [ row 0 values | ... | row R-1 values ]   R × rowValueBytes
/// [ row 0 scales | ... | row R-1 scales ]   R × rowScaleBytes
/// [ zero padding to pageBytes ]
/// ```
///
/// `rowsPerPage` is `pageBytes / (rowValueBytes + rowScaleBytes)` — 15 for the
/// published tables — which is the most a 264-byte record fits in 4096 bytes
/// without one crossing the boundary. A page never straddles a 4 KiB boundary
/// and a row never straddles a page, so **any row's values and its own scales
/// are both inside one aligned 4 KiB read**. The cost is 136 bytes a page,
/// 3.32%.
///
/// A **part** is one file with a 16 KiB header and up to two million pages.
/// 16384 and 4096 are both multiples of 4096, so page *p* begins at
/// `firstPageOffset + p * pageBytes` and stays 4 KiB-aligned.
///
/// ## Header, little-endian, 16384 bytes with everything after byte 88 zero
///
/// ```text
///  0  char[8]  magic "MNRNENG1"
///  8  uint32   version           = 1
/// 12  uint32   headerBytes       = 16384
/// 16  uint64   tableRows           rows in the whole table, across all parts
/// 24  uint64   rowBase             first table row in this part
/// 32  uint64   partRows            rows in this part
/// 40  uint32   rowValueBytes
/// 44  uint32   rowScaleBytes
/// 48  uint32   pageBytes
/// 52  uint32   rowsPerPage
/// 56  uint64   pageCount           ceil(partRows / rowsPerPage)
/// 64  uint64   firstPageOffset   = 16384
/// 72  uint32   valueDType          0 = fp8-e4m3
/// 76  uint32   scaleDType          0 = uint8-e8m0
/// 80  uint32   scaleGroupSize      32
/// 84  uint32   partIndex
/// ```
///
/// Every derivable field is **re-derived and compared** on open, so a header
/// that contradicts its own geometry is rejected rather than followed. That is
/// the same contract the writer states — `Tools/v41_flash/engram_container.py`
/// — and `EngramRowPageContainerTests` reads a fixture that writer produced, so
/// the two agree byte for byte rather than by description.
public enum EngramRowPageContainer {
    public static let magic = "MNRNENG1"
    public static let version: UInt32 = 1
    public static let readableVersions: [UInt32] = [1]
    /// One aligned unit, so page 0 starts aligned.
    public static let headerBytes = 16384
    /// The page size the published tables use. Carried in the header rather
    /// than assumed; this is only the writer's choice, restated.
    public static let defaultPageBytes = 4096
    /// Bytes of the fixed part of the header. Everything after is zero.
    static let headerFieldBytes = 88

    /// How a row's values are encoded.
    public enum ValueDType: UInt32, Sendable, Equatable, Codable, CaseIterable {
        case fp8E4M3 = 0

        public var name: String {
            switch self {
            case .fp8E4M3: return "fp8-e4m3"
            }
        }
    }

    /// How a row's scales are encoded.
    public enum ScaleDType: UInt32, Sendable, Equatable, Codable, CaseIterable {
        case uint8E8M0 = 0

        public var name: String {
            switch self {
            case .uint8E8M0: return "uint8-e8m0"
            }
        }
    }

    /// One part file's geometry. Every field is in that file's header.
    public struct Layout: Sendable, Equatable, Codable {
        public let tableRows: UInt64
        public let rowBase: UInt64
        public let partRows: UInt64
        public let rowValueBytes: Int
        public let rowScaleBytes: Int
        public let pageBytes: Int
        public let rowsPerPage: Int
        public let pageCount: UInt64
        public let firstPageOffset: UInt64
        public let valueDType: ValueDType
        public let scaleDType: ScaleDType
        public let scaleGroupSize: Int
        public let partIndex: Int

        /// Where a page's scale run begins, relative to the page.
        public var scaleOffsetInPage: Int { rowsPerPage * rowValueBytes }
        /// Zero bytes at the end of every page.
        public var padBytesPerPage: Int {
            pageBytes - rowsPerPage * (rowValueBytes + rowScaleBytes)
        }
        public var payloadBytes: UInt64 { pageCount * UInt64(pageBytes) }
        public var totalBytes: UInt64 { firstPageOffset + payloadBytes }
        /// One past the last table row this part holds.
        public var rowEnd: UInt64 { rowBase + partRows }

        public func contains(row: UInt64) -> Bool {
            row >= rowBase && row < rowEnd
        }

        public func pageOffset(_ page: UInt64) throws -> UInt64 {
            guard page < pageCount else {
                throw EngramContainerError.pageOutOfRange(
                    page: page, pageCount: pageCount)
            }
            return firstPageOffset + page * UInt64(pageBytes)
        }

        /// Where one table row lives: the aligned page to read, and the two
        /// offsets inside it. The page is one read — which is the entire point
        /// of the format.
        public func locate(row: UInt64) throws -> RowLocation {
            guard contains(row: row) else {
                throw EngramContainerError.rowOutOfRange(
                    row: row, rowBase: rowBase, rows: partRows)
            }
            let local = row - rowBase
            let page = local / UInt64(rowsPerPage)
            let within = Int(local % UInt64(rowsPerPage))
            return RowLocation(
                page: page,
                pageOffset: try pageOffset(page),
                valueOffsetInPage: within * rowValueBytes,
                scaleOffsetInPage: scaleOffsetInPage + within * rowScaleBytes)
        }
    }

    /// The address arithmetic for one row, resolved.
    public struct RowLocation: Sendable, Equatable {
        public let page: UInt64
        /// 4 KiB-aligned file offset of the page holding the row.
        public let pageOffset: UInt64
        public let valueOffsetInPage: Int
        public let scaleOffsetInPage: Int
    }

    /// One row's values and its own scales, from one aligned page read.
    public struct Row: Sendable, Equatable {
        public let row: UInt64
        public let values: [UInt8]
        public let scales: [UInt8]

        public init(row: UInt64, values: [UInt8], scales: [UInt8]) {
            self.row = row
            self.values = values
            self.scales = scales
        }
    }

    // MARK: - Opening

    /// Read and validate a part header, and check the file is exactly as long
    /// as that header describes.
    public static func open(path: String) throws -> Layout {
        let descriptor = Darwin.open(path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw StorageCoreError.posix(operation: "open", path: path, code: errno)
        }
        defer { close(descriptor) }
        return try open(fileDescriptor: descriptor, path: path)
    }

    /// Read a part through an already-authorized descriptor, which the caller
    /// keeps owning.
    public static func open(fileDescriptor descriptor: Int32, path: String) throws -> Layout {
        var header = [UInt8](repeating: 0, count: headerBytes)
        let read = header.withUnsafeMutableBytes { buffer -> Int in
            pread(descriptor, buffer.baseAddress, headerBytes, 0)
        }
        guard read == headerBytes else {
            throw EngramContainerError.truncated(
                path: path, expectedBytes: UInt64(headerBytes),
                actualBytes: UInt64(max(read, 0)))
        }
        let layout = try decodeHeader(header)

        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw StorageCoreError.posix(operation: "fstat", path: path, code: errno)
        }
        let actual = UInt64(max(status.st_size, 0))
        // Equality, not `>=`: a part is a whole number of pages after its
        // header by construction, so a longer file is as much a disagreement as
        // a shorter one.
        guard actual == layout.totalBytes else {
            throw EngramContainerError.truncated(
                path: path, expectedBytes: layout.totalBytes, actualBytes: actual)
        }
        return layout
    }

    static func decodeHeader(_ header: [UInt8]) throws -> Layout {
        guard header.count >= headerBytes else {
            throw EngramContainerError.malformedHeader(
                "header is \(header.count) bytes; the format reserves \(headerBytes)")
        }
        let found = String(decoding: header[0..<8], as: UTF8.self)
        guard found == magic else {
            throw EngramContainerError.badMagic(found: found)
        }
        var cursor = 8
        func take<T: FixedWidthInteger>(_ type: T.Type) -> T {
            let value = header.withUnsafeBytes {
                T(littleEndian: $0.loadUnaligned(fromByteOffset: cursor, as: T.self))
            }
            cursor += MemoryLayout<T>.size
            return value
        }

        let fileVersion = take(UInt32.self)
        guard readableVersions.contains(fileVersion) else {
            throw EngramContainerError.unsupportedVersion(fileVersion)
        }
        let declaredHeaderBytes = take(UInt32.self)
        guard declaredHeaderBytes == UInt32(headerBytes) else {
            throw EngramContainerError.malformedHeader(
                "headerBytes \(declaredHeaderBytes) is not \(headerBytes)")
        }
        let tableRows = take(UInt64.self)
        let rowBase = take(UInt64.self)
        let partRows = take(UInt64.self)
        let rowValueBytes = Int(take(UInt32.self))
        let rowScaleBytes = Int(take(UInt32.self))
        let pageBytes = Int(take(UInt32.self))
        let rowsPerPage = Int(take(UInt32.self))
        let pageCount = take(UInt64.self)
        let firstPageOffset = take(UInt64.self)
        let rawValueDType = take(UInt32.self)
        let rawScaleDType = take(UInt32.self)
        let scaleGroupSize = Int(take(UInt32.self))
        let partIndex = Int(take(UInt32.self))
        precondition(cursor == headerFieldBytes)

        // Reserved bytes are checked, not skipped: a future field written by a
        // producer this build does not understand must not be read as padding.
        guard header[headerFieldBytes..<headerBytes].allSatisfy({ $0 == 0 }) else {
            throw EngramContainerError.malformedHeader(
                "reserved header bytes are not zero")
        }
        guard let valueDType = ValueDType(rawValue: rawValueDType) else {
            throw EngramContainerError.geometryUnsupported(
                "value dtype \(rawValueDType) is unknown")
        }
        guard let scaleDType = ScaleDType(rawValue: rawScaleDType) else {
            throw EngramContainerError.geometryUnsupported(
                "scale dtype \(rawScaleDType) is unknown")
        }

        guard tableRows > 0, partRows > 0 else {
            throw EngramContainerError.geometryUnsupported(
                "a table and a part must both have rows")
        }
        let end = rowBase.addingReportingOverflow(partRows)
        guard !end.overflow, end.partialValue <= tableRows else {
            throw EngramContainerError.geometryUnsupported(
                "part rows [\(rowBase), \(rowBase)+\(partRows)) fall outside a table "
                    + "of \(tableRows)")
        }
        guard rowValueBytes > 0, rowScaleBytes > 0 else {
            throw EngramContainerError.geometryUnsupported(
                "a row needs both value and scale bytes")
        }
        guard pageBytes > 0, pageBytes % 4096 == 0 else {
            throw EngramContainerError.geometryUnsupported(
                "pageBytes \(pageBytes) must be a positive multiple of 4096 so a page "
                    + "is one aligned read")
        }
        guard headerBytes % pageBytes == 0 else {
            throw EngramContainerError.geometryUnsupported(
                "a \(headerBytes)-byte header is not a whole number of \(pageBytes)-byte "
                    + "pages, so no page in the file is aligned")
        }
        let densest = pageBytes / (rowValueBytes + rowScaleBytes)
        guard densest > 0 else {
            throw EngramContainerError.geometryUnsupported(
                "a \(rowValueBytes + rowScaleBytes)-byte row does not fit in a "
                    + "\(pageBytes)-byte page, so no single aligned read can return it")
        }
        guard rowsPerPage == densest else {
            throw EngramContainerError.geometryUnsupported(
                "rowsPerPage \(rowsPerPage) is not the densest packing \(densest) for a "
                    + "\(rowValueBytes)+\(rowScaleBytes) byte row in \(pageBytes) bytes")
        }
        guard scaleGroupSize > 0 else {
            throw EngramContainerError.geometryUnsupported(
                "scaleGroupSize must be positive")
        }
        guard rowValueBytes == rowScaleBytes * scaleGroupSize else {
            throw EngramContainerError.geometryUnsupported(
                "\(rowValueBytes) value bytes over \(rowScaleBytes) scales is not a "
                    + "group of \(scaleGroupSize)")
        }
        guard partIndex >= 0 else {
            throw EngramContainerError.geometryUnsupported(
                "partIndex must not be negative")
        }
        let derivedPageCount =
            (partRows + UInt64(rowsPerPage) - 1) / UInt64(rowsPerPage)
        guard pageCount == derivedPageCount else {
            throw EngramContainerError.malformedHeader(
                "pageCount \(pageCount) contradicts \(partRows) rows at \(rowsPerPage) "
                    + "per page (\(derivedPageCount))")
        }
        guard firstPageOffset == UInt64(headerBytes) else {
            throw EngramContainerError.malformedHeader(
                "firstPageOffset \(firstPageOffset) is not \(headerBytes)")
        }

        return Layout(
            tableRows: tableRows,
            rowBase: rowBase,
            partRows: partRows,
            rowValueBytes: rowValueBytes,
            rowScaleBytes: rowScaleBytes,
            pageBytes: pageBytes,
            rowsPerPage: rowsPerPage,
            pageCount: pageCount,
            firstPageOffset: firstPageOffset,
            valueDType: valueDType,
            scaleDType: scaleDType,
            scaleGroupSize: scaleGroupSize,
            partIndex: partIndex)
    }

    // MARK: - Reading

    /// One row, in **one** aligned `pageBytes` read.
    ///
    /// The caller keeps owning `descriptor`. This is the whole read path the
    /// format exists for: no second seek for the scales, and no read smaller
    /// than a page, because a page is what the device transfers anyway.
    public static func readRow(
        fileDescriptor descriptor: Int32,
        path: String,
        layout: Layout,
        row: UInt64
    ) throws -> Row {
        let location = try layout.locate(row: row)
        var page = [UInt8](repeating: 0, count: layout.pageBytes)
        try readPage(
            fileDescriptor: descriptor, path: path, layout: layout,
            offset: location.pageOffset, into: &page)
        return makeRow(row, from: page, at: location, layout: layout)
    }

    /// A batch of rows, in one aligned page read per **distinct** page.
    ///
    /// One Engram module reads `(maxNGram - 1) * heads` rows per token — 24 for
    /// the published configuration — and their addresses are known before block
    /// 0 runs. Rows that happen to share a page cost one read between them
    /// rather than one each; nothing is assumed about how often that happens.
    ///
    /// Results come back in the caller's order, including repeats.
    public static func readRows(
        fileDescriptor descriptor: Int32,
        path: String,
        layout: Layout,
        rows: [UInt64]
    ) throws -> [Row] {
        guard !rows.isEmpty else { return [] }
        let locations = try rows.map { try layout.locate(row: $0) }
        var pages = [UInt64: [UInt8]]()
        pages.reserveCapacity(locations.count)
        // Ascending page order, so the batch walks the file forward rather than
        // in whatever order the hash produced.
        for offset in Set(locations.map(\.pageOffset)).sorted() {
            var page = [UInt8](repeating: 0, count: layout.pageBytes)
            try readPage(
                fileDescriptor: descriptor, path: path, layout: layout,
                offset: offset, into: &page)
            pages[offset] = page
        }
        return zip(rows, locations).map { row, location in
            makeRow(row, from: pages[location.pageOffset]!, at: location, layout: layout)
        }
    }

    /// Distinct aligned page offsets a batch of rows touches, ascending.
    ///
    /// Exposed because the addresses are known before the token's first block
    /// runs, so a prefetcher can issue exactly these reads while block 0
    /// computes — which is what makes the Engram tables affordable at all.
    public static func pageOffsets(layout: Layout, rows: [UInt64]) throws -> [UInt64] {
        var seen = Set<UInt64>()
        var offsets = [UInt64]()
        for row in rows {
            let offset = try layout.locate(row: row).pageOffset
            if seen.insert(offset).inserted { offsets.append(offset) }
        }
        return offsets.sorted()
    }

    private static func makeRow(
        _ row: UInt64, from page: [UInt8], at location: RowLocation, layout: Layout
    ) -> Row {
        let valueStart = location.valueOffsetInPage
        let scaleStart = location.scaleOffsetInPage
        return Row(
            row: row,
            values: Array(page[valueStart..<(valueStart + layout.rowValueBytes)]),
            scales: Array(page[scaleStart..<(scaleStart + layout.rowScaleBytes)]))
    }

    private static func readPage(
        fileDescriptor descriptor: Int32,
        path: String,
        layout: Layout,
        offset: UInt64,
        into page: inout [UInt8]
    ) throws {
        guard let position = off_t(exactly: offset) else {
            throw EngramContainerError.malformedHeader(
                "page offset \(offset) is not representable by this process")
        }
        let want = layout.pageBytes
        var completed = 0
        try page.withUnsafeMutableBytes { buffer in
            guard let base = buffer.baseAddress else { return }
            while completed < want {
                let result = pread(
                    descriptor, base + completed, want - completed,
                    position + off_t(completed))
                if result < 0 {
                    if errno == EINTR { continue }
                    throw StorageCoreError.posix(
                        operation: "pread", path: path, code: errno)
                }
                guard result > 0 else {
                    throw StorageCoreError.shortTransfer(
                        operation: "pread", path: path,
                        offset: offset + UInt64(completed),
                        expected: want - completed, actual: 0)
                }
                completed += result
            }
        }
    }
}
