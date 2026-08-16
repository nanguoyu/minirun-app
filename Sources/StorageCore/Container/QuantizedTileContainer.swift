import Darwin
import Foundation

/// Errors specific to the tile container.
public enum TileContainerError: Error, CustomStringConvertible, LocalizedError {
    case badMagic(found: String)
    case unsupportedVersion(UInt32)
    case malformedHeader(String)
    case geometryUnsupported(String)
    case truncated(path: String, expectedBytes: UInt64, actualBytes: UInt64)
    case tileOutOfRange(index: Int, tileCount: Int)
    case contentKindMismatch(String)
    case digestMismatch(tile: Int, expected: String, actual: String)

    public var description: String {
        switch self {
        case .badMagic(let found):
            return "not a minirun tile container: magic was '\(found)'"
        case .unsupportedVersion(let version):
            return
                "tile container version \(version) is not supported (this build reads "
                + "\(QuantizedTileContainer.readableVersions.map(String.init).joined(separator: " and ")))"
        case .malformedHeader(let detail):
            return "tile container header is malformed: \(detail)"
        case .geometryUnsupported(let detail):
            return "tile geometry is not supported: \(detail)"
        case .truncated(let path, let expected, let actual):
            return
                "tile container '\(path)' is truncated: header describes \(expected) bytes, "
                + "file is \(actual)"
        case .tileOutOfRange(let index, let count):
            return "tile \(index) requested from a container of \(count) tiles"
        case .contentKindMismatch(let detail):
            return "tile container content kind does not permit this operation: \(detail)"
        case .digestMismatch(let tile, let expected, let actual):
            return
                "tile \(tile) fails its recorded SHA-256: header says \(expected), bytes hash to \(actual)"
        }
    }

    public var errorDescription: String? { description }
}

/// How a tile's elements are quantized.
///
/// This is the field that stops the container from being an MXFP4-only format.
/// Both modes are stock MLX kernel modes (spec §10.2 — the runtime dispatches on
/// tensor metadata), and they differ in the file in exactly two ways: the size of
/// a scale element, and whether a bias sub-region exists.
public enum TileQuantMode: UInt32, Sendable, Codable, CaseIterable {
    /// OCP MX FP4: e2m1 elements, one `uint8` e8m0 exponent per group, no bias.
    case mxfp4 = 0
    /// MLX affine: `w = scale * q + bias`, one `float16` scale AND one `float16`
    /// bias per group.
    case affine = 1
    /// FP8 e4m3 elements with a **two-dimensional** grid of e8m0 block scales:
    /// one exponent byte per `scaleBlockRows × groupSize` block of the matrix.
    ///
    /// Unlike the other two modes this is not an MLX kernel mode — MLX has no
    /// primitive that consumes it. It exists so the container can carry such a
    /// checkpoint's bytes *verbatim*, which is the whole requirement: a
    /// byte-preserving repack must not requantize into a shape MLX happens to
    /// like today. Whatever eventually consumes these bytes — a dequantizing
    /// pre-pass, a custom kernel, or a future MLX mode — reads the same bytes
    /// the checkpoint shipped.
    case fp8E4M3Block = 2

    public var name: String {
        switch self {
        case .mxfp4: return "mxfp4"
        case .affine: return "affine"
        case .fp8E4M3Block: return "fp8-e4m3-block"
        }
    }

    /// Whether the mode stores a bias alongside each scale.
    public var hasBiases: Bool {
        switch self {
        case .mxfp4, .fp8E4M3Block: return false
        case .affine: return true
        }
    }

    /// Whether scales form a 2-D block grid rather than one scale per row-group.
    ///
    /// The one-dimensional case is not a different mechanism, only the
    /// `scaleBlockRows == 1` corner of the same one: a group is a
    /// `1 × groupSize` block. Modes that set this expect a block taller than
    /// one row, and ``TileGeometry/scaleBlockRows`` says how tall.
    public var usesBlockScales: Bool {
        switch self {
        case .mxfp4, .affine: return false
        case .fp8E4M3Block: return true
        }
    }

    /// The scale element type this mode implies, when it implies one.
    ///
    /// MX fixes it: an MXFP4 scale is an e8m0 exponent byte. Affine does not —
    /// `mx.quantize(mode: .affine)` returns scales and biases in the *input*
    /// tensor's dtype, so a bfloat16 checkpoint yields bfloat16 scales and a
    /// float16 one yields float16. Measured on
    /// `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`, whose
    /// `switch_mlp.*.scales` are BF16 — an early version of this format assumed
    /// float16 and the converter's dtype check caught it. The type is therefore
    /// a header field, never an inference from the mode (spec §9.3:
    /// "quantization type per tensor/section").
    public var mandatoryScaleDType: TileScaleDType? {
        switch self {
        case .mxfp4: return .uint8E8M0
        case .affine: return nil
        // e8m0 is what "ue8m0 block scale" means; a float scale would be a
        // different format, not a variant of this one.
        case .fp8E4M3Block: return .uint8E8M0
        }
    }

    /// What a writer uses when the caller does not say. A reader never uses it.
    public var defaultScaleDType: TileScaleDType {
        mandatoryScaleDType ?? .bfloat16
    }

    /// The group size this mode is fixed at, when it is fixed at one.
    ///
    /// MX formats are fixed at 32 by OCP MX. Affine is not fixed by anything —
    /// it is a converter choice — so it is validated as a positive power of two
    /// and carried in the header rather than assumed.
    public var mandatoryGroupSize: Int? {
        switch self {
        case .mxfp4: return 32
        case .affine: return nil
        // Deliberately not fixed at 128. The checkpoint this mode was added for
        // uses 128x128, but that is a producer's choice recorded in its config,
        // not a property of the encoding, and spec 9.3 asks for such things to
        // be carried rather than assumed.
        case .fp8E4M3Block: return nil
        }
    }

    /// Bits per packed element this mode implies, when it implies one.
    public var mandatoryBits: Int? {
        switch self {
        case .mxfp4: return 4
        case .affine: return nil
        case .fp8E4M3Block: return 8
        }
    }
}

/// The element type of a tile's scale (and, where present, bias) sub-region.
///
/// Carried in the file rather than derived from the quantization mode, because
/// for affine it genuinely is not derivable: MLX returns scales in the source
/// tensor's dtype. See ``TileQuantMode/mandatoryScaleDType``.
public enum TileScaleDType: UInt32, Sendable, Codable, CaseIterable {
    /// One `uint8` e8m0 exponent per group. MX formats only.
    case uint8E8M0 = 0
    case float16 = 1
    case bfloat16 = 2
    case float32 = 3

    public var elementBytes: Int {
        switch self {
        case .uint8E8M0: return 1
        case .float16, .bfloat16: return 2
        case .float32: return 4
        }
    }

    public var name: String {
        switch self {
        case .uint8E8M0: return "uint8-e8m0"
        case .float16: return "float16"
        case .bfloat16: return "bfloat16"
        case .float32: return "float32"
        }
    }
}

/// Whether a container's bytes can be recomputed, or only checked.
///
/// Added in format version 2 on the specification's own recommendation (§9.3,
/// v0.4.2): "a future format version SHOULD carry a `contentKind` field
/// discriminating synthetic-deterministic from checkpoint-derived content; the
/// `seed` field and content verification currently apply only to the former with
/// nothing in the file saying which is which."
///
/// A version-1 file says nothing, so it is read as ``syntheticDeterministic`` —
/// which is what every version-1 *writer* in this repository produced, and what
/// the `seed` field means there. Checkpoint-derived version-1 containers exist
/// (the tiny and flagship expert converters wrote them with `seed = 0`); their
/// integrity comes from the sidecar manifest's SHA-256, and
/// ``verifyTile(_:layout:tile:)`` was simply documented as not applying. Version
/// 2 puts that in the file, so the check can be refused rather than remembered.
public enum TileContentKind: UInt32, Sendable, Codable, CaseIterable {
    /// Content is a pure function of `(seed, tile, offset)` and can be
    /// regenerated. ``QuantizedTileContainer/verifyTile(_:layout:tile:)`` applies.
    case syntheticDeterministic = 0
    /// Content came from a checkpoint. `seed` is meaningless; integrity comes
    /// from the per-tile digests when present, or the sidecar manifest.
    case checkpointDerived = 1

    public var name: String {
        switch self {
        case .syntheticDeterministic: return "synthetic-deterministic"
        case .checkpointDerived: return "checkpoint-derived"
        }
    }
}

/// Rows and columns of one output-row tile.
///
/// Tiling is along output rows only, never along `cols` (the reduction axis).
/// Row-tiled outputs concatenate with no cross-tile accumulation, so a paged
/// result is the same arithmetic as a resident one in the same order, and can
/// be required to be bit-identical rather than merely close. Splitting `cols`
/// would introduce a partial-sum order that depends on the tiling, and the
/// strongest available correctness gate would be lost (spec §16.1).
public struct TileGeometry: Sendable, Equatable, Codable {
    /// Output features in one tile.
    public let rows: Int
    /// Input features. The full reduction length, identical in every tile.
    public let cols: Int
    /// Elements sharing one scale (and, in affine mode, one bias).
    public let groupSize: Int
    /// Bits per packed element.
    public let bits: Int
    public let quantMode: TileQuantMode
    /// Element type of the scale and bias sub-regions.
    public let scaleDType: TileScaleDType
    /// Rows of the matrix covered by one scale.
    ///
    /// `1` — the only value versions 1 and 2 can express — means the familiar
    /// one-scale-per-`groupSize`-elements-of-one-row layout. A larger value
    /// means the scales form a 2-D grid: one scale per
    /// `scaleBlockRows × groupSize` block, in row-major block order. The
    /// column extent of a block is `groupSize`, which already meant exactly
    /// that; only the row extent is new.
    public let scaleBlockRows: Int

    public init(
        rows: Int,
        cols: Int,
        groupSize: Int = QuantizedTileContainer.mxfp4GroupSize,
        bits: Int = 4,
        quantMode: TileQuantMode = .mxfp4,
        scaleDType: TileScaleDType? = nil,
        scaleBlockRows: Int = 1
    ) {
        self.rows = rows
        self.cols = cols
        self.groupSize = groupSize
        self.bits = bits
        self.quantMode = quantMode
        self.scaleDType = scaleDType ?? quantMode.defaultScaleDType
        self.scaleBlockRows = scaleBlockRows
    }

    /// Groups along one row — equivalently, blocks along one block-row.
    public var groupsPerRow: Int { cols / groupSize }
    /// Rows of the scale grid.
    public var scaleGridRows: Int { rows / scaleBlockRows }
    /// Bytes of packed weights per tile.
    public var packedBytes: Int { rows * cols * bits / 8 }
    /// Bytes of scales per tile. Reduces to `rows * groupsPerRow * element` when
    /// `scaleBlockRows == 1`, which is every version-1 and version-2 container.
    public var scaleBytes: Int { scaleGridRows * groupsPerRow * scaleDType.elementBytes }
    /// Bytes of biases per tile, or 0 when the mode has none.
    public var biasBytes: Int {
        quantMode.hasBiases ? scaleGridRows * groupsPerRow * scaleDType.elementBytes : 0
    }

    /// True when this geometry is expressible in format version 1.
    ///
    /// Version 1 has no mode, no bits, no scale dtype and no bias fields, and
    /// hard-codes group size 32 — so it can only describe 4-bit MXFP4 with e8m0
    /// scales. Anything else needs version 2, and that is the only thing that
    /// decides which version the writer emits.
    public var isVersion1Expressible: Bool {
        quantMode == .mxfp4 && bits == 4
            && groupSize == QuantizedTileContainer.mxfp4GroupSize
            && scaleDType == .uint8E8M0
            && scaleBlockRows == 1
    }

    /// True when this geometry is expressible in format version 2.
    ///
    /// Version 2 has no `scaleBlockRows` field, so it can only describe a scale
    /// grid one row tall per group — and its readers compute `scaleBytes` as
    /// `rows * groupsPerRow`. A 2-D grid read by a version-2 reader would size
    /// the scale sub-region `scaleBlockRows` times too large and walk off every
    /// tile boundary after the first, so the field is genuinely required rather
    /// than merely convenient, and the version bump with it.
    public var isVersion2Expressible: Bool { scaleBlockRows == 1 }
}

/// A container of quantized row tiles, laid out for direct adoption.
///
/// ## Layout
///
/// ```text
/// [ header, headerBytes long, page-padded ]
/// [ tile 0: packed bytes | scale bytes | bias bytes (affine only) ]
/// [ tile 1: packed bytes | scale bytes | bias bytes (affine only) ]
/// ...
/// ```
///
/// Every tile starts on a 16 KiB boundary, and within a tile the packed, scale
/// and bias sub-regions each start on a 16 KiB boundary and are a whole number
/// of units long. That is not decoration: one tile is read with one `pread` into
/// one slot, and then the *same* slot is adopted as two or three `MLXArray`s —
/// packed at offset 0, scales at the scale offset, biases at the bias offset.
/// MLX wraps an adopted pointer for the Metal backend, which requires
/// page-aligned memory, so every sub-region has to be independently aligned or
/// the later adoptions would fall off the zero-copy path. Geometries that do not
/// satisfy this are rejected at construction rather than silently padded.
///
/// ## Two content kinds
///
/// Synthetic content is deterministic in the seed, so any tile's expected bytes
/// can be recomputed without a reference copy, at any offset, in any order — the
/// same property `DeterministicContent` provides for the storage benchmark.
/// Checkpoint content cannot be regenerated, so version 2 can carry a per-tile
/// SHA-256 table in the header instead. ``TileContentKind`` says which, in the
/// file, so ``verifyTile(_:layout:tile:)`` refuses rather than silently
/// comparing checkpoint bytes against a PRNG.
///
/// Scale bytes of *synthetic* content are constrained to `[100, 140]`. This is a
/// plumbing fixture, not a numerics fixture: byte 255 is the e8m0 NaN encoding
/// and the extremes produce subnormal or overflowing products, either of which
/// would make a bit-identity comparison fail for reasons that have nothing to do
/// with the pager. The window sits around 127 (2^0) and keeps every product
/// finite and normal.
///
/// ## Version history
///
/// - **1** — MXFP4 only, group size 32, no biases, no content kind, no digests.
///   Still written whenever the request is expressible in it, byte for byte as
///   before, and still read.
/// - **2** — adds `quantMode`, `bits`, `contentKind`, a feature-flag word, the
///   bias sub-region and an optional per-tile SHA-256 table. Written only when
///   version 1 cannot express the request (spec §9.3: feature flags rather than
///   implicit behaviour, and a clear compatibility policy).
public enum QuantizedTileContainer {
    public static let magic = "MNRNTIL1"

    /// The version a writer emits when the request needs no version-2 feature.
    public static let version: UInt32 = 1
    /// The newest version this build writes.
    public static let currentVersion: UInt32 = 3
    /// Every version this build reads.
    public static let readableVersions: [UInt32] = [1, 2, 3]

    /// Fixed at 32 by OCP MX for all MX formats. Not a default for other modes.
    public static let mxfp4GroupSize = 32

    /// Retained spelling of ``mxfp4GroupSize``.
    ///
    /// Callers that predate the affine mode mean the MXFP4 group size when they
    /// say `groupSize`, and every one of them is an MXFP4 call site.
    public static let groupSize = mxfp4GroupSize

    /// Alignment every tile and every sub-region inside a tile is padded to.
    ///
    /// Fixed at 16 KiB by the *format*, deliberately not taken from the host's
    /// page size. Apple Silicon pages are 16 KiB and x86-64 pages are 4 KiB, so
    /// a layout derived from `getpagesize()` would produce a file that only
    /// validates on the architecture that wrote it — a container generated on
    /// this development Mac would be rejected by a CI runner and vice versa.
    /// 16 KiB is a multiple of both, so a file written anywhere is correctly
    /// aligned everywhere.
    public static let regionAlignment = 16384

    /// One aligned unit, so tile 0 starts aligned.
    public static let headerBytes = regionAlignment

    /// Bytes of the fixed part of a version-1 header.
    static let version1HeaderFields = 80
    /// Bytes of the fixed part of a version-2 header.
    static let version2HeaderFields = 128
    /// Bytes of the fixed part of a version-3 header.
    static let version3HeaderFields = 160
    /// Where a version-2 per-tile digest table starts, when present.
    static let digestTableOffset = 128
    static let digestBytes = 32

    /// Where the per-tile digest table starts in a file of this version.
    ///
    /// Version 3 appends fixed fields where version 2 put the digest table, so
    /// the table moves. The offset was already a header field rather than an
    /// assumption, which is what makes moving it a compatible change at all.
    static func digestTableOffset(forVersion version: UInt32) -> Int {
        version >= 3 ? version3HeaderFields : version2HeaderFields
    }

    /// Round `value` up to the next multiple of `alignment`.
    static func alignUp(_ value: Int, _ alignment: Int) -> Int {
        let remainder = value % alignment
        return remainder == 0 ? value : value + (alignment - remainder)
    }

    /// Feature flags. Spec §9.3: "feature flags rather than implicit behaviour."
    public struct FeatureFlags: OptionSet, Sendable, Codable, Equatable {
        public let rawValue: UInt32
        public init(rawValue: UInt32) { self.rawValue = rawValue }
        /// A bias sub-region follows the scales in every tile.
        public static let biases = FeatureFlags(rawValue: 1 << 0)
        /// The header carries a SHA-256 per tile.
        public static let tileDigests = FeatureFlags(rawValue: 1 << 1)
        /// Scales form a 2-D block grid (`scaleBlockRows > 1`). Version 3 only.
        public static let blockScales2D = FeatureFlags(rawValue: 1 << 2)
        /// A sub-region's payload is followed by zero padding to the alignment.
        ///
        /// Version 1 and 2 geometries are rejected unless every sub-region is
        /// already a whole number of alignment units, so this bit never appears
        /// on them. A 2-D block grid cannot satisfy that — the largest scale
        /// grid in the checkpoint that motivated the mode is 24 KiB short of one
        /// unit — so the region is padded and the flag says the trailing bytes
        /// are padding rather than content.
        public static let paddedSubregions = FeatureFlags(rawValue: 1 << 3)

        /// Every bit this build understands. A file setting anything else is
        /// rejected rather than read with the unknown feature ignored.
        public static let known: FeatureFlags = [
            .biases, .tileDigests, .blockScales2D, .paddedSubregions,
        ]
    }

    /// Label mixed into the seed for packed content, so packed and scale
    /// streams do not correlate.
    static let packedSeedLabel: UInt64 = 0x5041_434B  // "PACK"
    static let scaleSeedLabel: UInt64 = 0x5343_414C  // "SCAL"

    /// Smallest and largest scale byte the writer emits.
    public static let minimumScaleByte: UInt8 = 100
    public static let scaleByteSpan: UInt64 = 41  // [100, 140]

    /// Fixed description of one container, derived from geometry and count.
    public struct Layout: Sendable, Equatable, Codable {
        public let geometry: TileGeometry
        public let tileCount: Int
        public let seed: UInt64
        /// Offset of a tile's scale sub-region within the tile.
        public let scaleOffsetInTile: Int
        /// Distance from one tile's start to the next.
        public let tileStride: Int
        public let firstTileOffset: UInt64
        /// Offset of a tile's bias sub-region within the tile, or `nil` when the
        /// mode has no biases.
        public let biasOffsetInTile: Int?
        public let contentKind: TileContentKind
        public let featureFlags: FeatureFlags
        /// The version this layout is written as, and the version a file it was
        /// decoded from carried.
        public let formatVersion: UInt32
        /// One SHA-256 per tile, in tile order, when the file carries them.
        public let tileDigests: [Data]?
        /// Bytes the packed sub-region occupies, payload plus any padding.
        /// Equal to `geometry.packedBytes` unless ``FeatureFlags/paddedSubregions``.
        public let packedRegionBytes: Int
        /// Bytes the scale sub-region occupies, payload plus any padding.
        /// Equal to `geometry.scaleBytes` unless ``FeatureFlags/paddedSubregions``.
        public let scaleRegionBytes: Int

        public var totalBytes: UInt64 {
            firstTileOffset + UInt64(tileCount) * UInt64(tileStride)
        }
        /// Bytes a full pass actually reads. Equal to `totalBytes` minus the
        /// header, since every tile is read whole (ADR-0001: full-tile reads).
        public var payloadBytes: UInt64 { UInt64(tileCount) * UInt64(tileStride) }
        /// Output features of the whole matrix.
        public var totalRows: Int { tileCount * geometry.rows }

        public var quantMode: TileQuantMode { geometry.quantMode }

        public func tileOffset(_ index: Int) -> UInt64 {
            firstTileOffset + UInt64(index) * UInt64(tileStride)
        }

        /// The pager request for one tile: whole tile, one read, one slot.
        public func region(_ index: Int, sourceID: UInt64 = 0) -> RegionDescriptor {
            RegionDescriptor(
                identity: RegionIdentity(sourceID: sourceID, index: UInt64(index)),
                offset: tileOffset(index),
                length: tileStride)
        }
    }

    /// Validate a geometry and compute its layout.
    ///
    /// - Parameters:
    ///   - contentKind: whether the bytes will be regenerable. Only recorded in
    ///     a version-2 file; a ``TileContentKind/checkpointDerived`` request
    ///     therefore forces version 2, because recording it is the whole point.
    ///   - tileDigests: one SHA-256 per tile, or `nil`. Forces version 2.
    public static func layout(
        geometry: TileGeometry,
        tileCount: Int,
        seed: UInt64,
        contentKind: TileContentKind = .syntheticDeterministic,
        tileDigests: [Data]? = nil,
        alignment: Int = QuantizedTileContainer.regionAlignment
    ) throws -> Layout {
        guard geometry.rows > 0, geometry.cols > 0 else {
            throw TileContainerError.geometryUnsupported("rows and cols must be positive")
        }
        guard tileCount > 0 else {
            throw TileContainerError.geometryUnsupported("tileCount must be positive")
        }
        guard geometry.bits == 4 || geometry.bits == 8 else {
            throw TileContainerError.geometryUnsupported(
                "bits \(geometry.bits) is not supported; this format packs 4 or 8")
        }
        guard geometry.groupSize > 0, geometry.groupSize & (geometry.groupSize - 1) == 0 else {
            throw TileContainerError.geometryUnsupported(
                "group size \(geometry.groupSize) must be a positive power of two")
        }
        if let mandatory = geometry.quantMode.mandatoryGroupSize,
            geometry.groupSize != mandatory
        {
            throw TileContainerError.geometryUnsupported(
                "\(geometry.quantMode.name) fixes the group size at \(mandatory), not "
                    + "\(geometry.groupSize)")
        }
        if let mandatory = geometry.quantMode.mandatoryScaleDType,
            geometry.scaleDType != mandatory
        {
            throw TileContainerError.geometryUnsupported(
                "\(geometry.quantMode.name) scales are \(mandatory.name), not "
                    + "\(geometry.scaleDType.name)")
        }
        if let mandatory = geometry.quantMode.mandatoryBits, geometry.bits != mandatory {
            throw TileContainerError.geometryUnsupported(
                "\(geometry.quantMode.name) elements are \(mandatory)-bit, not "
                    + "\(geometry.bits)-bit")
        }
        // The block-row extent and the mode have to agree in both directions:
        // a 2-D mode with a 1-row block is a mislabelled 1-D container, and a
        // 1-D mode with a taller block is a geometry no reader would interpret
        // the way the writer meant.
        guard geometry.scaleBlockRows > 0 else {
            throw TileContainerError.geometryUnsupported(
                "scaleBlockRows must be positive")
        }
        if geometry.quantMode.usesBlockScales {
            guard geometry.scaleBlockRows > 1 else {
                throw TileContainerError.geometryUnsupported(
                    "\(geometry.quantMode.name) is a 2-D block-scale mode; scaleBlockRows "
                        + "is 1, which is the one-scale-per-row-group layout the other "
                        + "modes already express")
            }
        } else {
            guard geometry.scaleBlockRows == 1 else {
                throw TileContainerError.geometryUnsupported(
                    "\(geometry.quantMode.name) has one scale per row-group; scaleBlockRows "
                        + "\(geometry.scaleBlockRows) needs a 2-D block-scale mode")
            }
        }
        guard geometry.rows % geometry.scaleBlockRows == 0 else {
            throw TileContainerError.geometryUnsupported(
                "rows \(geometry.rows) must be a multiple of scaleBlockRows "
                    + "\(geometry.scaleBlockRows); partial final blocks are not supported")
        }
        if geometry.quantMode == .affine, geometry.scaleDType == .uint8E8M0 {
            throw TileContainerError.geometryUnsupported(
                "affine scales are a floating-point type; e8m0 is an MX encoding")
        }
        guard geometry.cols % geometry.groupSize == 0 else {
            throw TileContainerError.geometryUnsupported(
                "cols \(geometry.cols) must be a multiple of the group size \(geometry.groupSize); "
                    + "partial final groups are not supported")
        }
        // cols % groupSize == 0 already makes packedBytes whole for these group
        // sizes; state the divisibility that the uint32 reinterpretation needs
        // explicitly anyway.
        let elementsPerWord = 32 / geometry.bits
        guard geometry.cols % elementsPerWord == 0 else {
            throw TileContainerError.geometryUnsupported(
                "cols \(geometry.cols) must be a multiple of \(elementsPerWord) "
                    + "(\(elementsPerWord) \(geometry.bits)-bit elements per uint32)")
        }
        // A 2-D block grid cannot be a whole number of alignment units: the
        // scale grid shrinks by the block area, so for any matrix that fits in
        // memory the grid is far smaller than 16 KiB. Such a mode pads instead,
        // and says so in the flags. Every other mode keeps the original
        // requirement, so every pre-existing geometry is accepted or rejected
        // exactly as before.
        let padded = geometry.quantMode.usesBlockScales
        if !padded {
            guard geometry.packedBytes % alignment == 0 else {
                throw TileContainerError.geometryUnsupported(
                    """
                    packed bytes per tile (\(geometry.packedBytes)) must be a multiple of \
                    \(alignment) so the scale sub-region starts aligned and can be adopted as its \
                    own zero-copy array; try rows*cols*bits divisible by \(alignment * 8)
                    """)
            }
            guard geometry.scaleBytes % alignment == 0 else {
                throw TileContainerError.geometryUnsupported(
                    """
                    scale bytes per tile (\(geometry.scaleBytes)) must be a multiple of \
                    \(alignment); try rows*cols divisible by \
                    \(alignment * geometry.groupSize / geometry.scaleDType.elementBytes)
                    """)
            }
            if geometry.quantMode.hasBiases {
                guard geometry.biasBytes % alignment == 0 else {
                    throw TileContainerError.geometryUnsupported(
                        """
                        bias bytes per tile (\(geometry.biasBytes)) must be a multiple of \
                        \(alignment) so the bias sub-region can be adopted as its own zero-copy array
                        """)
                }
            }
        }
        let packedRegion = alignUp(geometry.packedBytes, alignment)
        let scaleRegion = alignUp(geometry.scaleBytes, alignment)

        var flags: FeatureFlags = []
        if geometry.quantMode.hasBiases { flags.insert(.biases) }
        if tileDigests != nil { flags.insert(.tileDigests) }
        if geometry.scaleBlockRows > 1 { flags.insert(.blockScales2D) }
        if padded { flags.insert(.paddedSubregions) }

        // The lowest version that can express the request. Every pre-existing
        // caller lands on 1 or 2 and its file is byte-identical to what it was.
        let formatVersion: UInt32
        if !geometry.isVersion2Expressible || geometry.quantMode.usesBlockScales {
            formatVersion = 3
        } else if !geometry.isVersion1Expressible
            || contentKind != .syntheticDeterministic
            || !flags.isEmpty
        {
            formatVersion = 2
        } else {
            formatVersion = version
        }

        if let tileDigests {
            guard tileDigests.count == tileCount else {
                throw TileContainerError.geometryUnsupported(
                    "\(tileDigests.count) digests for \(tileCount) tiles")
            }
            guard tileDigests.allSatisfy({ $0.count == digestBytes }) else {
                throw TileContainerError.geometryUnsupported(
                    "every tile digest must be \(digestBytes) bytes of SHA-256")
            }
            let tableOffset = digestTableOffset(forVersion: formatVersion)
            let capacity = (headerBytes - tableOffset) / digestBytes
            guard tileCount <= capacity else {
                throw TileContainerError.geometryUnsupported(
                    "a \(headerBytes)-byte header holds at most \(capacity) tile digests, "
                        + "not \(tileCount); write the digests to the sidecar manifest instead")
            }
        }

        let biasOffset = geometry.quantMode.hasBiases ? packedRegion + scaleRegion : nil
        let biasRegion = alignUp(geometry.biasBytes, alignment)

        return Layout(
            geometry: geometry,
            tileCount: tileCount,
            seed: seed,
            scaleOffsetInTile: packedRegion,
            tileStride: packedRegion + scaleRegion + biasRegion,
            firstTileOffset: UInt64(headerBytes),
            biasOffsetInTile: biasOffset,
            contentKind: contentKind,
            featureFlags: flags,
            formatVersion: formatVersion,
            tileDigests: tileDigests,
            packedRegionBytes: packedRegion,
            scaleRegionBytes: scaleRegion)
    }

    // MARK: - Content

    /// Expected packed bytes of `tile` into `destination`, which must be
    /// exactly `geometry.packedBytes` long.
    public static func fillPacked(
        _ destination: UnsafeMutableRawBufferPointer,
        layout: Layout,
        tile: Int
    ) {
        // The packed stream is addressed by absolute tile-local byte offset, so
        // every tile has an independent derived seed and a tile's content never
        // depends on the tiles before it.
        let seed = SplitMix64.derive(
            seed: layout.seed &+ UInt64(tile) &* 0x9E37_79B9,
            label: packedSeedLabel)
        DeterministicContent.fill(destination, seed: seed, fileOffset: 0)
    }

    /// Expected scale bytes of `tile`, all within `[100, 140]`.
    public static func fillScales(
        _ destination: UnsafeMutableRawBufferPointer,
        layout: Layout,
        tile: Int
    ) {
        guard let base = destination.baseAddress else { return }
        let seed = SplitMix64.derive(
            seed: layout.seed &+ UInt64(tile) &* 0x9E37_79B9,
            label: scaleSeedLabel)
        for index in 0..<destination.count {
            base.storeBytes(
                of: scaleByte(seed: seed, index: UInt64(index)), toByteOffset: index, as: UInt8.self)
        }
    }

    /// One scale byte as a pure function of `(seed, index)`.
    @inline(__always)
    public static func scaleByte(seed: UInt64, index: UInt64) -> UInt8 {
        minimumScaleByte &+ UInt8(SplitMix64.value(seed: seed, index: index) % scaleByteSpan)
    }

    // MARK: - Writing

    /// Write a synthetic container. Returns the layout actually written.
    ///
    /// - Parameters:
    ///   - chunkTiles: tiles buffered per `write` call. Bounds the writer's own
    ///     memory so generating an 8 GiB fixture does not need 8 GiB.
    ///   - uncached: set `F_NOCACHE` for the tile writes, so the bytes are not
    ///     left sitting in the unified buffer cache. This matters for
    ///     measurement and not for correctness: a file written normally is warm
    ///     in the page cache the instant it is finished, and `F_NOCACHE` on the
    ///     *reader* stops reads populating the cache but does not evict what
    ///     the writer already put there (spec §5.7). Without this, a scale run
    ///     measures memcpy.
    @discardableResult
    public static func write(
        path: String,
        geometry: TileGeometry,
        tileCount: Int,
        seed: UInt64,
        chunkTiles: Int = 8,
        uncached: Bool = false
    ) throws -> Layout {
        guard geometry.quantMode == .mxfp4 else {
            // The synthetic generator has no bias stream and no block-scale
            // stream, and inventing either would make a fixture that looks
            // numerically meaningful and is not. It also writes no padding, so
            // a padded mode would leave uninitialised bytes in the file.
            throw TileContainerError.geometryUnsupported(
                "the synthetic writer produces \(TileQuantMode.mxfp4.name) content only; "
                    + "\(geometry.quantMode.name) containers come from a converter")
        }
        let layout = try layout(geometry: geometry, tileCount: tileCount, seed: seed)

        let descriptor = Darwin.open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        guard descriptor >= 0 else {
            throw StorageCoreError.posix(operation: "open", path: path, code: errno)
        }
        defer { close(descriptor) }

        // The header goes out first, while caching is still on: it comes from
        // an ordinary array rather than an aligned buffer, and 16 KiB in the
        // cache is not what confounds a multi-gigabyte measurement.
        var header = [UInt8](repeating: 0, count: headerBytes)
        encodeHeader(layout, into: &header)
        try writeAll(descriptor, header, at: 0, path: path)
        if uncached {
            _ = fcntl(descriptor, F_NOCACHE, 1)
        }

        let tilesPerChunk = max(1, chunkTiles)
        let buffer = try AlignedBuffer(count: layout.tileStride * tilesPerChunk)
        var tile = 0
        while tile < tileCount {
            let count = min(tilesPerChunk, tileCount - tile)
            for slot in 0..<count {
                let base = buffer.pointer + slot * layout.tileStride
                fillPacked(
                    UnsafeMutableRawBufferPointer(start: base, count: geometry.packedBytes),
                    layout: layout,
                    tile: tile + slot)
                fillScales(
                    UnsafeMutableRawBufferPointer(
                        start: base + layout.scaleOffsetInTile, count: geometry.scaleBytes),
                    layout: layout,
                    tile: tile + slot)
            }
            try writeAll(
                descriptor,
                UnsafeRawBufferPointer(start: buffer.pointer, count: count * layout.tileStride),
                at: layout.tileOffset(tile),
                path: path)
            tile += count
        }
        return layout
    }

    /// Encode a header for an already-computed layout.
    ///
    /// Exposed so a converter that streams tile bytes itself — the Python
    /// converters do, and so does any future Swift one — writes the same header
    /// as ``write(path:geometry:tileCount:seed:chunkTiles:uncached:)`` rather
    /// than its own transcription of it.
    public static func encodedHeader(_ layout: Layout) -> [UInt8] {
        var header = [UInt8](repeating: 0, count: headerBytes)
        encodeHeader(layout, into: &header)
        return header
    }

    private static func encodeHeader(_ layout: Layout, into header: inout [UInt8]) {
        var cursor = 0
        func put<T: FixedWidthInteger>(_ value: T) {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { bytes in
                for byte in bytes {
                    header[cursor] = byte
                    cursor += 1
                }
            }
        }
        for (index, scalar) in magic.utf8.enumerated() { header[index] = scalar }
        cursor = 8
        put(layout.formatVersion)
        put(UInt32(headerBytes))
        put(UInt32(layout.tileCount))
        put(UInt32(layout.geometry.rows))
        put(UInt32(layout.geometry.cols))
        put(UInt32(layout.geometry.groupSize))
        put(UInt64(layout.geometry.packedBytes))
        put(UInt64(layout.geometry.scaleBytes))
        put(UInt64(layout.scaleOffsetInTile))
        put(UInt64(layout.tileStride))
        put(layout.firstTileOffset)
        put(layout.seed)
        guard layout.formatVersion >= 2 else { return }
        put(layout.geometry.quantMode.rawValue)
        put(UInt32(layout.geometry.bits))
        put(layout.contentKind.rawValue)
        put(layout.featureFlags.rawValue)
        put(UInt64(layout.geometry.biasBytes))
        put(UInt64(layout.biasOffsetInTile ?? 0))
        let tableOffset = digestTableOffset(forVersion: layout.formatVersion)
        put(UInt64(layout.featureFlags.contains(.tileDigests) ? tableOffset : 0))
        put(layout.geometry.scaleDType.rawValue)
        put(UInt32(0))  // reserved, keeps the version-2 digest table starting at 128
        if layout.formatVersion >= 3 {
            put(UInt32(layout.geometry.scaleBlockRows))
            put(UInt64(layout.packedRegionBytes))
            put(UInt64(layout.scaleRegionBytes))
            put(UInt32(0))  // reserved
            put(UInt64(0))  // reserved, keeps the digest table starting at 160
        }
        if let digests = layout.tileDigests {
            var at = tableOffset
            for digest in digests {
                for byte in digest {
                    header[at] = byte
                    at += 1
                }
            }
        }
    }

    private static func writeAll(
        _ descriptor: Int32, _ bytes: [UInt8], at offset: UInt64, path: String
    ) throws {
        try bytes.withUnsafeBytes { try writeAll(descriptor, $0, at: offset, path: path) }
    }

    private static func writeAll(
        _ descriptor: Int32, _ bytes: UnsafeRawBufferPointer, at offset: UInt64, path: String
    ) throws {
        guard let base = bytes.baseAddress else { return }
        var written = 0
        while written < bytes.count {
            let result = pwrite(
                descriptor, base + written, bytes.count - written, off_t(offset) + off_t(written))
            if result < 0 {
                if errno == EINTR { continue }
                throw StorageCoreError.posix(operation: "pwrite", path: path, code: errno)
            }
            if result == 0 {
                throw StorageCoreError.shortTransfer(
                    operation: "pwrite", path: path, offset: offset,
                    expected: bytes.count, actual: written)
            }
            written += result
        }
    }

    // MARK: - Reading

    /// Read and validate a container header, and check the file is long enough
    /// for every tile it claims.
    public static func open(path: String) throws -> Layout {
        let descriptor = Darwin.open(path, O_RDONLY)
        guard descriptor >= 0 else {
            throw StorageCoreError.posix(operation: "open", path: path, code: errno)
        }
        defer { close(descriptor) }
        return try open(fileDescriptor: descriptor, path: path)
    }

    /// Read a container through an already-authorized descriptor.
    ///
    /// The descriptor remains owned by the caller. This is the same parser as
    /// ``open(path:)`` but avoids reopening a mutable pathname after a sandbox
    /// bookmark and rooted verification have selected the physical object.
    public static func open(fileDescriptor descriptor: Int32, path: String) throws -> Layout {

        var header = [UInt8](repeating: 0, count: headerBytes)
        let read = header.withUnsafeMutableBytes { buffer -> Int in
            pread(descriptor, buffer.baseAddress, headerBytes, 0)
        }
        guard read == headerBytes else {
            throw TileContainerError.truncated(
                path: path, expectedBytes: UInt64(headerBytes), actualBytes: UInt64(max(read, 0)))
        }

        let layout = try decodeHeader(header)

        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw StorageCoreError.posix(operation: "fstat", path: path, code: errno)
        }
        let actual = UInt64(status.st_size)
        guard actual >= layout.totalBytes else {
            throw TileContainerError.truncated(
                path: path, expectedBytes: layout.totalBytes, actualBytes: actual)
        }
        return layout
    }

    static func decodeHeader(_ header: [UInt8]) throws -> Layout {
        let found = String(decoding: header[0..<8], as: UTF8.self)
        guard found == magic else {
            throw TileContainerError.badMagic(found: found)
        }
        var cursor = 8
        func take<T: FixedWidthInteger>(_ type: T.Type) -> T {
            let size = MemoryLayout<T>.size
            let value = header.withUnsafeBytes {
                T(littleEndian: $0.loadUnaligned(fromByteOffset: cursor, as: T.self))
            }
            cursor += size
            return value
        }

        let fileVersion = take(UInt32.self)
        guard readableVersions.contains(fileVersion) else {
            throw TileContainerError.unsupportedVersion(fileVersion)
        }
        let declaredHeaderBytes = take(UInt32.self)
        let tileCount = take(UInt32.self)
        let rows = take(UInt32.self)
        let cols = take(UInt32.self)
        let group = take(UInt32.self)
        let packedBytes = take(UInt64.self)
        let scaleBytes = take(UInt64.self)
        let scaleOffsetInTile = take(UInt64.self)
        let tileStride = take(UInt64.self)
        let firstTileOffset = take(UInt64.self)
        let seed = take(UInt64.self)

        guard declaredHeaderBytes == UInt32(headerBytes) else {
            throw TileContainerError.malformedHeader(
                "header length \(declaredHeaderBytes) != \(headerBytes)")
        }

        // Version 1 says nothing about mode, bits, content kind or biases, and
        // every version-1 writer in this repository produced 4-bit MXFP4. Those
        // are the defaults, and they are defaults *of the version*, not guesses.
        var quantMode = TileQuantMode.mxfp4
        var bits = 4
        var contentKind = TileContentKind.syntheticDeterministic
        var flags: FeatureFlags = []
        var declaredBiasBytes: UInt64 = 0
        var declaredBiasOffset: UInt64 = 0
        var digests: [Data]?
        var scaleDType = TileScaleDType.uint8E8M0
        // Version 1 and 2 have no such field and mean the one-row block.
        var scaleBlockRows = 1
        var declaredPackedRegion: UInt64?
        var declaredScaleRegion: UInt64?

        if fileVersion >= 2 {
            let rawMode = take(UInt32.self)
            guard let mode = TileQuantMode(rawValue: rawMode) else {
                throw TileContainerError.malformedHeader("unknown quantization mode \(rawMode)")
            }
            quantMode = mode
            bits = Int(take(UInt32.self))
            let rawKind = take(UInt32.self)
            guard let kind = TileContentKind(rawValue: rawKind) else {
                throw TileContainerError.malformedHeader("unknown content kind \(rawKind)")
            }
            contentKind = kind
            let rawFlags = take(UInt32.self)
            flags = FeatureFlags(rawValue: rawFlags)
            // An unknown feature bit means the file uses something this build
            // does not implement. Reading it anyway would be the "silent
            // fallback" spec §19.2 forbids.
            guard FeatureFlags(rawValue: rawFlags & ~FeatureFlags.known.rawValue).isEmpty else {
                throw TileContainerError.malformedHeader(
                    "feature flags 0x\(String(rawFlags, radix: 16)) include bits this build does "
                        + "not implement (known: 0x\(String(FeatureFlags.known.rawValue, radix: 16)))")
            }
            declaredBiasBytes = take(UInt64.self)
            declaredBiasOffset = take(UInt64.self)
            let digestOffset = take(UInt64.self)
            let rawScaleDType = take(UInt32.self)
            guard let decodedScaleDType = TileScaleDType(rawValue: rawScaleDType) else {
                throw TileContainerError.malformedHeader(
                    "unknown scale element type \(rawScaleDType)")
            }
            scaleDType = decodedScaleDType
            let reserved = take(UInt32.self)
            guard reserved == 0 else {
                throw TileContainerError.malformedHeader(
                    "reserved header word is \(reserved), not zero")
            }

            guard flags.contains(.biases) == quantMode.hasBiases else {
                throw TileContainerError.malformedHeader(
                    "the biases feature flag disagrees with mode \(quantMode.name)")
            }

            if fileVersion >= 3 {
                scaleBlockRows = Int(take(UInt32.self))
                declaredPackedRegion = take(UInt64.self)
                declaredScaleRegion = take(UInt64.self)
                let reserved3 = take(UInt32.self)
                let reserved4 = take(UInt64.self)
                guard reserved3 == 0, reserved4 == 0 else {
                    throw TileContainerError.malformedHeader(
                        "reserved version-3 header words are not zero")
                }
            } else if flags.contains(.blockScales2D) || flags.contains(.paddedSubregions) {
                // The bits exist in this build, so the unknown-flag guard above
                // lets them through; a version-2 file has no field to back them.
                throw TileContainerError.malformedHeader(
                    "version \(fileVersion) sets a version-3 feature flag "
                        + "(0x\(String(flags.rawValue, radix: 16))) it has no field for")
            }

            let tableOffset = digestTableOffset(forVersion: fileVersion)
            if flags.contains(.tileDigests) {
                guard digestOffset == UInt64(tableOffset) else {
                    throw TileContainerError.malformedHeader(
                        "digest table offset \(digestOffset) != \(tableOffset)")
                }
                let end = tableOffset + Int(tileCount) * digestBytes
                guard end <= headerBytes else {
                    throw TileContainerError.malformedHeader(
                        "a digest table for \(tileCount) tiles does not fit in the header")
                }
                digests = (0..<Int(tileCount)).map { index in
                    let start = tableOffset + index * digestBytes
                    return Data(header[start..<(start + digestBytes)])
                }
            } else if digestOffset != 0 {
                throw TileContainerError.malformedHeader(
                    "a digest table offset is set but the feature flag is not")
            }
        }

        let geometry = TileGeometry(
            rows: Int(rows), cols: Int(cols), groupSize: Int(group), bits: bits,
            quantMode: quantMode, scaleDType: scaleDType, scaleBlockRows: scaleBlockRows)
        // Everything derivable is re-derived and compared, so a hand-edited or
        // corrupted header is rejected instead of steering reads off the rails.
        let expected = try layout(
            geometry: geometry, tileCount: Int(tileCount), seed: seed,
            contentKind: contentKind, tileDigests: digests)
        guard
            UInt64(geometry.packedBytes) == packedBytes,
            UInt64(geometry.scaleBytes) == scaleBytes,
            UInt64(expected.scaleOffsetInTile) == scaleOffsetInTile,
            UInt64(expected.tileStride) == tileStride,
            expected.firstTileOffset == firstTileOffset,
            UInt64(geometry.biasBytes) == declaredBiasBytes,
            UInt64(expected.biasOffsetInTile ?? 0) == declaredBiasOffset
        else {
            throw TileContainerError.malformedHeader(
                """
                header fields contradict the geometry: packed \(packedBytes) \
                (expected \(geometry.packedBytes)), scales \(scaleBytes) \
                (expected \(geometry.scaleBytes)), scaleOffset \(scaleOffsetInTile) \
                (expected \(expected.scaleOffsetInTile)), stride \(tileStride) \
                (expected \(expected.tileStride)), firstTile \(firstTileOffset) \
                (expected \(expected.firstTileOffset)), biasBytes \(declaredBiasBytes) \
                (expected \(geometry.biasBytes)), biasOffset \(declaredBiasOffset) \
                (expected \(expected.biasOffsetInTile ?? 0))
                """)
        }
        // Version-3 sub-region sizes are derivable too, so they get the same
        // treatment: recorded in the file for a reader that wants them without
        // recomputing, and rejected when they disagree with the geometry.
        if let declaredPackedRegion, let declaredScaleRegion {
            guard UInt64(expected.packedRegionBytes) == declaredPackedRegion,
                UInt64(expected.scaleRegionBytes) == declaredScaleRegion
            else {
                throw TileContainerError.malformedHeader(
                    "header sub-region sizes contradict the geometry: packed region "
                        + "\(declaredPackedRegion) (expected \(expected.packedRegionBytes)), "
                        + "scale region \(declaredScaleRegion) "
                        + "(expected \(expected.scaleRegionBytes))")
            }
        }
        // `layout` picks the lowest version that expresses the request, so a
        // version-2 file whose features all fit in version 1 is a writer bug or
        // a doctored header. Reading it would be harmless; letting it through
        // silently would make the compatibility policy unenforceable.
        guard expected.formatVersion == fileVersion else {
            throw TileContainerError.malformedHeader(
                "file declares version \(fileVersion) but its contents are a version "
                    + "\(expected.formatVersion) container")
        }
        guard expected.featureFlags == flags else {
            throw TileContainerError.malformedHeader(
                "feature flags 0x\(String(flags.rawValue, radix: 16)) contradict the geometry "
                    + "(expected 0x\(String(expected.featureFlags.rawValue, radix: 16)))")
        }
        return expected
    }

    // MARK: - Verification

    /// Compare a whole tile read into memory against the expected content.
    /// Returns nil when it matches, or a description of the first difference.
    ///
    /// Only meaningful for ``TileContentKind/syntheticDeterministic`` content —
    /// checkpoint bytes are not a function of the seed. Version 2 files say
    /// which they are, so this throws rather than reporting every byte of a
    /// checkpoint container as wrong. Version 1 files do not say, and are read
    /// as synthetic, which is what every version-1 caller of this function meant.
    public static func verifyTile(
        _ bytes: UnsafeRawBufferPointer,
        layout: Layout,
        tile: Int
    ) throws -> String? {
        guard layout.contentKind == .syntheticDeterministic else {
            throw TileContainerError.contentKindMismatch(
                "verifyTile regenerates content from the seed, and this container is "
                    + "\(layout.contentKind.name); check its tile digests instead")
        }
        guard bytes.count >= layout.tileStride else {
            return "tile \(tile): got \(bytes.count) bytes, expected \(layout.tileStride)"
        }
        var expectedPacked = [UInt8](repeating: 0, count: layout.geometry.packedBytes)
        expectedPacked.withUnsafeMutableBytes { fillPacked($0, layout: layout, tile: tile) }
        var expectedScales = [UInt8](repeating: 0, count: layout.geometry.scaleBytes)
        expectedScales.withUnsafeMutableBytes { fillScales($0, layout: layout, tile: tile) }

        for index in 0..<layout.geometry.packedBytes
        where bytes[index] != expectedPacked[index] {
            return
                "tile \(tile): packed byte \(index) is 0x\(String(bytes[index], radix: 16)), "
                + "expected 0x\(String(expectedPacked[index], radix: 16))"
        }
        for index in 0..<layout.geometry.scaleBytes {
            let observed = bytes[layout.scaleOffsetInTile + index]
            if observed != expectedScales[index] {
                return
                    "tile \(tile): scale byte \(index) is 0x\(String(observed, radix: 16)), "
                    + "expected 0x\(String(expectedScales[index], radix: 16))"
            }
        }
        return nil
    }

    /// Check a whole tile against the SHA-256 the header recorded for it.
    ///
    /// This is what replaces ``verifyTile(_:layout:tile:)`` for checkpoint
    /// content: the bytes cannot be regenerated, but they can be checked, and
    /// spec §9.3 asks for "checksums for independently transferable chunks" —
    /// a tile is exactly that, one `pread` into one slot.
    public static func verifyTileDigest(
        _ bytes: UnsafeRawBufferPointer,
        layout: Layout,
        tile: Int
    ) throws {
        guard let digests = layout.tileDigests else {
            throw TileContainerError.contentKindMismatch(
                "this container carries no per-tile digests")
        }
        guard tile >= 0, tile < digests.count else {
            throw TileContainerError.tileOutOfRange(index: tile, tileCount: layout.tileCount)
        }
        guard bytes.count >= layout.tileStride else {
            throw TileContainerError.truncated(
                path: "<memory>", expectedBytes: UInt64(layout.tileStride),
                actualBytes: UInt64(bytes.count))
        }
        let actual = SHA256.hash(
            UnsafeRawBufferPointer(start: bytes.baseAddress, count: layout.tileStride))
        guard actual == digests[tile] else {
            throw TileContainerError.digestMismatch(
                tile: tile,
                expected: digests[tile].map { String(format: "%02x", $0) }.joined(),
                actual: actual.map { String(format: "%02x", $0) }.joined())
        }
    }
}

/// The name this format had while it only described MXFP4.
///
/// Kept as an alias rather than renamed at every call site: version 1 files are
/// still version 1 files, the constants are the same constants, and a rename
/// sweep across fifteen files would bury the one change that matters.
public typealias MXFP4TileContainer = QuantizedTileContainer
