import Foundation

/// SHA-256 (FIPS 180-4), in plain Swift.
///
/// ## Why this is not CryptoKit
///
/// `StorageCore` links Foundation and Darwin and nothing else (spec §7.1), and
/// the CI step that builds this target alone is what keeps that true. CryptoKit
/// is neither, and it is Apple-only — while the tile container is explicitly a
/// cross-architecture format whose files are written on this Mac and validated
/// on an x86-64 runner (§9.3). A digest that only one of those two hosts can
/// compute would defeat the point of recording it.
///
/// This is not a general-purpose crypto primitive and is not offered as one. It
/// hashes container tiles so a checkpoint-derived container can be checked
/// rather than merely trusted; it is not constant-time and nothing here depends
/// on it being so. Correctness is pinned by the FIPS 180-4 vectors and by a
/// cross-language comparison against Python's `hashlib` on real container tiles.
public enum SHA256 {

    /// FIPS 180-4 §5.3.3.
    private static let initialState: [UInt32] = [
        0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
        0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
    ]

    /// FIPS 180-4 §4.2.2: the first 32 bits of the fractional parts of the cube
    /// roots of the first 64 primes.
    private static let k: [UInt32] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5,
        0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
        0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
        0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
        0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc,
        0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
        0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
        0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
        0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
        0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
        0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3,
        0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
        0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5,
        0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
        0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
        0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
    ]

    /// Incremental hasher, so a multi-gigabyte container is digested without
    /// being held in memory.
    public struct Hasher {
        private var state = SHA256.initialState
        private var block = [UInt8](repeating: 0, count: 64)
        private var blockFill = 0
        private var messageBytes: UInt64 = 0

        public init() {}

        public mutating func update(_ bytes: UnsafeRawBufferPointer) {
            guard let base = bytes.baseAddress else { return }
            messageBytes &+= UInt64(bytes.count)
            var consumed = 0
            // Top up a partial block first.
            if blockFill > 0 {
                let take = min(64 - blockFill, bytes.count)
                for index in 0..<take {
                    block[blockFill + index] = base.load(fromByteOffset: index, as: UInt8.self)
                }
                blockFill += take
                consumed = take
                // The whole input fitted inside the partial block. Returning here
                // is not an optimisation: the tail-copy below ends with
                // `blockFill = remaining`, computed from *this* call's buffer
                // alone, which would discard the bytes just topped up. Found by
                // the split-at-every-offset test at split 193 of 200 — three
                // whole blocks plus a one-byte remainder, then a seven-byte
                // second update.
                if blockFill < 64 { return }
                block.withUnsafeBytes { compress($0.baseAddress!, into: &state) }
                blockFill = 0
            }
            // Whole blocks straight from the caller's buffer.
            while bytes.count - consumed >= 64 {
                compress(base + consumed, into: &state)
                consumed += 64
            }
            // Keep the tail.
            let remaining = bytes.count - consumed
            for index in 0..<remaining {
                block[index] = base.load(fromByteOffset: consumed + index, as: UInt8.self)
            }
            blockFill = remaining
        }

        public mutating func update(_ bytes: [UInt8]) {
            bytes.withUnsafeBytes { update($0) }
        }

        public mutating func update(_ data: Data) {
            data.withUnsafeBytes { update($0) }
        }

        /// FIPS 180-4 §5.1.1 padding, then the big-endian state.
        public mutating func finalize() -> Data {
            let bitLength = messageBytes &* 8
            var padding: [UInt8] = [0x80]
            // Pad to 56 mod 64, leaving room for the 8-byte length.
            let target = (blockFill < 56) ? 56 : 120
            padding.append(contentsOf: [UInt8](repeating: 0, count: target - blockFill - 1))
            for shift in stride(from: 56, through: 0, by: -8) {
                padding.append(UInt8(truncatingIfNeeded: bitLength >> UInt64(shift)))
            }
            // `update` would re-count these into `messageBytes`; the length is
            // already fixed, so feed the blocks directly.
            var scratch = block[0..<blockFill] + padding[...]
            scratch.withUnsafeBytes { buffer in
                var offset = 0
                while offset + 64 <= buffer.count {
                    compress(buffer.baseAddress! + offset, into: &state)
                    offset += 64
                }
            }
            var digest = Data(capacity: 32)
            for word in state {
                digest.append(UInt8(truncatingIfNeeded: word >> 24))
                digest.append(UInt8(truncatingIfNeeded: word >> 16))
                digest.append(UInt8(truncatingIfNeeded: word >> 8))
                digest.append(UInt8(truncatingIfNeeded: word))
            }
            return digest
        }
    }

    /// One 64-byte block, FIPS 180-4 §6.2.2.
    private static func compress(_ block: UnsafeRawPointer, into state: inout [UInt32]) {
        var w = [UInt32](repeating: 0, count: 64)
        for index in 0..<16 {
            // Big-endian, loaded byte by byte so the pointer needs no alignment.
            let byte0 = UInt32(block.load(fromByteOffset: index * 4, as: UInt8.self))
            let byte1 = UInt32(block.load(fromByteOffset: index * 4 + 1, as: UInt8.self))
            let byte2 = UInt32(block.load(fromByteOffset: index * 4 + 2, as: UInt8.self))
            let byte3 = UInt32(block.load(fromByteOffset: index * 4 + 3, as: UInt8.self))
            w[index] = byte0 << 24 | byte1 << 16 | byte2 << 8 | byte3
        }
        for index in 16..<64 {
            let s0 =
                rotr(w[index - 15], 7) ^ rotr(w[index - 15], 18) ^ (w[index - 15] >> 3)
            let s1 = rotr(w[index - 2], 17) ^ rotr(w[index - 2], 19) ^ (w[index - 2] >> 10)
            w[index] = w[index - 16] &+ s0 &+ w[index - 7] &+ s1
        }

        var a = state[0]
        var b = state[1]
        var c = state[2]
        var d = state[3]
        var e = state[4]
        var f = state[5]
        var g = state[6]
        var h = state[7]

        for index in 0..<64 {
            let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            let ch = (e & f) ^ (~e & g)
            let temp1 = h &+ s1 &+ ch &+ k[index] &+ w[index]
            let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            let maj = (a & b) ^ (a & c) ^ (b & c)
            let temp2 = s0 &+ maj
            h = g
            g = f
            f = e
            e = d &+ temp1
            d = c
            c = b
            b = a
            a = temp1 &+ temp2
        }

        state[0] &+= a
        state[1] &+= b
        state[2] &+= c
        state[3] &+= d
        state[4] &+= e
        state[5] &+= f
        state[6] &+= g
        state[7] &+= h
    }

    @inline(__always)
    private static func rotr(_ value: UInt32, _ count: UInt32) -> UInt32 {
        (value >> count) | (value << (32 - count))
    }

    // MARK: - One-shot

    public static func hash(_ bytes: UnsafeRawBufferPointer) -> Data {
        var hasher = Hasher()
        hasher.update(bytes)
        return hasher.finalize()
    }

    public static func hash(_ bytes: [UInt8]) -> Data {
        bytes.withUnsafeBytes { hash($0) }
    }

    public static func hash(_ data: Data) -> Data {
        data.withUnsafeBytes { hash($0) }
    }

    /// Lowercase hex, the spelling every manifest in this repository uses.
    public static func hexString(_ digest: Data) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
