// The KV cache's f16 storage encoding, defined by this engine — not by the
// driver's `packHalf2x16`.
//
// WHY NOT `packHalf2x16`. Measured on the MI50 (gfx906, RADV 25.x): packing
// `0x3F5FDBFC` (0.8744) with `packHalf2x16` stores `0x3AFE`, while round-to-
// nearest-even gives `0x3AFF`. Every mismatching element of a 16 896-element
// sweep was exactly one ulp *low*, i.e. the driver's conversion **truncates**
// there. `unpackHalf2x16` is exact, so the whole discrepancy lands on the store
// side and is systematic (every |K| and |V| biased toward zero by up to 1 ulp =
// 2^-11 relative). That is a cache-precision decision, not a driver detail, so
// this file spells both directions out:
//
//   * `breccia_f32_to_f16_bits`: round-to-nearest-even, exponent rebias,
//     overflow -> inf, subnormal -> 0 (the model's K/V are O(1): a `head_dim`-wide
//     RMS-normed key or a raw value — documented rather than silently truncated),
//   * `breccia_f16_bits_to_f32`: exact (`e - 15` rebias).
//
// `qsa::cpu::to_f16` in `src/qsa/cpu.rs` is the same rule in Rust, and
// `tests/attn_conformance.rs::qsa_kv_cache_layout_and_f16_precision` holds the two
// equal element-for-element on both devices.

#ifndef BRECCIA_F16_STORAGE_GLSL
#define BRECCIA_F16_STORAGE_GLSL

/// Round-to-nearest-even f32 -> binary16 bits.
uint breccia_f32_to_f16_bits(float x) {
    uint b = floatBitsToUint(x);
    uint sign = (b >> 16) & 0x8000u;
    uint mag = b & 0x7fffffffu;
    if (mag >= 0x477ff000u) { // >= 65520 rounds to inf in binary16
        return sign | 0x7c00u;
    }
    int e = int((mag >> 23) & 0xffu) - 127;
    if (e > 15) {
        return sign | 0x7c00u;
    }
    if (e < -14) {
        return sign; // subnormal / zero: collapsed, see the header note
    }
    uint m = mag & 0x7fffffu;
    uint q = (m >> 13) | 0x400u; // implicit leading 1 + 10 mantissa bits
    uint rem = m & 0x1fffu;
    // Round to nearest, ties to even (the LSB of the 11-bit significand).
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u) == 1u)) {
        q += 1u;
    }
    if (q >= 0x800u) {
        q >>= 1;
        e += 1;
        if (e > 15) {
            return sign | 0x7c00u;
        }
    }
    return sign | (uint(e + 15) << 10) | (q & 0x3ffu);
}

/// Exact binary16 bits -> f32.
float breccia_f16_bits_to_f32(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u) {
        return uintBitsToFloat(sign); // zero or subnormal -> 0
    }
    if (e == 31u) {
        return uintBitsToFloat(sign | 0x7f800000u | (m << 13));
    }
    return uintBitsToFloat(sign | ((e + 112u) << 23) | (m << 13));
}

/// Pack two f32 values into one cache word (low half = `lo`).
uint breccia_pack_f16x2(float lo, float hi) {
    return (breccia_f32_to_f16_bits(hi) << 16) | breccia_f32_to_f16_bits(lo);
}

/// Read element `elem` (half index) out of a packed `uint` cache.
float breccia_load_f16(uint w, uint elem) {
    uint h = (elem & 1u) == 0u ? (w & 0xffffu) : (w >> 16);
    return breccia_f16_bits_to_f32(h);
}

#endif // BRECCIA_F16_STORAGE_GLSL
