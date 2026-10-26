//! Scalar CPU reference: block decode and dot product for every kernel type.
//!
//! This is a direct port of the reference decode loops in llama.cpp
//! `ggml/src/ggml-quants.c` (MIT, commit 6050b46 — the qwen4exp fork, which is
//! the one that defines `Q2_0`):
//!
//! | kernel type | reference function |
//! |---|---|
//! | IQ2_XXS | `dequantize_row_iq2_xxs` |
//! | IQ2_XS | `dequantize_row_iq2_xs` |
//! | IQ2_S | `dequantize_row_iq2_s` |
//! | IQ3_XXS | `dequantize_row_iq3_xxs` |
//! | IQ3_S | `dequantize_row_iq3_s` |
//! | IQ4_NL | `dequantize_row_iq4_nl` |
//! | IQ4_XS | `dequantize_row_iq4_xs` |
//! | Q4_K | `dequantize_row_q4_K` + `get_scale_min_k4` |
//! | Q5_K | `dequantize_row_q5_K` + `get_scale_min_k4` |
//! | Q6_K | `dequantize_row_q6_K` |
//! | Q2_0 | `dequantize_row_q2_0` |
//! | BF16 / F32 | raw widening |
//!
//! Written in plain scalar form (byte-addressed, no SIMD, no tables in LDS) so
//! it is obviously faithful to the C and can be diffed against it by eye. The
//! dot product accumulates in `f64`; that is a reference value, not a
//! bit-identical GPU emulation — conformance uses `f32`-scale tolerances.

use crate::quant::QuantType;
use crate::tables::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KVALUES_IQ4NL};

/// Exact fp16 → fp32 widening (keeps subnormals, `±inf`, NaN).
///
/// Deliberately implemented with `f64` intermediates instead of a bit-twiddling
/// path: fp16 values are exactly representable in `f32`, so every finite value
/// round-trips bit-exactly, and the NaN/Inf cases are explicit.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f64;
    if exp == 0 {
        sign * (mant * 2f64.powi(-24)) as f32
    } else if exp == 0x1f {
        if mant == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * ((1.0 + mant / 1024.0) * 2f64.powi(exp - 15)) as f32
    }
}

/// Little-endian 16-bit word `w` of `b`.
#[inline]
fn w16(b: &[u8], w: usize) -> u16 {
    u16::from_le_bytes([b[2 * w], b[2 * w + 1]])
}

/// Byte `i` of `b`'s 16-bit-word space.
#[inline]
fn b16(b: &[u8], w: usize, i: usize) -> u8 {
    (w16(b, w) >> (8 * i)) as u8
}

/// Little-endian 32-bit word made of 16-bit words `w`, `w+1`.
#[inline]
fn w32(b: &[u8], w: usize) -> u32 {
    u32::from(w16(b, w)) | (u32::from(w16(b, w + 1)) << 16)
}

/// `GGML_FP16_TO_FP32` of the block scale at word `w`.
#[inline]
fn scale(b: &[u8], w: usize) -> f32 {
    f16_to_f32(w16(b, w))
}

/// `kmask_iq2xs` / sign bit `j`.
#[inline]
fn sign_bit(mask: u32, j: usize) -> f32 {
    if mask & (1u32 << j) != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Byte `i & 3` of a value as a 4-byte little-endian view (`aux8`).
#[inline]
fn aux8(v: u32, i: usize) -> u8 {
    (v >> (8 * (i & 3))) as u8
}

/// Signs table lookup: `ksigns_iq2xs[idx]`.
#[inline]
fn ksigns(idx: u32) -> u32 {
    u32::from(crate::tables::KSIGNS_IQ2XS[(idx & 127) as usize])
}

/// Decode one block of `ty`, appending its values to `out`.
///
/// `blk` must be exactly `ty.block_bytes()` bytes. Values are appended in
/// element order; the caller decides how many of them are in range.
pub fn dequant_block(ty: QuantType, blk: &[u8], out: &mut Vec<f32>) {
    assert_eq!(blk.len(), ty.block_bytes(), "block size for {ty}");
    match ty {
        QuantType::F32 => {
            for j in 0..8 {
                out.push(f32::from_bits(w32(blk, 2 * j)));
            }
        }
        QuantType::Bf16 => {
            for j in 0..8 {
                out.push(f32::from_bits(u32::from(w16(blk, j)) << 16));
            }
        }
        QuantType::F16 => {
            for j in 0..8 {
                out.push(f16_to_f32(w16(blk, j)));
            }
        }
        QuantType::Iq2Xxs => {
            let d = scale(blk, 0);
            for ib32 in 0..8 {
                let q0 = w32(blk, 1 + 4 * ib32);
                let q1 = w32(blk, 3 + 4 * ib32);
                let db = d * (0.5 + f32::from((q1 >> 28) as u8)) * 0.25;
                for l in 0..4 {
                    let grid = IQ2XXS_GRID[aux8(q0, l) as usize];
                    let signs = ksigns(q1 >> (7 * l as u32));
                    for j in 0..8 {
                        let g = ((grid >> (8 * j)) & 0xff) as f32;
                        out.push(db * g * sign_bit(signs, j));
                    }
                }
            }
        }
        QuantType::Iq2Xs => {
            let d = scale(blk, 0);
            for ib32 in 0..8 {
                let sc = b16(blk, 33 + ib32 / 2, ib32 & 1);
                let lo = f32::from(sc & 0xf);
                let hi = f32::from(sc >> 4);
                for l in 0..4 {
                    let qs = w16(blk, 1 + 4 * ib32 + l);
                    let db = d * (0.5 + if l / 2 == 0 { lo } else { hi }) * 0.25;
                    let grid = IQ2XS_GRID[(qs & 511) as usize];
                    let signs = ksigns(u32::from(qs >> 9));
                    for j in 0..8 {
                        let g = ((grid >> (8 * j)) & 0xff) as f32;
                        out.push(db * g * sign_bit(signs, j));
                    }
                }
            }
        }
        QuantType::Iq2S => {
            // d: word 0 | qs: bytes 2..33 | signs: bytes 34..65 | qh: bytes
            // 66..73 | scales: bytes 74..81.
            let d = scale(blk, 0);
            for ib32 in 0..8usize {
                let h = blk[66 + ib32];
                let sc = blk[74 + ib32];
                let lo = f32::from(sc & 0xf);
                let hi = f32::from(sc >> 4);
                for l in 0..4usize {
                    let u = 4 * ib32 + l;
                    let dl = d * (0.5 + if l / 2 == 0 { lo } else { hi }) * 0.25;
                    let idx = usize::from(blk[2 + u]) | ((usize::from(h) << (8 - 2 * l)) & 0x300);
                    let grid = IQ2S_GRID[idx];
                    let signs = u32::from(blk[34 + u]);
                    for j in 0..8 {
                        let g = ((grid >> (8 * j)) & 0xff) as f32;
                        out.push(dl * g * sign_bit(signs, j));
                    }
                }
            }
        }
        QuantType::Iq3Xxs => {
            let d = scale(blk, 0);
            for ib32 in 0..8 {
                let q1 = w32(blk, 33 + 2 * ib32);
                let db = d * (0.5 + f32::from((q1 >> 28) as u8)) * 0.5;
                for l in 0..4 {
                    let signs = ksigns(q1 >> (7 * l as u32));
                    // qs advances 8 bytes per 32-element group, so indices 2l,
                    // 2l+1 live in the 32-bit word pair +2*(l/2).
                    let qa = w32(blk, 1 + 4 * ib32 + 2 * (l / 2));
                    let g1 = IQ3XXS_GRID[aux8(qa, 2 * l) as usize];
                    let g2 = IQ3XXS_GRID[aux8(qa, 2 * l + 1) as usize];
                    // The C loop writes y[j] = grid1[j] and y[j+4] = grid2[j] in
                    // one pass, i.e. elements 0..3 are grid1, 4..7 are grid2.
                    for j in 0..4 {
                        let a = db * ((g1 >> (8 * j)) & 0xff) as f32;
                        out.push(a * sign_bit(signs, j));
                    }
                    for j in 0..4 {
                        let b = db * ((g2 >> (8 * j)) & 0xff) as f32;
                        out.push(b * sign_bit(signs, j + 4));
                    }
                }
            }
        }
        QuantType::Iq3S => {
            let d = scale(blk, 0);
            for ib32 in 0..8 {
                let qhb = b16(blk, 33 + ib32 / 2, ib32 & 1);
                let sc = b16(blk, 53 + ib32 / 4, (ib32 / 2) & 1);
                let nib = if ib32 & 1 == 0 { sc & 0xf } else { sc >> 4 };
                let db = d * (1.0 + 2.0 * f32::from(nib));
                for l in 0..4 {
                    let qa = w32(blk, 1 + 4 * ib32 + 2 * (l / 2));
                    let g1 = (aux8(qa, 2 * l) as usize)
                        | (((qhb as u32) << (8 - 2 * l as u32)) as usize & 256);
                    let g2 = (aux8(qa, 2 * l + 1) as usize)
                        | (((qhb as u32) << (7 - 2 * l as u32)) as usize & 256);
                    let sb = 4 * ib32 + l;
                    let signs = u32::from(b16(blk, 37 + sb / 2, sb & 1));
                    let grid1 = IQ3S_GRID[g1];
                    let grid2 = IQ3S_GRID[g2];
                    for j in 0..4 {
                        let a = db * ((grid1 >> (8 * j)) & 0xff) as f32;
                        out.push(a * sign_bit(signs, j));
                    }
                    for j in 0..4 {
                        let b = db * ((grid2 >> (8 * j)) & 0xff) as f32;
                        out.push(b * sign_bit(signs, j + 4));
                    }
                }
            }
        }
        QuantType::Iq4Nl => {
            let d = scale(blk, 0);
            let qs = &blk[2..18];
            for j in 0..16 {
                out.push(d * f32::from(KVALUES_IQ4NL[(qs[j] & 0xf) as usize]));
            }
            for j in 0..16 {
                out.push(d * f32::from(KVALUES_IQ4NL[(qs[j] >> 4) as usize]));
            }
        }
        QuantType::Iq4Xs => {
            // d: word 0 | scales_h: u16 at byte 2 | scales_l: bytes 4..7 |
            // qs: 16 bytes per 32-element group from byte 8.
            let d = scale(blk, 0);
            let sh = w16(blk, 1);
            for ib in 0..8usize {
                let sl = blk[4 + ib / 2];
                let ls = ((sl >> (4 * (ib % 2))) & 0xf) | ((((sh >> (2 * ib)) & 3) << 4) as u8);
                let dl = d * (f32::from(ls) - 32.0);
                let qs = &blk[8 + 16 * ib..8 + 16 * ib + 16];
                for j in 0..16 {
                    out.push(dl * f32::from(KVALUES_IQ4NL[(qs[j] & 0xf) as usize]));
                }
                for j in 0..16 {
                    out.push(dl * f32::from(KVALUES_IQ4NL[(qs[j] >> 4) as usize]));
                }
            }
        }
        QuantType::Q4K => {
            // dm: words 0..1 | scales: bytes 4..15 | qs: bytes 16..143.
            let d = f16_to_f32(w16(blk, 0));
            let dmin = f16_to_f32(w16(blk, 1));
            let qscale = |i: usize| b16(blk, 2 + i / 2, i & 1);
            for group in 0..4usize {
                let (mut sc1, mut m1, mut sc2, mut m2) = (0u8, 0u8, 0u8, 0u8);
                for (k, j) in [(0usize, 2 * group), (1, 2 * group + 1)] {
                    let (s, m) = if j < 4 {
                        (qscale(j) & 63, qscale(j + 4) & 63)
                    } else {
                        (
                            (qscale(j + 4) & 0xf) | ((qscale(j - 4) >> 6) << 4),
                            (qscale(j + 4) >> 4) | ((qscale(j) >> 6) << 4),
                        )
                    };
                    if k == 0 {
                        sc1 = s;
                        m1 = m;
                    } else {
                        sc2 = s;
                        m2 = m;
                    }
                }
                let (d1, dm1) = (d * f32::from(sc1), dmin * f32::from(m1));
                let (d2, dm2) = (d * f32::from(sc2), dmin * f32::from(m2));
                let qs = &blk[16 + 32 * group..16 + 32 * group + 32];
                for &b in qs {
                    out.push(d1 * f32::from(b & 0xf) - dm1);
                }
                for &b in qs {
                    out.push(d2 * f32::from(b >> 4) - dm2);
                }
            }
        }
        QuantType::Q5K => {
            let d = f16_to_f32(w16(blk, 0));
            let dmin = f16_to_f32(w16(blk, 1));
            let qscale = |i: usize| b16(blk, 2 + i / 2, i & 1);
            for group in 0..4usize {
                let (mut sc1, mut m1, mut sc2, mut m2) = (0u8, 0u8, 0u8, 0u8);
                for (k, j) in [(0usize, 2 * group), (1, 2 * group + 1)] {
                    let (s, m) = if j < 4 {
                        (qscale(j) & 63, qscale(j + 4) & 63)
                    } else {
                        (
                            (qscale(j + 4) & 0xf) | ((qscale(j - 4) >> 6) << 4),
                            (qscale(j + 4) >> 4) | ((qscale(j) >> 6) << 4),
                        )
                    };
                    if k == 0 {
                        sc1 = s;
                        m1 = m;
                    } else {
                        sc2 = s;
                        m2 = m;
                    }
                }
                let (d1, dm1) = (d * f32::from(sc1), dmin * f32::from(m1));
                let (d2, dm2) = (d * f32::from(sc2), dmin * f32::from(m2));
                let ql = 24 + group * 16; // 32 ql bytes per 64-element group
                let qh = 8; // qh is 32 bytes, indexed by l (0..32)
                let (u1, u2) = (1u8 << (2 * group), 2u8 << (2 * group));
                for l in 0..32 {
                    let q = (b16(blk, ql + l / 2, l & 1) & 0xf)
                        + if b16(blk, qh + l / 2, l & 1) & u1 != 0 {
                            16
                        } else {
                            0
                        };
                    out.push(d1 * f32::from(q) - dm1);
                }
                for l in 0..32 {
                    let q = (b16(blk, ql + l / 2, l & 1) >> 4)
                        + if b16(blk, qh + l / 2, l & 1) & u2 != 0 {
                            16
                        } else {
                            0
                        };
                    out.push(d2 * f32::from(q) - dm2);
                }
            }
        }
        QuantType::Q6K => {
            // ql: bytes 0..127 | qh: bytes 128..191 | scales: bytes 192..207
            // (int8) | d: word 104. Element order is quarter-major inside each
            // 128-element half, so the pushes follow the element index.
            let d = f16_to_f32(w16(blk, 104));
            for g in 0..2usize {
                for q in 0..4usize {
                    for l in 0..32usize {
                        let is = l / 16;
                        let base = g * 64 + l;
                        let nb = blk[if q % 2 == 0 { base } else { base + 32 }];
                        let nib = if q >= 2 { nb >> 4 } else { nb & 0xf };
                        let hi2 = (blk[128 + g * 32 + l] >> (2 * q)) & 3;
                        let qv = i16::from(nib | (hi2 << 4)) - 32;
                        let sc = f32::from(blk[192 + g * 8 + is + 2 * q] as i8);
                        out.push(d * sc * f32::from(qv));
                    }
                }
            }
        }
        QuantType::Q2_0 => {
            let d = scale(blk, 0);
            for u in 0..8usize {
                let w = w16(blk, 1 + u);
                for j in 0..8 {
                    let q = (w >> (2 * j)) & 3;
                    out.push((f32::from(q) - 1.0) * d);
                }
            }
        }
    }
}

/// Dequantise one packed row of `cols` values.
///
/// `packed` must hold `ceil(cols / block_elems) * block_bytes` bytes: a
/// trailing partial block is stored in full and only its first
/// `cols % block_elems` values are returned.
pub fn dequant_row(ty: QuantType, packed: &[u8], cols: usize) -> Vec<f32> {
    let qk = ty.block_elems();
    let bb = ty.block_bytes();
    let nblocks = cols.div_ceil(qk);
    let mut out = Vec::with_capacity(nblocks * qk);
    let mut buf = Vec::with_capacity(qk);
    for b in 0..nblocks {
        buf.clear();
        dequant_block(ty, &packed[b * bb..(b + 1) * bb], &mut buf);
        debug_assert_eq!(buf.len(), qk);
        let take = (cols - b * qk).min(qk);
        out.extend_from_slice(&buf[..take]);
    }
    out
}

/// One GEMV row: `sum_j dequant(w[j]) * x[j]`, accumulated in `f64`.
pub fn dot_row(ty: QuantType, packed: &[u8], cols: usize, x: &[f32]) -> f64 {
    assert!(x.len() >= cols, "x shorter than cols");
    let w = dequant_row(ty, packed, cols);
    let mut acc = 0.0f64;
    for j in 0..cols {
        acc += f64::from(w[j]) * f64::from(x[j]);
    }
    acc
}

/// Reference GEMV for a full matrix (`rows` packed rows of `cols` values).
pub fn gemv(ty: QuantType, packed: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let row_bytes = ty.block_bytes() * cols.div_ceil(ty.block_elems());
    (0..rows)
        .map(|r| dot_row(ty, &packed[r * row_bytes..], cols, x) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_widening_is_exact_for_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x8001), -2f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7c00) > 0.0);
        assert!(f16_to_f32(0xfc00).is_infinite() && f16_to_f32(0xfc00) < 0.0);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn q2_0_codes_map_to_minus_one_zero_plus_one_plus_two() {
        // 0b11100100 repeated: codes 0,1,2,3 -> -1,0,+1,+2
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        for byte in blk.iter_mut().skip(2) {
            *byte = 0b1110_0100;
        }
        let v = dequant_row(QuantType::Q2_0, &blk, 64);
        assert_eq!(v[0], -1.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 1.0);
        assert_eq!(v[3], 2.0);
        assert_eq!(v[63], 2.0);
    }

    #[test]
    fn iq4_nl_layout_is_low_nibbles_then_high_nibbles() {
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        for j in 0..16 {
            blk[2 + j] = (j as u8) | (((15 - j) as u8) << 4);
        }
        let v = dequant_row(QuantType::Iq4Nl, &blk, 32);
        assert_eq!(v[0], f32::from(KVALUES_IQ4NL[0]));
        assert_eq!(v[15], f32::from(KVALUES_IQ4NL[15]));
        assert_eq!(v[16], f32::from(KVALUES_IQ4NL[15]));
        assert_eq!(v[31], f32::from(KVALUES_IQ4NL[0]));
    }
}
