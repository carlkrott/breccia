//! Fused scalar decode+dot: the portable fallback of the host expert lane.
//!
//! This is the same decode arithmetic as [`crate::reference`] (the port that
//! `tools/refcheck.py` diffs bit-for-bit against llama.cpp's `ggml-quants.c`),
//! but fused: each decoded value is multiplied straight into the activation
//! vector and accumulated in `f32`, so no weight block is ever materialised.
//! The reference materialises `f32` weights and sums in `f64`; that difference
//! is the only one, and it is bounded by `cols * f32::EPSILON` (measured ≤1e-6
//! relative for every type at `cols = 2560`, see `tests/cpu_experts.rs`).
//!
//! Everything here is safe code — no intrinsics, no `unsafe`. The AVX2 fast
//! path lives in [`super::avx2`] and is diffed against this module and against
//! [`crate::reference`] by the conformance suite.

use crate::quant::QuantType;
use crate::reference::f16_to_f32;
use crate::tables::{
    IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

/// Little-endian 16-bit word `w` of `b`.
#[inline]
fn w16(b: &[u8], w: usize) -> u16 {
    u16::from_le_bytes([b[2 * w], b[2 * w + 1]])
}

/// Byte `i` of `b`'s 16-bit-word space (`b16` in [`crate::reference`]).
#[inline]
fn b16(b: &[u8], w: usize, i: usize) -> u8 {
    (w16(b, w) >> (8 * i)) as u8
}

/// Little-endian 32-bit word made of 16-bit words `w`, `w+1`.
#[inline]
fn w32(b: &[u8], w: usize) -> u32 {
    u32::from(w16(b, w)) | (u32::from(w16(b, w + 1)) << 16)
}

/// `kmask_iq2xs` sign of element `j` for a [`KSIGNS_IQ2XS`] byte.
#[inline]
fn sgn(mask: u8, j: usize) -> f32 {
    if mask & (1 << j) != 0 {
        -1.0
    } else {
        1.0
    }
}

/// `ksigns_iq2xs[idx & 127]`.
#[inline]
fn ksign(idx: u32) -> u8 {
    KSIGNS_IQ2XS[(idx & 127) as usize]
}

/// One row of whole blocks: `cols` values packed as `block_bytes` blocks.
///
/// `x` must have at least as many elements as `row` has values.
///
/// The accumulator is `f64`. The decode is the reference decode, and keeping the
/// sum in `f64` makes this path *more* accurate than the vector path rather than
/// merely different, so it stays usable as a second oracle: with 2560 random-
/// magnitude terms the `f32` sum can lose 1.5e-5 relative to the `f64` value
/// through cancellation alone, which the 1e-5 conformance bar would flag as if
/// the decode were wrong.
pub(crate) fn dot(ty: QuantType, row: &[u8], x: &[f32]) -> f32 {
    (dot_f64(ty, row, x)) as f32
}

/// [`dot`] with the `f64` accumulator exposed (the conformance oracle uses it).
pub(crate) fn dot_f64(ty: QuantType, row: &[u8], x: &[f32]) -> f64 {
    match ty {
        QuantType::Iq2Xxs => iq2_xxs(row, x),
        QuantType::Iq2Xs => iq2_xs(row, x),
        QuantType::Iq2S => iq2_s(row, x),
        QuantType::Iq3Xxs => iq3_xxs(row, x),
        QuantType::Iq3S => iq3_s(row, x),
        QuantType::Iq4Nl => iq4_nl(row, x),
        QuantType::Iq4Xs => iq4_xs(row, x),
        QuantType::Q4K => q4_k(row, x),
        QuantType::Q5K => q5_k(row, x),
        QuantType::Q6K => q6_k(row, x),
        QuantType::Q2_0 => q2_0(row, x),
        QuantType::F16 => f16(row, x),
        QuantType::Bf16 => bf16(row, x),
        QuantType::F32 => f32_row(row, x),
    }
}

fn iq2_xxs(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(66) {
        let d = f16_to_f32(w16(blk, 0));
        for ib32 in 0..8 {
            let q0 = w32(blk, 1 + 4 * ib32);
            let q1 = w32(blk, 3 + 4 * ib32);
            let db = d * (0.5 + f32::from((q1 >> 28) as u8)) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[((q0 >> (8 * l)) & 0xff) as usize];
                let signs = ksign(q1 >> (7 * l));
                for j in 0..8 {
                    let g = ((grid >> (8 * j)) & 0xff) as f32;
                    acc += f64::from((db * g * sgn(signs, j)) * x[i]);
                    i += 1;
                }
            }
        }
    }
    acc
}

fn iq2_xs(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(74) {
        let d = f16_to_f32(w16(blk, 0));
        for ib32 in 0..8 {
            let sc = b16(blk, 33 + ib32 / 2, ib32 & 1);
            let lo = f32::from(sc & 0xf);
            let hi = f32::from(sc >> 4);
            for l in 0..4 {
                let qs = w16(blk, 1 + 4 * ib32 + l);
                let db = d * (0.5 + if l / 2 == 0 { lo } else { hi }) * 0.25;
                let grid = IQ2XS_GRID[(qs & 511) as usize];
                let signs = ksign(u32::from(qs >> 9));
                for j in 0..8 {
                    let g = ((grid >> (8 * j)) & 0xff) as f32;
                    acc += f64::from((db * g * sgn(signs, j)) * x[i]);
                    i += 1;
                }
            }
        }
    }
    acc
}

fn iq2_s(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(82) {
        let d = f16_to_f32(w16(blk, 0));
        for ib32 in 0..8 {
            let h = blk[66 + ib32];
            let sc = blk[74 + ib32];
            let lo = f32::from(sc & 0xf);
            let hi = f32::from(sc >> 4);
            for l in 0..4 {
                let u = 4 * ib32 + l;
                let dl = d * (0.5 + if l / 2 == 0 { lo } else { hi }) * 0.25;
                let idx = usize::from(blk[2 + u]) | ((usize::from(h) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx];
                let signs = blk[34 + u];
                for j in 0..8 {
                    let g = ((grid >> (8 * j)) & 0xff) as f32;
                    acc += f64::from((dl * g * sgn(signs, j)) * x[i]);
                    i += 1;
                }
            }
        }
    }
    acc
}

fn iq3_xxs(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(98) {
        let d = f16_to_f32(w16(blk, 0));
        for ib32 in 0..8 {
            let q1 = w32(blk, 33 + 2 * ib32);
            let db = d * (0.5 + f32::from((q1 >> 28) as u8)) * 0.5;
            for l in 0..4 {
                let signs = ksign(q1 >> (7 * l));
                let qa = w32(blk, 1 + 4 * ib32 + 2 * (l / 2));
                let g1 = IQ3XXS_GRID[(qa >> (8 * ((2 * l) & 3))) as u8 as usize];
                let g2 = IQ3XXS_GRID[(qa >> (8 * ((2 * l) & 3) + 8)) as u8 as usize];
                for j in 0..4 {
                    let a = db * ((g1 >> (8 * j)) & 0xff) as f32;
                    acc += f64::from((a * sgn(signs, j)) * x[i]);
                    i += 1;
                }
                for j in 0..4 {
                    let b = db * ((g2 >> (8 * j)) & 0xff) as f32;
                    acc += f64::from((b * sgn(signs, j + 4)) * x[i]);
                    i += 1;
                }
            }
        }
    }
    acc
}

fn iq3_s(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(110) {
        let d = f16_to_f32(w16(blk, 0));
        for ib32 in 0..8 {
            let qhb = b16(blk, 33 + ib32 / 2, ib32 & 1);
            let sc = b16(blk, 53 + ib32 / 4, (ib32 / 2) & 1);
            let nib = if ib32 & 1 == 0 { sc & 0xf } else { sc >> 4 };
            let db = d * (1.0 + 2.0 * f32::from(nib));
            for l in 0..4 {
                let qa = w32(blk, 1 + 4 * ib32 + 2 * (l / 2));
                let g1 = (qa >> (8 * ((2 * l) & 3))) as u8 as usize
                    | (((qhb as usize) << (8 - 2 * l)) & 256);
                let g2 = (qa >> (8 * ((2 * l) & 3) + 8)) as u8 as usize
                    | (((qhb as usize) << (7 - 2 * l)) & 256);
                let sb = 4 * ib32 + l;
                let signs = b16(blk, 37 + sb / 2, sb & 1);
                let grid1 = IQ3S_GRID[g1];
                let grid2 = IQ3S_GRID[g2];
                for j in 0..4 {
                    let a = db * ((grid1 >> (8 * j)) & 0xff) as f32;
                    acc += f64::from((a * sgn(signs, j)) * x[i]);
                    i += 1;
                }
                for j in 0..4 {
                    let b = db * ((grid2 >> (8 * j)) & 0xff) as f32;
                    acc += f64::from((b * sgn(signs, j + 4)) * x[i]);
                    i += 1;
                }
            }
        }
    }
    acc
}

fn iq4_nl(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(18) {
        let d = f16_to_f32(w16(blk, 0));
        for j in 0..16 {
            acc += f64::from((d * f32::from(KVALUES_IQ4NL[(blk[2 + j] & 0xf) as usize])) * x[i]);
            i += 1;
        }
        for j in 0..16 {
            acc += f64::from((d * f32::from(KVALUES_IQ4NL[(blk[2 + j] >> 4) as usize])) * x[i]);
            i += 1;
        }
    }
    acc
}

fn iq4_xs(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(136) {
        let d = f16_to_f32(w16(blk, 0));
        let sh = w16(blk, 1);
        for ib in 0..8usize {
            let sl = blk[4 + ib / 2];
            let ls = ((sl >> (4 * (ib % 2))) & 0xf) | ((((sh >> (2 * ib)) & 3) << 4) as u8);
            let dl = d * (f32::from(ls) - 32.0);
            for j in 0..16 {
                acc += f64::from(
                    (dl * f32::from(KVALUES_IQ4NL[(blk[8 + 16 * ib + j] & 0xf) as usize])) * x[i],
                );
                i += 1;
            }
            for j in 0..16 {
                acc += f64::from(
                    (dl * f32::from(KVALUES_IQ4NL[(blk[8 + 16 * ib + j] >> 4) as usize])) * x[i],
                );
                i += 1;
            }
        }
    }
    acc
}

/// `get_scale_min_k4` from `ggml-quants.c`: `(scale, min)` pair `j` of 8.
#[inline]
fn scale_min_k4(blk: &[u8], j: usize) -> (u8, u8) {
    let q = |i: usize| b16(blk, 2 + i / 2, i & 1);
    if j < 4 {
        (q(j) & 63, q(j + 4) & 63)
    } else {
        (
            (q(j + 4) & 0xf) | ((q(j - 4) >> 6) << 4),
            (q(j + 4) >> 4) | ((q(j) >> 6) << 4),
        )
    }
}

fn q4_k(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(144) {
        let d = f16_to_f32(w16(blk, 0));
        let dmin = f16_to_f32(w16(blk, 1));
        for group in 0..4usize {
            let (sc1, m1) = scale_min_k4(blk, 2 * group);
            let (sc2, m2) = scale_min_k4(blk, 2 * group + 1);
            let (d1, dm1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (d2, dm2) = (d * f32::from(sc2), dmin * f32::from(m2));
            let qs = &blk[16 + 32 * group..16 + 32 * group + 32];
            for &b in qs {
                acc += f64::from((d1 * f32::from(b & 0xf) - dm1) * x[i]);
                i += 1;
            }
            for &b in qs {
                acc += f64::from((d2 * f32::from(b >> 4) - dm2) * x[i]);
                i += 1;
            }
        }
    }
    acc
}

fn q5_k(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(176) {
        let d = f16_to_f32(w16(blk, 0));
        let dmin = f16_to_f32(w16(blk, 1));
        for group in 0..4usize {
            let (sc1, m1) = scale_min_k4(blk, 2 * group);
            let (sc2, m2) = scale_min_k4(blk, 2 * group + 1);
            let (d1, dm1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (d2, dm2) = (d * f32::from(sc2), dmin * f32::from(m2));
            let ql = 24 + group * 16;
            let (u1, u2) = (1u8 << (2 * group), 2u8 << (2 * group));
            for l in 0..32 {
                let q = (b16(blk, ql + l / 2, l & 1) & 0xf)
                    + if b16(blk, 8 + l / 2, l & 1) & u1 != 0 {
                        16
                    } else {
                        0
                    };
                acc += f64::from((d1 * f32::from(q) - dm1) * x[i]);
                i += 1;
            }
            for l in 0..32 {
                let q = (b16(blk, ql + l / 2, l & 1) >> 4)
                    + if b16(blk, 8 + l / 2, l & 1) & u2 != 0 {
                        16
                    } else {
                        0
                    };
                acc += f64::from((d2 * f32::from(q) - dm2) * x[i]);
                i += 1;
            }
        }
    }
    acc
}

fn q6_k(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(210) {
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
                    acc += f64::from((d * sc * f32::from(qv)) * x[i]);
                    i += 1;
                }
            }
        }
    }
    acc
}

fn q2_0(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    let mut i = 0usize;
    for blk in row.chunks_exact(18) {
        let d = f16_to_f32(w16(blk, 0));
        for u in 0..8usize {
            let w = w16(blk, 1 + u);
            for j in 0..8 {
                let q = (w >> (2 * j)) & 3;
                acc += f64::from(((f32::from(q) - 1.0) * d) * x[i]);
                i += 1;
            }
        }
    }
    acc
}

fn f16(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    for (i, blk) in row.chunks_exact(16).enumerate() {
        for j in 0..8 {
            acc += f64::from((f16_to_f32(w16(blk, j))) * x[8 * i + j]);
        }
    }
    acc
}

fn bf16(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    for (i, blk) in row.chunks_exact(16).enumerate() {
        for j in 0..8 {
            acc += f64::from((f32::from_bits(u32::from(w16(blk, j)) << 16)) * x[8 * i + j]);
        }
    }
    acc
}

fn f32_row(row: &[u8], x: &[f32]) -> f64 {
    let mut acc = 0f64;
    for (i, blk) in row.chunks_exact(32).enumerate() {
        for j in 0..8 {
            acc += f64::from((f32::from_bits(w32(blk, 2 * j))) * x[8 * i + j]);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference;

    /// The fused scalar path must decode exactly like the materialising
    /// reference: compare one block's contribution by handing it a `1` at a
    /// single position and `0` elsewhere.
    #[test]
    fn fused_scalar_decode_agrees_with_the_reference_value_by_value() {
        for ty in QuantType::ALL {
            let qk = ty.block_elems();
            let bb = ty.block_bytes();
            // Deterministic, valid packed bytes (scales re-written into a sane
            // range so nothing is NaN/inf).
            let mut blk = vec![0u8; bb];
            let mut s = 0x1234_5678u32;
            for b in blk.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *b = s as u8;
            }
            let scale = 0x3555u16; // fp16 0.33325
            for &off in crate::real::scale_offsets(ty) {
                blk[off..off + 2].copy_from_slice(&scale.to_le_bytes());
            }
            let want = reference::dequant_row(ty, &blk, qk);
            for j in 0..qk {
                let mut x = vec![0f32; qk];
                x[j] = 1.0;
                let got = dot(ty, &blk, &x);
                // f32 accumulation of one product: exact up to the product's
                // own rounding, both derived from the same f32 weight.
                assert!(
                    (got - want[j]).abs() <= 1e-6 * want[j].abs().max(1.0),
                    "{ty} element {j}: fused {got} vs reference {}",
                    want[j]
                );
            }
        }
    }

    #[test]
    fn repeated_runs_are_bit_identical() {
        for ty in QuantType::ALL {
            let cols: usize = 640;
            let bb = ty.block_bytes() * cols.div_ceil(ty.block_elems());
            let mut row = vec![0u8; bb];
            for (i, b) in row.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(37).wrapping_add(11);
            }
            // The dot walks whole blocks only, so the activation vector must be
            // as long as the packed row's value count.
            let nvals = row.len() / ty.block_bytes() * ty.block_elems();
            let x: Vec<f32> = (0..nvals).map(|i| (i as f32 * 0.017).sin()).collect();
            let a = dot(ty, &row, &x);
            let b = dot(ty, &row, &x);
            assert_eq!(a.to_bits(), b.to_bits(), "{ty}");
        }
    }
}
