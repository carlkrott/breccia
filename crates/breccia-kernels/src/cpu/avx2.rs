//! AVX2 + FMA fused decode+dot kernels for the host expert lane.
//!
//! **All `unsafe` in `breccia-kernels`' CPU lane lives in this file.** Every
//! kernel is a `#[target_feature(enable = "avx2,fma")]` function and is only
//! reachable through [`super::dot_row_with`], which checks
//! `is_x86_feature_detected!("avx2")` once per GEMV (see [`super::path`]); the
//! safe wrapper [`dot`] therefore cannot be called on a CPU that lacks the
//! features. `lib.rs` sets `#![forbid(unsafe_op_in_unsafe_fn)]`, so each kernel
//! body is one explicit `unsafe { }` block with a `// SAFETY:` note.
//!
//! # Shape of the code
//!
//! Every type decodes in units of eight consecutive elements, exactly like the
//! GLSL decode units (`shaders/blocks.glsl`) and the scalar reference
//! ([`crate::reference`]) — a 256-element super-block is 32 units, `IQ4_NL` is
//! 4 units, `Q2_0` is 8, and the unquantised types use a virtual 8-element
//! block. One unit costs:
//!
//! * one 8-byte load + `vpmovzxbd` + `vcvtdq2ps` to turn integer magnitudes
//!   into 8 `f32` lanes (the grids/values are exact small integers, so this is
//!   lossless), or a register shift/mask for the K-quants;
//! * one sign multiply (`IQ*`) or one scale/min transform (`K`-quants,
//!   `Q2_0`), which is *data* multiplied into the value — never a float→int
//!   conversion, so a NaN scale degrades the result instead of indexing memory
//!   out of bounds (`docs/phase-3/02` §4.3);
//! * one `vfmadd231ps` against the activation vector.
//!
//! Two accumulators alternate by unit so the FMA dependency chain does not
//! serialize the pipeline; the lane sums are combined in `f64` at the end,
//! which is why a 2560-element dot stays within 1e-6 relative of the `f64`
//! reference dot (`tests/cpu_experts.rs`).

use std::arch::x86_64::*;

use crate::quant::QuantType;
use crate::reference::f16_to_f32;
use crate::tables::{
    IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

/// Sign-bit masks (`0x8000_0000` per negative lane) for each `ksigns_iq2xs` byte.
///
/// The decode contract multiplies grid magnitudes by `±d`. Flipping the *sign
/// bit* of the already-scaled value with a `vxorps` costs one instruction and is
/// exact for every value (including a negative scale), whereas multiplying by a
/// `±1` table costs a load plus a multiply.
static SIGNBITS: [[u32; 8]; 256] = {
    let mut t = [[0u32; 8]; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut j = 0usize;
        while j < 8 {
            t[i][j] = if (i >> j) & 1 == 0 { 0 } else { 0x8000_0000 };
            j += 1;
        }
        i += 1;
    }
    t
};

/// [`SIGNBITS`] indexed by the *raw* `ksigns_iq2xs` index (0..128), i.e. with the
/// `KSIGNS_IQ2XS` permutation folded in at compile time.
///
/// The reference does `ksigns_iq2xs[idx]` and then tests its bits, which on the
/// SIMD path would be two dependent loads per unit of eight. Folding it removes
/// one load from the inner loop of every `ksigns`-carrying type.
static SIGNBITS_BY_IDX: [[u32; 8]; 128] = {
    let mut t = [[0u32; 8]; 128];
    let mut i = 0usize;
    while i < 128 {
        let mask = KSIGNS_IQ2XS[i] as usize;
        let mut j = 0usize;
        while j < 8 {
            t[i][j] = if (mask >> j) & 1 == 0 { 0 } else { 0x8000_0000 };
            j += 1;
        }
        i += 1;
    }
    t
};

/// Grid magnitudes widened to `f32` at compile time.
///
/// The grids hold small integers (≤127), so the widening is exact and the
/// decode result is bit-identical to widening at run time — but it costs one
/// 32-byte load instead of a load plus `vpmovzxbd` plus `vcvtdq2ps`, which was
/// the difference between 2.6 and 3.4 GB/s per core on this host
/// (`docs/phase-3/08-cpu-expert-lane.md` §3).
static IQ2XXS_F32: [[f32; 8]; 256] = {
    let mut t = [[0f32; 8]; 256];
    let mut i = 0usize;
    while i < 256 {
        let g = IQ2XXS_GRID[i].to_le_bytes();
        let mut j = 0usize;
        while j < 8 {
            t[i][j] = g[j] as f32;
            j += 1;
        }
        i += 1;
    }
    t
};

static IQ2XS_F32: [[f32; 8]; 512] = {
    let mut t = [[0f32; 8]; 512];
    let mut i = 0usize;
    while i < 512 {
        let g = IQ2XS_GRID[i].to_le_bytes();
        let mut j = 0usize;
        while j < 8 {
            t[i][j] = g[j] as f32;
            j += 1;
        }
        i += 1;
    }
    t
};

static IQ2S_F32: [[f32; 8]; 1024] = {
    let mut t = [[0f32; 8]; 1024];
    let mut i = 0usize;
    while i < 1024 {
        let g = IQ2S_GRID[i].to_le_bytes();
        let mut j = 0usize;
        while j < 8 {
            t[i][j] = g[j] as f32;
            j += 1;
        }
        i += 1;
    }
    t
};

/// The IQ3 grids carry four magnitudes per code; a unit of eight is two codes.
static IQ3XXS_F32: [[f32; 4]; 256] = {
    let mut t = [[0f32; 4]; 256];
    let mut i = 0usize;
    while i < 256 {
        let g = IQ3XXS_GRID[i].to_le_bytes();
        let mut j = 0usize;
        while j < 4 {
            t[i][j] = g[j] as f32;
            j += 1;
        }
        i += 1;
    }
    t
};

static IQ3S_F32: [[f32; 4]; 512] = {
    let mut t = [[0f32; 4]; 512];
    let mut i = 0usize;
    while i < 512 {
        let g = IQ3S_GRID[i].to_le_bytes();
        let mut j = 0usize;
        while j < 4 {
            t[i][j] = g[j] as f32;
            j += 1;
        }
        i += 1;
    }
    t
};

#[inline]
fn u16le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Safe wrapper over the AVX2 dispatcher.
///
/// # Safety
///
/// The caller must have verified `is_x86_feature_detected!("avx2")` and
/// `("fma")` and `("f16c")` (they ship together on every AVX2-era part; the
/// dispatch in [`super::path`] checks all three).
pub(super) fn dot(ty: QuantType, row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: guaranteed by the caller (see the doc comment); every path below
    // is one of the `#[target_feature]` kernels.
    unsafe { dispatch(ty, row, x) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dispatch(ty: QuantType, row: &[u8], x: &[f32]) -> f32 {
    match ty {
        // SAFETY: the caller of `dot` checked the features this function
        // declares, so every kernel below (same features) is legal here.
        QuantType::Iq2Xxs => unsafe { iq2_xxs(row, x) },
        QuantType::Iq2Xs => unsafe { iq2_xs(row, x) },
        QuantType::Iq2S => unsafe { iq2_s(row, x) },
        QuantType::Iq3Xxs => unsafe { iq3_xxs(row, x) },
        QuantType::Iq3S => unsafe { iq3_s(row, x) },
        QuantType::Iq4Nl => unsafe { iq4_nl(row, x) },
        QuantType::Iq4Xs => unsafe { iq4_xs(row, x) },
        QuantType::Q4K => unsafe { q4_k(row, x) },
        QuantType::Q5K => unsafe { q5_k(row, x) },
        QuantType::Q6K => unsafe { q6_k(row, x) },
        QuantType::Q2_0 => unsafe { q2_0(row, x) },
        QuantType::F16 => unsafe { f16_row(row, x) },
        QuantType::Bf16 => unsafe { bf16_row(row, x) },
        QuantType::F32 => unsafe { f32_row(row, x) },
    }
}

// ---------------------------------------------------------------------------
// Shared unit helpers
// ---------------------------------------------------------------------------

/// Sign-bit mask for a `ksigns_iq2xs` byte.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sign_bits(mask: u8) -> __m256 {
    // SAFETY: `mask` indexes the 256-entry table, so the load is 32 bytes
    // inside `SIGNBITS`.
    unsafe { _mm256_castsi256_ps(_mm256_loadu_si256(SIGNBITS[mask as usize].as_ptr().cast())) }
}

/// Sign-bit mask for a raw `ksigns_iq2xs` index (already unmasked by the caller).
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sign_bits_by_idx(idx: u32) -> __m256 {
    // SAFETY: `idx` is masked to 0..127 by the caller, so the load is 32 bytes
    // inside the 128-entry table.
    unsafe {
        _mm256_castsi256_ps(_mm256_loadu_si256(
            SIGNBITS_BY_IDX[idx as usize].as_ptr().cast(),
        ))
    }
}

/// `±(grid · scale)` for eight magnitudes, as one 32-byte grid load, one
/// multiply and one sign-bit flip.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn signed_grid(g: &[f32; 8], scale: f32, mask: u8) -> __m256 {
    // SAFETY: both operands are 32/8-byte loads from tables indexed by an
    // 8-bit mask; no pointer arithmetic escapes the tables.
    unsafe {
        _mm256_xor_ps(
            _mm256_mul_ps(_mm256_loadu_ps(g.as_ptr()), _mm256_set1_ps(scale)),
            sign_bits(mask),
        )
    }
}

/// `signed_grid` with the scale already broadcast (hoisted out of the inner
/// loop where consecutive units share the block scale).
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn signed_grid_v(g: &[f32; 8], scale: __m256, sign: __m256) -> __m256 {
    // SAFETY: one 32-byte load from an 8-entry table.
    unsafe { _mm256_xor_ps(_mm256_mul_ps(_mm256_loadu_ps(g.as_ptr()), scale), sign) }
}

/// Eight magnitudes from two four-entry IQ3 grids in one vector.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn signed_grid4(lo: &[f32; 4], hi: &[f32; 4], scale: f32, mask: u8) -> __m256 {
    // SAFETY: two 16-byte loads from 4-entry tables, combined with
    // `insertf128`; the result is 8 lanes and the sign table is 8-bit indexed.
    unsafe {
        let g = join4(lo, hi);
        _mm256_xor_ps(_mm256_mul_ps(g, _mm256_set1_ps(scale)), sign_bits(mask))
    }
}

/// [`signed_grid4`] with a precomputed sign vector.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn signed_grid4_v(lo: &[f32; 4], hi: &[f32; 4], scale: f32, sign: __m256) -> __m256 {
    // SAFETY: as `signed_grid4`.
    unsafe {
        let g = join4(lo, hi);
        _mm256_xor_ps(_mm256_mul_ps(g, _mm256_set1_ps(scale)), sign)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn join4(lo: &[f32; 4], hi: &[f32; 4]) -> __m256 {
    // SAFETY: two 16-byte loads from 4-entry tables.
    unsafe {
        let l = _mm_loadu_ps(lo.as_ptr());
        let h = _mm_loadu_ps(hi.as_ptr());
        _mm256_insertf128_ps(_mm256_castps128_ps256(l), h, 1)
    }
}

/// `acc += (q * scale − min) * x`
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn kq_fma(acc: __m256, q: __m256i, scale: __m256, min: __m256, xp: *const f32) -> __m256 {
    // SAFETY: `xp` points at 8 readable activation elements (the caller's loop
    // is bounded by the block count of the row).
    unsafe {
        let v = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q), scale), min);
        _mm256_fmadd_ps(v, _mm256_loadu_ps(xp), acc)
    }
}

/// `acc += (v * scale) * x` for a value vector.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn v_fma(acc: __m256, v: __m256, xp: *const f32) -> __m256 {
    // SAFETY: `xp` points at 8 readable activation elements.
    unsafe { _mm256_fmadd_ps(v, _mm256_loadu_ps(xp), acc) }
}

/// Combine the two 8-lane accumulators into one `f32` scalar.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn finish(a0: __m256, a1: __m256) -> f32 {
    let s = _mm256_add_ps(a0, a1);
    let lo = _mm256_castps256_ps128(s);
    let hi = _mm256_extractf128_ps(s, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    _mm_cvtss_f32(s)
}

// ---------------------------------------------------------------------------
// IQ2_XXS
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq2_xxs(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: `row` holds whole 66-byte blocks, `x` at least as many elements
    // as `row` has values (guaranteed by `super::dot_row_with`); every load is
    // inside those bounds and every grid/sign index is a packed-bit code masked
    // to its table.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        for blk in row.chunks_exact(66) {
            let d = f16_to_f32(u16le(blk, 0));
            for ib in 0..8 {
                let q0 = u32le(blk, 2 + 8 * ib);
                let q1 = u32le(blk, 6 + 8 * ib);
                // One scale for all four units of this 32-element group, so the
                // broadcast is hoisted out of the inner loop.
                let dbr = _mm256_set1_ps(d * (0.5 + ((q1 >> 28) as u8) as f32) * 0.25);
                for l in 0..4 {
                    let g = &IQ2XXS_F32[((q0 >> (8 * l)) & 0xff) as usize];
                    let v = signed_grid_v(g, dbr, sign_bits_by_idx((q1 >> (7 * l)) & 127));
                    if l & 1 == 0 {
                        a0 = v_fma(a0, v, x.as_ptr().add(i));
                    } else {
                        a1 = v_fma(a1, v, x.as_ptr().add(i));
                    }
                    i += 8;
                }
            }
        }
        finish(a0, a1)
    }
}

// ---------------------------------------------------------------------------
// IQ2_XS
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq2_xs(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 74-byte blocks and an in-range activation vector; the grid
    // index is `qs & 511` into a 512-entry table.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        for blk in row.chunks_exact(74) {
            let d = f16_to_f32(u16le(blk, 0));
            for ib in 0..8 {
                let sc = blk[66 + ib];
                let s0 = _mm256_set1_ps(d * (0.5 + f32::from(sc & 0xf)) * 0.25);
                let s1 = _mm256_set1_ps(d * (0.5 + f32::from(sc >> 4)) * 0.25);
                for l in 0..4 {
                    let qs = u16le(blk, 2 + 8 * ib + 2 * l);
                    let g = &IQ2XS_F32[(qs & 511) as usize];
                    let scale = if l / 2 == 0 { s0 } else { s1 };
                    let v = signed_grid_v(g, scale, sign_bits_by_idx(u32::from(qs >> 9) & 127));
                    if l & 1 == 0 {
                        a0 = v_fma(a0, v, x.as_ptr().add(i));
                    } else {
                        a1 = v_fma(a1, v, x.as_ptr().add(i));
                    }
                    i += 8;
                }
            }
        }
        finish(a0, a1)
    }
}

// ---------------------------------------------------------------------------
// IQ2_S
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq2_s(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 82-byte blocks; the grid index is `qs | (qh bits) << 8`,
    // ≤1023 into a 1024-entry table.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        for blk in row.chunks_exact(82) {
            let d = f16_to_f32(u16le(blk, 0));
            for ib in 0..8 {
                let h = usize::from(blk[66 + ib]);
                let sc = blk[74 + ib];
                let lo = f32::from(sc & 0xf);
                let hi = f32::from(sc >> 4);
                for l in 0..4 {
                    let u = 4 * ib + l;
                    let dbr = d * (0.5 + if l / 2 == 0 { lo } else { hi }) * 0.25;
                    let idx = usize::from(blk[2 + u]) | ((h << (8 - 2 * l)) & 0x300);
                    let g = &IQ2S_F32[idx];
                    let v = signed_grid(g, dbr, blk[34 + u]);
                    if u & 1 == 0 {
                        a0 = v_fma(a0, v, x.as_ptr().add(i));
                    } else {
                        a1 = v_fma(a1, v, x.as_ptr().add(i));
                    }
                    i += 8;
                }
            }
        }
        finish(a0, a1)
    }
}

// ---------------------------------------------------------------------------
// IQ3_XXS / IQ3_S  (four magnitudes from each of two grids per unit of 8)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq3_xxs(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 98-byte blocks; the code bytes live in bytes 2..66, the
    // scale words in 66..98, and both grid codes are single bytes into a
    // 256-entry table.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        for blk in row.chunks_exact(98) {
            let d = f16_to_f32(u16le(blk, 0));
            for ib in 0..8 {
                let q1 = u32le(blk, 66 + 4 * ib);
                let dbr = d * (0.5 + ((q1 >> 28) as u8) as f32) * 0.5;
                for l in 0..4 {
                    let base = 2 + 8 * ib + 4 * (l / 2) + ((2 * l) & 3);
                    let v = signed_grid4_v(
                        &IQ3XXS_F32[blk[base] as usize],
                        &IQ3XXS_F32[blk[base + 1] as usize],
                        dbr,
                        sign_bits_by_idx((q1 >> (7 * l)) & 127),
                    );
                    if l & 1 == 0 {
                        a0 = v_fma(a0, v, x.as_ptr().add(i));
                    } else {
                        a1 = v_fma(a1, v, x.as_ptr().add(i));
                    }
                    i += 8;
                }
            }
        }
        finish(a0, a1)
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq3_s(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 110-byte blocks (d 0..2, qs 2..66, qh 66..74, signs
    // 74..106, scales 106..110); grid codes are ≤511 into a 512-entry table.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        for blk in row.chunks_exact(110) {
            let d = f16_to_f32(u16le(blk, 0));
            for ib in 0..8 {
                let qhb = usize::from(blk[66 + ib]);
                // `scales` is 4 bytes at 106: the nibble for unit `ib` lives at
                // byte `106 + 2*(ib/4) + (ib/2)%2` (low nibble for even `ib`,
                // high for odd) — the `b16(blk, 53 + ib/4, (ib/2)&1)`
                // addressing of the reference.
                let sc = blk[106 + 2 * (ib / 4) + (ib / 2) % 2];
                let sc = if ib % 2 == 0 { sc & 0xf } else { sc >> 4 };
                let dbr = d * (1.0 + 2.0 * f32::from(sc));
                for l in 0..4 {
                    let qb = 2 + 8 * ib + 4 * (l / 2) + ((2 * l) & 3);
                    let g1 = (blk[qb] as usize) | ((qhb << (8 - 2 * l)) & 256);
                    let g2 = (blk[qb + 1] as usize) | ((qhb << (7 - 2 * l)) & 256);
                    let v = signed_grid4(&IQ3S_F32[g1], &IQ3S_F32[g2], dbr, blk[74 + 4 * ib + l]);
                    if l & 1 == 0 {
                        a0 = v_fma(a0, v, x.as_ptr().add(i));
                    } else {
                        a1 = v_fma(a1, v, x.as_ptr().add(i));
                    }
                    i += 8;
                }
            }
        }
        finish(a0, a1)
    }
}

// ---------------------------------------------------------------------------
// IQ4_NL / IQ4_XS  (16-entry non-linear value table, `vpshufb` lookup)
// ---------------------------------------------------------------------------

/// `acc += kv[nibble] * d * x` for 16 nibbles held in `idx` (low 8, then high 8).
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq4_nibbles(
    acc: __m256,
    idx: __m128i,
    table: __m128i,
    scale: __m256,
    xp: *const f32,
) -> __m256 {
    // SAFETY: `vpshufb` is lane-local with a 16-byte index and 16-byte table, so
    // it cannot read outside `table`; `xp` is in range (see the callers).
    unsafe {
        let vals = _mm_shuffle_epi8(table, idx);
        let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals));
        let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vals, 8)));
        let acc = _mm256_fmadd_ps(_mm256_mul_ps(lo, scale), _mm256_loadu_ps(xp), acc);
        _mm256_fmadd_ps(_mm256_mul_ps(hi, scale), _mm256_loadu_ps(xp.add(8)), acc)
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq4_nl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 18-byte blocks (d + 16 nibble bytes); the nibble table is
    // 16 bytes and `idx` is masked to 0..15 before the shuffle.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mask = _mm_set1_epi8(0x0f);
        let mut i = 0usize;
        for blk in row.chunks_exact(18) {
            let d = _mm256_set1_ps(f16_to_f32(u16le(blk, 0)));
            let qs = _mm_loadu_si128(blk[2..18].as_ptr() as *const __m128i);
            let lo = _mm_and_si128(qs, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), mask);
            a0 = iq4_nibbles(a0, lo, table, d, x.as_ptr().add(i));
            a0 = iq4_nibbles(a0, hi, table, d, x.as_ptr().add(i + 16));
            i += 32;
        }
        finish(a0, _mm256_setzero_ps())
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq4_xs(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 136-byte blocks (d 0..2, scales_h 2..4, scales_l 4..8,
    // 8 × 16 nibble bytes from 8); indices masked to 0..15.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mask = _mm_set1_epi8(0x0f);
        let mut i = 0usize;
        for blk in row.chunks_exact(136) {
            let d = f16_to_f32(u16le(blk, 0));
            let sh = u16le(blk, 2);
            for ib in 0..8usize {
                let sl = blk[4 + ib / 2];
                let ls = ((sl >> (4 * (ib % 2))) & 0xf) | ((((sh >> (2 * ib)) & 3) << 4) as u8);
                let dl = _mm256_set1_ps(d * (f32::from(ls) - 32.0));
                let qs = _mm_loadu_si128(blk[8 + 16 * ib..24 + 16 * ib].as_ptr() as *const __m128i);
                let lo = _mm_and_si128(qs, mask);
                let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), mask);
                a0 = iq4_nibbles(a0, lo, table, dl, x.as_ptr().add(i));
                a0 = iq4_nibbles(a0, hi, table, dl, x.as_ptr().add(i + 16));
                i += 32;
            }
        }
        finish(a0, _mm256_setzero_ps())
    }
}

/// Users the dequant-once kernels unroll (a chunk's expert is used by `m` tokens;
/// `m` above this is processed in groups).
pub const MANY_MAX: usize = 8;

/// **Dequantise once, dot many**: `IQ4_NL` rows against `m` activation rows.
///
/// `xs` are the users' activation pointers (all at the same column offset) and
/// `out` receives `m` dots. Per block the nibble table lookup, the int→float
/// conversion and the scale multiply happen **once**, and each user then costs one
/// FMA per 8 values. The per-user operation order is `mul` then `fmadd`, block
/// major, exactly as [`iq4_nl`] does it, so each dot is bit-identical to the
/// single-user kernel (§8t).
///
/// # Safety
/// `row` must be whole 18-byte blocks, every `x` must have `32 * blocks` values,
/// and `out` must hold `m` floats.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn iq4_nl_many(row: &[u8], xs: &[*const f32], out: *mut f32) {
    // SAFETY: callers pass whole blocks and in-range activation pointers.
    unsafe {
        let m = xs.len().min(MANY_MAX);
        let mut acc = [_mm256_setzero_ps(); MANY_MAX];
        let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mask = _mm_set1_epi8(0x0f);
        let mut i = 0usize;
        for blk in row.chunks_exact(18) {
            let d = _mm256_set1_ps(f16_to_f32(u16le(blk, 0)));
            let qs = _mm_loadu_si128(blk[2..18].as_ptr() as *const __m128i);
            let lo = _mm_shuffle_epi8(table, _mm_and_si128(qs, mask));
            let hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(qs, 4), mask));
            // Four dequantised f32x8 vectors, computed once for every user.
            let v = [
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), d),
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
                    d,
                ),
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), d),
                _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
                    d,
                ),
            ];
            for (u, xp) in xs.iter().take(m).enumerate() {
                let p = xp.add(i);
                let mut a = acc[u];
                a = _mm256_fmadd_ps(v[0], _mm256_loadu_ps(p), a);
                a = _mm256_fmadd_ps(v[1], _mm256_loadu_ps(p.add(8)), a);
                a = _mm256_fmadd_ps(v[2], _mm256_loadu_ps(p.add(16)), a);
                a = _mm256_fmadd_ps(v[3], _mm256_loadu_ps(p.add(24)), a);
                acc[u] = a;
            }
            i += 32;
        }
        for (u, a) in acc.iter().take(m).enumerate() {
            *out.add(u) = finish(*a, _mm256_setzero_ps());
        }
    }
}

/// [`iq4_nl_many`] for `Q2_0` (64 values in an 18-byte block, two alternating
/// accumulators exactly like [`q2_0`]).
///
/// # Safety
/// As [`iq4_nl_many`], with 64 values per block.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn q2_0_many(row: &[u8], xs: &[*const f32], out: *mut f32) {
    // SAFETY: callers pass whole blocks and in-range activation pointers.
    unsafe {
        let m = xs.len().min(MANY_MAX);
        let mut a0 = [_mm256_setzero_ps(); MANY_MAX];
        let mut a1 = [_mm256_setzero_ps(); MANY_MAX];
        let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        let three = _mm256_set1_epi32(3);
        let mut i = 0usize;
        for blk in row.chunks_exact(18) {
            let dv = _mm256_set1_ps(f16_to_f32(u16le(blk, 0)));
            for u in 0..8usize {
                let wv = _mm256_set1_epi32(u16le(blk, 2 + 2 * u) as i32);
                let q = _mm256_and_si256(_mm256_srlv_epi32(wv, shifts), three);
                let v = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q), dv), dv);
                if u & 1 == 0 {
                    for (j, xp) in xs.iter().take(m).enumerate() {
                        a0[j] = v_fma(a0[j], v, xp.add(i));
                    }
                } else {
                    for (j, xp) in xs.iter().take(m).enumerate() {
                        a1[j] = v_fma(a1[j], v, xp.add(i));
                    }
                }
                i += 8;
            }
        }
        for (u, (x, y)) in a0.iter().zip(a1.iter()).take(m).enumerate() {
            *out.add(u) = finish(*x, *y);
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K / Q5_K
// ---------------------------------------------------------------------------

/// `get_scale_min_k4` (see [`crate::reference`]).
#[inline]
fn scale_min_k4(blk: &[u8], j: usize) -> (u8, u8) {
    let q = |i: usize| {
        let w = u16le(blk, 4 + 2 * (i / 2));
        ((w >> (8 * (i & 1))) & 0xff) as u8
    };
    if j < 4 {
        (q(j) & 63, q(j + 4) & 63)
    } else {
        (
            (q(j + 4) & 0xf) | ((q(j - 4) >> 6) << 4),
            (q(j + 4) >> 4) | ((q(j) >> 6) << 4),
        )
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4_k(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 144-byte blocks (dm 0..4, scales 4..16, 4 × 32 nibble bytes
    // from 16); one `fma` per 8 elements, 8 chunks per block.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let nib = _mm256_set1_epi32(0x0f);
        let mut i = 0usize;
        for blk in row.chunks_exact(144) {
            let d = f16_to_f32(u16le(blk, 0));
            let dmin = f16_to_f32(u16le(blk, 2));
            for g in 0..4usize {
                let (s1, m1) = scale_min_k4(blk, 2 * g);
                let (s2, m2) = scale_min_k4(blk, 2 * g + 1);
                let (sc1, mn1) = (
                    _mm256_set1_ps(d * f32::from(s1)),
                    _mm256_set1_ps(dmin * f32::from(m1)),
                );
                let (sc2, mn2) = (
                    _mm256_set1_ps(d * f32::from(s2)),
                    _mm256_set1_ps(dmin * f32::from(m2)),
                );
                let qs = blk[16 + 32 * g..48 + 32 * g].as_ptr();
                for c in 0..4usize {
                    let bytes =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs.add(8 * c) as *const __m128i));
                    let lo = _mm256_and_si256(bytes, nib);
                    let hi = _mm256_and_si256(_mm256_srli_epi32(bytes, 4), nib);
                    a0 = kq_fma(a0, lo, sc1, mn1, x.as_ptr().add(i + 8 * c));
                    a1 = kq_fma(a1, hi, sc2, mn2, x.as_ptr().add(i + 32 + 8 * c));
                }
                i += 64;
            }
        }
        finish(a0, a1)
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q5_k(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 176-byte blocks (dm 0..4, scales 4..16, qh 16..48,
    // 4 × 32 nibble bytes from 48).
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let nib = _mm256_set1_epi32(0x0f);
        let mut i = 0usize;
        for blk in row.chunks_exact(176) {
            let d = f16_to_f32(u16le(blk, 0));
            let dmin = f16_to_f32(u16le(blk, 2));
            for g in 0..4usize {
                let (s1, m1) = scale_min_k4(blk, 2 * g);
                let (s2, m2) = scale_min_k4(blk, 2 * g + 1);
                let (sc1, mn1) = (
                    _mm256_set1_ps(d * f32::from(s1)),
                    _mm256_set1_ps(dmin * f32::from(m1)),
                );
                let (sc2, mn2) = (
                    _mm256_set1_ps(d * f32::from(s2)),
                    _mm256_set1_ps(dmin * f32::from(m2)),
                );
                let ql = blk[48 + 32 * g..80 + 32 * g].as_ptr();
                for c in 0..4usize {
                    let qlb =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.add(8 * c) as *const __m128i));
                    let qhb = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        blk[16 + 8 * c..].as_ptr() as *const __m128i
                    ));
                    let b1 = _mm256_and_si256(
                        _mm256_srlv_epi32(qhb, _mm256_set1_epi32(2 * g as i32)),
                        _mm256_set1_epi32(1),
                    );
                    let b2 = _mm256_and_si256(
                        _mm256_srlv_epi32(qhb, _mm256_set1_epi32(2 * g as i32)),
                        _mm256_set1_epi32(2),
                    );
                    let lo = _mm256_add_epi32(_mm256_and_si256(qlb, nib), _mm256_slli_epi32(b1, 4));
                    let hi = _mm256_add_epi32(
                        _mm256_and_si256(_mm256_srli_epi32(qlb, 4), nib),
                        _mm256_slli_epi32(b2, 3),
                    );
                    a0 = kq_fma(a0, lo, sc1, mn1, x.as_ptr().add(i + 8 * c));
                    a1 = kq_fma(a1, hi, sc2, mn2, x.as_ptr().add(i + 32 + 8 * c));
                }
                i += 64;
            }
        }
        finish(a0, a1)
    }
}

// ---------------------------------------------------------------------------
// Q6_K
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q6_k(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 210-byte blocks (ql 0..128, qh 128..192, scales 192..208,
    // d 208..210); the loop covers exactly 256 elements per block.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let nib = _mm256_set1_epi32(0x0f);
        let three = _mm256_set1_epi32(3);
        let mut i = 0usize;
        for blk in row.chunks_exact(210) {
            let d = f16_to_f32(u16le(blk, 208));
            for g in 0..2usize {
                for q in 0..4usize {
                    let ql_off = g * 64 + if q % 2 == 0 { 0 } else { 32 };
                    let shift = if q >= 2 { 4 } else { 0 };
                    for h in 0..2usize {
                        let sc = d * f32::from(blk[192 + g * 8 + h + 2 * q] as i8);
                        let dsc = _mm256_set1_ps(sc);
                        for c in 0..2usize {
                            let off = 16 * h + 8 * c;
                            let qlb = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                blk[ql_off + off..].as_ptr() as *const __m128i,
                            ));
                            let qhb = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                blk[128 + g * 32 + off..].as_ptr() as *const __m128i,
                            ));
                            let low = _mm256_and_si256(
                                _mm256_srlv_epi32(qlb, _mm256_set1_epi32(shift)),
                                nib,
                            );
                            let high = _mm256_slli_epi32(
                                _mm256_and_si256(
                                    _mm256_srlv_epi32(qhb, _mm256_set1_epi32(2 * q as i32)),
                                    three,
                                ),
                                4,
                            );
                            let qv =
                                _mm256_sub_epi32(_mm256_or_si256(low, high), _mm256_set1_epi32(32));
                            let v = _mm256_mul_ps(_mm256_cvtepi32_ps(qv), dsc);
                            if off & 8 == 0 {
                                a0 = v_fma(a0, v, x.as_ptr().add(i));
                            } else {
                                a1 = v_fma(a1, v, x.as_ptr().add(i));
                            }
                            i += 8;
                        }
                    }
                }
            }
        }
        finish(a0, a1)
    }
}

// ---------------------------------------------------------------------------
// Q2_0 (fork-only: 64 values in 18 bytes = 2.25 bpw)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q2_0(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 18-byte blocks; each of the 8 code words yields 8 values,
    // so a block reads exactly 8 activation vectors.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        let three = _mm256_set1_epi32(3);
        let mut i = 0usize;
        for blk in row.chunks_exact(18) {
            let dv = _mm256_set1_ps(f16_to_f32(u16le(blk, 0)));
            for u in 0..8usize {
                let wv = _mm256_set1_epi32(u16le(blk, 2 + 2 * u) as i32);
                let q = _mm256_and_si256(_mm256_srlv_epi32(wv, shifts), three);
                // (q - 1) * d, without subtracting in integer space so a NaN
                // scale stays a NaN value rather than an index.
                let v = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q), dv), dv);
                if u & 1 == 0 {
                    a0 = v_fma(a0, v, x.as_ptr().add(i));
                } else {
                    a1 = v_fma(a1, v, x.as_ptr().add(i));
                }
                i += 8;
            }
        }
        finish(a0, a1)
    }
}

// ---------------------------------------------------------------------------
// F16 / BF16 / F32
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f16_row(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 16-byte blocks (8 halves) and an in-range `x`.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        for blk in row.chunks_exact(16) {
            let v = _mm256_cvtph_ps(_mm_loadu_si128(blk.as_ptr() as *const __m128i));
            if i & 8 == 0 {
                a0 = v_fma(a0, v, x.as_ptr().add(i));
            } else {
                a1 = v_fma(a1, v, x.as_ptr().add(i));
            }
            i += 8;
        }
        finish(a0, a1)
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn bf16_row(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 16-byte blocks; BF16→F32 is the bit pattern shifted left by
    // 16, which is exactly what `u16 << 16` widening means.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        for blk in row.chunks_exact(16) {
            let w = _mm256_cvtepu16_epi32(_mm_loadu_si128(blk.as_ptr() as *const __m128i));
            let v = _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
            if i & 8 == 0 {
                a0 = v_fma(a0, v, x.as_ptr().add(i));
            } else {
                a1 = v_fma(a1, v, x.as_ptr().add(i));
            }
            i += 8;
        }
        finish(a0, a1)
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f32_row(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: whole 32-byte blocks (8 f32) and an in-range `x`.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0usize;
        for blk in row.chunks_exact(32) {
            let v = _mm256_loadu_ps(blk.as_ptr() as *const f32);
            if i & 8 == 0 {
                a0 = v_fma(a0, v, x.as_ptr().add(i));
            } else {
                a1 = v_fma(a1, v, x.as_ptr().add(i));
            }
            i += 8;
        }
        finish(a0, a1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::scalar;

    /// Every AVX2 kernel must agree with the fused scalar port element by
    /// element. A `1` at position `j` turns the dot into "weight `j`", so this
    /// is a per-value comparison, not just a per-row one.
    #[test]
    fn avx2_decodes_the_same_values_as_the_scalar_port() {
        if !super::super::avx2_available() {
            eprintln!("skipping: no AVX2 on this CPU");
            return;
        }
        for ty in QuantType::ALL {
            let qk = ty.block_elems();
            let bb = ty.block_bytes();
            let mut blk = vec![0u8; bb];
            let mut s = 0x9e37_79b9u32;
            for b in blk.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *b = s as u8;
            }
            for &off in crate::real::scale_offsets(ty) {
                blk[off..off + 2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
            }
            for j in 0..qk {
                let mut x = vec![0f32; qk];
                x[j] = 1.0;
                let want = scalar::dot(ty, &blk, &x);
                let got = dot(ty, &blk, &x);
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "{ty} element {j}: avx2 {got} vs scalar {want}"
                );
            }
        }
    }

    #[test]
    fn avx2_is_deterministic() {
        for ty in QuantType::ALL {
            let cols: usize = 2560;
            let bb = ty.block_bytes() * cols.div_ceil(ty.block_elems());
            let mut row = vec![0u8; bb];
            for (i, b) in row.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(53).wrapping_add(7);
            }
            for &off in crate::real::scale_offsets(ty) {
                row[off..off + 2].copy_from_slice(&0x3400u16.to_le_bytes());
            }
            let x: Vec<f32> = (0..cols).map(|i| ((i * 7) % 13) as f32 * 0.01).collect();
            let a = dot(ty, &row, &x);
            let b = dot(ty, &row, &x);
            assert_eq!(a.to_bits(), b.to_bits(), "{ty}");
        }
    }
}
