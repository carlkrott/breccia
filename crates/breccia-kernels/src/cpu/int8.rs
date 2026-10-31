//! Int8 dot kernels for the host expert lane (opt-in, approximate).
//!
//! The activation is quantised **once per GEMV** to signed 8-bit with one f32
//! scale per block (`d = amax / 127`, round to nearest even). Each weight block is
//! decoded to 8-bit integers and multiplied with `vpmaddubsw` + `vpmaddwd` (AVX2)
//! or `vpdpbusd` (AVX-512 VNNI). Both produce the same exact i32 lane sums and the
//! same float tail, so the two paths are bit-identical; against the f32 kernels
//! the only error is the activation rounding (at most `d / 2` per element).
//!
//! Activation layouts (one scale per block):
//! - `IQ4_NL`: 32 values in natural order.
//! - `Q2_0`: 64 values permuted so slot `p * 16 + b` holds `x[4b + p]`, the order
//!   a shift-and-mask of the 16 code bytes yields; the block's sum of the q8
//!   values is kept for the `q - 1` offset, applied in integers.
//! - `IQ3_S`: 32 values in natural order, one scale per 32-value sub-block. The
//!   grid magnitudes are unsigned bytes and the sign bits are applied to the
//!   activation (`vpsignb`), so the dot is still unsigned x signed.

use std::arch::x86_64::*;
use std::cell::RefCell;

use crate::quant::QuantType;
use crate::tables::{IQ3S_GRID, KVALUES_IQ4NL};

/// Types with an int8 kernel; everything else stays on the f32 AVX2 kernels.
pub(super) fn supported(ty: QuantType) -> bool {
    matches!(ty, QuantType::Iq4Nl | QuantType::Q2_0 | QuantType::Iq3S)
}

/// Activation values per q8 scale for `ty` (also a multiple of its weight block).
pub(super) fn act_block(ty: QuantType) -> usize {
    match ty {
        QuantType::Q2_0 => 64,
        _ => 32,
    }
}

/// One quantised activation vector.
#[derive(Default)]
pub(super) struct Q8 {
    qs: Vec<i8>,
    d: Vec<f32>,
    sum: Vec<i32>,
}

/// Quantise `x` (a multiple of [`act_block`] long) into `q` in `ty`'s layout.
///
/// A NaN anywhere in a block makes that block's scale NaN, so a bad activation
/// still poisons the dot product as it does on the f32 path.
pub(super) fn quantize(ty: QuantType, x: &[f32], q: &mut Q8) {
    let n = act_block(ty);
    debug_assert!(x.len().is_multiple_of(n));
    q.qs.clear();
    q.qs.resize(x.len(), 0);
    q.d.clear();
    q.sum.clear();
    let permute = ty == QuantType::Q2_0;
    for (bi, xb) in x.chunks_exact(n).enumerate() {
        let mut amax = 0f32;
        let mut nan = false;
        for v in xb {
            amax = amax.max(v.abs());
            nan |= v.is_nan();
        }
        let d = if nan { f32::NAN } else { amax / 127.0 };
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let dst = &mut q.qs[bi * n..(bi + 1) * n];
        let mut s = 0i32;
        for (i, v) in xb.iter().enumerate() {
            // |v * id| <= 127 up to rounding; `as` saturates and maps NaN to 0.
            let qv = (v * id).round_ties_even() as i8;
            let slot = if permute { (i % 4) * 16 + i / 4 } else { i };
            dst[slot] = qv;
            s += i32::from(qv);
        }
        q.d.push(d);
        q.sum.push(s);
    }
}

thread_local! {
    static SCRATCH: RefCell<Q8> = RefCell::new(Q8::default());
}

/// `out[r] = <w_r, x>` for `rows` whole-block rows, the activation quantised once.
///
/// # Safety
/// AVX2 + FMA + F16C must be available, plus AVX-512 VNNI + VL when `vnni`;
/// `ty` must be [`supported`], `cols` a multiple of [`act_block`] and `packed`
/// at least `rows * row_bytes` long.
pub(super) unsafe fn gemv(
    vnni: bool,
    ty: QuantType,
    packed: &[u8],
    rows: usize,
    row_bytes: usize,
    x: &[f32],
    out: &mut [f32],
) {
    SCRATCH.with_borrow_mut(|q| {
        quantize(ty, x, q);
        for (r, o) in out.iter_mut().enumerate().take(rows) {
            let row = &packed[r * row_bytes..(r + 1) * row_bytes];
            // SAFETY: forwarded from the caller; `row` is whole blocks and `q`
            // covers exactly `cols` values.
            *o = unsafe { dot(vnni, ty, row, q) };
        }
    });
}

/// # Safety
/// As [`gemv`]; `row` is whole weight blocks matching `q`'s length.
pub(super) unsafe fn dot(vnni: bool, ty: QuantType, row: &[u8], q: &Q8) -> f32 {
    debug_assert_eq!(row.len() / ty.block_bytes() * ty.block_elems(), q.qs.len());
    // SAFETY: the caller checked the features of whichever kernel runs.
    unsafe {
        match (ty, vnni) {
            (QuantType::Iq4Nl, false) => iq4_nl_avx2(row, q),
            (QuantType::Iq4Nl, true) => iq4_nl_vnni(row, q),
            (QuantType::Q2_0, false) => q2_0_avx2(row, q),
            (QuantType::Q2_0, true) => q2_0_vnni(row, q),
            (QuantType::Iq3S, false) => iq3_s_avx2(row, q),
            (QuantType::Iq3S, true) => iq3_s_vnni(row, q),
            _ => unreachable!("no int8 kernel for {ty}"),
        }
    }
}

/// The block's fp16 scale via F16C (exact, same value as `f16_to_f32`).
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn wscale(blk: &[u8]) -> f32 {
    super::avx2::f16c(u16::from_le_bytes([blk[0], blk[1]]))
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn hsum(a: __m256, b: __m256) -> f32 {
    let s = _mm256_add_ps(a, b);
    let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}

/// The 32 signed IQ4_NL weights of one block as `(|w|, x * sign(w))` inputs.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq4_unit(blk: &[u8], table: __m128i, xp: *const i8) -> (__m256i, __m256i) {
    let mask = _mm_set1_epi8(0x0f);
    // SAFETY: `blk` is one 18-byte block; `xp` points at 32 activation bytes.
    unsafe {
        let qs = _mm_loadu_si128(blk[2..18].as_ptr().cast());
        let lo = _mm_shuffle_epi8(table, _mm_and_si128(qs, mask));
        let hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(qs, 4), mask));
        let w = _mm256_set_m128i(hi, lo);
        let x = _mm256_loadu_si256(xp.cast());
        // No table value is 0, so sign(x, w) never drops a term.
        (_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w))
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq4_nl_avx2(row: &[u8], q: &Q8) -> f32 {
    // SAFETY: whole blocks, 32 activation bytes per block (see `dot`).
    unsafe {
        let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast());
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); 2];
        for (b, blk) in row.chunks_exact(18).enumerate() {
            let (w, x) = iq4_unit(blk, table, q.qs.as_ptr().add(b * 32));
            // |w|,|x| <= 127, so a pair sum (<= 32258) never saturates i16.
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), ones);
            let s = _mm256_set1_ps(wscale(blk) * q.d[b]);
            acc[b & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), s, acc[b & 1]);
        }
        hsum(acc[0], acc[1])
    }
}

#[target_feature(enable = "avx2,fma,f16c,avx512vnni,avx512vl")]
unsafe fn iq4_nl_vnni(row: &[u8], q: &Q8) -> f32 {
    // SAFETY: as `iq4_nl_avx2`.
    unsafe {
        let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast());
        let mut acc = [_mm256_setzero_ps(); 2];
        for (b, blk) in row.chunks_exact(18).enumerate() {
            let (w, x) = iq4_unit(blk, table, q.qs.as_ptr().add(b * 32));
            let p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, x);
            let s = _mm256_set1_ps(wscale(blk) * q.d[b]);
            acc[b & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), s, acc[b & 1]);
        }
        hsum(acc[0], acc[1])
    }
}

/// The 64 two-bit codes of one Q2_0 block as two vectors in the permuted order.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q2_codes(blk: &[u8]) -> (__m256i, __m256i) {
    let m3 = _mm_set1_epi8(3);
    // SAFETY: `blk` is one 18-byte block.
    unsafe {
        let c = _mm_loadu_si128(blk[2..18].as_ptr().cast());
        let t0 = _mm_and_si128(c, m3);
        let t1 = _mm_and_si128(_mm_srli_epi16(c, 2), m3);
        let t2 = _mm_and_si128(_mm_srli_epi16(c, 4), m3);
        let t3 = _mm_and_si128(_mm_srli_epi16(c, 6), m3);
        (_mm256_set_m128i(t1, t0), _mm256_set_m128i(t3, t2))
    }
}

/// `sum(q * x) - sum(x)` lanes for one block: the `- 1` offset of `(q - 1) * d`,
/// subtracted exactly in lane 0.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn q2_offset(p: __m256i, sum: i32) -> __m256i {
    _mm256_sub_epi32(p, _mm256_zextsi128_si256(_mm_cvtsi32_si128(sum)))
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q2_0_avx2(row: &[u8], q: &Q8) -> f32 {
    // SAFETY: whole blocks, 64 activation bytes per block (see `dot`).
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); 2];
        for (b, blk) in row.chunks_exact(18).enumerate() {
            let (wa, wb) = q2_codes(blk);
            let xp = q.qs.as_ptr().add(b * 64);
            let xa = _mm256_loadu_si256(xp.cast());
            let xb = _mm256_loadu_si256(xp.add(32).cast());
            // q <= 3: two pair sums add to <= 1524, far inside i16.
            let p16 = _mm256_add_epi16(_mm256_maddubs_epi16(wa, xa), _mm256_maddubs_epi16(wb, xb));
            let p = q2_offset(_mm256_madd_epi16(p16, ones), q.sum[b]);
            let s = _mm256_set1_ps(wscale(blk) * q.d[b]);
            acc[b & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), s, acc[b & 1]);
        }
        hsum(acc[0], acc[1])
    }
}

#[target_feature(enable = "avx2,fma,f16c,avx512vnni,avx512vl")]
unsafe fn q2_0_vnni(row: &[u8], q: &Q8) -> f32 {
    // SAFETY: as `q2_0_avx2`.
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 2];
        for (b, blk) in row.chunks_exact(18).enumerate() {
            let (wa, wb) = q2_codes(blk);
            let xp = q.qs.as_ptr().add(b * 64);
            let xa = _mm256_loadu_si256(xp.cast());
            let xb = _mm256_loadu_si256(xp.add(32).cast());
            let p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), wa, xa);
            let p = q2_offset(_mm256_dpbusd_epi32(p, wb, xb), q.sum[b]);
            let s = _mm256_set1_ps(wscale(blk) * q.d[b]);
            acc[b & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), s, acc[b & 1]);
        }
        hsum(acc[0], acc[1])
    }
}

/// One 32-value IQ3_S sub-block as `(|w|, x * sign(w))`: 8 grid entries (4
/// unsigned magnitudes each) and the sub-block's 32 sign bits applied to `x`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq3_unit(blk: &[u8], ib: usize, xp: *const i8) -> (__m256i, __m256i) {
    // SAFETY: `blk` is one 110-byte block (qs 2..66, qh 66..74, signs 74..106),
    // `ib < 8`; grid indices are <= 511 into the 512-entry table; `xp` points at
    // 32 activation bytes.
    unsafe {
        // 9-bit grid index k = qs[k] | qh bit k << 8, then one 8-lane gather.
        let qs = _mm256_cvtepu8_epi32(_mm_loadl_epi64(blk[2 + 8 * ib..].as_ptr().cast()));
        let qh = _mm256_sllv_epi32(
            _mm256_set1_epi32(i32::from(blk[66 + ib])),
            _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1),
        );
        let idx = _mm256_or_si256(qs, _mm256_and_si256(qh, _mm256_set1_epi32(256)));
        let w = _mm256_i32gather_epi32::<4>(IQ3S_GRID.as_ptr().cast(), idx);
        // Sign byte l covers values 8l..8l+8, bit j value 8l+j.
        let sb = u32::from_le_bytes([
            blk[74 + 4 * ib],
            blk[75 + 4 * ib],
            blk[76 + 4 * ib],
            blk[77 + 4 * ib],
        ]);
        let spread = _mm256_set_epi64x(
            0x0303_0303_0303_0303,
            0x0202_0202_0202_0202,
            0x0101_0101_0101_0101,
            0,
        );
        let bits = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
        let s = _mm256_shuffle_epi8(_mm256_set1_epi32(sb as i32), spread);
        let neg = _mm256_cmpeq_epi8(_mm256_and_si256(s, bits), bits);
        let sgn = _mm256_or_si256(neg, _mm256_set1_epi8(1));
        let x = _mm256_loadu_si256(xp.cast());
        (w, _mm256_sign_epi8(x, sgn))
    }
}

/// The sub-block multiplier `1 + 2 * scale_nibble` of IQ3_S unit `ib`.
#[inline]
fn iq3_sub(blk: &[u8], ib: usize) -> f32 {
    let sc = blk[106 + ib / 2];
    let sc = if ib & 1 == 0 { sc & 0xf } else { sc >> 4 };
    f32::from(1 + 2 * sc)
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq3_s_avx2(row: &[u8], q: &Q8) -> f32 {
    // SAFETY: whole 110-byte blocks, 256 activation bytes per block (see `dot`).
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); 2];
        for (b, blk) in row.chunks_exact(110).enumerate() {
            let d = wscale(blk);
            for ib in 0..8 {
                let (w, x) = iq3_unit(blk, ib, q.qs.as_ptr().add(b * 256 + ib * 32));
                // |w| <= 15, |x| <= 127: a pair sum (<= 3810) never saturates i16.
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), ones);
                let s = _mm256_set1_ps(d * iq3_sub(blk, ib) * q.d[b * 8 + ib]);
                acc[ib & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), s, acc[ib & 1]);
            }
        }
        hsum(acc[0], acc[1])
    }
}

#[target_feature(enable = "avx2,fma,f16c,avx512vnni,avx512vl")]
unsafe fn iq3_s_vnni(row: &[u8], q: &Q8) -> f32 {
    // SAFETY: as `iq3_s_avx2`.
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 2];
        for (b, blk) in row.chunks_exact(110).enumerate() {
            let d = wscale(blk);
            for ib in 0..8 {
                let (w, x) = iq3_unit(blk, ib, q.qs.as_ptr().add(b * 256 + ib * 32));
                let p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, x);
                let s = _mm256_set1_ps(d * iq3_sub(blk, ib) * q.d[b * 8 + ib]);
                acc[ib & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), s, acc[ib & 1]);
            }
        }
        hsum(acc[0], acc[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{avx2_available, dot_row_with, vnni_available, Path};
    use crate::reference::{dequant_block, f16_to_f32};

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// Random whole-block row with a sane fp16 scale per block.
    fn row(ty: QuantType, blocks: usize, seed: &mut u64) -> Vec<u8> {
        let mut r = vec![0u8; blocks * ty.block_bytes()];
        for blk in r.chunks_exact_mut(ty.block_bytes()) {
            for b in blk.iter_mut() {
                *b = lcg(seed) as u8;
            }
            // fp16 in [0.0039, 0.0156): exponent 0x1c..0x1f-ish, random mantissa.
            let h: u16 = 0x1c00 | (lcg(seed) as u16 & 0x07ff);
            blk[..2].copy_from_slice(&h.to_le_bytes());
        }
        r
    }

    fn act(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| (lcg(seed) % 20001) as f32 / 10000.0 - 1.0)
            .collect()
    }

    /// Independent reference: natural-order quantisation, integer weights from
    /// the reference dequantiser at unit scale, exact i64 block sums.
    /// Returns `(dot, sum of |terms|)`; the second sizes the f32 rounding tolerance.
    fn reference(ty: QuantType, r: &[u8], x: &[f32]) -> (f64, f64) {
        let n = act_block(ty);
        let mut w = Vec::new();
        for blk in r.chunks_exact(ty.block_bytes()) {
            let mut unit = blk.to_vec();
            unit[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // 1.0
            let mut vals = Vec::new();
            dequant_block(ty, &unit, &mut vals);
            let d = f64::from(f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
            w.extend(vals.iter().map(|v| (*v as i64, d)));
        }
        let mut acc = 0f64;
        let mut mag = 0f64;
        for (bi, xb) in x.chunks_exact(n).enumerate() {
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let dx = amax / 127.0;
            let id = if dx > 0.0 { 1.0 / dx } else { 0.0 };
            // An act block never straddles a weight block, so one weight scale.
            let mut isum = 0i64;
            let mut iabs = 0i64;
            for (i, v) in xb.iter().enumerate() {
                let t = w[bi * n + i].0 * i64::from((v * id).round_ties_even() as i8);
                isum += t;
                iabs += t.abs();
            }
            acc += w[bi * n].1 * f64::from(dx) * isum as f64;
            mag += w[bi * n].1 * f64::from(dx) * iabs as f64;
        }
        (acc, mag)
    }

    #[test]
    fn int8_kernels_match_the_integer_reference_and_each_other() {
        if !avx2_available() {
            return;
        }
        let mut seed = 0x1234_5678;
        for ty in [QuantType::Iq4Nl, QuantType::Q2_0, QuantType::Iq3S] {
            let wpb = ty.block_elems();
            for blocks in [2usize, 3, 10, 20, 40, 80] {
                let r = row(ty, blocks, &mut seed);
                let x = act(blocks * wpb, &mut seed);
                let mut q = Q8::default();
                quantize(ty, &x, &mut q);
                let a = unsafe { dot(false, ty, &r, &q) };
                let (want, mag) = reference(ty, &r, &x);
                let tol = 1e-6 * mag + 1e-9;
                assert!(
                    (f64::from(a) - want).abs() <= tol,
                    "{ty} {blocks}: {a} vs {want}"
                );
                if vnni_available() {
                    let v = unsafe { dot(true, ty, &r, &q) };
                    assert_eq!(a.to_bits(), v.to_bits(), "{ty} {blocks}: avx2 {a} vnni {v}");
                }
                // Against the f32 kernel: activation rounding <= d/2 per element.
                let f = dot_row_with(Path::Avx2, ty, &r, x.len(), &x);
                let mut bound = 0f64;
                let mut vals = Vec::new();
                for (bi, blk) in r.chunks_exact(ty.block_bytes()).enumerate() {
                    vals.clear();
                    dequant_block(ty, blk, &mut vals);
                    for (j, wv) in vals.iter().enumerate() {
                        let i = bi * wpb + j;
                        bound += f64::from(wv.abs()) * f64::from(q.d[i / act_block(ty)]) * 0.5;
                    }
                }
                assert!(
                    (f64::from(a) - f64::from(f)).abs() <= bound * 1.01 + 1e-4,
                    "{ty} {blocks}: int8 {a} f32 {f} bound {bound}"
                );
            }
        }
    }

    #[test]
    fn nan_activation_poisons_the_dot() {
        if !avx2_available() {
            return;
        }
        let mut seed = 7;
        for ty in [QuantType::Iq4Nl, QuantType::Q2_0, QuantType::Iq3S] {
            let r = row(ty, 2, &mut seed);
            let mut x = act(2 * ty.block_elems(), &mut seed);
            x[5] = f32::NAN;
            let mut q = Q8::default();
            quantize(ty, &x, &mut q);
            assert!(unsafe { dot(false, ty, &r, &q) }.is_nan(), "{ty}");
        }
    }
}
