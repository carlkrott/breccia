//! The host (CPU) resident-expert lane.
//!
//! # Why this lane exists
//!
//! The production baseline (`docs/phase-3/07-moe-cache-ab.md`) computes every
//! routed expert that is not resident on a GPU **on the CPU, straight out of the
//! host-mapped GGUF**. Staging those experts to VRAM instead is 18–35 % *slower*,
//! so the host lane is not a fallback: it is the fast path for this model's
//! experts. Its budget is fixed by the baseline — one decode token reads
//! `10 experts × 48 blocks × 1.746 MB ≈ 0.84 GB` of quantised weights, and the
//! 20.79 tok/s baseline sustains **≈17.5 GB/s** of host reads. Anything slower
//! than that is a regression against llama.cpp.
//!
//! # Pieces
//!
//! | Item | What it does |
//! |---|---|
//! | [`dot_row_with`] | one row of whole blocks: fused decode + dot, `f32` lanes accumulated in `f32`, lane sums in `f64` |
//! | [`gemv_into`] | `rows` of [`dot_row_with`] — the GEMV the FFN needs |
//! | [`Path`] + [`path`] | `Avx2` (AVX2 + FMA + F16C via [`std::arch`]) or `Scalar`; `BRECCIA_CPU_FORCE_SCALAR=1` forces `Scalar` |
//! | [`LayerPlan`] | one block's three routed-expert role tensors in `breccia_format` geometry: type, `rows`, `cols`, absolute mmap offset, bytes per expert |
//! | [`expert_ffn`] | gate GEMV → up GEMV → `SiLU(gate)·up` → down GEMV, weighted and accumulated, straight from [`ExpertSlices`] |
//! | [`ExpertPool`] | persistent `std::thread` pool + the per-token top-k MoE shape: one expert per worker, deterministic weighted accumulation |
//! | [`bench`] | `breccia bench cpu-experts`: per-type GB/s, GB/token, ms/token, and the comparison against the 17.5 GB/s bar |
//!
//! # Zero-copy
//!
//! Weights are `&[u8]` windows of the `breccia_format` mmap
//! ([`breccia_format::ExpertSlices`]); the only data copied anywhere in this
//! lane is the 10 KB activation vector handed to each worker and the per-expert
//! output accumulator. Nothing touches the model file except the bytes the
//! dot products actually read, and every entry point reports how many those
//! were.
//!
//! # Safety
//!
//! `unsafe` appears in exactly one module, [`avx2`], and only for `std::arch`
//! intrinsics on the SIMD fast path. Everything else — including the whole
//! scalar decode port and the thread pool — is safe Rust.

use std::sync::OnceLock;

use crate::quant::QuantType;
use crate::reference;

mod avx2;
mod bench;
mod expert;
mod scalar;

pub use bench::{
    print_summary, run_cpu_expert_bench, CpuEnvironment, CpuExpertBenchOptions, CpuExpertReport,
    CpuModelInfo, CpuTokenCell, CpuTypeCell, CPU_EXPERTS_SCHEMA_VERSION, DEFAULT_CPU_MODEL_GLOB,
    PARITY_BAR_GBPS,
};
pub use expert::{
    expert_ffn, moe_topk_sequential, silu, ChunkRoute, ExpertPool, FfnScratch, LayerPlan, RolePlan,
};

/// `gemv_into` for **m users at once**: row `r`'s weight block is dequantised once
/// and dotted with every user's activation, so the decode cost is amortised over
/// the users while each user's dot keeps the single-user operation order (§8t).
///
/// `out` is `[user][row]` (user-major). Types without a dequant-once kernel fall
/// back to one fused pass per user — the caller is told which by
/// [`many_supported`].
pub fn gemv_into_many(
    path: Path,
    ty: QuantType,
    packed: &[u8],
    rows: usize,
    cols: usize,
    xs: &[&[f32]],
    out: &mut [f32],
) -> usize {
    let m = xs.len();
    debug_assert!(out.len() >= m * rows);
    if m <= 1 {
        if let Some(x) = xs.first() {
            gemv_into(path, ty, packed, rows, cols, x, out);
        }
        return rows;
    }
    #[cfg(target_arch = "x86_64")]
    if many_supported(path, ty) {
        let row_bytes = crate::cpu::row_bytes(ty, cols);
        let ptrs: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
        let mut tmp = vec![0f32; m];
        for r in 0..rows {
            let row = &packed[r * row_bytes..(r + 1) * row_bytes];
            // SAFETY: `row` is one whole row, every activation has `cols` values
            // (padded to whole blocks by the caller, as `gemv_into` requires) and
            // `tmp` holds `m`.
            unsafe {
                match ty {
                    QuantType::Iq4Nl => avx2::iq4_nl_many(row, &ptrs, tmp.as_mut_ptr()),
                    QuantType::Q2_0 => avx2::q2_0_many(row, &ptrs, tmp.as_mut_ptr()),
                    _ => unreachable!(),
                }
            }
            for (u, v) in tmp.iter().enumerate() {
                out[u * rows + r] = *v;
            }
        }
        return rows;
    }
    for (u, x) in xs.iter().enumerate() {
        gemv_into(
            path,
            ty,
            packed,
            rows,
            cols,
            x,
            &mut out[u * rows..(u + 1) * rows],
        );
    }
    rows
}

/// Does [`gemv_into_many`] amortise the decode for `ty` on this machine?
pub fn many_supported(path: Path, ty: QuantType) -> bool {
    path == Path::Avx2 && matches!(ty, QuantType::Iq4Nl | QuantType::Q2_0)
}

/// Which code path a dot product takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Path {
    /// AVX2 + FMA + F16C intrinsics ([`avx2`]).
    Avx2,
    /// Portable fused scalar port ([`scalar`]).
    Scalar,
}

impl Path {
    pub const fn name(self) -> &'static str {
        match self {
            Path::Avx2 => "avx2",
            Path::Scalar => "scalar",
        }
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Is the AVX2 fast path usable on this CPU?
///
/// Requires `avx2`, `fma` and `f16c` together: FMA is needed for the
/// `fmsub`/`fmadd` transforms and F16C for the fp16 scale widening on the
/// `F16` path. Every AVX2-era x86 part ships all three.
pub fn avx2_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// `BRECCIA_CPU_FORCE_SCALAR=1` forces the portable path (conformance runs both).
pub fn forced_scalar() -> bool {
    std::env::var_os("BRECCIA_CPU_FORCE_SCALAR").is_some_and(|v| {
        let s = v.to_string_lossy();
        !s.is_empty() && s != "0" && !s.eq_ignore_ascii_case("false")
    })
}

/// The path a GEMV should take: [`Path::Avx2`] unless unavailable or forced off.
///
/// Resolved once per GEMV (not per row): the environment lookup is a syscall-free
/// `environ` scan but there is no reason to pay it 2560 times per row.
pub fn path() -> Path {
    if forced_scalar() || !avx2_available() {
        Path::Scalar
    } else {
        Path::Avx2
    }
}

/// One packed row of `cols` values from the mmap at `path`.
///
/// Handles the tail (`cols` not a multiple of the block) through the reference
/// dequantiser; the shipped model is block-aligned, so the tail path exists for
/// tests and future paged weights only.
pub fn dot_row_with(path: Path, ty: QuantType, row: &[u8], cols: usize, x: &[f32]) -> f32 {
    assert!(x.len() >= cols, "activation vector shorter than cols");
    let qk = ty.block_elems();
    let bb = ty.block_bytes();
    let full = cols / qk;
    let mut acc = 0f32;
    if full > 0 {
        let packed = &row[..full * bb];
        let xp = &x[..full * qk];
        acc = match normalized(path) {
            Path::Avx2 => avx2::dot(ty, packed, xp),
            Path::Scalar => scalar::dot(ty, packed, xp),
        };
    }
    let tail = cols - full * qk;
    if tail > 0 {
        let off = full * bb;
        let mut buf = Vec::with_capacity(qk);
        reference::dequant_block(ty, &row[off..off + bb], &mut buf);
        for (j, w) in buf.iter().enumerate().take(tail) {
            acc += w * x[full * qk + j];
        }
    }
    acc
}

/// Convenience wrapper that resolves [`path`] itself.
pub fn dot_row(ty: QuantType, row: &[u8], cols: usize, x: &[f32]) -> f32 {
    dot_row_with(path(), ty, row, cols, x)
}

/// `rows` packed rows of `cols` values → `out[r] = <w_r, x>`.
///
/// `out` must have at least `rows` elements; `x` at least `cols`.
pub fn gemv_into(
    path: Path,
    ty: QuantType,
    packed: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let path = normalized(path);
    let row_bytes = row_bytes(ty, cols);
    assert!(
        packed.len() >= rows * row_bytes,
        "{ty}: {} bytes for {rows}x{cols}, expected {}",
        packed.len(),
        rows * row_bytes
    );
    assert!(out.len() >= rows, "output shorter than rows");
    for r in 0..rows {
        out[r] = dot_row_with(path, ty, &packed[r * row_bytes..], cols, x);
    }
}

/// Bytes of one packed row of `cols` values (whole blocks; see [`row_bytes_exact`]).
pub fn row_bytes(ty: QuantType, cols: usize) -> usize {
    ty.block_bytes() * cols.div_ceil(ty.block_elems())
}

fn normalized(path: Path) -> Path {
    match path {
        Path::Avx2 if !avx2_available() => Path::Scalar,
        p => p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_and_alignment_paths_agree() {
        // 40 columns of IQ4_NL (block 32) exercises "full block + 8-element
        // tail" through the reference dequantiser.
        let ty = QuantType::Iq4Nl;
        let mut packed = vec![0u8; row_bytes(ty, 40)];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(97).wrapping_add(3);
        }
        packed[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        packed[18..20].copy_from_slice(&0x3800u16.to_le_bytes());
        let x: Vec<f32> = (0..40).map(|i| (i as f32 * 0.1).cos()).collect();
        let want = reference::dot_row(ty, &packed, 40, &x) as f32;
        for p in [Path::Scalar, Path::Avx2] {
            let got = dot_row_with(p, ty, &packed, 40, &x);
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "{p}: {got} vs reference {want}"
            );
        }
    }

    #[test]
    fn force_scalar_switch_is_honoured() {
        if !avx2_available() {
            assert_eq!(path(), Path::Scalar);
            return;
        }
        // SAFETY: single-threaded test process section; the variable is only
        // read by `path()`/`forced_scalar()` in this crate.
        std::env::set_var("BRECCIA_CPU_FORCE_SCALAR", "1");
        assert!(forced_scalar());
        assert_eq!(path(), Path::Scalar);
        std::env::remove_var("BRECCIA_CPU_FORCE_SCALAR");
        assert_eq!(path(), Path::Avx2);
    }
}
