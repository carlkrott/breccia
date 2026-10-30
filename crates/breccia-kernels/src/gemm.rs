//! Multi-user fused dequant + mat-vector: one expert, `m` users, one decode.
//!
//! The single-user kernel ([`crate::Gemv`]) decodes a quantised weight block once
//! per output row and dots it against one activation vector, so a prefill chunk of
//! `m` tokens that share an expert pays the decode and the DRAM read `m` times.
//! [`GemmMany`] decodes each block **once into registers** and dots it with all `m`
//! users before moving on (`shaders/gemm_body.glsl`,
//! `docs/phase-4/01-forward-pass.md` §8v).
//!
//! Delivered for the `down` role's `IQ4_NL` first; every other role/type keeps the
//! fused single-user path (the blend is reported in the phase-4 document).

use anyhow::{anyhow, Result};
use ash::vk;
use breccia_vk::ctx::{Buf, Gpu, Qc};

use crate::gemv::{table_bytes, upload, GemvTiming, ROWS_PER_WG, WG_X_MAX};
use crate::quant::QuantType;

/// Users one dispatch can hold in registers (the shader's `USERS_MAX`).
pub const USERS_MAX: usize = 8;

/// A compiled multi-user fused dequant + mat-vector pipeline.
pub struct GemmMany<'g> {
    ty: QuantType,
    gpu: &'g Gpu,
    kernel: breccia_vk::ctx::Kernel,
    /// The int8-dot variant (q8_1 activations, §11j), for the types that have one.
    q8: Option<breccia_vk::ctx::Kernel>,
    /// The fused gate + up + SwiGLU variant (§11l item 4), for the types that have one.
    pair: Option<breccia_vk::ctx::Kernel>,
    /// The register-blocked builds (§12 item 1): `(rows per lane group, kernel)`.
    blocked: Vec<(usize, breccia_vk::ctx::Kernel)>,
    /// Wanted rows per lane group (1 = the plain build).
    rpl: usize,
    /// Forced users per z slice (1..=USERS_MAX); `None` = [`upz_for`] per dispatch.
    upz: Option<usize>,
    tables: Buf,
}

/// `BRECCIA_GEMM_UPZ` (1..=8) forces the users per z slice (A/B only).
fn default_upz() -> Option<usize> {
    static UPZ: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *UPZ.get_or_init(|| {
        std::env::var("BRECCIA_GEMM_UPZ")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .map(|n: usize| n.clamp(1, USERS_MAX))
    })
}

/// `BRECCIA_GEMM_RPL` (1, 2 or 4): rows per lane group of the plain GEMM; 1 = off.
fn default_rpl() -> usize {
    static RPL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *RPL.get_or_init(|| {
        std::env::var("BRECCIA_GEMM_RPL")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n| [1, 2, 4].contains(n))
            .unwrap_or(1)
    })
}

/// Workgroups a dispatch should reach before users share a z slice.
// shortcut: 320 = 4 WGs per CU on the RX's 80 CUs (MI50 has 60); per-device
// target if MI50 numbers disagree.
const WG_TARGET: usize = 320;

/// Users per z slice: `USERS_MAX` when rows x slices already fill the GPU, else
/// the largest split that reaches [`WG_TARGET`], floor 2. A 640-row expert
/// GEMM at 8 users gets 2 (1.9x, per-user bits unchanged); down (2560 rows)
/// and wide dense GEMMs keep 8.
fn upz_for(rows: usize, users: usize) -> usize {
    let groups = rows.div_ceil(ROWS_PER_WG).max(1);
    // Powers of two only: 16 users at 5 per slice would leave a 1-user slice.
    [USERS_MAX, 4, 2]
        .into_iter()
        .find(|&z| groups * users.max(1).div_ceil(z) >= WG_TARGET)
        .unwrap_or(2)
}

macro_rules! blocked_spv {
    ($($t:ident => $s:literal),* $(,)?) => {
        /// The register-blocked builds of `ty`'s plain GEMM (`build.rs`, `_r2`/`_r4`).
        fn blocked_spv(ty: QuantType) -> Vec<(usize, &'static [u8])> {
            match ty {
                $(QuantType::$t => vec![
                    (2, include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r2.spv"))),
                    (4, include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r4.spv"))),
                ],)*
                _ => Vec::new(),
            }
        }
    };
}
blocked_spv!(
    Iq4Nl => "iq4_nl", Q2_0 => "q2_0", Iq2Xxs => "iq2_xxs", Iq2Xs => "iq2_xs",
    Iq2S => "iq2_s", Iq3Xxs => "iq3_xxs", Iq3S => "iq3_s", Iq4Xs => "iq4_xs",
    Q4K => "q4_k", Q5K => "q5_k", Q6K => "q6_k", Bf16 => "bf16",
);

/// Whether [`GemmMany::run_pair`] has a shader for `ty` (gate and up both `ty`).
pub fn has_pair_shader(ty: QuantType) -> bool {
    ty == QuantType::Iq3S
}

impl<'g> GemmMany<'g> {
    /// Compile the pipeline for `ty` (only the types with a multi-user shader).
    pub fn new(gpu: &'g Gpu, ty: QuantType) -> Result<Self> {
        let spv: &[u8] = match ty {
            QuantType::Iq4Nl => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq4_nl.spv")),
            QuantType::Q2_0 => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q2_0.spv")),
            QuantType::Iq2Xxs => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq2_xxs.spv")),
            QuantType::Iq2Xs => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq2_xs.spv")),
            QuantType::Iq2S => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq2_s.spv")),
            QuantType::Iq3Xxs => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq3_xxs.spv")),
            QuantType::Iq3S => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq3_s.spv")),
            QuantType::Iq4Xs => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq4_xs.spv")),
            QuantType::Q4K => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q4_k.spv")),
            QuantType::Q5K => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q5_k.spv")),
            QuantType::Q6K => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q6_k.spv")),
            QuantType::Bf16 => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_bf16.spv")),
            other => {
                return Err(anyhow!(
                    "no multi-user GEMM shader for {}: the delivered set is \
                     IQ4_NL, Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, \
                     Q5_K, Q6_K",
                    other.name()
                ))
            }
        };
        let kernel = gpu
            .kernel(spv, 4)
            .map_err(|e| anyhow!("creating the {ty} multi-user GEMM pipeline: {e}"))?;
        let q8_spv: Option<&[u8]> = match ty {
            QuantType::Iq3S => Some(include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq3_s_q8.spv"))),
            _ => None,
        };
        let q8 = match q8_spv.map(|spv| gpu.kernel(spv, 6)).transpose() {
            Ok(k) => k,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                return Err(anyhow!("creating the {ty} int8 GEMM pipeline: {e}"));
            }
        };
        let pair_spv: Option<&[u8]> = match ty {
            QuantType::Iq3S => Some(include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq3_s_pair.spv"))),
            _ => None,
        };
        debug_assert_eq!(pair_spv.is_some(), has_pair_shader(ty));
        let pair = match pair_spv.map(|spv| gpu.kernel(spv, 4)).transpose() {
            Ok(k) => k,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                if let Some(k) = &q8 {
                    gpu.destroy_kernel(k);
                }
                return Err(anyhow!("creating the {ty} gate+up GEMM pipeline: {e}"));
            }
        };
        let mut blocked = Vec::new();
        for (rpl, spv) in blocked_spv(ty) {
            match gpu.kernel(spv, 4) {
                Ok(k) => blocked.push((rpl, k)),
                Err(e) => {
                    for k in [&kernel].into_iter().chain(q8.iter()).chain(pair.iter()) {
                        gpu.destroy_kernel(k);
                    }
                    for (_, k) in &blocked {
                        gpu.destroy_kernel(k);
                    }
                    return Err(anyhow!("creating the {ty} r{rpl} GEMM pipeline: {e}"));
                }
            }
        }
        let tables = match upload(gpu, &table_bytes()) {
            Ok(b) => b,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                for k in q8.iter().chain(pair.iter()).chain(blocked.iter().map(|(_, k)| k)) {
                    gpu.destroy_kernel(k);
                }
                return Err(e.context("uploading the quant tables"));
            }
        };
        Ok(GemmMany {
            ty,
            gpu,
            kernel,
            q8,
            pair,
            blocked,
            rpl: default_rpl(),
            upz: default_upz(),
            tables,
        })
    }

    pub fn ty(&self) -> QuantType {
        self.ty
    }

    /// Force the users per z slice (clamped to 1..=USERS_MAX); `None` = adaptive.
    pub fn set_users_per_z(&mut self, n: Option<usize>) {
        self.upz = n.map(|n| n.clamp(1, USERS_MAX));
    }

    /// Force the rows per lane group (1, 2 or 4; A/B and tests).
    pub fn set_rows_per_lane(&mut self, n: usize) {
        self.rpl = n;
    }

    fn upz(&self, rows: usize, users: usize) -> usize {
        self.upz.unwrap_or_else(|| upz_for(rows, users))
    }

    /// Whether [`GemmMany::run_at_q8`] exists for this type.
    pub fn has_q8(&self) -> bool {
        self.q8.is_some()
    }

    /// Blocks per row for `cols`.
    pub fn blocks_per_row(&self, cols: usize) -> usize {
        cols.div_ceil(self.ty.block_elems())
    }

    /// Padded activation columns: `X` must hold this many f32 per user (the same
    /// zero-padded layout the single-user kernels take).
    pub fn padded_cols(&self, cols: usize) -> usize {
        self.blocks_per_row(cols) * self.ty.block_elems()
    }

    fn pc(&self, rows: usize, cols: usize, users: usize, base_words: u32) -> [u8; 24] {
        self.pc_rpl(rows, cols, users, base_words, 1)
    }

    /// [`GemmMany::pc`] for a build with `rpl` rows per lane group: the z split
    /// sees `rows / rpl` lane groups, the same count [`GemmMany::dispatch`] gets.
    fn pc_rpl(&self, rows: usize, cols: usize, users: usize, base_words: u32, rpl: usize) -> [u8; 24] {
        let bpr = self.blocks_per_row(cols);
        let mut pc = [0u8; 24];
        pc[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
        pc[4..8].copy_from_slice(&(cols as u32).to_le_bytes());
        pc[8..12].copy_from_slice(&(bpr as u32).to_le_bytes());
        pc[12..16].copy_from_slice(&(users as u32).to_le_bytes());
        pc[16..20].copy_from_slice(&base_words.to_le_bytes());
        pc[20..24].copy_from_slice(&(self.upz(rows / rpl, users) as u32).to_le_bytes());
        pc
    }

    /// Run one GEMM: `out[u*rows + r] = W[r] · x_u`, `u < users`.
    pub fn run(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
    ) -> Result<()> {
        self.run_at(w, x, out, rows, cols, users, 0)
    }

    /// [`GemmMany::run`] with the expert's blob at `base_words` (u16 words) inside
    /// `w` — the resident arena's addressing (`MoeMap`'s convention, §8v).
    #[allow(clippy::too_many_arguments)]
    pub fn run_at(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        base_words: u32,
    ) -> Result<()> {
        if users == 0 {
            return Err(anyhow!("users must be at least 1"));
        }
        let (kernel, rpl) = self.plain(rows, users);
        let pc = self.pc_rpl(rows, cols, users, base_words, rpl);
        self.dispatch(kernel, &[w, x, out, &self.tables], &pc, rows / rpl, users)
    }

    /// The plain-GEMM pipeline for `rows x users` and its rows per lane group: a
    /// register-blocked build when `BRECCIA_GEMM_RPL` asks for one, `rows` divides
    /// and the fewer workgroups still fill the GPU at full users per slice.
    /// Rows per lane group `run` would use for this shape (1 = plain build).
    pub fn rows_per_lane_for(&self, rows: usize, users: usize) -> usize {
        self.plain(rows, users).1
    }

    fn plain(&self, rows: usize, users: usize) -> (&breccia_vk::ctx::Kernel, usize) {
        let want = self.rpl;
        self.blocked
            .iter()
            .find(|(r, _)| {
                *r == want
                    && rows % r == 0
                    && (rows / r).div_ceil(ROWS_PER_WG) * users.div_ceil(USERS_MAX) >= WG_TARGET
            })
            .map_or((&self.kernel, 1), |(r, k)| (k, *r))
    }

    /// Whether [`GemmMany::run_pair`] exists for this type.
    pub fn has_pair(&self) -> bool {
        self.pair.is_some()
    }

    /// Fused gate + up + SwiGLU: `w` holds `rows` gate rows then `rows` up rows,
    /// `out[u*rows + r] = silu(gate_r · x_u) * (up_r · x_u)`.
    pub fn run_pair(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
    ) -> Result<()> {
        if users == 0 {
            return Err(anyhow!("users must be at least 1"));
        }
        let kernel = self
            .pair
            .as_ref()
            .ok_or_else(|| anyhow!("no gate+up GEMM shader for {}", self.ty))?;
        let pc = self.pc(rows, cols, users, 0);
        self.dispatch(kernel, &[w, x, out, &self.tables], &pc, rows, users)
    }

    /// [`GemmMany::run_at`] on int8 dot products: `xq` holds `users * cols` int8
    /// (row-major per user) and `xd` one f16 scale per 32 of them, as
    /// [`crate::Q8Quantize`] writes. Needs `cols` a multiple of the block size.
    #[allow(clippy::too_many_arguments)]
    pub fn run_at_q8(
        &self,
        w: &Buf,
        xq: &Buf,
        xd: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        base_words: u32,
    ) -> Result<()> {
        let (kernel, pc) = self.q8_launch(rows, cols, users, base_words)?;
        // Binding 1 (the fp32 X the body does not read) is bound to xq.
        self.dispatch(kernel, &[w, xq, out, &self.tables, xq, xd], &pc, rows, users)
    }

    fn q8_launch(
        &self,
        rows: usize,
        cols: usize,
        users: usize,
        base_words: u32,
    ) -> Result<(&breccia_vk::ctx::Kernel, [u8; 24])> {
        let kernel = self
            .q8
            .as_ref()
            .ok_or_else(|| anyhow!("no int8 GEMM for {}", self.ty))?;
        if users == 0 {
            return Err(anyhow!("users must be at least 1"));
        }
        if cols % self.ty.block_elems() != 0 {
            return Err(anyhow!(
                "int8 GEMM needs cols ({cols}) a multiple of {}",
                self.ty.block_elems()
            ));
        }
        Ok((kernel, self.pc(rows, cols, users, base_words)))
    }

    fn dispatch(
        &self,
        kernel: &breccia_vk::ctx::Kernel,
        bufs: &[&Buf],
        pc: &[u8; 24],
        rows: usize,
        users: usize,
    ) -> Result<()> {
        let groups = rows.div_ceil(ROWS_PER_WG).max(1) as u32;
        let gx = groups.min(WG_X_MAX);
        let gy = groups.div_ceil(WG_X_MAX);
        // One z slice of the dispatch per `upz` users.
        let gz = users.div_ceil(self.upz(rows, users)) as u32;
        // Inside a dispatch batch the dispatch is **recorded** (the same rule
        // `Pipe::run`, `Gemv::run` and `GemvBatch::run` follow): a chunked prefill
        // records ~200 expert groups per block, and one submit+wait each would be
        // 40-60 µs of round trip per GEMM.
        if crate::dispatch_batch::is_open() {
            return crate::dispatch_batch::record(self.gpu, kernel, bufs, pc, (gx, gy, gz));
        }
        let handles: Vec<vk::Buffer> = bufs.iter().map(|b| b.buf).collect();
        self.gpu
            .bind_kernel(kernel, &handles)
            .map_err(|e| anyhow!("{e}"))?;
        self.gpu
            .wall_run(Qc::C, |d, cmd| unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipe);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.layout,
                    0,
                    &[kernel.set],
                    &[],
                );
                d.cmd_push_constants(cmd, kernel.layout, vk::ShaderStageFlags::COMPUTE, 0, pc);
                d.cmd_dispatch(cmd, gx, gy, gz);
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    /// Time `users` GEMMs in one dispatch, GPU-timestamped.
    #[allow(clippy::too_many_arguments)]
    pub fn run_timed(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        let (kernel, rpl) = self.plain(rows, users);
        let pc = self.pc_rpl(rows, cols, users, 0, rpl);
        self.timed(kernel, &[w, x, out, &self.tables], &pc, rows / rpl, users, target_s)
    }

    /// Timed variant of [`GemmMany::run_at_q8`] (base 0).
    #[allow(clippy::too_many_arguments)]
    pub fn run_timed_q8(
        &self,
        w: &Buf,
        xq: &Buf,
        xd: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        let (kernel, pc) = self.q8_launch(rows, cols, users, 0)?;
        self.timed(kernel, &[w, xq, out, &self.tables, xq, xd], &pc, rows, users, target_s)
    }

    fn timed(
        &self,
        kernel: &breccia_vk::ctx::Kernel,
        bufs: &[&Buf],
        pc: &[u8; 24],
        rows: usize,
        users: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        let handles: Vec<vk::Buffer> = bufs.iter().map(|b| b.buf).collect();
        self.gpu
            .bind_kernel(kernel, &handles)
            .map_err(|e| anyhow!("{e}"))?;
        let groups = rows.div_ceil(ROWS_PER_WG).max(1) as u32;
        let gx = groups.min(WG_X_MAX);
        let gy = groups.div_ceil(WG_X_MAX);
        let gz = users.div_ceil(self.upz(rows, users)) as u32;
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                for _ in 0..n {
                    unsafe {
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipe);
                        d.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.layout,
                            0,
                            &[kernel.set],
                            &[],
                        );
                        d.cmd_push_constants(
                            cmd,
                            kernel.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            pc,
                        );
                        d.cmd_dispatch(cmd, gx, gy, gz);
                        Gpu::barrier_all(d, cmd);
                    }
                }
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(GemvTiming {
            per_iter_s: t.per_iter_s,
            iters: t.iters,
            total_s: t.total_s,
            used_timestamps: t.used_timestamps,
        })
    }

    /// Bytes of quantised weights the dispatch reads (one expert's role tensor).
    pub fn weight_bytes(&self, rows: usize, cols: usize) -> u64 {
        (rows * self.blocks_per_row(cols) * self.ty.block_bytes()) as u64
    }
}

impl Drop for GemmMany<'_> {
    fn drop(&mut self) {
        self.gpu.destroy_kernel(&self.kernel);
        for k in self.q8.iter().chain(self.pair.iter()) {
            self.gpu.destroy_kernel(k);
        }
        for (_, k) in &self.blocked {
            self.gpu.destroy_kernel(k);
        }
        self.gpu.destroy_buf(&self.tables);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn upz_splits_only_narrow_dispatches() {
        use super::upz_for;
        // 640-row expert gate/up: 80 WGs per slice.
        assert_eq!(upz_for(640, 8), 2);
        assert_eq!(upz_for(640, 16), 4);
        assert_eq!(upz_for(640, 1), 2);
        // down (2560 rows) and a 1024-token dense chunk keep USERS_MAX.
        assert_eq!(upz_for(2560, 8), 8);
        assert_eq!(upz_for(640, 1024), 8);
    }
}
