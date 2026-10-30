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
    tables: Buf,
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
        let tables = match upload(gpu, &table_bytes()) {
            Ok(b) => b,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                return Err(e.context("uploading the quant tables"));
            }
        };
        Ok(GemmMany {
            ty,
            gpu,
            kernel,
            tables,
        })
    }

    pub fn ty(&self) -> QuantType {
        self.ty
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

    fn pc(rows: usize, cols: usize, bpr: usize, users: usize, base_words: u32) -> [u8; 20] {
        let mut pc = [0u8; 20];
        pc[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
        pc[4..8].copy_from_slice(&(cols as u32).to_le_bytes());
        pc[8..12].copy_from_slice(&(bpr as u32).to_le_bytes());
        pc[12..16].copy_from_slice(&(users as u32).to_le_bytes());
        pc[16..20].copy_from_slice(&base_words.to_le_bytes());
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
        let pc = Self::pc(rows, cols, self.blocks_per_row(cols), users, base_words);
        let groups = rows.div_ceil(ROWS_PER_WG).max(1) as u32;
        let gx = groups.min(WG_X_MAX);
        let gy = groups.div_ceil(WG_X_MAX);
        // More than USERS_MAX users: one z slice of the dispatch per USERS_MAX.
        let gz = users.div_ceil(USERS_MAX) as u32;
        // Inside a dispatch batch the dispatch is **recorded** (the same rule
        // `Pipe::run`, `Gemv::run` and `GemvBatch::run` follow): a chunked prefill
        // records ~200 expert groups per block, and one submit+wait each would be
        // 40-60 µs of round trip per GEMM.
        if crate::dispatch_batch::is_open() {
            return crate::dispatch_batch::record(
                self.gpu,
                &self.kernel,
                &[w, x, out, &self.tables],
                &pc,
                (gx, gy, gz),
            );
        }
        self.gpu
            .bind_kernel(&self.kernel, &[w.buf, x.buf, out.buf, self.tables.buf])
            .map_err(|e| anyhow!("{e}"))?;
        let kernel = &self.kernel;
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
                d.cmd_push_constants(cmd, kernel.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
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
        let pc = Self::pc(rows, cols, self.blocks_per_row(cols), users, 0);
        self.gpu
            .bind_kernel(&self.kernel, &[w.buf, x.buf, out.buf, self.tables.buf])
            .map_err(|e| anyhow!("{e}"))?;
        let groups = rows.div_ceil(ROWS_PER_WG).max(1) as u32;
        let gx = groups.min(WG_X_MAX);
        let gy = groups.div_ceil(WG_X_MAX);
        let kernel = &self.kernel;
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
                            &pc,
                        );
                        d.cmd_dispatch(cmd, gx, gy, 1);
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
        self.gpu.destroy_buf(&self.tables);
    }
}
