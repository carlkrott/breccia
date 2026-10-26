//! Batched multi-expert dispatch: one GEMV over N routed experts.
//!
//! The per-expert decode shape of this model is 640x2560 (`ffn_down_exps`
//! `[640, 2560, 512]`, `ffn_{gate,up}_exps` `[2560, 640, 512]`), i.e. 80
//! workgroups for 60 CUs — 1.3 per CU. That shape is latency-bound (91–161 GB/s
//! measured for the quantised types, `docs/phase-3/01-dequant-gemv.md` §4),
//! while the *same* weights as a 10-expert matrix (6400 rows) reached
//! 251–448 GB/s. The mechanism is parallelism, not fewer bytes, so what a
//! forward pass needs is one dispatch whose grid covers the whole routed set.
//!
//! # Layout
//!
//! `W` is **one device buffer holding N expert blobs back to back** — the M3
//! slot-ring layout (`breccia_vk::transfer::TransferRing` slots hold exactly one
//! expert blob each, so a ring round-trip for a layer's top-10 is a contiguous
//! run of slots). Rows no longer start at `row * bpr` blocks, so the kernel
//! reads a per-row indirection table (binding 4):
//!
//! ```text
//! map[0        .. rows) = row's block base     (u16-word index into W)
//! map[rows     .. 2rows) = row's activation base (vec4 index into X)
//! ```
//!
//! Both halves are `u32`; the whole table is 8 bytes per row (51 KB for the
//! real 6400-row MoE shape, built on the host once per expert set and uploaded
//! once, because the top-10 selection changes per token but the *blob* offsets
//! do not).
//!
//! The activation half is what supports per-expert `x` vectors: the routed
//! `down` projection consumes a different SwiGLU intermediate per expert
//! ([`XLayout::PerExpert`]), while `gate`/`up` consume one shared `mixed`
//! vector ([`XLayout::Shared`], where every row's activation base is 0). One
//! vector costs `padded_cols()` floats, and `cols = 2560` is a multiple of 8 so
//! the vec4 stride is exact.
//!
//! Output stays `[expert][row]` (`out[rows_per_expert * e + r]`), i.e. simply
//! the concatenation of the per-expert `Gemv` results.
//!
//! # Shape
//!
//! `rows = experts * rows_per_expert` rows of `cols` values, no split-K: with
//! the whole routed set in one grid the dispatch already has 800 workgroups
//! (13 per CU), and split-K's reduce pass would need the indirection applied
//! twice for no gain (measured in `docs/phase-3/02-expert-batching-quants.md`
//! §3: the batched shape is bandwidth-bound, not latency-bound).

use anyhow::{anyhow, Result};
use ash::vk;
use breccia_vk::ctx::{Buf, Gpu, Qc};

use crate::gemv::{alloc, table_bytes, upload, Accum, GemvTiming, ROWS_PER_WG, WG_X_MAX};
use crate::quant::QuantType;

/// Where a batch's activation vectors live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XLayout {
    /// One vector, read by every expert (MoE gate/up: all routed experts see
    /// the same `mixed` vector).
    Shared,
    /// One vector per expert, concatenated in `X` (MoE down: each expert has
    /// its own intermediate).
    PerExpert,
}

impl XLayout {
    /// Vectors the `x` buffer must hold for `experts`.
    pub const fn vectors(self, experts: usize) -> usize {
        match self {
            XLayout::Shared => 1,
            XLayout::PerExpert => experts,
        }
    }
}

/// One compiled pipeline for `{type, rows_per_expert, cols}` with the row
/// indirection the batched body needs (descriptor set 0: W, X, O, tables, map).
pub struct GemvBatch<'g> {
    ty: QuantType,
    accum: Accum,
    rows_per_expert: usize,
    cols: usize,
    gpu: &'g Gpu,
    kernel: breccia_vk::ctx::Kernel,
    tables: Buf,
}

impl<'g> GemvBatch<'g> {
    /// Compile the batched pipeline for `ty` on `gpu` (fp32 accumulation).
    pub fn new(gpu: &'g Gpu, ty: QuantType, rows_per_expert: usize, cols: usize) -> Result<Self> {
        Self::with_accum(gpu, ty, rows_per_expert, cols, Accum::F32)
    }

    /// Compile the batched pipeline for `ty` on `gpu` with an explicit
    /// accumulator (see [`Accum`]).
    pub fn with_accum(
        gpu: &'g Gpu,
        ty: QuantType,
        rows_per_expert: usize,
        cols: usize,
        accum: Accum,
    ) -> Result<Self> {
        if rows_per_expert == 0 || cols == 0 {
            return Err(anyhow!(
                "expert shape must be non-empty ({rows_per_expert}x{cols})"
            ));
        }
        if accum == Accum::F16 && !gpu.feats.shader_float16 {
            return Err(anyhow!(
                "device {} does not support shaderFloat16, which fp16 accumulation requires",
                gpu.index
            ));
        }
        let kernel = gpu
            .kernel(gemv_moe_spv(ty, accum), 5)
            .map_err(|e| anyhow!("creating the {ty} batched GEMV pipeline ({accum} accum): {e}"))?;
        let tables = match upload(gpu, &table_bytes()) {
            Ok(b) => b,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                return Err(e.context("uploading the quant tables"));
            }
        };
        Ok(GemvBatch {
            ty,
            accum,
            rows_per_expert,
            cols,
            gpu,
            kernel,
            tables,
        })
    }

    pub fn ty(&self) -> QuantType {
        self.ty
    }
    /// Accumulator precision this pipeline was compiled for.
    pub fn accum(&self) -> Accum {
        self.accum
    }
    pub fn rows_per_expert(&self) -> usize {
        self.rows_per_expert
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Blocks per row (`ceil(cols / block_elems)`).
    pub fn blocks_per_row(&self) -> usize {
        self.cols.div_ceil(self.ty.block_elems())
    }
    /// Elements per row padded to a whole block (the `x` stride in values).
    pub fn padded_cols(&self) -> usize {
        self.blocks_per_row() * self.ty.block_elems()
    }
    /// Bytes of packed weights per expert (`rows_per_expert * blocks * block_bytes`).
    pub fn blob_bytes(&self) -> u64 {
        (self.rows_per_expert * self.blocks_per_row() * self.ty.block_bytes()) as u64
    }
    /// Blocks per expert blob (the unit the row base is expressed in, /2).
    pub fn blob_words(&self) -> u32 {
        (self.blob_bytes() / 2) as u32
    }
    /// Total weight bytes read by a batch of `experts`.
    pub fn total_bytes(&self, experts: usize) -> u64 {
        self.blob_bytes() * experts as u64
    }
    /// Rows the dispatch covers for `experts`.
    pub fn rows(&self, experts: usize) -> usize {
        self.rows_per_expert * experts
    }
    /// Bytes of the indirection table for `experts` (8 per row).
    pub fn map_bytes(&self, experts: usize) -> u64 {
        (self.rows(experts) * 8) as u64
    }
    /// Bytes the `x` buffer needs for `experts` under `layout`.
    pub fn x_bytes(&self, experts: usize, layout: XLayout) -> u64 {
        (layout.vectors(experts) * self.padded_cols() * 4) as u64
    }

    /// Row map for `experts` blobs laid end to end in `W` (expert `e` at
    /// `e * blob_bytes()`), which is the GGUF role-tensor order and the pack's
    /// slot-ring order.
    pub fn map(&self, experts: usize, layout: XLayout) -> Vec<u32> {
        let words: Vec<u32> = (0..experts).map(|e| e as u32 * self.blob_words()).collect();
        self.map_for_blobs(&words, layout)
    }

    /// Row map for arbitrary blob positions (`blob_words[e]` = the u16-word
    /// offset of expert `e`'s blob in `W`), which is what a top-10 selection out
    /// of 512 experts needs: the *set* is scattered even though each blob is
    /// contiguous.
    ///
    /// # Panics
    /// If `blob_words` is empty (a batch needs at least one expert).
    pub fn map_for_blobs(&self, blob_words: &[u32], layout: XLayout) -> Vec<u32> {
        assert!(!blob_words.is_empty(), "a batch needs at least one expert");
        let bpr = self.blocks_per_row() as u32;
        let rows = self.rows(blob_words.len());
        let x_stride = if layout == XLayout::PerExpert {
            (self.padded_cols() / 4) as u32
        } else {
            0
        };
        let mut map = vec![0u32; rows * 2];
        for (e, &base) in blob_words.iter().enumerate() {
            for r in 0..self.rows_per_expert {
                let i = e * self.rows_per_expert + r;
                map[i] = base + r as u32 * bpr * block_words(self.ty);
                map[rows + i] = x_stride * e as u32;
            }
        }
        map
    }

    /// Bind the five buffers of a batched dispatch.
    pub fn bind(&self, w: &Buf, x: &Buf, out: &Buf, map: &Buf) -> Result<()> {
        self.gpu
            .bind_kernel(
                &self.kernel,
                &[w.buf, x.buf, out.buf, self.tables.buf, map.buf],
            )
            .map_err(|e| anyhow!("{e}"))
    }

    /// Record one batched dispatch. `pub(crate)` so the HC mixer
    /// ([`crate::ops::HcMixer`]) can put a projection in the middle of its own
    /// command buffer instead of round-tripping through `run`.
    pub(crate) fn dispatch(&self, d: &ash::Device, cmd: vk::CommandBuffer, rows: usize) {
        // Rows per row-group = ROWS_PER_WG, wrapped over grid.y exactly like the
        // single-expert kernel (`grid.x <= 65535`).
        let groups = rows.div_ceil(ROWS_PER_WG).max(1) as u32;
        let gx = groups.min(WG_X_MAX);
        let gy = groups.div_ceil(WG_X_MAX);
        let mut pc = [0u8; 16];
        pc[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
        pc[4..8].copy_from_slice(&(self.cols as u32).to_le_bytes());
        pc[8..12].copy_from_slice(&(self.blocks_per_row() as u32).to_le_bytes());
        pc[12..16].copy_from_slice(&1u32.to_le_bytes()); // splits = 1
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.kernel.pipe);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &[self.kernel.set],
                &[],
            );
            d.cmd_push_constants(
                cmd,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc,
            );
            d.cmd_dispatch(cmd, gx, gy, 1);
        }
    }

    /// Run one batched dispatch: `out[experts*rows_per_expert]` = every expert's
    /// `W_e · x_e`, synchronously.
    pub fn run(&self, w: &Buf, x: &Buf, out: &Buf, map: &Buf, experts: usize) -> Result<()> {
        let rows = self.rows(experts);
        if crate::dispatch_batch::is_open() {
            // Same push constants and grid as `dispatch`, recorded instead of
            // submitted (`readme` in `dispatch_batch`).
            let groups = rows.div_ceil(ROWS_PER_WG).max(1) as u32;
            let gx = groups.min(WG_X_MAX);
            let gy = groups.div_ceil(WG_X_MAX);
            let mut pc = [0u8; 16];
            pc[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
            pc[4..8].copy_from_slice(&(self.cols as u32).to_le_bytes());
            pc[8..12].copy_from_slice(&(self.blocks_per_row() as u32).to_le_bytes());
            pc[12..16].copy_from_slice(&1u32.to_le_bytes());
            return crate::dispatch_batch::record(
                self.gpu,
                &self.kernel,
                &[w, x, out, &self.tables, map],
                &pc,
                (gx, gy, 1),
            );
        }
        self.bind(w, x, out, map)?;
        self.gpu
            .wall_run(Qc::C, |d, cmd| self.dispatch(d, cmd, rows))
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    /// Time `target_s` worth of repeated batched dispatches (full barrier
    /// between them, so the number is a latency, not a pipelined average).
    pub fn run_timed(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        map: &Buf,
        experts: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        self.bind(w, x, out, map)?;
        let rows = self.rows(experts);
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                for _ in 0..n {
                    self.dispatch(d, cmd, rows);
                    unsafe { Gpu::barrier_all(d, cmd) };
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
}

impl Drop for GemvBatch<'_> {
    fn drop(&mut self) {
        self.gpu.destroy_kernel(&self.kernel);
        self.gpu.destroy_buf(&self.tables);
    }
}

/// 16-bit words per block (`block_bytes / 2`), the row-base unit.
pub const fn block_words(ty: QuantType) -> u32 {
    (ty.block_bytes() / 2) as u32
}

/// `u32` slice → little-endian bytes for the indirection buffer.
pub fn map_to_bytes(map: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(map.len() * 4);
    for v in map {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Allocate a device-local buffer for a row map (`map_bytes(experts)`).
pub fn alloc_map(gpu: &Gpu, experts_rows: usize) -> Result<Buf> {
    alloc(gpu, (experts_rows * 8) as u64)
}

// ---------------------------------------------------------------------------
// Embedded SPIR-V (row-indirection variants, `shaders/gemv_<stem>_moe.comp`)
// ---------------------------------------------------------------------------

macro_rules! spv {
    ($stem:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $stem, ".spv"))
    };
}

fn gemv_moe_spv(ty: QuantType, accum: Accum) -> &'static [u8] {
    match (ty, accum) {
        (QuantType::Iq2Xxs, Accum::F32) => spv!("gemv_iq2_xxs_moe"),
        (QuantType::Iq2Xxs, Accum::F16) => spv!("gemv_iq2_xxs_moe_f16acc"),
        (QuantType::Iq2Xs, Accum::F32) => spv!("gemv_iq2_xs_moe"),
        (QuantType::Iq2Xs, Accum::F16) => spv!("gemv_iq2_xs_moe_f16acc"),
        (QuantType::Iq2S, Accum::F32) => spv!("gemv_iq2_s_moe"),
        (QuantType::Iq2S, Accum::F16) => spv!("gemv_iq2_s_moe_f16acc"),
        (QuantType::Iq3Xxs, Accum::F32) => spv!("gemv_iq3_xxs_moe"),
        (QuantType::Iq3Xxs, Accum::F16) => spv!("gemv_iq3_xxs_moe_f16acc"),
        (QuantType::Iq3S, Accum::F32) => spv!("gemv_iq3_s_moe"),
        (QuantType::Iq3S, Accum::F16) => spv!("gemv_iq3_s_moe_f16acc"),
        (QuantType::Iq4Nl, Accum::F32) => spv!("gemv_iq4_nl_moe"),
        (QuantType::Iq4Nl, Accum::F16) => spv!("gemv_iq4_nl_moe_f16acc"),
        (QuantType::Iq4Xs, Accum::F32) => spv!("gemv_iq4_xs_moe"),
        (QuantType::Iq4Xs, Accum::F16) => spv!("gemv_iq4_xs_moe_f16acc"),
        (QuantType::Q4K, Accum::F32) => spv!("gemv_q4_k_moe"),
        (QuantType::Q4K, Accum::F16) => spv!("gemv_q4_k_moe_f16acc"),
        (QuantType::Q5K, Accum::F32) => spv!("gemv_q5_k_moe"),
        (QuantType::Q5K, Accum::F16) => spv!("gemv_q5_k_moe_f16acc"),
        (QuantType::Q6K, Accum::F32) => spv!("gemv_q6_k_moe"),
        (QuantType::Q6K, Accum::F16) => spv!("gemv_q6_k_moe_f16acc"),
        (QuantType::Q2_0, Accum::F32) => spv!("gemv_q2_0_moe"),
        (QuantType::Q2_0, Accum::F16) => spv!("gemv_q2_0_moe_f16acc"),
        (QuantType::F16, Accum::F32) => spv!("gemv_f16_moe"),
        (QuantType::F16, Accum::F16) => spv!("gemv_f16_moe_f16acc"),
        (QuantType::Bf16, Accum::F32) => spv!("gemv_bf16_moe"),
        (QuantType::Bf16, Accum::F16) => spv!("gemv_bf16_moe_f16acc"),
        (QuantType::F32, Accum::F32) => spv!("gemv_f32_moe"),
        (QuantType::F32, Accum::F16) => spv!("gemv_f32_moe_f16acc"),
    }
}
