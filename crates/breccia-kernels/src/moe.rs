//! Host-free MoE: a **device-side** row map for the batched expert GEMV, and the
//! routed combine.
//!
//! `GemvBatch` indexes expert blobs through a `u32` map built on the host
//! (`map_for_blobs`), which forces a readback of the router's top-k ids and an
//! upload of the map (200 KiB for the `down` role at `XLayout::PerExpert`, per
//! block per token). That round trip is what kept the expert lane on the host
//! even when the weights are in VRAM.
//!
//! [`MoeMap`] builds the same map from the ids **in place on the device**, and
//! [`MoeCombine`] does `out = Σ_j w_j·down_j + shared·sigmoid(gate)` from the
//! device's own weights and gate logit. Together they make a block's MoE phase
//! fully device-resident: the host sees nothing.

use anyhow::{anyhow, Result};
use breccia_vk::ctx::{Buf, Gpu};

use crate::ops::{grid_for, Pc, Pipe};

/// Fills a `GemvBatch` row map from the router's ids, on the device.
pub struct MoeMap<'g> {
    pipe: Pipe<'g>,
}

impl<'g> MoeMap<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(MoeMap {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/moe_map.spv")),
                2,
                "moe map",
            )?,
        })
    }

    /// Write the map for `experts` blobs of one `batch`'s shape.
    ///
    /// Everything the row map depends on comes from the pipeline the map will be
    /// used with — `rows_per_expert`, `blocks_per_row`, the quant block size and
    /// the activation stride — so a call site cannot get the units wrong. Use a
    /// **separate map per role**: `ffn_gate_exps` and `ffn_up_exps` need not share
    /// a type (this artifact's blocks do, per `docs/specs/model-artifacts-inventory.md`
    /// §4.2, but nothing enforces it).
    ///
    /// * `base_words` — the block's role-tensor offset inside the arena, in `u16`
    ///   words (zero when the arena holds one block's tensor).
    /// * `per_expert` — `true` for `XLayout::PerExpert` (one activation row per
    ///   expert, the `down` role), `false` for `XLayout::Shared` (gate/up).
    pub fn run(
        &self,
        ids: &Buf,
        map: &Buf,
        batch: &crate::GemvBatch<'_>,
        experts: usize,
        base_words: u32,
        per_expert: bool,
    ) -> Result<()> {
        let rows = batch.rows(experts);
        let pc = Pc::new()
            .u(experts as u32)
            .u(batch.rows_per_expert() as u32)
            .u(batch.blob_words())
            .u(base_words)
            .u(if per_expert {
                (batch.padded_cols() / 4) as u32
            } else {
                0
            })
            .u(rows as u32)
            .u(batch.blocks_per_row() as u32)
            .u(crate::batch::block_words(batch.ty()));
        let (gx, gy) = grid_for(rows, 64);
        self.pipe.run(&[ids, map], &pc, (gx, gy, 1))
    }
}

/// `out = Σ_j w_j · in_j + shared · sigmoid(gate_logit)`.
pub struct MoeCombine<'g> {
    pipe: Pipe<'g>,
}

impl<'g> MoeCombine<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(MoeCombine {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/moe_combine.spv")),
                5,
                "moe combine",
            )?,
        })
    }

    /// `down` is `[experts, cols]`, `weights` `[experts]`, `shared` `[cols]`,
    /// `gate_logit` `[1]`, `out` `[cols]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        down: &Buf,
        weights: &Buf,
        shared: &Buf,
        gate_logit: &Buf,
        out: &Buf,
        experts: usize,
        cols: usize,
    ) -> Result<()> {
        if experts == 0 || cols == 0 {
            return Err(anyhow!("moe combine needs experts and cols > 0"));
        }
        let pc = Pc::new().u(experts as u32).u(cols as u32);
        let (gx, gy) = grid_for(cols, 256);
        self.pipe
            .run(&[down, weights, shared, gate_logit, out], &pc, (gx, gy, 1))
    }
}

/// Chunked-prefill fan-in: `out[u] = Σ_j w[u][j]·scratch[slot[u][j]] + shared[u]·σ(gate[u])`
/// for every token of a chunk (§8v).
///
/// The grouped GEMM writes each distinct expert's users contiguously, so the
/// `(token, expert)` pairs are resolved through a host-built `slots` map — the same
/// algebra [`MoeCombine`] does for one token, with the pair list made explicit.
pub struct MoeFanIn<'g> {
    pipe: Pipe<'g>,
}

impl<'g> MoeFanIn<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(MoeFanIn {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/moe_fanin.spv")),
                6,
                "moe fan-in",
            )?,
        })
    }

    /// `scratch` is `[slots, cols]`, `slots`/`weights` are `[tokens, k]`, `shared`
    /// `[tokens, cols]`, `gate` `[tokens]`, `out` `[tokens, cols]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        scratch: &Buf,
        slots: &Buf,
        weights: &Buf,
        shared: &Buf,
        gate: &Buf,
        out: &Buf,
        tokens: usize,
        k: usize,
        cols: usize,
    ) -> Result<()> {
        if tokens == 0 || k == 0 || cols == 0 {
            return Err(anyhow!("moe fan-in needs tokens, k and cols > 0"));
        }
        let pc = Pc::new().u(cols as u32).u(k as u32).u(tokens as u32);
        let gx = cols.div_ceil(64).max(1) as u32;
        self.pipe.run(
            &[scratch, slots, weights, shared, gate, out],
            &pc,
            (gx, tokens as u32, 1),
        )
    }
}
