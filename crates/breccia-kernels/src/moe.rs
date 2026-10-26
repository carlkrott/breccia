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

    /// Write the map for `experts` blobs of `rows_per_expert` rows each.
    ///
    /// * `blob_words` — one expert blob's length in `u16` words (`GemvBatch::blob_words`).
    /// * `base_words` — the block's role tensor offset inside the arena, same units.
    /// * `per_expert_pc4` — `0` for one shared activation vector, else
    ///   `padded_cols / 4` (the `XLayout::PerExpert` stride in `vec4`).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        ids: &Buf,
        map: &Buf,
        experts: usize,
        rows_per_expert: usize,
        blob_words: u32,
        base_words: u32,
        per_expert_pc4: u32,
    ) -> Result<()> {
        let rows = experts * rows_per_expert;
        let pc = Pc::new()
            .u(experts as u32)
            .u(rows_per_expert as u32)
            .u(blob_words)
            .u(base_words)
            .u(per_expert_pc4)
            .u(rows as u32);
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
