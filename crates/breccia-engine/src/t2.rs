//! Tier 2: the RX drives, a second card helps. A per-expert slot pool ([`crate::d1::Pool`],
//! built on the helper card) for the host-lane blocks. In the merged path's cold branch
//! (spec verify rows, batched seqs) the pool's hits run on the helper while the host
//! lane runs only the misses; the helper's rows are weighted into the host partial
//! before it goes up to the block card. Swaps follow D1's adaptive rule per token.
//!
//! Opt-in and output-changing (GPU vs host numerics on the hit experts):
//! * `BRECCIA_T2_GIB=<total>`: helper VRAM for the pools, split evenly by block.
//! * `BRECCIA_T2_DEVICE=<ordinal>`: the helper card (default 1).
//! * `BRECCIA_D1_PROFILE` / `BRECCIA_D1_FREEZE` / `BRECCIA_D1_EVERY..` as for D1.
//! * `BRECCIA_T2_DECODE=0`: keep plain (t=1) decode on the host lane. Default on; without
//!   `BRECCIA_D1_FREEZE=1` its swaps are synchronous on every decode token.

use std::collections::BTreeMap;
use std::sync::Mutex;

use anyhow::{ensure, Result};
use breccia_kernels::ops::{f32_from_bytes, f32_to_bytes, u32_to_bytes, SwiGlu};
use breccia_kernels::{batch_cmd, MoeMap};
use breccia_vk::ctx::{Buf, Gpu};

use crate::d1::Pool;
use crate::model::PipelineSet;
use crate::xfer::Mailbox;

/// Row cap of one helper segment (spec k<=7 verify, up to 8 batched seqs).
pub const MAX_ROWS: usize = 64;

pub fn bytes() -> Option<u64> {
    let g: f64 = std::env::var("BRECCIA_T2_GIB").ok()?.parse().ok()?;
    (g > 0.0).then(|| (g * (1u64 << 30) as f64) as u64)
}

pub fn device() -> usize {
    std::env::var("BRECCIA_T2_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

pub struct Tier2 {
    pub dev: usize,
    pub pools: BTreeMap<usize, Pool>,
    pub scratch: Mutex<Scratch>,
}

pub struct Scratch {
    x: Mailbox,
    /// `(slot, token)` then `(slot, pair)` u32 pairs; the helper builds the row maps.
    ids: [Mailbox; 2],
    maps: [Buf; 3],
    g: Buf,
    u: Buf,
    act: Buf,
    out: Mailbox,
    cap: usize,
}

impl Scratch {
    pub fn new(gpu: &Gpu, k: usize, n_embd: usize, n_ff: usize) -> Result<Self> {
        let cap = MAX_ROWS * k;
        let f = |n: usize| (n * 4) as u64;
        Ok(Scratch {
            x: Mailbox::new(gpu, f(MAX_ROWS * n_embd))?,
            ids: [Mailbox::new(gpu, (cap * 8) as u64)?, Mailbox::new(gpu, (cap * 8) as u64)?],
            // 8 bytes per weight row: gate/up have n_ff rows per expert, down n_embd.
            maps: [
                breccia_kernels::gemv::alloc(gpu, (cap * n_ff * 8) as u64)?,
                breccia_kernels::gemv::alloc(gpu, (cap * n_ff * 8) as u64)?,
                breccia_kernels::gemv::alloc(gpu, (cap * n_embd * 8) as u64)?,
            ],
            g: breccia_kernels::gemv::alloc(gpu, f(cap * n_ff))?,
            u: breccia_kernels::gemv::alloc(gpu, f(cap * n_ff))?,
            act: breccia_kernels::gemv::alloc(gpu, f(cap * n_ff))?,
            out: Mailbox::new_cached(gpu, f(cap * n_embd))?,
            cap,
        })
    }

    pub fn release(&self, gpu: &Gpu) {
        for m in [&self.x, &self.ids[0], &self.ids[1], &self.out] {
            m.release(gpu);
        }
        for b in [&self.maps[0], &self.maps[1], &self.maps[2], &self.g, &self.u, &self.act] {
            gpu.destroy_buf(b);
        }
    }
}

/// Record and submit the hits `(slot, token, weight)` of one block on the helper:
/// gate/up read the token rows `x` (`t x n_embd`), down writes one row per hit.
/// The caller waits with [`finish`]. No batch may be open on the helper.
pub fn launch(
    gpu: &Gpu,
    pipes: &PipelineSet<'_>,
    swiglu: &SwiGlu<'_>,
    moe_map: &MoeMap<'_>,
    pool: &Pool,
    sc: &Scratch,
    hits: &[(u32, usize, f32)],
    x: &[f32],
    n_embd: usize,
) -> Result<batch_cmd::Segment> {
    let n = hits.len();
    ensure!(n > 0 && n <= sc.cap && x.len() <= MAX_ROWS * n_embd, "tier-2 batch of {n} pairs");
    let (gate, up, down) = (pipes.batch(&pool.gate.key)?, pipes.batch(&pool.up.key)?, pipes.batch(&pool.down.key)?);
    let n_ff = gate.rows(1);
    ensure!(
        gate.padded_cols() == n_embd && up.padded_cols() == n_embd && down.padded_cols() == n_ff,
        "tier-2 needs unpadded expert rows"
    );
    let tok: Vec<u32> = hits.iter().flat_map(|h| [h.0, h.1 as u32]).collect();
    let pair: Vec<u32> = hits.iter().enumerate().flat_map(|(p, h)| [h.0, p as u32]).collect();
    sc.x.write(&f32_to_bytes(x))?;
    sc.ids[0].write(&u32_to_bytes(&tok))?;
    sc.ids[1].write(&u32_to_bytes(&pair))?;
    batch_cmd::begin(gpu)?;
    moe_map.run_pairs(&sc.ids[0].buf, &sc.maps[0], gate, n)?;
    moe_map.run_pairs(&sc.ids[0].buf, &sc.maps[1], up, n)?;
    moe_map.run_pairs(&sc.ids[1].buf, &sc.maps[2], down, n)?;
    gate.run(&pool.gate.buf, &sc.x.buf, &sc.g, &sc.maps[0], n)?;
    up.run(&pool.up.buf, &sc.x.buf, &sc.u, &sc.maps[1], n)?;
    swiglu.run(&sc.g, &sc.u, &sc.act, n * n_ff)?;
    down.run(&pool.down.buf, &sc.act, &sc.out.buf, &sc.maps[2], n)?;
    batch_cmd::end_submit(gpu, None)
}

/// Wait for [`launch`]'s segment and add `weight * row` of every hit into `part`
/// (`t x n_embd`, the host lane's partial).
pub fn finish(
    gpu: &Gpu,
    seg: batch_cmd::Segment,
    sc: &Scratch,
    hits: &[(u32, usize, f32)],
    part: &mut [f32],
    n_embd: usize,
) -> Result<f64> {
    let waited = batch_cmd::wait(&[gpu], seg)?;
    let mut raw = vec![0u8; hits.len() * n_embd * 4];
    sc.out.read(&mut raw)?;
    let rows = f32_from_bytes(&raw);
    for (p, &(_, ti, w)) in hits.iter().enumerate() {
        let (dst, src) = (&mut part[ti * n_embd..][..n_embd], &rows[p * n_embd..][..n_embd]);
        for (d, s) in dst.iter_mut().zip(src) {
            *d += w * s;
        }
    }
    Ok(waited)
}
