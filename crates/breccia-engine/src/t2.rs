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
//! * `BRECCIA_T2_RELOAD=<tokens>` (option 1): count every block's routed experts
//!   (prompt chunks and decode rows) and every `<tokens>` swap up to
//!   `BRECCIA_T2_RELOAD_MAX` (32) hot ones in, ignoring the freeze. The copy runs on
//!   the helper while the block computes and lands at the next block's start.
//!   `BRECCIA_T2_RELOAD_DECODE=1`: count decode rows only (prompt chunks over
//!   `MAX_ROWS` are skipped), i.e. async adaptive swaps from decode history.

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
    pub reload: Option<Mutex<Reload>>,
}

pub fn reload_every() -> Option<usize> {
    std::env::var("BRECCIA_T2_RELOAD").ok()?.parse().ok().filter(|&n: &usize| n > 0)
}

fn reload_max() -> usize {
    std::env::var("BRECCIA_T2_RELOAD_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(32)
}

/// Option 1's one in-flight pool copy (helper staging, the fence, which block and moves).
pub struct Reload {
    every: usize,
    stage: Buf,
    pending: Option<(crate::xfer::PendingCopy, usize, Vec<(usize, usize)>)>,
    reloads: u64,
    record_s: f64,
    wait_s: f64,
}

impl Reload {
    pub fn new(gpu: &Gpu, pools: &BTreeMap<usize, Pool>) -> Result<Option<Mutex<Self>>> {
        let Some(every) = reload_every() else { return Ok(None) };
        let most = pools.values().map(|p| p.per_expert()).max().unwrap_or(0);
        let stage = breccia_kernels::gemv::staging(gpu, (reload_max() as u64 * most).max(1))?;
        Ok(Some(Mutex::new(Reload { every, stage, pending: None, reloads: 0, record_s: 0.0, wait_s: 0.0 })))
    }
}

impl Tier2 {
    /// Option 1, at a block's routing: count `ids` (over `tokens` tokens) and, when the
    /// block is due, land the previous copy and start this block's.
    pub fn note(&self, gpu: &Gpu, b: usize, ids: impl Iterator<Item = usize>, tokens: usize) -> Result<()> {
        let (Some(rl), Some(p)) = (&self.reload, self.pools.get(&b)) else { return Ok(()) };
        static DECODE_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if tokens > MAX_ROWS && *DECODE_ONLY.get_or_init(|| std::env::var("BRECCIA_T2_RELOAD_DECODE").is_ok_and(|v| v == "1")) {
            return Ok(());
        }
        let mut r = rl.lock().expect("t2 reload");
        // One session only: another one may hold hits into a slot this would empty.
        if !p.note(ids, tokens, r.every) || crate::graph::LIVE_SESSIONS.load(std::sync::atomic::Ordering::SeqCst) != 1 {
            return Ok(());
        }
        self.land(gpu, &mut r)?;
        let t0 = std::time::Instant::now();
        let moves = p.take_reload(reload_max());
        r.reloads += 1;
        if moves.is_empty() {
            return Ok(());
        }
        let parts = p.reload_parts(&moves);
        let mut data = Vec::with_capacity(parts.iter().map(|x| x.0.len()).sum());
        let mut copies = Vec::with_capacity(parts.len());
        for &(blob, dst, off) in &parts {
            copies.push((&r.stage, dst, data.len() as u64, off, blob.len() as u64));
            data.extend_from_slice(blob);
        }
        gpu.write_host(&r.stage, &data).map_err(|e| anyhow::anyhow!("{e}"))?;
        let pc = crate::xfer::copy_submit(gpu, &copies)?;
        drop(copies);
        r.pending = Some((pc, b, moves));
        r.record_s += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// Land the in-flight reload (if any). Called at each block's start, before
    /// anything else records on the helper's one-shot buffer.
    pub fn settle(&self, gpu: &Gpu) -> Result<()> {
        match &self.reload {
            Some(rl) => self.land(gpu, &mut rl.lock().expect("t2 reload")),
            None => Ok(()),
        }
    }

    fn land(&self, gpu: &Gpu, r: &mut Reload) -> Result<()> {
        if let Some((pc, b, moves)) = r.pending.take() {
            let t0 = std::time::Instant::now();
            pc.wait(gpu)?;
            r.wait_s += t0.elapsed().as_secs_f64();
            self.pools[&b].land(&moves);
        }
        Ok(())
    }

    /// Hit and reload stats so far (serve prints them as each slot retires).
    pub fn report(&self) {
        crate::d1::report("t2", &self.pools);
        if let Some(Ok(r)) = self.reload.as_ref().map(|m| m.lock()) {
            eprintln!("t2 reload: {} checks, record {:.1} ms, landing waits {:.1} ms", r.reloads, r.record_s * 1e3, r.wait_s * 1e3);
        }
    }

    pub fn release(&self, gpu: &Gpu) {
        // The in-flight copy writes into a pool: land it before the pools go.
        if let Some(Ok(mut r)) = self.reload.as_ref().map(|m| m.lock()) {
            let _ = self.land(gpu, &mut r);
            eprintln!(
                "t2 reload: {} checks, record {:.1} ms, landing waits {:.1} ms",
                r.reloads,
                r.record_s * 1e3,
                r.wait_s * 1e3
            );
            gpu.destroy_buf(&r.stage);
        }
        for p in self.pools.values() {
            p.release(gpu);
        }
        if let Ok(sc) = self.scratch.lock() {
            sc.release(gpu);
        }
    }
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
