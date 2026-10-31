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

/// Most experts [`MoeBucket`] handles (its shared-memory histogram).
pub const BUCKET_MAX_EXPERTS: usize = 1024;

/// Device-side expert bucketing for the merged prefill (`BRECCIA_GPU_DISPATCH`): the
/// router's `[tokens, k]` ids in, the sorted-slot map, the per-slot token list and the
/// per-expert offsets out (layouts in `moe_bucket.comp`). One workgroup.
pub struct MoeBucket<'g> {
    pipe: Pipe<'g>,
}

impl<'g> MoeBucket<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(MoeBucket {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket.spv")),
                4,
                "moe bucket",
            )?,
        })
    }

    /// `ids`/`slots` are `[tokens, k]` u32, `tok` `[tokens * k]` u32, `off`
    /// `[n_expert + 2]` u32.
    pub fn run(
        &self,
        ids: &Buf,
        slots: &Buf,
        tok: &Buf,
        off: &Buf,
        tokens: usize,
        k: usize,
        n_expert: usize,
    ) -> Result<()> {
        if tokens == 0 || k == 0 || n_expert == 0 || n_expert > BUCKET_MAX_EXPERTS {
            return Err(anyhow!(
                "moe bucket needs tokens, k > 0 and 1..={BUCKET_MAX_EXPERTS} experts"
            ));
        }
        let pc = Pc::new()
            .u((tokens * k) as u32)
            .u(k as u32)
            .u(n_expert as u32);
        self.pipe.run(&[ids, slots, tok, off], &pc, (1, 1, 1))
    }
}

/// `dst[u] = src[tok[base + u]]` for `u < mu`, rows of `cols` f32 (the group's activation
/// gather, on the device).
pub struct MoeGatherRows<'g> {
    pipe: Pipe<'g>,
}

impl<'g> MoeGatherRows<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(MoeGatherRows {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/moe_gather_rows.spv")),
                3,
                "moe gather rows",
            )?,
        })
    }

    pub fn run(
        &self,
        src: &Buf,
        tok: &Buf,
        dst: &Buf,
        cols: usize,
        base: u32,
        mu: usize,
    ) -> Result<()> {
        if cols == 0 || mu == 0 {
            return Err(anyhow!("moe gather needs cols and mu > 0"));
        }
        let pc = Pc::new().u(cols as u32).u(base);
        self.pipe.run(
            &[src, tok, dst],
            &pc,
            (cols.div_ceil(64) as u32, mu as u32, 1),
        )
    }
}

/// One expert group of the merged prefill: `(expert, first slot, users, user tokens)`.
/// `users` is `mu` entries from [`bucket_host`], empty from [`groups_from_offsets`]
/// (the device gathers by slot instead).
pub type BucketGroup = (usize, u32, usize, Vec<usize>);

/// The host bucketing the merged prefill has always done: sort `(expert, token, j)`,
/// slot = sorted position, groups of at most `cap` users per expert.
/// Returns `(slots[token * k + j], groups, distinct experts)`.
pub fn bucket_host(
    ids: &[u32],
    k: usize,
    n_expert: usize,
    cap: usize,
) -> Result<(Vec<u32>, Vec<BucketGroup>, usize)> {
    let mut pairs: Vec<(usize, usize, usize)> = Vec::with_capacity(ids.len());
    for (p, &e) in ids.iter().enumerate() {
        let e = e as usize;
        if e >= n_expert {
            return Err(anyhow!(
                "router id {e} is outside the model's {n_expert} experts"
            ));
        }
        pairs.push((e, p / k, p % k));
    }
    // Router order within a token, expert order across: the slot map keeps
    // each distinct expert's users contiguous, which is exactly the GEMM's
    // `O[u * rows + row]` output layout.
    pairs.sort_unstable();
    let mut slots = vec![0u32; ids.len()];
    let mut groups: Vec<BucketGroup> = Vec::new();
    let mut slot = 0u32;
    let mut at = 0usize;
    let mut distinct = 0usize;
    while at < pairs.len() {
        let e = pairs[at].0;
        let end = pairs[at..]
            .iter()
            .position(|p| p.0 != e)
            .map_or(pairs.len(), |d| at + d);
        distinct += 1;
        let mut u = at;
        while u < end {
            let g_end = (u + cap).min(end);
            let base = slot;
            let mut users = Vec::with_capacity(g_end - u);
            for &(_, ti, j) in &pairs[u..g_end] {
                slots[ti * k + j] = slot;
                users.push(ti);
                slot += 1;
            }
            groups.push((e, base, users.len(), users));
            u = g_end;
        }
        at = end;
    }
    Ok((slots, groups, distinct))
}

/// The same groups from the device's `off[0..=n_expert]` (user lists left empty).
/// Returns `(groups, distinct experts)`.
pub fn groups_from_offsets(off: &[u32], cap: usize) -> (Vec<BucketGroup>, usize) {
    let mut groups = Vec::new();
    let mut distinct = 0;
    for (e, w) in off.windows(2).enumerate() {
        let (mut s, end) = (w[0], w[1]);
        distinct += usize::from(end > s);
        while s < end {
            let mu = ((end - s) as usize).min(cap);
            groups.push((e, s, mu, Vec::new()));
            s += mu as u32;
        }
    }
    (groups, distinct)
}

/// What `moe_bucket.comp` computes, as a plain counting sort: `(slots, tok, off)` with
/// `off` of length `n_expert + 2` (pair count, then the out-of-range id count).
/// Ids `>= n_expert` get slot 0 and no `tok` entry, like the shader.
pub fn bucket_model(ids: &[u32], k: usize, n_expert: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut off = vec![0u32; n_expert + 2];
    for &e in ids {
        match off.get_mut(e as usize).filter(|_| (e as usize) < n_expert) {
            Some(c) => *c += 1,
            None => off[n_expert + 1] += 1,
        }
    }
    let mut run = 0;
    for c in off.iter_mut().take(n_expert) {
        (*c, run) = (run, run + *c);
    }
    off[n_expert] = run;
    let mut next = off.clone();
    let (mut slots, mut tok) = (vec![0u32; ids.len()], vec![0u32; run as usize]);
    for (p, &e) in ids.iter().enumerate() {
        if (e as usize) < n_expert {
            slots[p] = next[e as usize];
            tok[next[e as usize] as usize] = (p / k) as u32;
            next[e as usize] += 1;
        }
    }
    (slots, tok, off)
}

#[cfg(test)]
mod bucket_tests {
    use super::*;

    /// Router-like ids: `k` distinct experts per token, deterministic xorshift.
    fn routing(tokens: usize, k: usize, n_expert: usize, seed: u64) -> Vec<u32> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut ids = Vec::new();
        for _ in 0..tokens {
            let mut row: Vec<u32> = Vec::new();
            while row.len() < k {
                let e = (next() % n_expert as u64) as u32;
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            ids.extend(row);
        }
        ids
    }

    #[test]
    fn device_model_matches_the_host_sort() {
        for &(t, k, ne, cap) in &[
            (64, 10, 512, 1024),
            (257, 10, 512, 7),
            (1024, 10, 512, 1024),
            (5, 3, 8, 2),
            (1, 10, 512, 1024),
        ] {
            let ids = routing(t, k, ne, (t * 31 + ne) as u64);
            let (hs, hg, hd) = bucket_host(&ids, k, ne, cap).unwrap();
            let (ms, mt, off) = bucket_model(&ids, k, ne);
            assert_eq!(hs, ms, "slots t={t}");
            assert_eq!(off[ne] as usize, t * k);
            assert_eq!(off[ne + 1], 0);
            // The host's per-slot token list is the groups' users, concatenated.
            let users: Vec<u32> = hg
                .iter()
                .flat_map(|g| g.3.iter().map(|&u| u as u32))
                .collect();
            assert_eq!(users, mt, "tok t={t}");
            let (og, od) = groups_from_offsets(&off[..=ne], cap);
            assert_eq!(od, hd);
            let strip = |g: &[BucketGroup]| g.iter().map(|g| (g.0, g.1, g.2)).collect::<Vec<_>>();
            assert_eq!(strip(&og), strip(&hg), "groups t={t} cap={cap}");
            assert!(hg.iter().all(|g| g.2 == g.3.len() && g.2 <= cap));
        }
    }

    #[test]
    fn out_of_range_ids_are_counted_not_slotted() {
        assert!(bucket_host(&[0, 9], 2, 8, 4).is_err());
        let (slots, tok, off) = bucket_model(&[3, 9, 3, 1], 2, 8);
        assert_eq!(off[8], 3);
        assert_eq!(off[9], 1);
        assert_eq!(tok.len(), 3);
        assert_eq!(slots[1], 0);
    }
}
