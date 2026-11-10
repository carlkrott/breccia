//! D1: a per-expert VRAM cache for the blocks whose experts run on the host lane.
//!
//! Each such block gets a slot pool on its own device (one buffer per role, in the
//! block's own quant types) plus a zero blob for misses, and a device table
//! `expert -> slot` (`MISS` when not cached). In the block's GPU segment,
//! `moe_remap` turns the router ids into slot ids, the resident kernels compute
//! `hit_out = Σ hits + shared·σ(g)` and that is read back in place of `shared`; the
//! host lane computes only the misses and adds `hit_out`.
//!
//! Opt-in and output-changing (GPU vs host numerics on the hit experts):
//! * `BRECCIA_D1_GIB=<total>`: VRAM for all pools, split evenly by bytes.
//! * `BRECCIA_D1_PROFILE=<file>`: Strata STRP v1 ranked `(layer, expert)` list that
//!   seeds the pools (static hit rate is low, but it lifts the first ~80 tokens).
//! * `BRECCIA_D1_FREEZE=1`: seed only, no swaps (conformance runs).
//!
//! Swaps follow the cachebw replay rule: heat decays by 0.95 per token, and every
//! 8 tokens of a block the hottest uncached experts replace the coldest cached
//! ones when hotter by 1.5×. Swaps only run while exactly one session uses the
//! model; with more, the table is frozen, which is always consistent.

use std::sync::atomic::Ordering;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use breccia_format::ShardSet;
use breccia_kernels::cpu::{LayerPlan, RolePlan};
use breccia_vk::ctx::{Buf, Gpu};

use crate::model::{PipeKey, W};

pub const MISS: u32 = u32::MAX;
const KEEP: f32 = 0.95;
/// Swap policy: check every `every` tokens, swap in up to `max` experts per
/// block that are `ratio` x hotter than their victim. Env BRECCIA_D1_EVERY /
/// _RATIO / _MAX_SWAPS override (defaults 8 / 1.5 / 8).
struct Policy {
    every: u32,
    ratio: f32,
    max: usize,
}

fn policy() -> &'static Policy {
    static P: std::sync::OnceLock<Policy> = std::sync::OnceLock::new();
    fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    P.get_or_init(|| Policy {
        every: env("BRECCIA_D1_EVERY", 8u32).max(1),
        ratio: env("BRECCIA_D1_RATIO", 1.5f32),
        max: env("BRECCIA_D1_MAX_SWAPS", 8usize),
    })
}

pub struct Pool {
    pub device: usize,
    /// Usable slots; slot `slots` is the zero blob misses point at.
    pub slots: usize,
    pub gate: W,
    pub up: W,
    pub down: W,
    /// `[n_expert]` u32: expert -> slot, or [`MISS`].
    pub table: Buf,
    /// `[k]` scratch the remap writes: slot ids and hit-only weights.
    pub slot_ids: Buf,
    pub hit_w: Buf,
    pub state: Mutex<State>,
    /// Swap staging, one per device, shared by its pools; `owns_stage` frees it.
    stage: std::sync::Arc<Buf>,
    owns_stage: bool,
    /// Blocks check on different tokens, so swaps do not all land on one token.
    phase: u32,
    /// Where the slot blobs come from (option 2: a lower-bit GGUF for T2 pools).
    pub src: std::sync::Arc<ShardSet>,
    pub plan: LayerPlan,
}

pub struct State {
    pub slot_of: Vec<u32>,
    pub expert_of: Vec<u32>,
    heat: Vec<f32>,
    tokens: u32,
    pub hits: u64,
    pub routed: u64,
    pub swaps: u64,
    /// Option 1 (`BRECCIA_T2_RELOAD`): routed counts since the last reload (prompt and
    /// decode rows), halved at each reload, and the tokens they cover.
    pf: Vec<f32>,
    pf_tokens: usize,
}

pub fn freeze() -> bool {
    std::env::var("BRECCIA_D1_FREEZE").is_ok_and(|v| v == "1")
}

pub fn total_bytes() -> Option<u64> {
    let g: f64 = std::env::var("BRECCIA_D1_GIB").ok()?.parse().ok()?;
    (g > 0.0).then(|| (g * (1u64 << 30) as f64) as u64)
}

/// Strata STRP v1: "STRP", u32 version, nl, ne, want, n_ranked, then `n_ranked`
/// `(u16 layer, u16 expert)` pairs, hottest first.
pub fn read_strp(path: &std::path::Path) -> Result<Vec<(usize, usize)>> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if b.len() < 24 || &b[..4] != b"STRP" {
        bail!("{}: not a STRP profile", path.display());
    }
    let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
    let (ver, n) = (u(4), u(20));
    if ver != 1 || b.len() < 24 + 4 * n {
        bail!("{}: STRP version {ver}, {n} pairs, {} bytes", path.display(), b.len());
    }
    let h = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize;
    Ok((0..n).map(|i| (h(24 + 4 * i), h(26 + 4 * i))).collect())
}

pub(crate) fn blob<'a>(shards: &'a ShardSet, r: &RolePlan, e: usize) -> &'a [u8] {
    let s = shards.parts()[r.shard].bytes();
    &s[r.offset + e * r.per_expert..][..r.per_expert]
}

fn per_expert(p: &LayerPlan) -> u64 {
    (p.gate.per_expert + p.up.per_expert + p.down.per_expert) as u64
}

/// Build the pools for `plans` (host-lane block, device, geometry, role keys), or
/// nothing when `BRECCIA_D1_GIB` is unset.
pub fn build(
    gpus: &[&Gpu],
    shards: &std::sync::Arc<ShardSet>,
    plans: &[(usize, usize, LayerPlan, [PipeKey; 3])],
    k: usize,
    total: Option<u64>,
    name: &str,
) -> Result<std::collections::BTreeMap<usize, Pool>> {
    let mut out = std::collections::BTreeMap::new();
    let Some(total) = total else { return Ok(out) };
    if plans.is_empty() {
        return Ok(out);
    }
    let seed = match std::env::var_os("BRECCIA_D1_PROFILE") {
        Some(p) => read_strp(std::path::Path::new(&p))?,
        None => Vec::new(),
    };
    let share = total / plans.len() as u64;
    let mut seeded = 0usize;
    let mut stages: std::collections::BTreeMap<usize, std::sync::Arc<Buf>> = Default::default();
    for (b, dev, plan, [g, u, d]) in plans {
        let (b, dev) = (*b, *dev);
        let slots = ((share / per_expert(plan)) as usize).min(plan.n_experts);
        if slots == 0 {
            continue;
        }
        let gpu = gpus[dev];
        let mut st = State {
            slot_of: vec![MISS; plan.n_experts],
            expert_of: vec![MISS; slots],
            heat: vec![0.0; plan.n_experts],
            tokens: 0,
            hits: 0,
            routed: 0,
            swaps: 0,
            pf: vec![0.0; plan.n_experts],
            pf_tokens: 0,
        };
        let mut n = 0;
        for &(l, e) in &seed {
            if n == slots {
                break;
            }
            if l == b && e < plan.n_experts && st.slot_of[e] == MISS {
                st.slot_of[e] = n as u32;
                st.expert_of[n] = e as u32;
                n += 1;
            }
        }
        seeded += n;
        // Unfilled slots and the miss slot stay zero: a zero blob dequantizes to 0.
        let fill = |r: &RolePlan| -> Vec<u8> {
            let mut v = vec![0u8; (slots + 1) * r.per_expert];
            for (s, &e) in st.expert_of.iter().enumerate() {
                if e != MISS {
                    v[s * r.per_expert..][..r.per_expert].copy_from_slice(blob(shards, r, e as usize));
                }
            }
            v
        };
        let up_w = |data: Vec<u8>, key: &PipeKey| -> Result<W> {
            Ok(W { buf: breccia_kernels::gemv::upload(gpu, &data)?, key: *key })
        };
        let gate = up_w(fill(&plan.gate), g).with_context(|| format!("D1 blk.{b} gate pool"))?;
        let up = up_w(fill(&plan.up), u).with_context(|| format!("D1 blk.{b} up pool"))?;
        let down = up_w(fill(&plan.down), d).with_context(|| format!("D1 blk.{b} down pool"))?;
        let table = breccia_kernels::gemv::upload(gpu, &u32_bytes(&st.slot_of))?;
        let slot_ids = breccia_kernels::gemv::alloc(gpu, (k * 4) as u64)?;
        let hit_w = breccia_kernels::gemv::alloc(gpu, (k * 4) as u64)?;
        let owns_stage = !stages.contains_key(&dev);
        if owns_stage {
            let most = plans.iter().filter(|p| p.1 == dev).map(|p| per_expert(&p.2)).max().unwrap_or(0);
            let n = policy().max.max(1) as u64 * most + 4 * plan.n_experts as u64;
            stages.insert(dev, std::sync::Arc::new(breccia_kernels::gemv::staging(gpu, n)?));
        }
        let stage = stages[&dev].clone();
        out.insert(
            b,
            Pool {
                device: dev, slots, gate, up, down, table, slot_ids, hit_w,
                state: Mutex::new(st), stage, owns_stage, phase: b as u32 % policy().every,
                src: shards.clone(), plan: *plan,
            },
        );
    }
    let bytes: u64 = out.values().map(|p| p.gate.buf.size + p.up.buf.size + p.down.buf.size).sum();
    eprintln!(
        "{name}: {} host-lane blocks, {} MiB of expert slots ({} slots, {seeded} seeded from {}){}",
        out.len(),
        bytes >> 20,
        out.values().map(|p| p.slots).sum::<usize>(),
        if seed.is_empty() { "nothing" } else { "the STRP profile" },
        if freeze() { ", frozen" } else { "" }
    );
    Ok(out)
}

/// Reload choice: uncached experts by descending `heat` against cached slots by
/// ascending heat (empty slots first), while the newcomer is `ratio` x hotter.
/// `busy`: moves still in flight (option 1 async), neither their experts nor slots are picked.
fn pick(heat: &[f32], slot_of: &[u32], expert_of: &[u32], max: usize, ratio: f32, busy: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut hot: Vec<usize> =
        (0..heat.len()).filter(|&e| slot_of[e] == MISS && heat[e] > 0.0 && !busy.iter().any(|m| m.0 == e)).collect();
    hot.sort_by(|a, b| heat[*b].total_cmp(&heat[*a]));
    let vh = |s: usize| match expert_of[s] {
        MISS => -1.0,
        e => heat[e as usize],
    };
    let mut cold: Vec<usize> = (0..expert_of.len()).filter(|&s| !busy.iter().any(|m| m.1 == s)).collect();
    cold.sort_by(|a, b| vh(*a).total_cmp(&vh(*b)));
    // A seeded slot that sat idle this window is not evidence: the newcomer must also be
    // `ratio` x the window's mean count, or one-off experts thrash the seed.
    let mean = heat.iter().sum::<f32>() / heat.len().max(1) as f32;
    hot.iter()
        .zip(cold.iter())
        .take(max)
        .take_while(|(&e, &s)| vh(s) < 0.0 || heat[e] > ratio * vh(s).max(mean))
        .map(|(&e, &s)| (e, s))
        .collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Pool {
    pub fn release(&self, gpu: &Gpu) {
        for b in [&self.gate.buf, &self.up.buf, &self.down.buf, &self.table, &self.slot_ids, &self.hit_w] {
            gpu.destroy_buf(b);
        }
        if self.owns_stage {
            gpu.destroy_buf(&self.stage);
        }
    }

    /// After the block's lane ran on `ids`: count hits, heat, and every
    /// `every` tokens swap hot uncached experts in. Call only after the
    /// block's segment has completed (the pool is not read by anything in flight).
    pub fn observe(&self, gpu: &Gpu, ids: &[usize]) -> Result<()> {
        let (shards, plan) = (&*self.src, &self.plan);
        let mut st = self.state.lock().map_err(|_| anyhow!("D1 state poisoned"))?;
        st.routed += ids.len() as u64;
        st.hits += ids.iter().filter(|&&e| st.slot_of[e] != MISS).count() as u64;
        for h in st.heat.iter_mut() {
            *h *= KEEP;
        }
        for &e in ids {
            st.heat[e] += 1.0;
        }
        st.tokens += 1;
        if freeze() || (st.tokens + self.phase) % policy().every != 0 || crate::graph::LIVE_SESSIONS.load(Ordering::SeqCst) != 1 {
            return Ok(());
        }
        let st = &mut *st;
        let mut hot: Vec<usize> = (0..st.heat.len()).filter(|&e| st.slot_of[e] == MISS && st.heat[e] > 0.0).collect();
        hot.sort_by(|a, b| st.heat[*b].total_cmp(&st.heat[*a]));
        let victim_heat = |s: usize| match st.expert_of[s] {
            MISS => -1.0,
            e => st.heat[e as usize],
        };
        let mut cold: Vec<usize> = (0..self.slots).collect();
        cold.sort_by(|a, b| victim_heat(*a).total_cmp(&victim_heat(*b)));
        let mut moves = Vec::new();
        for (&e, &s) in hot.iter().zip(cold.iter()).take(policy().max) {
            let vh = victim_heat(s);
            if vh >= 0.0 && st.heat[e] <= policy().ratio * vh {
                break;
            }
            moves.push((e, s));
        }
        if moves.is_empty() {
            return Ok(());
        }
        for &(e, s) in &moves {
            if st.expert_of[s] != MISS {
                st.slot_of[st.expert_of[s] as usize] = MISS;
            }
            st.expert_of[s] = e as u32;
            st.slot_of[e] = s as u32;
        }
        // One staging write and one submit per check (blobs of every role + the table).
        // shortcut: the submit is waited on; record it into the next segment if it shows up.
        let mut data = Vec::new();
        let mut parts = Vec::new();
        for (w, r) in [(&self.gate, &plan.gate), (&self.up, &plan.up), (&self.down, &plan.down)] {
            for &(e, s) in &moves {
                data.extend_from_slice(blob(shards, r, e));
                parts.push((&w.buf, (s * r.per_expert) as u64, r.per_expert as u64));
            }
        }
        let table = u32_bytes(&st.slot_of);
        data.extend_from_slice(&table);
        parts.push((&self.table, 0, table.len() as u64));
        breccia_kernels::gemv::upload_scatter(gpu, &self.stage, &parts, &data)?;
        st.swaps += moves.len() as u64;
        Ok(())
    }

    /// The routed experts this pool does not hold, for the host lane. Read before
    /// [`Pool::observe`] so it matches the table the GPU segment used.
    pub fn misses(&self, ids: &[usize], weights: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let st = self.state.lock().expect("D1 state");
        ids.iter().zip(weights).filter(|(e, _)| st.slot_of[**e] == MISS).map(|(e, w)| (*e, *w)).unzip()
    }

    /// Option 1: add routed experts to the reload counts; true when `every` tokens
    /// have passed since the last reload.
    pub fn note(&self, ids: impl Iterator<Item = usize>, tokens: usize, every: usize) -> bool {
        let mut st = self.state.lock().expect("D1 state");
        for e in ids {
            st.pf[e] += 1.0;
        }
        st.pf_tokens += tokens;
        st.pf_tokens >= every
    }

    /// Option 1: pick up to `max` swaps (hottest uncached by the reload counts in,
    /// coldest cached out, `ratio` rule) and empty their slots now: the old experts
    /// miss from here on and the new ones only hit after [`Pool::land`], so nothing
    /// reads a slot while its copy is in flight. Halves the counts.
    pub fn take_reload(&self, max: usize, busy: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let moves = self.pick_reload(max, busy);
        self.evict(&moves);
        moves
    }

    /// [`Pool::take_reload`] without emptying the slots (async: [`Pool::evict`] at submit).
    pub fn pick_reload(&self, max: usize, busy: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut st = self.state.lock().expect("D1 state");
        let st = &mut *st;
        let moves = pick(&st.pf, &st.slot_of, &st.expert_of, max, policy().ratio, busy);
        for h in st.pf.iter_mut() {
            *h *= 0.5;
        }
        st.pf_tokens = 0;
        moves
    }

    /// Empty the slots `moves` will overwrite: their old experts miss from now on.
    pub fn evict(&self, moves: &[(usize, usize)]) {
        let mut st = self.state.lock().expect("D1 state");
        for &(_, s) in moves {
            if st.expert_of[s] != MISS {
                let old = st.expert_of[s] as usize;
                st.slot_of[old] = MISS;
                st.expert_of[s] = MISS;
            }
        }
    }

    /// The copies for `moves`: `(blob, dst buffer, dst offset)` per role and move.
    pub fn reload_parts(&self, moves: &[(usize, usize)]) -> Vec<(&[u8], &Buf, u64)> {
        let mut v = Vec::new();
        for (w, r) in [(&self.gate, &self.plan.gate), (&self.up, &self.plan.up), (&self.down, &self.plan.down)] {
            for &(e, s) in moves {
                v.push((blob(&self.src, r, e), &w.buf, (s * r.per_expert) as u64));
            }
        }
        v
    }

    /// The reload's copy has completed: the new experts hit from now on.
    pub fn land(&self, moves: &[(usize, usize)]) {
        let mut st = self.state.lock().expect("D1 state");
        for &(e, s) in moves {
            st.slot_of[e] = s as u32;
            st.expert_of[s] = e as u32;
        }
        st.swaps += moves.len() as u64;
    }

    pub fn per_expert(&self) -> u64 {
        per_expert(&self.plan)
    }

    /// `(hits, routed, swaps)` so far.
    pub fn counts(&self) -> (u64, u64, u64) {
        self.state.lock().map(|s| (s.hits, s.routed, s.swaps)).unwrap_or_default()
    }
}

/// Hit stats, printed when the last session drops (the model may outlive it).
pub fn report(name: &str, pools: &std::collections::BTreeMap<usize, Pool>) {
    let (mut hits, mut routed, mut swaps) = (0, 0, 0);
    for p in pools.values() {
        let (h, r, s) = p.counts();
        (hits, routed, swaps) = (hits + h, routed + r, swaps + s);
    }
    if routed > 0 {
        eprintln!("{name}: {hits} of {routed} routed host-lane experts hit the VRAM pools ({:.1} %), {swaps} swaps", 100.0 * hits as f64 / routed as f64);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn strp_roundtrip() {
        let mut b = b"STRP".to_vec();
        for v in [1u32, 48, 512, 8, 2] {
            b.extend(v.to_le_bytes());
        }
        for v in [3u16, 7, 47, 511] {
            b.extend(v.to_le_bytes());
        }
        let p = std::env::temp_dir().join(format!("strp-test-{}", std::process::id()));
        std::fs::write(&p, &b).unwrap();
        assert_eq!(super::read_strp(&p).unwrap(), vec![(3, 7), (47, 511)]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn reload_pick_swaps_hot_misses_for_cold_slots() {
        use super::MISS;
        // experts 0..6, slots hold 0 (heat 5) and 1 (heat 0); 4 (heat 9) and 5 (heat 1) miss.
        let heat = [5.0, 0.0, 0.0, 0.0, 9.0, 1.0];
        let slot_of = [0, 1, MISS, MISS, MISS, MISS];
        let expert_of = [0, 1];
        // 4 replaces cold 1; 5 is not 1.5x hotter than 0, so it stops there.
        assert_eq!(super::pick(&heat, &slot_of, &expert_of, 8, 1.5, &[]), vec![(4, 1)]);
        // an empty slot takes anything hot
        assert_eq!(super::pick(&heat, &slot_of, &[0, MISS], 8, 1.5, &[]), vec![(4, 1)]);
        assert_eq!(super::pick(&heat, &slot_of, &expert_of, 0, 1.5, &[]), vec![]);
        // idle seeded slots (1, 2, 3) only go to misses above 1.5 x the mean (2.5)
        assert_eq!(super::pick(&heat, &[MISS, 0, 1, 2, MISS, MISS], &[1, 2, 3], 8, 1.5, &[]), vec![(4, 0), (0, 1)]);
        // In-flight moves: neither their expert nor their slot is picked again.
        assert_eq!(super::pick(&heat, &[MISS, 0, 1, 2, MISS, MISS], &[1, 2, 3], 8, 1.5, &[(4, 0)]), vec![(0, 1)]);
    }
}
