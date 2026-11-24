//! `inkling2_loop` forward pass (S11 of `docs/specs/inkling2-loop-port.md`).
//!
//! A straight port of llama.cpp `src/models/inkling2.cpp` `build_layer` /
//! `build_output` and `inkling2-loop.cpp`'s pass driver onto the generic
//! Breccia ops (S4-S10). [`InklingSeq`] carries one sequence between calls
//! (S13): per-slot K/V (SWA slots in a ring of `window + chunk - 1` cells,
//! full slots linear up to `ctx`) and the short-conv states, so prefill and
//! decode are the same `forward` over the next positions.
//!
//! Every shape comes from [`InklingConfig`] or the tensor dims, so a sibling
//! checkpoint with other widths, loop bounds or block size runs unchanged.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;

use anyhow::{anyhow, ensure, Context, Result};
use breccia_format::ShardSet;
use breccia_kernels::ops::{
    self, Act, ActMode, Add, AttnDense, AttnShape, ControlWeights, DepthMix, EmbGather, Glu,
    ControlWs, GluMode, LowRankControl, RelBias, RmsNorm, Router, RowClip, ShortConv, Sink,
};
use breccia_kernels::cpu::{ChunkRoute, ExpertPool, GluAct, LayerPlan};
use breccia_kernels::{batch_cmd, gemv, proj, GemvBatch, MoeFanIn, MoeMap, QuantType, XLayout};
use breccia_vk::ctx::{Buf, Gpu};

use crate::inkling::InklingConfig;

/// Called with `(node name, slot, values)` for every golden-named node (slot
/// -1 = global). Returning `Some` replaces the node (teacher forcing).
pub type Tap<'t> = dyn FnMut(&str, i32, &[f32]) -> Option<Vec<f32>> + 't;

/// A device buffer freed on drop.
pub struct Owned<'g>(&'g Gpu, Buf);

impl Deref for Owned<'_> {
    type Target = Buf;
    fn deref(&self) -> &Buf {
        &self.1
    }
}

impl Drop for Owned<'_> {
    fn drop(&mut self) {
        if batch_cmd::is_open_on(self.0) {
            let b = &self.1;
            let dead = Buf { buf: b.buf, mem: b.mem, size: b.size, id: b.id };
            GRAVE.with(|g| g.borrow_mut().push((key(self.0), dead)));
        } else {
            self.0.destroy_buf(&self.1);
        }
    }
}

thread_local! {
    /// Buffers dropped while their device's batch was open: recorded commands
    /// still use them, so they are destroyed once that batch has completed.
    static GRAVE: RefCell<Vec<(usize, Buf)>> = const { RefCell::new(Vec::new()) };
}

fn key(gpu: &Gpu) -> usize {
    gpu as *const Gpu as usize
}

/// Destroy `gpu`'s deferred buffers (its batch must be closed and waited for).
fn bury(gpu: &Gpu) {
    let dead: Vec<(usize, Buf)> = GRAVE.with(|g| {
        let mut g = g.borrow_mut();
        let (dead, keep) = g.drain(..).partition(|(k, _)| *k == key(gpu));
        *g = keep;
        dead
    });
    for (_, b) in dead {
        gpu.destroy_buf(&b);
    }
}

/// Close `gpu`'s open batch (submit + wait) so the host can read or upload;
/// returns whether one was open, for [`resume`].
fn pause(gpu: &Gpu) -> Result<bool> {
    if !batch_cmd::is_open_on(gpu) {
        return Ok(false);
    }
    let r = batch_cmd::end(gpu);
    bury(gpu);
    r?;
    Ok(true)
}

fn resume(gpu: &Gpu, was_open: bool) -> Result<()> {
    if was_open {
        batch_cmd::begin(gpu)?;
    }
    Ok(())
}

/// Run host transfers `f` with `gpu`'s batch closed around them.
fn host<T>(gpu: &Gpu, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let open = pause(gpu)?;
    let r = f();
    resume(gpu, open)?;
    r
}

const CTRL: [&str; 4] = ["q", "k", "v", "ffn"];

/// One slot's cache: K/V `[cells, kv_dim]`, cell positions (i32, -1 = empty)
/// and the short-conv state.
struct SlotKv<'g> {
    k: Owned<'g>,
    v: Owned<'g>,
    kpos: Owned<'g>,
    cells: usize,
    ring: bool,
    state: Owned<'g>,
}

/// One sequence's state between [`InklingForward::forward`] calls.
// shortcut: f32 K/V only (64 KiB/token on the 8 full slots); f16/q6 KV is the S14 memory step.
pub struct InklingSeq<'g> {
    pos: usize,
    ctx: usize,
    chunk: usize,
    slots: Vec<SlotKv<'g>>,
}

impl InklingSeq<'_> {
    /// Tokens consumed so far (the next position).
    pub fn pos(&self) -> usize {
        self.pos
    }
}

pub struct InklingForward<'g> {
    gpu: &'g Gpu,
    shards: &'g ShardSet,
    cfg: InklingConfig,
    /// Batched GEMV per `(type, rows, cols)`.
    gemvs: HashMap<(QuantType, usize, usize), GemvBatch<'g>>,
    norm: RmsNorm<'g>,
    head_norm: RmsNorm<'g>,
    mix: DepthMix<'g>,
    ctrl: LowRankControl<'g>,
    conv: ShortConv<'g>,
    attn: AttnDense<'g>,
    add: Add<'g>,
    act: Act<'g>,
    glu: Glu<'g>,
    router: Router<'g>,
    fanin: MoeFanIn<'g>,
    moe_map: MoeMap<'g>,
    emb: EmbGather<'g>,
    clip_q: RowClip<'g>,
    clip_q_tau: RowClip<'g>,
    clip_k: RowClip<'g>,
    clip_r: RowClip<'g>,
    clip_r_tau: RowClip<'g>,
    /// Device copies kept between steps, by tensor name.
    cache: RefCell<HashMap<String, Rc<Owned<'g>>>>,
    residency: Residency,
    /// CPU expert lane and its per-layer plan (`None` = layer runs on the GPU).
    host: Option<(ExpertPool, Vec<Option<LayerPlan>>)>,
    /// Record each step into dispatch batches (one submit per MoE readback)
    /// instead of one submit per op.
    batching: bool,
    /// Per-token-count device scratch reused across steps: GEMV maps by
    /// `(type, rows, cols, tokens)`, control workspaces, fan-in inputs.
    maps: RefCell<HashMap<(QuantType, usize, usize, usize), Owned<'g>>>,
    ctrl_ws: RefCell<HashMap<usize, ControlWs>>,
    fan: RefCell<HashMap<usize, [Owned<'g>; 3]>>,
    /// Host-visible staging (readback, upload) and its capacity in tokens.
    stage: RefCell<Option<(Owned<'g>, Owned<'g>, usize)>>,
    /// Routed (token, expert) picks per weight layer: `routed[layer][expert]`.
    /// Loop layers run several times per token, so they count once per pass.
    routed: Vec<Vec<u64>>,
    /// Hot experts held on the device per weight layer (host-lane layers only;
    /// their misses still run on the lane), and `(hits, routed)` through them.
    hot: Vec<Option<HotLayer<'g>>>,
    hot_stats: (u64, u64),
    hot_stage: RefCell<Vec<Option<Owned<'g>>>>,
    hot_swaps: HotSwaps,
    /// Tokens since the last reload, and experts swapped so far.
    hot_tokens: usize,
    hot_swapped: u64,
    /// Force the host-built expert maps even where the device path applies (A/B).
    pub host_maps: bool,
    /// Seconds spent in the CPU lane, in `moe` overall, and in reloads.
    /// shortcut: public test probe, move behind a stats accessor when serve reports it.
    pub prof: [f64; 3],
}

/// One weight layer's device copy of its hottest experts: raw GGUF blobs, slot
/// `s` at blob offset `s`, and `slot_of[expert]` (`MISS` when not held).
struct HotLayer<'g> {
    gu: Owned<'g>,
    dn: Owned<'g>,
    slot_of: Vec<u32>,
    expert_of: Vec<u32>,
    /// Picks since the last reload (halved at each), what swaps rank by.
    heat: Vec<f32>,
}

/// Swap policy: every `every` tokens, up to `max` experts in total move
/// (D1's ratio rule per layer); `every` 0 = frozen.
#[derive(Clone, Copy, Debug)]
pub struct HotSwaps {
    pub every: usize,
    pub max: usize,
}

const MISS: u32 = u32::MAX;

/// The `slots` hottest `(layer, expert)` pairs across all weight layers by
/// routed count (never-picked experts are left out), as experts per layer.
/// Ranking globally gives layers that run several times per token (loop
/// layers) more slots without knowing why they are hot.
pub fn hot_pick(heat: &[Vec<u64>], slots: usize) -> Vec<Vec<usize>> {
    let mut all: Vec<(u64, usize, usize)> = heat
        .iter()
        .enumerate()
        .flat_map(|(l, c)| c.iter().enumerate().filter(|(_, &n)| n > 0).map(move |(e, &n)| (n, l, e)))
        .collect();
    all.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut out = vec![Vec::new(); heat.len()];
    for &(_, l, e) in all.iter().take(slots) {
        out[l].push(e);
    }
    out
}

impl Drop for InklingForward<'_> {
    fn drop(&mut self) {
        for (_, ws) in self.ctrl_ws.get_mut().drain() {
            self.ctrl.release(self.gpu, ws);
        }
    }
}

/// Which weights stay on the device between steps. Default: none (upload per use).
#[derive(Clone, Copy, Debug, Default)]
pub struct Residency {
    /// Every non-expert tensor (dense projections, norms, embeddings' helpers).
    pub dense: bool,
    /// The `*_exps` tensors of layers not on the host lane.
    pub experts: bool,
}

impl<'g> InklingForward<'g> {
    pub fn new(gpu: &'g Gpu, shards: &'g ShardSet, cfg: InklingConfig) -> Result<Self> {
        let (n, hd, h, hkv) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_head_kv);
        let tau = Some((cfg.log_scale_floor, cfg.log_scale_alpha));
        let emb_ty = Self::ty_of(shards, "token_embd.weight")?;
        Ok(InklingForward {
            gpu,
            shards,
            gemvs: HashMap::new(),
            norm: RmsNorm::with_eps(gpu, n, 1, cfg.eps)?,
            head_norm: RmsNorm::with_eps(gpu, hd, 0, cfg.eps)?,
            mix: DepthMix::new(gpu, n, 1e-6)?,
            ctrl: LowRankControl::new(gpu, n, cfg.loop_rank, 1e-6)?,
            conv: ShortConv::new(gpu)?,
            attn: AttnDense::new(gpu)?,
            add: Add::new(gpu)?,
            act: Act::new(gpu, ActMode::TanhCap)?,
            glu: Glu::new(
                gpu,
                GluMode::Situ {
                    beta: cfg.situ_beta,
                    lb: cfg.situ_linear_beta,
                },
            )?,
            router: Router::sigmoid(
                gpu,
                cfg.n_expert,
                cfg.n_expert_used,
                1e-20,
                cfg.expert_weights_scale,
            )?,
            fanin: MoeFanIn::new(gpu)?,
            moe_map: MoeMap::new(gpu)?,
            emb: EmbGather::new(gpu, emb_ty, n)?,
            clip_q: RowClip::new(gpu, hd, h, cfg.qk_clip, None)?,
            clip_q_tau: RowClip::new(gpu, hd, h, cfg.qk_clip, tau)?,
            clip_k: RowClip::new(gpu, hd, hkv, cfg.qk_clip, None)?,
            clip_r: RowClip::new(gpu, cfg.n_rel, h, cfg.rel_clip, None)?,
            clip_r_tau: RowClip::new(gpu, cfg.n_rel, h, cfg.rel_clip, tau)?,
            cache: RefCell::new(HashMap::new()),
            residency: Residency::default(),
            host: None,
            batching: true,
            maps: RefCell::new(HashMap::new()),
            ctrl_ws: RefCell::new(HashMap::new()),
            fan: RefCell::new(HashMap::new()),
            stage: RefCell::new(None),
            routed: vec![vec![0; cfg.n_expert]; cfg.n_layers],
            hot: (0..cfg.n_layers).map(|_| None).collect(),
            hot_stats: (0, 0),
            hot_stage: RefCell::new(Vec::new()),
            hot_swaps: HotSwaps { every: 8, max: 32 },
            hot_tokens: 0,
            hot_swapped: 0,
            host_maps: false,
            prof: [0.0; 3],
            cfg,
        })
    }

    pub fn config(&self) -> &InklingConfig {
        &self.cfg
    }

    /// Batch dispatches (default on); off submits every op on its own.
    /// Routed picks per weight layer so far (`[layer][expert]`), the heat a
    /// hot-expert placement ranks by.
    pub fn routed(&self) -> &[Vec<u64>] {
        &self.routed
    }

    /// `(hits, routed)` expert picks served by the hot pool so far.
    pub fn hot_stats(&self) -> (u64, u64) {
        self.hot_stats
    }

    /// Experts swapped into the hot pool so far.
    pub fn hot_swapped(&self) -> u64 {
        self.hot_swapped
    }

    pub fn set_hot_swaps(&mut self, s: HotSwaps) {
        self.hot_swaps = s;
    }

    /// Reload step, between steps (batch closed): per held layer, D1's pick
    /// (hottest uncached in, coldest held out, 1.5x rule), capped at `max`
    /// experts in total, hottest moves first; then halve the heat.
    fn hot_reload(&mut self) -> Result<()> {
        let HotSwaps { every, max } = self.hot_swaps;
        if every == 0 || self.hot_tokens < every || self.hot.iter().all(Option::is_none) {
            return Ok(());
        }
        self.hot_tokens = 0;
        let mut moves: Vec<(f32, usize, usize, usize)> = Vec::new();
        for (l, h) in self.hot.iter_mut().enumerate() {
            let Some(h) = h else { continue };
            for (e, s) in crate::d1::pick(&h.heat, &h.slot_of, &h.expert_of, max, 1.5, &[]) {
                moves.push((h.heat[e], l, e, s));
            }
            h.heat.iter_mut().for_each(|x| *x *= 0.5);
        }
        moves.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        moves.truncate(max);
        if moves.is_empty() {
            return Ok(());
        }
        let ne = self.cfg.n_expert;
        let mut data = Vec::new();
        let mut spans = Vec::new();
        for &(_, l, e, s) in &moves {
            for (role, name) in ["ffn_gate_up_exps.weight", "ffn_down_exps.weight"].iter().enumerate() {
                let b = self.shards.tensor(&format!("blk.{l}.{name}")).ok_or_else(|| anyhow!("missing {name}"))?.as_slice();
                let pe = b.len() / ne;
                data.extend_from_slice(&b[e * pe..][..pe]);
                spans.push((l, role, (s * pe) as u64, pe as u64));
            }
        }
        let stage = Owned(self.gpu, gemv::staging(self.gpu, data.len() as u64)?);
        let parts: Vec<(&Buf, u64, u64)> = spans
            .iter()
            .map(|&(l, role, off, len)| {
                let h = self.hot[l].as_ref().expect("held");
                (if role == 0 { &*h.gu } else { &*h.dn }, off, len)
            })
            .collect();
        gemv::upload_scatter(self.gpu, &stage, &parts, &data)?;
        drop(parts);
        for &(_, l, e, s) in &moves {
            let h = self.hot[l].as_mut().expect("held");
            if h.expert_of[s] != MISS {
                h.slot_of[h.expert_of[s] as usize] = MISS;
            }
            h.expert_of[s] = e as u32;
            h.slot_of[e] = s as u32;
        }
        self.hot_swapped += moves.len() as u64;
        Ok(())
    }

    /// Keep the hottest experts of the host-lane layers on the device, within
    /// `budget` bytes, ranked by `heat` (`[layer][expert]` routed counts, e.g.
    /// [`InklingForward::routed`] of a warm-up run). Misses still run on the
    /// lane; hits run on the GPU and the two are summed. Returns the slots held.
    pub fn set_hot_experts(&mut self, budget: u64, heat: &[Vec<u64>]) -> Result<usize> {
        ensure!(heat.len() == self.cfg.n_layers, "heat has {} layers", heat.len());
        let ne = self.cfg.n_expert as u64;
        let bytes = |n: &str| -> Result<&'g [u8]> {
            let t = self.shards.tensor(n).ok_or_else(|| anyhow!("missing tensor `{n}`"))?;
            ensure!(t.as_slice().len() as u64 == t.n_bytes() && t.n_bytes() % ne == 0, "`{n}`: bad size");
            Ok(t.as_slice())
        };
        let names = |l: usize| [format!("blk.{l}.ffn_gate_up_exps.weight"), format!("blk.{l}.ffn_down_exps.weight")];
        let per = names(0).iter().map(|n| bytes(n).map(|b| b.len() as u64 / ne)).sum::<Result<u64>>()?;
        let pick = hot_pick(heat, (budget / per) as usize);
        let mut held = 0;
        for (l, es) in pick.into_iter().enumerate() {
            self.hot[l] = None;
            if es.is_empty() {
                continue;
            }
            ensure!(
                self.host.as_ref().is_some_and(|(_, v)| v[l].is_some()),
                "hot experts for layer {l}, which is not on the host lane"
            );
            let [gu, dn] = names(l).map(|n| bytes(&n));
            let blobs = |b: &[u8]| -> Vec<u8> {
                let pe = b.len() / ne as usize;
                es.iter().flat_map(|&e| &b[e * pe..][..pe]).copied().collect()
            };
            let (gu, dn) = (blobs(gu?), blobs(dn?));
            let mut slot_of = vec![MISS; ne as usize];
            for (s, &e) in es.iter().enumerate() {
                slot_of[e] = s as u32;
            }
            held += es.len();
            self.hot[l] = Some(HotLayer {
                gu: host(self.gpu, || self.bytes(&gu))?,
                dn: host(self.gpu, || self.bytes(&dn))?,
                slot_of,
                expert_of: es.iter().map(|&e| e as u32).collect(),
                heat: vec![0.0; ne as usize],
            });
        }
        Ok(held)
    }

    pub fn set_batching(&mut self, on: bool) {
        self.batching = on;
    }

    /// Keep weights on the device between steps; turning a class off drops its copies.
    pub fn set_residency(&mut self, r: Residency) {
        self.residency = r;
        self.cache
            .get_mut()
            .retain(|name, _| if is_expert(name) { r.experts } else { r.dense });
    }

    /// Run the routed experts of `layers` on the CPU lane (straight out of the
    /// mmap) instead of uploading them; the rest stay on the GPU path.
    pub fn set_host_experts(&mut self, pool: ExpertPool, layers: &[usize]) -> Result<()> {
        let act = GluAct::Situ {
            beta: self.cfg.situ_beta,
            lb: self.cfg.situ_linear_beta,
        };
        let mut plans = vec![None; self.cfg.n_layers];
        for &l in layers {
            ensure!(l < plans.len(), "host layer {l} out of range");
            let t = |s: &str| {
                let name = format!("blk.{l}.{s}");
                self.shards
                    .tensor(&name)
                    .ok_or_else(|| anyhow!("missing tensor `{name}`"))
            };
            plans[l] = Some(LayerPlan::merged(
                self.shards,
                l,
                &t("ffn_gate_up_exps.weight")?,
                &t("ffn_down_exps.weight")?,
                act,
            )?);
        }
        self.cache
            .get_mut()
            .retain(|name, _| !layers.iter().any(|l| name.starts_with(&format!("blk.{l}.ffn_")) && is_expert(name)));
        self.host = Some((pool, plans));
        Ok(())
    }

    fn ty_of(shards: &ShardSet, name: &str) -> Result<QuantType> {
        let t = shards
            .tensor(name)
            .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
        QuantType::from_ggml(t.ty()).ok_or_else(|| anyhow!("`{name}`: no kernel for its type"))
    }

    fn buf(&self, floats: usize) -> Result<Owned<'g>> {
        Ok(Owned(self.gpu, gemv::alloc(self.gpu, (floats * 4) as u64)?))
    }

    fn bytes(&self, b: &[u8]) -> Result<Owned<'g>> {
        Ok(Owned(self.gpu, gemv::upload(self.gpu, b)?))
    }

    fn f32s(&self, v: &[f32]) -> Result<Owned<'g>> {
        Ok(Owned(self.gpu, ops::upload_f32(self.gpu, v)?))
    }

    fn u32s(&self, v: &[u32]) -> Result<Owned<'g>> {
        self.bytes(&ops::u32_to_bytes(v))
    }

    /// Readback / upload staging big enough for a `t`-token MoE step.
    fn stage(&self, t: usize) -> Result<std::cell::Ref<'_, (Owned<'g>, Owned<'g>, usize)>> {
        if self.stage.borrow().as_ref().is_none_or(|s| s.2 < t) {
            let (kx, n) = (self.cfg.n_expert_used, self.cfg.n_embd);
            let mk = |words: usize| -> Result<Owned<'g>> {
                Ok(Owned(self.gpu, gemv::staging(self.gpu, (words * 4) as u64)?))
            };
            *self.stage.borrow_mut() = Some((mk(2 * t * kx + t * n)?, mk(t * n)?, t));
        }
        Ok(std::cell::Ref::map(self.stage.borrow(), |s| s.as_ref().expect("set above")))
    }

    /// Raw GGUF bytes of `name` on the device (dims, type).
    fn weight(&self, name: &str) -> Result<(Rc<Owned<'g>>, Vec<usize>, QuantType)> {
        let t = self
            .shards
            .tensor(name)
            .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
        let ty = QuantType::from_ggml(t.ty()).ok_or_else(|| anyhow!("`{name}`: no kernel"))?;
        let bytes = t.as_slice();
        ensure!(bytes.len() as u64 == t.n_bytes(), "`{name}` is truncated");
        let dims = t.dims().iter().map(|&d| d as usize).collect();
        if let Some(w) = self.cache.borrow().get(name) {
            return Ok((w.clone(), dims, ty));
        }
        let w = Rc::new(
            host(self.gpu, || self.bytes(bytes)).with_context(|| format!("uploading {name}"))?,
        );
        let keep = if is_expert(name) {
            self.residency.experts
        } else {
            self.residency.dense
        };
        if keep {
            self.cache.borrow_mut().insert(name.to_string(), w.clone());
        }
        Ok((w, dims, ty))
    }

    fn has(&self, name: &str) -> bool {
        self.shards.tensor(name).is_some()
    }

    fn gemv(&mut self, ty: QuantType, rows: usize, cols: usize) -> Result<&GemvBatch<'g>> {
        use std::collections::hash_map::Entry;
        match self.gemvs.entry((ty, rows, cols)) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let g = GemvBatch::new(self.gpu, ty, rows, cols)?;
                // activations are packed [T, cols]; the map's row stride is padded_cols
                ensure!(
                    g.padded_cols() == cols,
                    "{ty:?} {rows}x{cols}: padded cols {} != cols",
                    g.padded_cols()
                );
                Ok(e.insert(g))
            }
        }
    }

    /// `out[T, rows] = W . x[T, cols]` for a dense 2-D tensor.
    fn proj(&mut self, name: &str, x: &Buf, tokens: usize) -> Result<(Owned<'g>, usize)> {
        let (w, dims, ty) = self.weight(name)?;
        ensure!(dims.len() == 2, "`{name}` is not 2-D");
        let (cols, rows) = (dims[0], dims[1]);
        let out = self.buf(tokens * rows)?;
        let gpu = self.gpu;
        let k = (ty, rows, cols, tokens);
        if !self.maps.get_mut().contains_key(&k) {
            let g = self.gemv(ty, rows, cols)?;
            let m = Owned(gpu, host(gpu, || proj::batch_map_buf(gpu, g, tokens))?);
            self.maps.get_mut().insert(k, m);
        }
        let map = &self.maps.get_mut()[&k];
        self.gemvs[&(ty, rows, cols)].run(&w, x, &out, map, tokens)?;
        Ok((out, rows))
    }

    /// Download for the tap, and upload its replacement if it returns one.
    fn tap(
        &self,
        tap: &mut Option<&mut Tap>,
        name: &str,
        slot: i32,
        b: &Buf,
        n: usize,
    ) -> Result<()> {
        let Some(t) = tap.as_mut() else {
            return Ok(());
        };
        host(self.gpu, || {
            let v = ops::download_f32(self.gpu, b, n)?;
            if let Some(r) = t(name, slot, &v) {
                ensure!(
                    r.len() == n,
                    "tap {name}-{slot}: {} values, want {n}",
                    r.len()
                );
                gemv::upload_into(self.gpu, b, bytemuck_f32(&r))?;
            }
            Ok(())
        })
    }

    fn copy(&self, dst: &Buf, dst_floats: usize, src: &Buf, floats: usize) -> Result<()> {
        gemv::copy_at(
            self.gpu,
            dst,
            (dst_floats * 4) as u64,
            src,
            0,
            (floats * 4) as u64,
        )
    }

    fn dup(&self, src: &Buf, floats: usize) -> Result<Owned<'g>> {
        let b = self.buf(floats)?;
        self.copy(&b, 0, src, floats)?;
        Ok(b)
    }

    /// Depth mix of `count` stack slots from `first` plus `cur` with `query`.
    #[allow(clippy::too_many_arguments)]
    fn depth_mix(
        &self,
        stack: &Buf,
        first: usize,
        count: usize,
        cur: &Buf,
        query: &str,
        tokens: usize,
    ) -> Result<(Owned<'g>, usize)> {
        let (q, dims, _) = self.weight(query)?;
        let nq = dims.get(1).copied().unwrap_or(1);
        let out = self.buf(tokens * nq * self.cfg.n_embd)?;
        self.mix.run(
            stack,
            first,
            count,
            tokens * self.cfg.n_embd,
            cur,
            &q,
            nq,
            &out,
            tokens,
        )?;
        Ok((out, nq))
    }

    /// Low-rank control of a `[T, n]` view by message column `c` of `msgs` `[T, 4, n]`.
    fn control(
        &self,
        layer: usize,
        c: usize,
        cur: &Buf,
        cv: (usize, usize),
        msgs: &Buf,
        tokens: usize,
    ) -> Result<Owned<'g>> {
        let n = self.cfg.n_embd;
        let p = format!("blk.{layer}.loop_{}_ctrl_", CTRL[c]);
        let (down, ..) = self.weight(&format!("{p}down.weight"))?;
        let (gain, ..) = self.weight(&format!("{p}gain.weight"))?;
        let (forget, ..) = self.weight(&format!("{p}forget.weight"))?;
        let out = self.buf(tokens * n)?;
        if !self.ctrl_ws.borrow().contains_key(&tokens) {
            let ws = host(self.gpu, || self.ctrl.workspace(self.gpu, tokens))?;
            self.ctrl_ws.borrow_mut().insert(tokens, ws);
        }
        let wss = self.ctrl_ws.borrow();
        self.ctrl.run(
            &wss[&tokens],
            cur,
            cv,
            msgs,
            (CTRL.len() * n, c * n),
            &ControlWeights {
                down: &down,
                gain: &gain,
                forget: &forget,
            },
            &out,
        )?;
        Ok(out)
    }

    /// An empty sequence for up to `ctx` tokens, fed at most `chunk` tokens per
    /// [`Self::forward`] step (the SWA ring holds `window + chunk - 1` cells).
    pub fn new_seq(&self, ctx: usize, chunk: usize) -> Result<InklingSeq<'g>> {
        ensure!(ctx > 0 && chunk > 0, "ctx and chunk must be > 0");
        let cfg = &self.cfg;
        let kv = cfg.kv_dim();
        let plan = cfg.plan();
        let mut layer_of = vec![0; plan.n_slots()];
        for st in plan.steps() {
            layer_of[st.slot] = st.layer;
        }
        let zeros = vec![0f32; (cfg.n_conv - 1) * (2 * kv + 2 * cfg.n_embd)];
        let mut slots = Vec::with_capacity(layer_of.len());
        for &l in &layer_of {
            let ring = cfg.swa[l];
            let cells = if ring {
                (cfg.sliding_window + chunk - 1).min(ctx)
            } else {
                ctx
            };
            slots.push(SlotKv {
                k: self.buf(cells * kv)?,
                v: self.buf(cells * kv)?,
                kpos: self.bytes(bytemuck_i32(&vec![-1i32; cells]))?,
                cells,
                ring,
                state: self.f32s(&zeros)?,
            });
        }
        Ok(InklingSeq {
            pos: 0,
            ctx,
            chunk,
            slots,
        })
    }

    /// Prefill `ids` as one fresh sequence and return the last token's logits.
    pub fn prefill(&mut self, ids: &[u32], tap: Option<&mut Tap>) -> Result<Vec<f32>> {
        let mut seq = self.new_seq(ids.len().max(1), ids.len().max(1))?;
        self.forward(&mut seq, ids, tap)
    }

    /// Feed `ids` at the sequence's next positions (in steps of `seq.chunk`) and
    /// return the last token's logits (`vocab` values).
    pub fn forward(
        &mut self,
        seq: &mut InklingSeq<'g>,
        ids: &[u32],
        mut tap: Option<&mut Tap>,
    ) -> Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "empty input");
        ensure!(
            seq.pos + ids.len() <= seq.ctx,
            "context full: {} + {} > {}",
            seq.pos,
            ids.len(),
            seq.ctx
        );
        let chunks: Vec<&[u32]> = ids.chunks(seq.chunk).collect();
        let last = chunks.len() - 1;
        for (i, c) in chunks.into_iter().enumerate() {
            let logits = self.step(seq, c, tap.as_deref_mut(), i == last)?;
            if let Some(l) = logits {
                return Ok(l);
            }
        }
        unreachable!("the last chunk returns logits")
    }

    /// One chunk at positions `seq.pos..`; logits only when `head`. The batch
    /// opened here is always closed (and its dropped buffers freed) on return.
    fn step(
        &mut self,
        seq: &mut InklingSeq<'g>,
        ids: &[u32],
        tap: Option<&mut Tap>,
        head: bool,
    ) -> Result<Option<Vec<f32>>> {
        let r = self.step_inner(seq, ids, tap, head);
        let closed = pause(self.gpu);
        bury(self.gpu);
        let out = r?;
        closed?;
        self.hot_tokens += ids.len();
        let t0 = std::time::Instant::now();
        self.hot_reload()?;
        self.prof[2] += t0.elapsed().as_secs_f64();
        Ok(out)
    }

    fn step_inner(
        &mut self,
        seq: &mut InklingSeq<'g>,
        ids: &[u32],
        mut tap: Option<&mut Tap>,
        head: bool,
    ) -> Result<Option<Vec<f32>>> {
        let t = ids.len();
        let p0 = seq.pos;
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let plan = cfg.plan();
        let tn = t * n;

        // input: embedding gather + token_embd_norm
        let raw = self.buf(tn)?;
        let emb_table = self
            .shards
            .tensor("token_embd.weight")
            .ok_or_else(|| anyhow!("missing token_embd"))?;
        let stage = self
            .emb
            .gather_staged(self.gpu, emb_table.as_slice(), ids, &raw)?;
        self.emb.release(self.gpu, &stage);
        let pos: Vec<i32> = (p0 as i32..(p0 + t) as i32).collect();
        let pos = self.bytes(bytemuck_i32(&pos))?;
        if self.batching {
            batch_cmd::begin(self.gpu)?;
        }
        let (g, ..) = self.weight("token_embd_norm.weight")?;
        let mut cur = self.buf(tn)?;
        self.norm.run(&raw, Some(&g), &cur, t)?;
        drop((raw, g));
        self.tap(&mut tap, "inp_norm", -1, &cur, tn)?;

        // block stack (oldest first) and the previous pass's bank
        let max_blocks = DepthMix::MAX_STATES - 1;
        let stack = self.buf(max_blocks * tn)?;
        let mut nblocks = 0usize;
        let bank = self.buf(max_blocks * tn)?;
        let mut bank_count = 0usize;
        let mut bank_out: Option<Owned<'g>> = None;
        let mut prelude: Option<(Owned<'g>, usize)> = None;

        for st in plan.steps() {
            let in_core = (plan.start..plan.end).contains(&st.layer);
            if in_core && st.layer == plan.start {
                if st.pass == 0 {
                    prelude = Some((self.dup(&cur, tn)?, nblocks));
                } else {
                    let (p, pblocks) = prelude.as_ref().expect("prelude saved at pass 0");
                    bank_count = nblocks - pblocks - 1;
                    for i in 0..bank_count {
                        gemv::copy_at(
                            self.gpu,
                            &bank,
                            (i * tn * 4) as u64,
                            &stack,
                            ((pblocks + 1 + i) * tn * 4) as u64,
                            (tn * 4) as u64,
                        )?;
                    }
                    bank_out = Some(std::mem::replace(&mut cur, self.dup(p, tn)?));
                    nblocks = *pblocks;
                }
            }
            let msgs = if in_core && st.pass > 0 {
                let (m, _) = self.depth_mix(
                    &bank,
                    0,
                    bank_count,
                    bank_out.as_ref().expect("bank"),
                    &format!("blk.{}.loop_msg_query.weight", st.layer),
                    t,
                )?;
                self.tap(&mut tap, "loop_msgs", st.slot as i32, &m, CTRL.len() * tn)?;
                Some(m)
            } else {
                None
            };
            cur = self.layer(
                st.layer,
                st.slot,
                cur,
                &stack,
                &mut nblocks,
                msgs.as_deref(),
                &seq.slots[st.slot],
                (&pos, p0),
                t,
                &mut tap,
            )?;
        }
        seq.pos += t;
        if !head {
            return Ok(None);
        }

        // output head on the last token
        let (mixed, _) = self.depth_mix(&stack, 0, nblocks, &cur, "output_res_query.weight", t)?;
        self.tap(&mut tap, "out_res_mix", -1, &mixed, tn)?;
        let (g, ..) = self.weight("output_norm.weight")?;
        let last = self.buf(n)?;
        self.norm
            .run_view(&mixed, (n, (t - 1) * n), Some(&g), &last, (n, 0), 1)?;
        self.tap(&mut tap, "result_norm", -1, &last, n)?;
        let (logits, rows) = self.proj("output.weight", &last, 1)?;
        if cfg.softcap != 0.0 {
            self.act.run(&logits, &logits, rows, cfg.softcap)?;
        }
        self.tap(&mut tap, "result_output", -1, &logits, cfg.vocab)?;
        pause(self.gpu)?;
        ops::download_f32(self.gpu, &logits, cfg.vocab).map(Some)
    }

    /// Write this step's K/V rows and positions into the slot's cells
    /// (two segments when the ring wraps).
    fn store(&self, s: &SlotKv, k: &Buf, v: &Buf, pos: &Buf, p0: usize, t: usize) -> Result<()> {
        let kvb = self.cfg.kv_dim() * 4;
        ensure!(s.ring || p0 + t <= s.cells, "slot cache full");
        // a wrapping write must not evict a cell this chunk's first query still sees
        ensure!(
            p0 + t <= s.cells || t + self.cfg.sliding_window - 1 <= s.cells,
            "chunk over the ring"
        );
        let mut done = 0;
        while done < t {
            let cell = (p0 + done) % s.cells;
            let m = (t - done).min(s.cells - cell);
            let (dst, src, len) = ((cell * kvb) as u64, (done * kvb) as u64, (m * kvb) as u64);
            gemv::copy_at(self.gpu, &s.k, dst, k, src, len)?;
            gemv::copy_at(self.gpu, &s.v, dst, v, src, len)?;
            gemv::copy_at(
                self.gpu,
                &s.kpos,
                (cell * 4) as u64,
                pos,
                (done * 4) as u64,
                (m * 4) as u64,
            )?;
            done += m;
        }
        Ok(())
    }

    /// llama.cpp `build_layer(il = slot, ip = layer, ...)`.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &mut self,
        layer: usize,
        slot: usize,
        inp_sa: Owned<'g>,
        stack: &Buf,
        nblocks: &mut usize,
        msgs: Option<&Buf>,
        kvs: &SlotKv,
        (pos, p0): (&Buf, usize),
        t: usize,
        tap: &mut Option<&mut Tap>,
    ) -> Result<Owned<'g>> {
        let state = &*kvs.state;
        let cfg = self.cfg.clone();
        let (n, hd, h, hkv) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_head_kv);
        let (kv, qd) = (cfg.kv_dim(), cfg.q_dim());
        let tn = t * n;
        let s = slot as i32;
        let b = |x: &str| format!("blk.{layer}.{x}.weight");
        let swa = cfg.swa[layer];
        let state_stride = 2 * kv + 2 * n;

        // attention inputs: views (buffer, stride, offset) of cur or the depth mix
        let mixed = if self.has(&b("attn_res_query")) {
            let (m, nq) = self.depth_mix(stack, 0, *nblocks, &inp_sa, &b("attn_res_query"), t)?;
            self.tap(tap, "attn_res_mix", s, &m, nq * tn)?;
            Some((m, nq))
        } else {
            None
        };
        let view = |j: usize| -> (&Buf, (usize, usize)) {
            match &mixed {
                Some((m, nq)) => (m, (nq * n, if *nq == 1 { 0 } else { j * n })),
                None => (&inp_sa, (n, 0)),
            }
        };
        let mut ctrl_out: Vec<Owned<'g>> = Vec::new();
        if let Some(msgs) = msgs {
            for j in 0..3 {
                let (buf, cv) = view(j);
                let o = self.control(layer, j, buf, cv, msgs, t)?;
                self.tap(
                    tap,
                    ["loop_ctrl_q", "loop_ctrl_k", "loop_ctrl_v"][j],
                    s,
                    &o,
                    tn,
                )?;
                ctrl_out.push(o);
            }
        }
        let input = |j: usize| -> (&Buf, (usize, usize)) {
            if ctrl_out.is_empty() {
                view(j)
            } else {
                (&ctrl_out[j], (n, 0))
            }
        };
        // input norms: SWA shares attn_norm (one input unless messages split it)
        let norm_names: [String; 3] = if swa {
            [b("attn_norm"), b("attn_norm"), b("attn_norm")]
        } else {
            [
                b("attn_q_inp_norm"),
                b("attn_k_inp_norm"),
                b("attn_v_inp_norm"),
            ]
        };
        let shared = swa && msgs.is_none();
        let mut ins: Vec<Owned<'g>> = Vec::new();
        for (j, name) in norm_names
            .iter()
            .enumerate()
            .take(if shared { 1 } else { 3 })
        {
            let (g, ..) = self.weight(name)?;
            let (x, xv) = input(j);
            let o = self.buf(tn)?;
            self.norm.run_view(x, xv, Some(&g), &o, (n, 0), t)?;
            ins.push(o);
        }
        drop(ctrl_out);
        drop(mixed);
        let in_at = |j: usize| if shared { 0 } else { j };
        self.tap(tap, "attn_in_q", s, &ins[0], tn)?;
        if !shared {
            self.tap(tap, "attn_in_k", s, &ins[1], tn)?;
            self.tap(tap, "attn_in_v", s, &ins[2], tn)?;
        }
        let (in_q, in_k, in_v) = (&ins[in_at(0)], &ins[in_at(1)], &ins[in_at(2)]);

        // projections + short convolutions (dynamic taps on full layers)
        let (q, _) = self.proj(&b("attn_q"), in_q, t)?;
        let (kp, _) = self.proj(&b("attn_k"), in_k, t)?;
        let (vp, _) = self.proj(&b("attn_v"), in_v, t)?;
        let dynk = if swa {
            None
        } else {
            Some(self.proj(&b("attn_k_conv_proj"), in_k, t)?)
        };
        let dynv = if swa {
            None
        } else {
            Some(self.proj(&b("attn_v_conv_proj"), in_v, t)?)
        };
        let (kw, ..) = self.weight(&b("attn_k_conv"))?;
        let (vw, ..) = self.weight(&b("attn_v_conv"))?;
        let k = cfg.n_conv;
        self.tap(tap, "Kproj", s, &kp, t * kv)?;
        self.tap(tap, "Vproj", s, &vp, t * kv)?;
        let hk = dynk.as_ref().map(|(d, rows)| (&**d, rows / k));
        let hv = dynv.as_ref().map(|(d, rows)| (&**d, rows / k));
        self.conv
            .run(state, state_stride, 0, &kp, &kw, k, hk, &kp, kv, t)?;
        self.conv
            .run(state, state_stride, kv, &vp, &vw, k, hv, &vp, kv, t)?;
        drop((dynk, dynv, kw, vw));

        // clips: Q/R by channel scale (+ tau on full layers), K plain, V per-head RMS
        let (qc, ..) = self.weight(&b("attn_q_clip"))?;
        let qcur = self.buf(t * qd)?;
        if swa {
            self.clip_q.run(&q, Some(&qc), None, &qcur, t)?;
        } else {
            self.clip_q_tau.run(&q, Some(&qc), Some(pos), &qcur, t)?;
        }
        let kcur = self.buf(t * kv)?;
        self.clip_k.run(&kp, None, None, &kcur, t)?;
        let vcur = self.buf(t * kv)?;
        self.head_norm.run(&vp, None, &vcur, t * hkv)?;
        let (r, _) = self.proj(&b("attn_rel"), in_q, t)?;
        let (rc, ..) = self.weight(&b("attn_rel_clip"))?;
        let rcur = self.buf(t * cfg.n_rel * h)?;
        if swa {
            self.clip_r.run(&r, Some(&rc), None, &rcur, t)?;
        } else {
            self.clip_r_tau.run(&r, Some(&rc), Some(pos), &rcur, t)?;
        }
        drop((q, kp, vp, r, qc, rc));
        self.tap(tap, "Qcur", s, &qcur, t * qd)?;
        self.tap(tap, "Kcur", s, &kcur, t * kv)?;
        self.tap(tap, "Vcur", s, &vcur, t * kv)?;
        self.tap(tap, "Rcur", s, &rcur, t * cfg.n_rel * h)?;
        self.store(kvs, &kcur, &vcur, pos, p0, t)?;
        drop((kcur, vcur));

        // attention: rel bias + per-token sink logit
        let (sinks, _) = self.proj(&b("attn_sink_proj"), in_q, t)?;
        self.tap(tap, "attn_sink", s, &sinks, t * h)?;
        let (bias, bdims, _) = self.weight(&b("attn_rel_bias"))?;
        let kqv = self.buf(t * qd)?;
        self.attn.run(
            &AttnShape {
                heads: h,
                kv_heads: hkv,
                head_dim: hd,
                window: if swa { cfg.sliding_window } else { 0 },
                scale: 1.0 / (hd as f32).sqrt(),
            },
            &qcur,
            pos,
            t,
            &kvs.k,
            &kvs.v,
            &kvs.kpos,
            kvs.cells.min(p0 + t),
            Some(&RelBias {
                r: &rcur,
                bias: &bias,
                rel_dim: cfg.n_rel,
                extent: bdims[1],
            }),
            Some(&Sink {
                sink: &sinks,
                per_token: true,
                scale: 1.0,
            }),
            &kqv,
        )?;
        drop((qcur, rcur, sinks, bias));
        self.tap(tap, "kqv_out", s, &kqv, t * qd)?;

        let (gate, _) = self.proj(&b("attn_gate"), in_q, t)?;
        self.add.run_gated(&kqv, &gate, &kqv, t * qd)?;
        drop(gate);
        drop(ins);
        self.tap(tap, "attn_gated", s, &kqv, t * qd)?;
        let (wo, _) = self.proj(&b("attn_output"), &kqv, t)?;
        drop(kqv);
        self.tap(tap, "attn_wo", s, &wo, tn)?;
        let (pw, ..) = self.weight(&b("attn_post_conv"))?;
        self.conv
            .run(state, state_stride, 2 * kv, &wo, &pw, k, None, &wo, n, t)?;
        drop(pw);
        self.tap(tap, "attn_out", s, &wo, tn)?;

        // a block start pushes the old stream; otherwise the residual add
        if layer.is_multiple_of(cfg.attn_res_block) {
            ensure!(*nblocks < DepthMix::MAX_STATES - 1, "block stack full");
            self.copy(stack, *nblocks * tn, &inp_sa, tn)?;
            *nblocks += 1;
        } else {
            self.add.run(&inp_sa, &wo, &wo, tn, 1.0)?;
        }
        let ffn_inp = wo;
        drop(inp_sa);
        self.tap(tap, "ffn_inp", s, &ffn_inp, tn)?;

        // MoE
        let (mut x, _) = self.depth_mix(stack, 0, *nblocks, &ffn_inp, &b("ffn_res_query"), t)?;
        self.tap(tap, "ffn_res_mix", s, &x, tn)?;
        if let Some(msgs) = msgs {
            x = self.control(layer, 3, &x, (n, 0), msgs, t)?;
            self.tap(tap, "loop_ctrl_ffn", s, &x, tn)?;
        }
        let (g, ..) = self.weight(&b("ffn_norm"))?;
        let fin = self.buf(tn)?;
        self.norm.run(&x, Some(&g), &fin, t)?;
        drop((x, g));
        self.tap(tap, "ffn_in", s, &fin, tn)?;
        let t0 = std::time::Instant::now();
        let moe = self.moe(layer, &fin, t, slot, tap)?;
        self.prof[1] += t0.elapsed().as_secs_f64();
        drop(fin);
        self.tap(tap, "ffn_moe_out", s, &moe, tn)?;
        let (fw, ..) = self.weight(&b("ffn_post_conv"))?;
        self.conv.run(
            state,
            state_stride,
            2 * kv + n,
            &moe,
            &fw,
            k,
            None,
            &moe,
            n,
            t,
        )?;
        drop(fw);
        self.tap(tap, "ffn_out", s, &moe, tn)?;
        self.add.run(&ffn_inp, &moe, &ffn_inp, tn, 1.0)?;
        self.tap(tap, "l_out", s, &ffn_inp, tn)?;
        Ok(ffn_inp)
    }

    /// Sigmoid-routed SITU experts with the merged gate_up tensor.
    fn moe(
        &mut self,
        layer: usize,
        x: &Buf,
        t: usize,
        slot: usize,
        tap: &mut Option<&mut Tap>,
    ) -> Result<Owned<'g>> {
        let (ne, kx, n) = (self.cfg.n_expert, self.cfg.n_expert_used, self.cfg.n_embd);
        let b = |s: &str| format!("blk.{layer}.{s}");
        let (logits, _) = self.proj(&b("ffn_gate_inp.weight"), x, t)?;
        self.tap(tap, "ffn_moe_logits", slot as i32, &logits, t * ne)?;
        let (bias, ..) = self.weight(&b("exp_probs_b.bias"))?;
        let ids = Owned(self.gpu, gemv::alloc(self.gpu, self.router.ids_bytes(t))?);
        let wts = Owned(
            self.gpu,
            gemv::alloc(self.gpu, self.router.weights_bytes(t))?,
        );
        self.router.run_biased(&logits, &bias, &ids, &wts, t)?;
        drop((logits, bias));
        let pairs = t * kx;
        let gpu = self.gpu;
        let lane = self.host.as_ref().and_then(|(_, v)| v[layer]);

        // readback through staging (recorded into the batch): ids, then for the
        // host lane the weights and x; one pause covers it and every upload below
        // decode on an all-resident layer: maps built on the device from the ids,
        // no readback (and no routed counts), so the per-layer host round trip goes
        let device = lane.is_none() && t == 1 && !self.host_maps;
        let need = if lane.is_some() { 2 * pairs + t * n } else if device { 0 } else { pairs };
        let mut raw = vec![0u8; need * 4];
        let open = if device {
            false
        } else {
            let stage = self.stage(t)?;
            let dn = &stage.0;
            let pb = (pairs * 4) as u64;
            gemv::copy_at(gpu, dn, 0, &ids, 0, pb)?;
            if lane.is_some() {
                gemv::copy_at(gpu, dn, pb, &wts, 0, pb)?;
                gemv::copy_at(gpu, dn, 2 * pb, x, 0, (t * n * 4) as u64)?;
            }
            let open = pause(gpu)?;
            gpu.read_host(dn, &mut raw).map_err(|e| anyhow!("{e}"))?;
            open
        };
        let words: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sel = &words[..if device { 0 } else { pairs }];
        ensure!(
            sel.iter().all(|&e| (e as usize) < ne),
            "router id out of range"
        );
        for &e in sel {
            self.routed[layer][e as usize] += 1;
        }
        if let Some(plan) = lane {
            let f = |w: &[u32]| -> Vec<f32> { w.iter().map(|&b| f32::from_bits(b)).collect() };
            let w = f(&words[pairs..2 * pairs]);
            let xs = f(&words[2 * pairs..]);
            // hot pool split: held experts run on the GPU, the rest on the lane
            if let Some(h) = self.hot[layer].as_mut() {
                for &e in sel {
                    h.heat[e as usize] += 1.0;
                }
            }
            let slot_of = self.hot[layer].as_ref().map(|h| &h.slot_of);
            let mut hits: Vec<(usize, u32)> = Vec::new();
            let (mut mid, mut mw) = (vec![Vec::new(); t], vec![Vec::new(); t]);
            for (p, &e) in sel.iter().enumerate() {
                match slot_of.map_or(MISS, |s| s[e as usize]) {
                    MISS => {
                        mid[p / kx].push(e as usize);
                        mw[p / kx].push(w[p]);
                    }
                    s => hits.push((p, s)),
                }
            }
            self.hot_stats.0 += hits.len() as u64;
            self.hot_stats.1 += if slot_of.is_some() { pairs as u64 } else { 0 };
            let lane = Lane { plan, ids: &mid, weights: &mw, x: &xs };
            if !hits.is_empty() {
                return self.moe_hot(layer, x, t, &w, &hits, lane, open);
            }
            let out = self.run_lane(lane, t)?;
            gpu.write_host(&self.stage(t)?.1, bytemuck_f32(&out))
                .map_err(|e| anyhow!("{e}"))?;
            resume(gpu, open)?;
            let o = self.buf(t * n)?;
            gemv::copy_at(gpu, &o, 0, &self.stage(t)?.1, 0, (t * n * 4) as u64)?;
            return Ok(o);
        }

        // GPU experts: maps from the ids, uploaded while the batch is closed
        let (gu_w, gu_dims, gu_ty) = self.weight(&b("ffn_gate_up_exps.weight"))?;
        let (gu_rows, cols) = (gu_dims[1], gu_dims[0]);
        let half = gu_rows / 2;
        let (dn_w, dn_dims, dn_ty) = self.weight(&b("ffn_down_exps.weight"))?;
        ensure!(
            dn_dims[0] == half && dn_dims[1] == n,
            "ffn_down_exps shape {dn_dims:?}"
        );
        let gu_map = if device {
            self.gemv(gu_ty, gu_rows, cols)?;
            let g = &self.gemvs[&(gu_ty, gu_rows, cols)];
            let m = Owned(gpu, gemv::alloc(gpu, (2 * g.rows(pairs) * 4) as u64)?);
            self.moe_map.run(&ids, &m, g, pairs, 0, false)?;
            m
        } else {
            let g = self.gemv(gu_ty, gu_rows, cols)?;
            let words: Vec<u32> = sel.iter().map(|&e| e * g.blob_words()).collect();
            let mut map = g.map_for_blobs(&words, XLayout::PerExpert);
            let (rows, stride) = (g.rows(pairs), (g.padded_cols() / 4) as u32);
            // gate_up: pair p = t*k + j reads activation row t
            for p in 0..pairs {
                for r in 0..gu_rows {
                    map[rows + p * gu_rows + r] = stride * (p / kx) as u32;
                }
            }
            Owned(gpu, gemv::upload(gpu, &breccia_kernels::batch::map_to_bytes(&map))?)
        };
        let dn_map = if device {
            self.gemv(dn_ty, n, half)?;
            let g = &self.gemvs[&(dn_ty, n, half)];
            let m = Owned(gpu, gemv::alloc(gpu, (2 * g.rows(pairs) * 4) as u64)?);
            self.moe_map.run(&ids, &m, g, pairs, 0, true)?;
            m
        } else {
            let g = self.gemv(dn_ty, n, half)?;
            let words: Vec<u32> = sel.iter().map(|&e| e * g.blob_words()).collect();
            let map = g.map_for_blobs(&words, XLayout::PerExpert);
            Owned(gpu, gemv::upload(gpu, &breccia_kernels::batch::map_to_bytes(&map))?)
        };
        if !self.fan.get_mut().contains_key(&t) {
            let fan = [
                self.u32s(&(0..pairs as u32).collect::<Vec<_>>())?,
                self.f32s(&vec![0.0; t * n])?,
                self.f32s(&vec![0.0; t])?,
            ];
            self.fan.get_mut().insert(t, fan);
        }
        resume(gpu, open)?;

        let gu = self.buf(pairs * gu_rows)?;
        self.gemvs[&(gu_ty, gu_rows, cols)].run(&gu_w, x, &gu, &gu_map, pairs)?;
        drop(gu_w);
        let situ = self.buf(pairs * half)?;
        self.glu.run_fused(&gu, &situ, pairs, half)?;
        drop(gu);
        // down: pair p reads SITU row p
        let down = self.buf(pairs * n)?;
        self.gemvs[&(dn_ty, n, half)].run(&dn_w, &situ, &down, &dn_map, pairs)?;
        drop((dn_w, situ));

        let out = self.buf(t * n)?;
        let [slots, shared, gate0] = &self.fan.get_mut()[&t];
        self.fanin
            .run(&down, slots, &wts, shared, gate0, &out, t, kx, n)?;
        Ok(out)
    }
}

/// One CPU-lane MoE call: per-token expert ids/weights (any count, misses only
/// under a hot pool) over the read-back activations.
struct Lane<'a> {
    plan: LayerPlan,
    ids: &'a [Vec<usize>],
    weights: &'a [Vec<f32>],
    x: &'a [f32],
}

impl<'g> InklingForward<'g> {
    fn run_lane(&mut self, l: Lane<'_>, t: usize) -> Result<Vec<f32>> {
        let mut out = vec![0f32; t * self.cfg.n_embd];
        if l.ids.iter().all(Vec::is_empty) {
            return Ok(out);
        }
        let t0 = std::time::Instant::now();
        let r = self.run_lane_inner(l, t, &mut out);
        self.prof[0] += t0.elapsed().as_secs_f64();
        r.map(|()| out)
    }

    fn run_lane_inner(&mut self, l: Lane<'_>, t: usize, out: &mut [f32]) -> Result<()> {
        let pool = &mut self.host.as_mut().expect("lane").0;
        if t == 1 {
            pool.moe_topk(&l.plan, &l.ids[0], &l.weights[0], l.x, out)?;
        } else {
            let routes: Vec<ChunkRoute> = (0..t)
                .map(|i| ChunkRoute {
                    ids: &l.ids[i],
                    weights: &l.weights[i],
                })
                .collect();
            pool.moe_topk_chunk(&l.plan, &routes, l.x, out)?;
        }
        Ok(())
    }

    /// Host-visible scratch `i` holding `b`, copied into a fresh device buffer
    /// by a recorded copy. Reuse is safe: every MoE step drains the batch before
    /// the next one writes.
    fn hot_put(&self, i: usize, b: &[u8]) -> Result<Owned<'g>> {
        let mut st = self.hot_stage.borrow_mut();
        if st.len() <= i {
            st.resize_with(i + 1, || None);
        }
        if st[i].as_ref().is_none_or(|s: &Owned| s.size < b.len() as u64) {
            st[i] = Some(Owned(self.gpu, gemv::staging(self.gpu, (b.len() as u64).next_power_of_two())?));
        }
        let s = st[i].as_ref().expect("set above");
        self.gpu.write_host(s, b).map_err(|e| anyhow!("{e}"))?;
        let d = Owned(self.gpu, gemv::alloc(self.gpu, b.len() as u64)?);
        gemv::copy_at(self.gpu, &d, 0, s, 0, b.len() as u64)?;
        Ok(d)
    }

    /// A hot-pool step, entered with the batch paused (`open`): the `hits`
    /// (pair, slot) run through the layer's held blobs and are submitted; the
    /// lane computes the misses meanwhile; the fan-in adds the lane's sum as
    /// its shared row (gate σ(100) = 1) to the hits, misses weighted 0.
    #[allow(clippy::too_many_arguments)]
    fn moe_hot(
        &mut self,
        layer: usize,
        x: &Buf,
        t: usize,
        w: &[f32],
        hits: &[(usize, u32)],
        lane: Lane<'_>,
        open: bool,
    ) -> Result<Owned<'g>> {
        let (kx, n) = (self.cfg.n_expert_used, self.cfg.n_embd);
        let (pairs, h, gpu) = (t * kx, hits.len(), self.gpu);
        let b = |s: &str| format!("blk.{layer}.{s}");
        let dims = |name: &str| -> Result<Vec<usize>> {
            let t = self.shards.tensor(name).ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
            Ok(t.dims().iter().map(|&d| d as usize).collect())
        };
        let (gu_name, dn_name) = (b("ffn_gate_up_exps.weight"), b("ffn_down_exps.weight"));
        let (gu_dims, dn_dims) = (dims(&gu_name)?, dims(&dn_name)?);
        let (gu_ty, dn_ty) = (Self::ty_of(self.shards, &gu_name)?, Self::ty_of(self.shards, &dn_name)?);
        let (gu_rows, cols) = (gu_dims[1], gu_dims[0]);
        let half = gu_rows / 2;
        ensure!(dn_dims[0] == half && dn_dims[1] == n, "ffn_down_exps shape {dn_dims:?}");
        let gu_map = {
            let g = self.gemv(gu_ty, gu_rows, cols)?;
            let words: Vec<u32> = hits.iter().map(|&(_, s)| s * g.blob_words()).collect();
            let mut map = g.map_for_blobs(&words, XLayout::PerExpert);
            let (rows, stride) = (g.rows(h), (g.padded_cols() / 4) as u32);
            // hit q = pair p reads activation row p / k
            for (q, &(p, _)) in hits.iter().enumerate() {
                for r in 0..gu_rows {
                    map[rows + q * gu_rows + r] = stride * (p / kx) as u32;
                }
            }
            breccia_kernels::batch::map_to_bytes(&map)
        };
        let dn_map = {
            let g = self.gemv(dn_ty, n, half)?;
            let words: Vec<u32> = hits.iter().map(|&(_, s)| s * g.blob_words()).collect();
            let map = g.map_for_blobs(&words, XLayout::PerExpert);
            breccia_kernels::batch::map_to_bytes(&map)
        };
        let (mut slots, mut wv) = (vec![0u32; pairs], vec![0f32; pairs]);
        for (q, &(p, _)) in hits.iter().enumerate() {
            slots[p] = q as u32;
            wv[p] = w[p];
        }
        resume(gpu, open)?;
        let gu_map = self.hot_put(0, &gu_map)?;
        let dn_map = self.hot_put(1, &dn_map)?;
        let slots = self.hot_put(2, &ops::u32_to_bytes(&slots))?;
        let wv = self.hot_put(3, bytemuck_f32(&wv))?;
        let one = self.hot_put(4, bytemuck_f32(&vec![100.0; t]))?;
        let hl = self.hot[layer].as_ref().expect("hot layer");
        let gu = self.buf(h * gu_rows)?;
        self.gemvs[&(gu_ty, gu_rows, cols)].run(&hl.gu, x, &gu, &gu_map, h)?;
        let situ = self.buf(h * half)?;
        self.glu.run_fused(&gu, &situ, h, half)?;
        let down = self.buf(h * n)?;
        self.gemvs[&(dn_ty, n, half)].run(&hl.dn, &situ, &down, &dn_map, h)?;
        // overlap: the hits run while the lane computes the misses. Nothing
        // recorded above may drop before the wait (no batch open to defer it).
        let seg = if open { Some(batch_cmd::end_submit(gpu, None)?) } else { None };
        let lane_out = self.run_lane(lane, t);
        if let Some(seg) = seg {
            batch_cmd::wait(&[gpu], seg)?;
            bury(gpu);
        }
        let lane_out = lane_out?;
        gpu.write_host(&self.stage(t)?.1, bytemuck_f32(&lane_out))
            .map_err(|e| anyhow!("{e}"))?;
        resume(gpu, open)?;
        let shared = self.buf(t * n)?;
        gemv::copy_at(gpu, &shared, 0, &self.stage(t)?.1, 0, (t * n * 4) as u64)?;
        let out = self.buf(t * n)?;
        self.fanin.run(&down, &slots, &wv, &shared, &one, &out, t, kx, n)?;
        Ok(out)
    }
}

fn is_expert(name: &str) -> bool {
    name.ends_with("_exps.weight")
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn bytemuck_i32(v: &[i32]) -> &[u8] {
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
