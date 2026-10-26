//! The forward-pass graph: 48 blocks, one token at a time.
//!
//! ```text
//! token ──▶ emb gather (4 rows = the hc copies) ──▶ R [hc, n_embd]
//! for L in 0..48:
//!   if PLE(L):  R = PLE(R, ple_emb)                       host block, GPU projections
//!   HC_READ(attn) ──▶ mixed
//!   GDN(L) | QSA(L) ──▶ block_out
//!   HC_WRITE(attn)  R += block_out · 2σ(inject/hc)
//!   HC_READ(ffn)  ──▶ mixed
//!   router ──▶ ids, w   ·   shared expert ──▶ sh[n_embd] · scalar gate
//!   ▼ one readback                     ▲ one upload
//!   host expert lane: Σ_j w_j down_j(silu(gate_j x) · up_j x)
//!   HC_WRITE(ffn)   R += block_out · 2σ(inject/hc)
//! HC_READ_FINAL ──▶ cur [n_embd] ──▶ lm_head GEMV ──▶ logits [vocab] ──▶ sampler
//! ```
//!
//! # Why the host round trip is on the critical path
//!
//! The expert lane is `breccia-kernels::cpu`'s, whose entry point takes `&[f32]`
//! and `&mut [f32]`: the MoE activation is a host vector by contract
//! (`docs/phase-3/08` §1). One block therefore costs exactly one 10 KiB readback
//! and one 10 KiB upload — two `vkWaitForFences` per block, which is what
//! [`crate::xfer::Staging`] exists to keep at two rather than ten.
//!
//! # Dispatch discipline
//!
//! Every kernel entry point in `breccia-kernels` records, submits and waits
//! (`ops::Pipe::run` → `Gpu::wall_run`), so the host is synchronous with the GPU
//! at every call boundary. That is what makes the per-stage timers honest without
//! inserting a single extra sync: each delta is the real time that stage occupies
//! on the critical path, submit overhead included.

use anyhow::{anyhow, bail, Context, Result};
use breccia_vk::ctx::Buf;

use breccia_kernels::batch_cmd;
use breccia_kernels::cpu::{self, ExpertPool};
use breccia_kernels::ops::{
    f32_from_bytes, f32_to_bytes, EmbGather, HcMixer, HcWeights, HcWorkspace, Router, SwiGlu,
};
use breccia_kernels::{
    Gdn, GdnProjections, GdnState, GdnWeights, GdnWorkspace, Proj, Qsa, QsaProjections, QsaState,
    QsaWeights, QsaWorkspace, QuantType,
};

use crate::config::LayerKind;
use crate::model::{HcW, Model, PipelineSet};
use crate::ple::PleHost;
use crate::sampler::{Sampler, Sampling};
use crate::stats::{Checksum, GdnTrace, LayerTrace};
use crate::xfer::{Mailbox, Staging};

/// Per-stage timer slots, in graph order.
pub const STAGES: [&str; 16] = [
    "embed",
    "ple",
    "hc-read(attn)",
    "mixer",
    "hc-write(attn)",
    "hc-read(ffn)",
    "router",
    "shared-exp",
    "wait-gpu",
    "xfer-down",
    "experts(host)",
    "xfer-up",
    "hc-write(ffn)",
    "handoff",
    "final-hc",
    "head",
];

const EMBED: usize = 0;
const PLE: usize = 1;
const HC_READ_A: usize = 2;
const MIXER: usize = 3;
const HC_WRITE_A: usize = 4;
const HC_READ_F: usize = 5;
const ROUTER: usize = 6;
const SHARED: usize = 7;
const WAIT_GPU: usize = 8;
const XFER_DOWN: usize = 9;
const EXPERTS: usize = 10;
const XFER_UP: usize = 11;
const HC_WRITE_F: usize = 12;
const HANDOFF: usize = 13;
const FINAL_HC: usize = 14;
const HEAD: usize = 15;

/// Batches the `BRECCIA_GPU_TIME` probe holds before reading the pool out and
/// starting again (256 batches = ~5 tokens of a two-stream run).
const GPU_PROBE_BATCHES: usize = 256;

/// Wall-clock accounting: host-side deltas around synchronous dispatch groups.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Timers {
    pub on: bool,
    pub ns: [u64; STAGES.len()],
    /// Dispatches recorded while each stage was the current one.
    pub dispatches: [u32; STAGES.len()],
    #[serde(skip)]
    last: Option<std::time::Instant>,
    #[serde(skip)]
    stage: usize,
    #[serde(skip)]
    dispatch_mark: u32,
}

impl Timers {
    fn start(&mut self) {
        if self.on {
            self.stage = EMBED;
            self.last = Some(std::time::Instant::now());
        }
    }
    /// Attribute the time since the last mark to the current stage, then advance.
    fn mark(&mut self, stage: usize) {
        if !self.on {
            return;
        }
        let now = std::time::Instant::now();
        if let Some(last) = self.last {
            self.ns[self.stage] += now.duration_since(last).as_nanos() as u64;
        }
        self.last = Some(now);
        self.stage = stage;
        // Dispatches recorded so far belong to the stage we are leaving.
        self.dispatch_mark = batch_cmd::recorded();
        let _ = stage;
    }

    /// Account the dispatches recorded since the previous mark to `stage`.
    fn count_dispatches(&mut self, stage: usize, total: u32) {
        if !self.on {
            return;
        }
        self.dispatches[stage] += total.saturating_sub(self.dispatch_mark);
    }
    fn stop(&mut self) {
        if !self.on {
            return;
        }
        if let Some(last) = self.last.take() {
            self.ns[self.stage] += last.elapsed().as_nanos() as u64;
        }
    }
    pub fn clear(&mut self) {
        self.ns = [0; STAGES.len()];
        self.dispatches = [0; STAGES.len()];
        self.last = None;
        self.stage = 0;
        self.dispatch_mark = 0;
    }

    /// `(stage, dispatches per token)` for the report.
    pub fn dispatch_table(&self, tokens: usize) -> Vec<(&'static str, f64)> {
        let n = tokens.max(1) as f64;
        STAGES
            .iter()
            .zip(self.dispatches.iter())
            .map(|(name, d)| (*name, *d as f64 / n))
            .collect()
    }
    pub fn total_s(&self) -> f64 {
        self.ns.iter().sum::<u64>() as f64 / 1e9
    }
    /// `(stage, milliseconds per token)`, for the report table.
    pub fn table(&self, tokens: usize) -> Vec<(&'static str, f64)> {
        let n = tokens.max(1) as f64;
        STAGES
            .iter()
            .zip(self.ns.iter())
            .map(|(name, ns)| (*name, *ns as f64 / 1e6 / n))
            .collect()
    }
}

/// Everything one device owns during a session.
pub struct Venue<'m> {
    pub pipes: PipelineSet<'m>,
    pub hc: HcMixer<'m>,
    pub final_hc: HcMixer<'m>,
    /// `Some` for the whole session; taken only by `Drop`, which is why it is
    /// an `Option` (`HcMixer::release` consumes the workspace).
    pub hc_ws: Option<HcWorkspace>,
    pub gdn: Option<Gdn<'m>>,
    pub gdn_ws: Option<GdnWorkspace>,
    pub qsa: Option<Qsa<'m>>,
    pub router: Router<'m>,
    pub swiglu: SwiGlu<'m>,
    pub emb: EmbGather<'m>,
    pub emb_ple: EmbGather<'m>,
    /// The residual stack `[hc, n_embd]`.
    pub r: Buf,
    /// The token mixer's / MoE's output, as a **host-visible mailbox**: the MoE
    /// result is written straight into the mapping instead of being uploaded
    /// through a submit + fence.
    pub block_out: Mailbox,
    pub sh: Buf,
    pub sh_gate_logit: Buf,
    pub shexp_gate: Buf,
    pub shexp_up: Buf,
    pub shexp_act: Buf,
    pub router_logits: Buf,
    pub router_ids: Buf,
    pub router_w: Buf,
    pub ple_key_out: Buf,
    pub ple_value_out: Buf,
    pub ple_emb: Buf,
    pub logits: Option<Buf>,
    /// Device-resident MoE scratch (`--gpu-experts`): `[k, n_ff]` gate/up/act,
    /// `[k, n_embd]` down, and the two row maps the device fills in.
    pub moe_gate: Buf,
    pub moe_up: Buf,
    pub moe_act: Buf,
    pub moe_down: Buf,
    pub moe_map_gate: Buf,
    pub moe_map_up: Buf,
    pub moe_map_down: Buf,
    pub moe_map: breccia_kernels::MoeMap<'m>,
    pub moe_combine: breccia_kernels::MoeCombine<'m>,
    pub name: String,
}

/// Session setup.
#[derive(Clone, Default)]
pub struct SessionCfg {
    /// KV cells each QSA layer may use (0 = the default, 256).
    pub ctx_capacity: usize,
    /// Host expert-lane threads (`0` = the lane's default).
    pub threads: usize,
    /// Collect per-stage timings.
    pub timers: bool,
    /// Record each GPU phase into one command buffer and submit it once
    /// (`crate::graph`'s batching). **On by default**: 0 validation messages, and
    /// the generated tokens are bit-identical with it on and off on all three
    /// device configurations (`docs/phase-4/01-forward-pass.md` §8).
    pub batching: bool,
    /// Record a block's readback (`mixed`, router ids/weights, shared expert out
    /// and gate) as **copies inside the block's own batch**, and read them out of
    /// the staging mapping after the wait — instead of a `vkQueueSubmit` +
    /// `vkWaitForFences` of their own (§8o measured 0.55 ms/block of host time in
    /// a two-stream pair, where the GPU has slack).
    pub inbatch_read: bool,
    /// Enable the GPU-time probe (`BRECCIA_GPU_TIME=1` also turns it on).
    /// It stamps every batch, so a two-stream run can report per-card
    /// utilisation instead of only fence waits.
    pub probe: bool,
    /// Share one host expert lane between sessions (two-stream decode).
    ///
    /// The lane's workers are `threads` OS threads; two streams run their lanes
    /// **serially** (each one hides under the other's GPU work), so a shared pool
    /// is the right shape: 16 threads serve both streams, not 32.
    pub pool: Option<std::sync::Arc<std::sync::Mutex<ExpertPool>>>,
}

/// Reset the host-side record/wait accumulators (per measured window).
pub fn batch_cmd_reset() {
    batch_cmd::reset_record_seconds();
}

/// The per-stage digests [`Session::submit_block`] takes while recording — the
/// debug trace's inputs, carried to [`Session::finish_block`] because the trace
/// also needs the router's ids and the MoE result.
#[derive(Clone, Debug, Default)]
pub struct BlockDigests {
    pub mixed_attn: Option<Checksum>,
    pub mixer_out: Option<Checksum>,
    pub mixed_ffn: Option<Checksum>,
}

/// Byte offsets of a block's readback inside the staging block region:
/// `mixed | router ids | router weights | shared out | shared gate`.
fn lane_offsets(n_embd: usize, k: usize) -> [u64; 5] {
    let mut o = [0u64; 5];
    let mut at = 0u64;
    for (i, n) in [n_embd * 4, k * 4, k * 4, n_embd * 4, 4]
        .into_iter()
        .enumerate()
    {
        o[i] = at;
        at += n as u64;
    }
    debug_assert!(at <= crate::xfer::BLOCK_REGION);
    o
}

fn kind_name(m: &Model, b: usize) -> String {
    match m.layers[b].kind {
        LayerKind::Gdn => "gdn".to_string(),
        LayerKind::Qsa => "qsa".to_string(),
    }
}

/// A loaded, ready-to-run session: pipelines, state, and the host expert pool.
pub struct Session<'m> {
    pub model: &'m Model,
    venues: Vec<Venue<'m>>,
    /// One staging buffer per device — kept out of [`Venue`] so a transfer can
    /// borrow it mutably while reading the device's own buffers.
    stagings: Vec<Staging>,
    gdn_state: Vec<Option<GdnState>>,
    qsa_state: Vec<Option<QsaState>>,
    /// One `QsaWorkspace` per QSA layer, allocated at `ctx_capacity` and reused
    /// every token: the *buffers* stay fixed and only `n_blocks` / `max_ids` /
    /// `slots` are rewritten per call, which is what the forward pass reads.
    ///
    /// The per-token version cost two extra flushes per QSA block (allocate
    /// outside the batch, release after it) plus eleven allocations per layer
    /// per token — 24 round trips a token, ~19 ms.
    qsa_ws: Vec<Option<QsaWorkspace>>,
    pool: std::sync::Arc<std::sync::Mutex<ExpertPool>>,
    ple: Option<PleHost>,
    sampler: Sampler,
    pub sampling: Sampling,
    pub timers: Timers,
    /// Forward passes run since the last reset (the decode denominator).
    pub forwards: usize,
    pub expert_bytes: u64,
    /// Device whose dispatch batch is currently open (`None` = none). At most
    /// one, because the batching flag is per thread while each device owns its
    /// own command buffer.
    pending_batch: Option<usize>,
    /// The segment a block submitted but whose fence has not been waited for.
    /// Two-stream decode keeps one of these per session in flight, so stream A's
    /// block executes while stream B's expert lane runs (§8o).
    pending_seg: Option<breccia_kernels::batch_cmd::Segment>,
    pending_dev: Option<usize>,
    /// Position of the token being processed (QSA's rope/selection input).
    pos: usize,
    /// The token id being processed (the PLE hash needs it).
    cur_tok: u32,
    /// Host seconds inside the expert lane, and the share of them that ran while
    /// **another stream's** GPU work was in flight — the overlap efficiency.
    pub lane_s: f64,
    pub lane_covered_s: f64,
    /// Submits and recorded dispatches, for the report's before/after.
    pub submits: u64,
    pub dispatches: u64,
    /// See [`SessionCfg::batching`].
    pub batching: bool,
    /// See [`SessionCfg::inbatch_read`].
    pub inbatch_read: bool,
    /// GPU-time probe (`BRECCIA_GPU_TIME=1`): one timestamp pool per device and
    /// the number of batches stamped so far (`None` = done or disabled).
    probe_pools: Vec<ash::vk::QueryPool>,
    probe_batches: Option<u32>,
    /// Accumulated GPU-busy seconds per device (the probe's total).
    gpu_busy_s: Vec<f64>,
    /// Per device: fence-wait seconds and host↔device bytes, so a dual run can
    /// say what *each* card did instead of only a total.
    pub dev_wait_s: Vec<f64>,
    pub dev_down: Vec<u64>,
    pub dev_up: Vec<u64>,
    pub dev_blocks: Vec<usize>,
    /// Blocks whose experts ran on a device this run (for the report).
    pub placed_blocks: usize,
    pub host_blocks: usize,
}

impl<'m> Session<'m> {
    /// Compile every pipeline, allocate every buffer, zero the state.
    pub fn new(model: &'m Model, cfg: &SessionCfg) -> Result<Self> {
        let mc = &model.cfg;
        let ctx_capacity = if cfg.ctx_capacity == 0 {
            256
        } else {
            cfg.ctx_capacity
        };
        let head_dev = model.device_of(mc.n_layers - 1);
        let emb_ty = ty_of(model, "token_embd.weight")?;
        let ple_ty = ty_of(model, "per_layer_token_embd.weight")?;
        let mut venues = Vec::with_capacity(model.devices.len());
        let mut stagings = Vec::with_capacity(model.devices.len());
        for (i, dev) in model.devices.iter().enumerate() {
            let gpu = dev.gpu();
            let pipes = PipelineSet::build(gpu, &model.keys)?;
            let hc = HcMixer::new(gpu, QuantType::Bf16, mc.n_embd, mc.hc, mc.hc_lr, true)
                .context("compiling the hyper-connection mixer")?;
            let final_hc = HcMixer::new(gpu, QuantType::Bf16, mc.n_embd, mc.hc, mc.hc_lr, false)
                .context("compiling the final hyper-connection mixer")?;
            let hc_ws = Some(hc.workspace(gpu, 1)?);
            let mut needs_gdn = false;
            let mut needs_qsa = false;
            for b in 0..mc.n_layers {
                if model.device_of(b) != i {
                    continue;
                }
                match mc.layer_kind(b) {
                    LayerKind::Gdn => needs_gdn = true,
                    LayerKind::Qsa => needs_qsa = true,
                }
            }
            let (gdn, gdn_ws) = if needs_gdn {
                let g = Gdn::new(gpu)?;
                let ws = g.workspace(gpu, 1)?;
                (Some(g), Some(ws))
            } else {
                (None, None)
            };
            let qsa = if needs_qsa {
                Some(Qsa::new(gpu, ctx_capacity.max(8))?)
            } else {
                None
            };
            let router = Router::new(gpu, mc.n_expert, mc.n_expert_used)?;
            let swiglu = SwiGlu::new(gpu)?;
            let emb = EmbGather::new(gpu, emb_ty, mc.n_embd)?;
            let emb_ple = EmbGather::new(gpu, ple_ty, mc.ple.dim)?;
            // Big enough for the largest single readback: on the head device the
            // logits row (vocab × 4 = 993 KiB), elsewhere R 40 KiB + PLE key
            // 40 KiB + value 10 KiB.
            let stage_cap = if head_dev == i {
                mc.vocab as u64 * 4 + (64 << 10)
            } else {
                192 << 10
            };
            stagings.push(Staging::new(gpu, stage_cap)?);
            let alloc = |bytes: u64, what: &str| -> Result<Buf> {
                breccia_kernels::gemv::alloc(gpu, bytes)
                    .with_context(|| format!("allocating {what} on device {i}"))
            };
            let n = mc.n_embd as u64;
            let r = alloc(mc.hc_dim() as u64 * 4, "the residual stack")?;
            let block_out = Mailbox::new(gpu, n * 4)?;
            let sh = alloc(n * 4, "shared-expert out")?;
            let sh_gate_logit = alloc(4, "shared-expert gate")?;
            let shexp_gate = alloc(mc.n_ff as u64 * 4, "shexp gate")?;
            let shexp_up = alloc(mc.n_ff as u64 * 4, "shexp up")?;
            let shexp_act = alloc(mc.n_ff as u64 * 4, "shexp act")?;
            let router_logits = alloc(mc.n_expert as u64 * 4, "router logits")?;
            let router_ids = alloc(router.ids_bytes(1), "router ids")?;
            let router_w = alloc(router.weights_bytes(1), "router weights")?;
            let ple_key_out = alloc(mc.hc_dim() as u64 * 4, "PLE key")?;
            let ple_value_out = alloc(n * 4, "PLE value")?;
            let ple_emb = alloc(mc.n_embd as u64 * 4, "PLE embed")?;
            let logits = if head_dev == i {
                Some(alloc(mc.vocab as u64 * 4, "logits")?)
            } else {
                None
            };
            // Device-resident MoE scratch (unused unless `--gpu-experts` places a
            // block on this device, but allocated once so the path has no
            // per-token allocation).
            let k = mc.n_expert_used;
            let ff = mc.n_ff;
            let moe_gate = alloc((k * ff) as u64 * 4, "moe gate")?;
            let moe_up = alloc((k * ff) as u64 * 4, "moe up")?;
            let moe_act = alloc((k * ff) as u64 * 4, "moe act")?;
            let moe_down = alloc((k * mc.n_embd) as u64 * 4, "moe down")?;
            let moe_map_gate = alloc((2 * k * ff) as u64 * 4, "moe map gate")?;
            let moe_map_up = alloc((2 * k * ff) as u64 * 4, "moe map up")?;
            let moe_map_down = alloc((2 * k * mc.n_embd) as u64 * 4, "moe map down")?;
            let moe_map = breccia_kernels::MoeMap::new(gpu)?;
            let moe_combine = breccia_kernels::MoeCombine::new(gpu)?;
            venues.push(Venue {
                pipes,
                hc,
                final_hc,
                hc_ws,
                gdn,
                gdn_ws,
                qsa,
                router,
                swiglu,
                emb,
                emb_ple,
                r,
                block_out,
                sh,
                sh_gate_logit,
                shexp_gate,
                shexp_up,
                shexp_act,
                router_logits,
                router_ids,
                router_w,
                ple_key_out,
                ple_value_out,
                ple_emb,
                logits,
                moe_gate,
                moe_up,
                moe_act,
                moe_down,
                moe_map_gate,
                moe_map_up,
                moe_map_down,
                moe_map,
                moe_combine,
                name: dev.info.name.clone(),
            });
        }

        // Per-layer recurrent / KV state, on the device that owns the block.
        let mut gdn_state = Vec::with_capacity(mc.n_layers);
        let mut qsa_state = Vec::with_capacity(mc.n_layers);
        for b in 0..mc.n_layers {
            let gpu = model.devices[model.device_of(b)].gpu();
            match mc.layer_kind(b) {
                LayerKind::Gdn => {
                    let st = GdnState::new(gpu)
                        .with_context(|| format!("allocating the GDN state of block {b}"))?;
                    st.reset(gpu)?;
                    gdn_state.push(Some(st));
                    qsa_state.push(None);
                }
                LayerKind::Qsa => {
                    let st = QsaState::new(gpu, ctx_capacity, ctx_capacity.max(8))
                        .with_context(|| format!("allocating the QSA state of block {b}"))?;
                    gdn_state.push(None);
                    qsa_state.push(Some(st));
                }
            }
        }

        // Hoist the QSA workspaces: capacity-sized once, mutated per call.
        let mut qsa_ws: Vec<Option<QsaWorkspace>> = Vec::with_capacity(mc.n_layers);
        for b in 0..mc.n_layers {
            if mc.layer_kind(b) != LayerKind::Qsa {
                qsa_ws.push(None);
                continue;
            }
            let dev = model.device_of(b);
            let gpu = model.devices[dev].gpu();
            let qsa = venues[dev]
                .qsa
                .as_ref()
                .ok_or_else(|| anyhow!("device {dev} has no QSA pipelines"))?;
            qsa_ws.push(Some(qsa.workspace(gpu, 1, ctx_capacity)?));
        }

        let threads = if cfg.threads == 0 {
            default_threads()
        } else {
            cfg.threads
        };
        let pool = match &cfg.pool {
            Some(p) => std::sync::Arc::clone(p),
            None => std::sync::Arc::new(std::sync::Mutex::new(ExpertPool::new(
                std::sync::Arc::clone(&model.shards),
                threads,
                cpu::path(),
            ))),
        };
        let ple = match &model.ple {
            Some(p) => Some(PleHost::new(
                mc,
                &p.norm_key,
                &p.norm_query,
                &p.norm_conv,
                &p.conv,
            )?),
            None => None,
        };
        Ok(Session {
            model,
            venues,
            stagings,
            gdn_state,
            qsa_state,
            qsa_ws,
            pool,
            ple,
            sampler: Sampler::new(Sampling::default().seed),
            sampling: Sampling::default(),
            timers: Timers {
                on: cfg.timers,
                ..Default::default()
            },
            forwards: 0,
            expert_bytes: 0,
            pending_batch: None,
            pending_seg: None,
            pending_dev: None,
            pos: 0,
            cur_tok: 0,
            lane_s: 0.0,
            lane_covered_s: 0.0,
            submits: 0,
            dispatches: 0,
            batching: cfg.batching,
            inbatch_read: cfg.inbatch_read,
            probe_pools: {
                let on = cfg.probe || std::env::var_os("BRECCIA_GPU_TIME").is_some();
                let mut v = Vec::with_capacity(model.devices.len());
                for i in 0..model.devices.len() {
                    v.push(if on {
                        model.devices[i]
                            .gpu()
                            .query_pool(2 * GPU_PROBE_BATCHES as u32)
                            .map_err(|e| anyhow!("allocating the GPU-time probe: {e}"))?
                    } else {
                        ash::vk::QueryPool::null()
                    });
                }
                v
            },
            probe_batches: if cfg.probe || std::env::var_os("BRECCIA_GPU_TIME").is_some() {
                Some(0)
            } else {
                None
            },
            gpu_busy_s: vec![0.0; model.devices.len()],
            dev_wait_s: vec![0.0; model.devices.len()],
            dev_down: vec![0; model.devices.len()],
            dev_up: vec![0; model.devices.len()],
            placed_blocks: model.placed.len(),
            host_blocks: model.cfg.n_layers - model.placed.len(),
            dev_blocks: {
                let mut v = vec![0usize; model.devices.len()];
                for b in 0..model.cfg.n_layers {
                    v[model.device_of(b)] += 1;
                }
                v
            },
        })
    }

    /// This session's devices (one model's placement).
    fn device_gpus(&self) -> Vec<&breccia_vk::ctx::Gpu> {
        self.model.devices.iter().map(|d| d.gpu()).collect()
    }

    /// Is a segment of this session's still in flight?
    pub fn in_flight(&self) -> bool {
        self.pending_seg.is_some() || self.pending_batch.is_some()
    }

    /// Close the open batch and wait for **this session's** submissions.
    ///
    /// Scoped to `self.model`'s devices: with two sessions (one per stream, each
    /// on its own card) a flush of stream A must not wait for stream B's GPU
    /// work — that would serialise exactly what the interleave is for.
    fn flush(&mut self) -> Result<()> {
        if let Some(d) = self.pending_batch.take() {
            let rec = batch_cmd::recorded();
            let stage = self.timers.stage;
            self.timers.count_dispatches(stage, rec);
            let gpu = self.model.devices[d].gpu();
            self.probe_stamp(d, true)?;
            let seg = batch_cmd::end_submit(gpu, None)
                .map_err(|e| anyhow!("closing the dispatch batch on device {d}: {e}"))?;
            self.submits += 1;
            self.dispatches += u64::from(rec);
            self.pending_seg = Some(seg);
            self.pending_dev = Some(d);
        }
        if let Some(seg) = self.pending_seg.take() {
            let d = self.pending_dev.take();
            let wait = {
                let gpus = self.device_gpus();
                batch_cmd::wait(&gpus, seg)
                    .map_err(|e| anyhow!("waiting for a block segment: {e}"))?
            };
            if let Some(d) = d {
                self.dev_wait_s[d] += wait;
            } else if let Some(v) = self.dev_wait_s.first_mut() {
                *v += wait;
            }
        }
        let wait = {
            let gpus = self.device_gpus();
            batch_cmd::wait_all(&gpus)
                .map_err(|e| anyhow!("waiting for the in-flight segments: {e}"))?
        };
        if let Some(v) = self.dev_wait_s.first_mut() {
            *v += wait;
        }
        Ok(())
    }

    /// Submit the open batch without waiting; the fence is kept in
    /// [`Session::pending_seg`] so [`Session::finish_block`] can wait for it.
    fn submit_batch(&mut self) -> Result<()> {
        let Some(d) = self.pending_batch.take() else {
            return Ok(());
        };
        let gpu = self.model.devices[d].gpu();
        let rec = batch_cmd::recorded();
        let stage = self.timers.stage;
        self.timers.count_dispatches(stage, rec);
        self.probe_stamp(d, true)?;
        let seg = batch_cmd::end_submit(gpu, None)
            .map_err(|e| anyhow!("submitting the block on device {d}: {e}"))?;
        self.submits += 1;
        self.dispatches += u64::from(rec);
        self.pending_seg = Some(seg);
        self.pending_dev = Some(d);
        Ok(())
    }

    /// Make sure a batch is open on `dev`, flushing any batch on another device.
    ///
    /// Everything recorded between here and the next [`Session::flush`] costs one
    /// submit and one fence wait, instead of one per kernel: the ~29 dispatches
    /// of a block collapse to a single round trip.
    fn ensure_batch(&mut self, dev: usize) -> Result<()> {
        if !self.batching {
            return Ok(());
        }
        if self.pending_batch == Some(dev) {
            // A batch that has grown past the soft limit is closed and reopened:
            // it keeps the descriptor pool bounded and lets the driver start the
            // work we have already recorded.
            if batch_cmd::recorded() >= batch_cmd::SOFT_LIMIT {
                self.flush()?;
            } else {
                return Ok(());
            }
        }
        self.flush()?;
        batch_cmd::begin(self.model.devices[dev].gpu())
            .map_err(|e| anyhow!("opening a dispatch batch on device {dev}: {e}"))?;
        self.pending_batch = Some(dev);
        self.probe_stamp(dev, false)?;
        Ok(())
    }

    /// `BRECCIA_GPU_TIME=1`: bracket the first [`GPU_PROBE_BATCHES`] batches with
    /// GPU timestamps and print their durations at the end of the token.
    ///
    /// The fence wait a batch costs includes the host and driver's submission
    /// latency; these say how much of it the GPU was actually busy — the number
    /// that decides whether a bottleneck is arithmetic (fusions help) or
    /// submission (pipelining helps).
    fn probe_stamp(&mut self, dev: usize, end: bool) -> Result<()> {
        let Some(k) = self.probe_batches else {
            return Ok(());
        };
        let pool = self.probe_pools[dev];
        let query = 2 * k + u32::from(end);
        batch_cmd::record_timestamp(self.model.devices[dev].gpu(), pool, query)?;
        if end {
            let k = k + 1;
            if k as usize >= GPU_PROBE_BATCHES {
                // The pool is full: read it out and start again, so the probe
                // measures the whole run rather than its first 256 batches.
                self.probe_flush()?;
            } else {
                self.probe_batches = Some(k);
            }
        }
        Ok(())
    }

    /// Read the pending probe timestamps into [`Session::gpu_busy_s`] and reset
    /// the pools (the caller has waited for the submissions that wrote them).
    fn probe_flush(&mut self) -> Result<()> {
        let Some(k) = self.probe_batches else {
            return Ok(());
        };
        if k == 0 {
            return Ok(());
        }
        for (i, pool) in self.probe_pools.iter().enumerate() {
            if *pool == ash::vk::QueryPool::null() {
                continue;
            }
            let gpu = self.model.devices[i].gpu();
            let ts = gpu
                .timestamps(*pool, k)
                .map_err(|e| anyhow!("reading the GPU-time probe: {e}"))?;
            let mut sum = 0.0f64;
            for c in ts.chunks_exact(2) {
                // A batch that belongs to the other device leaves a (0, 0) pair.
                if c[1] >= c[0] {
                    sum += c[1] - c[0];
                }
            }
            self.gpu_busy_s[i] += sum / 1e9;
            gpu.reset_queries(*pool, 2 * GPU_PROBE_BATCHES as u32);
        }
        self.probe_batches = Some(0);
        Ok(())
    }

    /// GPU seconds each of this session's devices was busy, from the probe
    /// (flushed, so it is current). Zero when the probe is off.
    pub fn gpu_busy_s(&mut self) -> Result<Vec<f64>> {
        self.probe_flush()?;
        Ok(self.gpu_busy_s.clone())
    }

    /// Zero the per-run counters (the decode window starts after the prompt).
    pub fn reset_counters(&mut self) -> Result<()> {
        self.forwards = 0;
        self.expert_bytes = 0;
        self.lane_s = 0.0;
        self.lane_covered_s = 0.0;
        self.dev_wait_s.iter_mut().for_each(|x| *x = 0.0);
        self.dev_down.iter_mut().for_each(|x| *x = 0);
        self.dev_up.iter_mut().for_each(|x| *x = 0);
        self.timers.clear();
        self.reset_gpu_busy()
    }

    /// Zero the probe's accumulated busy time and drop any unread timestamps, so
    /// the next window is measured from scratch.
    pub fn reset_gpu_busy(&mut self) -> Result<()> {
        for (i, pool) in self.probe_pools.iter().enumerate() {
            if *pool != ash::vk::QueryPool::null() {
                self.model.devices[i]
                    .gpu()
                    .reset_queries(*pool, 2 * GPU_PROBE_BATCHES as u32);
            }
        }
        if self.probe_batches.is_some() {
            self.probe_batches = Some(0);
        }
        self.gpu_busy_s.iter_mut().for_each(|x| *x = 0.0);
        Ok(())
    }

    /// Print and disable the GPU-time probe (called once a token).
    fn probe_report(&mut self) -> Result<()> {
        self.probe_flush()?;
        if std::env::var_os("BRECCIA_GPU_TIME_VERBOSE").is_some() {
            for (i, s) in self.gpu_busy_s.iter().enumerate() {
                eprintln!(
                    "gpu-time device {i}: GPU busy {:.2} ms over {} token(s)",
                    s * 1e3,
                    self.forwards.max(1)
                );
            }
        }
        Ok(())
    }

    /// Threads the host expert lane is running.
    pub fn threads(&self) -> usize {
        self.pool.lock().map(|p| p.threads()).unwrap_or(0)
    }

    /// The pool this session draws its expert lane from (shared by design in a
    /// two-stream run).
    pub fn pool(&self) -> &std::sync::Arc<std::sync::Mutex<ExpertPool>> {
        &self.pool
    }

    /// Sample the next token from `logits` with this session's sampler and
    /// sampling settings.
    pub fn sample(&mut self, logits: &[f32]) -> Option<u32> {
        self.sampler.sample(logits, &self.sampling)
    }

    /// The device this session's block `b` runs on (for per-card reporting).
    pub fn device_of(&self, b: usize) -> usize {
        self.model.device_of(b)
    }

    /// `(name, pci address)` of each of this session's devices.
    pub fn device_names(&self) -> Vec<(String, String)> {
        self.model
            .devices
            .iter()
            .map(|d| (d.info.name.clone(), d.info.pci_address.clone()))
            .collect()
    }

    /// Zero every recurrent state and the PLE history — a new sequence.
    pub fn reset(&mut self) -> Result<()> {
        let m = self.model;
        for b in 0..m.cfg.n_layers {
            if let Some(st) = &self.gdn_state[b] {
                st.reset(m.devices[m.device_of(b)].gpu())?;
            }
        }
        if let Some(p) = self.ple.as_mut() {
            p.reset();
        }
        self.flush()?;
        self.expert_bytes = 0;
        self.forwards = 0;
        self.timers.clear();
        Ok(())
    }

    /// What each card actually did: blocks, fence-wait time, host traffic.
    pub fn per_device(&self, forwards: usize) -> Vec<String> {
        let n = forwards.max(1) as f64;
        (0..self.model.devices.len())
            .map(|i| {
                format!(
                    "device {i} {:<42} {:>2} blocks  {:>6.2} ms/token fence wait                       {:>5.1} MiB down {:>5.1} MiB up",
                    self.venues[i].name,
                    self.dev_blocks[i],
                    self.dev_wait_s[i] * 1e3 / n,
                    self.dev_down[i] as f64 / n / (1 << 20) as f64,
                    self.dev_up[i] as f64 / n / (1 << 20) as f64,
                )
            })
            .collect()
    }

    /// Bytes of VRAM the session itself allocated (state + scratch).
    pub fn session_bytes(&self) -> u64 {
        let mut total = 0u64;
        for v in &self.venues {
            for b in [
                &v.r,
                &v.sh,
                &v.sh_gate_logit,
                &v.shexp_gate,
                &v.shexp_up,
                &v.shexp_act,
                &v.router_logits,
                &v.router_ids,
                &v.router_w,
                &v.ple_key_out,
                &v.ple_value_out,
                &v.ple_emb,
                &v.moe_gate,
                &v.moe_up,
                &v.moe_act,
                &v.moe_down,
                &v.moe_map_gate,
                &v.moe_map_up,
                &v.moe_map_down,
            ] {
                total += b.size;
            }
            if let Some(l) = &v.logits {
                total += l.size;
            }
            if let Some(ws) = &v.hc_ws {
                total += ws.bytes();
            }
            if let Some(ws) = &v.gdn_ws {
                total += ws.bytes();
            }
        }
        total += self.stagings.iter().map(|s| s.bytes()).sum::<u64>();
        total += self
            .gdn_state
            .iter()
            .flatten()
            .map(|s| s.bytes())
            .sum::<u64>();
        total += self
            .qsa_state
            .iter()
            .flatten()
            .map(|s| s.bytes())
            .sum::<u64>();
        total
    }

    /// Run one token at `pos`; returns its logits.
    /// One token, start to finish: the single-stream convenience wrapper over
    /// [`Session::begin_token`] / [`Session::submit_block`] /
    /// [`Session::finish_block`] / [`Session::end_token`].
    pub fn token(
        &mut self,
        tok: u32,
        pos: usize,
        debug: Option<&mut Vec<LayerTrace>>,
    ) -> Result<Vec<f32>> {
        let mut trace = debug;
        self.begin_token(tok, pos)?;
        let n = self.model.cfg.n_layers;
        for b in 0..n {
            let digests = self.submit_block(b, &mut trace)?;
            self.finish_block(b, digests, &mut trace, false)?;
        }
        self.end_token()
    }

    /// Embed the token into the residual stack and start the per-token timers.
    ///
    /// Split out so two sessions can be driven block by block with one stream's
    /// GPU work running under the other stream's expert lane (§8o).
    pub fn begin_token(&mut self, tok: u32, pos: usize) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        if pos >= mc.n_ctx_max {
            bail!(
                "position {pos} is past the model's context limit {}",
                mc.n_ctx_max
            );
        }
        self.pos = pos;
        self.cur_tok = tok;
        // ---- embedding: 4 gathered rows ARE the hc copies of the vector ------
        self.flush()?;
        self.timers.start();
        {
            let dev = m.device_of(0);
            let gpu = m.devices[dev].gpu();
            let v = &self.venues[dev];
            let table = m.token_embd()?;
            let rows = vec![tok; mc.hc];
            let staged = v.emb.gather_staged(gpu, table, &rows, &v.r)?;
            v.emb.release(gpu, &staged);
        }
        self.timers.mark(PLE);
        Ok(())
    }

    /// Record block `b` and **submit it without waiting**.
    ///
    /// The segment it submits contains the previous block's ffn `HC_WRITE` (which
    /// reads the `block_out` mailbox this host wrote after the previous lane),
    /// this block's two HC reads, its mixer, the router and the shared expert —
    /// one submission per block, exactly as before; only the *wait* moved to
    /// [`Session::finish_block`], where the other stream's GPU work can run
    /// underneath it.
    pub fn submit_block(
        &mut self,
        b: usize,
        trace: &mut Option<&mut Vec<LayerTrace>>,
    ) -> Result<BlockDigests> {
        let m = self.model;
        let mc = &m.cfg;
        let pos = self.pos;
        let dev = m.device_of(b);

        // ---- move the residual stack if the block lives elsewhere -----------
        if b > 0 && m.device_of(b - 1) != dev {
            // The hand-off uses `Staging`, which records into its own command
            // buffer — so the open batch (the previous block's ffn `HC_WRITE`)
            // must be submitted first.
            self.flush()?;
            self.timers.mark(HANDOFF);
            let prev = m.device_of(b - 1);
            let bytes = self.stagings[prev].read_batch(
                m.devices[prev].gpu(),
                &[(&self.venues[prev].r, 0, mc.hc_dim() * 4)],
            )?;
            self.stagings[dev].write(m.devices[dev].gpu(), &self.venues[dev].r, &bytes[0])?;
        }

        // ---- (A) PLE insertion (block 1 only) --------------------------------
        if mc.is_ple_layer(b) {
            self.timers.mark(PLE);
            self.ple_step(b)?;
        }

        // `BRECCIA_TRACE_BATCH=1` prints one line per block: kind, whether a
        // batch is open, and the running submit/dispatch counters. It is how the
        // QSA-workspace lifecycle bug (docs/phase-4/01-forward-pass.md §8.1) was
        // found, so it stays.
        let trace_batch = std::env::var_os("BRECCIA_TRACE_BATCH").is_some();
        if trace_batch {
            eprintln!(
                "block {b:>2} {:?} open={} submits={} dispatches={}",
                m.layers[b].kind,
                batch_cmd::is_open(),
                self.submits,
                self.dispatches
            );
        }
        // Every dispatch from here to the MoE readback rides one command buffer.
        self.ensure_batch(dev)?;

        // ---- (B) HC_READ(attn) -----------------------------------------------
        self.timers.mark(HC_READ_A);
        {
            let v = &self.venues[dev];
            hc_read(v, &m.layers[b].hc_attn, &v.r)?;
        }
        let mixed_attn = self.maybe_checksum(trace.is_some(), dev, BufKind::Mixed, mc.n_embd)?;

        // ---- (C) the token mixer --------------------------------------------
        self.timers.mark(MIXER);
        match m.layers[b].kind {
            LayerKind::Gdn => self.run_gdn(b, dev)?,
            LayerKind::Qsa => self.run_qsa(b, dev, pos)?,
        }
        let mixer_out = self.maybe_checksum(trace.is_some(), dev, BufKind::BlockOut, mc.n_embd)?;

        // ---- (D) HC_WRITE(attn) ----------------------------------------------
        self.timers.mark(HC_WRITE_A);
        {
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.hc.write(ws, &v.r, &v.block_out.buf, &v.r)?;
        }

        // ---- (E) HC_READ(ffn) ------------------------------------------------
        self.timers.mark(HC_READ_F);
        {
            let v = &self.venues[dev];
            hc_read(v, &m.layers[b].hc_ffn, &v.r)?;
        }
        let mixed_ffn = self.maybe_checksum(trace.is_some(), dev, BufKind::Mixed, mc.n_embd)?;

        // ---- (F) router ------------------------------------------------------
        self.timers.mark(ROUTER);
        {
            let lw = &m.layers[b];
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.pipes.get(&lw.moe.router.key)?.run(
                &lw.moe.router.buf,
                &ws.mixed,
                &v.router_logits,
            )?;
            v.router
                .run(&v.router_logits, &v.router_ids, &v.router_w, 1)?;
        }

        // ---- (G) shared expert + its scalar sigmoid gate ---------------------
        self.timers.mark(SHARED);
        {
            let lw = &m.layers[b];
            let v = &self.venues[dev];
            let pipes = &v.pipes;
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            pipes
                .get(&lw.moe.gate.key)?
                .run(&lw.moe.gate.buf, &ws.mixed, &v.shexp_gate)?;
            pipes
                .get(&lw.moe.up.key)?
                .run(&lw.moe.up.buf, &ws.mixed, &v.shexp_up)?;
            v.swiglu
                .run(&v.shexp_gate, &v.shexp_up, &v.shexp_act, mc.n_ff)?;
            pipes
                .get(&lw.moe.down.key)?
                .run(&lw.moe.down.buf, &v.shexp_act, &v.sh)?;
            pipes.get(&lw.moe.shexp_gate.key)?.run(
                &lw.moe.shexp_gate.buf,
                &ws.mixed,
                &v.sh_gate_logit,
            )?;
        }

        // ---- a block whose experts are resident: no host, no readback --------
        if let Some(pe) = m.placed.get(&b) {
            self.timers.mark(EXPERTS);
            self.run_placed_moe(b, dev, pe, mc.n_embd, mc.n_ff, mc.n_expert_used)?;
            self.timers.mark(HC_WRITE_F);
            {
                let v = &self.venues[dev];
                let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
                v.hc.write(ws, &v.r, &v.block_out.buf, &v.r)?;
            }
            self.submit_batch()?;
            return Ok(BlockDigests {
                mixed_attn,
                mixer_out,
                mixed_ffn,
            });
        }

        // The block's readback can ride this submission: five `cmd_copy_buffer`s
        // into the staging block region plus a host-visibility barrier, so the
        // host reads the mapping after the wait instead of paying a submission
        // and a fence of its own (§8o).
        if self.inbatch_read {
            let k = mc.n_expert_used;
            let gpu = m.devices[dev].gpu();
            let v = &self.venues[dev];
            let st = &self.stagings[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            let offs = lane_offsets(mc.n_embd, k);
            batch_cmd::record_copy(
                gpu,
                st.buffer(),
                offs[0],
                &ws.mixed,
                0,
                (mc.n_embd * 4) as u64,
            )?;
            batch_cmd::record_copy(gpu, st.buffer(), offs[1], &v.router_ids, 0, (k * 4) as u64)?;
            batch_cmd::record_copy(gpu, st.buffer(), offs[2], &v.router_w, 0, (k * 4) as u64)?;
            batch_cmd::record_copy(gpu, st.buffer(), offs[3], &v.sh, 0, (mc.n_embd * 4) as u64)?;
            batch_cmd::record_copy(gpu, st.buffer(), offs[4], &v.sh_gate_logit, 0, 4)?;
            batch_cmd::record_barrier_to_host(gpu)?;
        }

        // The host block's submission ends here: the readback is
        // [`Session::finish_block`]'s job, after the wait.
        self.submit_batch()?;
        Ok(BlockDigests {
            mixed_attn,
            mixer_out,
            mixed_ffn,
        })
    }

    /// Wait for block `b`'s segment, read its MoE inputs, run the host expert
    /// lane, hand the result back through the mailbox and open the next batch
    /// with the ffn `HC_WRITE`.
    ///
    /// `covered` says whether **another stream's** GPU work was in flight while
    /// this lane ran (the multi-stream driver knows; the single-stream wrapper
    /// passes `false`). It only affects the reported overlap efficiency.
    pub fn finish_block(
        &mut self,
        b: usize,
        digests: BlockDigests,
        trace: &mut Option<&mut Vec<LayerTrace>>,
        covered: bool,
    ) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let dev = m.device_of(b);

        // ---- wait for this block's segment ----------------------------------
        self.timers.mark(WAIT_GPU);
        if let Some(seg) = self.pending_seg.take() {
            let d = self.pending_dev.take().unwrap_or(dev);
            let wait = {
                let gpus = self.device_gpus();
                batch_cmd::wait(&gpus, seg)
                    .map_err(|e| anyhow!("waiting for block {b}'s segment: {e}"))?
            };
            self.dev_wait_s[d] += wait;
        }

        // ---- a block whose experts are resident: nothing left to do ---------
        if m.placed.contains_key(&b) {
            self.timers.mark(HC_WRITE_F);
            if let Some(t) = trace.as_mut() {
                let residual = self.checksum(dev, BufKind::R, mc.hc_dim())?;
                let gdn = self.gdn_trace_if(b, dev, m)?;
                t.push(LayerTrace {
                    block: b,
                    kind: kind_name(m, b),
                    device: dev,
                    residual,
                    mixed_attn: digests.mixed_attn.unwrap_or_default(),
                    mixed_ffn: digests.mixed_ffn.unwrap_or_default(),
                    mixer_out: digests.mixer_out.unwrap_or_default(),
                    moe_out: Checksum::of(&[0.0f32; 0], 0),
                    router_ids: Vec::new(),
                    router_weights: Vec::new(),
                    ple_gate: None,
                    ple_conv: None,
                    gdn,
                });
            }
            return Ok(());
        }

        // ---- one readback: mixed, ids, weights, shared out, shared gate ------
        self.timers.mark(XFER_DOWN);
        let k = mc.n_expert_used;
        let ws_mixed = &self.venues[dev]
            .hc_ws
            .as_ref()
            .expect("the session's hc workspace")
            .mixed;
        let raw = if self.inbatch_read {
            let offs = lane_offsets(mc.n_embd, k);
            let st = &self.stagings[dev];
            vec![
                st.read_block(offs[0], mc.n_embd * 4).to_vec(),
                st.read_block(offs[1], k * 4).to_vec(),
                st.read_block(offs[2], k * 4).to_vec(),
                st.read_block(offs[3], mc.n_embd * 4).to_vec(),
                st.read_block(offs[4], 4).to_vec(),
            ]
        } else {
            self.stagings[dev].read_batch(
                m.devices[dev].gpu(),
                &[
                    (ws_mixed, 0, mc.n_embd * 4),
                    (&self.venues[dev].router_ids, 0, k * 4),
                    (&self.venues[dev].router_w, 0, k * 4),
                    (&self.venues[dev].sh, 0, mc.n_embd * 4),
                    (&self.venues[dev].sh_gate_logit, 0, 4),
                ],
            )?
        };
        self.dev_down[dev] += (mc.n_embd * 2 + k * 8 + 4) as u64;
        let mixed = f32_from_bytes(&raw[0]);
        let ids: Vec<usize> = raw[1]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let weights = f32_from_bytes(&raw[2]);
        let shared = f32_from_bytes(&raw[3]);
        let g = f32::from_le_bytes([raw[4][0], raw[4][1], raw[4][2], raw[4][3]]);
        let gate = 1.0 / (1.0 + (-g).exp());

        // ---- host expert lane -------------------------------------------------
        self.timers.mark(EXPERTS);
        let mut out = vec![0f32; mc.n_embd];
        let t_lane = std::time::Instant::now();
        let bytes = self
            .pool
            .lock()
            .expect("the expert pool's lock")
            .moe_topk(&m.layers[b].experts, &ids, &weights, &mixed, &mut out)
            .with_context(|| format!("the expert lane on block {b}"))?;
        let lane = t_lane.elapsed().as_secs_f64();
        self.lane_s += lane;
        if covered {
            self.lane_covered_s += lane;
        }
        self.expert_bytes += bytes as u64;
        for i in 0..mc.n_embd {
            out[i] += shared[i] * gate;
        }
        let moe_out = if trace.is_some() {
            Some(Checksum::of(&out, 0))
        } else {
            None
        };

        // ---- upload the FFN result -------------------------------------------
        self.timers.mark(XFER_UP);
        // A memcpy into the mailbox: no submit, no fence.
        self.venues[dev].block_out.write(&f32_to_bytes(&out))?;
        self.dev_up[dev] += (mc.n_embd * 4) as u64;

        // ---- (H) HC_WRITE(ffn): recorded, left open --------------------------
        //
        // It reads the mailbox written just above, so it must be *submitted after
        // that write*; the next block's `submit_block` records on top of it and
        // submits once. That is what keeps a block at one submission even though
        // the write happens on the host.
        self.timers.mark(HC_WRITE_F);
        self.ensure_batch(dev)?;
        {
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.hc.write(ws, &v.r, &v.block_out.buf, &v.r)?;
        }

        if let Some(t) = trace.as_mut() {
            let residual = self.checksum(dev, BufKind::R, mc.hc_dim())?;
            let gdn = self.gdn_trace_if(b, dev, m)?;
            t.push(LayerTrace {
                block: b,
                kind: kind_name(m, b),
                device: dev,
                residual,
                mixed_attn: digests.mixed_attn.unwrap_or_default(),
                mixed_ffn: digests.mixed_ffn.unwrap_or_default(),
                mixer_out: digests.mixer_out.unwrap_or_default(),
                moe_out: moe_out.unwrap_or_default(),
                router_ids: ids.iter().map(|&i| i as u32).collect(),
                router_weights: weights,
                ple_gate: None,
                ple_conv: None,
                gdn,
            });
        }
        Ok(())
    }

    fn gdn_trace_if(&mut self, b: usize, dev: usize, m: &Model) -> Result<Option<GdnTrace>> {
        if m.layers[b].kind == LayerKind::Gdn {
            Ok(Some(self.gdn_trace(dev)?))
        } else {
            Ok(None)
        }
    }

    /// The output norm, the head, and the logits row for the token just fed.
    pub fn end_token(&mut self) -> Result<Vec<f32>> {
        self.submit_head()?;
        self.finish_head()
    }

    /// Record and submit the output norm + head **without waiting**; pair it with
    /// [`Session::finish_head`] so two streams' heads run concurrently (they are
    /// the last serial GPU work of a stream, ~7-10 ms each).
    pub fn submit_head(&mut self) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        self.flush()?;
        self.timers.mark(FINAL_HC);
        let last = m.device_of(mc.n_layers - 1);
        self.ensure_batch(last)?;
        {
            let v = &self.venues[last];
            let hw = HcWeights {
                norm: &m.out_hc_norm,
                down: &m.out_hc_down.buf,
                up: &m.out_hc_up.buf,
                inject: None,
            };
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.final_hc.read(ws, &v.r, &hw)?;
            self.timers.mark(HEAD);
            let out = v
                .logits
                .as_ref()
                .ok_or_else(|| anyhow!("the last device has no logits buffer"))?;
            v.pipes.get(&m.head.key)?.run(&m.head.buf, &ws.mixed, out)?;
        }
        self.submit_batch()?;
        Ok(())
    }

    /// Wait for the head and read the logits row back.
    pub fn finish_head(&mut self) -> Result<Vec<f32>> {
        let m = self.model;
        let mc = &m.cfg;
        let last = m.device_of(mc.n_layers - 1);
        self.flush()?;
        let logits_buf = self.venues[last]
            .logits
            .as_ref()
            .ok_or_else(|| anyhow!("the last device has no logits buffer"))?;
        let raw = self.stagings[last]
            .read_batch(m.devices[last].gpu(), &[(logits_buf, 0, mc.vocab * 4)])?;
        self.timers.stop();
        self.probe_report()?;
        self.forwards += 1;
        let values = f32_from_bytes(&raw[0]);
        if values.len() != mc.vocab {
            bail!(
                "the head produced {} logits, expected {}",
                values.len(),
                mc.vocab
            );
        }
        // A non-finite logit is a numerical failure, not a sampling preference.
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            bail!("logit {i} is not finite ({})", values[i]);
        }
        Ok(values)
    }

    /// One block's MoE, entirely on the device: ids→map, gate/up/SwiGLU, down,
    /// combine (routed + shared·sigmoid(gate)) into `block_out`.
    ///
    /// Nothing here touches the host: the router's ids and weights are read by
    /// shaders straight out of their buffers, and the result lands in the
    /// `block_out` the next recorded dispatch already reads.
    fn run_placed_moe(
        &mut self,
        b: usize,
        dev: usize,
        pe: &crate::model::PlacedExperts,
        n_embd: usize,
        n_ff: usize,
        k: usize,
    ) -> Result<()> {
        let _ = n_ff;
        let v = &self.venues[dev];
        let pipes = &v.pipes;
        let mixed = &v.hc_ws.as_ref().expect("hc ws").mixed;
        // gate / up: one shared activation, and **one map per role** — the two
        // tensors need not share a type, and the row map's units depend on it.
        let gate = pipes.batch(&pe.gate.key)?;
        v.moe_map
            .run(&v.router_ids, &v.moe_map_gate, gate, k, 0, false)?;
        gate.run(&pe.gate.buf, mixed, &v.moe_gate, &v.moe_map_gate, k)?;
        let up = pipes.batch(&pe.up.key)?;
        v.moe_map
            .run(&v.router_ids, &v.moe_map_up, up, k, 0, false)?;
        up.run(&pe.up.buf, mixed, &v.moe_up, &v.moe_map_up, k)?;
        v.swiglu.run(&v.moe_gate, &v.moe_up, &v.moe_act, k * n_ff)?;
        // down: one activation row per expert, so the map carries the stride
        let down = pipes.batch(&pe.down.key)?;
        v.moe_map
            .run(&v.router_ids, &v.moe_map_down, down, k, 0, true)?;
        down.run(&pe.down.buf, &v.moe_act, &v.moe_down, &v.moe_map_down, k)?;
        v.moe_combine.run(
            &v.moe_down,
            &v.router_w,
            &v.sh,
            &v.sh_gate_logit,
            &v.block_out.buf,
            k,
            n_embd,
        )?;
        let _ = b;
        Ok(())
    }

    fn run_gdn(&mut self, b: usize, dev: usize) -> Result<()> {
        let m = self.model;
        let gw = m.layers[b]
            .gdn
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} is GDN but has no weights"))?;
        let v = &self.venues[dev];
        let proj = GdnProjections {
            qkv: Proj::Single(v.pipes.get(&gw.qkv.key)?),
            w_qkv: &gw.qkv.buf,
            z: Proj::Single(v.pipes.get(&gw.z.key)?),
            w_z: &gw.z.buf,
            beta: Proj::Single(v.pipes.get(&gw.beta.key)?),
            w_beta: &gw.beta.buf,
            alpha: Proj::Single(v.pipes.get(&gw.alpha.key)?),
            w_alpha: &gw.alpha.buf,
            out: Proj::Single(v.pipes.get(&gw.out.key)?),
            w_out: &gw.out.buf,
        };
        let weights = GdnWeights {
            conv: &gw.conv,
            dt: &gw.dt,
            a: &gw.a,
            ssm_norm: &gw.ssm_norm,
        };
        let gdn = v
            .gdn
            .as_ref()
            .ok_or_else(|| anyhow!("device {dev} has no GDN pipelines"))?;
        let ws = v
            .gdn_ws
            .as_ref()
            .ok_or_else(|| anyhow!("device {dev} has no GDN workspace"))?;
        let state = self.gdn_state[b]
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} has no GDN state"))?;
        let mixed = &v.hc_ws.as_ref().expect("the session's hc workspace").mixed;
        gdn.forward(state, ws, &weights, &proj, mixed, &v.block_out.buf)
    }

    fn run_qsa(&mut self, b: usize, dev: usize, pos: usize) -> Result<()> {
        let m = self.model;
        let n_kv_after = pos + 1;
        {
            let state = self.qsa_state[b]
                .as_ref()
                .ok_or_else(|| anyhow!("block {b} has no QSA state"))?;
            if n_kv_after > state.capacity_cells {
                bail!(
                    "block {b}: position {pos} needs {n_kv_after} KV cells but the cache holds {}",
                    state.capacity_cells
                );
            }
        }
        // The workspace is hoisted: rewrite the three shape scalars the forward
        // pass reads (`n_blocks`, `max_ids`, `slots`) and leave the buffers
        // alone. No allocation, no flush — the batch stays open across the block.
        {
            let Some(ws) = self.qsa_ws[b].as_mut() else {
                bail!("block {b} has no hoisted QSA workspace");
            };
            let n_blocks = n_kv_after / 4;
            let max_ids = n_blocks.min(2048 / 4) + 1;
            ws.n_blocks = n_blocks;
            ws.max_ids = max_ids;
            ws.slots = 4 * max_ids;
        }
        self.ensure_batch(dev)?;
        let ws: &QsaWorkspace = self.qsa_ws[b]
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} has no hoisted QSA workspace"))?;
        {
            let qw = m.layers[b]
                .qsa
                .as_ref()
                .ok_or_else(|| anyhow!("block {b} is QSA but has no weights"))?;
            let v = &self.venues[dev];
            let qsa = v
                .qsa
                .as_ref()
                .ok_or_else(|| anyhow!("device {dev} has no QSA pipelines"))?;
            let state = self.qsa_state[b]
                .as_ref()
                .ok_or_else(|| anyhow!("block {b} has no QSA state"))?;
            let weights = QsaWeights {
                q_norm: &qw.q_norm,
                k_norm: &qw.k_norm,
                idx_q_norm: &qw.idx_q_norm,
                idx_k_norm: &qw.idx_k_norm,
            };
            let proj = QsaProjections {
                q: Proj::Single(v.pipes.get(&qw.q.key)?),
                w_q: &qw.q.buf,
                k: Proj::Single(v.pipes.get(&qw.k.key)?),
                w_k: &qw.k.buf,
                v: Proj::Single(v.pipes.get(&qw.v.key)?),
                w_v: &qw.v.buf,
                idx_q: Proj::Single(v.pipes.get(&qw.idx_q.key)?),
                w_idx_q: &qw.idx_q.buf,
                idx_k: Proj::Single(v.pipes.get(&qw.idx_k.key)?),
                w_idx_k: &qw.idx_k.buf,
                out: Proj::Single(v.pipes.get(&qw.out.key)?),
                w_out: &qw.out.buf,
            };
            let hc = v.hc_ws.as_ref().expect("the session's hc workspace");
            qsa.forward(state, ws, &weights, &proj, &hc.mixed, &v.block_out.buf, pos)
        }
    }

    /// The PLE insertion: host hash → GPU projections → host block → write back.
    fn ple_step(&mut self, b: usize) -> Result<()> {
        let tok = self.cur_tok;
        // PLE needs the residual on the host, so it runs outside any batch.
        self.flush()?;
        let m = self.model;
        let mc = &m.cfg;
        let raw = m
            .ple
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} runs PLE but the model has no PLE tensors"))?;
        let table = m.ple_table()?;
        let dev = m.device_of(b);
        let hc_dim = mc.hc_dim();
        let n_embd = mc.n_embd;

        // 1. the host hash (§9.1): 16 table rows, head-major.
        let rows = self
            .ple
            .as_mut()
            .ok_or_else(|| anyhow!("no PLE host state"))?
            .rows_for(tok);

        // 2. fault the rows in concurrently, then gather them on the GPU.
        //
        // The PLE table is 26.8 GiB of lazily-mapped IQ4_NL, and one token needs
        // 16 scattered 90-byte rows = 16 page faults. Serialised, those cost
        // ~28 ms per token here whenever the page cache has evicted them (the
        // expert lane streams ~800 MiB/token through the same NVMe), against
        // 0.65 ms once hot. llama.cpp prefetches the same rows for the same
        // reason (`llama_prefetch_rows`, `qwen4exp.cpp:1283-1288`).
        prefetch_rows(table, &rows, m.ple_row_bytes());
        {
            let gpu = m.devices[dev].gpu();
            let v = &self.venues[dev];
            let staged = v.emb_ple.gather_staged(gpu, table, &rows, &v.ple_emb)?;
            v.emb_ple.release(gpu, &staged);
            v.pipes
                .get(&raw.key.key)?
                .run(&raw.key.buf, &v.ple_emb, &v.ple_key_out)?;
            v.pipes
                .get(&raw.value.key)?
                .run(&raw.value.buf, &v.ple_emb, &v.ple_value_out)?;
        }

        // 3. readback R, key, value; run the block on the host; write R back.
        let gpu = m.devices[dev].gpu();
        self.flush()?;
        let parts = {
            let v = &self.venues[dev];
            self.stagings[dev].read_batch(
                gpu,
                &[
                    (&v.r, 0, hc_dim * 4),
                    (&v.ple_key_out, 0, hc_dim * 4),
                    (&v.ple_value_out, 0, n_embd * 4),
                ],
            )?
        };
        let r_host = f32_from_bytes(&parts[0]);
        let key = f32_from_bytes(&parts[1]);
        let value = f32_from_bytes(&parts[2]);
        let mut out = vec![0f32; hc_dim];
        self.ple
            .as_mut()
            .ok_or_else(|| anyhow!("no PLE host state"))?
            .block(&r_host, &key, &value, &mut out)?;
        self.stagings[dev].write(gpu, &self.venues[dev].r, &f32_to_bytes(&out))?;
        Ok(())
    }

    /// Digest every stage of a GDN sublayer's workspace.
    fn gdn_trace(&mut self, dev: usize) -> Result<GdnTrace> {
        self.flush()?;
        let gpu = self.model.devices[dev].gpu();
        // Sizes are the workspace's own shapes: h = conv channels, the rest as
        // declared in `GdnWorkspace`.
        let (h_n, v_n, g_n) = (10240usize, 6144usize, 48usize);
        let items = {
            let ws = self.venues[dev]
                .gdn_ws
                .as_ref()
                .ok_or_else(|| anyhow!("no GDN workspace on device {dev}"))?;
            [
                (&ws.qkv, h_n * 4),
                (&ws.z, v_n * 4),
                (&ws.alpha, g_n * 4),
                (&ws.beta, g_n * 4),
                (&ws.gate, g_n * 4),
                (&ws.step, v_n * 4),
                (&ws.normed, v_n * 4),
            ]
        };
        let refs: Vec<(&Buf, u64, usize)> = items.iter().map(|(b, n)| (*b, 0u64, *n)).collect();
        let raw = self.stagings[dev].read_batch(gpu, &refs)?;
        let c = |i: usize| Checksum::of(&f32_from_bytes(&raw[i]), 4);
        Ok(GdnTrace {
            h: c(0),
            z: c(1),
            alpha: c(2),
            beta: c(3),
            gate: c(4),
            step: c(5),
            normed: c(6),
        })
    }

    fn maybe_checksum(
        &mut self,
        wanted: bool,
        dev: usize,
        what: BufKind,
        n: usize,
    ) -> Result<Option<Checksum>> {
        if !wanted {
            return Ok(None);
        }
        Ok(Some(self.checksum(dev, what, n)?))
    }

    /// Download one of a venue's scratch buffers and digest it.
    fn checksum(&mut self, dev: usize, what: BufKind, n: usize) -> Result<Checksum> {
        self.flush()?;
        let gpu = self.model.devices[dev].gpu();
        let v = &self.venues[dev];
        let buf = match what {
            BufKind::R => &v.r,
            BufKind::Mixed => &v.hc_ws.as_ref().expect("hc workspace").mixed,
            BufKind::BlockOut => &v.block_out.buf,
        };
        let raw = self.stagings[dev].read_batch(gpu, &[(buf, 0, n * 4)])?;
        Ok(Checksum::of(&f32_from_bytes(&raw[0]), 8))
    }

    /// Prompt + generation. Decode-only timing: the prompt is fed one token at a
    /// time (there is no batched prefill path yet) and `decode_seconds` covers
    /// only the forwards that produced generated tokens.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        n: usize,
        debug: Option<&mut Vec<LayerTrace>>,
    ) -> Result<Generation> {
        if prompt.is_empty() {
            bail!("the prompt is empty: the n-gram window and the first token both need history");
        }
        self.reset()?;
        let mut trace = debug;
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.token(t, pos, None)?;
            pos += 1;
        }
        // The prompt is not part of the decode measurement.
        self.forwards = 0;
        self.expert_bytes = 0;
        self.timers.clear();
        // The prompt passes must not inflate the per-device figures.
        batch_cmd::reset_record_seconds();
        self.dev_wait_s.iter_mut().for_each(|x| *x = 0.0);
        self.dev_down.iter_mut().for_each(|x| *x = 0);
        self.dev_up.iter_mut().for_each(|x| *x = 0);

        let mut out = Vec::with_capacity(n);
        let mut first_step_logits = Vec::new();
        let mut last_logits = logits.clone();
        let t_decode = std::time::Instant::now();
        for i in 0..n {
            let next = self
                .sampler
                .sample(&logits, &self.sampling)
                .ok_or_else(|| anyhow!("the sampler produced no token"))?;
            out.push(next);
            if i + 1 == n {
                break;
            }
            let dbg = if i == 0 { trace.take() } else { None };
            logits = self.token(next, pos, dbg)?;
            pos += 1;
            if first_step_logits.is_empty() {
                first_step_logits = logits.clone();
            }
            last_logits = logits.clone();
        }
        let decode_seconds = t_decode.elapsed().as_secs_f64();
        Ok(Generation {
            tokens: out,
            last_logits,
            first_step_logits,
            decode_seconds,
            forwards: self.forwards,
            expert_bytes: self.expert_bytes,
        })
    }
}

impl Drop for Session<'_> {
    /// Free the scratch, workspaces and state this session allocated.
    ///
    /// The kernel crate's types free what *they* own on `Drop` (pipelines,
    /// quant tables), but a `HcWorkspace` / `GdnWorkspace` / `GdnState` /
    /// `QsaState` is a plain bag of `Buf`s, and every scratch buffer is a bare
    /// `Buf`: those are the caller's to release, and a leak here shows up as a
    /// `VUID-vkDestroyDevice-device-05137` at teardown.
    fn drop(&mut self) {
        let m = self.model;
        for (i, v) in self.venues.iter_mut().enumerate() {
            let gpu = m.devices[i].gpu();
            v.block_out.release(gpu);
            for b in [
                &v.r,
                &v.sh,
                &v.sh_gate_logit,
                &v.shexp_gate,
                &v.shexp_up,
                &v.shexp_act,
                &v.router_logits,
                &v.router_ids,
                &v.router_w,
                &v.ple_key_out,
                &v.ple_value_out,
                &v.ple_emb,
                &v.moe_gate,
                &v.moe_up,
                &v.moe_act,
                &v.moe_down,
                &v.moe_map_gate,
                &v.moe_map_up,
                &v.moe_map_down,
            ] {
                gpu.destroy_buf(b);
            }
            if let Some(l) = &v.logits {
                gpu.destroy_buf(l);
            }
            if let Some(ws) = v.hc_ws.take() {
                v.hc.release(gpu, ws);
            }
            if let Some(ws) = &v.gdn_ws {
                for b in [
                    &ws.qkv,
                    &ws.z,
                    &ws.alpha,
                    &ws.beta_raw,
                    &ws.beta,
                    &ws.gate,
                    &ws.step,
                    &ws.normed,
                ] {
                    gpu.destroy_buf(b);
                }
            }
        }
        for b in 0..m.cfg.n_layers {
            let gpu = m.devices[m.device_of(b)].gpu();
            if let Some(st) = &self.gdn_state[b] {
                gpu.destroy_buf(&st.state);
                gpu.destroy_buf(&st.conv);
            }
            if let Some(st) = &self.qsa_state[b] {
                for buf in [
                    &st.k_pool,
                    &st.v_pool,
                    &st.page_table,
                    &st.idx_tail,
                    &st.idx_pooled,
                    &st.positions,
                ] {
                    gpu.destroy_buf(buf);
                }
            }
        }
        for (b, ws) in self.qsa_ws.iter_mut().enumerate() {
            if let Some(ws) = ws.take() {
                let dev = m.device_of(b);
                if let Some(qsa) = self.venues[dev].qsa.as_ref() {
                    qsa.release(m.devices[dev].gpu(), ws);
                }
            }
        }
        for (i, s) in self.stagings.iter().enumerate() {
            s.release(m.devices[i].gpu());
        }
    }
}

/// A finished generation.
#[derive(Clone, Debug)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub last_logits: Vec<f32>,
    pub first_step_logits: Vec<f32>,
    pub decode_seconds: f64,
    /// Forward passes run after the prompt (the decode denominator).
    pub forwards: usize,
    pub expert_bytes: u64,
}

impl Generation {
    pub fn tokens_per_s(&self) -> f64 {
        if self.decode_seconds <= 0.0 || self.forwards == 0 {
            return 0.0;
        }
        self.forwards as f64 / self.decode_seconds
    }
    pub fn ms_per_token(&self) -> f64 {
        if self.forwards == 0 {
            return 0.0;
        }
        self.decode_seconds * 1e3 / self.forwards as f64
    }
    /// Expert weight bytes read per generated token.
    pub fn bytes_per_token(&self) -> u64 {
        if self.forwards == 0 {
            0
        } else {
            self.expert_bytes / self.forwards as u64
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BufKind {
    R,
    Mixed,
    BlockOut,
}

/// Touch one byte of each of `rows` from up to four threads, so 16 page faults
/// cost one fault latency instead of sixteen.
fn prefetch_rows(table: &[u8], rows: &[u32], row_bytes: usize) {
    if rows.len() < 2 || table.is_empty() {
        return;
    }
    let threads = rows.len().min(4);
    let per = rows.len().div_ceil(threads);
    std::thread::scope(|s| {
        for chunk in rows.chunks(per) {
            s.spawn(move || {
                let mut sink = 0u8;
                for r in chunk {
                    let at = (*r as usize).saturating_mul(row_bytes);
                    if at < table.len() {
                        // SAFETY: `at` is inside the mapping; a volatile read
                        // only stops the optimiser from dropping the touch.
                        sink ^= unsafe { std::ptr::read_volatile(table.as_ptr().add(at)) };
                    }
                }
                std::hint::black_box(sink);
            });
        }
    });
}

fn hc_read(v: &Venue<'_>, w: &HcW, r: &Buf) -> Result<()> {
    let hw = HcWeights {
        norm: &w.norm,
        down: &w.down.buf,
        up: &w.up.buf,
        inject: w.inject.as_ref(),
    };
    let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
    v.hc.read(ws, r, &hw)
}

/// Physical cores / 2, capped at 16 — the same default the host lane's own
/// benchmark uses (`breccia-kernels::cpu::bench::default_threads`, which is not
/// exported, so the rule is restated here).
fn default_threads() -> usize {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return 8;
    };
    let mut physical = String::new();
    let mut seen: std::collections::BTreeSet<(String, String)> = std::collections::BTreeSet::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            match k.trim() {
                "physical id" => physical = v.trim().to_string(),
                "core id" => {
                    seen.insert((physical.clone(), v.trim().to_string()));
                }
                _ => {}
            }
        }
    }
    (seen.len().max(1) / 2).clamp(1, 16)
}

/// The quant type of one named tensor.
fn ty_of(m: &Model, name: &str) -> Result<QuantType> {
    let t = m
        .shards
        .tensor(name)
        .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
    QuantType::from_ggml(t.ty())
        .ok_or_else(|| anyhow!("{name}: ggml type {} has no kernel", t.ty().id()))
}

impl Venue<'_> {
    /// One line for the banner.
    pub fn summary(&self) -> String {
        format!("{} · {} compiled pipelines", self.name, self.pipes.len())
    }
}

/// A `Gpu`-independent check that the head is the shape the graph assumes.
pub fn assert_head(m: &Model) -> Result<()> {
    let want = m.cfg.vocab;
    if m.head.key.rows != want {
        bail!(
            "the head has {} rows, expected {want} (= token_embd.ne1)",
            m.head.key.rows
        );
    }
    if m.head.key.cols != m.cfg.n_embd {
        bail!(
            "the head has {} columns, expected {}",
            m.head.key.cols,
            m.cfg.n_embd
        );
    }
    Ok(())
}
