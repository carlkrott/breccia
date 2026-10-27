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

/// Live sessions on this process, so the last one to drop can release the shared
/// dispatch ring's pools.
static LIVE_SESSIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Batches the `BRECCIA_GPU_TIME` probe holds before reading the pool out and
/// starting again (256 batches = ~5 tokens of a two-stream run).
const GPU_PROBE_BATCHES: usize = 256;

/// Dispatches after which a prefill chunk's batch is closed and the next one
/// opened. The batch's descriptor pool holds `MAX_SETS = 8 192` sets, and a
/// 256-token chunk records ~30 dispatches per token in pass 1 alone, so the cap
/// keeps a chunk (and its merged expert pass) inside the pool without making the
/// batches so short that the submissions dominate.
const MERGED_BATCH_CAP: u32 = 6000;

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
    /// The stage that was current when the **open batch** was started. A batch's
    /// dispatches belong to it: everything recorded into one command buffer is one
    /// stage's work by construction here (§8ap).
    #[serde(skip)]
    batch_stage: Option<usize>,
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
        let _ = stage;
    }

    /// Note the stage a batch was opened in, so its dispatch count lands there.
    fn open_batch(&mut self) {
        if self.on {
            self.batch_stage = Some(self.stage);
        }
    }

    /// Account one closed batch's dispatches to the stage it was opened in. `total`
    /// is the batch's own dispatch count (`batch_cmd::recorded()` at `end_submit`),
    /// so the per-stage table sums to the run's dispatch total by construction.
    fn count_batch(&mut self, total: u32) {
        if !self.on {
            return;
        }
        let stage = self.batch_stage.take().unwrap_or(self.stage);
        self.dispatches[stage] += total;
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
        self.batch_stage = None;
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

/// The merged prefill pass's per-session chunk scratch (§8v).
///
/// Every buffer is allocated once for the configured `prefill_chunk`, because the
/// pass runs once per block per chunk (48 × 2 per prompt, at chunk 128) and a
/// per-call allocation would be the whole cost of the thing it is trying to
/// delete.
pub struct PfChunk {
    /// The chunk's per-token MoE inputs, gathered inside pass 1's submission:
    /// `[T, k]` router ids (host-readable — the bucket list needs them),
    /// `[T, k]` router weights, `[T, n_embd]` ffn-mixed activation,
    /// `[T, n_embd]` shared-expert out and `[T]` its gate logit.
    ids: Mailbox,
    w: Buf,
    mixed: Buf,
    shared: Buf,
    gate: Buf,
    /// The host-built `(token, expert) → slot` map the fan-in reads.
    slots: Mailbox,
    /// `[T*k, n_embd]`: each distinct expert group's users, written contiguously.
    scratch: Buf,
    /// One expert group's GEMM scratch: `[8, n_embd]` activation gather,
    /// `[8, n_ff]` gate, `[8, n_ff]` up, `[8, n_ff]` SwiGLU, `[8, n_embd]` down.
    x: Buf,
    gg: Buf,
    uu: Buf,
    act: Buf,
    down: Buf,
    /// `[T, n_embd]`, the fan-in's output.
    out: Buf,
    /// **Streamed** (non-resident) experts: `STREAM_SLOTS` host-visible slots, each
    /// laid out `[gate | up | down]`, so one host memcpy per distinct expert
    /// replaces the lane's per-token dequantisation (§8x). The slot is copied to
    /// the device inside the batch; eight experts ride one submit.
    stream: Mailbox,
    /// One expert's three blobs on the device, for the streamed path.
    sf_gate: Buf,
    sf_up: Buf,
    sf_down: Buf,
    /// Per-slot stride inside [`PfChunk::stream`], and the per-role byte counts.
    slot_stride: u64,
    g_bytes: u64,
    u_bytes: u64,
    d_bytes: u64,
}

/// In-flight streamed experts per submit (see [`PfChunk::stream`]).
pub const STREAM_SLOTS: usize = 8;

impl PfChunk {
    fn bytes(&self) -> u64 {
        [
            &self.ids.buf,
            &self.w,
            &self.mixed,
            &self.shared,
            &self.gate,
            &self.slots.buf,
        ]
        .iter()
        .map(|b| b.size)
        .sum::<u64>()
            + [
                &self.scratch,
                &self.x,
                &self.gg,
                &self.uu,
                &self.act,
                &self.down,
                &self.out,
                &self.stream.buf,
                &self.sf_gate,
                &self.sf_up,
                &self.sf_down,
            ]
            .iter()
            .map(|b| b.size)
            .sum::<u64>()
    }

    fn release(&self, gpu: &breccia_vk::ctx::Gpu) {
        self.ids.release(gpu);
        self.slots.release(gpu);
        self.stream.release(gpu);
        for b in [
            &self.w,
            &self.mixed,
            &self.shared,
            &self.gate,
            &self.scratch,
            &self.x,
            &self.gg,
            &self.uu,
            &self.act,
            &self.down,
            &self.out,
            &self.sf_gate,
            &self.sf_up,
            &self.sf_down,
        ] {
            gpu.destroy_buf(b);
        }
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
    /// The chunked PLE phase's scratch (`docs/phase-4` §8aj), `Some` only for a
    /// chunked session: the whole chunk's gathered rows as one stack
    /// (`[t * ple_n_head, ple_dim]`), the chunk's packed table rows, the identity
    /// index list that gather reads, and **one key / value output buffer per
    /// token** — a `Buf` has no offset field, so a projection's output cannot be
    /// a row range of a stack.
    pub ple_emb_chunk: Option<Buf>,
    pub ple_packed: Option<Buf>,
    pub ple_idx: Option<Buf>,
    pub ple_key_slots: Vec<Buf>,
    pub ple_value_slots: Vec<Buf>,
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
    /// The chunk's weighted expert fan-in (§8v), used by the merged prefill pass.
    pub moe_fanin: breccia_kernels::MoeFanIn<'m>,
    /// The merged prefill pass's chunk scratch (`Some` only when
    /// `--prefill-merged` is on and the placement is single-device).
    pub pf: Option<PfChunk>,
    /// Chunked-prefill scratch: the chunk's gathered residual, one residual slice
    /// per token (the dense chain runs per token, unchanged) and one host-visible
    /// MoE mailbox per token (`docs/phase-4` §8s).
    pub r_chunk: Option<Buf>,
    pub r_slots: Vec<Buf>,
    /// `HC_READ(ffn)`'s `inject` partials per token: the ffn `HC_WRITE` of every
    /// token in a chunk runs after all the chunk's reads, so each token's inject
    /// values have to be kept (§8s).
    pub inject_slots: Vec<Buf>,
    pub moe_out: Vec<Mailbox>,
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
    /// Prefill chunk size: the prompt is fed `t` tokens at a time with one
    /// submission per stage per chunk and the expert lane **grouped by expert**
    /// over the chunk (§8s). `0` = token by token. Requires a single-device
    /// placement (which is what a multi-stream session is).
    pub prefill_chunk: usize,
    /// **Merged** prefill (§8v): for a chunk of `t` tokens, route the whole chunk
    /// once, bucket the `(token, expert)` pairs by distinct expert, and run one
    /// multi-user `GemmMany` per expert role over the resident arena
    /// (`--gpu-experts` blocks) instead of one GEMV per token. Off by default.
    /// Blocks whose experts are not resident keep the per-token host path.
    pub prefill_merged: bool,
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
    /// Tokens per prefill chunk (0 = token by token). The chunk's buffers live on
    /// the venue; the dense chain runs **per token** (unchanged) and only the
    /// expert lane is grouped (§8s). This is the **cap** the scratch is sized for
    /// ([`prefill_cap`]); the planner uses the target to split a prompt.
    pub prefill_chunk: usize,
    /// The `--prefill-chunk` target the cap came from ([`prefill_plan`]).
    prefill_target: usize,
    /// See [`SessionCfg::prefill_merged`].
    pub prefill_merged: bool,
    /// Host-side per-token routing for the chunk's grouped lane, and the chunk's
    /// token ids (the PLE hash needs them).
    pf_ids: Vec<Vec<usize>>,
    pf_weights: Vec<Vec<f32>>,
    pf_mixed: Vec<f32>,
    pf_toks: Vec<u32>,
    /// The merged pass's own counters (reset per measured window with the rest):
    /// host wall seconds of the expert GEMMs and of the fan-in, the dispatches
    /// they recorded, the distinct experts they touched and the expert bytes they
    /// read out of the arena (§8v).
    pf_gemm_s: f64,
    pf_fanin_s: f64,
    /// Streamed (non-resident) experts: bytes memcpy'd from the mmap, host seconds
    /// in those copies, and host seconds waiting for the staging ring.
    pf_stream_bytes: u64,
    pf_stream_fill_s: f64,
    pf_stream_wait_s: f64,
    /// Host seconds spent faulting PLE table rows in for the whole chunk.
    pf_ple_prefetch_s: f64,
    /// Fine-grained `shared-exp` attribution (§8ad): gate, up, swiglu, down, scalar
    /// gate in host seconds, then the pass-1 flush wait. Measurement only.
    sh_sub: [f64; 6],
    /// Wall seconds spent inside [`Session::prefill`] and the prompt tokens it
    /// covered — the denominator for the stage table's `sum vs total` line (§8ah).
    pf_wall_s: f64,
    pf_merged_dispatches: u64,
    pf_arena_bytes: u64,
    pf_merged_experts: usize,
    pf_merged_slots: usize,
    pf_merged_chunks: usize,
    /// Distinct experts the chunk's grouped lane touched, and the expert bytes a
    /// **merged** (device-side, once-per-expert) pass would move for it — the
    /// measured budget of §8u.
    pf_distinct: usize,
    pf_merged_bytes: u64,
    pf_tokens: usize,
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
    /// Host↔device bytes the **head path** moved: the residual hand-off when the
    /// head lives on another device, plus the logits row readback. Counted every
    /// token, so a regression that re-uploads the head's 416 MiB per token is
    /// visible in the report *and* caught by `head_traffic_ok` — the failure mode
    /// `--head-device 0` was first (wrongly) accused of
    /// (`docs/phase-4/01-forward-pass.md` §8aw).
    pub head_up: u64,
    pub head_down: u64,
    /// `(Σ down, Σ up)` over every staging buffer when the head path started.
    #[allow(clippy::type_complexity)]
    head_traffic0: (u64, u64),
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
        let head_dev = model.head_device();
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
            let chunk_slots = if i == 0 && cfg.prefill_chunk > 0 && model.devices.len() == 1 {
                prefill_cap(cfg.prefill_chunk)
            } else {
                1
            };
            // A chunked session that will batch PLE reads `t` tokens' residual/key/
            // value stacks back in one call, so its cap is the chunk's need, not the
            // single-readback default (§8ai; `ple_stage_bytes` is the tested rule).
            let stage_cap = if chunk_slots > 1 {
                stage_cap.max(crate::xfer::ple_stage_bytes(
                    chunk_slots,
                    mc.hc_dim(),
                    mc.n_embd,
                ))
            } else {
                stage_cap
            };
            stagings.push(Staging::with_block_region(gpu, stage_cap, chunk_slots)?);
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
            let chunk = if i == 0 && cfg.prefill_chunk > 0 && model.devices.len() == 1 {
                prefill_cap(cfg.prefill_chunk)
            } else {
                0
            };
            let (r_chunk, r_slots, inject_slots, moe_out) = {
                let mut chunk_buf = None;
                let mut slots = Vec::new();
                let mut injects = Vec::new();
                let mut boxes = Vec::new();
                if chunk > 0 {
                    chunk_buf = Some(alloc(
                        (mc.hc_dim() * 4 * chunk) as u64,
                        "the prefill chunk residual",
                    )?);
                    for t in 0..chunk {
                        slots.push(alloc(
                            mc.hc_dim() as u64 * 4,
                            &format!("prefill residual slot {t}"),
                        )?);
                        injects.push(alloc(160, &format!("prefill inject slot {t}"))?);
                        boxes.push(Mailbox::new(gpu, n * 4)?);
                    }
                }
                (chunk_buf, slots, injects, boxes)
            };
            // The chunked PLE phase's scratch (§8aj): the chunk's gathered rows,
            // its packed table rows, the identity index list and one key/value
            // output per token. Allocated once for the configured chunk, because a
            // per-call allocation would cost more than the submits it deletes.
            let (ple_emb_chunk, ple_packed, ple_idx, ple_key_slots, ple_value_slots) =
                if chunk > 0 && !mc.ple.layers.is_empty() {
                    let rows = chunk * mc.ple.n_head();
                    let idx: Vec<u8> = (0..rows as u32).flat_map(|i| i.to_le_bytes()).collect();
                    let mut keys = Vec::with_capacity(chunk);
                    let mut values = Vec::with_capacity(chunk);
                    for t in 0..chunk {
                        keys.push(alloc(mc.hc_dim() as u64 * 4, &format!("PLE key slot {t}"))?);
                        values.push(alloc(n * 4, &format!("PLE value slot {t}"))?);
                    }
                    (
                        Some(alloc(
                            rows as u64 * emb_ple.padded_cols() as u64 * 4,
                            "the prefill chunk PLE embeddings",
                        )?),
                        Some(alloc(
                            rows as u64 * model.ple_row_bytes() as u64,
                            "the prefill chunk PLE packed rows",
                        )?),
                        Some(
                            breccia_kernels::gemv::upload(gpu, &idx)
                                .context("uploading the PLE gather index list")?,
                        ),
                        keys,
                        values,
                    )
                } else {
                    (None, None, None, Vec::new(), Vec::new())
                };
            let moe_map = breccia_kernels::MoeMap::new(gpu)?;
            let moe_combine = breccia_kernels::MoeCombine::new(gpu)?;
            let moe_fanin = breccia_kernels::MoeFanIn::new(gpu)?;
            // The merged prefill pass's chunk scratch (§8v): only when the flag is
            // on, so a session that does not use it pays nothing.
            let pf = if chunk > 0 && cfg.prefill_merged && !model.placed.is_empty() {
                let k = mc.n_expert_used;
                let nb = mc.n_embd as u64 * 4;
                let nfb = mc.n_ff as u64 * 4;
                let users = breccia_kernels::gemm::USERS_MAX as u64;
                // Streamed experts: the largest role blob among the blocks that
                // are **not** resident (the resident ones address the arena).
                let (mut g_max, mut u_max, mut d_max) = (0usize, 0usize, 0usize);
                for b in 0..mc.n_layers {
                    if model.placed.contains_key(&b) {
                        continue;
                    }
                    let e = &model.layers[b].experts;
                    g_max = g_max.max(e.gate.bytes());
                    u_max = u_max.max(e.up.bytes());
                    d_max = d_max.max(e.down.bytes());
                }
                let align = |v: usize| (v as u64 + 255) & !255;
                let (gb, ub, db) = (align(g_max), align(u_max), align(d_max));
                let slot_stride = gb + ub + db;
                Some(PfChunk {
                    ids: Mailbox::new(gpu, chunk as u64 * k as u64 * 4)?,
                    w: alloc(chunk as u64 * k as u64 * 4, "prefill router weights")?,
                    mixed: alloc(chunk as u64 * nb, "prefill chunk mixed")?,
                    shared: alloc(chunk as u64 * nb, "prefill chunk shared")?,
                    gate: alloc(chunk as u64 * 4, "prefill chunk shared gate")?,
                    slots: Mailbox::new(gpu, chunk as u64 * k as u64 * 4)?,
                    scratch: alloc(chunk as u64 * k as u64 * nb, "prefill merged scratch")?,
                    x: alloc(users * nb, "prefill expert gather")?,
                    gg: alloc(users * nfb, "prefill expert gate")?,
                    uu: alloc(users * nfb, "prefill expert up")?,
                    act: alloc(users * nfb, "prefill expert act")?,
                    down: alloc(users * nb, "prefill expert down")?,
                    out: alloc(chunk as u64 * nb, "prefill merged out")?,
                    stream: Mailbox::new(gpu, STREAM_SLOTS as u64 * slot_stride)?,
                    sf_gate: alloc(gb.max(16), "streamed expert gate")?,
                    sf_up: alloc(ub.max(16), "streamed expert up")?,
                    sf_down: alloc(db.max(16), "streamed expert down")?,
                    slot_stride,
                    g_bytes: gb,
                    u_bytes: ub,
                    d_bytes: db,
                })
            } else {
                None
            };
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
                ple_emb_chunk,
                ple_packed,
                ple_idx,
                ple_key_slots,
                ple_value_slots,
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
                moe_fanin,
                pf,
                r_chunk,
                r_slots,
                inject_slots,
                moe_out,
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
        LIVE_SESSIONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
            prefill_chunk: if model.devices.len() == 1 {
                prefill_cap(cfg.prefill_chunk)
            } else {
                0
            },
            prefill_target: if model.devices.len() == 1 {
                cfg.prefill_chunk
            } else {
                0
            },
            prefill_merged: if model.devices.len() == 1 {
                cfg.prefill_merged
            } else {
                false
            },
            pf_ids: Vec::new(),
            pf_weights: Vec::new(),
            pf_mixed: Vec::new(),
            pf_toks: Vec::new(),
            pf_gemm_s: 0.0,
            pf_fanin_s: 0.0,
            pf_stream_bytes: 0,
            pf_stream_fill_s: 0.0,
            pf_stream_wait_s: 0.0,
            pf_ple_prefetch_s: 0.0,
            sh_sub: [0.0; 6],
            pf_wall_s: 0.0,
            pf_merged_dispatches: 0,
            pf_arena_bytes: 0,
            pf_merged_experts: 0,
            pf_merged_slots: 0,
            pf_merged_chunks: 0,
            pf_distinct: 0,
            pf_merged_bytes: 0,
            pf_tokens: 0,
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
            head_up: 0,
            head_down: 0,
            head_traffic0: (0, 0),
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
            self.timers.count_batch(rec);
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

    /// Census-only batch split (`BRECCIA_TRACE_DISPATCH=1`): close the open batch and
    /// open the next one, so the dispatch counter can attribute a whole command
    /// buffer's dispatches to the phase that produced them (§8as). Costs a submit per
    /// call, which is why it is **only** on under the census flag — the numbers it
    /// prints are for attribution, not for quoting as performance.
    fn census_split(&mut self, dev: usize) -> Result<()> {
        if std::env::var_os("BRECCIA_TRACE_DISPATCH").is_none() {
            return Ok(());
        }
        self.flush()?;
        self.ensure_batch(dev)
    }

    /// Submit the open batch without waiting; the fence is kept in
    /// [`Session::pending_seg`] so [`Session::finish_block`] can wait for it.
    fn submit_batch(&mut self) -> Result<()> {
        let Some(d) = self.pending_batch.take() else {
            return Ok(());
        };
        let gpu = self.model.devices[d].gpu();
        let rec = batch_cmd::recorded();
        self.timers.count_batch(rec);
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
        self.timers.open_batch();
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
        // The pool holds `GPU_PROBE_BATCHES` pairs and **cannot be read while its
        // submissions are in flight** (`vkGetQueryPoolResults` without
        // `WAIT` returns NOT_READY, and this probe used to abort the run there).
        // Reading can only happen once the caller has waited, which is at the end
        // of the token — so when the pool is full the probe simply stops stamping
        // and keeps its samples (§8am).
        if k as usize >= GPU_PROBE_BATCHES {
            return Ok(());
        }
        let pool = self.probe_pools[dev];
        let query = 2 * k + u32::from(end);
        batch_cmd::record_timestamp(self.model.devices[dev].gpu(), pool, query)?;
        if end {
            self.probe_batches = Some(k + 1);
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
        // `vkGetQueryPoolResults` without `WAIT` returns NOT_READY if the
        // submission that wrote the stamps is still in flight — which it was, on
        // the first token, and this probe aborted the run there (§8am). Drain this
        // session's segments first; the callers (`gpu_busy_s`, the token report) are
        // outside any batch.
        self.flush()?;
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
        self.pf_distinct = 0;
        self.pf_merged_bytes = 0;
        self.pf_tokens = 0;
        self.pf_gemm_s = 0.0;
        self.pf_fanin_s = 0.0;
        self.pf_stream_bytes = 0;
        self.pf_stream_fill_s = 0.0;
        self.pf_stream_wait_s = 0.0;
        self.pf_ple_prefetch_s = 0.0;
        self.sh_sub = [0.0; 6];
        self.pf_wall_s = 0.0;
        self.pf_merged_dispatches = 0;
        self.pf_arena_bytes = 0;
        self.pf_merged_experts = 0;
        self.pf_merged_slots = 0;
        self.pf_merged_chunks = 0;
        self.forwards = 0;
        self.expert_bytes = 0;
        self.lane_s = 0.0;
        self.lane_covered_s = 0.0;
        self.dev_wait_s.iter_mut().for_each(|x| *x = 0.0);
        self.dev_down.iter_mut().for_each(|x| *x = 0);
        self.dev_up.iter_mut().for_each(|x| *x = 0);
        self.head_up = 0;
        self.head_down = 0;
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
        // **Dispatch census** (`BRECCIA_TRACE_DISPATCH=1`): dispatches per stage for
        // one token. The stage tags are the graph's own marks, so this says *where*
        // the ~4 700 dispatches/token come from without any GPU work (§8an).
        if std::env::var_os("BRECCIA_TRACE_DISPATCH").is_some() {
            let rows: Vec<String> = STAGES
                .iter()
                .zip(self.timers.dispatches.iter())
                .filter(|(_, n)| **n > 0)
                .map(|(name, n)| format!("{name} {n}"))
                .collect();
            eprintln!(
                "dispatch census token {}: {} · total {}",
                self.forwards,
                rows.join(" · "),
                self.timers.dispatches.iter().sum::<u32>()
            );
            self.timers.dispatches = [0; STAGES.len()];
        }
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
        self.head_up = 0;
        self.head_down = 0;
        self.timers.clear();
        Ok(())
    }

    /// `(Σ device→host, Σ host→device)` bytes over every staging buffer.
    fn staging_traffic(&self) -> (u64, u64) {
        self.stagings
            .iter()
            .fold((0, 0), |(d, u), s| (d + s.bytes_down, u + s.bytes_up))
    }

    /// The head path's per-token host traffic, and whether it is what it should be.
    ///
    /// The *only* thing that may travel towards the head device is the residual
    /// stack (`hc_dim` floats), and the only thing that may come back is that plus
    /// the logits row (`vocab` floats). Anything else would mean the 416 MiB of head
    /// weights — or worse, the whole residual stack per block — is being re-sent
    /// every token, which is the "738 ms/token" failure mode
    /// `--head-device 0` was first accused of (`docs/phase-4/01-forward-pass.md`
    /// §8aw: it did not reproduce; the window had a foreign GPU process on the
    /// cards). This is the assertion a slow path has to fail.
    pub fn head_traffic_ok(&self, forwards: usize) -> Result<(f64, f64)> {
        if self.head_device() == self.model.device_of(self.model.cfg.n_layers - 1) {
            // No hand-off: the head owns the residual, so nothing should cross at all
            // beyond the logits row.
            if self.head_up != 0 {
                bail!(
                    "the head path uploaded {} bytes with no hand-off (expected 0)",
                    self.head_up
                );
            }
        }
        let n = forwards.max(1) as f64;
        let up = self.head_up as f64 / n;
        let down = self.head_down as f64 / n;
        let mc = &self.model.cfg;
        let resid = crate::head::handoff_bytes(mc.hc_dim()) as f64;
        // 4x slack: one residual up, and up to one residual + the logits row down.
        if up > 4.0 * resid {
            bail!(
                "the head path uploaded {:.1} MiB/token towards device {} for a {:.0} KiB                  residual — the head's weights must not move per token",
                up / (1 << 20) as f64,
                self.head_device(),
                resid / 1024.0
            );
        }
        if down > resid + (mc.vocab * 4) as f64 {
            bail!(
                "the head path read back {:.1} MiB/token (residual {:.0} KiB + logits {:.0} KiB                  expected)",
                down / (1 << 20) as f64,
                resid / 1024.0,
                (mc.vocab * 4) as f64 / 1024.0
            );
        }
        Ok((up, down))
    }

    /// The device the head runs on (for the report).
    pub fn head_device(&self) -> usize {
        self.model.head_device()
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
            if let Some(pf) = &v.pf {
                total += pf.bytes();
            }
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

        let digests = self.record_dense_prelude(b, dev, pos, trace)?;

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
            return Ok(digests);
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
        Ok(digests)
    }

    /// One token's dense chain for block `b`: `HC_READ(attn)`, the token mixer,
    /// `HC_WRITE(attn)`, `HC_READ(ffn)`, the router and the shared expert — all
    /// recorded into the open batch, with the venue's single-token scratch.
    ///
    /// Shared by the decode path ([`Session::submit_block`]) and the chunked
    /// prefill, which swaps each token's residual into `v.r` and keeps the whole
    /// chunk in one submission.
    fn record_dense_prelude(
        &mut self,
        b: usize,
        dev: usize,
        pos: usize,
        trace: &mut Option<&mut Vec<LayerTrace>>,
    ) -> Result<BlockDigests> {
        let m = self.model;
        let mc = &m.cfg;
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
        let mut sub = [0f64; 5];
        {
            let lw = &m.layers[b];
            let v = &self.venues[dev];
            let pipes = &v.pipes;
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            // `gate` and `up` both read `ws.mixed` and write disjoint outputs, so
            // the fence between them is dead: drop *gate's* trailing fence (§8z).
            batch_cmd::skip_next_barrier();
            // Fine-grained attribution (§8ad): this stage measures as 15-21 ms/token
            // while the routed shared expert prices at ~0.9, so time each call.
            let t0 = std::time::Instant::now();
            pipes
                .get(&lw.moe.gate.key)?
                .run(&lw.moe.gate.buf, &ws.mixed, &v.shexp_gate)?;
            sub[0] = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            pipes
                .get(&lw.moe.up.key)?
                .run(&lw.moe.up.buf, &ws.mixed, &v.shexp_up)?;
            sub[1] = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            v.swiglu
                .run(&v.shexp_gate, &v.shexp_up, &v.shexp_act, mc.n_ff)?;
            sub[2] = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            pipes
                .get(&lw.moe.down.key)?
                .run(&lw.moe.down.buf, &v.shexp_act, &v.sh)?;
            sub[3] = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            pipes.get(&lw.moe.shexp_gate.key)?.run(
                &lw.moe.shexp_gate.buf,
                &ws.mixed,
                &v.sh_gate_logit,
            )?;
            sub[4] = t0.elapsed().as_secs_f64();
        }
        for (acc, d) in self.sh_sub.iter_mut().zip(sub) {
            *acc += d;
        }

        Ok(BlockDigests {
            mixed_attn,
            mixer_out,
            mixed_ffn,
        })
    }

    /// Feed a **chunk** of `toks` in one pass: the dense chain still runs per
    /// token (unchanged kernels, unchanged numerics), but the whole chunk rides
    /// **one submission per stage**, and the routed experts are computed **once
    /// per expert** across the chunk instead of once per token
    /// (`moe_topk_chunk`, §8s).
    ///
    /// Requires a single-device placement (what a multi-stream session is) and a
    /// configured [`SessionCfg::prefill_chunk`].
    pub fn prefill(&mut self, toks: &[u32], pos0: usize) -> Result<Vec<f32>> {
        let t = toks.len();
        if t == 0 {
            bail!("an empty prefill chunk");
        }
        let m = self.model;
        let mc = &m.cfg;
        if m.devices.len() != 1 {
            bail!(
                "chunked prefill needs a single-device placement (this model spans {} devices)",
                m.devices.len()
            );
        }
        if t > self.prefill_chunk {
            bail!(
                "prefill chunk of {t} tokens exceeds the session's chunk cap of {} \
                 (`prefill_cap(target)`; see `prefill_plan`)",
                self.prefill_chunk
            );
        }
        if pos0 + t > mc.n_ctx_max {
            bail!(
                "chunk at {pos0}+{t} is past the context limit {}",
                mc.n_ctx_max
            );
        }
        if !self.batching {
            bail!("chunked prefill records into a dispatch batch: it needs --batching");
        }
        let dev = 0;
        let gpu = m.devices[dev].gpu();
        let hc_dim = mc.hc_dim();
        let wall0 = std::time::Instant::now();
        let subs0 = self.submits;
        // Every queue submission, not just the batch ring's: the staging
        // readbacks and the one-shot uploads are what PLE batching removes, and a
        // batch-level counter cannot see them (§8aj).
        let q0 = breccia_vk::ctx::submits();
        self.pf_toks.clear();
        self.pf_toks.extend_from_slice(toks);
        self.pf_tokens += t;

        // ---- embed the chunk, then split it into per-token residual slots -----
        self.flush()?;
        self.timers.start();
        self.pos = pos0 + t - 1;
        if self.venues[dev].r_chunk.is_none() {
            bail!("no prefill chunk buffer: was prefill_chunk set?");
        }
        {
            let v = &self.venues[dev];
            let table = m.token_embd()?;
            let rows: Vec<u32> = toks
                .iter()
                .flat_map(|&x| std::iter::repeat_n(x, mc.hc))
                .collect();
            let rc = v
                .r_chunk
                .as_ref()
                .ok_or_else(|| anyhow!("no prefill chunk buffer"))?;
            let staged = v.emb.gather_staged(gpu, table, &rows, rc)?;
            v.emb.release(gpu, &staged);
        }
        self.ensure_batch(dev)?;
        for i in 0..t {
            let v = &self.venues[dev];
            batch_cmd::record_copy(
                gpu,
                &v.r_slots[i],
                0,
                v.r_chunk.as_ref().expect("the chunk residual"),
                (i * hc_dim * 4) as u64,
                (hc_dim * 4) as u64,
            )?;
        }
        self.submit_batch()?;
        self.flush()?;
        self.timers.mark(PLE);

        for b in 0..mc.n_layers {
            self.prefill_block(b, t, pos0)?;
        }
        // `BRECCIA_TRACE_PREFILL=1`: the chunk's own stage split, ms per prompt
        // token, so "the dense chain" and "the expert lane" are numbers and not
        // guesses. This is what decides whether a T-batched dense chain can pay.
        if std::env::var_os("BRECCIA_TRACE_SHARED").is_some() {
            let t = t.max(1) as f64;
            let names = [
                "sh_gate",
                "sh_up",
                "sh_swiglu",
                "sh_down",
                "sh_scalar_gate",
                "pass1_flush",
            ];
            let rows: Vec<String> = names
                .iter()
                .zip(self.sh_sub.iter())
                .map(|(n, s)| format!("{n} {:.2}", s * 1e3 / t))
                .collect();
            eprintln!(
                "shared-exp blk48 t={t} (ms/token): {} · total {:.2}",
                rows.join(" · "),
                self.sh_sub.iter().sum::<f64>() * 1e3 / t
            );
            self.sh_sub = [0.0; 6];
        }
        if std::env::var_os("BRECCIA_TRACE_PREFILL").is_some() {
            let rows: Vec<String> = self
                .timers
                .table(t)
                .into_iter()
                .filter(|(_, ms)| *ms > 0.01)
                .map(|(name, ms)| format!("{name} {ms:.2}"))
                .collect();
            let sum_ms = self.timers.total_s() * 1e3 / t as f64;
            // This chunk's own wall: the accumulator is only updated on return, so
            // using it here printed 0.00 (which is the whole point of the line).
            let wall_ms = wall0.elapsed().as_secs_f64() * 1e3 / t as f64;
            eprintln!(
                "prefill stages blk48 t={t} (ms/token): {}",
                rows.join(" · ")
            );
            eprintln!(
                "prefill sum {:>7.2} · wall {:>7.2} · residual {:>7.2} ms/token \
                 ({:.0}% unattributed) over {} tokens · {} submits ({:.2}/block-chunk,                  {} from the batch ring)",
                sum_ms,
                wall_ms,
                wall_ms - sum_ms,
                if wall_ms > 0.0 {
                    (wall_ms - sum_ms) / wall_ms * 100.0
                } else {
                    0.0
                },
                t,
                breccia_vk::ctx::submits() - q0,
                (breccia_vk::ctx::submits() - q0) as f64 / mc.n_layers.max(1) as f64,
                self.submits - subs0
            );
            self.timers.clear();
        }

        // ---- the head sees the chunk's last token only -----------------------
        self.timers.mark(FINAL_HC);
        self.swap_r_in(dev, t - 1);
        self.submit_head()?;
        let logits = self.finish_head()?;
        self.swap_r_in(dev, t - 1);
        self.timers.stop();
        self.pf_wall_s += wall0.elapsed().as_secs_f64();
        Ok(logits)
    }

    /// One block of a chunked prefill: the dense chain for every token of the
    /// chunk in one submission, then the **grouped** expert lane, then the ffn
    /// `HC_WRITE` for every token in a second submission.
    fn prefill_block(&mut self, b: usize, t: usize, pos0: usize) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let dev = 0;
        let n_embd = mc.n_embd;
        let k = mc.n_expert_used;
        let mut no_trace: Option<&mut Vec<LayerTrace>> = None;

        // ---- PLE: one batched phase for the whole chunk (§8aj) ---------------
        if mc.is_ple_layer(b) {
            self.timers.mark(PLE);
            // The batched form needs the chunk scratch, a chunk of more than one
            // token and no `BRECCIA_NO_PLE_BATCH` (the A/B arm, and the revert
            // path). Everything else keeps the per-token host round trip.
            let batched = t > 1
                && self.venues[dev].ple_emb_chunk.is_some()
                && std::env::var_os("BRECCIA_NO_PLE_BATCH").is_none();
            if batched {
                self.ple_chunk(b, t)?;
            } else {
                // Batch the IO half: hash every token's 16 table rows up front and
                // fault them in with **one** `prefetch_rows` over `t * 16` rows, so
                // the 90-byte-within-a-4-KiB-page reads of a 28.8 GiB table stop
                // serialising per token (`docs/research/03`: 65 µs per cold row,
                // 48x extent over-read). The host blocks stay sequential — the
                // rolling context and the conv state are per token by definition.
                let table = self.model.ple_table()?;
                let row_bytes = self.model.ple_row_bytes();
                let mut per: Vec<Vec<u32>> = Vec::with_capacity(t);
                let mut all: Vec<u32> = Vec::with_capacity(t * 16);
                {
                    let ple = self
                        .ple
                        .as_mut()
                        .ok_or_else(|| anyhow!("no PLE host state"))?;
                    for i in 0..t {
                        let r = ple.rows_for(self.pf_toks[i]);
                        all.extend_from_slice(&r);
                        per.push(r);
                    }
                }
                let t_prefetch = std::time::Instant::now();
                prefetch_rows(table, &all, row_bytes);
                self.pf_ple_prefetch_s += t_prefetch.elapsed().as_secs_f64();
                for (i, rows) in per.iter().enumerate() {
                    self.swap_r_in(dev, i);
                    self.cur_tok = self.pf_toks[i];
                    self.ple_step_rows(b, rows)?;
                    self.swap_r_in(dev, i);
                }
            }
        }

        // ---- merged expert pass (§8v): resident blocks only ------------------
        // `--prefill-merged` routes the chunk once per block, buckets the
        // `(token, expert)` pairs and runs one multi-user GEMM per distinct expert
        // instead of one GEMV per token. A block whose role type has no multi-user
        // GEMM, or a session without the chunk scratch, keeps the path below.
        if self.prefill_merged && self.venues[dev].pf.is_some() {
            let pe = m.placed.get(&b);
            let plan = &m.layers[b].experts;
            let (gt, ut, dt) = match pe {
                Some(pe) => (pe.gate.key.ty, pe.up.key.ty, pe.down.key.ty),
                None => (plan.gate.ty, plan.up.ty, plan.down.ty),
            };
            let pipes = &self.venues[dev].pipes;
            // A role type with no multi-user GEMM keeps the per-token path for
            // that block (documented fallback, §8v/§8x).
            let ready = [gt, ut, dt].iter().all(|ty| pipes.gemm(*ty).is_some());
            if ready {
                return self.prefill_block_merged(b, t, pos0, pe);
            }
        }

        // ---- pass 1: the whole chunk's dense chain, one submission -----------
        self.ensure_batch(dev)?;
        for i in 0..t {
            self.swap_r_in(dev, i);
            self.record_dense_prelude(b, dev, pos0 + i, &mut no_trace)?;
            if m.placed.contains_key(&b) {
                // A device-resident block needs no host round trip; run it whole
                // (documented per-token fallback for that block only).
                self.run_placed_moe(b, dev, &m.placed[&b], n_embd, mc.n_ff, k)?;
                let v = &self.venues[dev];
                let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
                v.hc.write(ws, &v.r, &v.block_out.buf, &v.r)?;
            } else {
                self.record_lane_copies(dev, i)?;
                let v = &self.venues[dev];
                let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
                batch_cmd::record_copy(
                    self.model.devices[dev].gpu(),
                    &v.inject_slots[i],
                    0,
                    &ws.inject,
                    0,
                    ws.inject.size.min(160),
                )?;
            }
            self.swap_r_in(dev, i);
            // A chunk of 256 tokens records ~30 dispatches per token, which would
            // overrun the batch's descriptor pool (8 192 sets): close it and open
            // the next one. Tokens are independent, so this changes nothing.
            if batch_cmd::recorded() >= MERGED_BATCH_CAP {
                self.flush()?;
                self.ensure_batch(dev)?;
            }
        }
        self.submit_batch()?;
        // Mark the drain explicitly (§8ae): the shared expert is the last stage a
        // block records, so without this mark the flush() below is charged to it —
        // which is exactly how `shared-exp` read 19-21 ms/token while its own work
        // was 0.10 ms (§8ad). A stage mark must never be able to swallow a wait.
        self.timers.mark(WAIT_GPU);
        let t_pass1 = std::time::Instant::now();
        self.flush()?;
        self.sh_sub[5] += t_pass1.elapsed().as_secs_f64();

        if m.placed.contains_key(&b) {
            return Ok(());
        }

        // ---- the grouped expert lane over the whole chunk --------------------
        self.timers.mark(EXPERTS);
        self.pf_ids.clear();
        self.pf_weights.clear();
        self.pf_mixed.clear();
        let mut shared: Vec<f32> = Vec::with_capacity(t * n_embd);
        let mut gates: Vec<f32> = Vec::with_capacity(t);
        {
            let st = &self.stagings[dev];
            let offs = lane_offsets(n_embd, k);
            for i in 0..t {
                let base = crate::xfer::BLOCK_REGION * i as u64;
                self.pf_mixed
                    .extend_from_slice(&f32_from_bytes(st.read_block(base + offs[0], n_embd * 4)));
                let ids: Vec<usize> = st
                    .read_block(base + offs[1], k * 4)
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                    .collect();
                self.pf_ids.push(ids);
                self.pf_weights
                    .push(f32_from_bytes(st.read_block(base + offs[2], k * 4)));
                shared
                    .extend_from_slice(&f32_from_bytes(st.read_block(base + offs[3], n_embd * 4)));
                let g = st.read_block(base + offs[4], 4);
                gates.push(f32::from_le_bytes([g[0], g[1], g[2], g[3]]));
            }
        }
        self.dev_down[dev] += (t * (n_embd * 2 + k * 8 + 4)) as u64;
        let t_lane = std::time::Instant::now();
        let mut out = vec![0f32; t * n_embd];
        {
            let routes: Vec<breccia_kernels::cpu::ChunkRoute<'_>> = (0..t)
                .map(|i| breccia_kernels::cpu::ChunkRoute {
                    ids: &self.pf_ids[i],
                    weights: &self.pf_weights[i],
                })
                .collect();
            let bytes = self
                .pool
                .lock()
                .expect("the expert pool's lock")
                .moe_topk_chunk(&m.layers[b].experts, &routes, &self.pf_mixed, &mut out)
                .with_context(|| format!("the grouped expert lane on block {b}"))?;
            self.expert_bytes += bytes as u64;
            let distinct = self
                .pool
                .lock()
                .map(|p| p.last_distinct)
                .unwrap_or_default();
            self.pf_distinct += distinct;
            if std::env::var_os("BRECCIA_TRACE_ROUTE").is_some() {
                eprintln!(
                    "route blk {b:>2} t={t} draws={} distinct={distinct} uses/token={:.2}",
                    t * k,
                    distinct as f64 / t as f64
                );
            }
            self.pf_merged_bytes += (distinct as u64) * m.layers[b].experts.expert_bytes() as u64;
        }
        // `BRECCIA_TRACE_CHUNK=1`: one line per (block, token) with the lane's
        // inputs and its raw output, so a chunk-size disagreement can be traced to
        // the readback, the lane or the ffn `HC_WRITE` instead of guessed at.
        if std::env::var_os("BRECCIA_TRACE_CHUNK").is_some() {
            for i in 0..t {
                let mx = &self.pf_mixed[i * n_embd..i * n_embd + 3];
                let sh = &shared[i * n_embd..i * n_embd + 3];
                let ids = self.pf_ids[i]
                    .iter()
                    .take(5)
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let o = &out[i * n_embd..i * n_embd + 3];
                eprintln!(
                    "chunk blk {b:>2} tok {i:>3} mixed={mx:?} sh={sh:?} gate={:.6} ids=[{ids}] out={o:?}",
                    gates[i]
                );
            }
        }
        self.lane_s += t_lane.elapsed().as_secs_f64();
        self.lane_covered_s += self.lane_s.min(0.0); // set by the driver when relevant
        for i in 0..t {
            let gate = 1.0 / (1.0 + (-gates[i]).exp());
            let row = &mut out[i * n_embd..(i + 1) * n_embd];
            for (o, sv) in row.iter_mut().zip(&shared[i * n_embd..(i + 1) * n_embd]) {
                *o += sv * gate;
            }
            self.venues[dev].moe_out[i].write(&f32_to_bytes(row))?;
            self.dev_up[dev] += (n_embd * 4) as u64;
        }

        // ---- pass 2: the ffn HC_WRITE for every token, one submission --------
        self.timers.mark(HC_WRITE_F);
        if std::env::var_os("BRECCIA_TRACE_CHUNK").is_some() {
            let mut rows = Vec::new();
            for i in 0..t {
                let v = &self.venues[dev];
                let gpu = m.devices[dev].gpu();
                let b = self.stagings[dev]
                    .read_batch(gpu, &[(&v.inject_slots[i], 0, 16)])
                    .map(|x| x[0].clone())
                    .unwrap_or_default();
                let f: Vec<f32> = b
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                rows.push(format!("{i}:{f:?}"));
            }
            eprintln!("chunk blk {b:>2} inject {rows:?}");
        }
        self.ensure_batch(dev)?;
        for i in 0..t {
            self.swap_r_in(dev, i);
            self.swap_inject_in(dev, i);
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.hc.write(ws, &v.r, &v.moe_out[i].buf, &v.r)?;
            self.swap_inject_in(dev, i);
            self.swap_r_in(dev, i);
        }
        self.submit_batch()?;
        self.flush()?;
        Ok(())
    }

    /// Copy expert `e` of block `b` out of the mmap into streamed slot `slot`.
    ///
    /// This is the whole replacement for the host lane's per-token work: the lane
    /// decoded the same expert `m` times (once per token that routed to it) and
    /// dotted each separately; here the blob is copied **once per chunk** and the
    /// dequantisation happens once per *user group* on the GPU. A chunk of 128
    /// tokens gives 8-13 users per distinct expert, so the bytes read from the
    /// mmap drop by that factor too — which matters because the lane is I/O-bound
    /// on this box (§8x).
    fn fill_stream_slot(&mut self, dev: usize, b: usize, expert: usize, slot: usize) -> Result<()> {
        let m = self.model;
        let t0 = std::time::Instant::now();
        let slices = m.layers[b]
            .experts
            .slices(&m.shards, expert)
            .with_context(|| format!("streaming blk.{b} expert {expert}"))?;
        let total = {
            let pf = self.venues[dev]
                .pf
                .as_ref()
                .expect("the merged prefill scratch");
            let off = slot as u64 * pf.slot_stride;
            let (g, u, d) = (pf.g_bytes, pf.u_bytes, pf.d_bytes);
            if (slices.gate.len() as u64) > g
                || (slices.up.len() as u64) > u
                || (slices.down.len() as u64) > d
            {
                bail!(
                    "blk.{b} expert {expert}: a role blob ({} / {} / {}) exceeds the streamed \
                     slots ({g} / {u} / {d})",
                    slices.gate.len(),
                    slices.up.len(),
                    slices.down.len()
                );
            }
            pf.stream.write_at(off, slices.gate)?;
            pf.stream.write_at(off + g, slices.up)?;
            pf.stream.write_at(off + g + u, slices.down)?;
            slices.gate.len() + slices.up.len() + slices.down.len()
        };
        self.pf_stream_bytes += total as u64;
        self.pf_stream_fill_s += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// One block of a chunk with the **merged** expert pass (§8v).
    ///
    /// Pass 1 is the same dense chain per token, but a block whose experts are
    /// resident no longer runs one GEMV per token: it keeps the chunk's router ids,
    /// weights, `mixed`, shared-expert output and gate logit in chunk buffers. The
    /// host then builds the `(token, expert)` bucket list, and per distinct expert
    /// **three multi-user GEMMs** (`gate`, `up`, `down`) read that expert's blob
    /// **once** out of the arena and dot it with all of its users, followed by one
    /// chunk-wide fan-in that does what `MoeCombine` does per token.
    ///
    /// Numerics: every user's GEMM dot is bit-identical to the single-user kernel
    /// it replaces (§8v, measured for every `m`), the SwiGLU is elementwise, and the
    /// fan-in is bit-identical to `MoeCombine` in router order — so the generated
    /// tokens must match the token-by-token path exactly.
    fn prefill_block_merged(
        &mut self,
        b: usize,
        t: usize,
        pos0: usize,
        pe: Option<&'m crate::model::PlacedExperts>,
    ) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let dev = 0usize;
        let n_embd = mc.n_embd;
        let n_ff = mc.n_ff;
        let k = mc.n_expert_used;
        let gpu = m.devices[dev].gpu();
        let plan = &m.layers[b].experts;
        let (gate_ty, up_ty, down_ty) = match pe {
            Some(pe) => (pe.gate.key.ty, pe.up.key.ty, pe.down.key.ty),
            None => (plan.gate.ty, plan.up.ty, plan.down.ty),
        };
        // A **streamed** block (no arena): each distinct expert's three blobs are
        // copied once per chunk out of the mmap into a host-visible slot and then
        // to the device, and the multi-user GEMM runs on them with base 0.
        let streamed = pe.is_none();
        // Each expert's blob, in the 16-bit words the GEMM's base push constant
        // takes (the arena holds all 512 experts back to back).
        let words = |ty: breccia_kernels::QuantType, rows: usize, cols: usize| -> u32 {
            let bytes = rows * cols.div_ceil(ty.block_elems()) * ty.block_bytes();
            assert!(
                bytes.is_multiple_of(2),
                "an expert blob must be a whole u16 word count"
            );
            (bytes / 2) as u32
        };
        let (wgate, wup, wdown) = (
            words(gate_ty, n_ff, n_embd),
            words(up_ty, n_ff, n_embd),
            words(down_ty, n_embd, n_ff),
        );
        let mut no_trace: Option<&mut Vec<LayerTrace>> = None;
        let disp_before = self.dispatches;

        // ---- pass 1: the dense chain per token, the MoE inputs saved ---------
        self.ensure_batch(dev)?;
        for i in 0..t {
            self.swap_r_in(dev, i);
            self.record_dense_prelude(b, dev, pos0 + i, &mut no_trace)?;
            {
                let v = &self.venues[dev];
                let pf = v.pf.as_ref().expect("the merged prefill scratch");
                let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
                let kb = (k * 4) as u64;
                let nb = (n_embd * 4) as u64;
                // Six copies, one region array (and one barrier) per token.
                batch_cmd::record_copies(
                    gpu,
                    &[
                        (&pf.ids.buf, i as u64 * kb, &v.router_ids, 0, kb),
                        (&pf.w, i as u64 * kb, &v.router_w, 0, kb),
                        (&pf.mixed, i as u64 * nb, &ws.mixed, 0, nb),
                        (&pf.shared, i as u64 * nb, &v.sh, 0, nb),
                        (&pf.gate, i as u64 * 4, &v.sh_gate_logit, 0, 4),
                        // The ffn `HC_WRITE` moves to pass 2, so this token's
                        // inject partials have to survive the rest of the chunk.
                        (
                            &v.inject_slots[i],
                            0,
                            &ws.inject,
                            0,
                            ws.inject.size.min(160),
                        ),
                    ],
                )?;
            }
            self.swap_r_in(dev, i);
            if batch_cmd::recorded() >= MERGED_BATCH_CAP {
                self.flush()?;
                self.ensure_batch(dev)?;
            }
        }
        batch_cmd::record_barrier_to_host(gpu)?;
        self.submit_batch()?;
        self.flush()?;

        // ---- the bucket list: per distinct expert, its users -----------------
        let mut raw = vec![0u8; t * k * 4];
        self.venues[dev]
            .pf
            .as_ref()
            .expect("the merged prefill scratch")
            .ids
            .read(&mut raw)?;
        let mut pairs: Vec<(usize, usize, usize)> = Vec::with_capacity(t * k);
        for i in 0..t {
            for j in 0..k {
                let o = (i * k + j) * 4;
                let e = u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]) as usize;
                if e >= mc.n_expert {
                    bail!(
                        "router id {e} is outside the model's {} experts",
                        mc.n_expert
                    );
                }
                pairs.push((e, i, j));
            }
        }
        // Router order within a token, expert order across: the slot map keeps
        // each distinct expert's users contiguous, which is exactly the GEMM's
        // `O[u * rows + row]` output layout.
        pairs.sort_unstable();
        let mut slots = vec![0u32; t * k];
        let mut groups: Vec<(usize, u32, Vec<usize>)> = Vec::new();
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
                let g_end = (u + breccia_kernels::gemm::USERS_MAX).min(end);
                let base = slot;
                let mut users = Vec::with_capacity(g_end - u);
                for &(_, ti, j) in &pairs[u..g_end] {
                    slots[ti * k + j] = slot;
                    users.push(ti);
                    slot += 1;
                }
                groups.push((e, base, users));
                u = g_end;
            }
            at = end;
        }
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        self.venues[dev]
            .pf
            .as_ref()
            .expect("the merged prefill scratch")
            .slots
            .write(&slot_bytes)?;

        // ---- the expert GEMMs: three per (distinct expert, group of <= 8) -----
        self.timers.mark(EXPERTS);
        self.ensure_batch(dev)?;
        let t_gemm = std::time::Instant::now();
        // Streamed blocks walk the distinct experts in slot order; one submit per
        // STREAM_SLOTS experts bounds the host-visible staging (and is the ring
        // wait this pass reports).
        let mut expert_index = 0usize;
        let mut prev_expert: Option<usize> = None;
        for (e, base, users) in groups.iter() {
            let mu = users.len();
            // First group of a new distinct expert: stream it (once per chunk).
            let new_expert = Some(*e) != prev_expert;
            if streamed && new_expert {
                prev_expert = Some(*e);
                // Refill a slot; when the slots are exhausted, drain first so the
                // in-flight copies never see the next expert's bytes.
                if expert_index > 0 && expert_index.is_multiple_of(STREAM_SLOTS) {
                    let t0 = std::time::Instant::now();
                    self.submit_batch()?;
                    self.flush()?;
                    self.pf_stream_wait_s += t0.elapsed().as_secs_f64();
                    self.ensure_batch(dev)?;
                }
                self.fill_stream_slot(dev, b, *e, expert_index % STREAM_SLOTS)?;
                expert_index += 1;
            } else if !streamed {
                prev_expert = Some(*e);
            }
            {
                let v = &self.venues[dev];
                let pf = v.pf.as_ref().expect("the merged prefill scratch");
                let pipes = &v.pipes;
                let gate = pipes.gemm(gate_ty).expect("the gate multi-user GEMM");
                let up = pipes.gemm(up_ty).expect("the up multi-user GEMM");
                let down = pipes.gemm(down_ty).expect("the down multi-user GEMM");
                if streamed && new_expert && expert_index > 0 {
                    let slot = (expert_index - 1) % STREAM_SLOTS;
                    // slot -> device, three region groups, one barrier.
                    let off = slot as u64 * pf.slot_stride;
                    batch_cmd::record_copies(
                        gpu,
                        &[
                            (&pf.sf_gate, 0, &pf.stream.buf, off, pf.g_bytes),
                            (&pf.sf_up, 0, &pf.stream.buf, off + pf.g_bytes, pf.u_bytes),
                            (
                                &pf.sf_down,
                                0,
                                &pf.stream.buf,
                                off + pf.g_bytes + pf.u_bytes,
                                pf.d_bytes,
                            ),
                        ],
                    )?;
                }
                // Gather the group's activation rows (a run of consecutive tokens
                // is one copy), then one weights read at the expert's arena base.
                let mut rows: Vec<(&Buf, u64, &Buf, u64, u64)> = Vec::new();
                let mut a = 0usize;
                while a < mu {
                    let mut z = a + 1;
                    while z < mu && users[z] == users[z - 1] + 1 {
                        z += 1;
                    }
                    rows.push((
                        &pf.x,
                        (a * n_embd * 4) as u64,
                        &pf.mixed,
                        (users[a] * n_embd * 4) as u64,
                        ((z - a) * n_embd * 4) as u64,
                    ));
                    a = z;
                }
                batch_cmd::record_copies(gpu, &rows)?;
                // Resident: the arena holds all 512 experts, so the base is the
                // expert's own offset. Streamed: the slot *is* expert `e`, base 0.
                let (gbuf, ubuf, dbuf, gb, ub, db) = match pe {
                    Some(pe) => (
                        &pe.gate.buf,
                        &pe.up.buf,
                        &pe.down.buf,
                        (*e as u32) * wgate,
                        (*e as u32) * wup,
                        (*e as u32) * wdown,
                    ),
                    None => (&pf.sf_gate, &pf.sf_up, &pf.sf_down, 0u32, 0u32, 0u32),
                };
                // `gate` and `up` read the same gather and write pf.gg / pf.uu:
                // drop gate's trailing fence (§8z).
                batch_cmd::skip_next_barrier();
                gate.run_at(gbuf, &pf.x, &pf.gg, n_ff, n_embd, mu, gb)?;
                up.run_at(ubuf, &pf.x, &pf.uu, n_ff, n_embd, mu, ub)?;
                v.swiglu.run(&pf.gg, &pf.uu, &pf.act, mu * n_ff)?;
                down.run_at(dbuf, &pf.act, &pf.down, n_embd, n_ff, mu, db)?;
                batch_cmd::record_copy(
                    gpu,
                    &pf.scratch,
                    (*base as u64) * (n_embd * 4) as u64,
                    &pf.down,
                    0,
                    (mu * n_embd * 4) as u64,
                )?;
            }
            if batch_cmd::recorded() >= MERGED_BATCH_CAP {
                self.flush()?;
                self.ensure_batch(dev)?;
            }
        }
        self.submit_batch()?;
        self.flush()?;
        let gemm_s = t_gemm.elapsed().as_secs_f64();

        // ---- the fan-in: one dispatch, then the chunk's MoE result per token --
        self.ensure_batch(dev)?;
        let t_fan = std::time::Instant::now();
        {
            let v = &self.venues[dev];
            let pf = v.pf.as_ref().expect("the merged prefill scratch");
            v.moe_fanin.run(
                &pf.scratch,
                &pf.slots.buf,
                &pf.w,
                &pf.shared,
                &pf.gate,
                &pf.out,
                t,
                k,
                n_embd,
            )?;
            let rows: Vec<(&Buf, u64, &Buf, u64, u64)> = (0..t)
                .map(|i| {
                    (
                        &v.moe_out[i].buf,
                        0u64,
                        &pf.out,
                        (i * n_embd * 4) as u64,
                        (n_embd * 4) as u64,
                    )
                })
                .collect();
            batch_cmd::record_copies(gpu, &rows)?;
            self.dev_up[dev] += (t * n_embd * 4) as u64;
        }
        self.submit_batch()?;
        self.flush()?;
        let fanin_s = t_fan.elapsed().as_secs_f64();

        // ---- pass 2: the ffn HC_WRITE for every token, one submission --------
        self.timers.mark(HC_WRITE_F);
        self.ensure_batch(dev)?;
        for i in 0..t {
            self.swap_r_in(dev, i);
            self.swap_inject_in(dev, i);
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.hc.write(ws, &v.r, &v.moe_out[i].buf, &v.r)?;
            self.swap_inject_in(dev, i);
            self.swap_r_in(dev, i);
        }
        self.submit_batch()?;
        self.flush()?;

        // ---- the counters (§8v): per chunk, per block ------------------------
        let dispatches = self.dispatches - disp_before;
        let expert_bytes = m.layers[b].experts.expert_bytes() as u64;
        self.pf_gemm_s += gemm_s;
        self.pf_fanin_s += fanin_s;
        self.pf_merged_dispatches += dispatches;
        self.pf_merged_experts += groups.len();
        self.pf_arena_bytes += distinct as u64 * expert_bytes;
        self.pf_merged_slots += t * k;
        self.pf_distinct += distinct;
        self.pf_merged_bytes += distinct as u64 * expert_bytes;
        if std::env::var_os("BRECCIA_TRACE_MERGED").is_some() {
            let (sb, sf, sw) = (
                self.pf_stream_bytes,
                self.pf_stream_fill_s,
                self.pf_stream_wait_s,
            );
            eprintln!(
                "merged blk {b:>2} t={t} {} distinct={distinct} groups={} slots={} \
                 gemm={:.2} ms fanin={:.3} ms dispatches={dispatches} arena={:.1} MiB \
                 streamed(cum)={:.1} MiB fill={:.1} ms wait={:.1} ms",
                if streamed { "streamed" } else { "resident" },
                groups.len(),
                t * k,
                gemm_s * 1e3,
                fanin_s * 1e3,
                (distinct as u64 * expert_bytes) as f64 / (1u64 << 20) as f64,
                sb as f64 / (1u64 << 20) as f64,
                sf * 1e3,
                sw * 1e3
            );
        }
        Ok(())
    }

    /// Swap token `i`'s residual slice into the venue's single-token `r` (and
    /// back), so the per-token dense chain runs verbatim on a chunk.
    fn swap_r_in(&mut self, dev: usize, i: usize) {
        let v = &mut self.venues[dev];
        std::mem::swap(&mut v.r, &mut v.r_slots[i]);
    }

    /// Swap token `i`'s `inject` partials into the hc workspace (and back): the
    /// ffn `HC_WRITE` needs the values its own `HC_READ(ffn)` produced.
    fn swap_inject_in(&mut self, dev: usize, i: usize) {
        let v = &mut self.venues[dev];
        let Some(ws) = v.hc_ws.as_mut() else { return };
        std::mem::swap(&mut ws.inject, &mut v.inject_slots[i]);
    }

    /// Record the five readback copies for token `slot` of a chunk into the
    /// staging block region, plus the host-visibility barrier.
    fn record_lane_copies(&self, dev: usize, slot: usize) -> Result<()> {
        let mc = &self.model.cfg;
        let k = mc.n_expert_used;
        let gpu = self.model.devices[dev].gpu();
        let v = &self.venues[dev];
        let st = &self.stagings[dev];
        let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
        let base = crate::xfer::BLOCK_REGION * slot as u64;
        let offs = lane_offsets(mc.n_embd, k);
        // One region array per token (five copies, one barrier packet) instead of
        // five `record_copy`s: ~5 µs of host record time saved per token per block,
        // which is the whole `xfer-down` stage of a chunk.
        let stb = st.buffer();
        batch_cmd::record_copies(
            gpu,
            &[
                (stb, base + offs[0], &ws.mixed, 0, (mc.n_embd * 4) as u64),
                (stb, base + offs[1], &v.router_ids, 0, (k * 4) as u64),
                (stb, base + offs[2], &v.router_w, 0, (k * 4) as u64),
                (stb, base + offs[3], &v.sh, 0, (mc.n_embd * 4) as u64),
                (stb, base + offs[4], &v.sh_gate_logit, 0, 4),
            ],
        )?;
        batch_cmd::record_barrier_to_host(gpu)?;
        Ok(())
    }

    /// The last chunked pass's merged-MoE budget: `(distinct experts summed over
    /// blocks, expert bytes a once-per-expert device pass would move, tokens)`.
    pub fn merged_prefill_budget(&self) -> (usize, u64, usize) {
        (self.pf_distinct, self.pf_merged_bytes, self.pf_tokens)
    }

    /// Streamed-expert accounting for the last window: `(bytes copied out of the
    /// mmap, host seconds in those copies, host seconds waiting on the staging
    /// ring)`.
    pub fn merged_stream_detail(&self) -> (u64, f64, f64) {
        (
            self.pf_stream_bytes,
            self.pf_stream_fill_s,
            self.pf_stream_wait_s,
        )
    }

    /// The merged prefill pass's cost breakdown for the last measured window:
    /// `(expert GEMM seconds, fan-in seconds, dispatches recorded, expert groups,
    /// arena bytes the merged pass read, slots)`.
    pub fn merged_prefill_detail(&self) -> (f64, f64, u64, usize, u64, usize) {
        (
            self.pf_gemm_s,
            self.pf_fanin_s,
            self.pf_merged_dispatches,
            self.pf_merged_experts,
            self.pf_arena_bytes,
            self.pf_merged_slots,
        )
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
        self.census_split(dev)?;
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
    ///
    /// The head may live on a device other than the last block's (`--head-device`):
    /// the residual then crosses the link once, through the same
    /// [`Staging`] read/write pair the block-to-block hand-off uses
    /// ([`Session::submit_block`]), because that machinery already owns the
    /// synchronisation (its own submit + fence) and a second path would be a second
    /// place to get the ordering wrong.
    pub fn submit_head(&mut self) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        self.flush()?;
        self.timers.mark(FINAL_HC);
        let head_dev = m.head_device();
        debug_assert_eq!(
            head_dev,
            m.device_of(mc.n_layers - 1),
            "the head runs where block {} runs: the device-crossing hand-off was removed \
             (docs/phase-4/01-forward-pass.md §8aw)",
            mc.n_layers - 1
        );
        self.head_traffic0 = self.staging_traffic();
        self.ensure_batch(head_dev)?;
        {
            let v = &self.venues[head_dev];
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
                .ok_or_else(|| anyhow!("device {head_dev} has no logits buffer"))?;
            v.pipes.get(&m.head.key)?.run(&m.head.buf, &ws.mixed, out)?;
        }
        self.submit_batch()?;
        Ok(())
    }

    /// Wait for the head and read the logits row back.
    pub fn finish_head(&mut self) -> Result<Vec<f32>> {
        let m = self.model;
        let mc = &m.cfg;
        let head_dev = m.head_device();
        self.flush()?;
        let logits_buf = self.venues[head_dev]
            .logits
            .as_ref()
            .ok_or_else(|| anyhow!("device {head_dev} has no logits buffer"))?;
        let raw = self.stagings[head_dev]
            .read_batch(m.devices[head_dev].gpu(), &[(logits_buf, 0, mc.vocab * 4)])?;
        let (down1, up1) = self.staging_traffic();
        self.head_down += down1.saturating_sub(self.head_traffic0.0);
        self.head_up += up1.saturating_sub(self.head_traffic0.1);
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
        // The host hash advances the rolling context, so it happens exactly once
        // per token; a chunk hashes its whole prompt up front so the table rows can
        // be faulted in **once, in parallel** (see `ple_step_rows`).
        let rows = self
            .ple
            .as_mut()
            .ok_or_else(|| anyhow!("no PLE host state"))?
            .rows_for(self.cur_tok);
        self.ple_step_rows(b, &rows)
    }

    /// [`Session::ple_step`] with the table rows already hashed: the GPU half and
    /// the host block for one token.
    fn ple_step_rows(&mut self, b: usize, rows: &[u32]) -> Result<()> {
        let _tok = self.cur_tok;
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

        // 2. fault the rows in concurrently, then gather them on the GPU.
        //
        // The PLE table is 26.8 GiB of lazily-mapped IQ4_NL, and one token needs
        // 16 scattered 90-byte rows = 16 page faults. Serialised, those cost
        // ~28 ms per token here whenever the page cache has evicted them (the
        // expert lane streams ~800 MiB/token through the same NVMe), against
        // 0.65 ms once hot. llama.cpp prefetches the same rows for the same
        // reason (`llama_prefetch_rows`, `qwen4exp.cpp:1283-1288`).
        prefetch_rows(table, rows, m.ple_row_bytes());
        {
            let gpu = m.devices[dev].gpu();
            let v = &self.venues[dev];
            let staged = v.emb_ple.gather_staged(gpu, table, rows, &v.ple_emb)?;
            v.emb_ple.release(gpu, &staged);
            // `key` and `value` read the same gathered rows: drop key's fence.
            batch_cmd::skip_next_barrier();
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

    /// The **chunked** PLE phase (§8aj): one flush, one concatenated upload, `t`
    /// gathers-worth of batched GPU work, one readback, `t` host blocks, one
    /// writeback — instead of a host round trip per token.
    ///
    /// The per-token preamble costs ~9 submits (measured: 13.6 per block-chunk
    /// against 2 for the dense passes), and a submit+fence is a fixed 56 µs
    /// regardless of what it carries, so the unit to delete is the submit. The
    /// shape here is:
    ///
    /// ```text
    /// host   hash `t` tokens (ordered), pack `t*16` rows, prefetch them
    /// batch  upload the packed rows → 1 gather over `t*16` rows → per token
    ///        copy the token's row range into `ple_emb`, key GEMV, value GEMV
    /// read   R, key and value for every token in **one** `read_batch`
    /// host   the `t` blocks, in sequence (the rolling context and conv state)
    /// write  the `t` new residuals in **one** region-array writeback
    /// ```
    ///
    /// The copies exist because `Buf` has no offset field: the gather's output is
    /// one stack, and a projection can only read a buffer from 0. The key/value
    /// outputs are one buffer per token for the same reason, which also lets the
    /// readback address them as an offset list.
    fn ple_chunk(&mut self, b: usize, t: usize) -> Result<()> {
        if t == 0 {
            return Ok(());
        }
        let m = self.model;
        let mc = &m.cfg;
        let raw = m
            .ple
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} runs PLE but the model has no PLE tensors"))?;
        let table = m.ple_table()?;
        let row_bytes = m.ple_row_bytes();
        let n_head = mc.ple.n_head();
        let hc_dim = mc.hc_dim();
        let n_embd = mc.n_embd;
        let dev = m.device_of(b);
        let gpu = m.devices[dev].gpu();

        // ---- 1. the host hash, in sequence, then one prefetch and one pack ----
        //
        // The hash advances the rolling context, so it cannot be batched; the
        // table rows it produces can. One `prefetch_rows` over `t*16` rows keeps
        // the 90-byte-within-a-4-KiB-page reads of the 28.8 GiB table from
        // serialising per token (`docs/research/03`, §8z).
        let mut rows: Vec<u32> = Vec::with_capacity(t * n_head);
        {
            let ple = self
                .ple
                .as_mut()
                .ok_or_else(|| anyhow!("no PLE host state"))?;
            for &tok in self.pf_toks.iter().take(t) {
                rows.extend_from_slice(&ple.rows_for(tok));
            }
        }
        let t_prefetch = std::time::Instant::now();
        prefetch_rows(table, &rows, row_bytes);
        self.pf_ple_prefetch_s += t_prefetch.elapsed().as_secs_f64();
        let packed = EmbGather::pack_rows(table, row_bytes, &rows)?;

        // ---- 2. one batch: upload, gather, `2t` projections, `t` view copies --
        self.ensure_batch(dev)?;
        {
            let v = &self.venues[dev];
            let emb_chunk = v
                .ple_emb_chunk
                .as_ref()
                .ok_or_else(|| anyhow!("no PLE chunk scratch on device {dev}"))?;
            let packed_buf = v
                .ple_packed
                .as_ref()
                .ok_or_else(|| anyhow!("no PLE packed-row buffer on device {dev}"))?;
            let idx = v
                .ple_idx
                .as_ref()
                .ok_or_else(|| anyhow!("no PLE gather index list on device {dev}"))?;
            let key_slots = &v.ple_key_slots;
            let value_slots = &v.ple_value_slots;
            if key_slots.len() < t || value_slots.len() < t {
                bail!(
                    "the PLE key/value slots cover {} tokens, not {t}",
                    key_slots.len()
                );
            }
            // The upload rides the batch: a `Staging::write` here would cost a
            // submit + fence in front of the work that consumes it.
            self.stagings[dev].record_upload(gpu, packed_buf, 0, &packed)?;
            // `t * 16` rows of the chunk at once, token-major, stride
            // `ple.padded_cols()`: token `i`'s rows are `i*n_head .. (i+1)*n_head`.
            v.emb_ple.run(packed_buf, idx, emb_chunk, t * n_head)?;
            let view = (n_head * v.emb_ple.padded_cols() * 4) as u64;
            for i in 0..t {
                batch_cmd::record_copy(
                    gpu,
                    &v.ple_emb,
                    0,
                    emb_chunk,
                    i as u64 * view,
                    (n_embd * 4) as u64,
                )?;
                v.pipes
                    .get(&raw.key.key)?
                    .run(&raw.key.buf, &v.ple_emb, &key_slots[i])?;
                v.pipes
                    .get(&raw.value.key)?
                    .run(&raw.value.buf, &v.ple_emb, &value_slots[i])?;
            }
        }
        self.submit_batch()?;

        // ---- 3. one readback for the whole chunk ------------------------------
        //
        // `read_batch` waits for the batch above (same queue, one fence), so the
        // `t` tokens' R/key/value arrive in a single round trip.
        let parts = {
            let v = &self.venues[dev];
            let mut items: Vec<(&Buf, u64, usize)> = Vec::with_capacity(3 * t);
            for i in 0..t {
                items.push((&v.r_slots[i], 0, hc_dim * 4));
                items.push((&v.ple_key_slots[i], 0, hc_dim * 4));
                items.push((&v.ple_value_slots[i], 0, n_embd * 4));
            }
            self.stagings[dev].read_batch(gpu, &items)?
        };

        // ---- 4. the host blocks, in sequence, then one writeback --------------
        let mut out = vec![0f32; t * hc_dim];
        {
            let ple = self
                .ple
                .as_mut()
                .ok_or_else(|| anyhow!("no PLE host state"))?;
            for i in 0..t {
                let r_host = f32_from_bytes(&parts[3 * i]);
                let key = f32_from_bytes(&parts[3 * i + 1]);
                let value = f32_from_bytes(&parts[3 * i + 2]);
                ple.block(
                    &r_host,
                    &key,
                    &value,
                    &mut out[i * hc_dim..(i + 1) * hc_dim],
                )?;
            }
        }
        let bytes = f32_to_bytes(&out);
        {
            let v = &self.venues[dev];
            let items: Vec<(&Buf, u64, &[u8])> = (0..t)
                .map(|i| {
                    (
                        &v.r_slots[i],
                        0,
                        &bytes[i * hc_dim * 4..(i + 1) * hc_dim * 4],
                    )
                })
                .collect();
            self.stagings[dev].write_regions(gpu, &items)?;
        }
        self.cur_tok = self.pf_toks[t - 1];
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
        if self.prefill_chunk > 0 {
            // Chunked prompt pass (§8s): same tokens, same arithmetic, grouped
            // expert lane. The chunk sizes come from the planner, so a prompt does
            // not end in a tiny tail chunk (`prefill_plan`). `pos` is the outer
            // one — decode continues from it.
            for len in prefill_plan(prompt.len(), self.prefill_target) {
                logits = self.prefill(&prompt[pos..pos + len], pos)?;
                pos += len;
            }
        } else {
            for &t in prompt {
                logits = self.token(t, pos, None)?;
                pos += 1;
            }
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
        self.head_up = 0;
        self.head_down = 0;

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
        // Every submission must be finished before anything it references is
        // freed, and the ring's descriptor pools are shared by every session on
        // this thread — the last session to drop frees them.
        let gpus: Vec<&breccia_vk::ctx::Gpu> = m.devices.iter().map(|d| d.gpu()).collect();
        let _ = breccia_kernels::batch_cmd::wait_all(&gpus);
        if LIVE_SESSIONS.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1 {
            let _ = breccia_kernels::batch_cmd::shutdown();
        }
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
            if let Some(rc) = &v.r_chunk {
                gpu.destroy_buf(rc);
            }
            for b in v
                .r_slots
                .iter()
                .chain(v.inject_slots.iter())
                .chain(v.ple_key_slots.iter())
                .chain(v.ple_value_slots.iter())
            {
                gpu.destroy_buf(b);
            }
            for b in [&v.ple_emb_chunk, &v.ple_packed, &v.ple_idx]
                .into_iter()
                .flatten()
            {
                gpu.destroy_buf(b);
            }
            for mb in &v.moe_out {
                mb.release(gpu);
            }
            if let Some(pf) = &v.pf {
                pf.release(gpu);
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
    let threads = rows.len().min(8);
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

/// The **hard cap** a chunked session allocates for, from the `--prefill-chunk`
/// target: `1.5 x target`.
///
/// A prompt's prefill cost is `n_chunks * fixed + len * per_token`, and the fixed
/// part is not small — one chunk runs the whole grouped expert lane over all 48
/// blocks (`docs/research/07` §C1.3: 4-token tail = 658 ms/token against 88 for a
/// full chunk). So the number of chunks is the thing to minimise, and
/// `ceil(len / target)` leaves a tiny tail whenever `len` is a little over a
/// multiple of the target (`132 = 64 + 64 + 4`). The slack lets the planner use
/// `ceil(len / cap)` chunks instead — two chunks of 66 for that prompt — at the
/// cost of 50 % more chunk scratch (~100 KiB/token), which is noise beside the
/// weights.
pub fn prefill_cap(target: usize) -> usize {
    if target == 0 {
        0
    } else {
        target + target / 2
    }
}

/// The chunk sizes a prompt of `len` tokens is prefilled in, for a
/// `--prefill-chunk` target: the **fewest** chunks allowed by
/// [`prefill_cap`], split **evenly**.
///
/// Even splitting is what removes the tail: with `n = ceil(len / cap)` every chunk
/// is at least `len / n > cap * (n-1) / n >= cap/2`, so no chunk can be tiny. It
/// also never exceeds the cap, so the scratch allocated for
/// [`prefill_cap`] is enough.
pub fn prefill_plan(len: usize, target: usize) -> Vec<usize> {
    let cap = prefill_cap(target);
    if len == 0 || cap == 0 || target == 0 {
        return Vec::new();
    }
    // A/B arm and revert path (`docs/phase-4` §8aj): the pre-planner split, greedy
    // `target`-sized chunks with whatever tail is left.
    if std::env::var_os("BRECCIA_NO_PREFILL_PLAN").is_some() {
        let n = len.div_ceil(target);
        let mut v = vec![target; n];
        v[n - 1] = len - (n - 1) * target;
        return v;
    }
    let n = len.div_ceil(cap);
    let (base, rem) = (len / n, len % n);
    (0..n).map(|i| base + usize::from(i < rem)).collect()
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

/// **Logical** CPUs, capped at 32 — the lane's thread count, and it is worth 41 %
/// of the decode gate.
///
/// This used to be physical cores / 2 (8 on this 7950X), which measured 14.7
/// tok/s aggregate on the two-stream gate against **20.2-20.8 at 32 threads**
/// (lane 86 -> 41 ms/step, A/B/A in one window, `docs/phase-4` §8y): the lane mixes
/// ALU-bound dequantisation with memory stalls, so SMT genuinely helps it, and the
/// recorded gate of §8q was always run with `--threads 32`. Making 8 the default
/// was the whole "decode regression".
fn default_threads() -> usize {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return 8;
    };
    let logical = text
        .lines()
        .filter(|l| l.starts_with("processor"))
        .count()
        .max(1);
    logical.clamp(1, 32)
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

#[cfg(test)]
mod tests {
    /// The chunk planner is pure, and the number it removes is measured: a
    /// 132-token prompt at `--prefill-chunk 64` used to be `64 + 64 + 4`, where the
    /// 4-token tail ran at 658 ms/token against 88 (`docs/research/07` §C1.3). The
    /// plan must instead be the fewest chunks the cap allows, split evenly.
    #[test]
    fn the_plan_never_leaves_a_tiny_tail_chunk() {
        let cap = super::prefill_cap(64);
        assert_eq!(cap, 96);
        // The gate prompt and the two agent prefixes.
        assert_eq!(super::prefill_plan(132, 64), vec![66, 66]);
        assert_eq!(super::prefill_plan(300, 64), vec![75, 75, 75, 75]);
        assert_eq!(super::prefill_plan(64, 64), vec![64]);
        // A prompt under the cap is one chunk, at the target or not.
        assert_eq!(super::prefill_plan(1, 64), vec![1]);
        assert_eq!(super::prefill_plan(95, 64), vec![95]);
        // Just over the cap is two balanced chunks, not cap + a stub.
        assert_eq!(super::prefill_plan(97, 64), vec![49, 48]);
        // Every plan tiles the prompt, never exceeds the cap, and every chunk is
        // at least half the cap (so no chunk is "pure overhead").
        for len in [1usize, 2, 63, 64, 65, 96, 97, 132, 193, 300, 4096] {
            let plan = super::prefill_plan(len, 64);
            assert_eq!(plan.iter().sum::<usize>(), len, "len {len}: {plan:?}");
            assert!(
                plan.iter().all(|t| *t <= cap),
                "len {len}: {plan:?} exceeds the cap"
            );
            // A prompt longer than half a chunk must never be split into a stub:
            // every chunk is at least half the cap.
            if len > cap / 2 {
                assert!(
                    plan.iter().all(|t| *t >= cap / 2),
                    "len {len}: {plan:?} has a tiny chunk"
                );
            }
            assert_eq!(
                plan.len(),
                len.div_ceil(cap),
                "len {len}: {plan:?} is not the fewest chunks"
            );
        }
        // `0` stays token by token, which is the caller's fallback.
        assert!(super::prefill_plan(132, 0).is_empty());
        assert!(super::prefill_plan(0, 64).is_empty());
    }
}
