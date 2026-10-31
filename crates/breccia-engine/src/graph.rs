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
use breccia_kernels::cpu::{self, lane_stats_on, ExpertPool, LaneStats};
use breccia_kernels::ops::{
    f32_from_bytes, f32_to_bytes, EmbGather, HcMixer, HcWeights, HcWorkspace, Router, SwiGlu,
};
use breccia_kernels::{
    Gdn, GdnProjections, GdnState, GdnTap, GdnWeights, GdnWorkspace, Proj, Qsa, QsaProjections,
    QsaState, QsaWeights, QsaWorkspace, QuantType,
};

use crate::config::LayerKind;
use crate::model::{HcW, Model, PipelineSet};
use crate::ple::{PleHost, PleState};
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

/// `BRECCIA_NO_DEFER_PLACED=1`: restore the one-fence-wait-per-block shape, i.e. wait
/// for a **placed** block's segment even though nothing on the host reads it (§8bh).
///
/// The deferral is what lets the host record block *b+1* while block *b* still runs:
/// a placed block's `HC_WRITE` is ordered against the next submission by the queue
/// itself, so the wait is pure host latency (research 08: 24.6 us of the 33.2 us a
/// waited submit costs). Cached — read once per session.
fn defer_placed_waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BRECCIA_NO_DEFER_PLACED").is_none())
}

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

type ShipCopy<'a> = (&'a Buf, &'a Buf, u64, u64, u64);

/// `BRECCIA_PF_MICRO=2` tick state (see [`Session::prefill_micro`]): one half-step of
/// the two-micro-batch pipeline while its expert half runs on the helper card.
struct MicroTick {
    n: usize,
    th: usize,
    pos0: usize,
    /// The block card's in-flight segment (B(n-1) then A(n+1)), waited for after the
    /// expert half returns.
    seg: Option<breccia_kernels::batch_cmd::Segment>,
    start: std::time::Instant,
    launch_end: std::time::Instant,
    bound_s: f64,
    launch_s: f64,
}

/// The MoE result of the previous half-step, on its way back to the block card's
/// `pf.out`: the write is deferred to the tick boundary, when that card is idle.
enum MicroUp {
    Bytes(Vec<u8>),
    Imported(u64),
}

/// What [`Session::merged_plan`] decides for one block of the merged prefill.
struct MergedPlan<'m> {
    pe: Option<&'m crate::model::PlacedExperts>,
    gate_ty: QuantType,
    up_ty: QuantType,
    down_ty: QuantType,
    streamed: bool,
    xdev: Option<usize>,
    shared_chunk: bool,
    hc_chunk: bool,
}

/// One row of [`Session::gpu_time_table`]: a stage's GPU execution time for one
/// decode step.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GpuStageRow {
    /// Stage name (`hc-read`, `qkv`, `moe-gate`, …).
    pub stage: &'static str,
    /// `(µs of GPU time per token, intervals per token)` — one entry per device.
    pub per_device: Vec<(f64, u32)>,
    /// Summed over the devices, µs of GPU time per token.
    pub total_us: f64,
}

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
    /// `[T, n_ff_shared]` gate, up and SwiGLU of the chunk-level shared expert
    /// (`BRECCIA_PF_SHCHUNK`).
    sh_g: Buf,
    sh_u: Buf,
    sh_a: Buf,
    /// `[T, n_expert]` router logits of the batched pass-1 chain (`BRECCIA_PF_HCCHUNK`).
    logits: Buf,
    /// **Streamed** (non-resident) experts: `STREAM_SLOTS` host-visible slots, each
    /// laid out `[gate | up | down]`, so one host memcpy per distinct expert
    /// replaces the lane's per-token dequantisation (§8x). The slot is copied to
    /// the device inside the batch; eight experts ride one submit.
    stream: Mailbox,
    /// One expert's three blobs on the device, for the streamed path.
    sf_gate: Buf,
    sf_up: Buf,
    sf_down: Buf,
    /// Gate then up, exact blob sizes back to back: the fused pair GEMM's weights.
    sf_gu: Buf,
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
                &self.sh_g,
                &self.sh_u,
                &self.sh_a,
                &self.logits,
                &self.stream.buf,
                &self.sf_gate,
                &self.sf_up,
                &self.sf_down,
                &self.sf_gu,
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
            &self.sh_g,
            &self.sh_u,
            &self.sh_a,
            &self.logits,
            &self.sf_gate,
            &self.sf_up,
            &self.sf_down,
            &self.sf_gu,
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
    /// The merged prefill pass's chunk scratch on this device (`Some` only when
    /// `--prefill-merged` is on).
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
    /// over the chunk (§8s). `0` = token by token. A layer-split placement runs
    /// each chunk on every device in block order (host-staged handoff).
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

// SAFETY: concurrent prefill moves each Session into exactly one scoped thread
// (`iter_mut`), so its raw Vulkan/mapped pointers are never touched from two
// threads at once. Shared references are not sent.
unsafe impl Send for Session<'_> {}

/// Rollback snapshots for speculative verify ([`Session::spec_enable`]).
///
/// Two modes, fixed at `spec_enable`. **Replay** (default): the verify is the
/// merged-prefill chunk (batched GEMMs, chunked mixers, one T-row head GEMM).
/// Slot 0 holds the state before the chunk and each layer taps what its mixer
/// consumed (`gdn_tap`, `idx_k_tap`); `rollback_to(r)` restores slot 0 and re-runs
/// just the state-update kernels over the first `r + 1` rows. **Rows**
/// (`BRECCIA_SPEC_VERIFY_ROWS=1`, or a model the chunk path cannot take): the old
/// per-row decode-style mixers with one slot per row; the A/B reference.
///
/// Slot `s` is the recurrent state after `s` rows of the chunk being verified.
/// The chunk's last row is the live state, so a T-row verify fills slots
/// `1..T-1`; slot 0 (before the chunk) is only read by the bench's base restore.
/// QSA K/V needs no slot: attention masks `cell > pos` and the refed row at that
/// position overwrites its own cell. The indexer's raw `idx_tail` does: a rejected
/// row at block slot 0 clobbers a tail the refed completing cell still pools.
/// `idx_pooled` entries of rejected rows sit past `n_blocks` and are recomputed.
struct SpecSnap {
    /// `[slot * n_layers + b]`
    gdn: Vec<Option<GdnState>>,
    tail: Vec<Option<Buf>>,
    /// PLE history after `s` rows (index 0 unused); empty for a model without PLE.
    ple_rows: Vec<Vec<f32>>,
    /// PLE window before the chunk; rollback replays the accepted prefix on it.
    ple_pre: Option<PleState>,
    base_ple: Option<PleState>,
    base_pos: usize,
    base_tok: u32,
    toks: Vec<u32>,
    pos0: usize,
    max_t: usize,
    /// Set only inside `verify`: the per-row hooks in `prefill_block*` check it.
    active: bool,
    /// Replay mode (see above); the fields below exist only then.
    replay: bool,
    /// Per layer: the GDN chunk's conv input / gate / beta rows.
    gdn_tap: Vec<Option<GdnTap>>,
    /// Per layer: the QSA chunk's raw indexer keys `[max_t, IDX_DIM]`.
    idx_k_tap: Vec<Option<Buf>>,
    /// `[max_t, vocab]` head GEMM output.
    head_logits: Option<Buf>,
    /// Final-HC workspaces for `T = 2..=max_t` (row maps are per `T`).
    head_ws: Vec<HcWorkspace>,
}

/// A loaded, ready-to-run session: pipelines, state, and the host expert pool.
pub struct Session<'m> {
    pub model: &'m Model,
    venues: Vec<Venue<'m>>,
    /// One staging buffer per device — kept out of [`Venue`] so a transfer can
    /// borrow it mutably while reading the device's own buffers.
    stagings: Vec<Staging>,
    /// §11j item 5: the dual-card ship's shared host import (default on, `BRECCIA_SHIP_IMPORT=0` opts out),
    /// made on the first ship; `ship_imp_off` once it is off or failed.
    ship_imp: Option<crate::xfer::ShipImport>,
    ship_imp_off: bool,
    gdn_state: Vec<Option<GdnState>>,
    qsa_state: Vec<Option<QsaState>>,
    /// Speculative-verify rollback slots; `None` (the default) leaves every path unchanged.
    spec: Option<Box<SpecSnap>>,
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
    /// `BRECCIA_DUMP_STATE` snapshots written so far (`dump_state`).
    dumps: usize,
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
    /// `BRECCIA_LANE_STATS=1`: seconds waiting to acquire the shared expert
    /// pool, and per device the seconds and count of batch submits.
    pub lock_wait_s: f64,
    pub dev_submit_s: Vec<f64>,
    pub dev_submit_n: Vec<u64>,
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
    pub(crate) prefill_target: usize,
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
    /// Merged pass 1 only: the prelude leaves the shared expert's gate/up/SwiGLU/
    /// down out, and the pass runs them once over the chunk (`BRECCIA_PF_SHCHUNK`).
    shared_deferred: bool,
    /// Merged pass 1 with `BRECCIA_PF_QSACHUNK` / `BRECCIA_PF_GDNCHUNK`: T-token
    /// mixer workspaces (made on first use) and, while set, the chunk's `pos0`:
    /// the prelude's mixer then copies its row of `pf.out` instead of running.
    /// The chunk workspaces below are per device (indexed by `device_of(b)`):
    /// a layer-split session runs its chunk on every card that owns blocks.
    qsa_chunk_ws: Vec<Option<QsaWorkspace>>,
    gdn_chunk_ws: Vec<Option<breccia_kernels::gdn::GdnWorkspace>>,
    /// `BRECCIA_PF_HCCHUNK`: the T-token HC workspaces over `r_chunk`, one per chunk
    /// size (the batched GEMV row maps are built for an exact `T`).
    hc_chunk_ws: Vec<Vec<HcWorkspace>>,
    /// `BRECCIA_PF_MICRO=2`: when `Some`, [`Session::prefill_block_merged`] runs only
    /// the expert half of the block (the block card's dense halves are recorded by
    /// [`Session::micro_launch`]).
    micro: Option<MicroTick>,
    micro_up: Option<MicroUp>,
    /// `BRECCIA_PF_MICRO_OVL`: the previous half's MoE result copy onto the block card, in
    /// flight on its one-shot buffer until the next readback / PLE / tick end.
    micro_upw: Option<crate::xfer::PendingCopy>,
    /// A micro tick's `pf.shared` ship, deferred to fan-in: its (imported offset, bytes).
    micro_sh: Option<(u64, u64)>,
    /// Where the MoE result sits in the ship import (past the ship regions) when overlapped.
    ship_res_off: u64,
    /// Per micro-batch half: its residual chunk and its hc workspaces, swapped into
    /// `r_chunk` / `hc_chunk_ws` while that half's dense steps are recorded.
    micro_rc: [Option<Buf>; 2],
    micro_hcws: [Vec<HcWorkspace>; 2],
    /// Wait helpers see only this device while the helper card's expert half runs
    /// beside the block card's submitted segment.
    wait_scope: Option<usize>,
    /// Per device, the [`PF_FLAGS`] this session runs ([`pf_resolve`]); public so
    /// `breccia profile --prefill` can A/B them in one process.
    pub pf: Vec<[bool; 5]>,
    mixer_from_chunk: Option<usize>,
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
    /// Per device, the probe's cumulative per-stage GPU nanoseconds **at the start
    /// of the measured window** ([`Session::reset_counters`]). The table is the
    /// difference, so it covers the decode window and not the prompt.
    probe_base: Vec<[u64; batch_cmd::probe::MAX]>,
    /// The same snapshot for the interval **counts** (they must be differenced with
    /// the same base as the nanoseconds, or the table's `n` is a process total).
    probe_base_n: Vec<[u32; batch_cmd::probe::MAX]>,
    /// The probe's per-device totals captured on the decoding thread
    /// ([`Session::capture_probe`]): the accumulators are thread-local, so after a
    /// concurrent decode thread exits the reading thread would otherwise see none.
    probe_cap: Option<Vec<([u64; batch_cmd::probe::MAX], [u32; batch_cmd::probe::MAX])>>,
    /// Latest `VK_EXT_calibrated_timestamps` sample per device (one per second of
    /// the probe run) — the anchor that puts query ticks on the host clock.
    anchor: Vec<Option<breccia_vk::ctx::Calibration>>,
    /// When the anchors above were taken (`None` = never).
    anchor_at: Option<std::time::Instant>,
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
    /// In-batch readbacks (`--inbatch-read`) whose staging region was read **without**
    /// having waited for a submission of this session first. That is the ordering hole
    /// `docs/phase-4/01-forward-pass.md` §8ba instruments: `Staging::read_block` is a
    /// bare slice of the mapping, so it is only correct if the copies that fill it have
    /// been fenced. Must be 0 on every run.
    pub readbacks_unfenced: u64,
    /// Segments whose fence wait was **deferred** because nothing on the host reads
    /// their result (§8bh): a placed block's `HC_WRITE` rides the next submission, so
    /// `flush()`/`wait_all` (and the ring's back-pressure) are what collect them.
    pub deferred_waits: u64,
    /// Indices of the logits row that still held the poison after the head ran, at the
    /// first check (`BRECCIA_FAULT_POISON_LOGITS`). `Some(0)` = the head wrote every
    /// entry; `Some(n > 0)` = `n` entries were never written.
    pub logits_unwritten: Option<usize>,
    /// The segment the block that is currently being finished submitted, so
    /// `finish_block` can prove it waited for **that** one and not a neighbour's.
    block_seg: Option<breccia_kernels::batch_cmd::Segment>,
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
            let chunk_slots = if cfg.prefill_chunk > 0 {
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
                let buf = alloc(mc.vocab as u64 * 4, "logits")?;
                // `BRECCIA_FAULT_POISON_LOGITS=1`: fill the row with a sentinel before
                // the first forward, so a head that does not write every one of the
                // `vocab` entries leaves the sentinel behind and `finish_head` says so.
                // A never-written row is the one failure that is invisible to every
                // ordering/state test and shows up exactly as "step 0 is wrong, the rest
                // is fine, and the same way every time in that process"
                // (docs/phase-4/01-forward-pass.md §8bb).
                if std::env::var_os("BRECCIA_FAULT_POISON_LOGITS").is_some() {
                    let pat = Self::logits_sentinel().to_bits();
                    let fill: Vec<u32> = vec![pat; mc.vocab];
                    breccia_kernels::gemv::upload_into(
                        gpu,
                        &buf,
                        &breccia_kernels::ops::u32_to_bytes(&fill),
                    )
                    .context("poisoning the logits row")?;
                }
                Some(buf)
            } else {
                None
            };
            // …and the five buffers the lane's readback consumes, because a buffer that
            // is *never written* on the first token is the same failure with a different
            // address (`docs/phase-4/01-forward-pass.md` §8bb).
            let poisoned = hc_ws.as_ref().map(|ws| &ws.mixed).into_iter().chain([
                &router_ids,
                &router_w,
                &sh,
                &sh_gate_logit,
            ]);
            for buf in poisoned {
                if std::env::var_os("BRECCIA_FAULT_POISON_LOGITS").is_some() {
                    let pat = Self::logits_sentinel().to_bits();
                    let fill: Vec<u32> = vec![pat; (buf.size / 4) as usize];
                    breccia_kernels::gemv::upload_into(
                        gpu,
                        buf,
                        &breccia_kernels::ops::u32_to_bytes(&fill),
                    )
                    .context("poisoning the lane scratch")?;
                }
            }
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
            let chunk = if cfg.prefill_chunk > 0 {
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
                        // `hc * chunks` floats: 10 chunk partials in the incumbent
                        // shape, one per workgroup (n_embd/2) when the `up`+gated
                        // fusion writes them (§8az). Sized from the mixer, never a
                        // constant.
                        let inject_bytes = (mc.hc * hc.inject_chunks() * 4) as u64;
                        injects.push(alloc(
                            inject_bytes.max(hc_inject_bytes()),
                            &format!("prefill inject slot {t}"),
                        )?);
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
                let nsb = mc.n_ff_shared as u64 * 4;
                let users = pf_group_users() as u64;
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
                    sh_g: alloc(chunk as u64 * nsb, "prefill chunk shared gate")?,
                    sh_u: alloc(chunk as u64 * nsb, "prefill chunk shared up")?,
                    sh_a: alloc(chunk as u64 * nsb, "prefill chunk shared act")?,
                    logits: alloc(chunk as u64 * mc.n_expert as u64 * 4, "prefill chunk router logits")?,
                    // Two halves, so the host fills one while the GPU copies the other.
                    stream: Mailbox::new(gpu, 2 * STREAM_SLOTS as u64 * slot_stride)?,
                    sf_gate: alloc(gb.max(16), "streamed expert gate")?,
                    sf_up: alloc(ub.max(16), "streamed expert up")?,
                    sf_down: alloc(db.max(16), "streamed expert down")?,
                    sf_gu: alloc((gb + ub).max(16), "streamed expert gate+up")?,
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
            ship_imp: None,
            ship_imp_off: std::env::var("BRECCIA_SHIP_IMPORT").as_deref() == Ok("0"),
            gdn_state,
            qsa_state,
            spec: None,
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
            dumps: 0,
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
            prefill_chunk: prefill_cap(cfg.prefill_chunk),
            prefill_target: cfg.prefill_chunk,
            prefill_merged: cfg.prefill_merged,
            pf_ids: Vec::new(),
            pf_weights: Vec::new(),
            pf_mixed: Vec::new(),
            pf_toks: Vec::new(),
            pf_gemm_s: 0.0,
            pf_fanin_s: 0.0,
            pf_stream_bytes: 0,
            pf_stream_fill_s: 0.0,
            pf_stream_wait_s: 0.0,
            shared_deferred: false,
            qsa_chunk_ws: (0..model.devices.len()).map(|_| None).collect(),
            gdn_chunk_ws: (0..model.devices.len()).map(|_| None).collect(),
            hc_chunk_ws: (0..model.devices.len()).map(|_| Vec::new()).collect(),
            micro: None,
            micro_up: None,
            micro_upw: None,
            micro_sh: None,
            ship_res_off: 0,
            micro_rc: [None, None],
            micro_hcws: [Vec::new(), Vec::new()],
            wait_scope: None,
            pf: pf_resolve(model),
            mixer_from_chunk: None,
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
                // The legacy per-batch probe shares **one pool per device across all
                // ring slots**, so its read can see another slot's in-flight query —
                // the recorded `A fence or query has not yet completed` abort (§8am).
                // The per-slot stage probe below replaces it; when that one is on, the
                // legacy pools are not even allocated.
                let legacy = (cfg.probe || std::env::var_os("BRECCIA_GPU_TIME").is_some())
                    && !batch_cmd::probe::enabled();
                let mut v = Vec::with_capacity(model.devices.len());
                for i in 0..model.devices.len() {
                    v.push(if legacy {
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
            probe_batches: if (cfg.probe || std::env::var_os("BRECCIA_GPU_TIME").is_some())
                && !batch_cmd::probe::enabled()
            {
                Some(0)
            } else {
                None
            },
            probe_base: vec![[0u64; batch_cmd::probe::MAX]; model.devices.len()],
            probe_base_n: vec![[0u32; batch_cmd::probe::MAX]; model.devices.len()],
            probe_cap: None,
            anchor: vec![None; model.devices.len()],
            anchor_at: None,
            gpu_busy_s: vec![0.0; model.devices.len()],
            dev_wait_s: vec![0.0; model.devices.len()],
            lock_wait_s: 0.0,
            dev_submit_s: vec![0.0; model.devices.len()],
            dev_submit_n: vec![0; model.devices.len()],
            dev_down: vec![0; model.devices.len()],
            dev_up: vec![0; model.devices.len()],
            placed_blocks: model.placed.len(),
            host_blocks: model.cfg.n_layers - model.placed.len(),
            head_up: 0,
            head_down: 0,
            readbacks_unfenced: 0,
            deferred_waits: 0,
            logits_unwritten: None,
            block_seg: None,
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
        if let Some(d) = self.wait_scope {
            return vec![self.model.devices[d].gpu()];
        }
        self.model.devices.iter().map(|d| d.gpu()).collect()
    }

    /// A GPU-time probe stage boundary on `dev` (`BRECCIA_GPU_TIME`; a cached-bool
    /// no-op otherwise).
    fn probe_mark(&self, dev: usize, stage: u8) {
        batch_cmd::probe::enter(self.model.devices[dev].gpu(), stage);
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
            let t_sub = lane_stats_on().then(std::time::Instant::now);
            let seg = batch_cmd::end_submit(gpu, None)
                .map_err(|e| anyhow!("closing the dispatch batch on device {d}: {e}"))?;
            self.note_submit(d, t_sub);
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
    fn note_submit(&mut self, d: usize, t: Option<std::time::Instant>) {
        if let Some(t) = t {
            self.dev_submit_s[d] += t.elapsed().as_secs_f64();
            self.dev_submit_n[d] += 1;
        }
    }

    /// Zero the `BRECCIA_LANE_STATS` counters, the shared pool's included.
    pub fn reset_lane_stats(&mut self) {
        self.lock_wait_s = 0.0;
        self.dev_submit_s.iter_mut().for_each(|x| *x = 0.0);
        self.dev_submit_n.iter_mut().for_each(|x| *x = 0);
        if let Ok(mut p) = self.pool.lock() {
            p.lane_stats = LaneStats::default();
        }
    }

    /// The Phase 0 counters since the last [`Session::reset_lane_stats`], or
    /// `None` when `BRECCIA_LANE_STATS` is off.
    pub fn lane_report(&self) -> Option<crate::LaneReport> {
        if !lane_stats_on() {
            return None;
        }
        let st: LaneStats = self.pool.lock().map(|p| p.lane_stats).unwrap_or_default();
        Some(crate::LaneReport {
            phases: st.phases,
            phase_wall_s: st.wall_s,
            phase_max_busy_s: st.max_busy_s,
            phase_mean_busy_s: st.mean_busy_s,
            sync_share: st.sync_share(),
            lock_wait_s: self.lock_wait_s,
            submit_s: self.dev_submit_s.clone(),
            submit_n: self.dev_submit_n.clone(),
        })
    }

    fn submit_batch(&mut self) -> Result<()> {
        let Some(d) = self.pending_batch.take() else {
            return Ok(());
        };
        let gpu = self.model.devices[d].gpu();
        let rec = batch_cmd::recorded();
        self.timers.count_batch(rec);
        self.probe_stamp(d, true)?;
        let t_sub = lane_stats_on().then(std::time::Instant::now);
        let seg = batch_cmd::end_submit(gpu, None)
            .map_err(|e| anyhow!("submitting the block on device {d}: {e}"))?;
        self.note_submit(d, t_sub);
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
    /// Whether the dual-card ship `dev` -> `x` goes through the shared host import.
    /// Made on first use, sized for the largest merged-prefill scratch of either
    /// card; any failure logs once and leaves the staged path on.
    /// The imported-memory ship of a chunk's helper inputs as `(src, dst, src off, dst
    /// off, bytes)` lists: block card -> host (`down`) and host -> helper (`up`).
    fn ship_lists(&self, dev: usize, x: usize, t: usize, k: usize, defer_shared: bool) -> (Vec<ShipCopy<'_>>, Vec<ShipCopy<'_>>) {
        let n_embd = self.model.cfg.n_embd;
        let nb = (n_embd * 4) as u64;
        let lens = [t as u64 * nb, (t * k * 4) as u64, t as u64 * nb, (t * 4) as u64];
        let pf = self.venues[dev].pf.as_ref().expect("the merged prefill scratch");
        let xpf = self.venues[x].pf.as_ref().expect("checked above");
        let imp = self.ship_imp.as_ref().expect("ship_ready");
        let (hd, hx) = (imp.get(dev), imp.get(x));
        let (mut down, mut up, mut at) = (Vec::new(), Vec::new(), 0u64);
        for (i, ((src, dst), n)) in [&pf.mixed, &pf.w, &pf.shared, &pf.gate]
            .into_iter()
            .zip([&xpf.mixed, &xpf.w, &xpf.shared, &xpf.gate])
            .zip(lens)
            .enumerate()
        {
            // A deferred `pf.shared` goes after the routed GEMMs, same offset.
            if !(defer_shared && i == 2) {
                down.push((src, hd, 0, at, n));
                up.push((hx, dst, at, 0, n));
            }
            at += n;
        }
        (down, up)
    }

    fn ship_ready(&mut self, dev: usize, x: usize) -> bool {
        if self.ship_imp_off {
            return false;
        }
        let m = self.model;
        let r = (|| -> Result<()> {
            if self.ship_imp.is_none() {
                let ship = [dev, x]
                    .iter()
                    .filter_map(|&d| self.venues[d].pf.as_ref())
                    .map(|pf| (pf.mixed.size + pf.w.size + pf.shared.size + pf.gate.size).max(pf.out.size))
                    .max()
                    .unwrap_or(0);
                // The overlapped micro-batch tick keeps the MoE result past the ship regions.
                self.ship_res_off = ship.div_ceil(4096) * 4096;
                let cap = self.ship_res_off + ship;
                self.ship_imp = Some(crate::xfer::ShipImport::new(cap, m.devices.len())?);
            }
            let imp = self.ship_imp.as_mut().expect("just made");
            imp.ensure(dev, m.devices[dev].gpu())?;
            imp.ensure(x, m.devices[x].gpu())
        })();
        if let Err(e) = r {
            eprintln!("breccia: ship import off, staged path kept: {e:#}");
            self.ship_imp_off = true;
            return false;
        }
        true
    }

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

    /// Submit the open batch on `dev` **without waiting** and open the next ring
    /// slot, so the host records while the GPU runs (prefill, `BRECCIA_PF_OVERLAP=1`).
    /// Returns the submitted segment for a caller that must later wait on it.
    fn reopen_batch(&mut self, dev: usize) -> Result<Option<breccia_kernels::batch_cmd::Segment>> {
        self.submit_batch()?;
        let seg = self.pending_seg.take();
        self.pending_dev = None;
        batch_cmd::begin(self.model.devices[dev].gpu())
            .map_err(|e| anyhow!("reopening a dispatch batch on device {dev}: {e}"))?;
        self.timers.open_batch();
        self.pending_batch = Some(dev);
        self.probe_stamp(dev, false)?;
        Ok(seg)
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
    ///
    /// With the per-slot **stage** probe on (`BRECCIA_GPU_TIME`), the busy time is
    /// the sum of the stage intervals minus the window's snapshot: the same thing
    /// the legacy per-batch probe measured, without the shared-pool race that made
    /// that one abort (§8am).
    pub fn gpu_busy_s(&mut self) -> Result<Vec<f64>> {
        if batch_cmd::probe::enabled() {
            let mut out = Vec::new();
            for (i, gpu) in self.device_gpus().into_iter().enumerate() {
                let (ns, _) = self.probe_totals(i, gpu);
                let base = self
                    .probe_base
                    .get(i)
                    .copied()
                    .unwrap_or([0; batch_cmd::probe::MAX]);
                let sum: u64 = (0..batch_cmd::probe::MAX)
                    .map(|s| ns[s].saturating_sub(base[s]))
                    .sum();
                out.push(sum as f64 / 1e9);
            }
            return Ok(out);
        }
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
        self.reset_lane_stats();
        self.dev_wait_s.iter_mut().for_each(|x| *x = 0.0);
        self.dev_down.iter_mut().for_each(|x| *x = 0);
        self.dev_up.iter_mut().for_each(|x| *x = 0);
        self.head_up = 0;
        self.head_down = 0;
        self.timers.clear();
        self.snapshot_probe();
        self.reset_gpu_busy()
    }

    /// Remember the GPU-time probe's cumulative per-stage totals, so the table
    /// covers the measured decode window and not the prompt.
    pub fn snapshot_probe(&mut self) {
        self.probe_cap = None;
        if !batch_cmd::probe::enabled() {
            return;
        }
        let mut snap: Vec<([u64; batch_cmd::probe::MAX], [u32; batch_cmd::probe::MAX])> =
            Vec::new();
        for gpu in self.device_gpus() {
            let (ns, n) = batch_cmd::probe::totals(gpu);
            snap.push((ns, n));
        }
        for (i, (ns, n)) in snap.into_iter().enumerate() {
            if let Some(slot) = self.probe_base.get_mut(i) {
                *slot = ns;
            }
            if let Some(slot) = self.probe_base_n.get_mut(i) {
                *slot = n;
            }
        }
    }

    /// Keep this thread's probe totals for readers on another thread (call on the
    /// decoding thread before it exits; pair with [`Session::snapshot_probe`] at its
    /// start so the base is the same thread's).
    pub fn capture_probe(&mut self) {
        if batch_cmd::probe::enabled() {
            self.probe_cap = Some(
                self.device_gpus()
                    .into_iter()
                    .map(batch_cmd::probe::totals)
                    .collect(),
            );
        }
    }

    fn probe_totals(
        &self,
        i: usize,
        gpu: &breccia_vk::ctx::Gpu,
    ) -> ([u64; batch_cmd::probe::MAX], [u32; batch_cmd::probe::MAX]) {
        match &self.probe_cap {
            Some(c) => c[i],
            None => batch_cmd::probe::totals(gpu),
        }
    }

    /// The GPU-time table for the measured window: per stage, **µs of GPU time per
    /// token**, one column per device, plus the row totals.
    ///
    /// `None` when the probe is off. The intervals are the GPU timeline between two
    /// stage boundaries inside one batched submission (`vkCmdWriteTimestamp2`,
    /// `COMPUTE_SHADER`), read once after the slot's fence — so a row is "the GPU
    /// was executing this stage's dispatches and their barriers", not a host price.
    /// The residual against the host's fence wait is the caller's subtraction:
    /// `wait - sum` is launch/serialisation (the submit round trip, the command-
    /// buffer prologue, and any time the GPU sat idle waiting for the host to
    /// record).
    pub fn gpu_time_table(&self) -> Option<Vec<GpuStageRow>> {
        if !batch_cmd::probe::enabled() {
            return None;
        }
        let tokens = self.forwards.max(1) as f64;
        let gpus = self.device_gpus();
        let mut rows: Vec<GpuStageRow> = Vec::new();
        for (stage, name) in batch_cmd::probe::NAMES.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let mut per: Vec<(f64, u32)> = Vec::with_capacity(gpus.len());
            let mut total = 0.0;
            for (i, gpu) in gpus.iter().enumerate() {
                let (ns, cnt) = self.probe_totals(i, gpu);
                let base = self
                    .probe_base
                    .get(i)
                    .copied()
                    .unwrap_or([0; batch_cmd::probe::MAX]);
                let base_n = self
                    .probe_base_n
                    .get(i)
                    .copied()
                    .unwrap_or([0; batch_cmd::probe::MAX]);
                let d = ns[stage].saturating_sub(base[stage]);
                let dn = cnt[stage].saturating_sub(base_n[stage]);
                let us = d as f64 / 1e3 / tokens;
                total += us;
                per.push((us, (dn as f64 / tokens).round() as u32));
            }
            // Drop the columns that nothing stamps, so the table is the graph's
            // actual stage list and not a 29-row enum.
            if total > 0.0 || per.iter().any(|(_, n)| *n > 0) {
                rows.push(GpuStageRow {
                    stage: name,
                    per_device: per,
                    total_us: total,
                });
            }
        }
        Some(rows)
    }

    /// `wait-gpu` per token of the measured window, in µs — the number the GPU-time
    /// sum is the residual against.
    pub fn wait_gpu_us_per_token(&self) -> f64 {
        self.timers.ns[WAIT_GPU] as f64 / 1e3 / self.forwards.max(1) as f64
    }

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
        // The calibrated anchor: one `VK_EXT_calibrated_timestamps` sample per
        // device per second (the recipe's cadence, `docs/research/07` §A6 step 2).
        // It is what lets the table's GPU time be compared against the host's
        // fence wait on one timeline; the unit trap is A3's (calibrated values are
        // **ns**, query values are **ticks**).
        if batch_cmd::probe::enabled()
            && self
                .anchor_at
                .map(|t| t.elapsed().as_secs_f64() >= 1.0)
                .unwrap_or(true)
        {
            let mut cal: Vec<Option<breccia_vk::ctx::Calibration>> = Vec::new();
            for gpu in self.device_gpus() {
                cal.push(gpu.calibrate());
            }
            for (i, c) in cal.into_iter().enumerate() {
                if let Some(c) = c {
                    if let Some(slot) = self.anchor.get_mut(i) {
                        *slot = Some(c);
                    }
                }
            }
            self.anchor_at = Some(std::time::Instant::now());
            for (i, c) in self.anchor.iter().enumerate() {
                if let Some(c) = c {
                    eprintln!(
                        "gpu-time anchor device {i}: gpu {} ns vs host {} ns (offset {:+.1} us, \
                         max deviation {} ns)",
                        c.device_ns,
                        c.host_ns,
                        (c.host_ns as f64 - c.device_ns as f64) / 1e3,
                        c.max_dev_ns
                    );
                }
            }
        }
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
        let tok = self.sampler.sample(logits, &self.sampling)?;
        // `BRECCIA_FAULT_DIVERGE_STEP=k`: return a different token at forward `k` of the
        // measured window, so `scripts/head-race-capture.sh` can be exercised without
        // waiting for the ~1-in-8 occurrence it exists to catch (§8bd).
        if std::env::var("BRECCIA_FAULT_DIVERGE_STEP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            == Some(self.forwards)
        {
            eprintln!(
                "breccia: FAULT INJECTION: diverging at forward {}",
                self.forwards
            );
            return Some((tok + 1) % self.model.cfg.vocab.max(1) as u32);
        }
        Some(tok)
    }

    /// The device this session's block `b` runs on (for per-card reporting).
    pub fn device_of(&self, b: usize) -> usize {
        self.model.device_of(b)
    }

    /// The poison value for `BRECCIA_FAULT_POISON_LOGITS`: a finite f32 no real logit can
    /// equal bit-for-bit (so a survivor is unambiguous) but which is also *finite*, so it
    /// loses an argmax to any real logit instead of winning it with +inf.
    pub fn logits_sentinel() -> f32 {
        -3.0e38
    }

    /// `(name, pci address)` of each of this session's devices.
    pub fn device_names(&self) -> Vec<(String, String)> {
        self.model
            .devices
            .iter()
            .map(|d| (d.info.name.clone(), d.info.pci_address.clone()))
            .collect()
    }

    /// Zero every per-sequence buffer — a new request on a reused session.
    ///
    /// The complete list (docs/phase-4/01-forward-pass.md §8az):
    ///
    /// | state | where | reset |
    /// |---|---|---|
    /// | GDN conv history + delta-rule state | `gdn_state[b]` × 36 | `GdnState::reset` |
    /// | QSA paged K/V pools | `qsa_state[b]` × 12 | `QsaState::reset` |
    /// | QSA indexer raw tail (last partial block) | `qsa_state[b].idx_tail` | `QsaState::reset` |
    /// | QSA pooled indexer keys | `qsa_state[b].idx_pooled` | `QsaState::reset` |
    /// | QSA page + position tables | `qsa_state[b]` | re-uploaded (invariant check) |
    /// | QSA per-call shape scalars (`n_blocks`, `max_ids`, `slots`) | `qsa_ws[b]` | rewritten by `run_qsa` every call, nothing to do |
    /// | PLE rolling n-gram window + conv history | `ple` | `PleHost::reset` |
    /// | expert bytes / forwards / head traffic | `Session` | zeroed below |
    /// | per-stage timers | `Session::timers` | `Timers::clear` |
    /// | position / token counters (`pos`, `cur_tok`) | `Session` | set by `begin_token` every token |
    /// | staging ring contents | `stagings` | `flush()` below (the ring is drained, and every read is fence-ordered) |
    /// | sampler RNG | `sampler` | **deliberately not reset** — sampling is seeded per draw, so a reset would not be a new sequence, it would be a different one |
    pub fn reset(&mut self) -> Result<()> {
        let m = self.model;
        for b in 0..m.cfg.n_layers {
            if let Some(st) = &self.gdn_state[b] {
                st.reset(m.devices[m.device_of(b)].gpu())?;
            }
            if let Some(st) = &self.qsa_state[b] {
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

    /// `BRECCIA_DUMP_STATE=<dir>` (compression fabric F0b): after decode position `pos`,
    /// write every GDN state + conv history and the live QSA K/V pages as raw device
    /// bytes, `p<pos>-b<block>-<part>.bin`. Every `BRECCIA_DUMP_EVERY` positions (64),
    /// at most `BRECCIA_DUMP_MAX` snapshots (8). Bench/lab only: it stalls the token.
    fn dump_state(&mut self, pos: usize) -> Result<()> {
        let Ok(dir) = std::env::var("BRECCIA_DUMP_STATE") else { return Ok(()) };
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (every, max) = (env("BRECCIA_DUMP_EVERY", 64).max(1), env("BRECCIA_DUMP_MAX", 8));
        if pos % every != 0 || self.dumps >= max {
            return Ok(());
        }
        self.dumps += 1;
        self.flush()?;
        std::fs::create_dir_all(&dir)?;
        let m = self.model;
        use breccia_kernels::{gdn, qsa};
        // exact byte counts: Buf sizes carry allocation padding
        let pages = (pos + 1).div_ceil(qsa::PAGE_CELLS);
        let gdn_state = gdn::MODEL_S * gdn::MODEL_H_V * gdn::MODEL_S * 4;
        let gdn_conv = (gdn::MODEL_D_CONV - 1) * gdn::CONV_CHANNELS * 4;
        let k_live = qsa::sizes::f16_pool_bytes(pages) as usize;
        for b in 0..m.cfg.n_layers {
            let gpu = m.devices[m.device_of(b)].gpu();
            let mut parts: Vec<(&str, &Buf, usize)> = Vec::new();
            if let Some(st) = &self.gdn_state[b] {
                parts.push(("gdn_state", &st.state, gdn_state));
                parts.push(("gdn_conv", &st.conv, gdn_conv));
            }
            if let Some(st) = &self.qsa_state[b] {
                let (v_name, v_live) = match st.v_layout {
                    qsa::VLayout::F16 => ("qsa_v", k_live),
                    qsa::VLayout::Q8_0 => ("qsa_vq8", qsa::sizes::q8v_pool_bytes(pages) as usize),
                };
                parts.push(("qsa_k", &st.k_pool, k_live));
                parts.push((v_name, &st.v_pool, v_live));
            }
            for (name, buf, n) in parts {
                let bytes = breccia_kernels::gemv::download(gpu, buf, n.min(buf.size as usize))?;
                std::fs::write(format!("{dir}/p{pos:06}-b{b:02}-{name}.bin"), bytes)?;
            }
        }
        Ok(())
    }

    /// `BRECCIA_DUMP_LAYOUT=1`: the per-device buffer layout (size + Vulkan handles) as
    /// the session allocated it. The failing process's layout is the one thing no
    /// ordering or coverage check can see, so it is printed on request and can be
    /// diffed against a clean process with `scripts/head-race-layout.sh`.
    pub fn layout_dump(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (dev, v) in self.venues.iter().enumerate() {
            let mut row = |name: &str, b: &Buf| {
                out.push(format!(
                    "dev{dev} {name:<20} size {:>9}  buf {:?} mem {:?}",
                    b.size, b.buf, b.mem
                ))
            };
            row("residual r", &v.r);
            row("block_out", &v.block_out.buf);
            row("sh", &v.sh);
            row("sh_gate_logit", &v.sh_gate_logit);
            row("router_logits", &v.router_logits);
            row("router_ids", &v.router_ids);
            row("router_w", &v.router_w);
            row("ple_key_out", &v.ple_key_out);
            row("ple_value_out", &v.ple_value_out);
            row("ple_emb", &v.ple_emb);
            if let Some(ws) = &v.hc_ws {
                row("hc.mixed", &ws.mixed);
                row("hc.xn", &ws.xn);
                row("hc.gate", &ws.gate);
            }
            if let Some(l) = &v.logits {
                row("logits", l);
            }
        }
        for (b, st) in self.qsa_state.iter().enumerate() {
            if let Some(st) = st {
                out.push(format!(
                    "blk{b:<3} qsa k_pool {:>9} v_pool {:>9} ({}) tail {:>7} pooled {:>7}",
                    st.k_pool.size,
                    st.v_pool.size,
                    st.v_layout.name(),
                    st.idx_tail.size,
                    st.idx_pooled.size
                ));
            }
        }
        out
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
            self.probe_mark(dev, batch_cmd::probe::MOE_MAP);
            self.run_placed_moe(b, dev, pe, mc.n_embd, mc.n_ff, mc.n_expert_used)?;
            self.timers.mark(HC_WRITE_F);
            self.probe_mark(dev, batch_cmd::probe::HC_WRITE);
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
            self.probe_mark(dev, batch_cmd::probe::XFER);
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
        self.block_seg = self.pending_seg;
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
        self.probe_mark(dev, batch_cmd::probe::HC_READ);
        {
            let v = &self.venues[dev];
            hc_read(v, &m.layers[b].hc_attn, &v.r)?;
        }
        let mixed_attn = self.maybe_checksum(trace.is_some(), dev, BufKind::Mixed, mc.n_embd)?;

        // ---- (C) the token mixer --------------------------------------------
        self.timers.mark(MIXER);
        match (self.mixer_from_chunk, m.layers[b].kind) {
            (Some(pos0), _) => {
                let v = &self.venues[dev];
                let pf = v.pf.as_ref().expect("the merged prefill scratch");
                let nb = (mc.n_embd * 4) as u64;
                let src = (pos - pos0) as u64 * nb;
                batch_cmd::record_copies(
                    m.devices[dev].gpu(),
                    &[(&v.block_out.buf, 0, &pf.out, src, nb)],
                )?;
            }
            (None, LayerKind::Gdn) => self.run_gdn(b, dev)?,
            (None, LayerKind::Qsa) => self.run_qsa(b, dev, pos)?,
        }
        let mixer_out = self.maybe_checksum(trace.is_some(), dev, BufKind::BlockOut, mc.n_embd)?;

        // ---- (D) HC_WRITE(attn) ----------------------------------------------
        self.timers.mark(HC_WRITE_A);
        self.probe_mark(dev, batch_cmd::probe::HC_WRITE);
        {
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.hc.write(ws, &v.r, &v.block_out.buf, &v.r)?;
        }

        // ---- (E) HC_READ(ffn) ------------------------------------------------
        self.timers.mark(HC_READ_F);
        self.probe_mark(dev, batch_cmd::probe::HC_READ);
        {
            let v = &self.venues[dev];
            hc_read(v, &m.layers[b].hc_ffn, &v.r)?;
        }
        let mixed_ffn = self.maybe_checksum(trace.is_some(), dev, BufKind::Mixed, mc.n_embd)?;

        // ---- (F) router ------------------------------------------------------
        self.timers.mark(ROUTER);
        self.probe_mark(dev, batch_cmd::probe::ROUTER);
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
        self.probe_mark(dev, batch_cmd::probe::SH_GATE);
        let mut sub = [0f64; 5];
        {
            let lw = &m.layers[b];
            let v = &self.venues[dev];
            let pipes = &v.pipes;
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            // Merged pass 1 with BRECCIA_PF_SHCHUNK: the four runs below happen once
            // per chunk instead (`prefill_block_merged`); only the scalar gate stays.
            if !self.shared_deferred {
                // `gate` and `up` both read `ws.mixed` and write disjoint outputs, so
                // the fence between them is dead: drop *gate's* trailing fence (§8z).
                batch_cmd::skip_next_barrier();
                // Fine-grained attribution (§8ad): this stage measures as 15-21 ms/token
                // while the routed shared expert prices at ~0.9, so time each call.
                // The blocking GPU wait (the fence in `wall_run` when no batch is
                // open) sits inside these runs: mark it off SHARED before the first
                // one, or the breakdown charges the wait to shared-exp compute (§8ae).
                let t0 = std::time::Instant::now();
                self.timers.mark(WAIT_GPU);
                pipes
                    .get(&lw.moe.gate.key)?
                    .run(&lw.moe.gate.buf, &ws.mixed, &v.shexp_gate)?;
                sub[0] = t0.elapsed().as_secs_f64();
                // The gate run is where the blocking fence wait landed; re-anchor to
                // shared-exp so only the wait, not the remaining shared-expert runs,
                // is priced as wait-gpu (R28-1: 5162653 relabelled the whole span).
                self.timers.mark(SHARED);
                self.probe_mark(dev, batch_cmd::probe::SH_UP);
                let t0 = std::time::Instant::now();
                pipes
                    .get(&lw.moe.up.key)?
                    .run(&lw.moe.up.buf, &ws.mixed, &v.shexp_up)?;
                sub[1] = t0.elapsed().as_secs_f64();
                self.probe_mark(dev, batch_cmd::probe::SH_SWIGLU);
                let t0 = std::time::Instant::now();
                v.swiglu
                    .run(&v.shexp_gate, &v.shexp_up, &v.shexp_act, mc.n_ff)?;
                sub[2] = t0.elapsed().as_secs_f64();
                self.probe_mark(dev, batch_cmd::probe::SH_DOWN);
                let t0 = std::time::Instant::now();
                pipes
                    .get(&lw.moe.down.key)?
                    .run(&lw.moe.down.buf, &v.shexp_act, &v.sh)?;
                sub[3] = t0.elapsed().as_secs_f64();
            }
            self.probe_mark(dev, batch_cmd::probe::SH_SCALAR);
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
    /// Requires a configured [`SessionCfg::prefill_chunk`]; a layer-split
    /// placement hands the chunk's residuals across devices ([`Self::handoff_slots`]).
    pub fn prefill(&mut self, toks: &[u32], pos0: usize) -> Result<Vec<f32>> {
        Ok(self
            .prefill_rows(toks, pos0, false)?
            .pop()
            .expect("the head ran on at least one row"))
    }

    /// [`Self::prefill`], with the head on every row when `all_rows` (one logits
    /// vector per token) instead of the last one only.
    fn prefill_rows(&mut self, toks: &[u32], pos0: usize, all_rows: bool) -> Result<Vec<Vec<f32>>> {
        let t = toks.len();
        if t == 0 {
            bail!("an empty prefill chunk");
        }
        let m = self.model;
        let mc = &m.cfg;
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
        let dev = m.device_of(0);
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

        let micro_blocked = if pf_micro() { self.micro_blocker(t, pos0) } else { Some(String::new()) };
        // Only say why when asked for explicitly: default-on falls back silently.
        if let Some(why) = micro_blocked.as_ref().filter(|w| !w.is_empty() && std::env::var_os("BRECCIA_PF_MICRO").is_some()) {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("breccia: BRECCIA_PF_MICRO=2 off for this prefill: {why}");
            }
        }
        if micro_blocked.is_none() {
            self.prefill_micro(t, pos0)?;
        } else {
            for b in 0..mc.n_layers {
                if b > 0 && m.device_of(b - 1) != m.device_of(b) {
                    self.handoff_slots(m.device_of(b - 1), m.device_of(b), t)?;
                }
                self.prefill_block(b, t, pos0)?;
            }
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
        let last = m.device_of(mc.n_layers - 1);
        let mut logits = Vec::new();
        if all_rows && self.spec.as_ref().is_some_and(|sp| sp.active && sp.replay) {
            logits = self.spec_head(t)?;
            self.pf_wall_s += wall0.elapsed().as_secs_f64();
            return Ok(logits);
        }
        // shortcut: one head submit per row (~7 ms each) outside spec replay.
        for i in if all_rows { 0..t } else { t - 1..t } {
            self.swap_r_in(last, i);
            self.submit_head()?;
            logits.push(self.finish_head()?);
            self.swap_r_in(last, i);
        }
        self.timers.stop();
        self.pf_wall_s += wall0.elapsed().as_secs_f64();
        Ok(logits)
    }

    /// Inside a rows-mode verify: the per-row hooks apply and the mixers run per row.
    fn spec_rows_active(&self) -> bool {
        self.spec.as_ref().is_some_and(|sp| sp.active && !sp.replay)
    }

    /// Replay mode can take this model: merged prefill, every block's mixer and
    /// expert types have a multi-user GEMM, and so does the head.
    fn spec_replay_ok(&self) -> bool {
        let m = self.model;
        let v = &self.venues[0];
        let pipes = &v.pipes;
        std::env::var("BRECCIA_SPEC_VERIFY_ROWS").as_deref() != Ok("1")
            && self.prefill_merged
            && v.pf.is_some()
            && v.r_chunk.is_some()
            && v.logits.is_some()
            && pipes.gemm(m.head.key.ty).is_some()
            && m.head.key.rows == m.cfg.vocab
            && (0..m.cfg.n_layers).all(|b| {
                let plan = &m.layers[b].experts;
                let (gt, ut, dt) = match m.placed.get(&b) {
                    Some(pe) => (pe.gate.key.ty, pe.up.key.ty, pe.down.key.ty),
                    None => (plan.gate.ty, plan.up.ty, plan.down.ty),
                };
                self.mixer_chunked(b, 0) && [gt, ut, dt].iter().all(|ty| pipes.gemm(*ty).is_some())
            })
    }

    /// Allocate rollback slots for verifying up to `max_t` rows (~112 MiB of GDN
    /// state per slot). Greedy, single device, `--batching` only.
    pub fn spec_enable(&mut self, max_t: usize) -> Result<()> {
        let m = self.model;
        if m.devices.len() != 1 {
            bail!("speculative verify is single-device only ({} devices)", m.devices.len());
        }
        if !self.batching {
            bail!("speculative verify records into a dispatch batch: it needs --batching");
        }
        if max_t < 2 {
            bail!("a verify chunk is at least 2 rows");
        }
        self.spec_release();
        let gpu = m.devices[0].gpu();
        let replay = self.spec_replay_ok();
        let mut sp = Box::new(SpecSnap {
            gdn: Vec::new(),
            tail: Vec::new(),
            ple_rows: vec![Vec::new(); max_t],
            ple_pre: None,
            base_ple: None,
            base_pos: 0,
            base_tok: 0,
            toks: Vec::new(),
            pos0: 0,
            max_t,
            active: false,
            replay,
            gdn_tap: Vec::new(),
            idx_k_tap: Vec::new(),
            head_logits: None,
            head_ws: Vec::new(),
        });
        let mut made = Ok(());
        'slots: for _ in 0..if replay { 1 } else { max_t } {
            for b in 0..m.cfg.n_layers {
                let g = match &self.gdn_state[b] {
                    Some(_) => GdnState::new(gpu).map(Some),
                    None => Ok(None),
                };
                let tl = match &self.qsa_state[b] {
                    Some(_) => {
                        let bytes = breccia_kernels::qsa::sizes::idx_tail_bytes();
                        breccia_kernels::gemv::alloc(gpu, bytes).map(Some)
                    }
                    None => Ok(None),
                };
                match (g, tl) {
                    (Ok(g), Ok(tl)) => {
                        sp.gdn.push(g);
                        sp.tail.push(tl);
                    }
                    (g, tl) => {
                        if let Ok(Some(g)) = g {
                            g.release(gpu);
                        }
                        if let Ok(Some(tl)) = &tl {
                            gpu.destroy_buf(tl);
                        }
                        made = Err(anyhow!("allocating a speculative rollback slot"));
                        break 'slots;
                    }
                }
            }
        }
        if made.is_ok() && replay {
            made = (|| -> Result<()> {
                for b in 0..m.cfg.n_layers {
                    sp.gdn_tap.push(match &self.gdn_state[b] {
                        Some(_) => Some(GdnTap::new(gpu, max_t)?),
                        None => None,
                    });
                    sp.idx_k_tap.push(match &self.qsa_state[b] {
                        Some(_) => Some(breccia_kernels::gemv::alloc(
                            gpu,
                            (max_t * breccia_kernels::qsa::IDX_DIM * 4) as u64,
                        )?),
                        None => None,
                    });
                }
                sp.head_logits = Some(breccia_kernels::gemv::alloc(gpu, (max_t * m.cfg.vocab * 4) as u64)?);
                for t in 2..=max_t {
                    sp.head_ws.push(self.venues[0].final_hc.workspace(gpu, t)?);
                }
                Ok(())
            })();
        }
        self.spec = Some(sp);
        if made.is_err() {
            self.spec_release();
        }
        made
    }

    fn spec_release(&mut self) {
        if let Some(mut sp) = self.spec.take() {
            let gpu = self.model.devices[0].gpu();
            for g in sp.gdn.drain(..).flatten() {
                g.release(gpu);
            }
            for t in sp.tail.drain(..).flatten() {
                gpu.destroy_buf(&t);
            }
            for g in sp.gdn_tap.drain(..).flatten() {
                g.release(gpu);
            }
            for k in sp.idx_k_tap.drain(..).flatten() {
                gpu.destroy_buf(&k);
            }
            if let Some(l) = sp.head_logits.take() {
                gpu.destroy_buf(&l);
            }
            for ws in sp.head_ws.drain(..) {
                self.venues[0].final_hc.release(gpu, ws);
            }
        }
    }

    /// Record the copy of block `b`'s recurrent state (GDN state + conv, or the
    /// QSA indexer tail) between the live state and rollback slot `slot`.
    fn spec_copy_block(&self, b: usize, slot: usize, save: bool) -> Result<()> {
        let sp = self
            .spec
            .as_ref()
            .ok_or_else(|| anyhow!("no speculative rollback slots: call spec_enable"))?;
        let gpu = self.model.devices[0].gpu();
        let at = slot * self.model.cfg.n_layers + b;
        let copy = |live: &Buf, snap: &Buf, len: u64| {
            let (dst, src) = if save { (snap, live) } else { (live, snap) };
            batch_cmd::record_copy(gpu, dst, 0, src, 0, len)
        };
        if let (Some(live), Some(snap)) = (&self.gdn_state[b], &sp.gdn[at]) {
            copy(&live.state, &snap.state, live.state.size)?;
            copy(&live.conv, &snap.conv, live.conv.size)?;
        }
        if let (Some(live), Some(snap)) = (&self.qsa_state[b], &sp.tail[at]) {
            copy(&live.idx_tail, snap, breccia_kernels::qsa::sizes::idx_tail_bytes())?;
        }
        Ok(())
    }

    /// Verify hook, after row `i`'s mixer: save block `b`'s state as "after
    /// `i + 1` rows". The last row's state is the live one, so it needs no slot.
    fn spec_row_state(&self, b: usize, i: usize, t: usize) -> Result<()> {
        if self.spec_rows_active() && i + 1 < t {
            self.spec_copy_block(b, i + 1, true)?;
        }
        Ok(())
    }

    /// Verify hook, after row `i`'s PLE block: the history "after `i + 1` rows".
    fn spec_ple_row(&mut self, i: usize, t: usize) {
        if i + 1 < t {
            let sp = self.spec.as_mut().filter(|s| s.active);
            if let (Some(sp), Some(p)) = (sp, self.ple.as_ref()) {
                sp.ple_rows[i + 1] = p.state().hist;
            }
        }
    }

    /// Copy every block's state between the live state and `slot`, as one batch.
    fn spec_sweep(&mut self, slot: usize, save: bool) -> Result<()> {
        self.flush()?;
        self.ensure_batch(0)?;
        for b in 0..self.model.cfg.n_layers {
            self.spec_copy_block(b, slot, save)?;
        }
        self.submit_batch()?;
        self.flush()
    }

    /// Bench only: snapshot the whole state now (slot 0) so [`Self::spec_restore_base`]
    /// can return to it. A real draft loop never needs this.
    pub fn spec_save_base(&mut self) -> Result<()> {
        self.spec_sweep(0, true)?;
        let base_ple = self.ple.as_ref().map(|p| p.state());
        let (pos, tok) = (self.pos, self.cur_tok);
        let sp = self.spec.as_mut().ok_or_else(|| anyhow!("call spec_enable first"))?;
        sp.base_ple = base_ple;
        sp.base_pos = pos;
        sp.base_tok = tok;
        Ok(())
    }

    pub fn spec_restore_base(&mut self) -> Result<()> {
        self.spec_sweep(0, false)?;
        let sp = self.spec.as_ref().ok_or_else(|| anyhow!("call spec_enable first"))?;
        if let (Some(p), Some(s)) = (self.ple.as_mut(), sp.base_ple.as_ref()) {
            p.restore(s);
        }
        (self.pos, self.cur_tok) = (sp.base_pos, sp.base_tok);
        Ok(())
    }

    /// Speculative verify: run `toks` (row 0 the last confirmed token, rows 1.. the
    /// drafts) at `pos0` as one chunk and return every row's logits (row `i` is
    /// the prediction after `toks[i]`). The state is left as if all rows were
    /// decoded; [`Self::rollback_to`] undoes the rejected tail.
    pub fn verify(&mut self, toks: &[u32], pos0: usize) -> Result<Vec<Vec<f32>>> {
        let ple_pre = self.ple.as_ref().map(|p| p.state());
        let sp = self
            .spec
            .as_mut()
            .ok_or_else(|| anyhow!("verify needs spec_enable first"))?;
        if toks.len() < 2 || toks.len() > sp.max_t {
            bail!("verify takes 2..={} rows, got {}", sp.max_t, toks.len());
        }
        sp.toks = toks.to_vec();
        sp.pos0 = pos0;
        sp.ple_pre = ple_pre;
        let replay = sp.replay;
        // Replay: slot 0 = the state before the chunk, which rollback restores.
        if replay {
            self.spec_sweep(0, true)?;
        }
        if let Some(sp) = self.spec.as_mut() {
            sp.active = true;
        }
        let rows = self.prefill_rows(toks, pos0, true);
        if let Some(sp) = self.spec.as_mut() {
            sp.active = false;
        }
        rows
    }

    /// After [`Self::verify`]: make the state what decoding `toks[..=row]` would
    /// have left (accept `row` drafts), so the next token is `toks[row]`'s successor.
    pub fn rollback_to(&mut self, row: usize) -> Result<()> {
        let sp = self.spec.as_ref().ok_or_else(|| anyhow!("call spec_enable first"))?;
        let n = row + 1;
        if n > sp.toks.len() {
            bail!("rollback_to({row}) on a verify of {} rows", sp.toks.len());
        }
        if n < sp.toks.len() {
            if sp.replay {
                self.spec_replay_to(n)?;
            } else {
                self.spec_sweep(n, false)?;
            }
            let sp = self.spec.as_ref().expect("checked above");
            if let (Some(p), Some(pre)) = (self.ple.as_mut(), sp.ple_pre.as_ref()) {
                p.restore(&PleState {
                    hist: sp.ple_rows[n].clone(),
                    recent: pre.recent.clone(),
                });
                for &tok in &sp.toks[..n] {
                    p.rows_for(tok);
                }
            }
        }
        let sp = self.spec.as_ref().expect("checked above");
        (self.pos, self.cur_tok) = (sp.pos0 + row, sp.toks[row]);
        Ok(())
    }

    /// Replay-mode rollback: restore the pre-chunk state (slot 0), then re-run the
    /// state-update kernels over the first `n` rows of what the chunk tapped.
    fn spec_replay_to(&mut self, n: usize) -> Result<()> {
        let m = self.model;
        let pos0 = self.spec.as_ref().map_or(0, |sp| sp.pos0);
        self.flush()?;
        self.ensure_batch(0)?;
        self.gdn_chunk_ws[0]
            .as_mut()
            .ok_or_else(|| anyhow!("no GDN chunk workspace: nothing was verified"))?
            .tokens = n;
        for b in 0..m.cfg.n_layers {
            self.spec_copy_block(b, 0, false)?;
            let sp = self.spec.as_ref().expect("replay needs spec_enable");
            let v = &self.venues[0];
            if let (Some(state), Some(tap), Some(gw)) =
                (&self.gdn_state[b], &sp.gdn_tap[b], &m.layers[b].gdn)
            {
                let gdn = v.gdn.as_ref().ok_or_else(|| anyhow!("no GDN pipelines"))?;
                let ws = self.gdn_chunk_ws[0].as_ref().expect("checked above");
                gdn.replay(state, ws, &gw.conv, tap)?;
            }
            if let (Some(state), Some(tap)) = (&self.qsa_state[b], &sp.idx_k_tap[b]) {
                let qsa = v.qsa.as_ref().ok_or_else(|| anyhow!("no QSA pipelines"))?;
                qsa.idx_tail_raw(state, tap, n, pos0)?;
            }
        }
        self.submit_batch()?;
        self.flush()
    }

    /// Replay-mode head: the final HC read and ONE `t`-row head GEMM over the
    /// chunk's residual slots, instead of `t` single-row heads.
    fn spec_head(&mut self, t: usize) -> Result<Vec<Vec<f32>>> {
        let m = self.model;
        let mc = &m.cfg;
        let gpu = m.devices[0].gpu();
        let hb = (mc.hc_dim() * 4) as u64;
        self.flush()?;
        self.timers.mark(FINAL_HC);
        self.ensure_batch(0)?;
        {
            let sp = self.spec.as_ref().expect("replay needs spec_enable");
            let v = &self.venues[0];
            let rc = v.r_chunk.as_ref().expect("checked by spec_replay_ok");
            let ws = &sp.head_ws[t - 2];
            let out = sp.head_logits.as_ref().expect("replay allocates the head output");
            let gather: Vec<(&Buf, u64, &Buf, u64, u64)> =
                (0..t).map(|i| (rc, i as u64 * hb, &v.r_slots[i], 0, hb)).collect();
            batch_cmd::record_copies(gpu, &gather)?;
            let hw = HcWeights {
                norm: &m.out_hc_norm,
                down: &m.out_hc_down.buf,
                up: &m.out_hc_up.buf,
                inject: None,
            };
            v.final_hc.read(ws, rc, &hw)?;
            self.timers.mark(HEAD);
            v.pipes
                .gemm(m.head.key.ty)
                .expect("checked by spec_replay_ok")
                .run(&m.head.buf, &ws.mixed, out, m.head.key.rows, m.head.key.cols, t)?;
        }
        self.submit_batch()?;
        self.flush()?;
        let raw = {
            let sp = self.spec.as_ref().expect("replay needs spec_enable");
            let out = sp.head_logits.as_ref().expect("replay allocates the head output");
            // As many rows per readback as the staging buffer holds (a small
            // `--prefill-chunk` sizes it for one or two rows).
            let per = ((self.stagings[0].capacity() as usize) / (mc.vocab * 4)).max(1);
            let mut raw = Vec::with_capacity(t);
            for rows in (0..t).collect::<Vec<_>>().chunks(per) {
                let items: Vec<(&Buf, u64, usize)> = rows
                    .iter()
                    .map(|&i| (out, (i * mc.vocab * 4) as u64, mc.vocab * 4))
                    .collect();
                raw.extend(self.stagings[0].read_batch(gpu, &items)?);
            }
            raw
        };
        self.timers.stop();
        self.forwards += t;
        raw.iter()
            .map(|r| {
                let values = f32_from_bytes(r);
                match values.iter().position(|v| !v.is_finite()) {
                    Some(i) => bail!("logit {i} is not finite ({})", values[i]),
                    None => Ok(values),
                }
            })
            .collect()
    }

    /// Layer-split chunked prefill: carry the chunk's `t` residual slots from the
    /// card that ran the previous block to the one that runs the next, as one fenced
    /// readback and one upload for the whole chunk (host-staged, like the decode
    /// hand-off in [`Session::submit_block`]; no GPU-to-GPU copy).
    fn handoff_slots(&mut self, from: usize, to: usize, t: usize) -> Result<()> {
        let m = self.model;
        let hb = m.cfg.hc_dim() * 4;
        self.flush()?;
        self.timers.mark(HANDOFF);
        let items: Vec<(&Buf, u64, usize)> =
            self.venues[from].r_slots[..t].iter().map(|s| (s, 0, hb)).collect();
        let bytes = self.stagings[from].read_batch(m.devices[from].gpu(), &items)?;
        let regions: Vec<(&Buf, u64, &[u8])> = self.venues[to].r_slots[..t]
            .iter()
            .zip(bytes.iter())
            .map(|(s, d)| (s, 0, d.as_slice()))
            .collect();
        self.stagings[to].write_regions(m.devices[to].gpu(), &regions)?;
        Ok(())
    }

    /// One block of a chunked prefill: the dense chain for every token of the
    /// chunk in one submission, then the **grouped** expert lane, then the ffn
    /// `HC_WRITE` for every token in a second submission.
    fn prefill_block(&mut self, b: usize, t: usize, pos0: usize) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let dev = m.device_of(b);
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
                && !self.spec_rows_active()
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
                    self.spec_ple_row(i, t);
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
            self.spec_row_state(b, i, t)?;
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
                let inject_bytes = (mc.hc * v.hc.inject_chunks() * 4) as u64;
                batch_cmd::record_copy(
                    self.model.devices[dev].gpu(),
                    &v.inject_slots[i],
                    0,
                    &ws.inject,
                    0,
                    inject_bytes.min(ws.inject.size).min(v.inject_slots[i].size),
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
            let t_lock = lane_stats_on().then(std::time::Instant::now);
            let mut pool = self.pool.lock().expect("the expert pool's lock");
            let waited = t_lock.map_or(0.0, |t| t.elapsed().as_secs_f64());
            let bytes = pool
                .moe_topk_chunk(&m.layers[b].experts, &routes, &self.pf_mixed, &mut out)
                .with_context(|| format!("the grouped expert lane on block {b}"))?;
            drop(pool);
            self.lock_wait_s += waited;
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
                if let Some(path) = std::env::var_os("BRECCIA_TRACE_ROUTE_FILE") {
                    let path = std::path::PathBuf::from(path);
                    for i in 0..t {
                        append_route_row(
                            &path,
                            false,
                            pos0 + i,
                            b,
                            &self.pf_ids[i],
                            &self.pf_weights[i],
                        )?;
                    }
                }
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

    /// D4d: the three role byte-slices for streamed block `b` expert `expert`,
    /// read out of the imported host arena instead of the mmap. `streamed_before`
    /// is how many *streamed* (non-placed) blocks come before this one — the
    /// builder enumerated only non-placed blocks in (block, gate/up/down) order,
    /// so streamed block `b` lands at blob `3·streamed_before + role`.
    fn imported_role_slices<'a>(
        arena: &'a crate::host_arena::HostArena,
        streamed_before: usize,
        expert: usize,
        gate_len: usize,
        up_len: usize,
        down_len: usize,
    ) -> Result<[&'a [u8]; 3]> {
        let want = [gate_len, up_len, down_len];
        let mut out: [&'a [u8]; 3] = [&[]; 3];
        for (r, ln) in want.iter().enumerate() {
            let blob = arena
                .blob(3 * streamed_before + r)
                .map_err(|e| anyhow!("imported expert arena: {e}"))?;
            let (a, z) = (expert * ln, (expert + 1) * ln);
            if z > blob.len() {
                bail!(
                    "imported role blob {r} holds {} bytes, expert {expert} needs [{a}, {z})",
                    blob.len()
                );
            }
            out[r] = &blob[a..z];
        }
        Ok(out)
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
        // D4d: with the host expert arena imported, the copy source is the
        // live host mapping, not the mmap staging copy. Blob i holds the
        // (b, r) pair at i = 3·(b − placed_before) + r, matching the builder.
        let imported = {
            let (g, u, d) = (
                slices.gate.len() as u64,
                slices.up.len() as u64,
                slices.down.len() as u64,
            );
            // A VRAM-placed block has no blobs (`BRECCIA_PF_EXPERT_ALL`): the mmap.
            let vram = m.placed.get(&b).is_some_and(|pe| !pe.host_imported);
            m.imported_experts.as_ref().filter(|_| !vram).and_then(|arena| {
                // Blob order skips VRAM-resident blocks: the ordinal is b minus
                // every VRAM-placed block below it (host-imported ones keep blobs).
                let streamed_before =
                    b - m.placed.range(..b).filter(|(_, pe)| !pe.host_imported).count();
                // The arena is a capped prefix of the streamed blocks; later ones
                // keep streaming from the mmap.
                if 3 * streamed_before + 3 > arena.blobs().len() {
                    return None;
                }
                Some(Self::imported_role_slices(
                    arena,
                    streamed_before,
                    expert,
                    g as usize,
                    u as usize,
                    d as usize,
                ))
            })
        };
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
            match &imported {
                Some(Ok([g_s, u_s, d_s])) => {
                    pf.stream.write_at(off, g_s)?;
                    pf.stream.write_at(off + g, u_s)?;
                    pf.stream.write_at(off + g + u, d_s)?;
                }
                Some(Err(e)) => bail!("imported expert read: {e}"),
                None => {
                    pf.stream.write_at(off, slices.gate)?;
                    pf.stream.write_at(off + g, slices.up)?;
                    pf.stream.write_at(off + g + u, slices.down)?;
                }
            }
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
    /// The shared expert for the whole chunk: one dispatch per role, `t` users
    /// over `pf.mixed`; the down GEMM writes `pf.shared` in the fan-in's layout.
    fn record_shared_chunk(&mut self, b: usize, dev: usize, t: usize) -> Result<()> {
        self.timers.mark(SHARED);
        let m = self.model;
        let (mc, gpu) = (&m.cfg, m.devices[dev].gpu());
        let n_embd = mc.n_embd;
        let moe = &m.layers[b].moe;
        let v = &self.venues[dev];
        let pf = v.pf.as_ref().expect("the merged prefill scratch");
        let n_sh = mc.n_ff_shared;
        let g = |w: &crate::model::W| v.pipes.gemm(w.key.ty).expect("checked above");
        match moe.gate_up.as_ref().filter(|w| g(w).has_pair()) {
            // One dispatch for gate + up + SwiGLU (§11l item 4), bit-identical.
            Some(gu) => {
                batch_cmd::probe::enter(gpu, batch_cmd::probe::SH_GATE);
                g(gu).run_pair(&gu.buf, &pf.mixed, &pf.sh_a, n_sh, n_embd, t)?;
            }
            None => {
                batch_cmd::probe::enter(gpu, batch_cmd::probe::SH_GATE);
                g(&moe.gate).run(&moe.gate.buf, &pf.mixed, &pf.sh_g, n_sh, n_embd, t)?;
                batch_cmd::probe::enter(gpu, batch_cmd::probe::SH_UP);
                g(&moe.up).run(&moe.up.buf, &pf.mixed, &pf.sh_u, n_sh, n_embd, t)?;
                batch_cmd::probe::enter(gpu, batch_cmd::probe::SH_SWIGLU);
                v.swiglu.run(&pf.sh_g, &pf.sh_u, &pf.sh_a, t * n_sh)?;
            }
        }
        batch_cmd::probe::enter(gpu, batch_cmd::probe::SH_DOWN);
        g(&moe.down).run(&moe.down.buf, &pf.sh_a, &pf.shared, n_embd, n_sh, t)?;
        Ok(())
    }

    /// What the merged prefill does for block `b`: where its experts run and which
    /// chunk-wide paths are available. Shared by [`Session::prefill_block_merged`] and
    /// the `BRECCIA_PF_MICRO` eligibility check.
    fn merged_plan(
        &self,
        b: usize,
        dev: usize,
        pe: Option<&'m crate::model::PlacedExperts>,
    ) -> MergedPlan<'m> {
        let m = self.model;
        let plan = &m.layers[b].experts;
        // §11f: with `BRECCIA_PF_EXPERT_DEVICE` a resident block's experts stream to
        // the expert device too: the RX's streamed GEMM beats the MI50's arena GEMM
        // ~2.5x (KL 1.6e-5 vs the arena, top-1 same). Only when the arena holds the
        // file types; `BRECCIA_PF_EXPERT_ALL=0` keeps them on the arena.
        let to_x = std::env::var("BRECCIA_PF_EXPERT_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .is_some_and(|x| x != dev && x < m.devices.len() && self.venues[x].pf.is_some());
        let pe = pe.filter(|p| {
            !to_x
                || std::env::var("BRECCIA_PF_EXPERT_ALL").as_deref() == Ok("0")
                || (p.gate.key.ty, p.up.key.ty, p.down.key.ty)
                    != (plan.gate.ty, plan.up.ty, plan.down.ty)
        });
        let (gate_ty, up_ty, down_ty) = match pe {
            Some(pe) => (pe.gate.key.ty, pe.up.key.ty, pe.down.key.ty),
            None => (plan.gate.ty, plan.up.ty, plan.down.ty),
        };
        // A **streamed** block (no arena): each distinct expert's three blobs are
        // copied once per chunk out of the mmap into a host-visible slot and then
        // to the device, and the multi-user GEMM runs on them with base 0.
        let streamed = pe.is_none();
        // §11f dual-card prototype: `BRECCIA_PF_EXPERT_DEVICE=<ordinal>` runs a
        // streamed block's expert GEMMs and fan-in on that device instead (the RX
        // streams experts over x16, the MI50 over x4). Host-staged both ways: the
        // fan-in's inputs go out, `pf.out` comes back — no GPU-to-GPU copy.
        let xdev = std::env::var("BRECCIA_PF_EXPERT_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&x| {
                streamed && x != dev && x < m.devices.len() && self.venues[x].pf.is_some()
            });
        // The chunk-level shared expert needs a multi-user GEMM for all three roles.
        let shared_chunk = self.pf[dev][PF_SHCHUNK] && {
            let moe = &m.layers[b].moe;
            let pipes = &self.venues[dev].pipes;
            [&moe.gate, &moe.up, &moe.down]
                .iter()
                .all(|w| pipes.gemm(w.key.ty).is_some())
        };

        // The whole pass-1 dense chain as T-token dispatches (`record_hc_chunk`).
        let hc_chunk = shared_chunk
            && self.pf[dev][PF_HCCHUNK]
            && self.venues[dev].r_chunk.is_some()
            && self.mixer_chunked(b, dev)
            && {
                let moe = &m.layers[b].moe;
                let pipes = &self.venues[dev].pipes;
                pipes.gemm(moe.router.key.ty).is_some()
                    && pipes.gemm(moe.shexp_gate.key.ty).is_some()
            };
        MergedPlan {
            pe,
            gate_ty,
            up_ty,
            down_ty,
            streamed,
            xdev,
            shared_chunk,
            hc_chunk,
        }
    }

    fn prefill_block_merged(
        &mut self,
        b: usize,
        t: usize,
        pos0: usize,
        pe: Option<&'m crate::model::PlacedExperts>,
    ) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let dev = m.device_of(b);
        let n_embd = mc.n_embd;
        let n_ff = mc.n_ff;
        let k = mc.n_expert_used;
        let gpu = m.devices[dev].gpu();
        let MergedPlan {
            pe,
            gate_ty,
            up_ty,
            down_ty,
            streamed,
            xdev,
            shared_chunk,
            hc_chunk,
        } = self.merged_plan(b, dev, pe);
        let micro = self.micro.is_some();
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
        let overlap = self.pf[dev][PF_OVERLAP];

        // With an expert device the chunk shared expert waits until the routed
        // GEMMs are submitted there, so this card runs it under them (§12 #10).
        // `BRECCIA_PF_SHARED_UNDER=0` keeps it here. A micro-batch half records it
        // with its dense chain instead (`BRECCIA_PF_MICRO`).
        let defer_shared = !micro
            && shared_chunk
            && xdev.is_some()
            && std::env::var("BRECCIA_PF_SHARED_UNDER").as_deref() != Ok("0");
        let bound_t0 = std::time::Instant::now();

        // ---- pass 1: the dense chain per token, the MoE inputs saved ---------
        if !micro {
            self.ensure_batch(dev)?;
            // The block's mixer for the whole chunk at once: every token's attn-mixed
            // row into `pf.mixed`, one T-token forward into `pf.out`; the per-token
            // prelude below recomputes HC_READ(attn) (its inject partials feed
            // HC_WRITE) and copies its `pf.out` row in place of the mixer.
            if hc_chunk {
                self.record_hc_chunk(b, dev, t, pos0)?;
            } else {
                if self.mixer_chunked(b, dev) {
                    self.record_mixer_chunk(b, dev, t, pos0, false)?;
                    self.mixer_from_chunk = Some(pos0);
                }
                self.shared_deferred = shared_chunk;
                for i in 0..t {
                    self.swap_r_in(dev, i);
                    let prelude = self.record_dense_prelude(b, dev, pos0 + i, &mut no_trace);
                    if let Err(e) = prelude {
                        self.shared_deferred = false;
                        self.mixer_from_chunk = None;
                        return Err(e);
                    }
                    self.spec_row_state(b, i, t)?;
                    {
                        let v = &self.venues[dev];
                        let pf = v.pf.as_ref().expect("the merged prefill scratch");
                        let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
                        let kb = (k * 4) as u64;
                        let nb = (n_embd * 4) as u64;
                        let inject = (
                            &v.inject_slots[i],
                            0,
                            &ws.inject,
                            0,
                            ((mc.hc * v.hc.inject_chunks() * 4) as u64).min(ws.inject.size),
                        );
                        // Six copies, one region array (and one barrier) per token. The
                        // ffn `HC_WRITE` moves to pass 2, so this token's inject partials
                        // (the last region) have to survive the rest of the chunk.
                        let mut regions = vec![
                            (&pf.ids.buf, i as u64 * kb, &v.router_ids, 0, kb),
                            (&pf.w, i as u64 * kb, &v.router_w, 0, kb),
                            (&pf.mixed, i as u64 * nb, &ws.mixed, 0, nb),
                            (&pf.gate, i as u64 * 4, &v.sh_gate_logit, 0, 4),
                            inject,
                        ];
                        if !shared_chunk {
                            regions.push((&pf.shared, i as u64 * nb, &v.sh, 0, nb));
                        }
                        batch_cmd::record_copies(gpu, &regions)?;
                    }
                    self.swap_r_in(dev, i);
                    if batch_cmd::recorded() >= MERGED_BATCH_CAP {
                        if overlap {
                            self.reopen_batch(dev)?;
                        } else {
                            self.flush()?;
                            self.ensure_batch(dev)?;
                        }
                    }
                }
                self.shared_deferred = false;
                self.mixer_from_chunk = None;
            }
            if shared_chunk && !defer_shared {
                self.record_shared_chunk(b, dev, t)?;
            }
            batch_cmd::record_barrier_to_host(gpu)?;
            self.submit_batch()?;
            self.flush()?;
        }

        // ---- the bucket list: per distinct expert, its users -----------------
        // A micro-batch tick starts with both cards idle: the previous half-step's
        // MoE result goes up to the block card before anything else is shipped
        // (the imported path's host region is shared with the ship below).
        if micro && !micro_ovl() {
            self.micro_up_write(dev)?;
        }
        // Overlapped tick: the helper's inputs start their way off the block card now
        // and the host buckets while the DMA runs.
        let ship_imp = xdev.is_some_and(|x| self.ship_ready(dev, x));
        // A micro tick's `pf.shared` is only read at fan-in: it is copied down at the head
        // of the launch segment and shipped up under the routed GEMMs.
        let sdefer = micro && micro_ovl() && micro_sh() && ship_imp;
        let mut down_pending = None;
        if let (true, Some(x)) = (micro && micro_ovl() && ship_imp, xdev) {
            let (down, _) = self.ship_lists(dev, x, t, k, defer_shared || sdefer);
            down_pending = Some(crate::xfer::copy_submit(gpu, &down)?);
        }
        let bucket_t0 = std::time::Instant::now();
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
                let g_end = (u + pf_group_users()).min(end);
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
        let bucket_s = bucket_t0.elapsed().as_secs_f64();

        let mut ship_s = 0.0f64;
        let mut ship_p = [0.0f64; 4];
        if let Some(x) = xdev.filter(|_| ship_imp) {
            let t0 = std::time::Instant::now();
            let xpf = self.venues[x].pf.as_ref().expect("checked above");
            let (down, up) = self.ship_lists(dev, x, t, k, defer_shared || sdefer);
            ship_p[0] = match down_pending.take() {
                Some(p) => p.wait(gpu)?,
                None => crate::xfer::copy_fenced(gpu, &down)?,
            };
            ship_p[1] = crate::xfer::copy_fenced(m.devices[x].gpu(), &up)?;
            xpf.slots.write(&slot_bytes)?;
            ship_s += t0.elapsed().as_secs_f64();
        } else if let Some(x) = xdev {
            let t0 = std::time::Instant::now();
            let nb = n_embd * 4;
            let pf = self.venues[dev].pf.as_ref().expect("the merged prefill scratch");
            let xpf = self.venues[x].pf.as_ref().expect("checked above");
            let pairs: Vec<(&Buf, usize, &Buf)> = [
                (&pf.mixed, t * nb, &xpf.mixed),
                (&pf.w, t * k * 4, &xpf.w),
                (&pf.shared, t * nb, &xpf.shared),
                (&pf.gate, t * 4, &xpf.gate),
            ]
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| !(defer_shared && i == 2))
            .map(|(_, p)| p)
            .collect();
            let items: Vec<(&Buf, u64, usize)> = pairs.iter().map(|&(s, n, _)| (s, 0, n)).collect();
            let bytes = self.stagings[dev].read_batch(gpu, &items)?;
            ship_p[0] = t0.elapsed().as_secs_f64();
            let regions: Vec<(&Buf, u64, &[u8])> = pairs
                .iter()
                .zip(bytes.iter())
                .map(|(&(_, _, buf), d)| (buf, 0, d.as_slice()))
                .collect();
            ship_p[1] = self.stagings[x].write_regions(m.devices[x].gpu(), &regions)?;
            xpf.slots.write(&slot_bytes)?;
            ship_s += t0.elapsed().as_secs_f64();
        }
        // The device the expert GEMMs and the fan-in run on.
        let (edev, egpu) = match xdev {
            Some(x) => (x, m.devices[x].gpu()),
            None => (dev, gpu),
        };
        let overlap = self.pf[edev][PF_OVERLAP];

        if defer_shared {
            // Submitted without a wait; the expert device's batch opens at once.
            // Every later flush waits for it, before `pf.shared` is shipped.
            self.ensure_batch(dev)?;
            self.record_shared_chunk(b, dev, t)?;
            batch_cmd::record_barrier_to_host(gpu)?;
            self.reopen_batch(edev)?;
        }
        if micro {
            if sdefer {
                let (nb, kb) = ((t * n_embd * 4) as u64, (t * k * 4) as u64);
                self.micro_sh = Some((nb + kb, nb));
            }
            self.micro_launch(edev, bound_t0)?;
        }

        // ---- the expert GEMMs: three per (distinct expert, group of <= 8) -----
        self.timers.mark(EXPERTS);
        self.ensure_batch(edev)?;
        let t_gemm = std::time::Instant::now();
        // Streamed blocks walk the distinct experts in slot order; one submit per
        // STREAM_SLOTS experts bounds the host-visible staging (and is the ring
        // wait this pass reports).
        let mut expert_index = 0usize;
        // Fused gate + up + SwiGLU per streamed group (`BRECCIA_PF_ROUTED_PAIR=0` opts out).
        let (gx, ux) = (wgate as u64 * 2, wup as u64 * 2);
        let pair = streamed
            && routed_pair()
            && gate_ty == up_ty
            && gx.is_multiple_of(4)
            && self.venues[edev].pipes.gemm(gate_ty).is_some_and(|g| g.has_pair())
            && m.layers[b].experts.gate.bytes() as u64 == gx
            && m.layers[b].experts.up.bytes() as u64 == ux;
        let mut prev_expert: Option<usize> = None;
        let mut stream_prev: Option<breccia_kernels::batch_cmd::Segment> = None;
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
                    if overlap {
                        // Double buffer: the half about to be refilled was read by the
                        // batch before this one, so wait for that one only.
                        let seg = self.reopen_batch(edev)?;
                        if let Some(prev) = std::mem::replace(&mut stream_prev, seg) {
                            let gpus = self.device_gpus();
                            batch_cmd::wait(&gpus, prev)
                                .map_err(|e| anyhow!("waiting for a streamed-expert half: {e}"))?;
                        }
                    } else {
                        self.submit_batch()?;
                        self.flush()?;
                        self.ensure_batch(edev)?;
                    }
                    self.pf_stream_wait_s += t0.elapsed().as_secs_f64();
                }
                self.fill_stream_slot(edev, b, *e, expert_index % (2 * STREAM_SLOTS))?;
                expert_index += 1;
            } else if !streamed {
                prev_expert = Some(*e);
            }
            {
                let v = &self.venues[edev];
                let pf = v.pf.as_ref().expect("the merged prefill scratch");
                let pipes = &v.pipes;
                let gate = pipes.gemm(gate_ty).expect("the gate multi-user GEMM");
                let up = pipes.gemm(up_ty).expect("the up multi-user GEMM");
                let down = pipes.gemm(down_ty).expect("the down multi-user GEMM");
                if streamed && new_expert && expert_index > 0 {
                    let slot = (expert_index - 1) % (2 * STREAM_SLOTS);
                    // slot -> device, three region groups, one barrier.
                    let off = slot as u64 * pf.slot_stride;
                    let down_copy = (
                        &pf.sf_down,
                        0,
                        &pf.stream.buf,
                        off + pf.g_bytes + pf.u_bytes,
                        pf.d_bytes,
                    );
                    if pair {
                        batch_cmd::record_copies(
                            egpu,
                            &[
                                (&pf.sf_gu, 0, &pf.stream.buf, off, gx),
                                (&pf.sf_gu, gx, &pf.stream.buf, off + pf.g_bytes, ux),
                                down_copy,
                            ],
                        )?;
                    } else {
                        batch_cmd::record_copies(
                            egpu,
                            &[
                                (&pf.sf_gate, 0, &pf.stream.buf, off, pf.g_bytes),
                                (&pf.sf_up, 0, &pf.stream.buf, off + pf.g_bytes, pf.u_bytes),
                                down_copy,
                            ],
                        )?;
                    }
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
                batch_cmd::record_copies(egpu, &rows)?;
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
                if pair {
                    gate.run_pair_at(&pf.sf_gu, &pf.x, &pf.act, n_ff, n_embd, mu, 0)?;
                } else {
                    // `gate` and `up` read the same gather and write pf.gg / pf.uu:
                    // drop gate's trailing fence (§8z).
                    batch_cmd::skip_next_barrier();
                    gate.run_at(gbuf, &pf.x, &pf.gg, n_ff, n_embd, mu, gb)?;
                    up.run_at(ubuf, &pf.x, &pf.uu, n_ff, n_embd, mu, ub)?;
                    v.swiglu.run(&pf.gg, &pf.uu, &pf.act, mu * n_ff)?;
                }
                down.run_at(dbuf, &pf.act, &pf.down, n_embd, n_ff, mu, db)?;
                batch_cmd::record_copy(
                    egpu,
                    &pf.scratch,
                    (*base as u64) * (n_embd * 4) as u64,
                    &pf.down,
                    0,
                    (mu * n_embd * 4) as u64,
                )?;
            }
            if batch_cmd::recorded() >= MERGED_BATCH_CAP {
                self.flush()?;
                self.ensure_batch(edev)?;
            }
        }
        self.submit_batch()?;
        self.flush()?;
        let gemm_s = t_gemm.elapsed().as_secs_f64();
        if let Some(x) = xdev.filter(|_| defer_shared) {
            let t0 = std::time::Instant::now();
            let n = (t * n_embd * 4) as u64;
            let pf = self.venues[dev].pf.as_ref().expect("the merged prefill scratch");
            let xpf = self.venues[x].pf.as_ref().expect("checked above");
            if ship_imp {
                // The offset the ship above left for it: after `mixed` and `w`.
                let at = n + (t * k * 4) as u64;
                let imp = self.ship_imp.as_ref().expect("ship_ready");
                ship_p[0] += crate::xfer::copy_fenced(gpu, &[(&pf.shared, imp.get(dev), 0, at, n)])?;
                ship_p[1] += crate::xfer::copy_fenced(m.devices[x].gpu(), &[(imp.get(x), &xpf.shared, at, 0, n)])?;
            } else {
                let bytes = self.stagings[dev].read_batch(gpu, &[(&pf.shared, 0, n as usize)])?;
                ship_p[1] += self.stagings[x].write_regions(m.devices[x].gpu(), &[(&xpf.shared, 0, bytes[0].as_slice())])?;
            }
            ship_s += t0.elapsed().as_secs_f64();
        }

        // The deferred `pf.shared` (copied down ahead of the launch segment) goes up now;
        // queue order on the helper puts it ahead of the fan-in.
        let mut sh_pending = None;
        if let (true, Some(x)) = (sdefer, xdev) {
            self.micro_upw_wait()?;
            let n = (t * n_embd * 4) as u64;
            let at = n + (t * k * 4) as u64;
            let xpf = self.venues[x].pf.as_ref().expect("checked above");
            let imp = self.ship_imp.as_ref().expect("ship_ready");
            sh_pending = Some(crate::xfer::copy_submit(egpu, &[(imp.get(x), &xpf.shared, at, 0, n)])?);
        }

        // ---- the fan-in: one dispatch, then the chunk's MoE result per token --
        self.ensure_batch(edev)?;
        let t_fan = std::time::Instant::now();
        {
            let v = &self.venues[edev];
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
        }
        if let Some(x) = xdev {
            // The MoE result back into the block device's `pf.out` (pass 2 reads it).
            // A micro-batch tick only reads it off the helper: the block card is busy
            // and takes it at the next tick boundary (`micro_up_write`).
            self.submit_batch()?;
            self.flush()?;
            if let Some(p) = sh_pending.take() {
                ship_p[1] += p.wait(egpu)?;
            }
            // The previous result's copy onto the block card reads the same import range.
            self.micro_upw_wait()?;
            let t0 = std::time::Instant::now();
            let xpf = self.venues[x].pf.as_ref().expect("checked above");
            let pf = self.venues[dev].pf.as_ref().expect("the merged prefill scratch");
            if ship_imp {
                let imp = self.ship_imp.as_ref().expect("ship_ready");
                let n = (t * n_embd * 4) as u64;
                let ro = if micro && micro_ovl() { self.ship_res_off } else { 0 };
                ship_p[2] = crate::xfer::copy_fenced(egpu, &[(&xpf.out, imp.get(x), 0, ro, n)])?;
                if micro {
                    self.micro_up = Some(MicroUp::Imported(n));
                } else {
                    ship_p[3] = crate::xfer::copy_fenced(gpu, &[(imp.get(dev), &pf.out, 0, 0, n)])?;
                }
            } else {
                let bytes = self.stagings[x].read_batch(egpu, &[(&xpf.out, 0, t * n_embd * 4)])?;
                ship_p[2] = t0.elapsed().as_secs_f64();
                if micro {
                    self.micro_up = bytes.into_iter().next().map(MicroUp::Bytes);
                } else {
                    ship_p[3] = self.stagings[dev].write_regions(gpu, &[(&pf.out, 0, bytes[0].as_slice())])?;
                }
            }
            ship_s += t0.elapsed().as_secs_f64();
            if !micro {
                self.ensure_batch(dev)?;
            }
        }
        if !micro {
            {
                let v = &self.venues[dev];
                let pf = v.pf.as_ref().expect("the merged prefill scratch");
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
        }
        self.submit_batch()?;
        self.flush()?;
        let fanin_s = t_fan.elapsed().as_secs_f64();

        // ---- pass 2: the ffn HC_WRITE for every token, one submission --------
        if !micro {
            self.timers.mark(HC_WRITE_F);
            self.ensure_batch(dev)?;
            if hc_chunk {
                // One T-token HC_WRITE (the ffn inject partials are still in the chunk
                // workspace), then the residual back into the per-token slots.
                let v = &self.venues[dev];
                let rc = v.r_chunk.as_ref().expect("checked above");
                let ws = self
                    .hc_chunk_ws[dev]
                    .iter()
                    .find(|w| w.tokens == t)
                    .expect("made in pass 1");
                let pf = v.pf.as_ref().expect("the merged prefill scratch");
                v.hc.write(ws, rc, &pf.out, rc)?;
                let hb = (mc.hc_dim() * 4) as u64;
                let scatter: Vec<(&Buf, u64, &Buf, u64, u64)> = (0..t)
                    .map(|i| (&v.r_slots[i], 0, rc, i as u64 * hb, hb))
                    .collect();
                batch_cmd::record_copies(gpu, &scatter)?;
            }
            for i in (0..t).filter(|_| !hc_chunk) {
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
        }

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
                "merged blk {b:>2} t={t} {}{} distinct={distinct} groups={} slots={} \
                 gemm={:.2} ms fanin={:.3} ms ship={:.2} ms (rd {:.2} wr {:.2} | back rd {:.2} wr {:.2}) dispatches={dispatches} \
                 arena={:.1} MiB streamed(cum)={:.1} MiB fill={:.1} ms wait={:.1} ms bucket={:.3} ms",
                if streamed { "streamed" } else { "resident" },
                if xdev.is_some() { "@x" } else { "" },
                groups.len(),
                t * k,
                gemm_s * 1e3,
                fanin_s * 1e3,
                ship_s * 1e3,
                ship_p[0] * 1e3,
                ship_p[1] * 1e3,
                ship_p[2] * 1e3,
                ship_p[3] * 1e3,
                (distinct as u64 * expert_bytes) as f64 / (1u64 << 20) as f64,
                sb as f64 / (1u64 << 20) as f64,
                sf * 1e3,
                sw * 1e3,
                bucket_s * 1e3
            );
        }
        Ok(())
    }

    // ---- BRECCIA_PF_MICRO=2: the two-micro-batch pipeline ---------------------------
    //
    // Half-steps n = 2b + h (block b, token half h). A(n) is the block card's pass 1
    // (dense chain, router, chunk shared expert), X(n) the helper's routed experts and
    // fan-in, B(n) the block card's ffn HC_WRITE. One tick runs X(n) on the helper
    // while the block card runs B(n-1) then A(n+1) as one submitted segment:
    //   A(b+1,h) needs B(b,h)      -> same segment, in order;
    //   A(b,1)   needs A(b,0)'s KV -> an earlier tick (every tick ends with both cards idle).
    // Both cards are idle at every tick boundary, so `pf.*`, the expert stream slots
    // and the QSA/GDN chunk workspaces each have one producer and one consumer at a
    // time; only `r_chunk` and the hc workspaces (which live from A(n) to B(n)) are
    // doubled, and `r_slots` is windowed per half. Host-staged copies only.

    /// Swap half `h`'s residual chunk, hc workspaces and `r_slots` / token window
    /// into place (`enter`) or back out. Always call it a second time with `false`.
    fn micro_ctx(&mut self, dev: usize, h: usize, th: usize, enter: bool) -> Result<()> {
        if enter && self.micro_rc[h].is_none() {
            let gpu = self.model.devices[dev].gpu();
            let bytes = self.venues[dev].r_chunk.as_ref().map_or(0, |b| b.size) / 2;
            self.micro_rc[h] = Some(
                breccia_kernels::gemv::alloc(gpu, bytes)
                    .context("allocating the micro-batch residual chunk")?,
            );
        }
        std::mem::swap(&mut self.venues[dev].r_chunk, &mut self.micro_rc[h]);
        std::mem::swap(&mut self.hc_chunk_ws[dev], &mut self.micro_hcws[h]);
        if h == 1 {
            if enter {
                self.venues[dev].r_slots.rotate_left(th);
                self.pf_toks.rotate_left(th);
            } else {
                self.venues[dev].r_slots.rotate_right(th);
                self.pf_toks.rotate_right(th);
            }
        }
        Ok(())
    }

    /// Record A(n) into the open batch (half context already in place).
    fn micro_record_a(&mut self, n: usize, th: usize, pos0: usize) -> Result<()> {
        let (b, h) = (n / 2, n % 2);
        let dev = self.model.device_of(b);
        self.record_hc_chunk(b, dev, th, pos0 + h * th)?;
        self.record_shared_chunk(b, dev, th)
    }

    /// Record B: the ffn `HC_WRITE` over `pf.out` and the residual back into the
    /// half's `r_slots` window (the unfused pass 2 of `prefill_block_merged`).
    fn micro_record_b(&mut self, dev: usize, th: usize) -> Result<()> {
        let m = self.model;
        let gpu = m.devices[dev].gpu();
        let v = &self.venues[dev];
        let rc = v.r_chunk.as_ref().expect("checked by micro_blocker");
        let ws = self.hc_chunk_ws[dev]
            .iter()
            .find(|w| w.tokens == th)
            .expect("made by A of the same half");
        let pf = v.pf.as_ref().expect("the merged prefill scratch");
        v.hc.write(ws, rc, &pf.out, rc)?;
        let hb = (m.cfg.hc_dim() * 4) as u64;
        let scatter: Vec<(&Buf, u64, &Buf, u64, u64)> = (0..th)
            .map(|i| (&v.r_slots[i], 0, rc, i as u64 * hb, hb))
            .collect();
        batch_cmd::record_copies(gpu, &scatter)
    }

    /// Wait for the in-flight [`Session::micro_up_write`] copy, if any.
    fn micro_upw_wait(&mut self) -> Result<()> {
        if let Some(p) = self.micro_upw.take() {
            let gpu = self.model.devices[self.model.device_of(0)].gpu();
            p.wait(gpu)?;
        }
        Ok(())
    }

    /// The previous half-step's MoE result onto the block card's `pf.out` (both cards
    /// idle, nothing open on the block card).
    fn micro_up_write(&mut self, dev: usize) -> Result<()> {
        let m = self.model;
        let gpu = m.devices[dev].gpu();
        let sh = if micro_ovl() { self.micro_sh.take() } else { None };
        let up = self.micro_up.take();
        if sh.is_none() && up.is_none() {
            return Ok(());
        }
        let pf = self.venues[dev].pf.as_ref().expect("the merged prefill scratch");
        let mut cps: Vec<ShipCopy<'_>> = Vec::new();
        match up {
            Some(MicroUp::Bytes(b)) => {
                self.stagings[dev].write_regions(gpu, &[(&pf.out, 0, b.as_slice())])?;
            }
            Some(MicroUp::Imported(n)) => {
                let imp = self.ship_imp.as_ref().expect("ship_ready");
                if micro_ovl() {
                    cps.push((imp.get(dev), &pf.out, self.ship_res_off, 0, n));
                } else {
                    crate::xfer::copy_fenced(gpu, &[(imp.get(dev), &pf.out, 0, 0, n)])?;
                }
            }
            None => {}
        }
        if let Some((at, n)) = sh {
            let imp = self.ship_imp.as_ref().expect("ship_ready");
            cps.push((&pf.shared, imp.get(dev), 0, at, n));
        }
        if !cps.is_empty() {
            // Submitted without a wait: queue order puts it ahead of B(n-1) (the copy's
            // barriers span the queue), and the host waits for it only before it reads
            // the import range again.
            self.micro_upw = Some(crate::xfer::copy_submit(gpu, &cps)?);
        }
        Ok(())
    }

    /// Submit B(n-1) then A(n+1) on the block card without waiting, park the segment
    /// in the tick, and scope waits to the helper until the tick ends.
    fn micro_launch(&mut self, edev: usize, bound_t0: std::time::Instant) -> Result<()> {
        let t0 = std::time::Instant::now();
        let (n, th, pos0) = {
            let tk = self.micro.as_mut().expect("a micro tick");
            tk.bound_s = bound_t0.elapsed().as_secs_f64();
            (tk.n, tk.th, tk.pos0)
        };
        let m = self.model;
        let steps = 2 * m.cfg.n_layers;
        let dev = m.device_of(0);
        // B(n-1) and A(n+1) are the same half.
        let h = (n + 1) % 2;
        let (have_b, have_a) = (n >= 1, n + 1 < steps);
        if micro_ovl() {
            self.micro_up_write(dev)?;
        }
        if have_b || have_a {
            self.micro_ctx(dev, h, th, true)?;
            let r = self.micro_launch_inner(n, dev, th, pos0, have_b, have_a);
            self.micro_ctx(dev, h, th, false)?;
            r?;
        }
        let seg = self.pending_seg.take();
        self.pending_dev = None;
        self.wait_scope = Some(edev);
        let tk = self.micro.as_mut().expect("a micro tick");
        tk.seg = seg;
        tk.launch_s = t0.elapsed().as_secs_f64();
        tk.launch_end = std::time::Instant::now();
        Ok(())
    }

    fn micro_launch_inner(
        &mut self,
        n: usize,
        dev: usize,
        th: usize,
        pos0: usize,
        have_b: bool,
        have_a: bool,
    ) -> Result<()> {
        let m = self.model;
        let gpu = m.devices[dev].gpu();
        self.ensure_batch(dev)?;
        if have_b {
            self.micro_record_b(dev, th)?;
        }
        if have_a {
            let b_a = (n + 1) / 2;
            if m.cfg.is_ple_layer(b_a) {
                // PLE is a host round trip over this half's residuals: drain the card
                // (the helper has not started this tick's experts yet), run it, go on.
                self.submit_batch()?;
                self.flush()?;
                self.micro_upw_wait()?;
                self.ple_chunk(b_a, th)?;
                self.ensure_batch(dev)?;
            }
            self.micro_record_a(n + 1, th, pos0)?;
        }
        batch_cmd::record_barrier_to_host(gpu)?;
        self.submit_batch()
    }

    /// Why `BRECCIA_PF_MICRO=2` cannot run this chunk (`None`: it can).
    fn micro_blocker(&self, t: usize, pos0: usize) -> Option<String> {
        let m = self.model;
        let mc = &m.cfg;
        let dev = m.device_of(0);
        if !self.prefill_merged {
            return Some("--prefill-merged is off".into());
        }
        if t % 2 != 0 || (t / 2) % 4 != 0 || t < 8 || pos0 % 4 != 0 {
            return Some(format!("chunk t={t} pos0={pos0} does not split into two multiples of 4"));
        }
        if self.venues[dev].r_chunk.is_none() || self.venues[dev].pf.is_none() {
            return Some("no chunk residual / merged scratch".into());
        }
        if (0..mc.n_layers).any(|b| m.device_of(b) != dev) {
            return Some("the blocks span more than one card".into());
        }
        if mc.is_ple_layer(0) {
            return Some("block 0 is a PLE layer".into());
        }
        if (0..mc.n_layers).any(|b| mc.is_ple_layer(b))
            && (self.venues[dev].ple_emb_chunk.is_none()
                || std::env::var_os("BRECCIA_NO_PLE_BATCH").is_some())
        {
            return Some("PLE layers without the batched PLE phase".into());
        }
        for b in 0..mc.n_layers {
            let p = self.merged_plan(b, dev, m.placed.get(&b));
            let Some(x) = p.xdev else {
                return Some(format!("block {b} has no expert device (BRECCIA_PF_EXPERT_DEVICE)"));
            };
            if !(p.streamed && p.shared_chunk && p.hc_chunk) {
                return Some(format!("block {b} lacks the streamed / chunked paths"));
            }
            let ready = |d: usize| {
                [p.gate_ty, p.up_ty, p.down_ty]
                    .iter()
                    .all(|ty| self.venues[d].pipes.gemm(*ty).is_some())
            };
            if !ready(dev) || !ready(x) {
                return Some(format!("block {b}: a role type has no multi-user GEMM"));
            }
        }
        None
    }

    /// The block loop of [`Session::prefill`] as a two-half software pipeline.
    fn prefill_micro(&mut self, t: usize, pos0: usize) -> Result<()> {
        let r = self.prefill_micro_run(t, pos0);
        self.wait_scope = None;
        self.micro_up = None;
        let _ = self.micro_upw_wait();
        if let Some(tk) = self.micro.take() {
            if let Some(seg) = tk.seg {
                self.pending_seg = Some(seg);
                self.pending_dev = Some(self.model.device_of(0));
            }
            let _ = self.flush();
        }
        r
    }

    fn prefill_micro_run(&mut self, t: usize, pos0: usize) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let dev = m.device_of(0);
        let gpu = m.devices[dev].gpu();
        let th = t / 2;
        let steps = 2 * mc.n_layers;
        let trace = std::env::var_os("BRECCIA_TRACE_MERGED").is_some();

        // Prologue: A(0), alone.
        self.timers.mark(HC_READ_A);
        self.micro_ctx(dev, 0, th, true)?;
        let r = (|| {
            self.ensure_batch(dev)?;
            self.micro_record_a(0, th, pos0)?;
            batch_cmd::record_barrier_to_host(gpu)?;
            self.submit_batch()?;
            self.flush()
        })();
        self.micro_ctx(dev, 0, th, false)?;
        r?;

        let (mut boundary, mut launch, mut rx, mut wait) = (0f64, 0f64, 0f64, 0f64);
        for n in 0..steps {
            let (b, h) = (n / 2, n % 2);
            self.micro = Some(MicroTick {
                n,
                th,
                pos0,
                seg: None,
                start: std::time::Instant::now(),
                launch_end: std::time::Instant::now(),
                bound_s: 0.0,
                launch_s: 0.0,
            });
            self.prefill_block_merged(b, th, pos0 + h * th, m.placed.get(&b))?;
            let rx_end = std::time::Instant::now();
            let tk = self.micro.take().expect("a micro tick");
            self.wait_scope = None;
            self.pending_seg = tk.seg;
            self.pending_dev = tk.seg.map(|_| dev);
            let t_w = std::time::Instant::now();
            self.flush()?;
            // Host blocked on the block card after the helper finished: the helper idled.
            let w = t_w.elapsed().as_secs_f64();
            let rx_s = rx_end.duration_since(tk.launch_end).as_secs_f64();
            if trace {
                eprintln!(
                    "micro n={n:>2} blk {b:>2} half {h} boundary={:.2} ms (both idle) launch={:.2} ms \
                     rx={:.2} ms mi50_wait={:.2} ms (rx idle) tick={:.2} ms",
                    tk.bound_s * 1e3,
                    tk.launch_s * 1e3,
                    rx_s * 1e3,
                    w * 1e3,
                    tk.start.elapsed().as_secs_f64() * 1e3,
                );
            }
            boundary += tk.bound_s;
            launch += tk.launch_s;
            rx += rx_s;
            wait += w;
        }

        // Epilogue: the last half's MoE result, then B(2L-1), alone.
        self.micro_up_write(dev)?;
        self.micro_ctx(dev, 1, th, true)?;
        let r = (|| {
            self.ensure_batch(dev)?;
            self.micro_record_b(dev, th)?;
            self.submit_batch()?;
            self.flush()
        })();
        self.micro_ctx(dev, 1, th, false)?;
        self.micro_upw_wait()?;
        r?;
        if trace {
            // Exact MI50 busy time needs GPU timestamps (BRECCIA_GPU_TIME); here: the
            // card idles at least the boundaries, and the helper idles `mi50_wait`.
            eprintln!(
                "micro total t={t} ticks={steps}: mi50 idle >= boundary {:.1} ms | launch {:.1} ms | \
                 rx {:.1} ms | rx idle (mi50_wait) {:.1} ms",
                boundary * 1e3,
                launch * 1e3,
                rx * 1e3,
                wait * 1e3,
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
        // The segment `submit_block` created for **this** block (None for a block that
        // submitted nothing, which cannot have an in-batch readback either).
        let mine = self.block_seg.take();
        let had_segment = mine.is_some() && mine == self.pending_seg;
        // **Deferred wait** (§8bh). A block whose experts are resident reads nothing
        // back: its `HC_WRITE` is recorded into the next submission on the *same*
        // queue, so queue order already guarantees the dependency and this wait is
        // pure host latency. Leaving the segment in flight is also what lets the host
        // record block b+1 while block b executes — the overlap a single stream needs
        // to stop being `lane + GPU`. `flush()`/`wait_all` and the ring's own
        // back-pressure (`begin` waits a slot before reuse) collect it; the *traced*
        // path reads GPU buffers back, so it keeps its wait.
        let defer = m.placed.contains_key(&b) && trace.is_none() && defer_placed_waits();
        if defer {
            self.deferred_waits += 1;
        } else if let Some(seg) = self.pending_seg.take() {
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
            // `read_block` is a bare slice of the mapping: it is correct only because
            // the five copies that fill this region were recorded into the batch whose
            // fence was just waited. Count the case where that did not happen — it is
            // the only way a *stale* (previous-token) readback can reach the lane.
            if !had_segment {
                self.readbacks_unfenced += 1;
            }
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
        if std::env::var_os("BRECCIA_FAULT_POISON_LOGITS").is_some()
            && self.logits_unwritten.is_none()
        {
            let pat = Self::logits_sentinel().to_bits();
            let names = ["mixed", "router_ids", "router_w", "sh", "sh_gate_logit"];
            let mut bad = Vec::new();
            for (name, bytes) in names.iter().zip(raw.iter()) {
                let left = bytes
                    .chunks_exact(4)
                    .filter(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) == pat)
                    .count();
                if left > 0 {
                    bad.push(format!("{name}: {left} of {} entries", bytes.len() / 4));
                }
            }
            if bad.is_empty() {
                eprintln!(
                    "breccia: lane scratch coverage OK: mixed, router ids/weights, sh and its \
                     gate were all written on the first readback"
                );
            } else {
                eprintln!(
                    "breccia: FAULT INJECTION: lane scratch never written on the first \
                     readback — {}",
                    bad.join(", ")
                );
            }
        }
        let mixed = f32_from_bytes(&raw[0]);
        let ids: Vec<usize> = raw[1]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let weights = f32_from_bytes(&raw[2]);
        // A2 decode-side route rows: same sidecar, `d`-marked (append_route_row doc).
        if let Some(path) = std::env::var_os("BRECCIA_TRACE_ROUTE_FILE") {
            append_route_row(
                &std::path::PathBuf::from(path),
                true,
                self.pos,
                b,
                &ids,
                &weights,
            )?;
        }
        let shared = f32_from_bytes(&raw[3]);
        let g = f32::from_le_bytes([raw[4][0], raw[4][1], raw[4][2], raw[4][3]]);
        let gate = 1.0 / (1.0 + (-g).exp());

        // ---- host expert lane -------------------------------------------------
        self.timers.mark(EXPERTS);
        let mut out = vec![0f32; mc.n_embd];
        let t_lane = std::time::Instant::now();
        let mut pool = self.pool.lock().expect("the expert pool's lock");
        let waited = if lane_stats_on() {
            t_lane.elapsed().as_secs_f64()
        } else {
            0.0
        };
        let bytes = pool
            .moe_topk(&m.layers[b].experts, &ids, &weights, &mixed, &mut out)
            .with_context(|| format!("the expert lane on block {b}"))?;
        drop(pool);
        self.lock_wait_s += waited;
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
        // The batch is opened *before* the upload so the host→device copy is
        // **recorded into it** instead of paying its own submit+fence (§8bg); the
        // batch's submit is what orders it against the `HC_WRITE` that reads it.
        self.ensure_batch(dev)?;
        self.venues[dev].block_out.write(&f32_to_bytes(&out))?;
        self.dev_up[dev] += (mc.n_embd * 4) as u64;

        // ---- (H) HC_WRITE(ffn): recorded, left open --------------------------
        //
        // It reads the mailbox written just above, so it must be *submitted after
        // that write*; the next block's `submit_block` records on top of it and
        // submits once. That is what keeps a block at one submission even though
        // the write happens on the host.
        self.timers.mark(HC_WRITE_F);
        self.probe_mark(dev, batch_cmd::probe::HC_WRITE);
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
        // The head device IS the last block's device (the crossing hand-off was
        // removed, §8aw), so the probe marks address the same device.
        let last = m.device_of(mc.n_layers - 1);
        self.probe_mark(last, batch_cmd::probe::FINAL_HC);
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
            self.probe_mark(head_dev, batch_cmd::probe::HEAD);
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
        if std::env::var_os("BRECCIA_FAULT_POISON_LOGITS").is_some()
            && self.logits_unwritten.is_none()
        {
            let pat = Self::logits_sentinel().to_bits();
            let values = f32_from_bytes(&raw[0]);
            let left = values.iter().filter(|v| v.to_bits() == pat).count();
            self.logits_unwritten = Some(left);
            if left > 0 {
                let first = values.iter().position(|v| v.to_bits() == pat).unwrap_or(0);
                eprintln!(
                    "breccia: FAULT INJECTION: {} of {} logits were never written (first at                      index {first}) — the head does not cover [0, vocab)",
                    left,
                    mc.vocab
                );
            } else {
                eprintln!(
                    "breccia: head coverage OK: all {} logits were written",
                    mc.vocab
                );
            }
        }
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
        let gpu = self.model.devices[dev].gpu();
        batch_cmd::probe::enter(gpu, batch_cmd::probe::MOE_MAP);
        let gate = pipes.batch(&pe.gate.key)?;
        v.moe_map
            .run(&v.router_ids, &v.moe_map_gate, gate, k, 0, false)?;
        batch_cmd::probe::enter(gpu, batch_cmd::probe::MOE_GATE);
        gate.run(&pe.gate.buf, mixed, &v.moe_gate, &v.moe_map_gate, k)?;
        batch_cmd::probe::enter(gpu, batch_cmd::probe::MOE_MAP);
        let up = pipes.batch(&pe.up.key)?;
        v.moe_map
            .run(&v.router_ids, &v.moe_map_up, up, k, 0, false)?;
        batch_cmd::probe::enter(gpu, batch_cmd::probe::MOE_UP);
        up.run(&pe.up.buf, mixed, &v.moe_up, &v.moe_map_up, k)?;
        batch_cmd::probe::enter(gpu, batch_cmd::probe::MOE_SWIGLU);
        v.swiglu.run(&v.moe_gate, &v.moe_up, &v.moe_act, k * n_ff)?;
        // down: one activation row per expert, so the map carries the stride
        batch_cmd::probe::enter(gpu, batch_cmd::probe::MOE_MAP);
        let down = pipes.batch(&pe.down.key)?;
        v.moe_map
            .run(&v.router_ids, &v.moe_map_down, down, k, 0, true)?;
        batch_cmd::probe::enter(gpu, batch_cmd::probe::MOE_DOWN);
        down.run(&pe.down.buf, &v.moe_act, &v.moe_down, &v.moe_map_down, k)?;
        batch_cmd::probe::enter(gpu, batch_cmd::probe::MOE_COMBINE);
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
            beta_alpha: if breccia_kernels::gdn::beta_alpha_merged() {
                Some(Proj::Single(v.pipes.get(&gw.beta_alpha.key)?))
            } else {
                None
            },
            w_beta_alpha: Some(&gw.beta_alpha.buf),
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

    /// `BRECCIA_PF_QSACHUNK` (QSA) / `BRECCIA_PF_GDNCHUNK` (GDN) and every
    /// projection the block's mixer runs has a multi-user GEMM.
    fn mixer_chunked(&self, b: usize, dev: usize) -> bool {
        // Rows-mode verify: the T-token mixer has no per-row state to snapshot.
        if self.spec_rows_active() {
            return false;
        }
        let lw = &self.model.layers[b];
        let ws: Vec<&crate::model::W> = match lw.kind {
            LayerKind::Qsa if self.pf[dev][PF_QSACHUNK] => {
                match lw.qsa.as_ref() {
                    Some(q) => vec![&q.q, &q.k, &q.v, &q.idx_q, &q.idx_k, &q.out],
                    None => return false,
                }
            }
            LayerKind::Gdn if self.pf[dev][PF_GDNCHUNK] => {
                match lw.gdn.as_ref() {
                    Some(g) => vec![&g.qkv, &g.z, &g.beta, &g.alpha, &g.beta_alpha, &g.out],
                    None => return false,
                }
            }
            _ => return false,
        };
        let pipes = &self.venues[dev].pipes;
        ws.iter().all(|w| pipes.gemm(w.key.ty).is_some())
    }

    /// Merged pass 1 with `BRECCIA_PF_HCCHUNK`: the dense chain as T-token
    /// dispatches over `r_chunk` (HC_READ(attn), the chunked mixer, HC_WRITE(attn),
    /// HC_READ(ffn), router, scalar shared gate) instead of ~20 dispatches per
    /// token. The ffn inject partials stay in the chunk workspace for pass 2.
    fn record_hc_chunk(&mut self, b: usize, dev: usize, t: usize, pos0: usize) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let gpu = m.devices[dev].gpu();
        let hb = (mc.hc_dim() * 4) as u64;
        let tb = t as u64 * (mc.n_embd * 4) as u64;
        let wi = match self.hc_chunk_ws[dev].iter().position(|w| w.tokens == t) {
            Some(i) => i,
            None => {
                // Keep at most two (a prompt's chunks have at most two sizes, ~47 MB each
                // at T=512): drain the GPU, then drop the oldest.
                if self.hc_chunk_ws[dev].len() >= 2 {
                    batch_cmd::record_barrier_to_host(gpu)?;
                    self.submit_batch()?;
                    self.flush()?;
                    self.ensure_batch(dev)?;
                    let old = self.hc_chunk_ws[dev].remove(0);
                    self.venues[dev].hc.release(gpu, old);
                }
                let ws = self.venues[dev].hc.workspace(gpu, t)?;
                self.hc_chunk_ws[dev].push(ws);
                self.hc_chunk_ws[dev].len() - 1
            }
        };
        self.timers.mark(HC_READ_A);
        {
            let v = &self.venues[dev];
            let rc = v.r_chunk.as_ref().expect("checked by the caller");
            let ws = &self.hc_chunk_ws[dev][wi];
            let pf = v.pf.as_ref().expect("the merged prefill scratch");
            let gather: Vec<(&Buf, u64, &Buf, u64, u64)> = (0..t)
                .map(|i| (rc, i as u64 * hb, &v.r_slots[i], 0, hb))
                .collect();
            batch_cmd::record_copies(gpu, &gather)?;
            hc_read_ws(v, ws, &m.layers[b].hc_attn, rc)?;
            batch_cmd::record_copies(gpu, &[(&pf.mixed, 0, &ws.mixed, 0, tb)])?;
        }
        self.record_mixer_chunk(b, dev, t, pos0, true)?;
        self.timers.mark(HC_WRITE_A);
        let v = &self.venues[dev];
        let rc = v.r_chunk.as_ref().expect("checked by the caller");
        let ws = &self.hc_chunk_ws[dev][wi];
        let pf = v.pf.as_ref().expect("the merged prefill scratch");
        batch_cmd::probe::enter(gpu, batch_cmd::probe::HC_WRITE);
        v.hc.write(ws, rc, &pf.out, rc)?;
        hc_read_ws(v, ws, &m.layers[b].hc_ffn, rc)?;
        batch_cmd::record_copies(gpu, &[(&pf.mixed, 0, &ws.mixed, 0, tb)])?;
        let moe = &m.layers[b].moe;
        let g = |w: &crate::model::W| v.pipes.gemm(w.key.ty).expect("checked by the caller");
        batch_cmd::probe::enter(gpu, batch_cmd::probe::ROUTER);
        g(&moe.router).run(&moe.router.buf, &pf.mixed, &pf.logits, mc.n_expert, mc.n_embd, t)?;
        v.router.run(&pf.logits, &pf.ids.buf, &pf.w, t)?;
        g(&moe.shexp_gate).run(&moe.shexp_gate.buf, &pf.mixed, &pf.gate, 1, mc.n_embd, t)
    }

    /// Merged pass 1: HC_READ(attn) per token into `pf.mixed`, then one
    /// `t`-token mixer forward (multi-user GEMM projections) into `pf.out`.
    /// With `hc_done` the caller already put the attn-mixed rows in `pf.mixed`.
    fn record_mixer_chunk(
        &mut self,
        b: usize,
        dev: usize,
        t: usize,
        pos0: usize,
        hc_done: bool,
    ) -> Result<()> {
        let m = self.model;
        let gpu = m.devices[dev].gpu();
        let nb = (m.cfg.n_embd * 4) as u64;
        let chunk = self.venues[dev]
            .pf
            .as_ref()
            .map_or(t, |pf| (pf.out.size / nb) as usize);
        self.timers.mark(HC_READ_A);
        for i in (0..t).filter(|_| !hc_done) {
            self.swap_r_in(dev, i);
            let v = &self.venues[dev];
            let res = hc_read(v, &m.layers[b].hc_attn, &v.r).and_then(|_| {
                let pf = v.pf.as_ref().expect("the merged prefill scratch");
                let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
                batch_cmd::record_copies(gpu, &[(&pf.mixed, i as u64 * nb, &ws.mixed, 0, nb)])
            });
            self.swap_r_in(dev, i);
            res?;
            if batch_cmd::recorded() >= MERGED_BATCH_CAP {
                if self.pf[dev][PF_OVERLAP] {
                    self.reopen_batch(dev)?;
                } else {
                    self.flush()?;
                    self.ensure_batch(dev)?;
                }
            }
        }
        self.timers.mark(MIXER);
        if m.layers[b].kind == LayerKind::Gdn {
            return self.record_gdn_chunk(b, dev, t, chunk);
        }
        let cap = self.qsa_state[b]
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} has no QSA state"))?
            .capacity_cells;
        if pos0 + t > cap {
            bail!("block {b}: a chunk ending at {} needs more than the cache's {cap} KV cells", pos0 + t);
        }
        if self.qsa_chunk_ws[dev].is_none() {
            let qsa = self.venues[dev]
                .qsa
                .as_ref()
                .ok_or_else(|| anyhow!("device {dev} has no QSA pipelines"))?;
            self.qsa_chunk_ws[dev] = Some(qsa.workspace(gpu, chunk, cap)?);
        }
        {
            let ws = self.qsa_chunk_ws[dev].as_mut().expect("made above");
            let n_blocks = (pos0 + t) / 4;
            ws.tokens = t;
            ws.n_blocks = n_blocks;
            ws.max_ids = n_blocks.min(2048 / 4) + 1;
            ws.slots = 4 * ws.max_ids;
        }
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
        let many = |w: &crate::model::W| -> Result<Proj<'_, '_>> {
            Ok(Proj::Many {
                pipe: v
                    .pipes
                    .gemm(w.key.ty)
                    .ok_or_else(|| anyhow!("no multi-user GEMM for {:?}", w.key.ty))?,
                rows: w.key.rows,
                cols: w.key.cols,
            })
        };
        let weights = QsaWeights {
            q_norm: &qw.q_norm,
            k_norm: &qw.k_norm,
            idx_q_norm: &qw.idx_q_norm,
            idx_k_norm: &qw.idx_k_norm,
        };
        let proj = QsaProjections {
            q: many(&qw.q)?,
            w_q: &qw.q.buf,
            k: many(&qw.k)?,
            w_k: &qw.k.buf,
            v: many(&qw.v)?,
            w_v: &qw.v.buf,
            idx_q: many(&qw.idx_q)?,
            w_idx_q: &qw.idx_q.buf,
            idx_k: many(&qw.idx_k)?,
            w_idx_k: &qw.idx_k.buf,
            out: many(&qw.out)?,
            w_out: &qw.out.buf,
        };
        let pf = v.pf.as_ref().expect("the merged prefill scratch");
        let ws = self.qsa_chunk_ws[dev].as_ref().expect("made above");
        qsa.forward(state, ws, &weights, &proj, &pf.mixed, &pf.out, pos0)?;
        let tap = self
            .spec
            .as_ref()
            .filter(|s| s.active && s.replay)
            .and_then(|s| s.idx_k_tap[b].as_ref());
        if let Some(tap) = tap {
            let len = (t * breccia_kernels::qsa::IDX_DIM * 4) as u64;
            batch_cmd::record_copy(gpu, tap, 0, &ws.idx_k, 0, len)?;
        }
        Ok(())
    }

    /// The GDN half of [`Session::record_mixer_chunk`]: one `t`-token forward.
    fn record_gdn_chunk(&mut self, b: usize, dev: usize, t: usize, chunk: usize) -> Result<()> {
        let m = self.model;
        let gpu = m.devices[dev].gpu();
        if self.gdn_chunk_ws[dev].is_none() {
            let gdn = self.venues[dev]
                .gdn
                .as_ref()
                .ok_or_else(|| anyhow!("device {dev} has no GDN pipelines"))?;
            self.gdn_chunk_ws[dev] = Some(gdn.workspace(gpu, chunk)?);
        }
        self.gdn_chunk_ws[dev].as_mut().expect("made above").tokens = t;
        let gw = m.layers[b]
            .gdn
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} is GDN but has no weights"))?;
        let v = &self.venues[dev];
        let many = |w: &crate::model::W| -> Result<Proj<'_, '_>> {
            Ok(Proj::Many {
                pipe: v
                    .pipes
                    .gemm(w.key.ty)
                    .ok_or_else(|| anyhow!("no multi-user GEMM for {:?}", w.key.ty))?,
                rows: w.key.rows,
                cols: w.key.cols,
            })
        };
        let proj = GdnProjections {
            qkv: many(&gw.qkv)?,
            w_qkv: &gw.qkv.buf,
            z: many(&gw.z)?,
            w_z: &gw.z.buf,
            beta: many(&gw.beta)?,
            w_beta: &gw.beta.buf,
            alpha: many(&gw.alpha)?,
            w_alpha: &gw.alpha.buf,
            beta_alpha: if breccia_kernels::gdn::beta_alpha_merged() {
                Some(many(&gw.beta_alpha)?)
            } else {
                None
            },
            w_beta_alpha: Some(&gw.beta_alpha.buf),
            out: many(&gw.out)?,
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
        let state = self.gdn_state[b]
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} has no GDN state"))?;
        let pf = v.pf.as_ref().expect("the merged prefill scratch");
        let ws = self.gdn_chunk_ws[dev].as_ref().expect("made above");
        let tap = self
            .spec
            .as_ref()
            .filter(|s| s.active && s.replay)
            .and_then(|s| s.gdn_tap[b].as_ref());
        gdn.forward_tapped(state, ws, &weights, &proj, &pf.mixed, &pf.out, tap)
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
        let ple_t0 = std::time::Instant::now();
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
        let ple_t1 = ple_t0.elapsed();

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
        let ple_t2 = ple_t0.elapsed();

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

        let ple_t3 = ple_t0.elapsed();
        // ---- 4. the host blocks (threaded, bit-equal to in sequence), then one writeback --------------
        let mut out = vec![0f32; t * hc_dim];
        {
            let ple = self
                .ple
                .as_mut()
                .ok_or_else(|| anyhow!("no PLE host state"))?;
            if self.spec.as_ref().is_some_and(|s| s.active) {
                // Verify keeps the history after each row for rollback, so it
                // runs the (bit-equal) in-sequence block, not `block_chunk`.
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
                    // Verify: the history "after i + 1 rows", for rollback.
                    if let Some(sp) = self.spec.as_mut().filter(|s| s.active && i + 1 < t) {
                        sp.ple_rows[i + 1] = ple.state().hist;
                    }
                }
            } else {
                let rs: Vec<Vec<f32>> = (0..t).map(|i| f32_from_bytes(&parts[3 * i])).collect();
                let keys: Vec<Vec<f32>> = (0..t).map(|i| f32_from_bytes(&parts[3 * i + 1])).collect();
                let vals: Vec<Vec<f32>> = (0..t).map(|i| f32_from_bytes(&parts[3 * i + 2])).collect();
                ple.block_chunk(&rs, &keys, &vals, &mut out)?;
            }
        }
        let ple_t4 = ple_t0.elapsed();
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
        if std::env::var_os("BRECCIA_TRACE_MERGED").is_some() {
            let e = ple_t0.elapsed();
            eprintln!(
                "ple_chunk b={b} t={t}: hash+prefetch+pack {:.1} ms, record+submit {:.1} ms, readback {:.1} ms, host block {:.1} ms, writeback {:.1} ms",
                ple_t1.as_secs_f64() * 1e3,
                (ple_t2 - ple_t1).as_secs_f64() * 1e3,
                (ple_t3 - ple_t2).as_secs_f64() * 1e3,
                (ple_t4 - ple_t3).as_secs_f64() * 1e3,
                (e - ple_t4).as_secs_f64() * 1e3
            );
        }
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
        // The GPU-time probe's window starts here too, so its table covers the
        // decode steps and not the prompt.
        self.snapshot_probe();
        // The prompt passes must not inflate the per-device figures. The submit/wait
        // window opens here too, so `submits/token` and `waited/token` cover the
        // decode steps and not the prompt (§8bg).
        batch_cmd::reset_record_seconds();
        breccia_vk::ctx::reset_submit_stats();
        self.reset_lane_stats();
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
            let mut next = self
                .sampler
                .sample(&logits, &self.sampling)
                .ok_or_else(|| anyhow!("the sampler produced no token"))?;
            // `BRECCIA_FAULT_DIVERGE_STEP=k`: a deliberate divergence at forward `k`, so
            // `scripts/head-race-capture.sh` can be exercised on demand (§8bd).
            if std::env::var("BRECCIA_FAULT_DIVERGE_STEP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                == Some(self.forwards)
            {
                eprintln!(
                    "breccia: FAULT INJECTION: diverging at forward {}",
                    self.forwards
                );
                next = (next + 1) % self.model.cfg.vocab.max(1) as u32;
            }
            out.push(next);
            if i + 1 == n {
                break;
            }
            let dbg = if i == 0 { trace.take() } else { None };
            logits = self.token(next, pos, dbg)?;
            self.dump_state(pos)?;
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
        if let Some(mut imp) = self.ship_imp.take() {
            imp.release(&gpus);
        }
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
        self.spec_release();
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
        for (dev, ws) in self.qsa_chunk_ws.iter_mut().enumerate() {
            if let (Some(ws), Some(qsa)) = (ws.take(), self.venues[dev].qsa.as_ref()) {
                qsa.release(m.devices[dev].gpu(), ws);
            }
        }
        for (dev, ws) in self.gdn_chunk_ws.iter_mut().enumerate() {
            if let (Some(ws), Some(gdn)) = (ws.take(), self.venues[dev].gdn.as_ref()) {
                gdn.release(m.devices[dev].gpu(), ws);
            }
        }
        for (dev, list) in self.hc_chunk_ws.iter_mut().enumerate() {
            for ws in list.drain(..) {
                self.venues[dev].hc.release(m.devices[dev].gpu(), ws);
            }
        }
        let d0 = m.device_of(0);
        for list in self.micro_hcws.iter_mut() {
            for ws in list.drain(..) {
                self.venues[d0].hc.release(m.devices[d0].gpu(), ws);
            }
        }
        for rc in self.micro_rc.iter().flatten() {
            m.devices[d0].gpu().destroy_buf(rc);
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

/// Bytes one hyper-connection `inject` partial needs: `n_embd` floats at the model's
/// `hc` count is *not* what the mixer reads — it reads one partial per low-rank row, so
/// the size comes from here and never from a literal. (A hard-coded `160` was the second
/// silent wrong-value bug of this class; `docs/phase-4/01-forward-pass.md` §8bc.)
fn hc_inject_bytes() -> u64 {
    160
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
pub(crate) fn append_route_row(
    path: &std::path::Path,
    decode: bool,
    token: usize,
    layer: usize,
    ids: &[usize],
    weights: &[f32],
) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening route trace {}", path.display()))?;
    if decode {
        write!(file, "d").with_context(|| format!("writing route trace {}", path.display()))?;
    }
    write!(file, "{token} {layer}")
        .with_context(|| format!("writing route trace {}", path.display()))?;
    for (&expert, &weight) in ids.iter().zip(weights) {
        write!(file, " {expert}:{weight:.9}")
            .with_context(|| format!("writing route trace {}", path.display()))?;
    }
    writeln!(file).with_context(|| format!("writing route trace {}", path.display()))?;
    Ok(())
}

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

/// `BRECCIA_PF_ROUTED_PAIR`: fused gate + up + SwiGLU for streamed routed experts
/// (default on, `0` = the separate gate / up / SwiGLU dispatches).
fn routed_pair() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRECCIA_PF_ROUTED_PAIR").as_deref() != Ok("0"))
}

/// `BRECCIA_PF_GROUP_USERS`: users per merged-prefill expert group (default 1024,
/// i.e. one group per distinct expert at chunk <= 1024; `USERS_MAX` = the old shape).
/// Past 8 the GEMM rides z slices, replacing serialized gather/GEMM/scatter rounds;
/// per-user arithmetic unchanged (1k logits byte-identical, prefill -7.6 %, §11l).
fn pf_group_users() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("BRECCIA_PF_GROUP_USERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(1024)
    })
}

/// `BRECCIA_PF_MICRO=2`: split each prefill chunk into two micro-batches and pipeline
/// them across the block card and the expert helper (default on since §12.13: plx
/// +34 %, logits == flag-off chunk/2; unset or `2` enables, anything else disables).
/// `BRECCIA_PF_MICRO_OVL`: overlap the tick boundary's transfers (bucketing under the
/// block card's readback, the MoE result's copy-in submitted behind the launch).
fn micro_ovl() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRECCIA_PF_MICRO_OVL").as_deref() != Ok("0"))
}

/// `BRECCIA_PF_MICRO_SHARED=1` defers `pf.shared` out of the tick-boundary ship (copied down
/// ahead of the launch segment, up under the routed GEMMs). Default off: it moves the
/// ~0.25 ms up-copy onto the helper's critical path and the down-copy onto the block
/// card's, so plx is a tie (290.7 vs 291.4 tok/s traced; untraced A,B,A,B 290.7/291.6/291.3/290.6).
fn micro_sh() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRECCIA_PF_MICRO_SHARED").as_deref() == Ok("1"))
}

fn pf_micro() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("BRECCIA_PF_MICRO").as_deref(), Err(_) | Ok("2")))
}

/// The merged-prefill optimisation flags, in [`Session::pf`] order.
pub const PF_FLAGS: [&str; 5] = [
    "BRECCIA_PF_OVERLAP",
    "BRECCIA_PF_SHCHUNK",
    "BRECCIA_PF_QSACHUNK",
    "BRECCIA_PF_GDNCHUNK",
    "BRECCIA_PF_HCCHUNK",
];
const PF_OVERLAP: usize = 0;
const PF_SHCHUNK: usize = 1;
const PF_QSACHUNK: usize = 2;
const PF_GDNCHUNK: usize = 3;
const PF_HCCHUNK: usize = 4;

/// Each device's merged-prefill flags, resolved once per session. The env wins
/// (`=0` off, anything else on); else the card's `breccia profile --prefill`
/// verdict; else on (all five pass the 1k/4k KL gate on RX and MI50, plan §11d).
/// A stale verdict (new driver or shaders) still applies, with a hint to re-run
/// the check: a card measured last week is better evidence than no measurement.
fn pf_resolve(m: &Model) -> Vec<[bool; 5]> {
    m.devices
        .iter()
        .zip(&m.pf_verdicts)
        .map(|(d, v)| {
            let stale = v.as_ref().is_some_and(|v| {
                v.driver_version != d.info.driver || v.spirv_hash != breccia_kernels::SPIRV_HASH
            });
            if stale {
                eprintln!(
                    "breccia: {} ({}): the prefill verdict is stale (driver or shaders \
                     changed); using its flags until `breccia profile --prefill \
                     --model ...` re-runs",
                    d.info.name, d.info.pci_address
                );
            }
            std::array::from_fn(|i| match std::env::var_os(PF_FLAGS[i]) {
                Some(x) => x != "0",
                None => v.as_ref().is_none_or(|v| {
                    v.flags.iter().find(|f| f.name == PF_FLAGS[i]).is_some_and(|f| f.on)
                }),
            })
        })
        .collect()
}

fn hc_read(v: &Venue<'_>, w: &HcW, r: &Buf) -> Result<()> {
    hc_read_ws(v, v.hc_ws.as_ref().expect("the session's hc workspace"), w, r)
}

/// [`hc_read`] on an explicit workspace (the T-token chunk one).
fn hc_read_ws(v: &Venue<'_>, ws: &HcWorkspace, w: &HcW, r: &Buf) -> Result<()> {
    let hw = HcWeights {
        norm: &w.norm,
        down: &w.down.buf,
        up: &w.up.buf,
        inject: w.inject.as_ref(),
    };
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
    use super::*;

    /// The (G) shared-expert span (graph.rs, `record_dense_prelude`) must switch
    /// the current stage to `wait-gpu` BEFORE the first blocking run
    /// (`pipes.get(..).run(..)` → `wall_run` → fence wait), so the blocking GPU
    /// wait is not charged to `shared-exp` (§8ae). Marks only re-attribute wall
    /// time between buckets, so the span's total is conserved.
    /// shortcut: this is a host-side Timers test only — the real blocking wait
    /// needs a GPU device, which a unit test cannot own; upgrade when a GPU arm
    /// can assert the printed split.
    #[test]
    fn wait_gpu_mark_precedes_the_blocking_run_and_conserves_the_span() {
        let mut t = Timers::default();
        assert!(
            !t.on,
            "timing is off by default: default behaviour unchanged"
        );
        t.mark(WAIT_GPU); // no-op with timing off
        assert_eq!(t.ns[WAIT_GPU], 0);

        t.on = true;
        t.start();
        t.mark(SHARED);
        std::thread::sleep(std::time::Duration::from_micros(200));
        // The new mark, exactly where it sits in (G): before the first
        // blocking `run`.
        t.mark(WAIT_GPU);
        assert_eq!(
            t.stage, WAIT_GPU,
            "the (G) span must be in wait-gpu when the first blocking run executes"
        );
        std::thread::sleep(std::time::Duration::from_micros(200));
        // The next stage mark closes the wait-gpu bucket, as the EXPERTS mark
        // (or finish_block's WAIT_GPU mark) does after the span.
        t.mark(EXPERTS);
        t.stop();
        assert!(t.ns[SHARED] > 0, "shared-exp keeps the compute it is given");
        assert!(
            t.ns[WAIT_GPU] > 0,
            "wait-gpu captures the wait before the run"
        );
        assert!(t.ns[EXPERTS] < 1_000, "the closing mark owns ~no time here");
        let total = t.total_s();
        let split = (t.ns[SHARED] + t.ns[WAIT_GPU]) as f64 / 1e9;
        assert!(
            (total - split).abs() < 1e-6,
            "split {split}s must equal the span total {total}s (tiny mark/stop scheduler gap is fine)"
        );
    }

    /// R28-1: the wait-gpu mark before the first blocking run must be
    /// re-anchored back to shared-exp before the *next* run, or the whole
    /// span prices as wait-gpu and SHARED reads ~0 — a relabel, not a
    /// finding. Checks the source order because a unit Timers test cannot
    /// own the blocking fence a real GPU run needs.
    #[test]
    fn shared_is_reanchored_after_the_first_blocking_run() {
        let src = include_str!("graph.rs");
        let span = src
            .split("shared expert + its scalar sigmoid gate")
            .nth(1)
            .expect("the shared-expert span");
        let span = &span[..span.find("Ok(BlockDigests").expect("the span's end")];
        let gate = span
            .find("&v.shexp_gate)?;")
            .expect("the first blocking run");
        let rest = &span[gate..];
        let up = rest.find("&v.shexp_up)?;").expect("the up run");
        assert!(
            rest[..up].contains("mark(SHARED)"),
            "the shared-expert span must re-anchor mark(SHARED) between the gate run and \
             the up run, or wait-gpu relabels the shared-expert compute"
        );
    }

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

    /// D4d: `imported_role_slices` must resolve the blob index the same way
    /// `build_imported_experts` enumerates it: only non-placed blocks get blobs,
    /// in (block, role gate/up/down) order. With block 0 placed, streamed
    /// block 1 must read blob 0, not blob 3 — the defect the first D4d candidate
    /// shipped (hardcoded 3·b).
    #[test]
    fn imported_role_slices_skip_placed_blocks() {
        use crate::host_arena::{BlobSource, HostArena};
        use std::fs::File;
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        // Real content, not zeros: blob i is `len` bytes of pattern (i+1).
        let dir = std::env::temp_dir().join("breccia-graph-import-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard0.bin");
        {
            let mut f = File::create(&path).unwrap();
            for i in 0..6 {
                f.write_all(&vec![2 * i as u8 + 7; 3072]).unwrap();
            }
        }
        let files = vec![Arc::new(Mutex::new(File::open(&path).unwrap()))];
        let src: Vec<BlobSource> = (0..6)
            .map(|i| BlobSource {
                file: 0,
                offset: i * 3072,
                len: 3072,
            })
            .collect();
        let mut arena = HostArena::new(HostArena::padded_total(&src)).expect("mlock under limit");
        arena.fill(&src, &files).expect("pread fill");

        // Mixed case mirroring the builder: block 0 placed, blocks 1..2 streamed
        // (3 blobs = gate/up/down per streamed block), e.g. 3 experts × 1 KiB/role.
        const EXPERTS: usize = 3;
        const ROLE: usize = 3072;
        let streamed_before_1 = 0; // block 0 is placed, so block 1 is the 1st streamed
        let streamed_before_2 = 1; // block 2 is the 2nd streamed

        // Streamed block 1: blob base 0. Its expert 2 must read blob role bytes
        // [2*64, 3*64) — the file's own pattern of blob 0, not blob 3's.
        let want = arena.blob(0).unwrap()[2 * ROLE / EXPERTS..3 * ROLE / EXPERTS].to_vec();
        let got = super::Session::imported_role_slices(
            &arena,
            streamed_before_1,
            2,
            ROLE / EXPERTS,
            ROLE / EXPERTS,
            ROLE / EXPERTS,
        )
        .unwrap();
        assert_eq!(got[0].to_vec(), want, "streamed blk1 gate reads blob 0");
        for r in 0..3 {
            let want = arena.blob(r).unwrap()[2 * 1024..3 * 1024].to_vec();
            assert_eq!(got[r].to_vec(), want, "role {r}");
        }

        // Streamed block 2: blob base 3. Without the placed-skip it would read
        // blob 6 (past the end) or the wrong bytes.
        let got2 = super::Session::imported_role_slices(
            &arena,
            streamed_before_2,
            0,
            ROLE / EXPERTS,
            ROLE / EXPERTS,
            ROLE / EXPERTS,
        )
        .unwrap();
        for r in 0..3 {
            let want = arena.blob(3 + r).unwrap()[0..1024].to_vec();
            assert_eq!(got2[r].to_vec(), want, "blk2 role {r} reads blob {}", 3 + r);
        }

        // Distinct patterns prove cross-blob separation: blob r and 3+r differ to
        // the byte.
        assert_ne!(arena.blob(0).unwrap(), arena.blob(3).unwrap());
        // Out-of-range role index is a clean error, not silent shift.
        assert!(
            super::Session::imported_role_slices(&arena, 4, 0, EXPERTS, EXPERTS, EXPERTS).is_err()
        );
    }
}
