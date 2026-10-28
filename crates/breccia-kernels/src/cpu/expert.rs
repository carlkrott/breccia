//! Expert FFN pipeline + the per-token top-k MoE dispatch.
//!
//! The routed-expert shape of this model is `n_embd = 2560`, `n_ff = 640`,
//! `n_expert = 512`, `k = 10`. One expert blob is `[gate | up | down]`:
//!
//! ```text
//! gate: [cols=n_embd, rows=n_ff]   → 640 outputs from 2560 inputs
//! up:   [cols=n_embd, rows=n_ff]
//! down: [cols=n_ff,   rows=n_embd] → 2560 outputs from 640 inputs
//! ```
//!
//! so a full expert FFN is `640·2560 + 640·2560 + 2560·640` MACs over
//! 1.746 MB of quantised weights — a pure weight-bandwidth problem.
//!
//! [`expert_ffn`] runs the three GEMVs with the SwiGLU in between, reading
//! straight from [`ExpertSlices`] (mmap windows, no copy). [`ExpertPool`]
//! runs the ten selected experts of one token's block in parallel, one expert
//! per worker, and accumulates `Σ w_j · down_j(SiLU(gate_j x) · up_j x)` in
//! dispatch order so the result is bit-reproducible regardless of which worker
//! finishes first.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use anyhow::{bail, Context, Result};
use breccia_format::{ExpertBlock, ExpertRole, ExpertSlices, ShardSet};

use crate::cpu::{gemv_into, Path};
use crate::quant::QuantType;

/// `silu(x) = x / (1 + exp(-x))`, ggml's `GGML_SILU_FP32`, with `-inf → 0`.
#[inline]
pub fn silu(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        0.0
    } else {
        x / (1.0 + (-x).exp())
    }
}

/// One role tensor (gate / up / down) of one block, in mmap geometry.
#[derive(Clone, Copy, Debug)]
pub struct RolePlan {
    pub role: ExpertRole,
    pub ty: QuantType,
    /// Output width (`ne1`).
    pub rows: usize,
    /// Input width (`ne0`).
    pub cols: usize,
    /// Shard holding the tensor (0 for every expert tensor of this model).
    pub shard: usize,
    /// Absolute byte offset of expert 0 inside that shard.
    pub offset: usize,
    /// Bytes of one expert's slice.
    pub per_expert: usize,
}

impl RolePlan {
    pub fn bytes(&self) -> usize {
        self.rows * crate::cpu::row_bytes(self.ty, self.cols)
    }
}

/// One block's routed experts: geometry only, `Copy`, cheap to send to a worker.
#[derive(Clone, Copy, Debug)]
pub struct LayerPlan {
    pub block: usize,
    pub n_experts: usize,
    /// `gate.ne0()` — the residual width (2560).
    pub n_embd: usize,
    /// `gate.ne1()` — the expert hidden width (640).
    pub n_ff: usize,
    pub gate: RolePlan,
    pub up: RolePlan,
    pub down: RolePlan,
}

impl LayerPlan {
    /// Build the plan for `block` from a `breccia_format` layout.
    pub fn from_block(set: &ShardSet, block: &ExpertBlock<'_>) -> Result<Self> {
        let plan = |role: ExpertRole| -> Result<RolePlan> {
            let t = block.role(role);
            let ty = QuantType::from_ggml(t.ty()).with_context(|| {
                format!("{}: ggml type {} has no CPU kernel", t.name(), t.ty().id())
            })?;
            Ok(RolePlan {
                role,
                ty,
                rows: t.ne1() as usize,
                cols: t.ne0() as usize,
                shard: shard_of(set, t.as_slice())?,
                offset: t.file_range().start as usize,
                per_expert: block.bytes_per_expert(role) as usize,
            })
        };
        let gate = plan(ExpertRole::Gate)?;
        let up = plan(ExpertRole::Up)?;
        let down = plan(ExpertRole::Down)?;
        if down.cols != gate.rows {
            bail!(
                "blk.{}: down.cols ({}) != gate.rows ({})",
                block.block,
                down.cols,
                gate.rows
            );
        }
        Ok(Self {
            block: block.block,
            n_experts: block.n_experts,
            n_embd: gate.cols,
            n_ff: gate.rows,
            gate,
            up,
            down,
        })
    }

    /// Bytes of one expert blob (`gate | up | down`) — the unit of read.
    pub fn expert_bytes(&self) -> usize {
        self.gate.per_expert + self.up.per_expert + self.down.per_expert
    }

    /// One role by name (`"gate"` / `"up"` / `"down"`), for CLI/report code.
    pub fn role(&self, role: &str) -> &RolePlan {
        match role {
            "gate" => &self.gate,
            "up" => &self.up,
            _ => &self.down,
        }
    }

    /// Bytes the served kernels read for one expert (block-aligned rows).
    pub fn expert_kernel_bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }

    /// Zero-copy `gate | up | down` slices of expert `e`.
    pub fn slices<'a>(&self, set: &'a ShardSet, expert: usize) -> Result<ExpertSlices<'a>> {
        if expert >= self.n_experts {
            bail!("blk.{}: expert {expert} >= {}", self.block, self.n_experts);
        }
        let part = |r: &RolePlan| -> Result<&'a [u8]> {
            let bytes = set
                .parts()
                .get(r.shard)
                .with_context(|| format!("blk.{}: no shard {}", self.block, r.shard))?
                .bytes();
            let start = r.offset + expert * r.per_expert;
            bytes.get(start..start + r.per_expert).with_context(|| {
                format!(
                    "blk.{}: {} slice [{start}, +{}) outside shard {}",
                    self.block,
                    r.role.name(),
                    r.per_expert,
                    r.shard
                )
            })
        };
        Ok(ExpertSlices {
            gate: part(&self.gate)?,
            up: part(&self.up)?,
            down: part(&self.down)?,
        })
    }
}

/// Which shard of `set` a tensor slice points into (by mapping range).
fn shard_of(set: &ShardSet, slice: &[u8]) -> Result<usize> {
    let p = slice.as_ptr() as usize;
    set.parts()
        .iter()
        .position(|s| {
            let b = s.bytes();
            let start = b.as_ptr() as usize;
            p >= start && p < start + b.len()
        })
        .with_context(|| "tensor slice is not inside any mapped shard".to_string())
}

/// Per-worker (or caller-owned) FFN scratch: no allocation inside a token.
#[derive(Clone, Debug)]
pub struct FfnScratch {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub act: Vec<f32>,
    pub tmp: Vec<f32>,
}

impl FfnScratch {
    pub fn new(plan: &LayerPlan) -> Self {
        Self {
            gate: vec![0.0; plan.n_ff],
            up: vec![0.0; plan.n_ff],
            act: vec![0.0; plan.n_ff],
            tmp: vec![0.0; plan.n_embd],
        }
    }
}

/// One expert's SwiGLU FFN, accumulated with weight `weight` into `out`.
///
/// `out[k] += weight * down(SiLU(gate·x) · up·x)[k]` — exactly the routed-expert
/// term of `Σ_j w_j · down_j(SiLU(gate_j x) · up_j x)`
/// (`docs/specs/qwen4exp-architecture.md` §5.4).
///
/// Returns the bytes of weight data read.
///
/// The SwiGLU is a **separate pass** over `n_ff = 640` values, not fused into
/// the up GEMV: the up GEMV emits one output at a time, and fusing would need
/// the gate result of the same row, i.e. either a second read of the gate
/// weights or a transposed loop order that destroys the coalesced weight
/// stream. 640 extra `exp` calls and 640 extra 4-byte reads per expert are
/// noise against 1.746 MB of weights (measured cost in
/// `docs/phase-3/08-cpu-expert-lane.md` §5).
pub fn expert_ffn(
    path: Path,
    plan: &LayerPlan,
    slices: &ExpertSlices<'_>,
    x: &[f32],
    scratch: &mut FfnScratch,
    weight: f32,
    out: &mut [f32],
) -> usize {
    gemv_into(
        path,
        plan.gate.ty,
        slices.gate,
        plan.n_ff,
        plan.n_embd,
        x,
        &mut scratch.gate,
    );
    gemv_into(
        path,
        plan.up.ty,
        slices.up,
        plan.n_ff,
        plan.n_embd,
        x,
        &mut scratch.up,
    );
    for k in 0..plan.n_ff {
        scratch.act[k] = silu(scratch.gate[k]) * scratch.up[k];
    }
    gemv_into(
        path,
        plan.down.ty,
        slices.down,
        plan.n_embd,
        plan.n_ff,
        &scratch.act,
        &mut scratch.tmp,
    );
    for (o, v) in out.iter_mut().zip(&scratch.tmp) {
        *o += weight * v;
    }
    slices.len()
}

/// The per-token top-k MoE dispatch as a persistent thread pool.
///
/// # Why the work is split twice, not once
///
/// The model selects **10** experts per block, so "one expert per worker" wastes
/// a third of an 8-thread machine: ten ~1.75 MB jobs cannot fill eight workers
/// evenly (two workers take two experts, the batch takes two job-times).
/// Measured on this host, per-expert dispatch with 8 threads reached
/// 13.1 GB/s while 12 threads reached 24.1 GB/s — same kernels, same bytes, only
/// the packing changed (`docs/phase-3/08-cpu-expert-lane.md` §4).
///
/// So each expert is cut into row shards, and because a routed expert's FFN is
/// *sequential* (`down` consumes `SiLU(gate·x)·up·x`, i.e. all 640 gate and `up`
/// rows) the pool runs two phases with one barrier between them:
///
/// ```text
/// phase 1  gate rows of every expert (in parallel with the up rows)   → act
///          SwiGLU on the main thread: 640 exp() per expert, ~1 µs
/// phase 2  down rows of every expert, accumulated as Σ_j w_j · down_j(act_j)
/// ```
///
/// Shard counts come from [`choose_shards`], which picks the smallest split that
/// keeps the predicted makespan within 15 % of perfect packing, so a 4-thread run
/// still sends whole-role jobs (40 messages/token-block) while a 16-thread run
/// splits into thirds.
///
/// # Determinism
///
/// Dispatch is round-robin over workers, so which worker runs a shard depends
/// only on the thread count; the **result** does not: phase-1 shards write
/// disjoint row ranges of `gate`/`up`, and phase-2 shards are accumulated into
/// `out` in `(expert, shard)` order, which is the order the single-threaded path
/// uses. `tests/cpu_experts.rs` asserts the pool is bit-identical to
/// [`moe_topk_sequential`] for 2, 4 and 8 threads.
///
/// # Safety
///
/// Workers hold an `Arc<ShardSet>`, so nothing is borrowed across threads that
/// is not `Sync` and no raw pointers appear: the mmap is read-only and every
/// worker re-derives its slices from the same mapping.
pub struct ExpertPool {
    threads: usize,
    /// Distinct experts the last [`ExpertPool::moe_topk_chunk`] call touched (the
    /// number the merged-MoE byte budget is built on, §8u).
    pub last_distinct: usize,
    tx: Vec<Sender<Msg>>,
    rx: Receiver<Done>,
    handles: Vec<JoinHandle<()>>,
}

/// One worker's work for one phase.
enum Msg {
    Batch(Batch),
    Exit,
}

/// A worker's whole share of a phase, in one message.
///
/// Batching matters: measured on this host a parked-receiver wake-up costs
/// ~3 µs, so one message per shard (up to 90 per token-block, 4 300 per token)
/// cost more than the imbalance it removed — 12 threads fell from 24.1 to
/// 16.2 GB/s. One message per worker per phase makes the synchronisation cost
/// `O(threads)`, not `O(shards)`.
struct Batch {
    tasks: Vec<Shard>,
}

/// What a shard computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShardKind {
    /// `gate` **and** `up` rows `[r0, r1)` of one expert, then the SwiGLU over
    /// exactly those rows: the worker that owns the rows owns their
    /// `SiLU(gate)·up`, so the 640 `exp()` calls per expert cost nothing serial.
    GateUp,
    /// `down` rows `[r0, r1)` of one expert, consuming `act`.
    Down,
}

struct Shard {
    /// Dispatch index: identifies the reply's destination.
    seq: usize,
    expert: usize,
    plan: LayerPlan,
    kind: ShardKind,
    /// Row range `[r0, r1)` of the role tensor.
    r0: usize,
    r1: usize,
    x: Arc<Vec<f32>>,
    /// Offset of this shard's activation vector inside `x` (the whole
    /// activation for gate/up; `act_j` for down).
    x_off: usize,
    /// Grouped form: `(destination slot, activation offset, weight)` for every
    /// token that uses this expert. Empty means "the single-user form above" —
    /// one activation at `x_off`, weight applied by the caller.
    users: Arc<Vec<(usize, usize, f32)>>,
}

struct Done {
    results: Vec<(usize, Vec<f32>)>,
}

/// Rows per shard below which splitting costs more than it balances.
const MIN_SHARD_ROWS: usize = 16;
/// Largest split per role (messages are not free).
const MAX_SHARDS: usize = 4;
/// Accept a split whose predicted makespan is within this factor of perfect.
const PACK_TOLERANCE: f64 = 1.15;

/// Smallest row split (1..=[`MAX_SHARDS`]) that packs `tasks` jobs onto
/// `threads` workers within [`PACK_TOLERANCE`] of perfect.
fn choose_shards(tasks: usize, threads: usize, rows: usize) -> usize {
    let ideal = tasks as f64 / threads as f64;
    let mut best = 1usize;
    for s in 1..=MAX_SHARDS {
        if rows / s < MIN_SHARD_ROWS {
            break;
        }
        let makespan = (tasks * s).div_ceil(threads) as f64 / s as f64;
        best = s;
        if makespan <= ideal * PACK_TOLERANCE {
            break;
        }
    }
    best
}

impl ExpertPool {
    /// Spawn `threads` workers over `set` (`threads` is clamped to ≥1).
    pub fn new(set: Arc<ShardSet>, threads: usize, path: Path) -> Self {
        let threads = threads.max(1);
        let (done_tx, rx) = mpsc::channel::<Done>();
        let mut tx = Vec::with_capacity(threads);
        let mut handles = Vec::with_capacity(threads);
        let pin = crate::cpu::cpu_pin();
        for id in 0..threads {
            let (job_tx, job_rx) = mpsc::channel::<Msg>();
            let done = done_tx.clone();
            let set = Arc::clone(&set);
            let cpu = pin.as_ref().map(|p| p.workers[id % p.workers.len()]);
            let handle = thread::Builder::new()
                .name(format!("breccia-expert-{id}"))
                .spawn(move || {
                    if let Some(c) = cpu {
                        crate::cpu::pin_current(&[c]);
                    }
                    worker_loop(&set, path, &job_rx, &done)
                })
                .expect("spawning an expert worker");
            tx.push(job_tx);
            handles.push(handle);
        }
        drop(done_tx);
        Self {
            threads,
            last_distinct: 0,
            tx,
            rx,
            handles,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run one phase: round-robin the tasks over the workers, one message each,
    /// and scatter the replies back into dispatch order.
    ///
    /// Round-robin keeps the per-worker task counts within one of each other,
    /// and every shard of a phase has the same number of rows, so that is
    /// balanced work.
    fn run_phase(&self, tasks: Vec<Shard>) -> Result<Vec<Option<Vec<f32>>>> {
        let n = tasks.len();
        let mut slots: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
        let mut buckets: Vec<Vec<Shard>> = (0..self.threads).map(|_| Vec::new()).collect();
        for (i, task) in tasks.into_iter().enumerate() {
            buckets[i % self.threads].push(task);
        }
        let mut busy = 0usize;
        for (w, batch) in buckets.into_iter().enumerate() {
            if batch.is_empty() {
                continue;
            }
            self.tx[w]
                .send(Msg::Batch(Batch { tasks: batch }))
                .map_err(|_| anyhow::anyhow!("expert worker {w} is gone"))?;
            busy += 1;
        }
        for _ in 0..busy {
            let done = self
                .rx
                .recv()
                .map_err(|_| anyhow::anyhow!("expert pool result channel closed"))?;
            for (seq, values) in done.results {
                if let Some(slot) = slots.get_mut(seq) {
                    *slot = Some(values);
                }
            }
        }
        Ok(slots)
    }

    /// Routed-expert contribution of one token's block, accumulated into `out`.
    ///
    /// `ids` are the top-k expert indices and `weights` their renormalised
    /// router probabilities; `out` receives `Σ_j w_j · down_j(SiLU(gate_j x) ·
    /// up_j x)` added to whatever it already holds.
    ///
    /// Returns the bytes of expert weights read.
    pub fn moe_topk(
        &mut self,
        plan: &LayerPlan,
        ids: &[usize],
        weights: &[f32],
        x: &[f32],
        out: &mut [f32],
    ) -> Result<usize> {
        if ids.len() != weights.len() {
            bail!("{} expert ids vs {} weights", ids.len(), weights.len());
        }
        if ids.is_empty() {
            return Ok(0);
        }
        let k = ids.len();
        let (n_ff, n_embd) = (plan.n_ff, plan.n_embd);
        let x = Arc::new(x.to_vec());

        // ---- phase 1: gate+up+SwiGLU rows of every selected expert --------
        let s1 = choose_shards(k, self.threads, n_ff);
        let mut first: Vec<(usize, usize, usize)> = Vec::new();
        for j in 0..k {
            for s in 0..s1 {
                let (r0, r1) = shard_rows(n_ff, s, s1);
                first.push((j, r0, r1));
            }
        }
        let tasks1: Vec<Shard> = first
            .iter()
            .enumerate()
            .map(|(seq, &(j, r0, r1))| Shard {
                seq,
                expert: ids[j],
                plan: *plan,
                kind: ShardKind::GateUp,
                r0,
                r1,
                x: Arc::clone(&x),
                x_off: 0,
                users: no_users(),
            })
            .collect();
        let done1 = self.run_phase(tasks1)?;
        let mut act = vec![0f32; k * n_ff];
        for (seq, &(j, r0, r1)) in first.iter().enumerate() {
            let values = done1[seq]
                .as_ref()
                .with_context(|| format!("phase-1 shard {seq} never returned"))?;
            act[j * n_ff + r0..j * n_ff + r1].copy_from_slice(values);
        }

        // ---- phase 2: down rows, weighted and accumulated ------------------
        let act = Arc::new(act);
        let s2 = choose_shards(k, self.threads, n_embd);
        let mut second: Vec<(usize, usize, usize)> = Vec::new();
        for j in 0..k {
            for s in 0..s2 {
                let (r0, r1) = shard_rows(n_embd, s, s2);
                second.push((j, r0, r1));
            }
        }
        let tasks2: Vec<Shard> = second
            .iter()
            .enumerate()
            .map(|(seq, &(j, r0, r1))| Shard {
                seq,
                expert: ids[j],
                plan: *plan,
                kind: ShardKind::Down,
                r0,
                r1,
                x: Arc::clone(&act),
                x_off: j * n_ff,
                users: no_users(),
            })
            .collect();
        let done2 = self.run_phase(tasks2)?;
        for (seq, &(j, r0, _r1)) in second.iter().enumerate() {
            let values = done2[seq]
                .as_ref()
                .with_context(|| format!("phase-2 shard {seq} never returned"))?;
            let w = weights[j];
            for (i, v) in values.iter().enumerate() {
                out[r0 + i] += w * v;
            }
        }
        Ok(plan.expert_kernel_bytes() * k)
    }
}

/// The shared empty user list: the single-token path must not allocate per shard
/// (that cost ~40 % of the decode lane's time when it did).
fn no_users() -> Arc<Vec<(usize, usize, f32)>> {
    static EMPTY: std::sync::OnceLock<Arc<Vec<(usize, usize, f32)>>> = std::sync::OnceLock::new();
    Arc::clone(EMPTY.get_or_init(|| Arc::new(Vec::new())))
}

/// `(expert, row range, users)` for one grouped phase task: the shard's row range
/// and the `(slot, activation offset, weight)` of every token that uses it.
type GroupedTask = (usize, usize, usize, Vec<(usize, usize, f32)>);

/// One token's routing inside a prefill chunk: its top-k experts and weights.
#[derive(Clone, Copy, Debug)]
pub struct ChunkRoute<'a> {
    pub ids: &'a [usize],
    pub weights: &'a [f32],
}

/// One row-range GEMV straight out of the mmap. `None` if the slice geometry
/// does not hold (a malformed plan; the caller skips the shard rather than
/// panicking a worker).
fn run_shard(set: &ShardSet, path: Path, shard: &Shard) -> Option<Vec<f32>> {
    let slices = shard.plan.slices(set, shard.expert).ok()?;
    let plan = shard.plan;
    let rows = shard.r1 - shard.r0;
    if shard.users.is_empty() {
        // Single-token form: one activation, at `x_off`, weight applied by caller.
        let mut values = vec![0f32; rows];
        let ok = one_user(
            &slices,
            &plan,
            path,
            shard.kind,
            rows,
            shard.r0,
            &shard.x,
            shard.x_off,
            &mut values,
        );
        return ok.then_some(values);
    }
    // Grouped form: every token that routed to this expert. The shard's weight
    // rows are read once and then served from L2/L3 for the other users - that
    // cache reuse is the whole point of grouping a chunk (§8r/§8s).
    let m = shard.users.len();
    let mut values = vec![0f32; rows * m];
    // The `down` role is the biggest byte consumer, and its users share the same
    // expert rows: dequantise once and dot every user (bit-identical per user,
    // §8t). Below 2 users there is nothing to amortise.
    if m > 1 && shard.kind == ShardKind::Down && crate::cpu::many_supported(path, plan.down.ty) {
        let act = &shard.x;
        let xs: Vec<&[f32]> = shard
            .users
            .iter()
            .map(|&(_, x_off, _)| &act[x_off..x_off + plan.n_ff])
            .collect();
        let rb = crate::cpu::row_bytes(plan.down.ty, plan.down.cols);
        let dv = slices.down.get(shard.r0 * rb..)?;
        let bytes = rows * rb;
        if dv.len() < bytes {
            return None;
        }
        // SAFETY: the slice holds `rows` whole rows and every activation has
        // `plan.down.cols` values (checked by the caller's chunk layout).
        crate::cpu::gemv_into_many(
            path,
            plan.down.ty,
            &dv[..bytes],
            rows,
            plan.down.cols,
            &xs,
            &mut values,
        );
        return Some(values);
    }
    for (u, &(_, x_off, _)) in shard.users.iter().enumerate() {
        let ok = one_user(
            &slices,
            &plan,
            path,
            shard.kind,
            rows,
            shard.r0,
            &shard.x,
            x_off,
            &mut values[u * rows..(u + 1) * rows],
        );
        if !ok {
            return None;
        }
    }
    Some(values)
}

/// One user's row-range GEMV against already-resolved slices: `values` receives
/// `rows` results. `false` when the slice geometry does not hold (a malformed
/// plan; the worker skips the shard rather than panicking).
#[allow(clippy::too_many_arguments)]
fn one_user(
    slices: &ExpertSlices<'_>,
    plan: &LayerPlan,
    path: Path,
    kind: ShardKind,
    rows: usize,
    r0: usize,
    x: &[f32],
    x_off: usize,
    values: &mut [f32],
) -> bool {
    match kind {
        ShardKind::GateUp => {
            let Some(xs) = x.get(x_off..x_off + plan.n_embd) else {
                return false;
            };
            let mut up = vec![0f32; rows];
            let g = plan.gate;
            let g_rb = crate::cpu::row_bytes(g.ty, g.cols);
            let Some(gv) = slices.gate.get(r0 * g_rb..) else {
                return false;
            };
            gemv_into(path, g.ty, gv, rows, g.cols, xs, values);
            let u = plan.up;
            let u_rb = crate::cpu::row_bytes(u.ty, u.cols);
            let Some(uv) = slices.up.get(r0 * u_rb..) else {
                return false;
            };
            gemv_into(path, u.ty, uv, rows, u.cols, xs, &mut up);
            for (v, u) in values.iter_mut().zip(&up) {
                *v = silu(*v) * u;
            }
        }
        ShardKind::Down => {
            let r = plan.down;
            let rb = crate::cpu::row_bytes(r.ty, r.cols);
            let Some(dv) = slices.down.get(r0 * rb..) else {
                return false;
            };
            let Some(xs) = x.get(x_off..x_off + r.cols) else {
                return false;
            };
            gemv_into(path, r.ty, dv, rows, r.cols, xs, values);
        }
    }
    true
}

fn shard_rows(rows: usize, s: usize, shards: usize) -> (usize, usize) {
    let per = rows.div_ceil(shards);
    let r0 = (s * per).min(rows);
    let r1 = (r0 + per).min(rows);
    (r0, r1)
}

impl ExpertPool {
    /// Routed-expert contribution of **T tokens at once**, grouped by expert.
    ///
    /// `x` is the chunk's `[T, n_embd]` activations and `out` its `[T, n_embd]`
    /// outputs (accumulated, like [`ExpertPool::moe_topk`]). Instead of running
    /// each token's top-10 separately, this buckets the `(token, expert)` pairs,
    /// and each expert's rows are read **once per chunk** and reused for every
    /// token that routed to it — the whole point of a prefill chunk
    /// (`docs/phase-4/01-forward-pass.md` §8r/§8s).
    ///
    /// # Determinism
    ///
    /// Each row's dot product is bit-identical to the per-token path (same
    /// kernel, same slice, same order). The *sum over an expert set* is
    /// accumulated in **expert-id order** rather than the router's `j` order,
    /// because grouping reorders the work; that changes the routed sum in the
    /// last ulp (the same class of deviation the split-K `hc_down` takes, §8n).
    /// The greedy-token parity gate is what decides whether that is acceptable.
    pub fn moe_topk_chunk(
        &mut self,
        plan: &LayerPlan,
        routes: &[ChunkRoute<'_>],
        x: &[f32],
        out: &mut [f32],
    ) -> Result<usize> {
        let t = routes.len();
        let (n_ff, n_embd) = (plan.n_ff, plan.n_embd);
        if t == 0 {
            return Ok(0);
        }
        if x.len() < t * n_embd || out.len() < t * n_embd {
            bail!(
                "chunk of {t} needs {} activations and outputs, got {} and {}",
                t * n_embd,
                x.len(),
                out.len()
            );
        }
        // expert -> [(token, j)] in router order, so a token's pairs stay ordered.
        let mut by: std::collections::BTreeMap<usize, Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();
        let mut k = 0usize;
        for (token, r) in routes.iter().enumerate() {
            if r.ids.len() != r.weights.len() {
                bail!("{} expert ids vs {} weights", r.ids.len(), r.weights.len());
            }
            k = k.max(r.ids.len());
            for (j, &id) in r.ids.iter().enumerate() {
                by.entry(id).or_default().push((token, j));
            }
        }
        let distinct = by.len();
        self.last_distinct = distinct;
        let words = Arc::new(x[..t * n_embd].to_vec());

        // ---- phase 1: gate+up+SwiGLU rows, one task per (expert, row shard) ----
        let s1 = choose_shards(distinct * k, self.threads, n_ff).max(1);
        let mut first: Vec<GroupedTask> = Vec::new();
        for (&expert, users) in by.iter() {
            let slots: Vec<(usize, usize, f32)> = users
                .iter()
                .map(|&(token, j)| (token * k + j, token * n_embd, 0.0))
                .collect();
            for s in 0..s1 {
                let (r0, r1) = shard_rows(n_ff, s, s1);
                first.push((expert, r0, r1, slots.clone()));
            }
        }
        let tasks1: Vec<Shard> = first
            .iter()
            .enumerate()
            .map(|(seq, &(expert, r0, r1, ref users))| Shard {
                seq,
                expert,
                plan: *plan,
                kind: ShardKind::GateUp,
                r0,
                r1,
                x: Arc::clone(&words),
                x_off: 0,
                users: Arc::new(users.clone()),
            })
            .collect();
        let done1 = self.run_phase(tasks1)?;
        let mut act = vec![0f32; t * k * n_ff];
        for (seq, &(_, r0, r1, ref users)) in first.iter().enumerate() {
            let values = done1[seq]
                .as_ref()
                .with_context(|| format!("chunk phase-1 shard {seq} never returned"))?;
            let rows = r1 - r0;
            for (u, &(slot, _, _)) in users.iter().enumerate() {
                act[slot * n_ff + r0..slot * n_ff + r1]
                    .copy_from_slice(&values[u * rows..(u + 1) * rows]);
            }
        }

        // ---- phase 2: down rows, weighted into each token's output ------------
        let act = Arc::new(act);
        let s2 = choose_shards(distinct * k, self.threads, n_embd).max(1);
        let mut second: Vec<GroupedTask> = Vec::new();
        for (&expert, users) in by.iter() {
            let slots: Vec<(usize, usize, f32)> = users
                .iter()
                .map(|&(token, j)| (token, (token * k + j) * n_ff, routes[token].weights[j]))
                .collect();
            for s in 0..s2 {
                let (r0, r1) = shard_rows(n_embd, s, s2);
                second.push((expert, r0, r1, slots.clone()));
            }
        }
        let tasks2: Vec<Shard> = second
            .iter()
            .enumerate()
            .map(|(seq, &(expert, r0, r1, ref users))| Shard {
                seq,
                expert,
                plan: *plan,
                kind: ShardKind::Down,
                r0,
                r1,
                x: Arc::clone(&act),
                x_off: 0,
                users: Arc::new(users.clone()),
            })
            .collect();
        let done2 = self.run_phase(tasks2)?;
        for (seq, &(_, r0, r1, ref users)) in second.iter().enumerate() {
            let values = done2[seq]
                .as_ref()
                .with_context(|| format!("chunk phase-2 shard {seq} never returned"))?;
            let rows = r1 - r0;
            for (u, &(token, _, w)) in users.iter().enumerate() {
                let dst = &mut out[token * n_embd + r0..token * n_embd + r1];
                for (o, v) in dst.iter_mut().zip(&values[u * rows..(u + 1) * rows]) {
                    *o += w * v;
                }
            }
        }
        Ok(plan.expert_kernel_bytes() * k * t)
    }
}

impl Drop for ExpertPool {
    fn drop(&mut self) {
        for tx in &self.tx {
            let _ = tx.send(Msg::Exit);
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker_loop(set: &ShardSet, path: Path, rx: &Receiver<Msg>, done: &Sender<Done>) {
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Exit => break,
            Msg::Batch(batch) => {
                let mut results = Vec::with_capacity(batch.tasks.len());
                for shard in batch.tasks {
                    let Some(values) = run_shard(set, path, &shard) else {
                        continue;
                    };
                    results.push((shard.seq, values));
                }
                if done.send(Done { results }).is_err() {
                    break;
                }
            }
        }
    }
}

/// Single-threaded reference for [`ExpertPool::moe_topk`] (and the `--threads 1`
/// path), sharing every line of the arithmetic.
pub fn moe_topk_sequential(
    set: &ShardSet,
    path: Path,
    plan: &LayerPlan,
    ids: &[usize],
    weights: &[f32],
    x: &[f32],
    out: &mut [f32],
) -> Result<usize> {
    if ids.len() != weights.len() {
        bail!("{} expert ids vs {} weights", ids.len(), weights.len());
    }
    let mut scratch = FfnScratch::new(plan);
    let mut bytes = 0usize;
    for (&expert, &weight) in ids.iter().zip(weights) {
        let slices = plan.slices(set, expert)?;
        bytes += expert_ffn(path, plan, &slices, x, &mut scratch, weight, out);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_matches_the_reference_definition() {
        for x in [-8.0f32, -1.0, 0.0, 0.5, 3.0, 20.0] {
            let want = f64::from(x) / (1.0 + (-f64::from(x)).exp());
            assert!(
                (f64::from(silu(x)) - want).abs() < 1e-6,
                "silu({x}) = {} vs {want}",
                silu(x)
            );
        }
        assert_eq!(silu(f32::NEG_INFINITY), 0.0);
    }
}
