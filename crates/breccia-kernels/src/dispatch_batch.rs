//! Record-only dispatch batching: **one command buffer and one submission per
//! segment**, with a ring of segments in flight.
//!
//! # Why
//!
//! Every kernel entry point here submits and waits (`Gpu::wall_run`), which is
//! the right shape for a benchmark cell — the phase-3 numbers are GPU timestamps
//! around exactly one dispatch — and the wrong shape for a token: the graph
//! issues ~1 500 dispatches per token and each one costs a host round trip.
//! Measured on this machine (`examples/dispatch_overhead`, MI50 / RX): **36 / 64
//! µs** per synchronous dispatch, i.e. 51–97 ms of a token before any arithmetic.
//!
//! # What
//!
//! [`begin`] picks the next ring slot, starts recording into that slot's command
//! buffer and sets a thread-local flag; while it is set, [`crate::ops::Pipe::run`],
//! [`crate::Gemv::run`] and [`crate::GemvBatch::run`] **record** their dispatch
//! instead of submitting it. [`end_submit`] closes the command buffer and submits
//! it **without waiting**; [`wait_all`] waits for every submission still in
//! flight.
//!
//! # The two-segment block (`docs/phase-4/01-forward-pass.md` §8n)
//!
//! Segmenting a block lets the **host expert lane overlap the GPU**: segment A
//! holds everything up to and including the router and is submitted with a
//! timeline-semaphore signal; segment B holds the shared expert and is submitted
//! immediately behind it on the same queue. The host waits for the *marker*
//! ([`Gpu::timeline`] + [`breccia_vk::ctx::Timeline::wait_host`]), reads the
//! router's ids out of the staging buffer and runs the expert lane while segment
//! B is still executing. That wait is bounded and falls back to the fence, so a
//! stuck GPU cannot hang the host.
//!
//! The ring exists because a command buffer that is pending execution cannot be
//! reset: block *n*'s segments stay in flight while block *n+1* is recorded.
//!
//! # The descriptor-set rule that makes this work
//!
//! A `Kernel` owns **one** descriptor set, and `vkUpdateDescriptorSets` may not
//! touch a set that a recorded (not yet submitted) command buffer references —
//! the GPU would see the *last* write for every recorded dispatch. The first
//! version of this module did exactly that and produced a GPUVM fault on the
//! MI50 within one token. So a batch allocates a **fresh** set per recorded
//! dispatch from its own pool (compatible with the kernel's layout, since it is
//! allocated from the kernel's own `dsl`), writes that set and binds it. A batch
//! therefore costs one pool create/destroy plus one set per dispatch, which is
//! tens of microseconds — against one submit+fence per dispatch.
//!
//! # Discipline the caller must keep
//!
//! * No host readback, upload or allocation through `Gpu::record`/`wall_run` may
//!   happen inside a batch (they reset the one-shot command buffer). Close the
//!   batch first. Copies *into* a staging buffer are fine: [`record_copy`]
//!   records them into the open batch.
//! * Every recorded dispatch is followed by a full barrier, so a batch may mix
//!   dependant kernels freely.
//! * **One open batch per device** (`begin` refuses to nest on the same device).
//!   Two-stream decode keeps stream A's block open on card A while stream B
//!   records on card B; the per-device state is what makes that legal, since each
//!   device has its own command buffer (§8o).

use anyhow::{anyhow, Result};
use ash::vk;
use breccia_vk::ctx::{Buf, Gpu, Kernel, Qc, Timeline};
use std::cell::{Cell, RefCell};

thread_local! {
    /// Set by [`skip_next_barrier`]: the next recorded dispatch is provably
    /// independent of the one before it, so its trailing fence can be dropped.
    static SKIP_BARRIER: Cell<bool> = const { Cell::new(false) };
    /// Batches currently being recorded, one per device.
    static OPEN: RefCell<Vec<Open>> = const { RefCell::new(Vec::new()) };
    /// Round-robin cursor for [`begin`].
    static NEXT: Cell<usize> = const { Cell::new(0) };
    /// Per-slot descriptor pool of the **in-flight** submission (freed when its
    /// fence is waited on).
    static POOLS: RefCell<Vec<vk::DescriptorPool>> = const { RefCell::new(Vec::new()) };
    /// Per-slot fence of the in-flight submission (`null` = idle).
    static FENCES: RefCell<Vec<vk::Fence>> = const { RefCell::new(Vec::new()) };
    /// Per-slot device handle that owns the slot's fence.
    static DEVICES: RefCell<Vec<ash::vk::Device>> = const { RefCell::new(Vec::new()) };
    /// Per-slot `ash::Device` clone, so [`shutdown`] can free the slot's pool even
    /// when the caller no longer has that device's `Gpu`.
    static SLOT_DEV: RefCell<Vec<Option<ash::Device>>> = const { RefCell::new(Vec::new()) };
    /// Cumulative host seconds spent inside `record` (set allocation, descriptor
    /// write, bind, push constants, dispatch, barrier) — the cost a pre-recorded
    /// command buffer would remove.
    static RECORD_NS: Cell<u64> = const { Cell::new(0) };
    /// Dispatches recorded since the same reset point as [`RECORD_NS`]: the
    /// **matching denominator**, so `ns = RECORD_NS / RECORDED` is one window's
    /// per-dispatch price and not an artefact of mixing two (§8aq, `docs/research/11`
    /// §A6).
    static RECORDED: Cell<u64> = const { Cell::new(0) };
    /// Host seconds spent waiting for slots in [`wait_all`]/[`begin`].
    static WAIT_NS: Cell<u64> = const { Cell::new(0) };
    /// Per-resource dependency tracking (§8al, ncnn's model): the `vk::Buffer`s
    /// **written since the last barrier** that separates them from a later reader.
    /// A barrier is emitted only when the next dispatch binds one of these.
    static DIRTY: RefCell<Vec<vk::Buffer>> = const { RefCell::new(Vec::new()) };
    /// **Persistent descriptor sets** (§8ao): one set per
    /// `(pipeline layout, buffer set)` — the decode shape is frozen per config, so
    /// every token asks for exactly the same bindings and the set can be written
    /// once, off the hot path. Keyed by the pipeline handle and the bound buffer
    /// handles; allocated from a per-device pool that lives until `shutdown`.
    static SETS: RefCell<std::collections::HashMap<(u64, Vec<vk::Buffer>), vk::DescriptorSet>> =
        RefCell::new(std::collections::HashMap::new());
    static SET_POOLS: RefCell<Vec<(ash::vk::Device, vk::DescriptorPool, ash::Device)>> =
        const { RefCell::new(Vec::new()) };
    /// Per-phase host nanoseconds inside `record` (§8aq): descriptor key+cache,
    /// set write, bind_pipeline, bind_descriptor_sets, push_constants, dispatch,
    /// barrier, and our own tracker bookkeeping.
    static PROF: [Cell<u64>; 8] = const { [const { Cell::new(0) }; 8] };
    /// Counters: sets written (cold) vs dispatches that reused one (hot).
    static SETS_NEW: Cell<u64> = const { Cell::new(0) };
    static SETS_HIT: Cell<u64> = const { Cell::new(0) };
    /// Barriers emitted into batches since the process started, and the subset the
    /// per-resource tracker **pruned** for comparison.
    static BARRIERS: Cell<u64> = const { Cell::new(0) };
    static BARRIERS_SAVED: Cell<u64> = const { Cell::new(0) };

    // ---- GPU-time probe (`BRECCIA_GPU_TIME`) — one query pool per ring slot ----
    //
    // The old probe had **one pool per device shared by every ring slot**, so a
    // read could land while another slot's queries were still in flight and
    // `vkGetQueryPoolResults` returned "A fence or query has not yet completed"
    // (the recorded blocker, §8am). The pools are now per slot: a slot's pool is
    // reset when the slot is reused (its previous submission has been waited on at
    // that point, `begin`) and read immediately after that slot's fence signals
    // (`wait`/`wait_all`), so a read can never see an in-flight query.
    static PROBE_POOL: RefCell<Vec<vk::QueryPool>> = const { RefCell::new(Vec::new()) };
    /// Per slot: the device that **created** the slot's probe pool. A pool is a
    /// device object and a ring slot can change owner between devices (a free slot
    /// is picked by whichever device needs one), so ownership has to be tracked
    /// separately from `DEVICES` — destroying or stamping across devices is invalid
    /// and reads back as `NOT_READY`.
    static PROBE_OWNER: RefCell<Vec<Option<ash::Device>>> = const { RefCell::new(Vec::new()) };
    /// Per slot: the command buffer being recorded into it (for stamping).
    static PROBE_CMD: RefCell<Vec<vk::CommandBuffer>> = const { RefCell::new(Vec::new()) };
    /// Per slot: stamps recorded into the **open** batch.
    static PROBE_N: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    /// Per slot: `(query index, stage that starts here)` for the open batch; the
    /// final entry is the closing stamp (stage [`probe::CLOSE`]).
    static PROBE_LOG: RefCell<Vec<Vec<(u32, u8)>>> = const { RefCell::new(Vec::new()) };
    /// Per device: accumulated GPU nanoseconds per stage and the interval count.
    static PROBE_ACC: RefCell<Vec<ProbeAcc>> = const { RefCell::new(Vec::new()) };
}

/// Segments that may be in flight at once. Two are enough for a block (the
/// pre-router segment and the shared-expert segment); three leaves room for the
/// next block's first segment to be recorded while the previous one drains.
pub const RING: usize = 6;

/// One device's open recording.
#[derive(Clone, Copy, Debug)]
struct Open {
    dev: vk::Device,
    slot: usize,
    count: u32,
    /// The slot's command buffer, fixed at [`begin`]: the device's own ring buffer,
    /// or this thread's on a shared device (`Gpu::ring_cmd`). Held here so the
    /// per-dispatch path does no lookup.
    cmd: vk::CommandBuffer,
}

fn open_slot(gpu: &Gpu) -> Option<usize> {
    open_cmd(gpu).map(|(slot, _)| slot)
}

fn open_cmd(gpu: &Gpu) -> Option<(usize, vk::CommandBuffer)> {
    let dev = gpu.device.handle();
    OPEN.with(|o| o.borrow().iter().find(|x| x.dev == dev).map(|x| (x.slot, x.cmd)))
}

fn bump_open(gpu: &Gpu) {
    let dev = gpu.device.handle();
    OPEN.with(|o| {
        if let Some(x) = o.borrow_mut().iter_mut().find(|x| x.dev == dev) {
            x.count += 1;
        }
    });
}

/// Sets a batch budgets for.
///
/// A decode block records ~30 dispatches, but a **prefill chunk** records ~30 per
/// token in one submission (§8s): 4 096 sets covers a 128-token chunk, and the
/// pool is reset (not recreated) per batch. A bigger chunk needs a bigger pool.
pub const MAX_SETS: u32 = 8192;
/// Storage bindings per set (the widest kernel in this crate declares 5).
const MAX_BINDINGS: u32 = 8;

/// Dispatches after which the caller *may* consider flushing. Kept for callers
/// that want a bound on the descriptor pool; the engine's segmented pipeline does
/// not need it (a segment is one block's pre-router work).
pub const SOFT_LIMIT: u32 = 200;

fn ensure_slots() {
    POOLS.with(|p| {
        let mut p = p.borrow_mut();
        if p.is_empty() {
            p.extend(std::iter::repeat_n(vk::DescriptorPool::null(), RING));
        }
    });
    FENCES.with(|f| {
        let mut f = f.borrow_mut();
        if f.is_empty() {
            f.extend(std::iter::repeat_n(vk::Fence::null(), RING));
        }
    });
    DEVICES.with(|d| {
        let mut d = d.borrow_mut();
        if d.is_empty() {
            d.extend(std::iter::repeat_n(vk::Device::null(), RING));
        }
    });
    SLOT_DEV.with(|d| {
        let mut d = d.borrow_mut();
        if d.is_empty() {
            d.extend(std::iter::repeat_n(None, RING));
        }
    });
    PROBE_POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.is_empty() {
            p.extend(std::iter::repeat_n(vk::QueryPool::null(), RING));
        }
    });
    PROBE_OWNER.with(|o| {
        let mut o = o.borrow_mut();
        if o.is_empty() {
            o.extend(std::iter::repeat_n(None, RING));
        }
    });
    PROBE_CMD.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_empty() {
            c.extend(std::iter::repeat_n(vk::CommandBuffer::null(), RING));
        }
    });
    PROBE_N.with(|n| {
        let mut n = n.borrow_mut();
        if n.is_empty() {
            n.extend(std::iter::repeat_n(0u32, RING));
        }
    });
    PROBE_LOG.with(|l| {
        let mut l = l.borrow_mut();
        if l.is_empty() {
            l.extend(std::iter::repeat_with(Vec::new).take(RING));
        }
    });
}

/// Release every ring slot's descriptor pool and fence.
///
/// The pools outlive individual batches (they are reset, not recreated, which is
/// what removed the per-batch create/destroy), so they are freed once — by the
/// **last** session to drop, before the logical devices go away.
pub fn shutdown() -> Result<()> {
    // An unsubmitted recording is simply dropped: its command buffer and pool are
    // about to be freed with the device.
    OPEN.with(|o| o.borrow_mut().clear());
    ensure_slots();
    // The GPU-time probe's pools are per slot and owned by the device that created
    // them, so they are freed here with `PROBE_OWNER` — never with the slot's
    // *current* owner — and before the loop below, which `continue`s on a free slot.
    for slot in 0..RING {
        let pool = PROBE_POOL.with(|p| p.borrow()[slot]);
        if pool == vk::QueryPool::null() {
            continue;
        }
        if let Some(dev) = PROBE_OWNER.with(|o| o.borrow()[slot].clone()) {
            unsafe { dev.destroy_query_pool(pool, None) };
        }
        PROBE_POOL.with(|p| p.borrow_mut()[slot] = vk::QueryPool::null());
        PROBE_OWNER.with(|o| o.borrow_mut()[slot] = None);
    }
    let mut freed = 0usize;
    for slot in 0..RING {
        let dev = SLOT_DEV.with(|d| d.borrow()[slot].clone());
        let Some(dev) = dev else { continue };
        let fence = FENCES.with(|f| f.borrow()[slot]);
        if fence != vk::Fence::null() {
            unsafe {
                // Bounded: an unbounded wait on the display GPU rebooted this box
                // (AGENTS.md). On timeout the process aborts instead of hanging.
                let _ = dev.wait_for_fences(&[fence], true, 10_000_000_000);
            }
            FENCES.with(|f| f.borrow_mut()[slot] = vk::Fence::null());
        }
        let pool = POOLS.with(|p| p.borrow()[slot]);
        if pool != vk::DescriptorPool::null() {
            unsafe { dev.destroy_descriptor_pool(pool, None) };
            POOLS.with(|p| p.borrow_mut()[slot] = vk::DescriptorPool::null());
            freed += 1;
        }
        SLOT_DEV.with(|d| d.borrow_mut()[slot] = None);
    }
    for (_, pool, dev) in SET_POOLS.with(|v| std::mem::take(&mut *v.borrow_mut())) {
        unsafe { dev.destroy_descriptor_pool(pool, None) };
    }
    SETS.with(|c| c.borrow_mut().clear());
    if std::env::var_os("BRECCIA_TRACE_BATCH").is_some() {
        eprintln!("batch ring: freed {freed} descriptor pool(s) at shutdown");
    }
    Ok(())
}

/// Drop the fence **after the dispatch that is recorded next**, for the case where
/// the dispatch after it provably does not read anything that one wrote (one
/// dispatch only; the flag clears itself).
///
/// The distinction matters: the fence a recorded dispatch emits is the one that
/// separates it from its *successor*, so the flag has to be set before the
/// **first** member of an independent pair. Set before the second, it skips the
/// fence the pair still needs (§8z — that mistake measured 0/32 parity).
///
/// # When this is valid
///
/// Only when the next dispatch reads nothing the previous one wrote. The call
/// sites here are pairs that share an *input* and write disjoint *outputs* — the
/// shared expert's `gate`/`up` (both read `mixed`), the PLE `key`/`value`
/// projections (both read `ple_emb`) and the merged pass's `gate`/`up` GEMMs (both
/// read `pf.x`) — so the result is bit-identical by construction, not by luck.
/// A buffer-overlap heuristic would prune almost nothing here (most kernels are
/// in-place), which is why pruning is declared per call site
/// (`docs/phase-4/01-forward-pass.md` §8z).
pub fn skip_next_barrier() {
    // PARKED (§8z/§8aa). Measured, and it failed its own parity gate: with the
    // flag armed the gate read 20/32 at chunk 0 and a wrong top-1 logit at chunk
    // 128, because a *call site* is not a *dispatch* — the split-K forms record
    // several dispatches per `run`, so the flag is consumed by the first of them
    // and the reduction loses its fence. Re-arming it needs per-dispatch
    // dependency bookkeeping (each kernel declaring its reads and writes), not a
    // call-site annotation. The measured ceiling is small anyway: 0.98 ms (RX) /
    // 4.32 ms (MI50) of the GPU half for ~1 100 pruned barriers, which sits under
    // this box's CPU-contention noise. Kept as a no-op so the call sites document
    // where the independent pairs are, and so nothing unsafe is reachable.
    let _ = SKIP_BARRIER.with(|c| c.replace(false));
}

/// Is a batch open on this thread (on any device)?
pub fn is_open() -> bool {
    OPEN.with(|o| !o.borrow().is_empty())
}

/// Is a batch open on `gpu`'s device?
pub fn is_open_on(gpu: &Gpu) -> bool {
    open_slot(gpu).is_some()
}

/// Dispatches recorded into the open batches (all devices).
pub fn recorded() -> u32 {
    OPEN.with(|o| o.borrow().iter().map(|x| x.count).sum())
}

/// Dispatches recorded into `gpu`'s open batch (0 when none is open).
pub fn recorded_on(gpu: &Gpu) -> u32 {
    let dev = gpu.device.handle();
    OPEN.with(|o| {
        o.borrow()
            .iter()
            .find(|x| x.dev == dev)
            .map(|x| x.count)
            .unwrap_or(0)
    })
}

/// Dispatch barriers emitted into batches (process-wide), and how many the
/// per-resource tracker pruned. `barriers + pruned` is what the old
/// always-barrier shape would have emitted (§8al).
pub fn barriers() -> u64 {
    BARRIERS.with(|c| c.get())
}

/// Per-phase host nanoseconds inside `record`, in the order
/// `key+cache, set write, bind_pipeline, bind_set, push_constants, dispatch,
/// barrier, tracker`.
pub fn record_profile() -> [u64; 8] {
    std::array::from_fn(|i| PROF.with(|p| p[i].get()))
}

fn prof(i: usize, t0: std::time::Instant) {
    PROF.with(|p| p[i].set(p[i].get() + t0.elapsed().as_nanos() as u64));
}

/// Descriptor sets written once (cold) and dispatches that reused one (hot). The
/// ratio is the per-dispatch descriptor cost the persistent cache removes.
pub fn descriptor_sets() -> (u64, u64) {
    (SETS_NEW.with(|c| c.get()), SETS_HIT.with(|c| c.get()))
}

/// Barriers the per-resource tracker decided were not needed.
pub fn barriers_pruned() -> u64 {
    BARRIERS_SAVED.with(|c| c.get())
}

/// `BRECCIA_NO_BARRIER_TRACK=1`: emit the pre-§8al always-barrier shape (the A/B
/// arm; never a correctness aid — the tracker is the conservative-correct one).
///
/// Cached: this used to run **two `env::var_os` per dispatch** on the hot path
/// (60-78 ns of the 266 ns measured in `docs/research/11` §A2), and it is read
/// twice inside [`record`] plus once in [`end_submit`].
fn track_disabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BRECCIA_NO_BARRIER_TRACK").is_some())
}

/// `BRECCIA_RECORD_PROFILE=1`: take the five `Instant::now()` pairs that attribute
/// the per-phase cost inside [`record`] (§8aq).
///
/// **Off by default, and cached.** Those five pairs measure **165 ns of the
/// 266 ns** recorded dispatch (`docs/research/11` §A2) — 0.3 ms/token of pure
/// instrumentation. The one pair that stays is the `RECORD_NS` total, because that
/// is the number the CLI's `record` line quotes.
fn profile_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BRECCIA_RECORD_PROFILE").is_some())
}

/// The GPU-time probe: `BRECCIA_GPU_TIME=1` records one timestamp per **stage
/// boundary** inside every batched submission, so a decode step can be attributed
/// on the GPU timeline instead of guessed at from host timers.
///
/// The recipe is `docs/research/07` §A6, measured and confirmed in §C2:
///
/// * `vkCmdWriteTimestamp2(COMPUTE_SHADER)` — v1 `TOP/BOTTOM_OF_PIPE` costs
///   +210 ns/stamp and +5.8 % on a dispatch chain, so it is not used.
/// * **One pool per ring slot**, allocated with the slot, host-reset when the slot
///   is reused (its previous submission has been waited at that point), and read
///   **once, right after that slot's fence has signalled**. That is the fix for
///   the recorded blocker: the old shape shared one pool across slots and read a
///   slot's results while another slot's queries were still in flight, which is
///   `vkGetQueryPoolResults: A fence or query has not yet completed` (§8am).
/// * One stamp **per stage boundary**, which is the recipe's "two stamps per
///   stage": the closing stamp of stage *i* is the opening stamp of stage *i+1*,
///   and [`probe::close`] stamps the end of the last stage of a submission.
/// * **Never per dispatch**: 2 stamps × 1 843 dispatches measured **+14.6 %**
///   (§C2); ~15 stage boundaries per block is ~0.05-0.1 % of the token.
/// * A `VK_EXT_calibrated_timestamps` anchor (`Gpu::calibrate`) puts the tick
///   timeline on the host clock so the table can be set against `wait-gpu`.
pub mod probe {
    use super::*;

    /// Stage ids. One namespace shared by the engine (`graph.rs`) and the kernel
    /// families that have internal stages (`qsa`, `gdn`), so a mark inside a
    /// kernel and a mark in the graph land in the same column. Append-only: the
    /// ids are what a probe table is compared against between rounds (§8ax).
    pub const MAX: usize = 40;
    /// The closing stamp of a submission (not a stage).
    pub const CLOSE: u8 = 255;

    pub const OTHER: u8 = 0;
    pub const HC_READ: u8 = 1;
    pub const RMSNORM: u8 = 2;
    pub const QKV: u8 = 3;
    pub const ROPE: u8 = 4;
    pub const KV_APPEND: u8 = 5;
    pub const QSA_AUX: u8 = 6;
    pub const ATTN: u8 = 7;
    pub const ATTN_OUT: u8 = 8;
    pub const GDN_PROJ: u8 = 9;
    pub const GDN_SCAN: u8 = 10;
    pub const GDN_OUT: u8 = 11;
    pub const HC_WRITE: u8 = 12;
    pub const ROUTER: u8 = 13;
    pub const MOE_MAP: u8 = 14;
    pub const MOE_GATE: u8 = 15;
    pub const MOE_UP: u8 = 16;
    pub const MOE_SWIGLU: u8 = 17;
    pub const MOE_DOWN: u8 = 18;
    pub const MOE_COMBINE: u8 = 19;
    pub const SH_GATE: u8 = 20;
    pub const SH_UP: u8 = 21;
    pub const SH_SWIGLU: u8 = 22;
    pub const SH_DOWN: u8 = 23;
    pub const SH_SCALAR: u8 = 24;
    pub const FINAL_HC: u8 = 25;
    pub const HEAD: u8 = 26;
    pub const XFER: u8 = 27;
    pub const PLE: u8 = 28;
    /// The GDN projections, split (§8ax): `GDN_PROJ` is `attn_qkv`, then `z`
    /// (the output gate), and the two 48-row `beta`/`alpha` projections, which are
    /// 6 workgroups each and therefore pure latency.
    /// The HC_READ chain, split (§8ay): the RMSNorm, the 320-row bottleneck `down`
    /// (+ its split-K `reduce`), the 10 240-row `up`, and the gated mean. Two of the
    /// five are **one workgroup** (`norm`, `gated`), i.e. pure dispatch latency.
    pub const HC_NORM: u8 = 32;
    pub const HC_DOWN: u8 = 33;
    pub const HC_REDUCE: u8 = 34;
    pub const HC_UP: u8 = 35;
    pub const HC_GATED: u8 = 36;
    pub const GDN_Z: u8 = 29;
    pub const GDN_BETA: u8 = 30;
    pub const GDN_ALPHA: u8 = 31;

    /// Column names, indexed by id. `""` is a column that nothing stamps.
    pub const NAMES: [&str; MAX] = [
        "other",
        "hc-read",
        "rmsnorm",
        "qkv",
        "rope",
        "kv-append",
        "qsa-aux",
        "attn",
        "attn-out",
        "gdn-proj",
        "gdn-scan",
        "gdn-out",
        "hc-write",
        "router",
        "moe-map",
        "moe-gate",
        "moe-up",
        "moe-swiglu",
        "moe-down",
        "moe-combine",
        "shared-gate",
        "shared-up",
        "shared-swiglu",
        "shared-down",
        "shared-scalar",
        "final-hc",
        "head",
        "xfer",
        "ple",
        "gdn-z",
        "gdn-beta",
        "gdn-alpha",
        "hc-norm",
        "hc-down",
        "hc-reduce",
        "hc-up",
        "hc-gated",
        "",
        "",
        "",
    ];

    /// Stamps a single submission may hold. A decode block's prelude has ~15 stage
    /// boundaries (6 of them inside the mixer) and the head 3; 96 leaves room for a
    /// prefill chunk's per-token marks without ever silently dropping one.
    pub const CAP: u32 = 96;

    /// Is the probe on? Cached — it is called once per stage boundary.
    pub fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var_os("BRECCIA_GPU_TIME").is_some();
            if on {
                eprintln!(
                    "gpu-time probe: on (stage boundaries only, one pool per ring slot; \
                     never per dispatch — that costs +14.6 %, docs/research/07 §C2)"
                );
            }
            on
        })
    }

    fn acc_index(dev: vk::Device) -> usize {
        PROBE_ACC.with(|a| {
            let mut a = a.borrow_mut();
            match a.iter().position(|x| x.dev == dev) {
                Some(i) => i,
                None => {
                    a.push(ProbeAcc {
                        dev,
                        ns: [0; MAX],
                        n: [0; MAX],
                    });
                    a.len() - 1
                }
            }
        })
    }

    /// Ensure this slot has a pool on `gpu`'s device. A pool is a device object, so
    /// a slot whose owner changed (only possible after [`shutdown`] of a session)
    /// gets a new one.
    pub(super) fn ensure_pool(gpu: &Gpu, slot: usize) -> Result<()> {
        if !enabled() {
            return Ok(());
        }
        ensure_slots();
        let dev = gpu.device.handle();
        let old = PROBE_POOL.with(|p| p.borrow()[slot]);
        let owner = PROBE_OWNER.with(|o| o.borrow()[slot].as_ref().map(|d| d.handle()));
        if old != vk::QueryPool::null() && owner == Some(dev) {
            return Ok(());
        }
        if old != vk::QueryPool::null() {
            // Free it with the device that **created** it, never with the slot's new
            // owner: a ring slot is free for another device once the previous
            // submission has been waited, and a pool is a device object.
            if let Some(old_dev) = PROBE_OWNER.with(|o| o.borrow()[slot].clone()) {
                unsafe { old_dev.destroy_query_pool(old, None) };
            }
            PROBE_POOL.with(|p| p.borrow_mut()[slot] = vk::QueryPool::null());
        }
        let pool = gpu
            .query_pool(CAP)
            .map_err(|e| anyhow!("allocating a GPU-time probe pool: {e}"))?;
        PROBE_POOL.with(|p| p.borrow_mut()[slot] = pool);
        PROBE_OWNER.with(|o| o.borrow_mut()[slot] = Some(gpu.device.clone()));
        Ok(())
    }

    /// Start recording into `slot`: reset its pool and forget the previous batch's
    /// stamps. Called from [`begin`] **after** the slot's fence was waited, which is
    /// what makes a host-side reset legal and the pool reusable.
    pub(super) fn reset_slot(gpu: &Gpu, slot: usize, cmd: vk::CommandBuffer) -> Result<()> {
        if !enabled() {
            return Ok(());
        }
        ensure_pool(gpu, slot)?;
        let pool = PROBE_POOL.with(|p| p.borrow()[slot]);
        gpu.reset_queries(pool, CAP);
        PROBE_N.with(|n| n.borrow_mut()[slot] = 0);
        PROBE_LOG.with(|l| l.borrow_mut()[slot].clear());
        PROBE_CMD.with(|c| c.borrow_mut()[slot] = cmd);
        Ok(())
    }

    /// A stage boundary: `stage` starts here. Cheap when the probe is off (one
    /// cached-bool load).
    pub fn enter(gpu: &Gpu, stage: u8) {
        if !enabled() || stage as usize >= MAX {
            return;
        }
        let Some(slot) = open_slot(gpu) else {
            // Not inside a batch (a one-shot `wall_run` path): nothing to stamp.
            return;
        };
        let Some(dev) = SLOT_DEV.with(|d| d.borrow()[slot].clone()) else {
            return;
        };
        let n = PROBE_N.with(|c| c.borrow()[slot]);
        if n >= CAP {
            return;
        }
        let cmd = PROBE_CMD.with(|c| c.borrow()[slot]);
        let pool = PROBE_POOL.with(|p| p.borrow()[slot]);
        if cmd == vk::CommandBuffer::null() || pool == vk::QueryPool::null() {
            return;
        }
        unsafe {
            dev.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::COMPUTE_SHADER, pool, n);
        }
        PROBE_LOG.with(|l| l.borrow_mut()[slot].push((n, stage)));
        PROBE_N.with(|c| c.borrow_mut()[slot] = n + 1);
    }

    /// The closing stamp of the open batch (from [`end_submit`], which has already
    /// removed the slot from `OPEN` — so the slot is passed in, not looked up): the
    /// last stage's end. Without it the last region of every submission would be
    /// unattributed.
    pub(super) fn close(slot: usize) {
        if !enabled() {
            return;
        }
        ensure_slots();
        let Some(dev) = SLOT_DEV.with(|d| d.borrow()[slot].clone()) else {
            return;
        };
        let n = PROBE_N.with(|c| c.borrow()[slot]);
        if n >= CAP {
            return;
        }
        let cmd = PROBE_CMD.with(|c| c.borrow()[slot]);
        let pool = PROBE_POOL.with(|p| p.borrow()[slot]);
        if cmd == vk::CommandBuffer::null() || pool == vk::QueryPool::null() {
            return;
        }
        unsafe {
            dev.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::COMPUTE_SHADER, pool, n);
        }
        PROBE_LOG.with(|l| l.borrow_mut()[slot].push((n, CLOSE)));
        PROBE_N.with(|c| c.borrow_mut()[slot] = n + 1);
    }

    /// Read the slot's stamps **after its fence has signalled** and fold the
    /// intervals into the per-device accumulators. Called from [`wait`] and
    /// [`wait_all`]; never while a query can still be pending.
    pub(super) fn read_slot(gpu: &Gpu, slot: usize) {
        if !enabled() {
            return;
        }
        let log = PROBE_LOG.with(|l| std::mem::take(&mut l.borrow_mut()[slot]));
        if log.len() < 2 {
            return;
        }
        let pool = PROBE_POOL.with(|p| p.borrow()[slot]);
        if pool == vk::QueryPool::null() {
            return;
        }
        // A fence has signalled, so every query in the pool that was written is
        // complete: this cannot return NOT_READY unless the ring's invariants broke.
        let ticks = match gpu.timestamp_stamps(pool, log.len() as u32) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("gpu-time probe: slot {slot}: {e}");
                return;
            }
        };
        let dev = gpu.device.handle();
        let idx = acc_index(dev);
        let period = gpu.timestamp_period_ns;
        if std::env::var_os("BRECCIA_GPU_TIME_DUMP").is_some() {
            static DUMPED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let k = DUMPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if k < 400 {
                let span = ticks[log.len() - 1].saturating_sub(ticks[0]);
                let zeros = (0..log.len() - 1)
                    .filter(|k| ticks[k + 1] == ticks[*k])
                    .count();
                let stages: Vec<&str> = log
                    .iter()
                    .map(|(_, s)| {
                        if *s == CLOSE {
                            "close"
                        } else {
                            NAMES[(*s).min(MAX as u8 - 1) as usize]
                        }
                    })
                    .collect();
                eprintln!(
                    "probe dump #{k} slot {slot}: {} stamps, span {:.1}us, zero-deltas {zeros} [{}]",
                    log.len(),
                    span as f64 * period / 1e3,
                    stages.join(",")
                );
            }
        }
        let (ns, n) = intervals(&log, &ticks, period);
        PROBE_ACC.with(|a| {
            let mut a = a.borrow_mut();
            let acc = &mut a[idx];
            for s in 0..MAX {
                acc.ns[s] += ns[s];
                acc.n[s] += n[s];
            }
        });
    }

    /// Charge every `(stamp k, stamp k+1)` interval to the stage that **starts** at
    /// stamp `k`; the closing stamp ends the last stage. Pure, so the arithmetic is
    /// testable without a GPU (see `probe_interval_tests`).
    pub(super) fn intervals(
        log: &[(u32, u8)],
        ticks: &[u64],
        period_ns: f64,
    ) -> ([u64; MAX], [u32; MAX]) {
        let mut ns = [0u64; MAX];
        let mut n = [0u32; MAX];
        if log.len() < 2 {
            return (ns, n);
        }
        for k in 0..log.len() - 1 {
            let stage = log[k].1;
            if stage == CLOSE || stage as usize >= MAX {
                continue;
            }
            let (q0, q1) = (log[k].0 as usize, log[k + 1].0 as usize);
            if q1 >= ticks.len() || q0 >= ticks.len() {
                continue;
            }
            let d = ticks[q1].saturating_sub(ticks[q0]);
            ns[stage as usize] += (d as f64 * period_ns) as u64;
            n[stage as usize] += 1;
        }
        (ns, n)
    }

    /// GPU nanoseconds per stage accumulated for `gpu`'s device since the process
    /// started, with the number of intervals behind each.
    pub fn totals(gpu: &Gpu) -> ([u64; MAX], [u32; MAX]) {
        if !enabled() {
            return ([0; MAX], [0; MAX]);
        }
        let dev = gpu.device.handle();
        PROBE_ACC.with(|a| {
            a.borrow()
                .iter()
                .find(|x| x.dev == dev)
                .map(|x| (x.ns, x.n))
                .unwrap_or(([0; MAX], [0; MAX]))
        })
    }
}

/// Per-device accumulators of the GPU-time probe.
#[derive(Clone, Copy, Debug)]
struct ProbeAcc {
    dev: vk::Device,
    ns: [u64; probe::MAX],
    n: [u32; probe::MAX],
}

/// Host seconds spent recording since the last [`reset_record_seconds`].
pub fn record_seconds() -> f64 {
    RECORD_NS.with(|c| c.get()) as f64 / 1e9
}

/// Host seconds spent waiting for in-flight segments since the last
/// [`reset_record_seconds`].
pub fn wait_seconds() -> f64 {
    WAIT_NS.with(|c| c.get()) as f64 / 1e9
}

/// Clear the host-time accumulators (the engine calls this per run).
pub fn reset_record_seconds() {
    RECORD_NS.with(|c| c.set(0));
    RECORDED.with(|c| c.set(0));
    WAIT_NS.with(|c| c.set(0));
}

/// Dispatches recorded since the same reset point as [`record_seconds`] — the
/// denominator that matches that numerator, so their ratio is a real per-dispatch
/// price on one window (`docs/research/11` §A6, §8aq).
pub fn recorded_dispatches() -> u64 {
    RECORDED.with(|c| c.get())
}

/// Start recording a new segment: the next ring slot's command buffer.
///
/// If that slot's previous submission is still in flight this **waits for it**
/// (that is the ring's back-pressure — with the engine's two-segment block it
/// never actually blocks, because the slot was drained two segments ago).
pub fn begin(gpu: &Gpu) -> Result<()> {
    if is_open_on(gpu) {
        return Err(anyhow!(
            "a dispatch batch is already open on device {} of this thread",
            gpu.index
        ));
    }
    ensure_slots();
    // A new command buffer starts with no in-batch writes to order against.
    DIRTY.with(|d| d.borrow_mut().clear());
    let dev = gpu.device.handle();
    // Slots another open batch is recording into are off limits, and a slot whose
    // fence belongs to another device can be neither waited nor freed here (§8o:
    // two streams, one card each).
    //
    // Two passes: a slot with **no fence** first (nothing to wait for — that is
    // what keeps the pipeline deep), and only if every slot is busy do we wait on
    // one of our own. (Taking an own-but-in-flight slot first would make every
    // block wait for the previous block's submission and serialise the stream.)
    //
    // Slots are **device-affine**: a slot's descriptor pool is a device object
    // (it is reset per batch, not recreated), so a slot may only ever serve one
    // device. Sharing a slot across devices corrupts the recorded sets.
    let open_slots: Vec<usize> = OPEN.with(|o| o.borrow().iter().map(|x| x.slot).collect());
    let mut slot = usize::MAX;
    for i in 0..RING {
        if open_slots.contains(&i) {
            continue;
        }
        let owner = DEVICES.with(|d| d.borrow()[i]);
        if owner != vk::Device::null() && owner != dev {
            continue;
        }
        if FENCES.with(|f| f.borrow()[i]) == vk::Fence::null() {
            slot = i;
            break;
        }
    }
    if slot == usize::MAX {
        for i in 0..RING {
            if open_slots.contains(&i) {
                continue;
            }
            if DEVICES.with(|d| d.borrow()[i]) == dev {
                slot = i;
                break;
            }
        }
    }
    if slot == usize::MAX {
        return Err(anyhow!(
            "no free ring slot for device {} ({RING} slots, {} open, {} in flight)",
            gpu.index,
            open_slots.len(),
            FENCES.with(|f| f
                .borrow()
                .iter()
                .filter(|x| **x != vk::Fence::null())
                .count()),
        ));
    }
    let old = FENCES.with(|f| f.borrow()[slot]);
    if old != vk::Fence::null() {
        let t0 = std::time::Instant::now();
        gpu.wait_fence(old)
            .map_err(|e| anyhow!("waiting for ring slot {slot}: {e}"))?;
        WAIT_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos() as u64));
        FENCES.with(|f| f.borrow_mut()[slot] = vk::Fence::null());
        // This submission was reclaimed by back-pressure rather than by `wait`, so
        // its GPU-time stamps have not been read yet — read them now, while the
        // pool is still valid and its fence has signalled.
        probe::read_slot(gpu, slot);
    }
    // The slot keeps its descriptor pool across batches: `vkResetDescriptorPool`
    // is ~free against a create/destroy pair per batch, and a two-stream pair runs
    // ~96 batches per token (§8q).
    let pool = POOLS.with(|p| p.borrow()[slot]);
    if pool != vk::DescriptorPool::null() {
        unsafe {
            let _ = gpu
                .device
                .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty());
        }
    } else {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_SETS * MAX_BINDINGS)];
        let pool = unsafe {
            gpu.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_SETS)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| anyhow!("creating the batch descriptor pool: {e}"))?;
        POOLS.with(|p| p.borrow_mut()[slot] = pool);
    }
    SLOT_DEV.with(|d| d.borrow_mut()[slot] = Some(gpu.device.clone()));
    DEVICES.with(|d| d.borrow_mut()[slot] = dev);
    let cmd = gpu.ring_cmd(slot).map_err(|e| anyhow!("batch command buffer: {e}"))?;
    unsafe {
        gpu.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| anyhow!("resetting the batch command buffer: {e}"))?;
        gpu.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| anyhow!("beginning the batch command buffer: {e}"))?;
    }
    // The GPU-time probe's pool for this slot is reset here, **after** the fence
    // wait above: the previous submission has completed, so a host reset is legal
    // and the pool cannot hold an in-flight query (§8am's blocker).
    probe::reset_slot(gpu, slot, cmd)?;
    OPEN.with(|o| {
        o.borrow_mut().push(Open {
            dev: gpu.device.handle(),
            slot,
            count: 0,
            cmd,
        })
    });
    Ok(())
}

/// Close the recording and submit it **without waiting**; optionally have the
/// submission signal `(timeline, value)` when it completes.
///
/// This is the mid-batch sync point: the caller can submit a second segment
/// immediately and then wait on the timeline to learn that *this* segment is
/// done while the second one runs.
pub fn end_submit(gpu: &Gpu, signal: Option<(&Timeline, u64)>) -> Result<Segment> {
    let dev = gpu.device.handle();
    let Some((slot, cmd)) = OPEN.with(|o| {
        let mut o = o.borrow_mut();
        let i = o.iter().position(|x| x.dev == dev)?;
        let x = o.remove(i);
        Some((x.slot, x.cmd))
    }) else {
        return Err(anyhow!(
            "no dispatch batch is open on device {} of this thread",
            gpu.index
        ));
    };
    // **Cross-submission edge.** The tracker prunes the barrier *after* a dispatch
    // when nothing in the batch reads it, so the batch must close with everything
    // written visible to whatever the next submission (or the host) reads — queue
    // order alone does not make memory visible. One barrier per batch against the
    // ~30-100 it replaces.
    // The GPU-time probe's closing stamp for this submission: the last stage's end.
    // It goes in **before** the closing barrier so the barrier is attributed to the
    // stage that needed it, not to a phantom one. The slot is passed explicitly:
    // `OPEN` no longer holds it (it was taken above).
    probe::close(slot);
    if !track_disabled() && DIRTY.with(|d| !d.borrow().is_empty()) {
        unsafe { gpu.barrier_compute_on(cmd) };
        BARRIERS.with(|c| c.set(c.get() + 1));
        DIRTY.with(|d| d.borrow_mut().clear());
    }
    unsafe {
        gpu.device
            .end_command_buffer(cmd)
            .map_err(|e| anyhow!("ending the batch command buffer: {e}"))?;
    }
    let fence = gpu
        .submit_cmd(Qc::C, cmd, signal)
        .map_err(|e| anyhow!("submitting the batch: {e}"))?;
    FENCES.with(|f| f.borrow_mut()[slot] = fence);
    DEVICES.with(|d| d.borrow_mut()[slot] = gpu.device.handle());
    Ok(Segment(slot))
}

/// A handle to a submitted segment: [`wait`] on it to learn that its GPU work
/// is done (and to free its descriptor pool).
///
/// Two-stream decode keeps one of these per stream in flight, so the host can
/// wait for stream A's block while stream B's block is still executing — the
/// whole point of interleaving at the block level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment(usize);

impl Segment {
    pub fn slot(self) -> usize {
        self.0
    }
}

/// Wait for one segment (the device that submitted it must be in `gpus`).
/// Returns the host seconds spent waiting.
pub fn wait(gpus: &[&Gpu], seg: Segment) -> Result<f64> {
    ensure_slots();
    let slot = seg.0;
    let fence = FENCES.with(|f| f.borrow()[slot]);
    if fence == vk::Fence::null() {
        return Ok(0.0);
    }
    let gpu = DEVICES.with(|d| {
        d.borrow()
            .get(slot)
            .and_then(|dev| gpus.iter().find(|g| g.device.handle() == *dev))
            .copied()
    });
    let gpu = gpu.ok_or_else(|| {
        anyhow!("segment {slot} belongs to a device that was not passed to `wait`")
    })?;
    let t0 = std::time::Instant::now();
    gpu.wait_fence(fence)
        .map_err(|e| anyhow!("waiting for segment {slot}: {e}"))?;
    FENCES.with(|f| f.borrow_mut()[slot] = vk::Fence::null());
    let secs = t0.elapsed().as_secs_f64();
    WAIT_NS.with(|c| c.set(c.get() + (secs * 1e9) as u64));
    // The slot's fence has signalled: this is the one place its GPU-time stamps may
    // be read (§8am).
    probe::read_slot(gpu, slot);
    Ok(secs)
}

/// Close the recording, submit and wait for **every** segment in flight.
/// Returns the host seconds spent waiting.
pub fn end(gpu: &Gpu) -> Result<f64> {
    let seg = end_submit(gpu, None)?;
    wait(&[gpu], seg)
}

/// Wait for every submission still in flight (and free its descriptor pool).
/// Returns the host seconds spent waiting.
///
/// `gpus` is the set of devices the caller wants drained: a fence belongs to the
/// device that created it, so each slot is waited on with its own device, and a
/// slot belonging to a device **outside** the list is left alone. That is what
/// makes a flush scoped: with one stream per card, stream A's flush must not
/// wait for stream B's GPU work (§8o).
pub fn wait_all(gpus: &[&Gpu]) -> Result<f64> {
    // An open batch is only an error when it is one of *these* devices': recorded
    // but unsubmitted work on a device being waited on would be silently skipped.
    // With one stream per card, another session's open batch is not our business.
    if is_open() {
        let open: Vec<vk::Device> = OPEN.with(|o| o.borrow().iter().map(|x| x.dev).collect());
        if gpus.iter().any(|g| open.contains(&g.device.handle())) {
            return Err(anyhow!(
                "wait_all with an open batch on one of these devices: close it first \
                 (its work would be lost)"
            ));
        }
    }
    ensure_slots();
    let t0 = std::time::Instant::now();
    for slot in 0..RING {
        let fence = FENCES.with(|f| f.borrow()[slot]);
        if fence != vk::Fence::null() {
            let gpu = DEVICES.with(|d| {
                d.borrow()
                    .get(slot)
                    .and_then(|dev| gpus.iter().find(|g| g.device.handle() == *dev))
                    .copied()
            });
            let Some(gpu) = gpu else {
                // Another session's device owns this slot.
                continue;
            };
            // Only now: the recorded command buffer references these sets, so
            // freeing the pool before the submission completes is a use-after-free
            // (`VUID-vkCmdDispatch-None-08114`).
            gpu.wait_fence(fence)
                .map_err(|e| anyhow!("waiting for ring slot {slot}: {e}"))?;
            FENCES.with(|f| f.borrow_mut()[slot] = vk::Fence::null());
            probe::read_slot(gpu, slot);
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    WAIT_NS.with(|c| c.set(c.get() + (secs * 1e9) as u64));
    Ok(secs)
}

/// Submissions still in flight on this thread.
pub fn in_flight() -> usize {
    ensure_slots();
    FENCES.with(|f| {
        f.borrow()
            .iter()
            .filter(|x| **x != vk::Fence::null())
            .count()
    })
}

/// Record a buffer copy into the open batch: `dst[dst_off..] = src[src_off..]`.
///
/// This is how a device-local result reaches the host **without a submission of
/// its own**: the copy rides the segment, and the segment's marker (or fence)
/// is what tells the host the bytes are there to read out of the staging
/// mapping.
pub fn record_copy(
    gpu: &Gpu,
    dst: &Buf,
    dst_off: u64,
    src: &Buf,
    src_off: u64,
    len: u64,
) -> Result<()> {
    if !is_open() {
        return Err(anyhow!("record_copy without an open batch"));
    }
    if src_off.saturating_add(len) > src.size {
        return Err(anyhow!(
            "record_copy: {len} bytes at {src_off} is outside a {}-byte source",
            src.size
        ));
    }
    if dst_off.saturating_add(len) > dst.size {
        return Err(anyhow!(
            "record_copy: {len} bytes at {dst_off} is outside a {}-byte destination",
            dst.size
        ));
    }
    let (_, cmd) =
open_cmd(gpu).ok_or_else(|| anyhow!("record_copy without an open batch"))?;
    let region = vk::BufferCopy::default()
        .src_offset(src_off)
        .dst_offset(dst_off)
        .size(len);
    unsafe {
        gpu.copy_dependency(cmd, true);
        gpu.device
            .cmd_copy_buffer(cmd, src.buf, dst.buf, std::slice::from_ref(&region));
        // See `record_copies`: the compute-scoped dispatch barrier does not cover
        // COPY -> COMPUTE, so the copy closes its own dependency.
        gpu.copy_dependency(cmd, false);
        // The copy's own barriers fence every prior write, so the dirty set starts
        // empty for the dispatches that follow.
        DIRTY.with(|d| d.borrow_mut().clear());
    }
    Ok(())
}

/// Record **many** copies into the open batch as one `vkCmdCopyBuffer` region
/// array, behind a single transfer barrier.
///
/// [`record_copy`] costs 1.0-1.5 µs of host time per call (a barrier packet plus
/// a copy packet); a region array costs ~0.13 µs per added region
/// (`docs/research/02-vulkan-followups-and-powerinfer.md` §C). The graph's chunks
/// are copy-dense — five lane-inputs per token, six per merged-prefill token, plus
/// one per expert group — so this replaces ~5-8 record calls with one.
///
/// `items` are `(dst, dst_offset, src, src_offset, len)`; every range is bounds
/// checked against its buffer (an out-of-range copy is a GPU page fault, not a
/// wrong number).
pub fn record_copies(gpu: &Gpu, items: &[(&Buf, u64, &Buf, u64, u64)]) -> Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    // A/B arm for the region-array change (`docs/research/02-…` §C): one
    // `record_copy` per region, i.e. the pre-§8w shape.
    if std::env::var_os("BRECCIA_NO_COPY_ARRAYS").is_some() {
        for (dst, dst_off, src, src_off, len) in items {
            record_copy(gpu, dst, *dst_off, src, *src_off, *len)?;
        }
        return Ok(());
    }
    let (_, cmd) =
open_cmd(gpu).ok_or_else(|| anyhow!("record_copies without an open batch"))?;
    // A region array is per source *and* per destination buffer: Vulkan's
    // `cmd_copy_buffer` takes one (src, dst) pair, so group by pair and emit one
    // packet per group with every region of that group inside it.
    let mut groups: Vec<(vk::Buffer, vk::Buffer, Vec<vk::BufferCopy>)> = Vec::new();
    for (dst, dst_off, src, src_off, len) in items {
        if src_off.saturating_add(*len) > src.size {
            return Err(anyhow!(
                "record_copies: {len} bytes at {src_off} is outside a {}-byte source",
                src.size
            ));
        }
        if dst_off.saturating_add(*len) > dst.size {
            return Err(anyhow!(
                "record_copies: {len} bytes at {dst_off} is outside a {}-byte destination",
                dst.size
            ));
        }
        let region = vk::BufferCopy::default()
            .src_offset(*src_off)
            .dst_offset(*dst_off)
            .size(*len);
        // Vulkan requires the regions of one `cmd_copy_buffer` to not overlap, so an
        // in-place copy (source == destination) gets a packet of its own even when
        // the ranges happen to be disjoint.
        if src.buf == dst.buf {
            groups.push((src.buf, dst.buf, vec![region]));
            continue;
        }
        match groups
            .iter_mut()
            .find(|(sb, db, _)| *sb == src.buf && *db == dst.buf)
        {
            Some((_, _, regions)) => regions.push(region),
            None => groups.push((src.buf, dst.buf, vec![region])),
        }
    }
    if groups.is_empty() {
        return Ok(());
    }
    unsafe {
        // Before: the writes this copy consumes (shader or earlier transfer).
        gpu.copy_dependency(cmd, true);
        for (src, dst, regions) in &groups {
            gpu.device.cmd_copy_buffer(cmd, *src, *dst, regions);
        }
        // **After**: the dispatches that read the copied bytes. The per-dispatch
        // fence is compute-scoped now (`Gpu::barrier_compute_on`, §8w), which by
        // construction does not cover COPY -> COMPUTE, so a copy that feeds a
        // dispatch must close its own dependency. Without this the streamed-expert
        // GEMMs race their uploads — measured as non-deterministic logits on the
        // MI50 (max |Δlogit| 2.46 between two identical prompts, §8w/§8x).
        gpu.copy_dependency(cmd, false);
        // The copy's own barriers fence every prior write, so the dirty set starts
        // empty for the dispatches that follow.
        DIRTY.with(|d| d.borrow_mut().clear());
    }
    Ok(())
}

/// Record a `BOTTOM_OF_PIPE` timestamp at `query` in the open batch.
///
/// Two of these around a batch (one just after [`begin`], one just before
/// [`end`]) give the submission's **own** GPU span, which is the number to
/// compare against its fence wait: the difference between them is the
/// submission latency the host pays on top of the GPU's work.
pub fn record_timestamp(gpu: &Gpu, pool: vk::QueryPool, query: u32) -> Result<()> {
    let (_, cmd) =
open_cmd(gpu).ok_or_else(|| anyhow!("record_timestamp without an open batch"))?;
    unsafe {
        gpu.device
            .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, query);
    }
    Ok(())
}

/// Barrier making prior shader writes visible to the **host**.
/// Required between a copy of a device-local result into a host-visible staging
/// buffer and the host reading that mapping after a timeline marker (or fence).
pub fn record_barrier_to_host(gpu: &Gpu) -> Result<()> {
    if !is_open() {
        return Err(anyhow!("record_barrier_to_host without an open batch"));
    }
    let (_, cmd) =
open_cmd(gpu).ok_or_else(|| anyhow!("record_barrier_to_host without an open batch"))?;
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    unsafe {
        gpu.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        );
    }
    Ok(())
}

/// The coarse `ALL_COMMANDS -> ALL_COMMANDS` memory barrier. The copy paths use
/// [`Gpu::copy_dependency`] instead (precise stages, no global stall); this stays
/// for callers that need to fence everything at once.
#[allow(dead_code)]
unsafe fn transfer_barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe {
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        );
    }
}

/// Record one dispatch into the open batch, followed by a full barrier.
///
/// `bufs` are the dispatch's storage buffers in binding order; a fresh
/// descriptor set is allocated for them (see the module docs for why it cannot
/// be the kernel's own set).
/// The persistent descriptor pool for `gpu`'s device (created on first use, freed by
/// [`shutdown`]). Sized for the distinct `(pipeline, buffers)` combinations a session
/// has — a few hundred, not one per dispatch.
fn set_pool(gpu: &Gpu) -> Result<vk::DescriptorPool> {
    ensure_slots();
    let dev = gpu.device.handle();
    if let Some((_, p, _)) =
        SET_POOLS.with(|v| v.borrow().iter().find(|(d, _, _)| *d == dev).cloned())
    {
        return Ok(p);
    }
    let sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(MAX_SETS * MAX_BINDINGS)];
    let pool = unsafe {
        gpu.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(MAX_SETS * 4),
            None,
        )
    }
    .map_err(|e| anyhow!("creating the persistent descriptor pool: {e}"))?;
    SET_POOLS.with(|v| v.borrow_mut().push((dev, pool, gpu.device.clone())));
    Ok(pool)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    gpu: &Gpu,
    kernel: &Kernel,
    bufs: &[&Buf],
    pc: &[u8],
    grid: (u32, u32, u32),
) -> Result<()> {
    // **One** clock pair, and it is the total the CLI's `record` line quotes. The
    // five per-phase pairs that used to sit here cost 165 ns of the 266 ns recorded
    // dispatch (`docs/research/11` §A2); they are now behind `BRECCIA_RECORD_PROFILE`
    // (checked once, cached), which takes the path from 266 to ~70 ns.
    let t0 = std::time::Instant::now();
    let (slot, cmd) =
open_cmd(gpu).ok_or_else(|| anyhow!("recording a dispatch without an open batch"))?;
    let pool = POOLS.with(|p| p.borrow()[slot]);
    if pool == vk::DescriptorPool::null() {
        return Err(anyhow!("recording a dispatch without an open batch"));
    }
    // **The set is written once per (pipeline, buffer set), not once per
    // dispatch** (§8ao). The decode shape is frozen per config, so token *n* binds
    // exactly the buffers token *n-1* did and the set can be cached. A cached set
    // is only ever written before any command buffer references it, which is the
    // rule the module docs impose on `vkUpdateDescriptorSets`; the per-batch pool's
    // fresh set was the ~3 µs/dispatch this removes.
    let prof_on = profile_on();
    let mut t_ph = prof_on.then(std::time::Instant::now);
    let key: (u64, Vec<vk::Buffer>) = (
        ash::vk::Handle::as_raw(kernel.pipe),
        bufs.iter().map(|b| b.buf).collect(),
    );
    let cached = SETS.with(|c| c.borrow().get(&key).copied());
    let persistent = if track_disabled() {
        None
    } else {
        cached.or_else(|| {
            let pool = set_pool(gpu).ok()?;
            let layouts = [kernel.dsl];
            let sets = unsafe {
                gpu.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
            }
            .ok()?;
            let set = sets[0];
            let infos: Vec<vk::DescriptorBufferInfo> = bufs
                .iter()
                .map(|b| vk::DescriptorBufferInfo {
                    buffer: b.buf,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
            SETS.with(|c| c.borrow_mut().insert(key, set));
            SETS_NEW.with(|c| c.set(c.get() + 1));
            Some(set)
        })
    };
    if persistent.is_some() && cached.is_some() {
        SETS_HIT.with(|c| c.set(c.get() + 1));
    }
    if let Some(t) = t_ph.take() {
        prof(0, t);
    }
    unsafe {
        let batch_set;
        let set = match persistent {
            Some(s) => s,
            None => {
                // The A/B arm (`BRECCIA_NO_BARRIER_TRACK=1`) and any allocation
                // failure keep the per-dispatch set from the batch's own pool.
                let layouts = [kernel.dsl];
                let sets = gpu
                    .device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(pool)
                            .set_layouts(&layouts),
                    )
                    .map_err(|e| anyhow!("allocating a batch descriptor set: {e}"))?;
                batch_set = sets[0];
                let infos: Vec<vk::DescriptorBufferInfo> = bufs
                    .iter()
                    .map(|b| vk::DescriptorBufferInfo {
                        buffer: b.buf,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    })
                    .collect();
                let writes: Vec<vk::WriteDescriptorSet> = infos
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(batch_set)
                            .dst_binding(i as u32)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(info))
                    })
                    .collect();
                gpu.device.update_descriptor_sets(&writes, &[]);
                batch_set
            }
        };

        if prof_on {
            t_ph = Some(std::time::Instant::now());
        }
        gpu.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipe);
        if let Some(t) = t_ph.take() {
            prof(2, t);
        }
        if prof_on {
            t_ph = Some(std::time::Instant::now());
        }
        gpu.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            kernel.layout,
            0,
            &[set],
            &[],
        );
        if let Some(t) = t_ph.take() {
            prof(3, t);
        }
        gpu.device
            .cmd_push_constants(cmd, kernel.layout, vk::ShaderStageFlags::COMPUTE, 0, pc);
        // **Per-resource dependency tracking** (`docs/phase-4` §8al, ncnn's model).
        //
        // The old shape emitted a compute→compute barrier after *every* dispatch,
        // which is correct but prices ~4 800 barriers/token (§8af). A barrier is
        // only needed when the dispatch about to be recorded touches a buffer that
        // an earlier dispatch **wrote** and no barrier has separated since. The
        // dirty set is conservative in one direction — every bound buffer is
        // treated as possibly written, because a kernel does not declare its
        // writes — so this can never drop a required barrier; it only drops the
        // ones where the two dispatches' storage sets are disjoint.
        //
        // This is per *dispatch*, not per call site: that distinction is exactly
        // why `skip_next_barrier` failed its parity gate (§8z/§8aa).
        let need = if SKIP_BARRIER.with(|c| c.replace(false)) || track_disabled() {
            false
        } else {
            DIRTY.with(|d| {
                let d = d.borrow();
                let hit = bufs.iter().any(|b| d.contains(&b.buf));
                if hit {
                    // The barrier below fences everything recorded so far.
                    drop(d);
                    DIRTY.with(|d| d.borrow_mut().clear());
                }
                hit
            })
        };
        if need {
            if prof_on {
                t_ph = Some(std::time::Instant::now());
            }
            gpu.barrier_compute_on(cmd);
            if let Some(t) = t_ph.take() {
                prof(6, t);
            }
            BARRIERS.with(|c| c.set(c.get() + 1));
        } else if track_disabled() {
            // A/B arm: the pre-§8al always-barrier shape.
            gpu.barrier_compute_on(cmd);
            BARRIERS.with(|c| c.set(c.get() + 1));
        } else {
            BARRIERS_SAVED.with(|c| c.set(c.get() + 1));
        }
        gpu.device
            .cmd_dispatch(cmd, grid.0.max(1), grid.1.max(1), grid.2.max(1));
        // Everything bound here may now have been written.
        DIRTY.with(|d| {
            let mut d = d.borrow_mut();
            for b in bufs {
                if !d.contains(&b.buf) {
                    d.push(b.buf);
                }
            }
        });
    }
    bump_open(gpu);
    RECORD_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos() as u64));
    RECORDED.with(|c| c.set(c.get() + 1));
    Ok(())
}

#[cfg(test)]
mod probe_interval_tests {
    use super::probe::{intervals, ATTN, CLOSE, HC_READ, HEAD, QKV};

    /// The interval arithmetic behind the GPU-time table: each interval is charged
    /// to the stage that **starts** at its first stamp, ticks scale by
    /// `timestampPeriod`, and the closing stamp only ends the last stage.
    #[test]
    fn intervals_are_charged_to_the_stage_that_starts_them() {
        // 2 ns/tick, chosen so the arithmetic is checkable by eye.
        let log = [(0u32, HC_READ), (1, QKV), (2, ATTN), (3, HEAD), (4, CLOSE)];
        let ticks = [100u64, 150, 210, 260, 400];
        let (ns, n) = intervals(&log, &ticks, 2.0);
        assert_eq!(ns[HC_READ as usize], 100); // 50 ticks x 2 ns
        assert_eq!(ns[QKV as usize], 120); // 60 ticks
        assert_eq!(ns[ATTN as usize], 100); // 50 ticks
        assert_eq!(ns[HEAD as usize], 280); // 140 ticks, up to the closing stamp
        assert_eq!(n[HC_READ as usize], 1);
        assert_eq!(n[HEAD as usize], 1);
        // Nothing else is charged, and the sum telescopes to the whole span.
        let total: u64 = ns.iter().sum();
        assert_eq!(total, (ticks[4] - ticks[0]) * 2);
    }

    /// A stamp sequence that is not monotonic (a reused pool's stale tail) must not
    /// wrap: `saturating_sub` gives a zero-length interval rather than 2^64 ns.
    #[test]
    fn a_stale_tail_contributes_no_time() {
        let log = [(0u32, HC_READ), (1, CLOSE)];
        let ticks = [500u64, 100];
        let (ns, n) = intervals(&log, &ticks, 10.0);
        assert_eq!(ns[HC_READ as usize], 0);
        assert_eq!(n[HC_READ as usize], 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_batch_is_open_on_a_fresh_thread() {
        assert!(!is_open());
        assert_eq!(recorded(), 0);
        assert_eq!(in_flight(), 0);
    }

    /// The probe is off unless asked for, and asking for it is a cached decision:
    /// the hot path only ever sees a `bool` load.
    #[test]
    fn the_probe_is_off_by_default() {
        assert!(!super::probe::enabled() || std::env::var_os("BRECCIA_GPU_TIME").is_some());
    }

    #[test]
    fn the_probe_stage_ids_fit_the_table() {
        assert!(super::probe::NAMES.len() == super::probe::MAX);
        assert!((super::probe::PLE as usize) < super::probe::MAX);
        assert!(super::probe::CLOSE as usize >= super::probe::MAX);
    }
}
