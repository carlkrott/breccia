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
    /// Counters: sets written (cold) vs dispatches that reused one (hot).
    static SETS_NEW: Cell<u64> = const { Cell::new(0) };
    static SETS_HIT: Cell<u64> = const { Cell::new(0) };
    /// Barriers emitted into batches since the process started, and the subset the
    /// per-resource tracker **pruned** for comparison.
    static BARRIERS: Cell<u64> = const { Cell::new(0) };
    static BARRIERS_SAVED: Cell<u64> = const { Cell::new(0) };
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
}

fn open_slot(gpu: &Gpu) -> Option<usize> {
    let dev = gpu.device.handle();
    OPEN.with(|o| o.borrow().iter().find(|x| x.dev == dev).map(|x| x.slot))
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
fn track_disabled() -> bool {
    std::env::var_os("BRECCIA_NO_BARRIER_TRACK").is_some()
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
    WAIT_NS.with(|c| c.set(0));
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
    let cmd = gpu.cmds_c[1 + slot];
    unsafe {
        gpu.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| anyhow!("resetting the batch command buffer: {e}"))?;
        gpu.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| anyhow!("beginning the batch command buffer: {e}"))?;
    }
    OPEN.with(|o| {
        o.borrow_mut().push(Open {
            dev: gpu.device.handle(),
            slot,
            count: 0,
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
    let slot = OPEN.with(|o| {
        let mut o = o.borrow_mut();
        match o.iter().position(|x| x.dev == dev) {
            Some(i) => o.remove(i).slot,
            None => usize::MAX,
        }
    });
    if slot == usize::MAX {
        return Err(anyhow!(
            "no dispatch batch is open on device {} of this thread",
            gpu.index
        ));
    }
    let cmd = gpu.cmds_c[1 + slot];
    // **Cross-submission edge.** The tracker prunes the barrier *after* a dispatch
    // when nothing in the batch reads it, so the batch must close with everything
    // written visible to whatever the next submission (or the host) reads — queue
    // order alone does not make memory visible. One barrier per batch against the
    // ~30-100 it replaces.
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
    let slot = open_slot(gpu).ok_or_else(|| anyhow!("record_copy without an open batch"))?;
    let cmd = gpu.cmds_c[1 + slot];
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
    let slot = open_slot(gpu).ok_or_else(|| anyhow!("record_copies without an open batch"))?;
    let cmd = gpu.cmds_c[1 + slot];
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
    let slot = open_slot(gpu).ok_or_else(|| anyhow!("record_timestamp without an open batch"))?;
    let cmd = gpu.cmds_c[1 + slot];
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
    let slot =
        open_slot(gpu).ok_or_else(|| anyhow!("record_barrier_to_host without an open batch"))?;
    let cmd = gpu.cmds_c[1 + slot];
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
    let t0 = std::time::Instant::now();
    let slot =
        open_slot(gpu).ok_or_else(|| anyhow!("recording a dispatch without an open batch"))?;
    let cmd = gpu.cmds_c[1 + slot];
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

        gpu.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipe);
        gpu.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            kernel.layout,
            0,
            &[set],
            &[],
        );
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
            gpu.barrier_compute_on(cmd);
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
    Ok(())
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
}
