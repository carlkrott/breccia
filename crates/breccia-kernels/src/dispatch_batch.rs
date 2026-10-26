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
    /// Cumulative host seconds spent inside `record` (set allocation, descriptor
    /// write, bind, push constants, dispatch, barrier) — the cost a pre-recorded
    /// command buffer would remove.
    static RECORD_NS: Cell<u64> = const { Cell::new(0) };
    /// Host seconds spent waiting for slots in [`wait_all`]/[`begin`].
    static WAIT_NS: Cell<u64> = const { Cell::new(0) };
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

/// Sets a batch budgets for. A segment holds at most a block's ~30 dispatches,
/// so 256 is a bounded, predictable budget with room to spare.
pub const MAX_SETS: u32 = 256;
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
    let dev = gpu.device.handle();
    // Slots another open batch is recording into are off limits, and a slot whose
    // fence belongs to another device can be neither waited nor freed here (§8o:
    // two streams, one card each).
    //
    // Two passes: a slot with **no fence** first (nothing to wait for — that is
    // what keeps the pipeline deep), and only if every slot is busy do we wait on
    // one of our own. (Taking an own-but-in-flight slot first would make every
    // block wait for the previous block's submission and serialise the stream.)
    let open_slots: Vec<usize> = OPEN.with(|o| o.borrow().iter().map(|x| x.slot).collect());
    let mut slot = usize::MAX;
    for i in 0..RING {
        if open_slots.contains(&i) {
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
    let old_pool = POOLS.with(|p| p.borrow()[slot]);
    if old_pool != vk::DescriptorPool::null() {
        unsafe { gpu.device.destroy_descriptor_pool(old_pool, None) };
        POOLS.with(|p| p.borrow_mut()[slot] = vk::DescriptorPool::null());
    }
    let cmd = gpu.cmds_c[1 + slot];
    unsafe {
        gpu.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| anyhow!("resetting the batch command buffer: {e}"))?;
        gpu.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| anyhow!("beginning the batch command buffer: {e}"))?;

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_SETS * MAX_BINDINGS)];
        let pool = gpu
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_SETS)
                    .pool_sizes(&sizes),
                None,
            )
            .map_err(|e| anyhow!("creating the batch descriptor pool: {e}"))?;
        POOLS.with(|p| p.borrow_mut()[slot] = pool);
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
    let pool = POOLS.with(|p| p.borrow()[slot]);
    if pool != vk::DescriptorPool::null() {
        unsafe { gpu.device.destroy_descriptor_pool(pool, None) };
        POOLS.with(|p| p.borrow_mut()[slot] = vk::DescriptorPool::null());
    }
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
            let pool = POOLS.with(|p| p.borrow()[slot]);
            if pool != vk::DescriptorPool::null() {
                unsafe { gpu.device.destroy_descriptor_pool(pool, None) };
                POOLS.with(|p| p.borrow_mut()[slot] = vk::DescriptorPool::null());
            }
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
        transfer_barrier(&gpu.device, cmd);
        gpu.device
            .cmd_copy_buffer(cmd, src.buf, dst.buf, std::slice::from_ref(&region));
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
    unsafe {
        let layouts = [kernel.dsl];
        let sets = gpu
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
            .map_err(|e| anyhow!("allocating a batch descriptor set: {e}"))?;
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
        gpu.device.update_descriptor_sets(&writes, &[]);

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
        gpu.device
            .cmd_dispatch(cmd, grid.0.max(1), grid.1.max(1), grid.2.max(1));
        Gpu::barrier_all(&gpu.device, cmd);
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
