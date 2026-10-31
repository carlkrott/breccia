//! Phase 2 (M1b) — asynchronous transfer layer.
//!
//! This is engine infrastructure, not a benchmark: [`TransferRing`] is the shape
//! M3 (expert streaming) and M5 (KV paging) are expected to drive directly.
//!
//! * **Timeline-semaphore DMA ring** — `N` slots × fixed-size device-local buffers,
//!   `N` pinned host staging chunks, `N` transfer command buffers. Slot reuse is
//!   gated by timeline values on the dedicated transfer queue family (family 1 on
//!   both cards here: 4 queues, timeline support). No `vkQueueWaitIdle` in steady
//!   state; the only blocking waits are (a) a slot whose previous consumer has not
//!   released it yet, and (b) the final drain.
//! * **Pinned host staging** — `HOST_VISIBLE | HOST_COHERENT` allocations, mapped
//!   once and kept mapped (the driver pins them while they are device-visible;
//!   Vulkan's equivalent of registered host memory, not pageable `malloc`).
//! * **Compute/transfer overlap primitive** — two timeline semaphores
//!   (`arrive`: transfer → compute, `release`: compute → transfer) let a producer
//!   thread stream uploads while the compute queue consumes the same slots without
//!   any CPU round-trip:
//!
//!   ```text
//!   ring.submit_upload(slot, host_slice, v);        // transfer queue, signals arrive>=v
//!   ring.submit_compute(slot, v, |dev, cmd| {..});  // compute queue, waits arrive>=v, signals release>=v
//!   ring.wait(v);                                   // host drain (once, at the end)
//!   ```
//!
//!   A one-value-per-round mapping keeps both timelines strictly increasing and
//!   makes "which round is this slot on" trivially checkable.
//!
//! Fallback: devices without a distinct transfer-capable family still work — the
//! ring uses whatever queue [`crate::ctx::Gpu`] selected (`second queue of the
//! compute family`, else the compute queue itself) and reports the mode, so the
//! same code path is exercised on the iGPU. Overlap is then expected to be ~0.

use ash::vk;
use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::time::Instant;

use crate::ctx::{find_mem_type, vk, Buf, Gpu, R};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Fixed geometry of a ring. `slot_bytes` is one DMA chunk = one expert blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingConfig {
    pub slots: u32,
    pub slot_bytes: u64,
}

impl RingConfig {
    pub fn new(slots: u32, slot_bytes: u64) -> Self {
        Self { slots, slot_bytes }
    }

    /// Reject geometry the Vulkan calls cannot express.
    pub fn validate(&self) -> R<()> {
        if self.slots == 0 {
            return Err("ring needs at least 1 slot".to_string());
        }
        if self.slots > 64 {
            return Err(format!("ring slots {} exceeds the 64-slot cap", self.slots));
        }
        if self.slot_bytes == 0 {
            return Err("ring slot_bytes must be > 0".to_string());
        }
        if !self.slot_bytes.is_multiple_of(4) {
            return Err(format!(
                "ring slot_bytes {} must be a multiple of 4 (vkCmdCopyBuffer alignment)",
                self.slot_bytes
            ));
        }
        Ok(())
    }
}

/// Ring counters. `stalls` counts *slot-reuse* blocks (a producer that had to wait
/// for a consumer, or for its own earlier DMA, before rewriting staging/slot);
/// `host_waits` counts every blocking timeline wait including the final drain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RingStats {
    pub submits: u64,
    pub consumer_submits: u64,
    pub bytes: u64,
    pub slot_reuses: u64,
    pub stalls: u64,
    pub stall_seconds: f64,
    pub host_waits: u64,
    pub host_wait_seconds: f64,
}

impl RingStats {
    /// Effective upload bandwidth over a wall-clock window that includes this
    /// ring's submissions.
    pub fn effective_gbps(&self, seconds: f64) -> f64 {
        if seconds <= 0.0 {
            return 0.0;
        }
        self.bytes as f64 / seconds / 1e9
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gate {
    /// Slot never used.
    None,
    /// A copy at `v` is (or may be) in flight into this slot.
    Arrive(u64),
    /// A consumer that read the slot must finish (`release >= v`) before reuse.
    Release(u64),
}

// ---------------------------------------------------------------------------
// Ring
// ---------------------------------------------------------------------------

/// Reusable DMA ring. Borrows the device for its lifetime; [`TransferRing::destroy`]
/// releases its Vulkan objects.
pub struct TransferRing<'a> {
    gpu: &'a Gpu,
    cfg: RingConfig,
    staging: Vec<Buf>,
    staging_ptr: Vec<*mut u8>,
    slots: Vec<Buf>,
    cmd_pool_x: vk::CommandPool,
    cmd_pool_c: vk::CommandPool,
    cmd_x: Vec<vk::CommandBuffer>,
    cmd_c: Vec<vk::CommandBuffer>,
    arrive: vk::Semaphore,
    release: vk::Semaphore,
    gates: Vec<Gate>,
    value: u64,
    stats: RingStats,
    /// diag: (pool, query) for the next `submit_copy` to stamp its copy begin/end
    /// at `query`, `query + 1`; taken (cleared) by that submit.
    pub ts_copy: Option<(vk::QueryPool, u32)>,
}

impl<'a> TransferRing<'a> {
    /// Create the ring: pinned staging, device-local slots (concurrently shared by
    /// the compute + transfer families), two timeline semaphores, command buffers.
    pub fn new(gpu: &'a Gpu, cfg: RingConfig) -> R<Self> {
        cfg.validate()?;
        if !gpu.feats.timeline_semaphore {
            return Err("device lacks the timelineSemaphore feature (required by the ring)".into());
        }
        let n = cfg.slots as usize;
        let fams = [gpu.family_c, gpu.family_x];

        let mut slots = Vec::with_capacity(n);
        for i in 0..n {
            let b = gpu
                .buf_with(
                    cfg.slot_bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::empty(),
                    Some(&fams),
                )
                .map_err(|e| format!("ring slot {i} device allocation: {e}"))?;
            slots.push(b);
        }

        let mut staging = Vec::with_capacity(n);
        let mut staging_ptr = Vec::with_capacity(n);
        for i in 0..n {
            let b = gpu
                .buf(
                    cfg.slot_bytes,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    vk::MemoryPropertyFlags::empty(),
                )
                .map_err(|e| format!("ring staging {i} host allocation: {e}"))?;
            // Map once and keep it mapped: staging is host-written every round.
            let ptr = unsafe {
                vk(
                    gpu.device
                        .map_memory(b.mem, 0, cfg.slot_bytes, vk::MemoryMapFlags::empty()),
                    "vkMapMemory(staging)",
                )?
            } as *mut u8;
            staging.push(b);
            staging_ptr.push(ptr);
        }

        let arrive = make_timeline_semaphore(gpu)?;
        let release = make_timeline_semaphore(gpu)?;

        let (cmd_pool_x, cmd_x) = make_cmd_buffers(gpu, gpu.family_x, n)
            .map_err(|e| format!("ring transfer command buffers: {e}"))?;
        let (cmd_pool_c, cmd_c) = make_cmd_buffers(gpu, gpu.family_c, n)
            .map_err(|e| format!("ring compute command buffers: {e}"))?;

        Ok(TransferRing {
            gpu,
            cfg,
            staging,
            staging_ptr,
            slots,
            cmd_pool_x,
            cmd_pool_c,
            cmd_x,
            cmd_c,
            arrive,
            release,
            gates: vec![Gate::None; n],
            value: 0,
            stats: RingStats::default(),
            ts_copy: None,
        })
    }

    pub fn config(&self) -> RingConfig {
        self.cfg
    }
    pub fn stats(&self) -> RingStats {
        self.stats
    }
    pub fn reset_stats(&mut self) {
        self.stats = RingStats::default();
    }
    /// Highest timeline value handed out so far (both timelines use the same value).
    pub fn last_value(&self) -> u64 {
        self.value
    }
    /// Device-local buffer for `slot` — bind this to whatever consumes the data.
    pub fn slot_buffer(&self, slot: u32) -> vk::Buffer {
        self.slots[slot as usize].buf
    }
    pub fn slot(&self, slot: u32) -> &Buf {
        &self.slots[slot as usize]
    }
    /// Timeline semaphore signalled by the transfer queue (upload arrivals).
    pub fn arrive_semaphore(&self) -> vk::Semaphore {
        self.arrive
    }
    /// Timeline semaphore signalled by the compute queue (slot releases).
    pub fn release_semaphore(&self) -> vk::Semaphore {
        self.release
    }

    /// Copy `src` into the slot's pinned staging chunk (host side, no GPU work).
    pub fn stage(&self, slot: u32, src: &[u8]) -> R<()> {
        let s = self.check_slot(slot)?;
        if src.len() as u64 > self.cfg.slot_bytes {
            return Err(format!(
                "stage overflow: {} bytes into a {} byte slot",
                src.len(),
                self.cfg.slot_bytes
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.staging_ptr[s], src.len());
        }
        Ok(())
    }

    /// Stage `src` then submit its upload in one call (the engine-facing shortcut).
    pub fn submit_upload(&mut self, slot: u32, src: &[u8], timeline_value: u64) -> R<()> {
        self.stage(slot, src)?;
        self.submit_copy(slot, src.len() as u64, timeline_value)
    }

    /// Submit an H2D copy of `bytes` from the slot's (already staged) pinned chunk.
    ///
    /// Blocks on the slot's previous use if it has not completed yet, and records
    /// how long that block took.
    pub fn submit_copy(&mut self, slot: u32, bytes: u64, timeline_value: u64) -> R<()> {
        let s = self.check_slot(slot)?;
        if bytes == 0 || bytes > self.cfg.slot_bytes {
            return Err(format!(
                "copy size {bytes} outside 1..={} for slot {s}",
                self.cfg.slot_bytes
            ));
        }
        if timeline_value <= self.value {
            return Err(format!(
                "timeline values must increase: {timeline_value} <= {}",
                self.value
            ));
        }
        let gate = self.gate(s)?;

        let dev = &self.gpu.device;
        let cmd = self.cmd_x[s];
        unsafe {
            vk(
                dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()),
                "vkResetCommandBuffer",
            )?;
            let bi = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            vk(dev.begin_command_buffer(cmd, &bi), "vkBeginCommandBuffer")?;

            // Previous consumer read (or this slot's earlier copy) -> transfer write.
            // A transfer-only family cannot name shader access; the release semaphore
            // already orders the other queue's reads before this write.
            let x_compute = self.gpu.x_compute;
            let src = if x_compute {
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE
            } else {
                vk::AccessFlags::TRANSFER_WRITE
            };
            let start = vk::MemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[start],
                &[],
                &[],
            );
            let region = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: bytes,
            }];
            let ts = self.ts_copy.take();
            if let Some((pool, q)) = ts {
                dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pool, q);
            }
            dev.cmd_copy_buffer(cmd, self.staging[s].buf, self.slots[s].buf, &region);
            if let Some((pool, q)) = ts {
                dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, q + 1);
            }
            // Make the write available to the compute stage on the other queue; the
            // arrive semaphore provides the execution dependency (and, on a
            // transfer-only family, the availability: its signal covers all prior writes).
            if x_compute {
                let end = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[end],
                    &[],
                    &[],
                );
            }
            vk(dev.end_command_buffer(cmd), "vkEndCommandBuffer")?;
        }

        let _q = self.gpu.queue_guard();
        submit_one(
            dev,
            self.gpu.queue_x,
            cmd,
            gate.map(|(sem, v)| (sem, v, vk::PipelineStageFlags::TRANSFER)),
            Some((self.arrive, timeline_value)),
        )?;

        self.gates[s] = Gate::Arrive(timeline_value);
        self.value = timeline_value;
        self.stats.submits += 1;
        self.stats.bytes += bytes;
        Ok(())
    }

    /// Submit a compute payload that consumes `slot`, gated on the upload that
    /// filled it (`arrive >= timeline_value`) and releasing it on completion
    /// (`release >= timeline_value`). No CPU round-trip: the host may submit the
    /// next round immediately.
    pub fn submit_compute<F>(&mut self, slot: u32, timeline_value: u64, record: F) -> R<()>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> R<()>,
    {
        let s = self.check_slot(slot)?;
        match self.gates[s] {
            Gate::Arrive(v) if v == timeline_value => {}
            other => {
                return Err(format!(
                    "slot {s} has no pending upload for timeline value {timeline_value} \
                     (gate {other:?}); submit_copy first"
                ))
            }
        }
        let dev = &self.gpu.device;
        let cmd = self.cmd_c[s];
        unsafe {
            vk(
                dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()),
                "vkResetCommandBuffer",
            )?;
            let bi = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            vk(dev.begin_command_buffer(cmd, &bi), "vkBeginCommandBuffer")?;

            let start = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[start],
                &[],
                &[],
            );

            let recorded = record(dev, cmd);

            // Publish the consumer's reads/writes so a later overwrite sees them.
            let end = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[end],
                &[],
                &[],
            );
            vk(dev.end_command_buffer(cmd), "vkEndCommandBuffer")?;
            recorded?;
        }

        let _q = self.gpu.queue_guard();
        submit_one(
            dev,
            self.gpu.queue_c,
            cmd,
            Some((
                self.arrive,
                timeline_value,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            )),
            Some((self.release, timeline_value)),
        )?;

        self.gates[s] = Gate::Release(timeline_value);
        self.stats.consumer_submits += 1;
        Ok(())
    }

    /// Host-block until every upload with value `<= timeline_value` has landed.
    /// Returns the seconds spent blocked.
    pub fn wait(&mut self, timeline_value: u64) -> R<f64> {
        let dt = self.wait_sem(self.arrive, timeline_value)?;
        if dt > 0.0 {
            self.stats.host_waits += 1;
            self.stats.host_wait_seconds += dt;
        }
        Ok(dt)
    }

    /// Host-block until every consumer with value `<= timeline_value` has finished.
    pub fn wait_release(&mut self, timeline_value: u64) -> R<f64> {
        let dt = self.wait_sem(self.release, timeline_value)?;
        if dt > 0.0 {
            self.stats.host_waits += 1;
            self.stats.host_wait_seconds += dt;
        }
        Ok(dt)
    }

    /// Non-blocking read of how many values are currently signalled.
    pub fn arrived(&self, sem: vk::Semaphore) -> R<u64> {
        unsafe {
            vk(
                self.gpu.device.get_semaphore_counter_value(sem),
                "vkGetSemaphoreCounterValue",
            )
        }
    }

    // --- internals -----------------------------------------------------------

    fn check_slot(&self, slot: u32) -> R<usize> {
        if slot as usize >= self.slots.len() {
            return Err(format!(
                "slot {slot} out of range (ring has {} slots)",
                self.slots.len()
            ));
        }
        Ok(slot as usize)
    }

    /// Wait for the slot's previous use; returns the semaphore/value to also pass
    /// to the queue (harmless when already signalled, correct when not).
    fn gate(&mut self, slot: usize) -> R<Option<(vk::Semaphore, u64)>> {
        let g = self.gates[slot];
        let (sem, value) = match g {
            Gate::None => return Ok(None),
            Gate::Arrive(v) => (self.arrive, v),
            Gate::Release(v) => (self.release, v),
        };
        self.stats.slot_reuses += 1;
        let dt = self.wait_sem(sem, value)?;
        if dt > 0.0 {
            self.stats.stalls += 1;
            self.stats.stall_seconds += dt;
        }
        Ok(Some((sem, value)))
    }

    /// Blocking timeline wait; returns 0.0 when the value was already reached.
    fn wait_sem(&self, sem: vk::Semaphore, value: u64) -> R<f64> {
        if value == 0 {
            return Ok(0.0);
        }
        let current = self.arrived(sem)?;
        if current >= value {
            return Ok(0.0);
        }
        let sems = [sem];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&values);
        let t0 = Instant::now();
        // **Bounded** (§8ad): the last unbounded wait in the engine. An infinite
        // wait on the display GPU is what let the kernel watchdog issue a BACO reset
        // on 2026-10-26; this one gives up, says what it was waiting for, and aborts,
        // which leaves the device healthy (AGENTS.md incident log).
        let r = unsafe {
            self.gpu
                .device
                .wait_semaphores(&info, crate::ctx::Gpu::WAIT_TIMEOUT_NS)
        };
        if let Err(e) = r {
            eprintln!("breccia: waiting on a transfer-ring semaphore: {e}");
            crate::ctx::Gpu::fault_abort("a transfer-ring semaphore");
        }
        let _ = vk::Result::SUCCESS;
        Ok(t0.elapsed().as_secs_f64())
    }

    pub fn destroy(self) {
        let dev = &self.gpu.device;
        unsafe {
            let _ = {
                let _q = self.gpu.queue_guard();
                dev.device_wait_idle()
            };
            dev.destroy_semaphore(self.release, None);
            dev.destroy_semaphore(self.arrive, None);
            dev.destroy_command_pool(self.cmd_pool_c, None);
            dev.destroy_command_pool(self.cmd_pool_x, None);
            for (b, ptr) in self.staging.iter().zip(self.staging_ptr.iter()) {
                dev.unmap_memory(b.mem);
                let _ = ptr;
            }
            for b in &self.staging {
                self.gpu.destroy_buf(b);
            }
            for b in &self.slots {
                self.gpu.destroy_buf(b);
            }
        }
    }
}

fn make_timeline_semaphore(gpu: &Gpu) -> R<vk::Semaphore> {
    let mut tci = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let sci = vk::SemaphoreCreateInfo::default().push_next(&mut tci);
    unsafe {
        vk(
            gpu.device.create_semaphore(&sci, None),
            "vkCreateSemaphore(timeline)",
        )
    }
}

fn make_cmd_buffers(
    gpu: &Gpu,
    family: u32,
    count: usize,
) -> R<(vk::CommandPool, Vec<vk::CommandBuffer>)> {
    unsafe {
        let pci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = vk(
            gpu.device.create_command_pool(&pci, None),
            "vkCreateCommandPool",
        )?;
        let ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count as u32);
        match gpu.device.allocate_command_buffers(&ai) {
            Ok(v) => Ok((pool, v)),
            Err(e) => {
                gpu.device.destroy_command_pool(pool, None);
                Err(format!("vkAllocateCommandBuffers: {e}"))
            }
        }
    }
}

/// Single-command-buffer submit with optional timeline wait/signal.
///
/// Timeline waits ignore `pWaitDstStageMask` (Vulkan 1.2), but a valid mask is
/// still supplied for clarity and for drivers that guard on it.
fn submit_one(
    dev: &ash::Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    wait: Option<(vk::Semaphore, u64, vk::PipelineStageFlags)>,
    signal: Option<(vk::Semaphore, u64)>,
) -> R<()> {
    let cmds = [cmd];
    let wait_sems: [vk::Semaphore; 1];
    let wait_vals: [u64; 1];
    let wait_stages: [vk::PipelineStageFlags; 1];
    let sig_sems: [vk::Semaphore; 1];
    let sig_vals: [u64; 1];

    let mut tsi = vk::TimelineSemaphoreSubmitInfo::default();
    let mut si = vk::SubmitInfo::default().command_buffers(&cmds);

    if let Some((sem, value, stage)) = wait {
        wait_sems = [sem];
        wait_vals = [value];
        wait_stages = [stage];
        si = si
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages);
        tsi = tsi.wait_semaphore_values(&wait_vals);
    }
    if let Some((sem, value)) = signal {
        sig_sems = [sem];
        sig_vals = [value];
        si = si.signal_semaphores(&sig_sems);
        tsi = tsi.signal_semaphore_values(&sig_vals);
    }
    si = si.push_next(&mut tsi);

    unsafe {
        vk(
            dev.queue_submit(queue, &[si], vk::Fence::null()),
            "vkQueueSubmit",
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Host-memory capability probe (VK_EXT_external_memory_host)
// ---------------------------------------------------------------------------

/// What this device can do with zero-copy host memory, and whether it was proven.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HostMemoryCaps {
    pub extension_listed: bool,
    pub extension_enabled: bool,
    /// `VkPhysicalDeviceExternalMemoryHostPropertiesEXT::minImportedHostPointerAlignment`.
    pub min_imported_host_pointer_alignment: u64,
    /// Memory types the driver accepted the host pointer for.
    pub imported_memory_type_bits: u32,
    pub import_tested: bool,
    pub import_ok: bool,
    /// Read-back of the first words matched the host pattern (import really works).
    pub import_verified: bool,
    /// H2D rate from imported host memory to a device-local buffer.
    pub h2d_gbps: Option<f64>,
    pub error: Option<String>,
    pub notes: Vec<String>,
}

/// Probe `VK_EXT_external_memory_host`: presence, alignment, and (when enabled and
/// supported) a real import + verified DMA from mapped host memory.
///
/// Never fails the caller: problems come back in `error`/`notes`.
pub fn probe_host_memory(dev: &crate::ctx::Dev, gpu: &Gpu, bytes: u64) -> HostMemoryCaps {
    let mut caps = HostMemoryCaps {
        extension_listed: dev.has_ext("VK_EXT_external_memory_host"),
        extension_enabled: gpu.external_memory_host,
        ..Default::default()
    };
    if !caps.extension_listed {
        caps.notes.push(
            "VK_EXT_external_memory_host is not advertised; staging copies stay required"
                .to_string(),
        );
        return caps;
    }
    // Alignment requirement (extension-listed only; otherwise the struct is unusable).
    let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host_props);
    unsafe {
        gpu.instance
            .get_physical_device_properties2(dev.pdev, &mut p2);
    }
    caps.min_imported_host_pointer_alignment = host_props.min_imported_host_pointer_alignment;

    if !caps.extension_enabled {
        caps.notes.push(
            "VK_EXT_external_memory_host is advertised but was not enabled on the logical \
             device; zero-copy import not attempted"
                .to_string(),
        );
        return caps;
    }

    match probe_import(
        gpu,
        bytes,
        caps.min_imported_host_pointer_alignment.max(4096),
    ) {
        Ok((bits, verified, gbps)) => {
            caps.import_tested = true;
            caps.import_ok = true;
            caps.imported_memory_type_bits = bits;
            caps.import_verified = verified;
            caps.h2d_gbps = Some(gbps);
            caps.notes.push(
                "host pointer imported with vkImportMemoryHostPointerInfoEXT and used as a \
                 vkCmdCopyBuffer source: zero-copy upload path is available"
                    .to_string(),
            );
        }
        Err(e) => {
            caps.import_tested = true;
            caps.import_ok = false;
            caps.error = Some(e);
        }
    }
    caps
}

/// Aligned host allocation usable as an import source (freed on drop).
struct AlignedHost {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedHost {
    fn new(size: u64, align: u64) -> R<Self> {
        let layout = std::alloc::Layout::from_size_align(size as usize, align as usize)
            .map_err(|e| format!("host allocation layout: {e}"))?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(format!(
                "host allocation of {size} bytes @ {align}-byte alignment failed"
            ));
        }
        // Zero it: every later `&mut [u8]` over this range must reference
        // initialised bytes.
        unsafe { std::ptr::write_bytes(ptr, 0, layout.size()) };
        Ok(Self { ptr, layout })
    }
    fn fill_pattern(&mut self) {
        let buf = unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) };
        for (i, chunk) in buf.chunks_mut(4).enumerate() {
            chunk.copy_from_slice(&(1.0f32 + (i % 8) as f32).to_le_bytes());
        }
    }
}

impl Drop for AlignedHost {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// Picks the memory type common to both the host-pointer's importable types and
/// the imported buffer's requirements, preferring HOST_VISIBLE. `None` when the
/// two bit sets are disjoint.
fn pick_import_type(
    host_bits: u32,
    buf_bits: u32,
    mem: &vk::PhysicalDeviceMemoryProperties,
) -> Option<u32> {
    find_mem_type(
        mem,
        host_bits & buf_bits,
        vk::MemoryPropertyFlags::empty(),
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    )
}

/// Import a host pointer as device memory backing a new buffer, doing exactly the
/// import half of the host-pointer probe: allocate the buffer, find a common
/// memory type, import the host range, and bind it.
pub fn import_from_host_ptr(
    gpu: &Gpu,
    ptr: *mut u8,
    bytes: u64,
    usage: vk::BufferUsageFlags,
) -> R<Buf> {
    let fns = ash::ext::external_memory_host::Device::new(&gpu.instance, &gpu.device);

    let mut props = vk::MemoryHostPointerPropertiesEXT::default();
    let r = unsafe {
        (fns.fp().get_memory_host_pointer_properties_ext)(
            gpu.device.handle(),
            vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            ptr as *const c_void,
            &mut props,
        )
    };
    r.result()
        .map_err(|e| format!("host pointer of {} bytes @ {:p}: {}", bytes, ptr, e))?;
    if props.memory_type_bits == 0 {
        return Err("driver reported no importable memory type for the host pointer".to_string());
    }

    // Buffer whose lifetime is the imported host range. The import handle type must
    // also be declared on the buffer (VUID-vkBindBufferMemory-memory-02985).
    unsafe {
        let mut ext_mem = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let bci = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_mem);
        let buf = vk(
            gpu.device.create_buffer(&bci, None),
            "vkCreateBuffer(import)",
        )?;
        let reqs = gpu.device.get_buffer_memory_requirements(buf);
        let idx = match pick_import_type(props.memory_type_bits, reqs.memory_type_bits, &gpu.mem) {
            Some(i) => i,
            None => {
                gpu.device.destroy_buffer(buf, None);
                return Err(format!(
                    "no common memory type between host pointer ({:#x}) and buffer ({:#x})",
                    props.memory_type_bits, reqs.memory_type_bits
                ));
            }
        };
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(ptr as *mut c_void);
        let mai = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(idx)
            .push_next(&mut import);
        let mem = match gpu.device.allocate_memory(&mai, None) {
            Ok(m) => m,
            Err(e) => {
                gpu.device.destroy_buffer(buf, None);
                return Err(format!("vkAllocateMemory(host import) failed: {e}"));
            }
        };
        if let Err(e) = gpu.device.bind_buffer_memory(buf, mem, 0) {
            gpu.device.free_memory(mem, None);
            gpu.device.destroy_buffer(buf, None);
            return Err(format!("vkBindBufferMemory(host import) failed: {e}"));
        }
        Ok(Buf {
            buf,
            mem,
            size: bytes,
            id: Buf::next_id(),
        })
    }
}

/// Returns `(memory_type_bits, verified, h2d_gbps)`.
fn probe_import(gpu: &Gpu, bytes: u64, align: u64) -> R<(u32, bool, f64)> {
    let mut host = AlignedHost::new(bytes, align.max(4096))?;
    host.fill_pattern();

    let mut props = vk::MemoryHostPointerPropertiesEXT::default();
    let fns = ash::ext::external_memory_host::Device::new(&gpu.instance, &gpu.device);
    unsafe {
        (fns.fp().get_memory_host_pointer_properties_ext)(
            gpu.device.handle(),
            vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            host.ptr as *const c_void,
            &mut props,
        )
    }
    .result()
    .map_err(|e| format!("host pointer of {bytes} bytes @ align {align}: {e}"))?;

    let imported = import_from_host_ptr(gpu, host.ptr, bytes, vk::BufferUsageFlags::TRANSFER_SRC)?;

    let dst = match gpu.buf(
        bytes,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    ) {
        Ok(b) => b,
        Err(e) => {
            gpu.destroy_buf(&imported);
            return Err(format!("device-local import target: {e}"));
        }
    };

    let result = (|| -> R<(bool, f64)> {
        let timing = gpu.run_timed(crate::ctx::Qc::X, 0.15, |d, cmd, n| unsafe {
            for _ in 0..n {
                let region = [vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes,
                }];
                d.cmd_copy_buffer(cmd, imported.buf, dst.buf, &region);
            }
        })?;
        let gbps = bytes as f64 / timing.per_iter_s / 1e9;

        // Verify the copy actually transported the host pattern.
        let probe_len = 4096u64.min(bytes);
        let readback = gpu.buf(
            probe_len,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::empty(),
        )?;
        gpu.wall_run(crate::ctx::Qc::X, |d, cmd| unsafe {
            let region = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: probe_len,
            }];
            d.cmd_copy_buffer(cmd, dst.buf, readback.buf, &region);
        })?;
        let mut got = vec![0u8; probe_len as usize];
        gpu.read_host(&readback, &mut got)?;
        let want = unsafe { std::slice::from_raw_parts(host.ptr, probe_len as usize).to_vec() };
        gpu.destroy_buf(&readback);
        Ok((got == want, gbps))
    })();

    gpu.destroy_buf(&dst);
    gpu.destroy_buf(&imported);
    let (verified, gbps) = result?;
    Ok((props.memory_type_bits, verified, gbps))
}

#[cfg(test)]
mod import_type_tests {
    use super::*;

    /// Memory types: 0 = DEVICE_LOCAL only, 1 = HOST_VISIBLE, 2 = HOST_VISIBLE|COHERENT.
    fn mem_props() -> vk::PhysicalDeviceMemoryProperties {
        let mut mem = vk::PhysicalDeviceMemoryProperties::default();
        mem.memory_type_count = 3;
        mem.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        mem.memory_types[1].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        mem.memory_types[2].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        mem
    }

    #[test]
    fn common_bits_pick_the_host_visible_type() {
        // host pointer importable as type 1, buffer accepts 0|1 -> only 1 is common.
        assert_eq!(pick_import_type(0b010, 0b011, &mem_props()), Some(1));
        // common set is {2}: the HOST_VISIBLE preference skips type 0.
        assert_eq!(pick_import_type(0b110, 0b100, &mem_props()), Some(2));
    }

    #[test]
    fn disjoint_bits_give_none() {
        assert_eq!(pick_import_type(0b001, 0b110, &mem_props()), None);
        assert_eq!(pick_import_type(0, 0b111, &mem_props()), None);
    }
}
