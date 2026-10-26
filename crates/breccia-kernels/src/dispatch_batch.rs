//! Record-only dispatch batching: one command buffer and **one fence per phase**
//! instead of one per kernel.
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
//! [`begin`] starts recording into the compute queue's command buffer and sets a
//! thread-local flag; while it is set, [`crate::ops::Pipe::run`],
//! [`crate::Gemv::run`] and [`crate::GemvBatch::run`] **record** their dispatch
//! instead of submitting it. [`end`] closes the command buffer, submits once and
//! waits once.
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
//! * No host readback, upload or allocation may happen inside a batch (they
//!   submit their own command buffers). Close the batch first.
//! * Every recorded dispatch is followed by a full barrier, so a batch may mix
//!   dependant kernels freely.
//! * One batch per thread; [`begin`] refuses to nest.

use anyhow::{anyhow, Result};
use ash::vk;
use breccia_vk::ctx::{Buf, Gpu, Kernel, Qc};
use std::cell::Cell;

thread_local! {
    /// Is a batch open on this thread?
    static OPEN: Cell<bool> = const { Cell::new(false) };
    /// Dispatches recorded into the open batch.
    static COUNT: Cell<u32> = const { Cell::new(0) };
    /// This thread's descriptor pool for the open batch.
    static POOL: Cell<vk::DescriptorPool> = const { Cell::new(vk::DescriptorPool::null()) };
}

/// Sets a batch budgets for (the biggest phase here is ~35 dispatches).
const MAX_SETS: u32 = 64;
/// Storage bindings per set (the widest kernel in this crate declares 5).
const MAX_BINDINGS: u32 = 8;

/// Is a batch open on this thread?
pub fn is_open() -> bool {
    OPEN.with(|c| c.get())
}

/// Dispatches recorded into the batch that is (or was last) open.
pub fn recorded() -> u32 {
    COUNT.with(|c| c.get())
}

/// Start recording on the compute queue's command buffer. Nothing is submitted.
pub fn begin(gpu: &Gpu) -> Result<()> {
    if is_open() {
        return Err(anyhow!("a dispatch batch is already open on this thread"));
    }
    unsafe {
        gpu.device
            .reset_command_buffer(gpu.cmd_c, vk::CommandBufferResetFlags::empty())
            .map_err(|e| anyhow!("resetting the batch command buffer: {e}"))?;
        gpu.device
            .begin_command_buffer(gpu.cmd_c, &vk::CommandBufferBeginInfo::default())
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
        POOL.with(|c| c.set(pool));
    }
    OPEN.with(|c| c.set(true));
    COUNT.with(|c| c.set(0));
    Ok(())
}

/// Close the batch, submit it once and wait once. Returns the host seconds the
/// round trip took (the GPU work is inside it).
pub fn end(gpu: &Gpu) -> Result<f64> {
    if !is_open() {
        return Err(anyhow!("no dispatch batch is open on this thread"));
    }
    OPEN.with(|c| c.set(false));
    unsafe {
        gpu.device
            .end_command_buffer(gpu.cmd_c)
            .map_err(|e| anyhow!("ending the batch command buffer: {e}"))?;
    }
    let t0 = std::time::Instant::now();
    let fence = gpu
        .submit(Qc::C)
        .map_err(|e| anyhow!("submitting the batch: {e}"))?;
    gpu.wait_fence(fence)
        .map_err(|e| anyhow!("waiting for the batch: {e}"))?;
    // Only now: the recorded command buffer references these sets, so freeing
    // the pool before the submission completes is a use-after-free (it shows up
    // as `VUID-vkCmdDispatch-None-08114`, "descriptors must be valid").
    unsafe {
        let pool = POOL.with(|c| c.replace(vk::DescriptorPool::null()));
        if pool != vk::DescriptorPool::null() {
            gpu.device.destroy_descriptor_pool(pool, None);
        }
    }
    Ok(t0.elapsed().as_secs_f64())
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
    let cmd = gpu.cmd_c;
    let pool = POOL.with(|c| c.get());
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
    COUNT.with(|c| c.set(c.get() + 1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_batch_is_open_on_a_fresh_thread() {
        assert!(!is_open());
        assert_eq!(recorded(), 0);
        assert_eq!(POOL.with(|c| c.get()), vk::DescriptorPool::null());
    }
}
