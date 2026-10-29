//! Phase 2 (M1b) — transfer benchmark: ring throughput, capability probe, and the
//! M3-shaped "expert slot churn" (DMA hidden under compute?).
//!
//! Everything here is driven through [`crate::transfer::TransferRing`] — the same
//! API M3/M5 will use — so the numbers describe the engine's transfer layer, not a
//! side-channel harness. Results are recorded in `docs/phase-2/01-transfer-harness.md`.

use ash::vk;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::bench;
use crate::ctx::{api_version_string, Dev, Gpu, Host, Kernel, Qc, R};
use crate::report::DeviceId;
use crate::transfer::{probe_host_memory, HostMemoryCaps, RingConfig, RingStats, TransferRing};

pub(crate) const SPV_SLOT_CONSUMER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/slot_consumer.spv"));

/// Bump when the JSON shape changes incompatibly.
pub const TRANSFER_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Options / selection
// ---------------------------------------------------------------------------

/// How a device was requested on the command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSelector {
    /// Vulkan enumeration index at run time.
    Index(u32),
    /// PCI address — the stable key (Vulkan ordinals are not stable on this host).
    Pci(String),
}

#[derive(Debug, Clone, Default)]
pub struct TransferBenchOptions {
    /// Short windows and smaller buffers (≈3–4× faster, noisier).
    pub quick: bool,
    /// Empty = every discrete GPU.
    pub devices: Vec<DeviceSelector>,
}

pub(crate) fn select_devices(devices: &[Dev], opts: &TransferBenchOptions) -> Result<Vec<u32>, String> {
    if opts.devices.is_empty() {
        let discrete: Vec<u32> = devices
            .iter()
            .filter(|d| d.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
            .map(|d| d.index)
            .collect();
        if !discrete.is_empty() {
            return Ok(discrete);
        }
        return Ok(devices
            .iter()
            .filter(|d| d.device_type != vk::PhysicalDeviceType::INTEGRATED_GPU)
            .map(|d| d.index)
            .collect());
    }

    let mut out = Vec::new();
    for sel in &opts.devices {
        let hit = match sel {
            DeviceSelector::Index(i) => devices.iter().find(|d| d.index == *i).map(|d| d.index),
            DeviceSelector::Pci(addr) => {
                let want = crate::pci::normalize_address(addr);
                devices
                    .iter()
                    .find(|d| crate::pci::normalize_address(&d.pci_address) == want)
                    .map(|d| d.index)
            }
        };
        match hit {
            Some(i) => {
                if !out.contains(&i) {
                    out.push(i);
                }
            }
            None => {
                let avail = devices
                    .iter()
                    .map(|d| {
                        format!(
                            "{}={} [pci {}]",
                            d.index,
                            d.name,
                            if d.pci_address.is_empty() {
                                "?"
                            } else {
                                &d.pci_address
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(format!(
                    "device selector {sel:?} matched nothing. available: {avail}"
                ));
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Report schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferReport {
    pub schema_version: u32,
    pub environment: TransferEnvironment,
    pub devices: Vec<DeviceTransfer>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEnvironment {
    pub hostname: String,
    pub timestamp_unix_s: u64,
    pub timestamp: String,
    pub instance_api_version: String,
    pub quick: bool,
    /// Instance layers requested through the environment (validation runs).
    pub instance_layers: Vec<String>,
    /// Processes holding `/dev/dri/*` at benchmark time (self excluded).
    pub competing_gpu_processes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTransfer {
    pub device: DeviceId,
    pub queue: TransferQueue,
    pub capabilities: TransferCaps,
    pub ring: Option<RingThroughput>,
    pub churn: Option<Churn>,
    pub notes: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferQueue {
    pub compute_family: u32,
    /// One of `distinct_transfer_queue_family`, `second_queue_of_compute_family`,
    /// `same_queue_as_compute` (fallback path, reported honestly).
    pub mode: String,
    pub transfer_family: u32,
    pub transfer_queue_index: u32,
    pub distinct_from_compute: bool,
    pub transfer_family_queue_count: u32,
    pub compute_family_queue_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferCaps {
    pub timeline_semaphore_supported: bool,
    /// True when the ring actually gated submissions on timeline values.
    pub timeline_semaphore_used: bool,
    pub host_memory: HostMemoryCaps,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingThroughput {
    pub slots: u32,
    pub sizes: Vec<RingSizeStat>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingSizeStat {
    pub size_mb: u64,
    pub slots: u32,
    pub rounds: u32,
    pub bytes: u64,
    /// Pre-staged pinned staging → device-local, DMA only (engine steady state with
    /// an io_uring-fed pinned ring).
    pub dma_seconds: f64,
    pub dma_gbps: f64,
    /// Same loop including the host memcpy into pinned staging each round.
    pub staged_seconds: f64,
    pub staged_gbps: f64,
    pub submits: u64,
    pub slot_reuses: u64,
    pub stall_count: u64,
    pub stall_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Churn {
    pub slot_bytes: u64,
    pub slots: u32,
    pub pipeline_depth: u32,
    pub rounds: u32,
    pub bytes_total: u64,
    pub blobs_4mib: u32,
    pub blobs_16mib: u32,
    /// Consumer alone over the same blob sequence with all slots resident.
    pub compute_only_seconds: f64,
    /// Producer alone over the same blob sequence, no consumer at all.
    pub transfer_only_seconds: f64,
    /// Producer + consumer pipelined through the ring (the M3 shape).
    pub stream_seconds: f64,
    pub overlap_efficiency_percent: f64,
    pub overlap_formula: String,
    /// `stream_seconds - compute_only_seconds`, clamped at 0: time the consumer
    /// spent waiting for DMA that hardware could not hide.
    pub consumer_stall_seconds: f64,
    pub consumer_stall_ms_per_mb: f64,
    /// Effective streaming rate of the pipelined run (bytes / stream time).
    pub stream_h2d_gbps: f64,
    /// Full-rate streaming with no consumer (bytes / transfer-only time).
    pub full_rate_h2d_gbps: f64,
    pub consumer_mib_per_s: f64,
    pub calibrated_iters_per_mib: u32,
    pub consumer_threads: u32,
    pub compute_seconds_per_blob: f64,
    pub transfer_seconds_per_blob: f64,
    pub compute_queue_family: u32,
    pub transfer_queue_family: u32,
    pub producer: RingStats,
    pub notes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run_transfer_bench(opts: &TransferBenchOptions) -> anyhow::Result<TransferReport> {
    let host = Host::new().map_err(|e| anyhow::anyhow!("breccia-vk: {e}"))?;
    let mut warnings = Vec::new();
    if host.device_select_was_set {
        warnings.push(format!(
            "MESA_VK_DEVICE_SELECT={} was set in the environment and has been cleared: it \
             would otherwise pin enumeration to a single device and hide both discrete GPUs \
             (docs/PROJECT-SETUP.md)",
            host.device_select_value.as_deref().unwrap_or("")
        ));
    }

    let devices = host
        .devices()
        .map_err(|e| anyhow::anyhow!("breccia-vk: {e}"))?;
    if devices.is_empty() {
        host.destroy();
        return Err(anyhow::anyhow!(
            "breccia-vk: no Vulkan physical devices found (is the RADV driver visible?)"
        ));
    }
    let selected = match select_devices(&devices, opts) {
        Ok(s) => s,
        Err(e) => {
            host.destroy();
            return Err(anyhow::anyhow!("breccia-vk: {e}"));
        }
    };
    if selected.is_empty() {
        host.destroy();
        return Err(anyhow::anyhow!(
            "breccia-vk: no devices selected for the transfer bench"
        ));
    }

    let environment = TransferEnvironment {
        hostname: hostname(),
        timestamp_unix_s: now_unix(),
        timestamp: iso8601_utc(now_unix()),
        instance_api_version: api_version_string(host.api_version),
        quick: opts.quick,
        instance_layers: requested_layers(),
        competing_gpu_processes: gpu_users(),
    };

    println!(
        "breccia-vk: transfer bench ({} mode) on device(s) {:?} of {} enumerated{}",
        if opts.quick { "quick" } else { "full" },
        selected,
        devices.len(),
        if environment.instance_layers.is_empty() {
            String::new()
        } else {
            format!(", instance layers {:?}", environment.instance_layers)
        }
    );
    if !environment.competing_gpu_processes.is_empty() {
        println!(
            "  note: other processes hold /dev/dri: {:?}",
            environment.competing_gpu_processes
        );
    }

    let mut results = Vec::new();
    for index in selected {
        let dev = devices
            .iter()
            .find(|d| d.index == index)
            .expect("selected device exists");
        let gpu = match Gpu::create(&host, dev, dev.index) {
            Ok(g) => g,
            Err(e) => {
                results.push(DeviceTransfer {
                    device: bench::device_id(dev),
                    queue: queue_info(dev, None),
                    capabilities: empty_caps(dev),
                    ring: None,
                    churn: None,
                    notes: Vec::new(),
                    error: Some(e),
                });
                continue;
            }
        };
        let rep = bench_device(dev, &gpu, opts.quick);
        results.push(rep);
        gpu.destroy();
    }
    host.destroy();

    let any_ok = results
        .iter()
        .any(|r| r.ring.is_some() || r.churn.is_some());
    if !any_ok {
        let errs: Vec<String> = results.iter().filter_map(|r| r.error.clone()).collect();
        return Err(anyhow::anyhow!(
            "breccia-vk: transfer bench produced no results: {errs:?}"
        ));
    }

    Ok(TransferReport {
        schema_version: TRANSFER_SCHEMA_VERSION,
        environment,
        devices: results,
        warnings,
    })
}

fn queue_info(dev: &Dev, gpu: Option<&Gpu>) -> TransferQueue {
    let (fam_c, fam_x, idx, mode) = match gpu {
        Some(g) => (
            g.family_c,
            g.family_x,
            g.queue_x_index,
            g.xfer_mode.as_str().to_string(),
        ),
        None => (0, 0, 0, "unknown".to_string()),
    };
    let count = |f: u32| {
        dev.families
            .get(f as usize)
            .map(|q| q.queue_count)
            .unwrap_or(0)
    };
    TransferQueue {
        compute_family: fam_c,
        mode,
        transfer_family: fam_x,
        transfer_queue_index: idx,
        distinct_from_compute: fam_c != fam_x || idx != 0,
        transfer_family_queue_count: count(fam_x),
        compute_family_queue_count: count(fam_c),
    }
}

fn empty_caps(dev: &Dev) -> TransferCaps {
    TransferCaps {
        timeline_semaphore_supported: dev.feats.timeline_semaphore,
        timeline_semaphore_used: false,
        host_memory: HostMemoryCaps {
            extension_listed: dev.has_ext("VK_EXT_external_memory_host"),
            ..Default::default()
        },
    }
}

// ---------------------------------------------------------------------------
// Per-device orchestration
// ---------------------------------------------------------------------------

fn bench_device(dev: &Dev, gpu: &Gpu, quick: bool) -> DeviceTransfer {
    let mut dt = DeviceTransfer {
        device: bench::device_id(dev),
        queue: queue_info(dev, Some(gpu)),
        capabilities: empty_caps(dev),
        ring: None,
        churn: None,
        notes: Vec::new(),
        error: None,
    };

    if !gpu.feats.timeline_semaphore {
        dt.error = Some("device lacks timelineSemaphore; the ring cannot run".to_string());
        return dt;
    }
    if dt.queue.mode != "distinct_transfer_queue_family" {
        dt.notes.push(format!(
            "no distinct transfer queue family: transfers fall back to '{}' — the same code \
             path is exercised, but overlap is expected to be ~0",
            dt.queue.mode
        ));
    }

    // --- capability probe (host memory import) ------------------------------
    let host_bytes = if quick { 4 << 20 } else { 16 << 20 };
    dt.capabilities.host_memory = probe_host_memory(dev, gpu, host_bytes);
    let hm = &dt.capabilities.host_memory;
    println!(
        "  [{}] external_memory_host: listed={} enabled={} align={} import={} \
         verified={} h2d={}",
        gpu.index,
        hm.extension_listed,
        hm.extension_enabled,
        hm.min_imported_host_pointer_alignment,
        if hm.import_ok { "ok" } else { "n/a" },
        hm.import_verified,
        hm.h2d_gbps
            .map(|v| format!("{v:.1} GB/s"))
            .unwrap_or_else(|| "-".to_string())
    );
    if let Some(e) = &hm.error {
        println!("  [{}] host import failed: {e}", gpu.index);
    }

    // --- ring throughput ----------------------------------------------------
    match bench_ring_throughput(gpu, quick) {
        Ok(rt) => {
            for s in &rt.sizes {
                println!(
                    "  [{}] ring {:>3} MiB x{} slots: {:.1} GB/s DMA ({:.2} s, {} rounds), \
                     {:.1} GB/s with host staging",
                    gpu.index,
                    s.size_mb,
                    s.slots,
                    s.dma_gbps,
                    s.dma_seconds,
                    s.rounds,
                    s.staged_gbps
                );
            }
            dt.capabilities.timeline_semaphore_used = true;
            dt.ring = Some(rt);
        }
        Err(e) => {
            dt.notes.push(format!("ring throughput skipped: {e}"));
            println!("  [{}] ring throughput failed: {e}", gpu.index);
        }
    }

    // --- churn (M3 shape) ---------------------------------------------------
    let hint = dt
        .ring
        .as_ref()
        .and_then(|r| r.sizes.iter().find(|s| s.size_mb == 16).or(r.sizes.first()))
        .map(|s| s.dma_gbps)
        .filter(|v| *v > 0.05)
        .unwrap_or(8.0);
    match bench_churn(gpu, quick, hint) {
        Ok(c) => {
            println!(
                "  [{}] churn: {} rounds / {} MiB, depth {}: stream {:.3} s vs compute-only \
                 {:.3} s, transfer-only {:.3} s -> overlap {:.1}%, consumer stall {:.2} ms/MB",
                gpu.index,
                c.rounds,
                c.bytes_total >> 20,
                c.pipeline_depth,
                c.stream_seconds,
                c.compute_only_seconds,
                c.transfer_only_seconds,
                c.overlap_efficiency_percent,
                c.consumer_stall_ms_per_mb
            );
            dt.capabilities.timeline_semaphore_used = true;
            dt.churn = Some(c);
        }
        Err(e) => {
            dt.notes.push(format!("churn skipped: {e}"));
            println!("  [{}] churn failed: {e}", gpu.index);
        }
    }

    dt
}

// ---------------------------------------------------------------------------
// Ring throughput
// ---------------------------------------------------------------------------

fn bench_ring_throughput(gpu: &Gpu, quick: bool) -> R<RingThroughput> {
    let sizes_mb: &[u64] = if quick { &[16, 64] } else { &[16, 64, 256] };
    let target = if quick { 0.10 } else { 0.35 };
    let mut sizes = Vec::new();
    let mut ring_slots = 0u32;

    for &mb in sizes_mb {
        let slot_bytes = mb << 20;
        // 256 MiB slots: keep VRAM + pinned host use bounded.
        let slots: u32 = if slot_bytes >= (256 << 20) { 3 } else { 4 };
        ring_slots = ring_slots.max(slots);
        let mut ring = TransferRing::new(gpu, RingConfig::new(slots, slot_bytes))?;

        let mut pattern = vec![0u8; slot_bytes as usize];
        for chunk in pattern.chunks_mut(1 << 20) {
            chunk.fill(0xA5);
        }
        for s in 0..slots {
            ring.stage(s, &pattern)?;
        }

        // Probe one cycle to size the measured window.
        let v = ring.last_value();
        let t0 = Instant::now();
        for r in 0..slots {
            ring.submit_copy(r, slot_bytes, v + 1 + r as u64)?;
        }
        ring.wait(v + slots as u64)?;
        let per_round = t0.elapsed().as_secs_f64() / slots as f64;
        let rounds =
            ((target / per_round.max(1e-9)).ceil() as u64).clamp(slots as u64, 4096) as u32;

        // (a) DMA only, staging pre-filled.
        ring.reset_stats();
        let v = ring.last_value();
        let t0 = Instant::now();
        for r in 0..rounds {
            ring.submit_copy(r % slots, slot_bytes, v + 1 + r as u64)?;
        }
        ring.wait(v + rounds as u64)?;
        let dma_seconds = t0.elapsed().as_secs_f64();
        let dma_stats = ring.stats();

        // (b) with the host memcpy into pinned staging on every round.
        ring.reset_stats();
        let v = ring.last_value();
        let t0 = Instant::now();
        for r in 0..rounds {
            ring.submit_upload(r % slots, &pattern, v + 1 + r as u64)?;
        }
        ring.wait(v + rounds as u64)?;
        let staged_seconds = t0.elapsed().as_secs_f64();

        let bytes = rounds as u64 * slot_bytes;
        sizes.push(RingSizeStat {
            size_mb: mb,
            slots,
            rounds,
            bytes,
            dma_seconds,
            dma_gbps: bytes as f64 / dma_seconds / 1e9,
            staged_seconds,
            staged_gbps: bytes as f64 / staged_seconds / 1e9,
            submits: dma_stats.submits,
            slot_reuses: dma_stats.slot_reuses,
            stall_count: dma_stats.stalls,
            stall_seconds: dma_stats.stall_seconds,
        });
        ring.destroy();
        drop(pattern);
    }

    Ok(RingThroughput {
        slots: ring_slots,
        sizes,
        notes: vec![
            "timeline-semaphore ring: N pinned staging chunks -> N device-local slots on the \
             transfer queue family; slot reuse is gated by arrive/release timeline values, no \
             vkQueueWaitIdle in steady state"
                .to_string(),
            "dma_gbps = pre-staged pinned host chunk -> device-local slot (steady state with an \
             io_uring-fed pinned ring); staged_gbps includes the host memcpy of every round"
                .to_string(),
        ],
    })
}

// ---------------------------------------------------------------------------
// Churn simulation (the M3 shape)
// ---------------------------------------------------------------------------

pub(crate) struct ConsumerCal {
    pub(crate) iters_per_mib: u32,
    read_seconds_16mib: f64,
    pub(crate) slope_seconds_per_iter: f64,
    pub(crate) target_ms_per_mib: f64,
}

fn blob_sequence(total_bytes: u64) -> Vec<u64> {
    const MIB4: u64 = 4 << 20;
    const MIB16: u64 = 16 << 20;
    let mut out = Vec::new();
    let mut total = 0u64;
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    while total < total_bytes {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let b = if (s >> 33) & 1 == 0 { MIB4 } else { MIB16 };
        out.push(b);
        total += b;
    }
    out
}

pub(crate) fn push4(a: u32, b: u32, c: u32, d: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a.to_le_bytes());
    out[4..8].copy_from_slice(&b.to_le_bytes());
    out[8..12].copy_from_slice(&c.to_le_bytes());
    out[12..16].copy_from_slice(&d.to_le_bytes());
    out
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_dispatch(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    k: &Kernel,
    set: vk::DescriptorSet,
    groups: u32,
    count: u32,
    threads: u32,
    iters: u32,
) {
    unsafe {
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[set],
            &[],
        );
        // count, stride, iters, acc_elems
        d.cmd_push_constants(
            cmd,
            k.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push4(count, threads, iters, threads),
        );
        d.cmd_dispatch(cmd, groups, 1, 1);
        Gpu::barrier_all(d, cmd);
    }
}

/// One descriptor set per ring slot (binding 0 is the slot's own buffer).
struct SlotSets {
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl SlotSets {
    fn new(gpu: &Gpu, k: &Kernel, ring: &TransferRing<'_>, acc: &crate::ctx::Buf) -> R<Self> {
        let n = ring.config().slots;
        unsafe {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2 * n)];
            let dpci = vk::DescriptorPoolCreateInfo::default()
                .max_sets(n)
                .pool_sizes(&sizes);
            let pool = crate::ctx::vk(
                gpu.device.create_descriptor_pool(&dpci, None),
                "vkCreateDescriptorPool(slot consumer)",
            )?;
            let layouts = vec![k.dsl; n as usize];
            let dai = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            let sets = match gpu.device.allocate_descriptor_sets(&dai) {
                Ok(s) => s,
                Err(e) => {
                    gpu.device.destroy_descriptor_pool(pool, None);
                    return Err(format!("vkAllocateDescriptorSets(slot consumer): {e}"));
                }
            };
            for (i, set) in sets.iter().enumerate() {
                let slot_info = vk::DescriptorBufferInfo {
                    buffer: ring.slot_buffer(i as u32),
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                };
                let acc_info = vk::DescriptorBufferInfo {
                    buffer: acc.buf,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                };
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&slot_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(1)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&acc_info)),
                ];
                gpu.device.update_descriptor_sets(&writes, &[]);
            }
            Ok(Self { pool, sets })
        }
    }

    fn at(&self, slot: u32) -> vk::DescriptorSet {
        self.sets[slot as usize]
    }

    fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }
}

fn bench_churn(gpu: &Gpu, quick: bool, h2d_hint_gbps: f64) -> R<Churn> {
    let slot_bytes: u64 = 16 << 20;
    let slots: u32 = if quick { 4 } else { 6 };
    let total_bytes: u64 = if quick { 1024 << 20 } else { 4096 << 20 };
    let threads: u32 = if quick { 1 << 17 } else { 1 << 18 };
    let groups = threads / 256;

    let mut ring = TransferRing::new(gpu, RingConfig::new(slots, slot_bytes))?;

    // Consumer kernel + one descriptor set per slot (binding 0 differs per slot).
    let k = gpu.kernel(SPV_SLOT_CONSUMER, 2)?;
    let acc = gpu.buf(
        (threads as u64) * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;
    let sets = SlotSets::new(gpu, &k, &ring, &acc)?;
    let set_at = |i: u32| sets.at(i);

    // Pattern data: small integers, safe for an unbounded FMA chain.
    let mut pattern = vec![0u8; slot_bytes as usize];
    for (i, chunk) in pattern.chunks_mut(4).enumerate() {
        chunk.copy_from_slice(&(1.0f32 + (i % 8) as f32).to_le_bytes());
    }
    for s in 0..slots {
        ring.stage(s, &pattern)?;
    }
    // Make every slot resident so the compute-only leg reads real device data.
    let v = ring.last_value();
    for s in 0..slots {
        ring.submit_copy(s, slot_bytes, v + 1 + s as u64)?;
    }
    ring.wait(v + slots as u64)?;
    ring.reset_stats();

    // --- calibrate the compute leg against the measured DMA rate -------------
    let target_per_mib_s = (1u64 << 20) as f64 / (h2d_hint_gbps.max(0.05) * 1e9) * 1.05;
    let cal = calibrate_consumer(gpu, &k, set_at(0), threads, groups, target_per_mib_s)?;
    let iters_for = |bytes: u64| -> u32 {
        let mib = (bytes >> 20).max(1);
        (cal.iters_per_mib as u64 * mib).min(4_000_000) as u32
    };

    let seq = blob_sequence(total_bytes);
    let rounds = seq.len() as u32;
    let streamed: u64 = seq.iter().sum();
    let blobs_4 = seq.iter().filter(|b| **b == 4 << 20).count() as u32;
    let blobs_16 = seq.iter().filter(|b| **b == 16 << 20).count() as u32;

    // --- (a) consumer alone, all slots resident ------------------------------
    let compute_only_seconds = {
        let target = if quick { 0.3 } else { 0.6 };
        let t = gpu.run_timed(Qc::C, target, |d, cmd, n| unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
            for _ in 0..n {
                for (i, &bytes) in seq.iter().enumerate() {
                    let slot = (i as u32) % slots;
                    record_dispatch(
                        d,
                        cmd,
                        &k,
                        set_at(slot),
                        groups,
                        (bytes / 16) as u32,
                        threads,
                        iters_for(bytes),
                    );
                }
            }
        })?;
        t.per_iter_s
    };

    // --- (b) producer alone --------------------------------------------------
    let transfer_only_seconds = {
        let v0 = ring.last_value();
        let t0 = Instant::now();
        for (i, &bytes) in seq.iter().enumerate() {
            ring.submit_copy((i as u32) % slots, bytes, v0 + 1 + i as u64)?;
        }
        ring.wait(v0 + rounds as u64)?;
        t0.elapsed().as_secs_f64()
    };

    // --- (c) pipelined: producer streams while the consumer eats slots -------
    let stream_seconds = {
        // Counters below describe the pipelined run only (not the transfer-only pass).
        ring.reset_stats();
        let v0 = ring.last_value();
        let t0 = Instant::now();
        for (i, &bytes) in seq.iter().enumerate() {
            let slot = (i as u32) % slots;
            let value = v0 + 1 + i as u64;
            ring.submit_copy(slot, bytes, value)?;
            let iters = iters_for(bytes);
            let count = (bytes / 16) as u32;
            ring.submit_compute(slot, value, |d, cmd| {
                record_dispatch(d, cmd, &k, set_at(slot), groups, count, threads, iters);
                Ok(())
            })?;
        }
        ring.wait_release(v0 + rounds as u64)?;
        t0.elapsed().as_secs_f64()
    };
    let producer = ring.stats();

    let (mx, mn) = (
        compute_only_seconds.max(transfer_only_seconds),
        compute_only_seconds.min(transfer_only_seconds).max(1e-9),
    );
    let overlap = 1.0 - ((stream_seconds - mx) / mn);
    let stall = (stream_seconds - compute_only_seconds).max(0.0);
    let mb = streamed as f64 / 1e6;

    let mut notes = vec![
        format!(
            "pipeline depth {} slots of {} MiB; blobs alternate 4 MiB / 16 MiB ({} / {}) from a \
             seeded LCG, the M3 expert-blob shape",
            slots,
            slot_bytes >> 20,
            blobs_4,
            blobs_16
        ),
        format!(
            "consumer leg is calibrated against the measured DMA rate: target {:.2} ms/MiB, \
             {} FMA iterations per MiB per thread, {} threads (read pass {:.2} ms per 16 MiB, \
             {:.2} ns per FMA iteration)",
            cal.target_ms_per_mib,
            cal.iters_per_mib,
            threads,
            cal.read_seconds_16mib * 1000.0,
            cal.slope_seconds_per_iter * 1e9
        ),
        "overlap efficiency uses the phase-0 formula (docs/phase-0/02-vulkan-raw-probe.md §3.4): \
         eff = 1 - (T_both - max(T_compute, T_transfer)) / min(T_compute, T_transfer)"
            .to_string(),
        "consumer stall = T_stream - T_compute_only (time the consumer spent waiting for DMA \
         the hardware could not hide); it includes the pipeline drain of the last slots"
            .to_string(),
    ];
    if gpu.xfer_mode != crate::ctx::XferMode::DistinctFamily {
        notes.push(
            "transfers share the compute queue family on this device; serialization is expected \
             and not a RADV limitation of the ring"
                .to_string(),
        );
    }

    ring.destroy();
    sets.destroy(gpu);
    gpu.destroy_buf(&acc);
    gpu.destroy_kernel(&k);

    Ok(Churn {
        slot_bytes,
        slots,
        pipeline_depth: slots,
        rounds,
        bytes_total: streamed,
        blobs_4mib: blobs_4,
        blobs_16mib: blobs_16,
        compute_only_seconds,
        transfer_only_seconds,
        stream_seconds,
        overlap_efficiency_percent: overlap * 100.0,
        overlap_formula:
            "eff = 1 - (T_both - max(T_compute, T_transfer)) / min(T_compute, T_transfer)"
                .to_string(),
        consumer_stall_seconds: stall,
        consumer_stall_ms_per_mb: stall * 1000.0 / mb.max(1e-9),
        stream_h2d_gbps: streamed as f64 / stream_seconds / 1e9,
        full_rate_h2d_gbps: streamed as f64 / transfer_only_seconds / 1e9,
        consumer_mib_per_s: (streamed >> 20) as f64 / stream_seconds,
        calibrated_iters_per_mib: cal.iters_per_mib,
        consumer_threads: threads,
        compute_seconds_per_blob: compute_only_seconds / rounds as f64,
        transfer_seconds_per_blob: transfer_only_seconds / rounds as f64,
        compute_queue_family: gpu.family_c,
        transfer_queue_family: gpu.family_x,
        producer,
        notes,
    })
}

/// Two-point calibration of the consumer kernel: `iters = 0` gives the read cost,
/// a nominal iteration count gives the FMA slope. `iters_per_mib` then makes the
/// compute leg ≈ the measured DMA leg for the same number of MiB.
pub(crate) fn calibrate_consumer(
    gpu: &Gpu,
    k: &Kernel,
    set0: vk::DescriptorSet,
    threads: u32,
    groups: u32,
    target_per_mib_s: f64,
) -> R<ConsumerCal> {
    let slot_mib = 16.0f64; // calibration pass reads a full 16 MiB slot
    let count = ((16u64 << 20) / 16) as u32;
    let time_once = |iters: u32| -> R<f64> {
        let t = gpu.run_timed(Qc::C, 0.10, |d, cmd, n| unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                k.layout,
                0,
                &[set0],
                &[],
            );
            d.cmd_push_constants(
                cmd,
                k.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push4(count, threads, iters, threads),
            );
            for _ in 0..n {
                d.cmd_dispatch(cmd, groups, 1, 1);
            }
        })?;
        Ok(t.per_iter_s)
    };

    let it0: u32 = 1024;
    let read_s = time_once(0)?;
    let t1 = time_once(it0)?;
    let slope = (t1 - read_s) / f64::from(it0);
    let read_per_mib = read_s / slot_mib;
    let iters_per_mib = if slope > 1e-12 {
        (((target_per_mib_s - read_per_mib) / slope).max(0.0) as u64).min(300_000) as u32
    } else {
        4096
    };
    Ok(ConsumerCal {
        iters_per_mib,
        read_seconds_16mib: read_s,
        slope_seconds_per_iter: slope,
        target_ms_per_mib: target_per_mib_s * 1000.0,
    })
}

// ---------------------------------------------------------------------------
// Environment helpers

pub(crate) fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

pub(crate) fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Layers requested through the loader environment (e.g. the validation run).
fn requested_layers() -> Vec<String> {
    let mut out = Vec::new();
    for key in [
        "VK_INSTANCE_LAYERS",
        "VK_LOADER_LAYERS_ENABLE",
        "VK_LAYER_PATH",
    ] {
        if let Ok(v) = std::env::var(key) {
            for part in v.split([':', ',', ';']) {
                let p = part.trim();
                if !p.is_empty() {
                    out.push(format!("{key}={p}"));
                }
            }
        }
    }
    out
}

/// Processes currently holding a DRM node (measurement-quality evidence).
pub(crate) fn gpu_users() -> Vec<String> {
    let me = std::process::id();
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return out;
    };
    for e in entries.flatten() {
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        let Ok(pid) = name.parse::<u32>() else {
            continue;
        };
        if pid == me {
            continue;
        }
        let Ok(fds) = std::fs::read_dir(e.path().join("fd")) else {
            continue;
        };
        let mut holds = false;
        for fd in fds.flatten() {
            if let Ok(t) = std::fs::read_link(fd.path()) {
                let s = t.to_string_lossy();
                if s.contains("/dev/dri/renderD") || s.contains("/dev/dri/card") {
                    holds = true;
                    break;
                }
            }
        }
        if holds {
            let comm = std::fs::read_to_string(e.path().join("comm")).unwrap_or_default();
            out.push(format!("{pid}:{}", comm.trim()));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `1761464400` → `"2026-10-26T01:00:00Z"` (Howard Hinnant's civil-from-days).
pub(crate) fn iso8601_utc(unix_s: u64) -> String {
    let z = (unix_s / 86_400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let (y, m) = if m <= 2 { (y + 1, m) } else { (y, m) };
    let rem = unix_s % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_sequence_alternates_and_sums() {
        let seq = blob_sequence(64 << 20);
        assert!(seq.len() >= 4, "{seq:?}");
        assert!(seq.iter().all(|b| *b == 4 << 20 || *b == 16 << 20));
        let total: u64 = seq.iter().sum();
        assert!(total >= 64 << 20);
        assert!(total < (64 << 20) + (16 << 20));
        // deterministic across calls (seeded LCG, no rand dependency)
        assert_eq!(seq, blob_sequence(64 << 20));
    }

    #[test]
    fn iso8601_matches_known_epochs() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(1_792_972_800), "2026-10-26T00:00:00Z");
    }

    #[test]
    fn schema_round_trips() {
        let rep = TransferReport {
            schema_version: TRANSFER_SCHEMA_VERSION,
            environment: TransferEnvironment {
                hostname: "h".into(),
                timestamp_unix_s: 1,
                timestamp: iso8601_utc(1),
                instance_api_version: "1.4.363".into(),
                quick: true,
                instance_layers: vec![],
                competing_gpu_processes: vec![],
            },
            devices: vec![DeviceTransfer {
                device: DeviceId::default(),
                queue: TransferQueue {
                    compute_family: 0,
                    mode: "distinct_transfer_queue_family".into(),
                    transfer_family: 1,
                    transfer_queue_index: 0,
                    distinct_from_compute: true,
                    transfer_family_queue_count: 4,
                    compute_family_queue_count: 1,
                },
                capabilities: TransferCaps {
                    timeline_semaphore_supported: true,
                    timeline_semaphore_used: true,
                    host_memory: HostMemoryCaps::default(),
                },
                ring: None,
                churn: None,
                notes: vec![],
                error: None,
            }],
            warnings: vec![],
        };
        let s = serde_json::to_string(&rep).expect("serialize");
        let back: TransferReport = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back.schema_version, TRANSFER_SCHEMA_VERSION);
        assert_eq!(back.devices[0].queue.transfer_family, 1);
    }
}
