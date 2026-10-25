//! Phase-0.2 benchmarks: VRAM bandwidth, FMA throughput, PCIe H2D/D2H, compute/transfer overlap.

use crate::ctx::{api_version_string, driver_version_string, Dev, Gpu, Kernel, Qc, R};
use crate::report::*;
use ash::vk;

const SPV_BANDWIDTH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bandwidth.spv"));
const SPV_FMA_FP32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fma_fp32.spv"));
const SPV_FMA_FP16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fma_fp16.spv"));

const FMA_CHAINS: u32 = 8;

fn push2(a: u32, b: u32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(&a.to_le_bytes());
    out[4..8].copy_from_slice(&b.to_le_bytes());
    out
}

// ---------------------------------------------------------------------------
// Discovery conversion
// ---------------------------------------------------------------------------

fn hex4(v: u32) -> String {
    format!("0x{v:04x}")
}

fn dev_type_name(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::OTHER => "OTHER",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "INTEGRATED_GPU",
        vk::PhysicalDeviceType::DISCRETE_GPU => "DISCRETE_GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "VIRTUAL_GPU",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "UNKNOWN",
    }
}

fn queue_flag_names(f: vk::QueueFlags) -> Vec<String> {
    let all = [
        (vk::QueueFlags::GRAPHICS, "GRAPHICS"),
        (vk::QueueFlags::COMPUTE, "COMPUTE"),
        (vk::QueueFlags::TRANSFER, "TRANSFER"),
        (vk::QueueFlags::SPARSE_BINDING, "SPARSE_BINDING"),
        (vk::QueueFlags::PROTECTED, "PROTECTED"),
    ];
    all.iter()
        .filter(|(b, _)| f.contains(*b))
        .map(|(_, n)| n.to_string())
        .collect()
}

fn subgroup_op_names(f: vk::SubgroupFeatureFlags) -> Vec<String> {
    let all = [
        (vk::SubgroupFeatureFlags::BASIC, "BASIC"),
        (vk::SubgroupFeatureFlags::VOTE, "VOTE"),
        (vk::SubgroupFeatureFlags::ARITHMETIC, "ARITHMETIC"),
        (vk::SubgroupFeatureFlags::BALLOT, "BALLOT"),
        (vk::SubgroupFeatureFlags::SHUFFLE, "SHUFFLE"),
        (vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE, "SHUFFLE_RELATIVE"),
        (vk::SubgroupFeatureFlags::CLUSTERED, "CLUSTERED"),
        (vk::SubgroupFeatureFlags::QUAD, "QUAD"),
    ];
    all.iter()
        .filter(|(b, _)| f.contains(*b))
        .map(|(_, n)| n.to_string())
        .collect()
}

fn subgroup_stage_names(f: vk::ShaderStageFlags) -> Vec<String> {
    let all = [
        (vk::ShaderStageFlags::VERTEX, "VERTEX"),
        (
            vk::ShaderStageFlags::TESSELLATION_CONTROL,
            "TESS_CONTROL",
        ),
        (
            vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            "TESS_EVAL",
        ),
        (vk::ShaderStageFlags::GEOMETRY, "GEOMETRY"),
        (vk::ShaderStageFlags::FRAGMENT, "FRAGMENT"),
        (vk::ShaderStageFlags::COMPUTE, "COMPUTE"),
    ];
    all.iter()
        .filter(|(b, _)| f.contains(*b))
        .map(|(_, n)| n.to_string())
        .collect()
}

pub fn device_id(dev: &Dev) -> DeviceId {
    DeviceId {
        index: dev.index,
        name: dev.name.clone(),
        vendor_id: hex4(dev.vendor_id),
        device_id: hex4(dev.device_id),
        device_type: dev_type_name(dev.device_type).to_string(),
        api_version: api_version_string(dev.api_version),
        driver_version: driver_version_string(dev.driver_version),
        driver_name: dev.driver_name.clone(),
        driver_info: dev.driver_info.clone(),
    }
}

pub fn properties(dev: &Dev) -> Properties {
    let heaps = (0..dev.mem.memory_heap_count)
        .map(|i| Heap {
            index: i,
            size_bytes: dev.mem.memory_heaps[i as usize].size,
            device_local: dev.mem.memory_heaps[i as usize]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
        })
        .collect();
    let families = dev
        .families
        .iter()
        .enumerate()
        .map(|(i, f)| QueueFamily {
            index: i as u32,
            flags: queue_flag_names(f.queue_flags),
            queue_count: f.queue_count,
            timestamp_valid_bits: f.timestamp_valid_bits,
            supports_compute: f.queue_flags.contains(vk::QueueFlags::COMPUTE),
            supports_transfer: f.queue_flags.contains(vk::QueueFlags::TRANSFER),
            supports_graphics: f.queue_flags.contains(vk::QueueFlags::GRAPHICS),
        })
        .collect();

    Properties {
        memory_heaps: heaps,
        queue_families: families,
        subgroup: Subgroup {
            subgroup_size: dev.subgroup_size,
            supported_stages: subgroup_stage_names(dev.subgroup_stages),
            supported_operations: subgroup_op_names(dev.subgroup_ops),
            quad_operations_in_all_stages: dev.quad_all_stages,
            min_subgroup_size: dev.min_subgroup_size,
            max_subgroup_size: dev.max_subgroup_size,
            max_compute_workgroup_subgroups: dev.max_compute_workgroup_subgroups,
        },
        extensions: ExtPresence {
            khr_timeline_semaphore: dev.has_ext("VK_KHR_timeline_semaphore"),
            ext_external_memory_host: dev.has_ext("VK_EXT_external_memory_host"),
            khr_cooperative_matrix: dev.has_ext("VK_KHR_cooperative_matrix"),
            ext_subgroup_size_control: dev.has_ext("VK_EXT_subgroup_size_control"),
            khr_shader_atomic_int64: dev.has_ext("VK_KHR_shader_atomic_int64"),
            total_device_extensions: dev.exts.len() as u32,
        },
        features: FeatPresence {
            shader_float16: dev.feats.shader_float16,
            shader_int8: dev.feats.shader_int8,
            shader_atomic_int64_buffer: dev.feats.atomic64_buffer,
            shader_atomic_int64_shared: dev.feats.atomic64_shared,
            timeline_semaphore: dev.feats.timeline_semaphore,
            subgroup_size_control: dev.feats.subgroup_size_control,
            compute_full_subgroups: dev.feats.compute_full_subgroups,
            cooperative_matrix: dev.feats.cooperative_matrix,
        },
        timestamp_period_ns: dev.timestamp_period_ns,
    }
}

// ---------------------------------------------------------------------------
// 2. VRAM bandwidth
// ---------------------------------------------------------------------------

fn bench_bandwidth(gpu: &Gpu, quick: bool) -> R<Bandwidth> {
    let bytes: u64 = if quick { 128 << 20 } else { 512 << 20 };
    let target = if quick { 0.05 } else { 0.2 };

    let src = gpu.buf(
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;
    let dst = gpu.buf(
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;

    let k = gpu.kernel(SPV_BANDWIDTH, 2)?;
    gpu.bind_kernel(&k, &[src.buf, dst.buf])?;

    let count = (bytes / 16) as u32; // vec4 elements per buffer
    let threads_raw = count / 4; // strided pattern: 4 elements per thread
    let threads = (threads_raw.div_ceil(256)) * 256;
    let groups = (threads / 256).max(1);
    let push = push2(count, threads);

    let t = gpu.run_timed(Qc::C, target, |d, cmd, n| unsafe {
        d.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            k.pipe,
        );
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[k.set],
            &[],
        );
        d.cmd_push_constants(
            cmd,
            k.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        Gpu::barrier_all(d, cmd);
        for _ in 0..n {
            d.cmd_dispatch(cmd, groups, 1, 1);
            Gpu::barrier_all(d, cmd);
        }
    })?;

    let b = bytes as f64;
    let read = b / t.per_iter_s / 1e9;
    let write = b / t.per_iter_s / 1e9;
    let copy = 2.0 * b / t.per_iter_s / 1e9;

    gpu.destroy_kernel(&k);
    gpu.destroy_buf(&src);
    gpu.destroy_buf(&dst);
    gpu.wait_idle()?;

    Ok(Bandwidth {
        buffer_bytes: bytes,
        iterations: t.iters,
        seconds: t.total_s,
        wall_seconds: t.wall_s,
        read_gbps: read,
        write_gbps: write,
        copy_gbps: copy,
        notes: vec![format!(
            "kernel: strided vec4 read(src)+write(dst), {} x {} MiB device-local, \
             timed by queue timestamps ({} ticks @ {} ns/tick)",
            2,
            bytes >> 20,
            "TYPE_64",
            gpu.timestamp_period_ns
        )],
    })
}

// ---------------------------------------------------------------------------
// 4. PCIe host <-> device
// ---------------------------------------------------------------------------

pub struct PcieRes {
    pub report: Pcie,
    pub host: crate::ctx::Buf,
    pub dev: crate::ctx::Buf,
}

fn bench_pcie(gpu: &Gpu, quick: bool) -> R<PcieRes> {
    let sizes_mb: Vec<u64> = if quick {
        vec![16, 64]
    } else {
        vec![16, 64, 256, 1024]
    };
    let max_mb = *sizes_mb.iter().max().expect("non-empty");
    let target = if quick { 0.05 } else { 0.2 };

    let host = gpu.buf(
        max_mb << 20,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(),
    )?;
    let dev = gpu.buf(
        max_mb << 20,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;

    // Fill the staging buffer with a pattern so every page is committed.
    let mut whole = vec![0u8; host.size as usize];
    for chunk in whole.chunks_mut(1 << 20) {
        chunk.fill(0xA5);
    }
    gpu.write_host(&host, &whole)?;
    drop(whole);

    let mut sizes = Vec::new();
    for &mb in &sizes_mb {
        let bytes = mb << 20;

        let h2d = gpu.run_timed(Qc::X, target, |d, cmd, n| unsafe {
            Gpu::barrier_all(d, cmd);
            for _ in 0..n {
                let r = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes,
                };
                d.cmd_copy_buffer(cmd, host.buf, dev.buf, &[r]);
                Gpu::barrier_all(d, cmd);
            }
        })?;
        let d2h = gpu.run_timed(Qc::X, target, |d, cmd, n| unsafe {
            Gpu::barrier_all(d, cmd);
            for _ in 0..n {
                let r = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes,
                };
                d.cmd_copy_buffer(cmd, dev.buf, host.buf, &[r]);
                Gpu::barrier_all(d, cmd);
            }
        })?;

        sizes.push(PcieSize {
            size_mb: mb,
            h2d_gbps: bytes as f64 / h2d.per_iter_s / 1e9,
            d2h_gbps: bytes as f64 / d2h.per_iter_s / 1e9,
            h2d_seconds_per_copy: h2d.per_iter_s,
            d2h_seconds_per_copy: d2h.per_iter_s,
        });
    }

    let distinct = gpu.family_x != gpu.family_c || gpu.queue_x_index != 0;
    let report = Pcie {
        queue_family: gpu.family_x,
        queue_index: gpu.queue_x_index,
        distinct_from_compute_queue: distinct,
        staging_memory: "HOST_VISIBLE|HOST_COHERENT".to_string(),
        device_memory: "DEVICE_LOCAL".to_string(),
        sizes,
        notes: vec![
            "staging buffer is vkAllocateMemory'd host memory; the driver pins \
             (page-locks) it while it is device-visible, which is the Vulkan \
             equivalent of a CUDA host registration, not malloc'd pageable RAM"
                .to_string(),
            format!(
                "copies submitted on queue family {} index {} ({})",
                gpu.family_x,
                gpu.queue_x_index,
                gpu.xfer_mode.as_str()
            ),
        ],
    };
    Ok(PcieRes { report, host, dev })
}

// ---------------------------------------------------------------------------
// 3. Compute throughput (FMA)
// ---------------------------------------------------------------------------

fn upload_pattern(gpu: &Gpu, staging: &crate::ctx::Buf, dst: &crate::ctx::Buf, bytes: u64) -> R<()> {
    // values 1..=8 (exact in both f32 and f16)
    let mut buf = vec![0u8; bytes as usize];
    for (i, b) in buf.chunks_mut(4).enumerate() {
        let v = (1 + (i % 8)) as f32;
        b.copy_from_slice(&v.to_le_bytes());
    }
    gpu.write_host(staging, &buf)?;
    gpu.copy_once(staging, dst, bytes)
}

fn bench_compute(
    gpu: &Gpu,
    quick: bool,
    staging: &crate::ctx::Buf,
) -> R<(Compute, Kernel, crate::ctx::Buf, u32, f64)> {
    let threads: u32 = if quick { 1 << 18 } else { 1 << 20 };
    let rounds: u32 = if quick { 1024 } else { 4096 };
    let target = if quick { 0.05 } else { 0.2 };
    let groups = threads / 256;

    let k = gpu.kernel(SPV_FMA_FP32, 1)?;
    let data = gpu.buf(
        (threads as u64) * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;
    upload_pattern(gpu, staging, &data, (threads as u64) * 4)?;
    gpu.bind_kernel(&k, &[data.buf])?;

    let push = push2(rounds, threads);
    let t = gpu.run_timed(Qc::C, target, |d, cmd, n| unsafe {
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[k.set],
            &[],
        );
        d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
        Gpu::barrier_all(d, cmd);
        for _ in 0..n {
            d.cmd_dispatch(cmd, groups, 1, 1);
            Gpu::barrier_all(d, cmd);
        }
    })?;

    let flops_per_dispatch =
        threads as f64 * rounds as f64 * FMA_CHAINS as f64 * 2.0;
    let fp32_gflops = flops_per_dispatch / t.per_iter_s / 1e9;

    let mut notes = vec![format!(
        "fp32: {threads} threads x {rounds} rounds x {FMA_CHAINS} independent fma() chains \
         (fma counted as 2 flops); {} timestamps",
        if t.used_timestamps { "queue" } else { "wall-clock fallback" }
    )];

    // --- fp16 ---
    let mut fp16_gflops = None;
    let mut fp16_supported = gpu.feats.shader_float16;
    if fp16_supported {
        match bench_fp16(gpu, staging, threads, rounds, target) {
            Ok(v) => {
                fp16_gflops = Some(v.0);
                notes.push(v.1);
            }
            Err(e) => {
                fp16_supported = false;
                notes.push(format!("fp16 test failed, skipped: {e}"));
            }
        }
    } else {
        notes.push(
            "shaderFloat16 not supported by this device -> fp16 test skipped".to_string(),
        );
    }

    let report = Compute {
        fp32_gflops,
        fp16_gflops,
        fp16_supported,
        threads,
        fma_chains: FMA_CHAINS,
        fma_rounds_per_dispatch: rounds,
        seconds_per_dispatch: t.per_iter_s,
        notes,
    };
    Ok((report, k, data, rounds, t.per_iter_s))
}

fn bench_fp16(
    gpu: &Gpu,
    staging: &crate::ctx::Buf,
    threads: u32,
    rounds: u32,
    target: f64,
) -> R<(f64, String)> {
    let groups = threads / 256;
    let k = gpu.kernel(SPV_FMA_FP16, 1)?;
    let data = gpu.buf(
        (threads as u64) * 2,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;
    // f16 pattern: integers 1..=8 are exactly representable in binary16.
    let mut buf = vec![0u8; (threads as u64 * 2) as usize];
    for (i, b) in buf.chunks_mut(2).enumerate() {
        let f = (1 + (i % 8)) as f32;
        let bits = f32_to_f16(f);
        b.copy_from_slice(&bits.to_le_bytes());
    }
    gpu.write_host(staging, &buf)?;
    gpu.copy_once(staging, &data, (threads as u64) * 2)?;
    gpu.bind_kernel(&k, &[data.buf])?;

    let push = push2(rounds, threads);
    let t = gpu.run_timed(Qc::C, target, |d, cmd, n| unsafe {
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[k.set],
            &[],
        );
        d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
        Gpu::barrier_all(d, cmd);
        for _ in 0..n {
            d.cmd_dispatch(cmd, groups, 1, 1);
            Gpu::barrier_all(d, cmd);
        }
    })?;

    let flops = threads as f64 * rounds as f64 * FMA_CHAINS as f64 * 2.0;
    let gflops = flops / t.per_iter_s / 1e9;

    gpu.destroy_kernel(&k);
    gpu.destroy_buf(&data);

    Ok((
        gflops,
        format!(
            "fp16: shaderFloat16 enabled, {threads} threads x {rounds} rounds x \
             {FMA_CHAINS} chains; packed/pair math means reported GFLOPS counts \
             binary16 ops as issued (see report caveats)"
        ),
    ))
}

/// Round-to-nearest (actually truncation; inputs are small integers) f32 -> f16 bits.
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = (((b >> 23) & 0xff) as i32) - 127 + 15;
    let man = b & 0x007f_ffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((man >> 13) as u16)
}

// ---------------------------------------------------------------------------
// 5. Compute / transfer overlap
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench_overlap(
    gpu: &Gpu,
    quick: bool,
    k: &Kernel,
    _data: &crate::ctx::Buf,
    threads: u32,
    rounds0: u32,
    t_dispatch0: f64,
    pcie: &PcieRes,
) -> R<Overlap> {
    let target = if quick { 0.3 } else { 1.0 };

    // Scale the FMA kernel so one dispatch lasts ~target seconds.
    let rounds1 = {
        let scaled = (rounds0 as f64) * (target / t_dispatch0.max(1e-9));
        (scaled.ceil() as u32).clamp(1, 200_000_000)
    };
    let groups = threads / 256;
    let push_c = push2(rounds1, threads);

    // Transfer leg: 64 MiB H2D + D2H pairs, repeated to last ~target seconds.
    let pair_mb: u64 = 64;
    let pair = pcie
        .report
        .sizes
        .iter()
        .find(|s| s.size_mb == pair_mb)
        .or_else(|| pcie.report.sizes.last())
        .ok_or_else(|| "no pcie sizes measured".to_string())?;
    let pair_bytes = pair.size_mb << 20;
    let pair_s = pair.h2d_seconds_per_copy + pair.d2h_seconds_per_copy;
    let copies = ((target / pair_s.max(1e-9)).ceil() as u64).clamp(1, 100_000) as u32;
    let transfer_bytes = (copies as u64) * pair_bytes * 2;

    // --- (a) compute alone -------------------------------------------------
    let compute_s = gpu.wall_run(Qc::C, |d, cmd| unsafe {
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[k.set],
            &[],
        );
        d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, &push_c);
        Gpu::barrier_all(d, cmd);
        d.cmd_dispatch(cmd, groups, 1, 1);
        Gpu::barrier_all(d, cmd);
    })?;

    // --- (b) transfer alone ------------------------------------------------
    let host = &pcie.host;
    let devb = &pcie.dev;
    let transfer_s = gpu.wall_run(Qc::X, |d, cmd| unsafe {
        Gpu::barrier_all(d, cmd);
        for _ in 0..copies {
            let up = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: pair_bytes,
            };
            d.cmd_copy_buffer(cmd, host.buf, devb.buf, &[up]);
            Gpu::barrier_all(d, cmd);
            let down = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: pair_bytes,
            };
            d.cmd_copy_buffer(cmd, devb.buf, host.buf, &[down]);
            Gpu::barrier_all(d, cmd);
        }
    })?;

    // --- (c) both at once --------------------------------------------------
    // Record both command buffers first, then submit them back-to-back and wait
    // for both fences; wall clock spans the whole concurrent window.
    gpu.record(Qc::C, |d, cmd| unsafe {
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[k.set],
            &[],
        );
        d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, &push_c);
        Gpu::barrier_all(d, cmd);
        d.cmd_dispatch(cmd, groups, 1, 1);
        Gpu::barrier_all(d, cmd);
    })?;
    gpu.record(Qc::X, |d, cmd| unsafe {
        Gpu::barrier_all(d, cmd);
        for _ in 0..copies {
            let up = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: pair_bytes,
            };
            d.cmd_copy_buffer(cmd, host.buf, devb.buf, &[up]);
            Gpu::barrier_all(d, cmd);
            let down = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: pair_bytes,
            };
            d.cmd_copy_buffer(cmd, devb.buf, host.buf, &[down]);
            Gpu::barrier_all(d, cmd);
        }
    })?;
    let t0 = std::time::Instant::now();
    let f1 = gpu.submit(Qc::C)?;
    let f2 = gpu.submit(Qc::X)?;
    gpu.wait_fence(f1)?;
    gpu.wait_fence(f2)?;
    let both_s = t0.elapsed().as_secs_f64();

    // efficiency = 1 - (T_both - max(Tc, Tt)) / min(Tc, Tt)
    let (mx, mn) = (
        compute_s.max(transfer_s),
        compute_s.min(transfer_s).max(1e-9),
    );
    let efficiency = 1.0 - ((both_s - mx) / mn);
    let mut notes = vec![
        format!(
            "compute leg: 1 dispatch of {rounds1} FMA rounds (~{target:.1}s of work); \
             transfer leg: {copies} x {pair_mb} MiB H2D + D2H ({transfer_bytes} bytes total)"
        ),
        format!(
            "queue setup: compute on family {} index 0, transfer on family {} index {} ({})",
            gpu.family_c,
            gpu.family_x,
            gpu.queue_x_index,
            gpu.xfer_mode.as_str()
        ),
    ];
    if gpu.xfer_mode == crate::ctx::XferMode::SameQueue {
        notes.push(
            "no separate transfer queue on this device: both submissions land on the \
             same VkQueue and serialize, so overlap efficiency is expected to be ~0%"
                .to_string(),
        );
    }

    Ok(Overlap {
        compute_queue_family: gpu.family_c,
        transfer_queue_family: gpu.family_x,
        queue_mode: gpu.xfer_mode.as_str().to_string(),
        compute_seconds: compute_s,
        transfer_seconds: transfer_s,
        both_seconds: both_s,
        efficiency_percent: efficiency * 100.0,
        formula: "eff = 1 - (T_both - max(T_compute, T_transfer)) / min(T_compute, T_transfer)".to_string(),
        compute_work_seconds: target,
        transfer_work_bytes: transfer_bytes,
        notes,
    })
}

// ---------------------------------------------------------------------------
// Orchestration for one device
// ---------------------------------------------------------------------------

pub fn run_device(dev: &Dev, gpu: &Gpu, quick: bool) -> DeviceReport {
    let mut rep = DeviceReport {
        device: device_id(dev),
        properties: properties(dev),
        notes: Vec::new(),
        ..Default::default()
    };

    // 2. VRAM bandwidth
    match bench_bandwidth(gpu, quick) {
        Ok(b) => {
            println!(
                "  [{}] VRAM bandwidth: copy {:.1} GB/s (read {:.1}, write {:.1})",
                gpu.index, b.copy_gbps, b.read_gbps, b.write_gbps
            );
            rep.vram_bandwidth_gbps = Some(b);
        }
        Err(e) => {
            rep.notes.push(format!("vram_bandwidth skipped: {e}"));
            eprintln!("  [{}] bandwidth failed: {e}", gpu.index);
        }
    }

    // 4. PCIe (creates staging + device buffers reused by compute init and overlap)
    let pcie = match bench_pcie(gpu, quick) {
        Ok(p) => {
            for s in &p.report.sizes {
                println!(
                    "  [{}] PCIe {:>4} MiB: H2D {:.2} GB/s, D2H {:.2} GB/s",
                    gpu.index, s.size_mb, s.h2d_gbps, s.d2h_gbps
                );
            }
            Some(p)
        }
        Err(e) => {
            rep.notes.push(format!("pcie skipped: {e}"));
            eprintln!("  [{}] pcie failed: {e}", gpu.index);
            None
        }
    };

    // 3. Compute (fp32 + fp16), initialized through the staging buffer
    let compute = match &pcie {
        Some(p) => match bench_compute(gpu, quick, &p.host) {
            Ok((c, k, data, rounds, t_disp)) => {
                println!(
                    "  [{}] FP32: {:.0} GFLOPS{}",
                    gpu.index,
                    c.fp32_gflops,
                    c.fp16_gflops
                        .map(|v| format!(", FP16: {v:.0} GFLOPS"))
                        .unwrap_or_default()
                );
                let threads = c.threads;
                rep.compute_gflops = Some(c);
                Some((k, data, threads, rounds, t_disp))
            }
            Err(e) => {
                rep.notes.push(format!("compute skipped: {e}"));
                eprintln!("  [{}] compute failed: {e}", gpu.index);
                None
            }
        },
        None => {
            rep.notes.push("compute skipped: no staging buffer (pcie test failed)".into());
            None
        }
    };

    // 5. Overlap (needs compute kernel + pcie buffers)
    if let (Some(p), Some((k, data, threads, rounds, t_disp))) = (&pcie, &compute) {
        match bench_overlap(gpu, quick, k, data, *threads, *rounds, *t_disp, p) {
            Ok(o) => {
                println!(
                    "  [{}] overlap: compute {:.2}s, transfer {:.2}s, both {:.2}s -> {:.1}% efficiency",
                    gpu.index, o.compute_seconds, o.transfer_seconds, o.both_seconds,
                    o.efficiency_percent
                );
                rep.overlap = Some(o);
            }
            Err(e) => {
                rep.notes.push(format!("overlap skipped: {e}"));
                eprintln!("  [{}] overlap failed: {e}", gpu.index);
            }
        }
    }

    // Cleanup (scoped: pcie staging/dev are still alive here).
    if let Some((k, data, ..)) = &compute {
        gpu.destroy_kernel(k);
        gpu.destroy_buf(data);
    }
    if let Some(p) = &pcie {
        gpu.destroy_buf(&p.host);
        gpu.destroy_buf(&p.dev);
    }
    let _ = gpu.wait_idle();

    rep
}
