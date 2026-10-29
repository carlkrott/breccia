//! Compression fabric F2/F3 (`breccia bench codec`): GPU static interleaved-rANS decode
//! rate, then delivered raw GB/s through the transfer ring (host -> DMA -> decode) against
//! a raw DMA of the same bytes, with and without a calibrated compute consumer.
//!
//! Frames come from `docs/coord/codecframes.py` (`.brcf`); the format is documented there
//! and in `shaders/rans_decode.comp`. Every decoded frame is compared bit-exact against the
//! raw bytes shipped in the same file.

use ash::vk;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::bench;
use crate::ctx::{Buf, Dev, Gpu, Host, Kernel, Qc, R};
use crate::report::DeviceId;
use crate::transfer::{RingConfig, TransferRing};
use crate::transfer_bench::{
    calibrate_consumer, gpu_users, hostname, iso8601_utc, now_unix, record_dispatch,
    select_devices, DeviceSelector, TransferBenchOptions, SPV_SLOT_CONSUMER,
};

const SPV_RANS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rans_decode.spv"));
pub const CODEC_SCHEMA_VERSION: u32 = 1;
const MAGIC: u32 = 0x4643_5242;
const MAX_CTX: u32 = 4; // rans_decode.comp MAX_CTX
const CTX_BYTES: usize = 4096 + 256 * 4;
/// Consumer load as a multiple of the measured raw DMA time per MiB (0 = idle receiver).
const FACTORS: [f64; 3] = [0.0, 0.5, 1.05];

#[derive(Debug, Clone, Default)]
pub struct CodecBenchOptions {
    pub quick: bool,
    pub devices: Vec<DeviceSelector>,
    pub frames: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecReport {
    pub schema_version: u32,
    pub hostname: String,
    pub timestamp: String,
    pub quick: bool,
    pub competing_gpu_processes: Vec<String>,
    pub devices: Vec<DeviceCodec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCodec {
    pub device: DeviceId,
    pub classes: Vec<ClassResult>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassResult {
    pub file: String,
    pub mode: u32,
    pub blk: u32,
    pub n_ctx: u32,
    pub lanes: u32,
    pub frame_raw: u64,
    pub n_frames: u32,
    pub image_bytes: Vec<u64>,
    /// Sum of image bytes / sum of raw bytes. Excludes the static tables (pre-shared).
    pub ratio: f64,
    pub table_bytes: u64,
    /// Idle decode, raw bytes out per second, one entry per frame.
    pub decode_gbps: Vec<f64>,
    pub decode_gbps_all: f64,
    pub decode_bit_exact: bool,
    /// Raw DMA rate of this frame set through the ring, no consumer.
    pub raw_dma_gbps: f64,
    /// Go/no-go: decode must beat L / r.
    pub needed_decode_gbps: f64,
    /// Diagnostics, no consumer: raw-equivalent GB/s when only the images are copied,
    /// and when only the decode runs (256 B copies over images already in the slots).
    pub image_dma_raw_gbps: f64,
    pub ring_decode_gbps: f64,
    pub arms: Vec<ArmPoint>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmPoint {
    pub consumer_factor: f64,
    pub consumer_iters_per_mib: u32,
    pub blobs_per_run: u32,
    /// Delivered raw GB/s per run (raw arm: DMA of raw; codec arm: DMA of image + decode).
    pub raw_gbps: Vec<f64>,
    pub codec_gbps: Vec<f64>,
    pub raw_mean: f64,
    pub codec_mean: f64,
    pub raw_spread: f64,
    pub codec_spread: f64,
    pub gain: f64,
    pub codec_bit_exact: bool,
}

struct Frames {
    file: String,
    mode: u32,
    blk: u32,
    n_ctx: u32,
    lanes: u32,
    frame_raw: usize,
    tables: Vec<u8>,
    /// (image, raw)
    frames: Vec<(Vec<u8>, Vec<u8>)>,
}

impl Frames {
    fn wpl(&self) -> u32 {
        (self.frame_raw / 4) as u32 / self.lanes
    }

    /// The 7 push-constant words of `rans_decode.comp` for frame `i`.
    fn push(&self, i: usize) -> [u8; 28] {
        let img = &self.frames[i].0;
        let off_l = u32::from_le_bytes(img[4 * self.lanes as usize..][..4].try_into().unwrap());
        let words = [
            self.lanes,
            self.wpl(),
            self.lanes + 1,
            self.lanes + 1 + off_l,
            self.blk,
            self.mode,
            self.n_ctx,
        ];
        let mut out = [0u8; 28];
        for (k, w) in words.iter().enumerate() {
            out[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

fn le(b: &[u8], at: usize) -> R<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| format!("truncated at byte {at}"))
}

fn load_frames(path: &Path) -> R<Frames> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let h: Vec<u32> = (0..8).map(|k| le(&b, 4 * k)).collect::<R<_>>()?;
    let (magic, ver, mode, blk, n_ctx, lanes, frame_raw, n_frames) =
        (h[0], h[1], h[2], h[3], h[4], h[5], h[6] as usize, h[7]);
    if magic != MAGIC || ver != 1 {
        return Err(format!("{}: not a v1 .brcf file", path.display()));
    }
    if mode > 1 || n_ctx == 0 || n_ctx > MAX_CTX || blk == 0 || blk > 256 || lanes == 0 {
        return Err(format!("{}: bad header {h:?}", path.display()));
    }
    if frame_raw % (4 * lanes as usize) != 0 {
        return Err(format!("{}: frame_raw {frame_raw} not a multiple of 4*lanes", path.display()));
    }
    let tab_end = 32 + 256 + n_ctx as usize * CTX_BYTES;
    let tables = b.get(32..tab_end).ok_or("truncated tables")?.to_vec();
    let mut at = tab_end;
    let mut frames = Vec::new();
    for fi in 0..n_frames {
        let n = le(&b, at)? as usize;
        at += 4;
        let img = b.get(at..at + n).ok_or_else(|| format!("frame {fi}: truncated image"))?;
        let raw = b
            .get(at + n..at + n + frame_raw)
            .ok_or_else(|| format!("frame {fi}: truncated raw"))?;
        // the shader trusts the offsets: check they stay inside the image
        let words = n / 4;
        let off_l = le(img, 4 * lanes as usize)? as usize;
        let plane = if mode == 1 { frame_raw / 2 } else { 0 };
        if n % 4 != 0 || lanes as usize + 1 + off_l + plane.div_ceil(4) > words {
            return Err(format!("frame {fi}: offsets/plane overrun the image"));
        }
        frames.push((img.to_vec(), raw.to_vec()));
        at += n + frame_raw;
    }
    Ok(Frames {
        file: path.display().to_string(),
        mode,
        blk,
        n_ctx,
        lanes,
        frame_raw,
        tables,
        frames,
    })
}

/// Descriptor sets whose bindings are arbitrary buffers (one row of buffers per set).
struct Sets {
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl Sets {
    fn new(gpu: &Gpu, k: &Kernel, rows: &[Vec<vk::Buffer>]) -> R<Self> {
        let n = rows.len() as u32;
        let per = rows.iter().map(|r| r.len()).max().unwrap_or(1) as u32;
        unsafe {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n * per)];
            let dpci = vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes);
            let pool = crate::ctx::vk(
                gpu.device.create_descriptor_pool(&dpci, None),
                "vkCreateDescriptorPool(codec)",
            )?;
            let layouts = vec![k.dsl; n as usize];
            let dai = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            let sets = match gpu.device.allocate_descriptor_sets(&dai) {
                Ok(s) => s,
                Err(e) => {
                    gpu.device.destroy_descriptor_pool(pool, None);
                    return Err(format!("vkAllocateDescriptorSets(codec): {e}"));
                }
            };
            for (set, row) in sets.iter().zip(rows) {
                let infos: Vec<vk::DescriptorBufferInfo> = row
                    .iter()
                    .map(|&buffer| vk::DescriptorBufferInfo {
                        buffer,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    })
                    .collect();
                let writes: Vec<vk::WriteDescriptorSet> = infos
                    .iter()
                    .enumerate()
                    .map(|(b, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(b as u32)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(info))
                    })
                    .collect();
                gpu.device.update_descriptor_sets(&writes, &[]);
            }
            Ok(Self { pool, sets })
        }
    }

    fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// Device-local buffer readable by compute and the transfer family (copy_once runs on X).
fn dev_buf(gpu: &Gpu, size: u64) -> R<Buf> {
    gpu.buf_with(
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
        Some(&[gpu.family_c, gpu.family_x]),
    )
}

fn host_buf(gpu: &Gpu, size: u64) -> R<Buf> {
    gpu.buf(
        size,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::HOST_CACHED,
    )
}

fn upload(gpu: &Gpu, stage: &Buf, dst: &Buf, data: &[u8]) -> R<()> {
    gpu.write_host(stage, data)?;
    gpu.copy_once(stage, dst, data.len() as u64)
}

fn download_eq(gpu: &Gpu, stage: &Buf, src: &Buf, want: &[u8]) -> R<bool> {
    gpu.copy_once(src, stage, want.len() as u64)?;
    let mut got = vec![0u8; want.len()];
    gpu.read_host(stage, &mut got)?;
    Ok(got == want)
}

fn record_decode(d: &ash::Device, cmd: vk::CommandBuffer, k: &Kernel, set: vk::DescriptorSet, push: &[u8; 28], lanes: u32) {
    unsafe {
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
        d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, k.layout, 0, &[set], &[]);
        d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
        d.cmd_dispatch(cmd, lanes.div_ceil(256), 1, 1);
        Gpu::barrier_all(d, cmd);
    }
}

fn mean_spread(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let hi = v.iter().cloned().fold(f64::MIN, f64::max);
    let lo = v.iter().cloned().fold(f64::MAX, f64::min);
    (m, hi - lo)
}

pub fn run_codec_bench(opts: &CodecBenchOptions) -> anyhow::Result<CodecReport> {
    let sets: Vec<Frames> = opts
        .frames
        .iter()
        .map(|p| load_frames(p))
        .collect::<R<_>>()
        .map_err(|e| anyhow::anyhow!("breccia-vk codec: {e}"))?;
    let host = Host::new().map_err(|e| anyhow::anyhow!("breccia-vk: {e}"))?;
    let devices = host.devices().map_err(|e| anyhow::anyhow!("breccia-vk: {e}"))?;
    let topts = TransferBenchOptions {
        quick: opts.quick,
        devices: opts.devices.clone(),
    };
    let selected = match select_devices(&devices, &topts) {
        Ok(s) => s,
        Err(e) => {
            host.destroy();
            return Err(anyhow::anyhow!("breccia-vk: {e}"));
        }
    };
    let mut out = Vec::new();
    for index in selected {
        let dev: &Dev = devices.iter().find(|d| d.index == index).expect("selected device exists");
        println!("breccia-vk: codec bench on {} [{}]", dev.name, dev.pci_address);
        let gpu = match Gpu::create(&host, dev, dev.index) {
            Ok(g) => g,
            Err(e) => {
                out.push(DeviceCodec { device: bench::device_id(dev), classes: vec![], error: Some(e) });
                continue;
            }
        };
        let classes = sets.iter().map(|f| bench_class(&gpu, f, opts.quick)).collect();
        gpu.destroy();
        out.push(DeviceCodec { device: bench::device_id(dev), classes, error: None });
    }
    host.destroy();
    Ok(CodecReport {
        schema_version: CODEC_SCHEMA_VERSION,
        hostname: hostname(),
        timestamp: iso8601_utc(now_unix()),
        quick: opts.quick,
        competing_gpu_processes: gpu_users(),
        devices: out,
    })
}

fn bench_class(gpu: &Gpu, f: &Frames, quick: bool) -> ClassResult {
    let image_bytes: Vec<u64> = f.frames.iter().map(|(i, _)| i.len() as u64).collect();
    let mut r = ClassResult {
        file: f.file.clone(),
        mode: f.mode,
        blk: f.blk,
        n_ctx: f.n_ctx,
        lanes: f.lanes,
        frame_raw: f.frame_raw as u64,
        n_frames: f.frames.len() as u32,
        ratio: image_bytes.iter().sum::<u64>() as f64 / (f.frame_raw * f.frames.len()) as f64,
        image_bytes,
        table_bytes: f.tables.len() as u64,
        decode_gbps: vec![],
        decode_gbps_all: 0.0,
        decode_bit_exact: false,
        raw_dma_gbps: 0.0,
        needed_decode_gbps: 0.0,
        image_dma_raw_gbps: 0.0,
        ring_decode_gbps: 0.0,
        arms: vec![],
        error: None,
    };
    if let Err(e) = bench_class_inner(gpu, f, quick, &mut r) {
        r.error = Some(e);
    }
    println!(
        "  {}: ratio {:.4}, decode {:.2} GB/s (bit-exact {}), raw DMA {:.2} GB/s, need > {:.2}, \
         ring image-DMA-only {:.2}, ring decode-only {:.2}{}",
        Path::new(&f.file).file_name().map(|s| s.to_string_lossy()).unwrap_or_default(),
        r.ratio,
        r.decode_gbps_all,
        r.decode_bit_exact,
        r.raw_dma_gbps,
        r.needed_decode_gbps,
        r.image_dma_raw_gbps,
        r.ring_decode_gbps,
        r.error.as_deref().map(|e| format!(" ERROR {e}")).unwrap_or_default()
    );
    for a in &r.arms {
        println!(
            "    consumer x{:.2}: raw {:.3} ±{:.3}  codec {:.3} ±{:.3} GB/s  gain {:.3}x  bit-exact {}",
            a.consumer_factor, a.raw_mean, a.raw_spread, a.codec_mean, a.codec_spread, a.gain, a.codec_bit_exact
        );
    }
    r
}

fn bench_class_inner(gpu: &Gpu, f: &Frames, quick: bool, r: &mut ClassResult) -> R<()> {
    let n = f.frames.len();
    let max_img = f.frames.iter().map(|(i, _)| i.len()).max().unwrap_or(0);
    // >= 16 MiB: calibrate_consumer reads a full 16 MiB through binding 0
    let slot_bytes = (max_img as u64 + 64).max(f.frame_raw as u64).max(16 << 20); // +64: decoder read-ahead
    let kd = gpu.kernel(SPV_RANS, 3)?;
    let kc = gpu.kernel(SPV_SLOT_CONSUMER, 2)?;
    let stage = host_buf(gpu, slot_bytes)?;
    let tabs = dev_buf(gpu, f.tables.len() as u64)?;
    upload(gpu, &stage, &tabs, &f.tables)?;
    let mut bufs = vec![stage, tabs];

    // --- F2: idle decode rate per frame + bit-exact --------------------------
    let res = (|| -> R<()> {
        let imgs: Vec<Buf> = (0..n).map(|_| dev_buf(gpu, max_img as u64 + 64)).collect::<R<_>>()?;
        let out = dev_buf(gpu, slot_bytes)?;
        let rows: Vec<Vec<vk::Buffer>> = imgs.iter().map(|i| vec![bufs[1].buf, i.buf, out.buf]).collect();
        let ds = Sets::new(gpu, &kd, &rows)?;
        let mut exact = true;
        let mut total_s = 0.0;
        let res = (|| -> R<()> {
            for (i, (img, raw)) in f.frames.iter().enumerate() {
                upload(gpu, &bufs[0], &imgs[i], img)?;
                let push = f.push(i);
                let t = gpu.run_timed(Qc::C, if quick { 0.2 } else { 0.5 }, |d, cmd, k| {
                    for _ in 0..k {
                        record_decode(d, cmd, &kd, ds.sets[i], &push, f.lanes);
                    }
                })?;
                total_s += t.per_iter_s;
                r.decode_gbps.push(f.frame_raw as f64 / t.per_iter_s / 1e9);
                exact &= download_eq(gpu, &bufs[0], &out, raw)?;
            }
            Ok(())
        })();
        ds.destroy(gpu);
        for b in imgs.iter().chain(std::iter::once(&out)) {
            gpu.destroy_buf(b);
        }
        res?;
        r.decode_gbps_all = (f.frame_raw * n) as f64 / total_s / 1e9;
        r.decode_bit_exact = exact;
        Ok(())
    })();
    if res.is_err() {
        cleanup(gpu, &bufs, &kd, &kc);
        return res;
    }

    // --- F3: ring arms --------------------------------------------------------
    let slots = n.min(6) as u32;
    let res = (|| -> R<()> {
        let mut ring = TransferRing::new(gpu, RingConfig::new(slots, slot_bytes))?;
        let threads: u32 = if quick { 1 << 17 } else { 1 << 18 };
        let groups = threads / 256;
        let outs: Vec<Buf> = (0..slots).map(|_| dev_buf(gpu, slot_bytes)).collect::<R<_>>()?;
        let acc = gpu.buf(
            threads as u64 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        let dec_rows: Vec<Vec<vk::Buffer>> =
            (0..slots).map(|s| vec![bufs[1].buf, ring.slot_buffer(s), outs[s as usize].buf]).collect();
        let raw_rows: Vec<Vec<vk::Buffer>> = (0..slots).map(|s| vec![ring.slot_buffer(s), acc.buf]).collect();
        let out_rows: Vec<Vec<vk::Buffer>> = outs.iter().map(|o| vec![o.buf, acc.buf]).collect();
        let ds = Sets::new(gpu, &kd, &dec_rows)?;
        let rs = Sets::new(gpu, &kc, &raw_rows)?;
        let os = Sets::new(gpu, &kc, &out_rows)?;
        let blobs: u32 = if quick { 24 } else { 64 };
        let repeats = if quick { 3 } else { 5 };
        let pushes: Vec<[u8; 28]> = (0..slots as usize).map(|s| f.push(s % n)).collect();
        let raw_len = f.frame_raw as u64;
        let img_len: Vec<u64> = (0..slots as usize).map(|s| f.frames[s % n].0.len() as u64).collect();

        // diag only: 1 drops the pre-decode barrier, 2 adds one before the consumer
        let nobar: u8 = std::env::var("BRECCIA_CODEC_BAR").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        // diag only: 1 decode without the output store, 2 fill 16 MiB of outs[s] after the consumer
        let extra = std::cell::Cell::new(0u8);
        let res = (|| -> R<()> {
            let stage_all = |ring: &TransferRing<'_>, codec: bool| -> R<()> {
                for s in 0..slots {
                    let (img, raw) = &f.frames[s as usize % n];
                    ring.stage(s, if codec { img } else { raw })?;
                }
                Ok(())
            };
            // one run: `blobs` frames through the ring; returns delivered raw GB/s.
            // copy: 0 raw frame, 1 image, 2 a 256 B prefix of the image already in the slot
            let run = |ring: &mut TransferRing<'_>, dsets: &Sets, copy: u8, decode: bool, iters: Option<u32>| -> R<f64> {
                let codec = decode;
                let v0 = ring.last_value();
                let t0 = Instant::now();
                for b in 0..blobs {
                    let s = b % slots;
                    let v = v0 + 1 + b as u64;
                    let bytes = match copy {
                        0 => raw_len,
                        1 => img_len[s as usize],
                        _ => 256,
                    };
                    ring.submit_copy(s, bytes, v)?;
                    if codec || iters.is_some() {
                        ring.submit_compute(s, v, |d, cmd| {
                            if codec {
                                if nobar == 0 {
                                    unsafe { Gpu::barrier_all(d, cmd) };
                                }
                                let mut push = pushes[s as usize];
                                let mut lanes = f.lanes;
                                if extra.get() == 1 {
                                    push[21] |= 1; // mode bit 8
                                }
                                if extra.get() == 5 {
                                    // one workgroup, one word per lane: same pipeline, ~no work (output is garbage)
                                    lanes = 256;
                                    push[..8].copy_from_slice(&[0, 1, 0, 0, 1, 0, 0, 0]);
                                }
                                record_decode(d, cmd, &kd, dsets.sets[s as usize], &push, lanes);
                            }
                            if let Some(it) = iters {
                                if nobar == 2 {
                                    unsafe { Gpu::barrier_all(d, cmd) };
                                }
                                let set = if codec { os.sets[s as usize] } else { rs.sets[s as usize] };
                                record_dispatch(d, cmd, &kc, set, groups, (raw_len / 16) as u32, threads, it);
                                if extra.get() == 2 {
                                    unsafe { d.cmd_fill_buffer(cmd, outs[s as usize].buf, 0, raw_len, 0) };
                                }
                            }
                            Ok(())
                        })?;
                    }
                }
                let last = v0 + blobs as u64;
                if codec || iters.is_some() {
                    ring.wait_release(last)?;
                } else {
                    ring.wait(last)?;
                }
                Ok((raw_len * blobs as u64) as f64 / t0.elapsed().as_secs_f64() / 1e9)
            };

            // raw DMA rate L of these frames (warm-up run discarded)
            stage_all(&ring, false)?;
            run(&mut ring, &ds, 0, false, None)?;
            let l: Vec<f64> = (0..3).map(|_| run(&mut ring, &ds, 0, false, None)).collect::<R<_>>()?;
            r.raw_dma_gbps = mean_spread(&l).0;
            r.needed_decode_gbps = r.raw_dma_gbps * r.ratio;
            stage_all(&ring, true)?;
            run(&mut ring, &ds, 1, false, None)?;
            r.image_dma_raw_gbps = (0..3).map(|_| run(&mut ring, &ds, 1, false, None)).collect::<R<Vec<_>>>()?.iter().sum::<f64>() / 3.0;
            r.ring_decode_gbps = (0..3).map(|_| run(&mut ring, &ds, 2, true, None)).collect::<R<Vec<_>>>()?.iter().sum::<f64>() / 3.0;
            if std::env::var_os("BRECCIA_CODEC_DIAG").is_some() {
                ring.reset_stats();
                let g = run(&mut ring, &ds, 1, true, None)?;
                println!("    diag codec x0 {g:.3} GB/s stats {:?}", ring.stats());
                // a consumer as long as one decode, on image copies: does plain compute overlap the DMA?
                let target = (1u64 << 20) as f64 / (r.ring_decode_gbps * 1e9);
                let cal = calibrate_consumer(gpu, &kc, rs.sets[0], threads, groups, target)?;
                let it = Some(((cal.iters_per_mib as u64 * (raw_len >> 20).max(1)).min(4_000_000)) as u32);
                ring.reset_stats();
                let g = run(&mut ring, &ds, 1, false, it)?;
                println!("    diag image copy + decode-length consumer {g:.3} GB/s stats {:?}", ring.stats());
                // decode reading a resident copy of the image while the ring copies: contention or slot coupling?
                let res: Vec<Buf> = (0..slots).map(|_| dev_buf(gpu, max_img as u64 + 64)).collect::<R<_>>()?;
                for s in 0..slots as usize {
                    upload(gpu, &bufs[0], &res[s], &f.frames[s % n].0)?;
                }
                let rows: Vec<Vec<vk::Buffer>> =
                    (0..slots as usize).map(|s| vec![bufs[1].buf, res[s].buf, outs[s].buf]).collect();
                let rds = Sets::new(gpu, &kd, &rows)?;
                ring.reset_stats();
                let g = run(&mut ring, &rds, 1, true, None)?;
                println!("    diag image copy + decode of a resident image {g:.3} GB/s stats {:?}", ring.stats());
                let g = run(&mut ring, &rds, 2, true, None)?;
                println!("    diag resident decode, no copy {g:.3} GB/s");
                extra.set(1);
                let g = run(&mut ring, &rds, 1, true, None)?;
                println!("    diag image copy + resident decode WITHOUT output store {g:.3} GB/s");
                let g = run(&mut ring, &rds, 2, true, None)?;
                println!("    diag resident decode without store, no copy {g:.3} GB/s");
                extra.set(2);
                let g = run(&mut ring, &ds, 1, false, it)?;
                println!("    diag image copy + decode-length consumer + 16 MiB fill {g:.3} GB/s");
                let g = run(&mut ring, &ds, 2, false, it)?;
                println!("    diag consumer + 16 MiB fill, no copy {g:.3} GB/s");
                extra.set(0);
                // the x0.5 consumer on image copies, with and without the decode in front of it
                let target = (1u64 << 20) as f64 / (r.raw_dma_gbps * 1e9) * 0.5;
                let cal = calibrate_consumer(gpu, &kc, rs.sets[0], threads, groups, target)?;
                let it5 = Some(((cal.iters_per_mib as u64 * (raw_len >> 20).max(1)).min(4_000_000)) as u32);
                let g = run(&mut ring, &ds, 1, false, it5)?;
                println!("    diag image copy + x0.5 consumer (slot) {g:.3} GB/s");
                let g = run(&mut ring, &ds, 1, true, it5)?;
                println!("    diag image copy + decode + x0.5 consumer (outs) {g:.3} GB/s");
                let g = run(&mut ring, &ds, 2, false, it5)?;
                println!("    diag x0.5 consumer alone, no copy {g:.3} GB/s");
                extra.set(5);
                let g = run(&mut ring, &ds, 1, true, it5)?;
                println!("    diag image copy + EMPTY decode + x0.5 consumer (outs) {g:.3} GB/s");
                let g = run(&mut ring, &ds, 1, true, None)?;
                println!("    diag image copy + EMPTY decode {g:.3} GB/s");
                extra.set(0);
                rds.destroy(gpu);
                for b in &res {
                    gpu.destroy_buf(b);
                }
            }

            for &factor in &FACTORS {
                let iters = if factor > 0.0 {
                    let target = (1u64 << 20) as f64 / (r.raw_dma_gbps * 1e9) * factor;
                    let cal = calibrate_consumer(gpu, &kc, rs.sets[0], threads, groups, target)?;
                    Some(((cal.iters_per_mib as u64 * (raw_len >> 20).max(1)).min(4_000_000)) as u32)
                } else {
                    None
                };
                let (mut rg, mut cg) = (vec![], vec![]);
                for _ in 0..repeats {
                    stage_all(&ring, false)?;
                    rg.push(run(&mut ring, &ds, 0, false, iters)?);
                    stage_all(&ring, true)?;
                    cg.push(run(&mut ring, &ds, 1, true, iters)?);
                }
                let mut exact = true;
                for s in 0..slots as usize {
                    exact &= download_eq(gpu, &bufs[0], &outs[s], &f.frames[s % n].1)?;
                }
                let (rm, rsd) = mean_spread(&rg);
                let (cm, csd) = mean_spread(&cg);
                r.arms.push(ArmPoint {
                    consumer_factor: factor,
                    consumer_iters_per_mib: iters.map(|i| i / (raw_len >> 20).max(1) as u32).unwrap_or(0),
                    blobs_per_run: blobs,
                    raw_gbps: rg,
                    codec_gbps: cg,
                    raw_mean: rm,
                    codec_mean: cm,
                    raw_spread: rsd,
                    codec_spread: csd,
                    gain: cm / rm.max(1e-12),
                    codec_bit_exact: exact,
                });
            }
            Ok(())
        })();
        ring.destroy();
        for s in [ds, rs, os] {
            s.destroy(gpu);
        }
        for b in outs.iter().chain(std::iter::once(&acc)) {
            gpu.destroy_buf(b);
        }
        res
    })();
    cleanup(gpu, &bufs, &kd, &kc);
    bufs.clear();
    res
}

fn cleanup(gpu: &Gpu, bufs: &[Buf], kd: &Kernel, kc: &Kernel) {
    for b in bufs {
        gpu.destroy_buf(b);
    }
    gpu.destroy_kernel(kd);
    gpu.destroy_kernel(kc);
}
