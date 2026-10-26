//! Host side of the decode kernels: buffer upload, dispatch, timing.
//!
//! One [`Gemv`] is one compiled pipeline for a `{type, rows, cols}` shape. The
//! shader is `SPIR-V` embedded at build time and compiled once per
//! `breccia_vk::ctx::Gpu` (pipelines belong to a device). Buffers are passed to
//! [`Gemv::run`] so the caller can keep weights resident across calls — the
//! decode path runs this kernel thousands of times against the same weights.

use anyhow::{anyhow, Context, Result};
use ash::vk;
use breccia_vk::ctx::{Buf, Gpu, Qc};

use crate::quant::QuantType;

/// Timing of one measured kernel invocation.
#[derive(Clone, Copy, Debug, Default)]
pub struct GemvTiming {
    /// GPU seconds per GEMV (queue timestamps when available, else wall clock).
    pub per_iter_s: f64,
    /// Iterations inside the measured window.
    pub iters: u32,
    /// Total GPU seconds for the window.
    pub total_s: f64,
    /// True when the number came from `vkCmdWriteTimestamp` rather than the host clock.
    pub used_timestamps: bool,
}

impl GemvTiming {
    /// Packed-weight bytes read per second.
    pub fn gbps(&self, weight_bytes: u64) -> f64 {
        weight_bytes as f64 / self.per_iter_s / 1e9
    }

    /// KDE-scale check: `per_iter_s` is the budget one token's worth of this
    /// op would consume, so `1 / per_iter_s` is the "tokens/s if this were the
    /// whole token". Reported for scale, never as a decode claim.
    pub fn tokens_per_s_equivalent(&self) -> f64 {
        1.0 / self.per_iter_s
    }
}

/// A fused dequant + mat-vector multiply for one quant type and shape.
pub struct Gemv<'g> {
    ty: QuantType,
    rows: usize,
    cols: usize,
    gpu: &'g Gpu,
    kernel: breccia_vk::ctx::Kernel,
    /// The generated quant tables (7 KB), resident for the pipeline's lifetime.
    tables: Buf,
    /// Split-K reduction pipeline (`shaders/gemv_reduce.comp`).
    reduce: breccia_vk::ctx::Kernel,
}

impl<'g> Gemv<'g> {
    /// Compile the pipeline for `ty` on `gpu`.
    pub fn new(gpu: &'g Gpu, ty: QuantType, rows: usize, cols: usize) -> Result<Gemv<'g>> {
        if rows == 0 || cols == 0 {
            return Err(anyhow!("gemv shape must be non-empty ({rows}x{cols})"));
        }
        let spv = gemv_spv(ty);
        let kernel = gpu
            .kernel(spv, 4)
            .map_err(|e| anyhow!("creating the {ty} GEMV pipeline: {e}"))?;
        let tables = match upload(gpu, &table_bytes()) {
            Ok(b) => b,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                return Err(e.context("uploading the quant tables"));
            }
        };
        let reduce = match gpu.kernel(
            include_bytes!(concat!(env!("OUT_DIR"), "/gemv_reduce.spv")),
            2,
        ) {
            Ok(k) => k,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                gpu.destroy_buf(&tables);
                return Err(anyhow!("creating the split-K reduce pipeline: {e}"));
            }
        };
        Ok(Gemv {
            ty,
            rows,
            cols,
            gpu,
            kernel,
            tables,
            reduce,
        })
    }

    pub fn ty(&self) -> QuantType {
        self.ty
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Blocks per row (`ceil(cols / block_elems)`).
    pub fn blocks_per_row(&self) -> usize {
        self.cols.div_ceil(self.ty.block_elems())
    }

    /// Bytes of packed weights this shape needs (what the kernel reads once).
    pub fn weight_bytes(&self) -> u64 {
        (self.rows * self.blocks_per_row() * self.ty.block_bytes()) as u64
    }

    /// Elements-per-row padded to a whole block (the row stride in values).
    pub fn padded_cols(&self) -> usize {
        self.blocks_per_row() * self.ty.block_elems()
    }

    fn push_constants(&self, splits: u32) -> [u8; 16] {
        let mut pc = [0u8; 16];
        pc[0..4].copy_from_slice(&(self.rows as u32).to_le_bytes());
        pc[4..8].copy_from_slice(&(self.cols as u32).to_le_bytes());
        pc[8..12].copy_from_slice(&(self.blocks_per_row() as u32).to_le_bytes());
        pc[12..16].copy_from_slice(&splits.to_le_bytes());
        pc
    }

    /// Row-group workgroups split over `(x, y)` so `x` never exceeds the
    /// guaranteed `maxComputeWorkGroupCount[0]` of 65535 (see `gemv_body.glsl`).
    fn grid_for(&self, rows_per_wg: usize) -> (u32, u32) {
        let groups = self.rows.div_ceil(rows_per_wg).max(1) as u32;
        let x = groups.min(WG_X_MAX);
        let y = groups.div_ceil(WG_X_MAX);
        (x, y)
    }

    fn dispatch_split(&self, d: &ash::Device, cmd: vk::CommandBuffer, splits: u32) {
        let (gx, gy) = self.grid_for(ROWS_PER_WG);
        let push = self.push_constants(splits);
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.kernel.pipe);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &[self.kernel.set],
                &[],
            );
            d.cmd_push_constants(
                cmd,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            d.cmd_dispatch(cmd, gx, gy, splits.max(1));
        }
    }

    fn dispatch_reduce(&self, d: &ash::Device, cmd: vk::CommandBuffer, splits: u32) {
        let (gx, gy) = self.grid_for(256);
        let push = self.push_constants(splits);
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.reduce.pipe);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.reduce.layout,
                0,
                &[self.reduce.set],
                &[],
            );
            d.cmd_push_constants(
                cmd,
                self.reduce.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            d.cmd_dispatch(cmd, gx, gy, 1);
        }
    }

    /// Split factor for this shape (see [`Gemv::recommended_splits_for`]).
    pub fn recommended_splits(&self) -> u32 {
        self.recommended_splits_for(60)
    }

    /// Split heuristic: split only when row-parallelism alone leaves the device
    /// under-filled **and** the kernel is not bandwidth-bound.
    ///
    /// Measured on the RX 6950 XT at 640x2560 (`docs/phase-3/01-dequant-gemv.md`
    /// §4.2): 2–4 splits buy **+7 % (IQ2_XXS) to +27 % (Q5_K)** on the quantised
    /// types, because those cells are ALU/latency-bound; the same split costs
    /// **−15 % to −29 %** on F32, which already saturates the memory path at
    /// these shapes, since the reduce dispatch plus two full barriers is 2–3 µs
    /// — as much as the GEMV itself. So F32 never splits, and neither does any
    /// shape that already has ≥ 4 workgroups per CU.
    pub fn recommended_splits_for(&self, cus: u32) -> u32 {
        // F32 is the one type that already saturates the cache/DRAM path at these
        // shapes (1161 GB/s at 640x2560), so splitting it only adds the reduce cost.
        if self.ty == QuantType::F32 {
            return 1;
        }
        let wgs = self.rows.div_ceil(ROWS_PER_WG) as u32;
        let target = (cus * 4).max(1);
        if wgs >= target {
            return 1;
        }
        target
            .div_ceil(wgs.max(1))
            .clamp(1, 4)
            .min((self.blocks_per_row() as u32).max(1))
    }

    /// Partial-sum buffer size for `splits` (one f32 per row per split).
    pub fn partial_bytes(&self, splits: u32) -> u64 {
        (self.rows as u64) * u64::from(splits.max(1)) * 4
    }

    /// Bind the three buffers this shape needs.
    pub fn bind(&self, w: &Buf, x: &Buf, out: &Buf) -> Result<()> {
        self.gpu
            .bind_kernel(&self.kernel, &[w.buf, x.buf, out.buf, self.tables.buf])
            .map_err(|e| anyhow!("{e}"))
    }

    /// The table buffer this pipeline binds (for callers that bind manually).
    pub fn tables(&self) -> &Buf {
        &self.tables
    }

    /// Run one GEMV: `out[rows] = W[rows, cols] · x[cols]`, synchronously.
    ///
    /// `w` holds the packed blocks (`weight_bytes()` bytes), `x` holds
    /// `padded_cols()` fp32 values (zero-padded past `cols`), `out` receives
    /// `rows` fp32 values.
    pub fn run(&self, w: &Buf, x: &Buf, out: &Buf) -> Result<()> {
        self.bind(w, x, out)?;
        self.gpu
            .wall_run(Qc::C, |d, cmd| self.dispatch_split(d, cmd, 1))
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    /// Split-K run: `partials` holds `rows * splits` fp32 partial products,
    /// `out` receives `rows`. Bind the partials buffer to binding 2 first (use
    /// [`Gemv::bind_partials`]), the reduce pipeline to bindings 0/1.
    pub fn run_split(
        &self,
        w: &Buf,
        x: &Buf,
        partials: &Buf,
        out: &Buf,
        splits: u32,
    ) -> Result<()> {
        let splits = splits.max(1);
        self.bind(w, x, partials)?;
        self.gpu
            .bind_kernel(&self.reduce, &[partials.buf, out.buf])
            .map_err(|e| anyhow!("{e}"))?;
        self.gpu
            .wall_run(Qc::C, |d, cmd| {
                self.dispatch_split(d, cmd, splits);
                unsafe { Gpu::barrier_all(d, cmd) };
                self.dispatch_reduce(d, cmd, splits);
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    /// Same as [`Gemv::run_split`] but timed.
    pub fn run_split_timed(
        &self,
        w: &Buf,
        x: &Buf,
        partials: &Buf,
        out: &Buf,
        splits: u32,
        target_s: f64,
    ) -> Result<GemvTiming> {
        let splits = splits.max(1);
        self.bind(w, x, partials)?;
        self.gpu
            .bind_kernel(&self.reduce, &[partials.buf, out.buf])
            .map_err(|e| anyhow!("{e}"))?;
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                for _ in 0..n {
                    self.dispatch_split(d, cmd, splits);
                    unsafe { Gpu::barrier_all(d, cmd) };
                    self.dispatch_reduce(d, cmd, splits);
                    unsafe { Gpu::barrier_all(d, cmd) };
                }
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(GemvTiming {
            per_iter_s: t.per_iter_s,
            iters: t.iters,
            total_s: t.total_s,
            used_timestamps: t.used_timestamps,
        })
    }

    /// Bind the partial buffer for a split run (alias of `bind(w, x, partials)`).
    pub fn bind_partials(&self, w: &Buf, x: &Buf, partials: &Buf) -> Result<()> {
        self.bind(w, x, partials)
    }

    /// Time `reps` consecutive single-pass dispatches per iteration — the
    /// per-expert loop a MoE layer issues when it does *not* batch
    /// ([`crate::GemvBatch`]). `per_iter_s` is the whole group's latency.
    pub fn run_looped_timed(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        reps: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        self.bind(w, x, out)?;
        let reps = reps.max(1) as u32;
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                for _ in 0..n {
                    for _ in 0..reps {
                        self.dispatch_split(d, cmd, 1);
                        unsafe { Gpu::barrier_all(d, cmd) };
                    }
                }
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(GemvTiming {
            per_iter_s: t.per_iter_s,
            iters: t.iters,
            total_s: t.total_s,
            used_timestamps: t.used_timestamps,
        })
    }

    /// Split-K variant of [`Gemv::run_looped_timed`] (loop baseline with the
    /// per-expert split factor the engine would otherwise use).
    #[allow(clippy::too_many_arguments)] // buffers + splits + reps + window
    pub fn run_split_looped_timed(
        &self,
        w: &Buf,
        x: &Buf,
        partials: &Buf,
        out: &Buf,
        splits: u32,
        reps: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        let splits = splits.max(1);
        self.bind(w, x, partials)?;
        self.gpu
            .bind_kernel(&self.reduce, &[partials.buf, out.buf])
            .map_err(|e| anyhow!("{e}"))?;
        let reps = reps.max(1) as u32;
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                for _ in 0..n {
                    for _ in 0..reps {
                        self.dispatch_split(d, cmd, splits);
                        unsafe { Gpu::barrier_all(d, cmd) };
                        self.dispatch_reduce(d, cmd, splits);
                        unsafe { Gpu::barrier_all(d, cmd) };
                    }
                }
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(GemvTiming {
            per_iter_s: t.per_iter_s,
            iters: t.iters,
            total_s: t.total_s,
            used_timestamps: t.used_timestamps,
        })
    }

    /// Time `target_s` worth of repeated GEMVs (timestamps bracket the window).
    pub fn run_timed(&self, w: &Buf, x: &Buf, out: &Buf, target_s: f64) -> Result<GemvTiming> {
        self.bind(w, x, out)?;
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                // `run_timed` hands us the repeat count for the window; a full
                // barrier between dispatches keeps them serialised so the
                // per-iteration number is a latency, not a pipelined average.
                for _ in 0..n {
                    self.dispatch_split(d, cmd, 1);
                    unsafe { Gpu::barrier_all(d, cmd) };
                }
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(GemvTiming {
            per_iter_s: t.per_iter_s,
            iters: t.iters,
            total_s: t.total_s,
            used_timestamps: t.used_timestamps,
        })
    }
}

impl Drop for Gemv<'_> {
    fn drop(&mut self) {
        self.gpu.destroy_kernel(&self.reduce);
        self.gpu.destroy_kernel(&self.kernel);
        self.gpu.destroy_buf(&self.tables);
    }
}

/// The quant tables as little-endian u32 bytes, ready for `binding = 3`.
pub fn table_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(crate::tables::GPU_TABLE_WORDS.len() * 4);
    for w in crate::tables::GPU_TABLE_WORDS.iter() {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Rows per workgroup, matching `ROWS_PER_WG` in `shaders/gemv_body.glsl`
/// (8 groups of 32 lanes in a 256-thread workgroup).
pub const ROWS_PER_WG: usize = 8;

/// `maxComputeWorkGroupCount[0]`, the value every Vulkan device guarantees.
/// Larger dispatches are wrapped over `grid.y` (and `grid.z` for split-K).
pub const WG_X_MAX: u32 = 65535;

// ---------------------------------------------------------------------------
// Standalone dequantiser (conformance tests + callers that need f32 weights)
// ---------------------------------------------------------------------------

/// Dequantise `rows` packed rows of `cols` values into fp32.
///
/// The result is `rows * padded_cols()` values (row-major, each row padded to a
/// whole block); callers that want `rows * cols` should pass whole-block shapes.
pub fn dequant_to_f32(
    gpu: &Gpu,
    ty: QuantType,
    rows: usize,
    cols: usize,
    packed: &[u8],
) -> Result<Vec<f32>> {
    if rows == 0 || cols == 0 {
        return Err(anyhow!("dequant shape must be non-empty ({rows}x{cols})"));
    }
    let qk = ty.block_elems();
    let bpr = cols.div_ceil(qk);
    let padded = bpr * qk;
    let expect = rows * bpr * ty.block_bytes();
    if packed.len() != expect {
        return Err(anyhow!(
            "packed length {} != rows*bpr*block_bytes = {} for {ty} {rows}x{cols}",
            packed.len(),
            expect
        ));
    }

    let kernel = gpu
        .kernel(dequant_spv(ty), 4)
        .map_err(|e| anyhow!("creating the {ty} dequant pipeline: {e}"))?;
    let w = upload(gpu, packed).context("uploading packed weights")?;
    let tables = upload(gpu, &table_bytes()).context("uploading the quant tables")?;
    // The dequant shader does not statically use binding 1 (it is shared with the
    // fused GEMV header); a tiny dummy keeps every declared binding bound.
    let x = upload(gpu, &[0u8; 16]).context("allocating the unused x binding")?;
    let out = alloc(gpu, (padded * rows * 4) as u64).context("allocating the dequant output")?;

    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
    pc[4..8].copy_from_slice(&(cols as u32).to_le_bytes());
    pc[8..12].copy_from_slice(&(bpr as u32).to_le_bytes());

    let result = (|| -> Result<Vec<f32>> {
        gpu.bind_kernel(&kernel, &[w.buf, x.buf, out.buf, tables.buf])
            .map_err(|e| anyhow!("{e}"))?;
        let blocks = (rows * bpr) as u32;
        let groups = blocks.div_ceil(64);
        let gx = groups.min(WG_X_MAX);
        let gy = groups.div_ceil(WG_X_MAX);
        gpu.wall_run(Qc::C, |d, cmd| unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipe);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.layout,
                0,
                &[kernel.set],
                &[],
            );
            d.cmd_push_constants(cmd, kernel.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
            d.cmd_dispatch(cmd, gx, gy, 1);
        })
        .map_err(|e| anyhow!("{e}"))?;

        let bytes = download(gpu, &out, padded * rows * 4).context("reading the dequant output")?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    })();

    gpu.destroy_kernel(&kernel);
    for b in [&w, &x, &out, &tables] {
        gpu.destroy_buf(b);
    }
    result
}

// ---------------------------------------------------------------------------
// Buffer helpers (device-local + staging; never raw host-visible weights)
// ---------------------------------------------------------------------------

/// Bytes added to every device-local allocation.
///
/// A GPU read one element past the end of a storage buffer is not "garbage", it
/// is a UTCL2 permission fault that kills the process and, on a display card,
/// can take the machine down (2026-10-26: a conformance run did exactly that —
/// `docs/phase-3/02-expert-batching-quants.md` §4). Decode kernels read blocks
/// with `u16`/`u32`/`vec4` loads whose last access can be wider than the field
/// being consumed on some compilations, and every one of these buffers ends
/// exactly at the last byte the kernel uses. 64 B of slack costs nothing at
/// these sizes and makes a tail read land in mapped memory.
pub const TAIL_PAD: u64 = 64;

/// Allocate a device-local storage buffer (always `size + TAIL_PAD` bytes long;
/// callers may only rely on the first `size`, and the kernels only touch those).
pub fn alloc(gpu: &Gpu, size: u64) -> Result<Buf> {
    let size = size + TAIL_PAD;
    gpu.buf(
        size.max(4),
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .map_err(|e| anyhow!("{e}"))
}

fn staging(gpu: &Gpu, size: u64) -> Result<Buf> {
    gpu.buf(
        size.max(4),
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .map_err(|e| anyhow!("{e}"))
}

/// Upload `data` into a fresh device-local buffer (staging + one DMA copy).
pub fn upload(gpu: &Gpu, data: &[u8]) -> Result<Buf> {
    let dst = alloc(gpu, data.len() as u64)?;
    if let Err(e) = upload_into(gpu, &dst, data) {
        gpu.destroy_buf(&dst);
        return Err(e);
    }
    Ok(dst)
}

/// Upload into an existing device-local buffer.
pub fn upload_into(gpu: &Gpu, dst: &Buf, data: &[u8]) -> Result<()> {
    let stage = staging(gpu, data.len() as u64)?;
    let r = (|| -> Result<()> {
        gpu.write_host(&stage, data).map_err(|e| anyhow!("{e}"))?;
        gpu.copy_once(&stage, dst, data.len() as u64)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    })();
    gpu.destroy_buf(&stage);
    r
}

/// Read `n` bytes back from a device-local buffer.
pub fn download(gpu: &Gpu, src: &Buf, n: usize) -> Result<Vec<u8>> {
    let stage = staging(gpu, n as u64)?;
    let r = (|| -> Result<Vec<u8>> {
        gpu.copy_once(src, &stage, n as u64)
            .map_err(|e| anyhow!("{e}"))?;
        let mut out = vec![0u8; n];
        gpu.read_host(&stage, &mut out)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(out)
    })();
    gpu.destroy_buf(&stage);
    r
}

// ---------------------------------------------------------------------------
// Embedded SPIR-V
// ---------------------------------------------------------------------------

macro_rules! spv {
    ($stem:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $stem, ".spv"))
    };
}

fn gemv_spv(ty: QuantType) -> &'static [u8] {
    match ty {
        QuantType::Iq2Xxs => spv!("gemv_iq2_xxs"),
        QuantType::Iq2Xs => spv!("gemv_iq2_xs"),
        QuantType::Iq2S => spv!("gemv_iq2_s"),
        QuantType::Iq3Xxs => spv!("gemv_iq3_xxs"),
        QuantType::Iq3S => spv!("gemv_iq3_s"),
        QuantType::Iq4Nl => spv!("gemv_iq4_nl"),
        QuantType::Iq4Xs => spv!("gemv_iq4_xs"),
        QuantType::Q4K => spv!("gemv_q4_k"),
        QuantType::Q5K => spv!("gemv_q5_k"),
        QuantType::Q6K => spv!("gemv_q6_k"),
        QuantType::Q2_0 => spv!("gemv_q2_0"),
        QuantType::Bf16 => spv!("gemv_bf16"),
        QuantType::F32 => spv!("gemv_f32"),
    }
}

fn dequant_spv(ty: QuantType) -> &'static [u8] {
    match ty {
        QuantType::Iq2Xxs => spv!("dequant_iq2_xxs"),
        QuantType::Iq2Xs => spv!("dequant_iq2_xs"),
        QuantType::Iq2S => spv!("dequant_iq2_s"),
        QuantType::Iq3Xxs => spv!("dequant_iq3_xxs"),
        QuantType::Iq3S => spv!("dequant_iq3_s"),
        QuantType::Iq4Nl => spv!("dequant_iq4_nl"),
        QuantType::Iq4Xs => spv!("dequant_iq4_xs"),
        QuantType::Q4K => spv!("dequant_q4_k"),
        QuantType::Q5K => spv!("dequant_q5_k"),
        QuantType::Q6K => spv!("dequant_q6_k"),
        QuantType::Q2_0 => spv!("dequant_q2_0"),
        QuantType::Bf16 => spv!("dequant_bf16"),
        QuantType::F32 => spv!("dequant_f32"),
    }
}
