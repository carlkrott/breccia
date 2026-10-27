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

/// Accumulator precision of the fused dot product.
///
/// `F32` is the default and the only setting the conformance gate accepts
/// (≤ 1e-3 relative). `F16` accumulates the per-lane running sum of a row in
/// fp16 and converts back to f32 before the cross-lane reduce; it exists to
/// measure what the lower precision costs and buys
/// (`docs/phase-3/03-gaps-closed.md` §2) and is **not recommended**: the
/// measured error is two orders of magnitude above the default bar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Accum {
    /// fp32 dot product, two alternating accumulators (the default, unchanged).
    #[default]
    F32,
    /// fp16 per-lane accumulation (opt-in research path, `gemv_*_f16acc.spv`).
    F16,
}

impl Accum {
    /// Every accumulator mode the kernels implement.
    pub const ALL: [Accum; 2] = [Accum::F32, Accum::F16];

    pub const fn name(self) -> &'static str {
        match self {
            Accum::F32 => "f32",
            Accum::F16 => "f16",
        }
    }

    /// Parse `f32` / `f16` (case-insensitive).
    pub fn from_name(s: &str) -> Option<Accum> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Some(Accum::F32),
            "f16" | "fp16" => Some(Accum::F16),
            _ => None,
        }
    }
}

impl std::fmt::Display for Accum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Accum {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Accum::from_name(s).ok_or_else(|| {
            format!(
                "unknown accumulator {s:?}; expected one of {}",
                Accum::ALL
                    .iter()
                    .map(|a| a.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }
}

/// `BRECCIA_GEMV_DP4A=1`: run IQ3_S GEMVs on **packed int8 dot products** (§8bb) —
/// `q8_1` activations from a small quantiser, int8 codes from the IQ3_S grid, and
/// `dotPacked4x8EXT` (= SPIR-V `OpSDot`, `PackedVectorFormat4x8Bit`) doing 4 MACs per
/// slot with the dequant's `u2f`/FMA/scale work gone.
///
/// Off by default: the fp32 path stays the reference (bit-reproducible) and this one
/// is a rounding change (int32 accumulation, one scale per 32-element group).
pub fn dp4a_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BRECCIA_GEMV_DP4A").is_some())
}

/// The int8-dot state one [`Gemv`] owns when the dp4a path is on: the two pipelines
/// and the activation's `q8_1` buffers, sized from the shape (never a literal).
struct Q8 {
    quant: breccia_vk::ctx::Kernel,
    gemv: breccia_vk::ctx::Kernel,
    /// `cols` int8 quants, read as packed 4-byte words by the dot.
    q8: Buf,
    /// One f16 scale per 32-element group.
    dq: Buf,
    /// `cols / 32`, rounded up.
    groups: usize,
}

/// Build the int8-dot state for a `cols`-wide row, sizing every buffer from `cols`.
fn build_q8(gpu: &Gpu, cols: usize) -> Result<Q8> {
    if !cols.is_multiple_of(32) {
        return Err(anyhow!(
            "the int8-dot GEMV needs cols % 32 == 0 (the q8_1 group), got {cols}"
        ));
    }
    let groups = cols / 32;
    let quant = gpu
        .kernel(
            include_bytes!(concat!(env!("OUT_DIR"), "/ops_q8_quantize.spv")),
            3,
        )
        .map_err(|e| anyhow!("creating the q8_1 quantiser: {e}"))?;
    let gemv = match gpu.kernel(
        include_bytes!(concat!(env!("OUT_DIR"), "/gemv_iq3s_q8.spv")),
        6,
    ) {
        Ok(k) => k,
        Err(e) => {
            gpu.destroy_kernel(&quant);
            return Err(anyhow!("creating the IQ3_S dp4a GEMV pipeline: {e}"));
        }
    };
    let q8 = match alloc(gpu, cols as u64) {
        Ok(b) => b,
        Err(e) => {
            gpu.destroy_kernel(&quant);
            gpu.destroy_kernel(&gemv);
            return Err(e.context("allocating the activation quants"));
        }
    };
    let dq = match alloc(gpu, (groups * 2) as u64) {
        Ok(b) => b,
        Err(e) => {
            gpu.destroy_kernel(&quant);
            gpu.destroy_kernel(&gemv);
            gpu.destroy_buf(&q8);
            return Err(e.context("allocating the activation scales"));
        }
    };
    Ok(Q8 {
        quant,
        gemv,
        q8,
        dq,
        groups,
    })
}

/// The q8_1 quantiser's push constants: `k`, `groups`, then two unused words.
fn q8_quant_pc(cols: usize, groups: usize) -> [u8; 32] {
    let mut pc = [0u8; 32];
    for (i, v) in [cols as u32, groups as u32, 0u32, 0u32].iter().enumerate() {
        pc[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc
}

/// A fused dequant + mat-vector multiply for one quant type and shape.
pub struct Gemv<'g> {
    ty: QuantType,
    accum: Accum,
    rows: usize,
    cols: usize,
    gpu: &'g Gpu,
    kernel: breccia_vk::ctx::Kernel,
    /// The generated quant tables (7 KB), resident for the pipeline's lifetime.
    tables: Buf,
    /// Split-K reduction pipeline (`shaders/gemv_reduce.comp`).
    reduce: breccia_vk::ctx::Kernel,
    /// `Some(s)` when the `silu_div` epilogue is fused (see
    /// [`Gemv::with_epilogue`]): `s` is the divisor.
    epilogue: Option<f32>,
    /// The `_hcact` build of [`Gemv::kernel`] (used when `splits == 1`).
    ep_kernel: Option<breccia_vk::ctx::Kernel>,
    /// The `_hcact` build of [`Gemv::reduce`] (used when `splits > 1`: the
    /// partials have to stay raw, so the activation moves into the sum).
    ep_reduce: Option<breccia_vk::ctx::Kernel>,
    /// `Some` when this shape runs on the packed int8-dot path (§8bb).
    q8: Option<Q8>,
}

impl<'g> Gemv<'g> {
    /// Compile the pipeline for `ty` on `gpu` with the default fp32 accumulator.
    pub fn new(gpu: &'g Gpu, ty: QuantType, rows: usize, cols: usize) -> Result<Gemv<'g>> {
        Self::with_accum(gpu, ty, rows, cols, Accum::F32)
    }

    /// Compile the pipeline for `ty` on `gpu` with an explicit accumulator.
    pub fn with_accum(
        gpu: &'g Gpu,
        ty: QuantType,
        rows: usize,
        cols: usize,
        accum: Accum,
    ) -> Result<Gemv<'g>> {
        Self::with_variant(gpu, ty, rows, cols, accum, wide_dequant())
    }

    /// Compile the **word-level pack** build (§8bc) explicitly, for the A/B and the
    /// bit-identity test. Falls back to the default build for types without one.
    pub fn with_words(
        gpu: &'g Gpu,
        ty: QuantType,
        rows: usize,
        cols: usize,
        accum: Accum,
    ) -> Result<Gemv<'g>> {
        let spv = gemv_words_spv(ty).unwrap_or_else(|| gemv_spv_variant(ty, accum, false));
        Self::glsl_variant(gpu, ty, rows, cols, accum, spv, false)
    }

    /// Compile an explicit **dequant variant**: `wide` selects the coalesced
    /// (`_wide`) build when this type has one, `false` the incumbent per-element
    /// one. This is the A/B constructor — [`Gemv::with_accum`] passes
    /// [`wide_dequant`] (the `BRECCIA_GEMV_WIDE` flag) — and it is what the
    /// conformance suite uses to prove the two builds are **bit-identical**
    /// (`wide_dequant_is_bit_identical_to_the_default`).
    pub fn with_variant(
        gpu: &'g Gpu,
        ty: QuantType,
        rows: usize,
        cols: usize,
        accum: Accum,
        wide: bool,
    ) -> Result<Gemv<'g>> {
        if rows == 0 || cols == 0 {
            return Err(anyhow!("gemv shape must be non-empty ({rows}x{cols})"));
        }
        if accum == Accum::F16 && !gpu.feats.shader_float16 {
            return Err(anyhow!(
                "device {} does not support shaderFloat16, which fp16 accumulation requires",
                gpu.index
            ));
        }
        let spv = gemv_spv_variant(ty, accum, wide);
        Self::glsl_variant(gpu, ty, rows, cols, accum, spv, wide)
    }

    /// The shared body of the variant constructors: `spv` is the pipeline's module,
    /// `dp4a` says whether the int8-dot state should be built.
    fn glsl_variant(
        gpu: &'g Gpu,
        ty: QuantType,
        rows: usize,
        cols: usize,
        accum: Accum,
        spv: &'static [u8],
        _wide: bool,
    ) -> Result<Gemv<'g>> {
        let kernel = gpu
            .kernel(spv, 4)
            .map_err(|e| anyhow!("creating the {ty} GEMV pipeline ({accum} accum): {e}"))?;
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
        // The int8-dot path (§8bb): only IQ3_S has a dp4a kernel, and only the fp32
        // accumulator shape makes sense (int8 accumulation is its own thing).
        let q8 = if ty == QuantType::Iq3S && accum == Accum::F32 && dp4a_enabled() {
            match build_q8(gpu, cols) {
                Ok(q) => Some(q),
                Err(e) => {
                    gpu.destroy_kernel(&kernel);
                    gpu.destroy_kernel(&reduce);
                    gpu.destroy_buf(&tables);
                    return Err(e);
                }
            }
        } else {
            None
        };
        Ok(Gemv {
            ty,
            accum,
            rows,
            cols,
            gpu,
            kernel,
            tables,
            reduce,
            epilogue: None,
            ep_kernel: None,
            ep_reduce: None,
            q8,
        })
    }

    /// Fuse the HC bottleneck activation (`ActMode::SiluDiv`, `docs/specs/
    /// qwen4exp-architecture.md` §4.1) into this projection's epilogue:
    /// `out = silu(acc / s)`.
    ///
    /// `s` is the divisor the separate [`crate::ops::Act`] dispatch would apply,
    /// so the fused result is **bit-identical by construction**: the epilogue runs
    /// on the fully accumulated row value, in the same order, with the same
    /// `breccia_silu`. With `splits > 1` the partial products must stay raw, so the
    /// activation rides in the reduce pass instead of the main kernel — either way
    /// the value in `out` is what `Act` would have written.
    ///
    /// `Err` when this `{type, accumulator}` has no `_hcact` build (see
    /// [`has_hc_epilogue`]).
    pub fn with_epilogue(mut self, s: f32) -> Result<Self> {
        if !s.is_finite() || s == 0.0 {
            return Err(anyhow!("the epilogue divisor must be finite and non-zero"));
        }
        let spv = gemv_hcact_spv(self.ty, self.accum).ok_or_else(|| {
            anyhow!(
                "no `_hcact` GEMV build for {} ({} accum); the epilogue fusion \
                 is compiled for BF16 and F32 only",
                self.ty,
                self.accum
            )
        })?;
        let ep = self
            .gpu
            .kernel(spv, 4)
            .map_err(|e| anyhow!("creating the {} GEMV epilogue pipeline: {e}", self.ty))?;
        let ep_reduce = match self.gpu.kernel(
            include_bytes!(concat!(env!("OUT_DIR"), "/gemv_reduce_hcact.spv")),
            2,
        ) {
            Ok(k) => k,
            Err(e) => {
                self.gpu.destroy_kernel(&ep);
                return Err(anyhow!("creating the epilogue reduce pipeline: {e}"));
            }
        };
        self.ep_kernel = Some(ep);
        self.ep_reduce = Some(ep_reduce);
        self.epilogue = Some(s);
        Ok(self)
    }

    /// The divisor of the fused `silu_div` epilogue (`None` = separate `Act`).
    pub fn epilogue(&self) -> Option<f32> {
        self.epilogue
    }

    pub fn ty(&self) -> QuantType {
        self.ty
    }
    /// Accumulator precision this pipeline was compiled for.
    pub fn accum(&self) -> Accum {
        self.accum
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

    /// The push constants a pipeline reads: the 16-byte prefix every GEMV build
    /// declares, extended to 24 bytes (word 5 `ep`) for the `_hcact` builds.
    /// Returns the length to push.
    fn fill_pc(&self, out: &mut [u8; 24], splits: u32, ep: Option<f32>) -> usize {
        out[..16].copy_from_slice(&self.push_constants(splits));
        match ep {
            None => 16,
            Some(s) => {
                out[20..24].copy_from_slice(&s.to_le_bytes());
                24
            }
        }
    }

    /// The pipeline that writes `O`, and the epilogue it applies: the main
    /// kernel with the activation wired in when there is no split-K, otherwise
    /// the plain one (its output is raw partials).
    fn out_kernel(&self, splits: u32) -> (&breccia_vk::ctx::Kernel, Option<f32>) {
        match (&self.ep_kernel, self.epilogue) {
            (Some(k), Some(s)) if splits.max(1) == 1 => (k, Some(s)),
            _ => (&self.kernel, None),
        }
    }

    /// The pipeline that sums the split-K partials into `O`, and the epilogue it
    /// applies (`Some` for the `_hcact` reduce build).
    fn sum_kernel(&self) -> (&breccia_vk::ctx::Kernel, Option<f32>) {
        match (&self.ep_reduce, self.epilogue) {
            (Some(k), Some(s)) => (k, Some(s)),
            _ => (&self.reduce, None),
        }
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
        let (k, ep) = self.out_kernel(splits);
        let mut pc = [0u8; 24];
        let n = self.fill_pc(&mut pc, splits, ep);
        let (gx, gy) = self.grid_for(ROWS_PER_WG);
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                k.layout,
                0,
                &[k.set],
                &[],
            );
            d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc[..n]);
            d.cmd_dispatch(cmd, gx, gy, splits.max(1));
        }
    }

    fn dispatch_reduce(&self, d: &ash::Device, cmd: vk::CommandBuffer, splits: u32) {
        let (k, ep) = self.sum_kernel();
        let mut pc = [0u8; 24];
        let n = self.fill_pc(&mut pc, splits, ep);
        let (gx, gy) = self.grid_for(256);
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                k.layout,
                0,
                &[k.set],
                &[],
            );
            d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc[..n]);
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
            .map_err(|e| anyhow!("{e}"))?;
        // The epilogue build is a pipeline of its own, hence a descriptor set of
        // its own; binding it here keeps `run` a single wall_run.
        if let Some(k) = &self.ep_kernel {
            self.gpu
                .bind_kernel(k, &[w.buf, x.buf, out.buf, self.tables.buf])
                .map_err(|e| anyhow!("{e}"))?;
        }
        Ok(())
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
        if let Some(q8) = &self.q8 {
            if crate::dispatch_batch::is_open() {
                self.record_q8(q8, w, x, out)?;
            } else {
                // One command buffer, two dispatches: the q8_1 quantiser then the
                // dp4a GEMV. The quantiser is one workgroup per 8 groups.
                let quant_pc = q8_quant_pc(self.cols, q8.groups);
                let gemv_pc = self.q8_gemv_pc();
                let qgrid = (q8.groups.div_ceil(8).max(1) as u32, 1, 1);
                self.gpu
                    .bind_kernel(&q8.quant, &[x.buf, q8.q8.buf, q8.dq.buf])
                    .map_err(|e| anyhow!("{e}"))?;
                self.gpu
                    .bind_kernel(
                        &q8.gemv,
                        &[w.buf, x.buf, out.buf, self.tables.buf, q8.q8.buf, q8.dq.buf],
                    )
                    .map_err(|e| anyhow!("{e}"))?;
                let (gx, gy) = self.grid_for(ROWS_PER_WG);
                self.gpu
                    .wall_run(Qc::C, |d, cmd| {
                        crate::ops::record(d, cmd, &q8.quant, &quant_pc, qgrid);
                        crate::ops::record(d, cmd, &q8.gemv, &gemv_pc, (gx, gy, 1));
                    })
                    .map_err(|e| anyhow!("{e}"))?;
            }
            return Ok(());
        }
        if crate::dispatch_batch::is_open() {
            return self.record_split(self.gpu, 1, &[w, x, out, &self.tables], None);
        }
        self.bind(w, x, out)?;
        self.gpu
            .wall_run(Qc::C, |d, cmd| self.dispatch_split(d, cmd, 1))
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    /// The IQ3_S dp4a kernel's push constants (`blocks.glsl`'s five words).
    fn q8_gemv_pc(&self) -> [u8; 32] {
        let mut pc = [0u8; 32];
        for (i, v) in [
            self.rows as u32,
            self.cols as u32,
            self.blocks_per_row() as u32,
            1,
            0,
        ]
        .iter()
        .enumerate()
        {
            pc[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        pc
    }

    /// Record the q8_1 producer and the dp4a GEMV into the open batch (§8bb): two
    /// dispatches, the first a one-workgroup-per-8-groups quantiser.
    fn record_q8(&self, q8: &Q8, w: &Buf, x: &Buf, out: &Buf) -> Result<()> {
        let quant_pc = q8_quant_pc(self.cols, q8.groups);
        crate::dispatch_batch::record(
            self.gpu,
            &q8.quant,
            &[x, &q8.q8, &q8.dq],
            &quant_pc,
            (q8.groups.div_ceil(8).max(1) as u32, 1, 1),
        )?;
        crate::dispatch_batch::record(
            self.gpu,
            &q8.gemv,
            &[w, x, out, &self.tables, &q8.q8, &q8.dq],
            &self.q8_gemv_pc(),
            {
                let (gx, gy) = self.grid_for(ROWS_PER_WG);
                (gx, gy, 1)
            },
        )
    }

    /// Record the split dispatch (+ reduce when `splits > 1`) into the batch that
    /// is open on the compute queue. The batch allocates its own descriptor sets,
    /// so nothing needs binding first.
    fn record_split(
        &self,
        gpu: &Gpu,
        splits: u32,
        bufs: &[&Buf],
        reduce_bufs: Option<&[&Buf]>,
    ) -> Result<()> {
        let (k, ep) = self.out_kernel(splits);
        let mut pc = [0u8; 24];
        let n = self.fill_pc(&mut pc, splits, ep);
        let (gx, gy) = self.grid_for(ROWS_PER_WG);
        crate::dispatch_batch::record(gpu, k, bufs, &pc[..n], (gx, gy, splits.max(1)))?;
        if splits > 1 {
            let (rk, rep) = self.sum_kernel();
            let mut rpc = [0u8; 24];
            let rn = self.fill_pc(&mut rpc, splits, rep);
            let (rx, ry) = self.grid_for(256);
            if let Some(rb) = reduce_bufs {
                crate::dispatch_batch::record(gpu, rk, rb, &rpc[..rn], (rx, ry, 1))?;
            }
        }
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
        if crate::dispatch_batch::is_open() {
            return self.record_split(
                self.gpu,
                splits,
                &[w, x, partials, &self.tables],
                Some(&[partials, out]),
            );
        }
        self.bind(w, x, partials)?;
        self.gpu
            .bind_kernel(self.sum_kernel().0, &[partials.buf, out.buf])
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
            .bind_kernel(self.sum_kernel().0, &[partials.buf, out.buf])
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
            .bind_kernel(self.sum_kernel().0, &[partials.buf, out.buf])
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
        // The int8-dot path (§8bb) is two dispatches (q8_1 quantiser + dp4a GEMV), so
        // the timing window must record both or it would price the dot alone.
        let q8 = self.q8.as_ref();
        if let Some(q) = q8 {
            self.gpu
                .bind_kernel(&q.quant, &[x.buf, q.q8.buf, q.dq.buf])
                .map_err(|e| anyhow!("{e}"))?;
            self.gpu
                .bind_kernel(
                    &q.gemv,
                    &[w.buf, x.buf, out.buf, self.tables.buf, q.q8.buf, q.dq.buf],
                )
                .map_err(|e| anyhow!("{e}"))?;
        }
        let quant_pc = q8.map(|q| q8_quant_pc(self.cols, q.groups));
        let gemv_pc = self.q8_gemv_pc();
        let qgrid = q8.map(|q| (q.groups.div_ceil(8).max(1) as u32, 1, 1));
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                // `run_timed` hands us the repeat count for the window; a full
                // barrier between dispatches keeps them serialised so the
                // per-iteration number is a latency, not a pipelined average.
                for _ in 0..n {
                    if let (Some(q), Some(qpc), Some(qg)) = (q8, quant_pc.as_ref(), qgrid) {
                        crate::ops::record(d, cmd, &q.quant, qpc, qg);
                        let (gx, gy) = self.grid_for(ROWS_PER_WG);
                        crate::ops::record(d, cmd, &q.gemv, &gemv_pc, (gx, gy, 1));
                    } else {
                        self.dispatch_split(d, cmd, 1);
                    }
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
        if let Some(k) = &self.ep_reduce {
            self.gpu.destroy_kernel(k);
        }
        if let Some(k) = &self.ep_kernel {
            self.gpu.destroy_kernel(k);
        }
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

/// Use the **coalesced-dequant** build of the GEMV where one exists (`_wide`,
/// `docs/phase-4/01` §8ax). That build replaces the decode unit's per-element byte
/// loads with one wide load per half-unit; the arithmetic and every output value
/// are **bit-identical** (`conformance::wide_dequant_is_bit_identical_to_the_default`),
/// so this is a pure throughput switch.
///
/// **On by default**: the head alone goes 4.25 → 1.77 ms/token on the MI50-resident
/// placement (`Q5_K` 248320x2560: 96 → 217 GB/s) and 1.60 → 1.01 ms on the RX
/// (274 → 433 GB/s, 97 % of that card's measured DRAM peak). `BRECCIA_GEMV_NO_WIDE=1`
/// restores the per-element build — that is the A/B arm and the revert path.
///
/// Cached: the decode path asks once per pipeline construction, never per dispatch.
/// Use the **word-level pack** build (`_words`, §8bc) where
/// one exists — four codes extracted per instruction and the codebook in LDS. Same
/// arithmetic, fewer instructions; bit-identical by construction.
pub fn word_pack() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // **On by default**: bit-identical (`conformance::word_pack_is_bit_identical_*`)
    // and 2.0x / 1.4x on the IQ4_XS shapes. `BRECCIA_NO_GEMV_WORDS=1` restores the
    // per-element build, which is the A/B arm and the revert.
    *ON.get_or_init(|| std::env::var_os("BRECCIA_NO_GEMV_WORDS").is_none())
}

/// The `_words` build of the GEMV for `ty`, when that type has one.
pub fn gemv_words_spv(ty: QuantType) -> Option<&'static [u8]> {
    match ty {
        QuantType::Iq4Xs => Some(spv!("gemv_iq4_xs_words")),
        _ => None,
    }
}

pub fn wide_dequant() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BRECCIA_GEMV_NO_WIDE").is_none())
}

/// The `_wide` build of the GEMV for `ty`, when that type's decode unit has been
/// converted (see `build.rs::WIDE_SHADERS`). The fp16/F32/BF16 GEMVs have no
/// byte-load loop and already run at 98-103 % of DRAM, so they never qualify.
pub fn gemv_wide_spv(ty: QuantType) -> Option<&'static [u8]> {
    match ty {
        QuantType::Q5K => Some(spv!("gemv_q5_k_wide")),
        QuantType::Iq4Xs => Some(spv!("gemv_iq4_xs_wide")),
        _ => None,
    }
}

/// The SPIR-V for `{ty, accum}` in the requested dequant variant. `wide` falls
/// back to the incumbent build for types with no `_wide` pipeline.
fn gemv_spv_variant(ty: QuantType, accum: Accum, wide: bool) -> &'static [u8] {
    if word_pack() && accum == Accum::F32 {
        if let Some(spv) = gemv_words_spv(ty) {
            return spv;
        }
    }
    if wide && accum == Accum::F32 {
        if let Some(spv) = gemv_wide_spv(ty) {
            return spv;
        }
    }
    match (ty, accum) {
        (QuantType::Iq2Xxs, Accum::F32) => spv!("gemv_iq2_xxs"),
        (QuantType::Iq2Xxs, Accum::F16) => spv!("gemv_iq2_xxs_f16acc"),
        (QuantType::Iq2Xs, Accum::F32) => spv!("gemv_iq2_xs"),
        (QuantType::Iq2Xs, Accum::F16) => spv!("gemv_iq2_xs_f16acc"),
        (QuantType::Iq2S, Accum::F32) => spv!("gemv_iq2_s"),
        (QuantType::Iq2S, Accum::F16) => spv!("gemv_iq2_s_f16acc"),
        (QuantType::Iq3Xxs, Accum::F32) => spv!("gemv_iq3_xxs"),
        (QuantType::Iq3Xxs, Accum::F16) => spv!("gemv_iq3_xxs_f16acc"),
        (QuantType::Iq3S, Accum::F32) => spv!("gemv_iq3_s"),
        (QuantType::Iq3S, Accum::F16) => spv!("gemv_iq3_s_f16acc"),
        (QuantType::Iq4Nl, Accum::F32) => spv!("gemv_iq4_nl"),
        (QuantType::Iq4Nl, Accum::F16) => spv!("gemv_iq4_nl_f16acc"),
        (QuantType::Iq4Xs, Accum::F32) => spv!("gemv_iq4_xs"),
        (QuantType::Iq4Xs, Accum::F16) => spv!("gemv_iq4_xs_f16acc"),
        (QuantType::Q4K, Accum::F32) => spv!("gemv_q4_k"),
        (QuantType::Q4K, Accum::F16) => spv!("gemv_q4_k_f16acc"),
        (QuantType::Q5K, Accum::F32) => spv!("gemv_q5_k"),
        (QuantType::Q5K, Accum::F16) => spv!("gemv_q5_k_f16acc"),
        (QuantType::Q6K, Accum::F32) => spv!("gemv_q6_k"),
        (QuantType::Q6K, Accum::F16) => spv!("gemv_q6_k_f16acc"),
        (QuantType::Q2_0, Accum::F32) => spv!("gemv_q2_0"),
        (QuantType::Q2_0, Accum::F16) => spv!("gemv_q2_0_f16acc"),
        (QuantType::F16, Accum::F32) => spv!("gemv_f16"),
        (QuantType::F16, Accum::F16) => spv!("gemv_f16_f16acc"),
        (QuantType::Bf16, Accum::F32) => spv!("gemv_bf16"),
        (QuantType::Bf16, Accum::F16) => spv!("gemv_bf16_f16acc"),
        (QuantType::F32, Accum::F32) => spv!("gemv_f32"),
        (QuantType::F32, Accum::F16) => spv!("gemv_f32_f16acc"),
    }
}

/// The `_hcact` (epilogue-fused) build of the GEMV for `{ty, accum}`, or `None`
/// when this type has none — `build.rs::EPILOGUE_SHADERS` compiles BF16 and F32
/// only, which covers the artifact's `hc_*` type and the conformance suite.
pub fn gemv_hcact_spv(ty: QuantType, accum: Accum) -> Option<&'static [u8]> {
    match (ty, accum) {
        (QuantType::Bf16, Accum::F32) => Some(spv!("gemv_bf16_hcact")),
        (QuantType::F32, Accum::F32) => Some(spv!("gemv_f32_hcact")),
        _ => None,
    }
}

/// `true` when [`Gemv::with_epilogue`] can be used for `ty` — the predicate the
/// HC mixer uses to decide whether the `down` GEMV can carry the `Act`.
pub fn has_hc_epilogue(ty: QuantType) -> bool {
    gemv_hcact_spv(ty, Accum::F32).is_some()
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
        QuantType::F16 => spv!("dequant_f16"),
        QuantType::Bf16 => spv!("dequant_bf16"),
        QuantType::F32 => spv!("dequant_f32"),
    }
}
