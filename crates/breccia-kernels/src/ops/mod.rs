//! Phase-3 op kernels: the elementwise / reduction / gather primitives the
//! forward-pass graph needs *around* the GEMVs.
//!
//! Vulkan is the primary backend, so each op is one GLSL compute shader compiled
//! by `build.rs` (`glslc` + `spirv-val`) and driven through `breccia_vk::ctx`,
//! exactly like the decode kernels in [`crate::gemv`]. Each op has a scalar CPU
//! reference in [`cpu`] and a conformance test in
//! `tests/ops_conformance.rs`; [`bench`] is `breccia bench ops`.
//!
//! | Op | Shader | Spec |
//! |---|---|---|
//! | [`RmsNorm`] | `ops_rmsnorm.comp` | `docs/specs/qwen4exp-architecture.md` §13 ops 3/4/5, §14 (`eps = 1e-6`) |
//! | [`Rope`] + [`RopeTable`] | `ops_rope.comp` | §13 op 13, §10.2-§10.3 (NEOX-paired partial rope, `n_rot = 64` of 256, base 1e7) |
//! | [`SwiGlu`] | `ops_swiglu.comp` | §5.4 (`silu(gate) * up`, 640-wide) |
//! | [`Softmax`] | `ops_softmax.comp` | §5.4 (512 router), §10.4 (512..2048 attention) |
//! | [`Router`] | `ops_router.comp` | §5.4/§8 (softmax(all) → top-10 → `max(sum, 2^-14)`) |
//! | [`EmbGather`] | `ops_gather_<type>.comp` | §6.5, §9.1 (`token_embd` IQ3_S, PLE IQ4_NL) |
//! | [`HcMixer`] | `ops_hc_*.comp` + [`crate::GemvBatch`] | §4.1 (HC_READ / HC_WRITE, `hc = 4`, `hc_lr = 320`) |
//! | [`Act`] | `ops_act.comp` | §4.1 (`silu(x/hc)`), §14 (sigmoid gates) |
//! | [`Add`] | `ops_add.comp` | residual plumbing (§4.0) |
//!
//! # Calling convention
//!
//! Every op is a compiled pipeline plus the shape facts its push constants
//! cannot carry, and every `run` takes device buffers so the caller can keep
//! weights and activations resident. Buffers are **flat**: an op that works on a
//! `[d, h, t]` tensor takes the GGML `ne`-order row-major layout directly (rows
//! of `cols` values), so nothing is transposed or copied on the way in.
//!
//! `run_timed` on each op reports GPU seconds per dispatch from queue timestamps
//! (`breccia_vk::ctx::Timing`), which is what [`bench`] reports.
//!
//! # Geometry conventions
//!
//! * Everything is `f32`. The model's weights are quantised or BF16, but the
//!   activations this layer sees are `f32` (the GEMV kernels write `f32`), and
//!   the ops are activation-side only.
//! * The 32-byte push-constant range `breccia_vk::ctx` reserves is enough for
//!   every op (four words each), so no `breccia-vk` change was needed
//!   (`docs/phase-3/03-gaps-closed.md` §1.1 reserved it).
//! * Dispatches wrap over `grid.y` so `grid.x` never exceeds the guaranteed
//!   65535 workgroups (`docs/phase-3/01` §1).

use anyhow::{anyhow, Context, Result};
use ash::vk;
use breccia_vk::ctx::{Buf, Gpu, Kernel, Qc};

use crate::gemv::{alloc, table_bytes, upload, GemvTiming, TAIL_PAD, WG_X_MAX};
use crate::quant::QuantType;

pub mod bench;
pub mod cpu;

pub use bench::{run_ops_bench, OpKind, OpsBenchOptions, OpsReport, OPS_SCHEMA_VERSION};

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/// `f32` slice → little-endian bytes (what the storage buffers hold).
pub fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Little-endian bytes → `f32`.
pub fn f32_from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Upload an `f32` vector into a fresh device-local buffer.
pub fn upload_f32(gpu: &Gpu, v: &[f32]) -> Result<Buf> {
    upload(gpu, &f32_to_bytes(v))
}

/// Download `n` `f32` values from a device-local buffer.
pub fn download_f32(gpu: &Gpu, src: &Buf, n: usize) -> Result<Vec<f32>> {
    let bytes = crate::gemv::download(gpu, src, n * 4)?;
    Ok(f32_from_bytes(&bytes))
}

/// Download `n` `u32` values from a device-local buffer.
pub fn download_u32(gpu: &Gpu, src: &Buf, n: usize) -> Result<Vec<u32>> {
    let bytes = crate::gemv::download(gpu, src, n * 4)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// `u32` slice → little-endian bytes.
pub fn u32_to_bytes(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// `(grid.x, grid.y)` for `units` invocations at `per_wg` invocations per
/// workgroup, wrapped over `grid.y` so `grid.x <= 65535` (the only value every
/// Vulkan device guarantees).
pub(crate) fn grid_for(units: usize, per_wg: usize) -> (u32, u32) {
    let groups = units.div_ceil(per_wg.max(1)).max(1) as u64;
    let gx = groups.min(u64::from(WG_X_MAX)) as u32;
    let gy = groups
        .div_ceil(u64::from(WG_X_MAX))
        .min(u64::from(u32::MAX)) as u32;
    (gx, gy)
}

/// A 32-byte push-constant block builder (every op uses four words or fewer).
#[derive(Clone, Copy)]
pub(crate) struct Pc {
    bytes: [u8; 32],
    len: usize,
}

impl Pc {
    pub(crate) fn new() -> Self {
        Pc {
            bytes: [0u8; 32],
            len: 0,
        }
    }

    pub(crate) fn u(mut self, v: u32) -> Self {
        let at = self.len;
        if at + 4 <= 32 {
            self.bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
            self.len = at + 4;
        }
        self
    }

    pub(crate) fn f(self, v: f32) -> Self {
        self.u(v.to_bits())
    }

    /// A signed 32-bit word (sequence positions in the QSA kernels).
    pub(crate) fn i(self, v: i32) -> Self {
        self.u(v as u32)
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

/// One compiled pipeline plus the device that owns it.
pub(crate) struct Pipe<'g> {
    gpu: &'g Gpu,
    kernel: Kernel,
    what: &'static str,
}

impl<'g> Pipe<'g> {
    /// The device that owns this pipeline (pipeline `Drop` frees the kernel, so
    /// an owning struct can free its own buffers in the same order).
    pub(crate) fn gpu(&self) -> &'g Gpu {
        self.gpu
    }

    pub(crate) fn new(
        gpu: &'g Gpu,
        spv: &'static [u8],
        n_buffers: u32,
        what: &'static str,
    ) -> Result<Self> {
        let kernel = gpu
            .kernel(spv, n_buffers)
            .map_err(|e| anyhow!("creating the {what} pipeline: {e}"))?;
        Ok(Pipe { gpu, kernel, what })
    }

    pub(crate) fn bind(&self, bufs: &[&Buf]) -> Result<()> {
        let handles: Vec<vk::Buffer> = bufs.iter().map(|b| b.buf).collect();
        self.gpu
            .bind_kernel(&self.kernel, &handles)
            .map_err(|e| anyhow!("{}: {e}", self.what))
    }

    /// One dispatch, synchronously.
    pub(crate) fn run(&self, bufs: &[&Buf], pc: &Pc, grid: (u32, u32, u32)) -> Result<()> {
        self.bind(bufs)?;
        self.gpu
            .wall_run(Qc::C, |d, cmd| {
                record(d, cmd, &self.kernel, pc.as_bytes(), grid);
            })
            .map_err(|e| anyhow!("{}: {e}", self.what))?;
        Ok(())
    }

    /// One dispatch repeated for `target_s`, GPU-timestamped. A full barrier
    /// between dispatches keeps the reported number a latency, not a pipelined
    /// average (same protocol as `Gemv::run_timed`).
    pub(crate) fn run_timed(
        &self,
        bufs: &[&Buf],
        pc: &Pc,
        grid: (u32, u32, u32),
        target_s: f64,
    ) -> Result<GemvTiming> {
        self.bind(bufs)?;
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                for _ in 0..n {
                    record(d, cmd, &self.kernel, pc.as_bytes(), grid);
                    unsafe { Gpu::barrier_all(d, cmd) };
                }
            })
            .map_err(|e| anyhow!("{}: {e}", self.what))?;
        Ok(GemvTiming {
            per_iter_s: t.per_iter_s,
            iters: t.iters,
            total_s: t.total_s,
            used_timestamps: t.used_timestamps,
        })
    }
}

pub(crate) fn record(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    k: &Kernel,
    pc: &[u8],
    grid: (u32, u32, u32),
) {
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
        d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, pc);
        d.cmd_dispatch(cmd, grid.0.max(1), grid.1.max(1), grid.2.max(1));
    }
}

impl Drop for Pipe<'_> {
    fn drop(&mut self) {
        self.gpu.destroy_kernel(&self.kernel);
    }
}

/// 4-element device buffer bound where a shader declares an optional input.
pub(crate) fn empty_buf(gpu: &Gpu, what: &str) -> Result<Buf> {
    upload(gpu, &[0u8; 16]).with_context(|| format!("allocating the unused {what} binding"))
}

// ---------------------------------------------------------------------------
// 1. RMSNorm
// ---------------------------------------------------------------------------

/// Model RMSNorm epsilon (`attention.layer_norm_rms_epsilon = 1e-6`,
/// `docs/specs/qwen4exp-architecture.md` §1/§14).
pub const MODEL_RMS_EPS: f32 = 1e-6;

/// Row-wise RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * gamma`.
///
/// One pipeline serves every norm shape in the model, because they differ only
/// in `rows` (per call), `cols` and the gamma grouping (per pipeline):
///
/// | use | cols | rows | `wgroups` |
/// |---|---|---|---|
/// | HC stack norm (`hc_*_norm`, `output_hc_norm`) | 2560 | `hc*T` | `hc` (gamma `[2560, hc]`) |
/// | per-head Q/K norm, indexer q/k norm | 256 | `24*T` / `4*T` / `T` | 1 |
/// | GDN post-recurrence `ssm_norm` | 128 | `48*T` | 1 |
/// | PLE `*_norm_*` | 2560 / 10240 | `T` / `hc*T` | 1 / `hc` |
///
/// `wgroups = 0` means "no gamma". That is not a shortcut for a norm without a
/// weight: it is also exactly the GDN q/k **L2 normalisation**
/// (`x / sqrt(sum(x^2) + eps)` with `eps/n` folded in, `LC/src/models/models.h:14-18`),
/// which is this kernel with gamma = 1.
pub struct RmsNorm<'g> {
    pipe: Pipe<'g>,
    cols: usize,
    wgroups: usize,
    eps: f32,
    /// Bound in place of `gamma` when `wgroups == 0` (every declared binding
    /// must be bound).
    no_gamma: Buf,
}

impl<'g> RmsNorm<'g> {
    /// Pipeline with the model's epsilon (1e-6) and `wgroups` gamma slices.
    pub fn new(gpu: &'g Gpu, cols: usize, wgroups: usize) -> Result<Self> {
        Self::with_eps(gpu, cols, wgroups, MODEL_RMS_EPS)
    }

    /// Pipeline with an explicit epsilon (the L2-norm case passes `eps / cols`).
    pub fn with_eps(gpu: &'g Gpu, cols: usize, wgroups: usize, eps: f32) -> Result<Self> {
        if cols == 0 {
            return Err(anyhow!("rmsnorm cols must be non-zero"));
        }
        let pipe = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/ops_rmsnorm.spv")),
            3,
            "rmsnorm",
        )?;
        let no_gamma = match empty_buf(gpu, "gamma") {
            Ok(b) => b,
            Err(e) => {
                drop(pipe);
                return Err(e);
            }
        };
        Ok(RmsNorm {
            pipe,
            cols,
            wgroups,
            eps,
            no_gamma,
        })
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn wgroups(&self) -> usize {
        self.wgroups
    }
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Bytes of gamma this pipeline expects (`wgroups * cols` values, or 0).
    pub fn gamma_values(&self) -> usize {
        self.wgroups * self.cols
    }

    fn launch<'a>(
        &'a self,
        x: &'a Buf,
        gamma: Option<&'a Buf>,
        out: &'a Buf,
        rows: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let g = match self.wgroups {
            0 => &self.no_gamma,
            _ => gamma.expect("gamma is required when wgroups > 0"),
        };
        let bufs = vec![x, g, out];
        let pc = Pc::new()
            .u(rows as u32)
            .u(self.cols as u32)
            .u(self.wgroups as u32)
            .f(self.eps);
        (bufs, pc, (rows.max(1) as u32, 1, 1))
    }

    /// Normalise `rows` rows of `cols` values. `gamma` must be `Some` iff
    /// `wgroups > 0`; `out` may alias `x`.
    pub fn run(&self, x: &Buf, gamma: Option<&Buf>, out: &Buf, rows: usize) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(x, gamma, out, rows);
        self.pipe.run(&bufs, &pc, grid)
    }

    /// Timed variant of [`RmsNorm::run`].
    pub fn run_timed(
        &self,
        x: &Buf,
        gamma: Option<&Buf>,
        out: &Buf,
        rows: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if rows == 0 {
            return Err(anyhow!("rmsnorm rows must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(x, gamma, out, rows);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

impl Drop for RmsNorm<'_> {
    fn drop(&mut self) {
        // The kernel is destroyed by `Pipe`'s own `Drop`; the placeholder gamma
        // buffer is this struct's only other device object.
        self.pipe.gpu.destroy_buf(&self.no_gamma);
    }
}

// ---------------------------------------------------------------------------
// 2. IMRoPE
// ---------------------------------------------------------------------------

/// `rope.freq_base` of the artifact (`docs/specs/qwen4exp-architecture.md` §1).
pub const MODEL_ROPE_BASE: f64 = 1e7;
/// `rope.dimension_count` — rotated dims of a 256-wide head (§1, §13 note).
pub const MODEL_N_ROT: usize = 64;
/// `attention.key_length` / `value_length` — the head width.
pub const MODEL_HEAD_DIM: usize = 256;

/// Host-precomputed `[max_pos][n_rot/2]` `(cos, sin)` table, `f32`.
///
/// Built in `f64` by [`cpu::rope_table`] (the same loop as strata's
/// `build_rope_table`) and uploaded once per sequence position range. Scaling —
/// linear or YaRN — would enter here and nowhere else: the kernel is a pure
/// rotation over these values, so it cannot tell a scaled table from an
/// unscaled one (`ST/include/strata/kernels/rope.hpp:7-11`). The artifact ships
/// no `rope.scaling` keys, so only the plain table is implemented.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTable {
    cos_sin: Vec<f32>,
    n_rot: usize,
    max_pos: usize,
}

impl RopeTable {
    /// Build a table of `max_pos` positions for `n_rot` rotated dims.
    pub fn build(freq_base: f64, n_rot: usize, max_pos: usize) -> Result<Self> {
        if n_rot == 0 || !n_rot.is_multiple_of(2) {
            return Err(anyhow!("n_rot must be even and non-zero, got {n_rot}"));
        }
        if max_pos == 0 {
            return Err(anyhow!("max_pos must be non-zero"));
        }
        Ok(RopeTable {
            cos_sin: cpu::rope_table(freq_base, n_rot, max_pos),
            n_rot,
            max_pos,
        })
    }

    /// The artifact's table: `base = 1e7`, `n_rot = 64` (§1).
    pub fn model(max_pos: usize) -> Result<Self> {
        Self::build(MODEL_ROPE_BASE, MODEL_N_ROT, max_pos)
    }

    pub fn n_rot(&self) -> usize {
        self.n_rot
    }
    /// Pairs per position (`n_rot / 2`).
    pub fn pairs(&self) -> usize {
        self.n_rot / 2
    }
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }
    /// `[pos][pair] = (cos, sin)` as a flat `f32` slice.
    pub fn cos_sin(&self) -> &[f32] {
        &self.cos_sin
    }
    pub fn bytes(&self) -> u64 {
        self.cos_sin.len() as u64 * 4
    }

    /// Upload for binding 1 of [`Rope`].
    pub fn upload(&self, gpu: &Gpu) -> Result<Buf> {
        upload_f32(gpu, &self.cos_sin)
    }
}

/// Partial NEOX-paired rotary embedding, applied to Q and K.
///
/// Every QSA layer rotates both (`docs/specs/qwen4exp-architecture.md` §4.3
/// steps 4 and §10.3 for the indexer query/pooled key); the GDN layers do not
/// rotate anything. The op is the same for both: `rows = heads * T`, one int32
/// position per row, `head_dim` wide rows, `n_rot` of them rotated.
///
/// # Layout
///
/// A Q activation of shape `[head_dim, n_head, T]` (GGML `ne` order) is exactly
/// `n_head * T` contiguous rows of `head_dim` values, in row order
/// `head + n_head * token` — which is the order `inp_pos` repeats across, so the
/// caller builds `positions[row] = pos[row / n_head]` once per token batch.
///
/// # In place
///
/// Bind the same buffer for `x` and `out`: under NEOX no two pairs share an
/// element, so each invocation reads both of its inputs before writing both of
/// its outputs.
pub struct Rope<'g> {
    pipe: Pipe<'g>,
    head_dim: usize,
    n_rot: usize,
    max_pos: usize,
}

impl<'g> Rope<'g> {
    /// `head_dim` wide rows, `n_rot` rotated dims, a table of `max_pos`
    /// positions.
    pub fn new(gpu: &'g Gpu, head_dim: usize, n_rot: usize, max_pos: usize) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(anyhow!(
                "head_dim must be even and non-zero, got {head_dim}"
            ));
        }
        if n_rot == 0 || !n_rot.is_multiple_of(2) || n_rot > head_dim {
            return Err(anyhow!(
                "n_rot must be even, non-zero and <= head_dim ({head_dim}), got {n_rot}"
            ));
        }
        if max_pos == 0 {
            return Err(anyhow!("max_pos must be non-zero"));
        }
        let pipe = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/ops_rope.spv")),
            4,
            "rope",
        )?;
        Ok(Rope {
            pipe,
            head_dim,
            n_rot,
            max_pos,
        })
    }

    /// The artifact's Q/K rotation: 64 of 256 dims.
    pub fn model(gpu: &'g Gpu, max_pos: usize) -> Result<Self> {
        Self::new(gpu, MODEL_HEAD_DIM, MODEL_N_ROT, max_pos)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn n_rot(&self) -> usize {
        self.n_rot
    }
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    fn launch<'a>(
        &'a self,
        x: &'a Buf,
        table: &'a Buf,
        positions: &'a Buf,
        out: &'a Buf,
        rows: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        self.launch_at(x, table, positions, out, rows, 0, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_at<'a>(
        &'a self,
        x: &'a Buf,
        table: &'a Buf,
        positions: &'a Buf,
        out: &'a Buf,
        rows: usize,
        pos_base: i32,
        pos_div: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let bufs = vec![x, table, positions, out];
        let pc = Pc::new()
            .u(rows as u32)
            .u(self.head_dim as u32)
            .u(self.n_rot as u32)
            .u(self.max_pos as u32)
            .i(pos_base)
            .u(pos_div.max(1) as u32);
        let units = rows * (self.head_dim / 2);
        let (gx, gy) = grid_for(units, 64);
        (bufs, pc, (gx, gy, 1))
    }

    /// Rotate `rows` rows. `positions` holds `rows` int32 values (device array).
    pub fn run(&self, x: &Buf, table: &Buf, positions: &Buf, out: &Buf, rows: usize) -> Result<()> {
        self.run_at(x, table, positions, out, rows, 0, 1)
    }

    /// Rotate `rows` rows reading the position of row `r` from
    /// `positions[pos_base + r/pos_div]`.
    ///
    /// `pos_div = heads_per_token` lets a caller pass one identity position
    /// table for the whole context — `positions[i] = i` — and address a token
    /// batch by its base position, so a decode step never re-uploads a
    /// positions array (and a captured graph cannot freeze one). `run` is this
    /// with `(0, 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_at(
        &self,
        x: &Buf,
        table: &Buf,
        positions: &Buf,
        out: &Buf,
        rows: usize,
        pos_base: i32,
        pos_div: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch_at(x, table, positions, out, rows, pos_base, pos_div);
        self.pipe.run(&bufs, &pc, grid)
    }

    /// Timed variant of [`Rope::run_at`] (the `pos_base`/`pos_div` form).
    #[allow(clippy::too_many_arguments)]
    pub fn run_at_timed(
        &self,
        x: &Buf,
        table: &Buf,
        positions: &Buf,
        out: &Buf,
        rows: usize,
        pos_base: i32,
        pos_div: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if rows == 0 {
            return Err(anyhow!("rope rows must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch_at(x, table, positions, out, rows, pos_base, pos_div);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }

    /// Timed variant of [`Rope::run`].
    pub fn run_timed(
        &self,
        x: &Buf,
        table: &Buf,
        positions: &Buf,
        out: &Buf,
        rows: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if rows == 0 {
            return Err(anyhow!("rope rows must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(x, table, positions, out, rows);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

// ---------------------------------------------------------------------------
// 3. SwiGLU
// ---------------------------------------------------------------------------

/// `out = silu(gate) * up`, elementwise over `n` values.
///
/// The layer's routed set is one dispatch: `gate`/`up` are the two halves of the
/// batched GEMV output (`10 experts × 640 = 6400` values, or `6400*T` for a
/// prefill), and the shared expert is another (`640*T`). `docs/phase-3/01` §7
/// item 5 and `02` §7 item 7 both list this as the companion of the batched
/// expert dispatch: it removes one round trip of the 640-wide intermediate and
/// one dispatch per layer.
pub struct SwiGlu<'g> {
    pipe: Pipe<'g>,
}

impl<'g> SwiGlu<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(SwiGlu {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/ops_swiglu.spv")),
                3,
                "swiglu",
            )?,
        })
    }

    fn launch<'a>(
        &'a self,
        gate: &'a Buf,
        up: &'a Buf,
        out: &'a Buf,
        n: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let (gx, gy) = grid_for(n, 256);
        (vec![gate, up, out], Pc::new().u(n as u32), (gx, gy, 1))
    }

    /// `out[i] = silu(gate[i]) * up[i]`; `out` may alias either input.
    pub fn run(&self, gate: &Buf, up: &Buf, out: &Buf, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(gate, up, out, n);
        self.pipe.run(&bufs, &pc, grid)
    }

    /// Timed variant of [`SwiGlu::run`].
    pub fn run_timed(
        &self,
        gate: &Buf,
        up: &Buf,
        out: &Buf,
        n: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if n == 0 {
            return Err(anyhow!("swiglu n must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(gate, up, out, n);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

// ---------------------------------------------------------------------------
// 4. Softmax
// ---------------------------------------------------------------------------

/// Stable row-wise softmax.
///
/// Serves the router (512 logits) and, when the attention kernel lands, the
/// attention probabilities over the selection width (`min(n_kv, 2051)`, i.e.
/// 512..2048 with a ragged tail, §10.4). No fixed width is assumed.
pub struct Softmax<'g> {
    pipe: Pipe<'g>,
}

impl<'g> Softmax<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(Softmax {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/ops_softmax.spv")),
                2,
                "softmax",
            )?,
        })
    }

    fn launch<'a>(
        &'a self,
        x: &'a Buf,
        out: &'a Buf,
        rows: usize,
        cols: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let (gx, gy) = grid_for(rows, 1);
        (
            vec![x, out],
            Pc::new().u(rows as u32).u(cols as u32),
            (gx, gy, 1),
        )
    }

    /// Normalise `rows` rows of `cols` values; `out` may alias `x`.
    pub fn run(&self, x: &Buf, out: &Buf, rows: usize, cols: usize) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(x, out, rows, cols);
        self.pipe.run(&bufs, &pc, grid)
    }

    /// Timed variant of [`Softmax::run`].
    pub fn run_timed(
        &self,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if rows == 0 || cols == 0 {
            return Err(anyhow!("softmax shape must be non-empty"));
        }
        let (bufs, pc, grid) = self.launch(x, out, rows, cols);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

// ---------------------------------------------------------------------------
// 5. Router top-k
// ---------------------------------------------------------------------------

/// `expert_count` of the model (`docs/specs/qwen4exp-architecture.md` §1).
pub const MODEL_N_EXPERT: usize = 512;
/// `expert_used_count` — the top-k.
pub const MODEL_TOP_K: usize = 10;
/// ggml's renormalisation clamp, `2^-14` (§5.4 / §8).
// `6.103515625e-05` is exactly 2^-14; written as the power so the constant and
// the shader's `6.103515625e-05` literal are visibly the same number.
pub const ROUTER_CLAMP: f32 = 6.103_515_6e-5;
/// The kernel keeps the selected probabilities in shared memory: `k <= 16`.
pub const ROUTER_MAX_K: usize = 16;

/// MoE router: softmax over **all** experts → top-k → clamp-renormalise.
///
/// One dispatch per token batch. `ids` receives `rows * k` `u32` expert ids in
/// descending probability (ties to the lowest id, deterministic) and `weights`
/// `rows * k` `f32` values summing to 1.
///
/// # Precision
///
/// The production kernel accumulates in `f64` and this one cannot (`breccia-vk`
/// does not enable `shaderFloat64`), so the *ids* are contractual and the
/// weights are accurate to ~1e-7 relative (measured 1.5e-8 … 5.9e-8 in
/// `tests/ops_conformance.rs`). Exact ties still tie exactly: `exp` is
/// deterministic and applied to identical inputs, so equal logits give bitwise
/// equal probabilities and the documented lowest-id rule decides.
pub struct Router<'g> {
    pipe: Pipe<'g>,
    n_expert: usize,
    k: usize,
}

impl<'g> Router<'g> {
    /// `n_expert <= 512` (the kernel's workgroup), `k <= 16`.
    pub fn new(gpu: &'g Gpu, n_expert: usize, k: usize) -> Result<Self> {
        if n_expert == 0 || n_expert > 512 {
            return Err(anyhow!(
                "router n_expert must be in 1..=512 (one 512-thread workgroup), got {n_expert}"
            ));
        }
        if k == 0 || k > n_expert || k > ROUTER_MAX_K {
            return Err(anyhow!(
                "router k must be in 1..={} and <= n_expert ({n_expert}), got {k}",
                ROUTER_MAX_K.min(n_expert)
            ));
        }
        Ok(Router {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/ops_router.spv")),
                3,
                "router",
            )?,
            n_expert,
            k,
        })
    }

    /// The artifact's router: 512 experts, top-10.
    pub fn model(gpu: &'g Gpu) -> Result<Self> {
        Self::new(gpu, MODEL_N_EXPERT, MODEL_TOP_K)
    }

    pub fn n_expert(&self) -> usize {
        self.n_expert
    }
    pub fn k(&self) -> usize {
        self.k
    }
    /// Bytes of the `ids` buffer for `rows` tokens.
    pub fn ids_bytes(&self, rows: usize) -> u64 {
        rows as u64 * self.k as u64 * 4
    }
    /// Bytes of the `weights` buffer for `rows` tokens.
    pub fn weights_bytes(&self, rows: usize) -> u64 {
        rows as u64 * self.k as u64 * 4
    }

    fn launch<'a>(
        &'a self,
        logits: &'a Buf,
        ids: &'a Buf,
        weights: &'a Buf,
        rows: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let (gx, gy) = grid_for(rows, 1);
        (
            vec![logits, ids, weights],
            Pc::new()
                .u(rows as u32)
                .u(self.n_expert as u32)
                .u(self.k as u32),
            (gx, gy, 1),
        )
    }

    /// `logits` holds `rows * n_expert` values.
    pub fn run(&self, logits: &Buf, ids: &Buf, weights: &Buf, rows: usize) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(logits, ids, weights, rows);
        self.pipe.run(&bufs, &pc, grid)
    }

    /// Timed variant of [`Router::run`].
    pub fn run_timed(
        &self,
        logits: &Buf,
        ids: &Buf,
        weights: &Buf,
        rows: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if rows == 0 {
            return Err(anyhow!("router rows must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(logits, ids, weights, rows);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

// ---------------------------------------------------------------------------
// 6. Quantised row gather + dequant
// ---------------------------------------------------------------------------

/// Gather `rows` rows of a packed table and dequantise them to `f32`.
///
/// Both embedding paths of the model are this op:
///
/// | table | shape | type | bytes | path |
/// |---|---|---|---|---|
/// | `token_embd.weight` | `[2560, 248320]` | IQ3_S | 260 MiB | resident or staged |
/// | `per_layer_token_embd.weight` | `[160, 320001536]` | IQ4_NL | 26.8 GiB | staged, on demand |
///
/// (`docs/specs/qwen4exp-architecture.md` §6.5, §9.1.) The table is never
/// copied: [`EmbGather::pack_rows`] copies **only the requested rows** out of a
/// `breccia_format` mapping (`rows * row_bytes` bytes, one staging upload) and
/// [`EmbGather::run`] dequantises them in one dispatch. A caller that has the
/// table resident instead binds it whole and passes the token ids as the index
/// list — same kernel, no staging copy.
///
/// The PLE gather is the head-slowest flatten of §9.1 for free: gathering
/// `16 * T` rows in token-major order (`row = t*16 + head`) produces
/// `[16*160, T]` = `[2560, T]` exactly as `qwen4exp.cpp:1361-1364` writes it.
pub struct EmbGather<'g> {
    pipe: Pipe<'g>,
    ty: QuantType,
    cols: usize,
    blocks_per_row: usize,
    padded_cols: usize,
    row_bytes: usize,
    tables: Buf,
    no_x: Buf,
}

impl<'g> EmbGather<'g> {
    /// Pipeline for a table of `cols`-wide rows of `ty`.
    pub fn new(gpu: &'g Gpu, ty: QuantType, cols: usize) -> Result<Self> {
        if cols == 0 {
            return Err(anyhow!("emb gather cols must be non-zero"));
        }
        let blocks_per_row = cols.div_ceil(ty.block_elems());
        let padded_cols = blocks_per_row * ty.block_elems();
        let row_bytes = blocks_per_row * ty.block_bytes();
        let pipe = Pipe::new(gpu, gather_spv(ty), 5, "emb gather")?;
        let tables = match upload(gpu, &table_bytes()) {
            Ok(b) => b,
            Err(e) => {
                drop(pipe);
                return Err(e.context("uploading the quant tables"));
            }
        };
        let no_x = match empty_buf(gpu, "x") {
            Ok(b) => b,
            Err(e) => {
                gpu.destroy_buf(&tables);
                drop(pipe);
                return Err(e);
            }
        };
        Ok(EmbGather {
            pipe,
            ty,
            cols,
            blocks_per_row,
            padded_cols,
            row_bytes,
            tables,
            no_x,
        })
    }

    pub fn ty(&self) -> QuantType {
        self.ty
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Packed bytes of one table row (`blocks_per_row * block_bytes`).
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    /// Floats per output row (each row padded to a whole block).
    pub fn padded_cols(&self) -> usize {
        self.padded_cols
    }
    pub fn blocks_per_row(&self) -> usize {
        self.blocks_per_row
    }
    /// Bytes the `out` buffer needs for `rows` gathered rows.
    pub fn out_bytes(&self, rows: usize) -> u64 {
        (rows * self.padded_cols * 4) as u64
    }
    /// Bytes the packed-rows buffer needs for `rows` (`rows * row_bytes`).
    pub fn stage_bytes(&self, rows: usize) -> u64 {
        (rows * self.row_bytes) as u64
    }

    /// Copy `rows` table rows out of a mapped table (`&[u8]` — a
    /// `breccia_format::TensorView::as_slice()`) into one contiguous block.
    ///
    /// `row_bytes` is [`EmbGather::row_bytes`]; an out-of-range row id is an
    /// error, never a read past the mapping.
    pub fn pack_rows(table: &[u8], row_bytes: usize, rows: &[u32]) -> Result<Vec<u8>> {
        if row_bytes == 0 {
            return Err(anyhow!("row_bytes must be non-zero"));
        }
        let n_table_rows = table.len() / row_bytes;
        let mut out = Vec::with_capacity(rows.len() * row_bytes);
        for &r in rows {
            let r = r as usize;
            if r >= n_table_rows {
                return Err(anyhow!(
                    "row {r} is outside a table of {n_table_rows} rows ({row_bytes} B/row)"
                ));
            }
            out.extend_from_slice(&table[r * row_bytes..(r + 1) * row_bytes]);
        }
        Ok(out)
    }

    /// Identity index list (`0..batch`) for the staged path.
    pub fn identity_indices(batch: usize) -> Vec<u32> {
        (0..batch as u32).collect()
    }

    fn launch<'a>(
        &'a self,
        packed: &'a Buf,
        idx: &'a Buf,
        out: &'a Buf,
        batch: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let bufs = vec![packed, &self.no_x, out, &self.tables, idx];
        let pc = Pc::new()
            .u(batch as u32)
            .u(self.cols as u32)
            .u(self.blocks_per_row as u32)
            .u(1u32);
        // One subgroup per row, 8 subgroups (rows) per 256-thread workgroup —
        // `ROWS_PER_WG` in `shaders/ops_gather.glsl`.
        let (gx, gy) = grid_for(batch, 8);
        (bufs, pc, (gx, gy, 1))
    }

    /// Dequantise `batch` rows. `idx` holds `batch` `u32` table-row numbers
    /// (identity for a staged buffer).
    pub fn run(&self, packed: &Buf, idx: &Buf, out: &Buf, batch: usize) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(packed, idx, out, batch);
        self.pipe.run(&bufs, &pc, grid)
    }

    /// Timed variant of [`EmbGather::run`] (includes the index read).
    pub fn run_timed(
        &self,
        packed: &Buf,
        idx: &Buf,
        out: &Buf,
        batch: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if batch == 0 {
            return Err(anyhow!("emb gather batch must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(packed, idx, out, batch);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }

    /// One-call staged gather: pack `rows` out of `table`, upload them, and
    /// dequantise into `out`. Returns the staging buffer so the caller can
    /// release it (or reuse it) explicitly.
    pub fn gather_staged(&self, gpu: &Gpu, table: &[u8], rows: &[u32], out: &Buf) -> Result<Buf> {
        let staged = Self::pack_rows(table, self.row_bytes, rows)?;
        let packed = upload(gpu, &staged).context("uploading the gathered rows")?;
        let idx = match upload(gpu, &u32_to_bytes(&Self::identity_indices(rows.len()))) {
            Ok(b) => b,
            Err(e) => {
                gpu.destroy_buf(&packed);
                return Err(e);
            }
        };
        let r = self.run(&packed, &idx, out, rows.len());
        gpu.destroy_buf(&idx);
        match r {
            Ok(()) => Ok(packed),
            Err(e) => {
                gpu.destroy_buf(&packed);
                Err(e)
            }
        }
    }

    /// Release a buffer returned by [`EmbGather::gather_staged`].
    pub fn release(&self, gpu: &Gpu, b: &Buf) {
        gpu.destroy_buf(b);
    }
}

impl Drop for EmbGather<'_> {
    fn drop(&mut self) {
        let gpu = self.pipe.gpu;
        gpu.destroy_buf(&self.no_x);
        gpu.destroy_buf(&self.tables);
    }
}

// ---------------------------------------------------------------------------
// 7. HC mixer primitives
// ---------------------------------------------------------------------------

/// Hyper-connection stream count (`hyper_connection.count = 4`).
pub const MODEL_HC: usize = 4;
/// Hyper-connection bottleneck rank (`hyper_connection.low_rank = 320`).
pub const MODEL_HC_LR: usize = 320;

/// The five elementwise modes of `ops_act.comp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActMode {
    /// `out = silu(x)`.
    Silu,
    /// `out = silu(x / s)` — the HC bottleneck, §4.1 `silu((xn @ w_down.T)/hc)`.
    SiluDiv,
    /// `out = sigmoid(x)` — the attention/output gates and the GDN beta.
    Sigmoid,
    /// `out = x * s` — residual/attention scaling, the shared expert's gate.
    Scale,
    /// `out = silu(x) * s`.
    SiluScale,
}

impl ActMode {
    const fn code(self) -> u32 {
        match self {
            ActMode::Silu => 0,
            ActMode::SiluDiv => 1,
            ActMode::Sigmoid => 2,
            ActMode::Scale => 3,
            ActMode::SiluScale => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            ActMode::Silu => "silu",
            ActMode::SiluDiv => "silu_div",
            ActMode::Sigmoid => "sigmoid",
            ActMode::Scale => "scale",
            ActMode::SiluScale => "silu_scale",
        }
    }
}

/// One-dispatch elementwise activation / rescale (see [`ActMode`]).
pub struct Act<'g> {
    pipe: Pipe<'g>,
    mode: ActMode,
}

impl<'g> Act<'g> {
    pub fn new(gpu: &'g Gpu, mode: ActMode) -> Result<Self> {
        Ok(Act {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/ops_act.spv")),
                2,
                "act",
            )?,
            mode,
        })
    }

    pub fn mode(&self) -> ActMode {
        self.mode
    }

    fn launch<'a>(
        &'a self,
        x: &'a Buf,
        out: &'a Buf,
        n: usize,
        s: f32,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let (gx, gy) = grid_for(n, 256);
        (
            vec![x, out],
            Pc::new().u(n as u32).u(self.mode.code()).f(s),
            (gx, gy, 1),
        )
    }

    /// Apply the mode to `n` values. `s` is the multiplier (`Scale`,
    /// `SiluScale`) or divisor (`SiluDiv`) and is ignored by `Silu`/`Sigmoid`.
    /// `out` may alias `x`.
    pub fn run(&self, x: &Buf, out: &Buf, n: usize, s: f32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(x, out, n, s);
        self.pipe.run(&bufs, &pc, grid)
    }

    /// Timed variant of [`Act::run`].
    pub fn run_timed(
        &self,
        x: &Buf,
        out: &Buf,
        n: usize,
        s: f32,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if n == 0 {
            return Err(anyhow!("act n must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(x, out, n, s);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

/// HC_READ tail: `mixed[d] = (1/hc) * sum_c xn[c][d] * sigmoid(gate[c][d])`,
/// **and** the four `inject` dot products, folded in.
///
/// Layouts (`GGML ne` order, see the shader):
/// `xn`, `gate` are `[n_embd, hc, T]`; `mixed` is `[n_embd, T]`;
/// `w_inject` is `[hc, n_embd]`; `part` is `[T, hc, chunks]` with
/// `chunks = n_embd/256` partial sums per (token, stream).
///
/// # Why `inject` is here (`docs/phase-3/04` §7.2 item 1)
///
/// `inject = xn @ w_inject.T` is a 4 x 10240 projection: 82 KiB of BF16 read by
/// one 8-row workgroup, measured at **21.85 us** on the RX 6950 XT — pure dispatch
/// latency, 36 % of the whole HC_READ chain (2.1 ms/token across the model's 96
/// HC_READs). This kernel already walks every `xn` value for `mixed`, so the four
/// dot products ride along in the same registers.
///
/// The dot products span all `n_embd` values while the grid gives one workgroup
/// per 256-element chunk, so each workgroup writes its **partial** sums to
/// `part[t][c][k]`; [`HcWrite`] — which reads the four gates once per token anyway —
/// sums the `chunks` of them. One dispatch removed per HC_READ, no atomics, and a
/// summation order that is fixed by the chunk index rather than by race.
pub struct HcGatedMean<'g> {
    pipe: Pipe<'g>,
    n_embd: usize,
    hc: usize,
    /// Partial columns per token (`n_embd / 256`); 0 disables the fold.
    chunks: usize,
    /// 0 = the inject weight is BF16, 1 = F32.
    dt: u32,
}

impl<'g> HcGatedMean<'g> {
    /// The strided mean on its own (no `inject`).
    pub fn new(gpu: &'g Gpu, n_embd: usize, hc: usize) -> Result<Self> {
        Self::with_inject(gpu, n_embd, hc, None)
    }

    /// `inject = Some((ty, chunks))` folds the 4 x `n_embd` projection in; `ty`
    /// must be BF16 (the artifact's `hc_*` type, §14) or F32 (tests), and
    /// `chunks` must be `n_embd / 256` so that a workgroup is never split across
    /// a token boundary.
    pub fn with_inject(
        gpu: &'g Gpu,
        n_embd: usize,
        hc: usize,
        inject: Option<(QuantType, usize)>,
    ) -> Result<Self> {
        if n_embd == 0 || hc == 0 {
            return Err(anyhow!("hc gated mean needs n_embd > 0 and hc > 0"));
        }
        if hc > 8 {
            return Err(anyhow!(
                "hc gated mean caches {hc} stream partials in shared memory (max 8)"
            ));
        }
        let (dt, chunks) = match inject {
            None => (0u32, 0usize),
            Some((ty, chunks)) => {
                if !n_embd.is_multiple_of(256) {
                    return Err(anyhow!(
                        "the fused inject needs n_embd % 256 == 0 (got {n_embd}): one \
                         workgroup per token chunk is what makes the partial sums \
                         unambiguous"
                    ));
                }
                if chunks != n_embd / 256 {
                    return Err(anyhow!(
                        "fused inject chunks must be n_embd/256 = {}, got {chunks}",
                        n_embd / 256
                    ));
                }
                match ty {
                    QuantType::Bf16 => (0u32, chunks),
                    QuantType::F32 => (1u32, chunks),
                    other => {
                        return Err(anyhow!(
                            "the fused hc inject reads BF16 or F32 weights (the artifact \
                             ships BF16); got {}",
                            other.name()
                        ))
                    }
                }
            }
        };
        Ok(HcGatedMean {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/ops_hc_gated_mean.spv")),
                5,
                "hc gated mean",
            )?,
            n_embd,
            hc,
            chunks,
            dt,
        })
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
    pub fn hc(&self) -> usize {
        self.hc
    }
    /// Partial sums per (token, stream); 0 when `inject` is not folded in.
    pub fn chunks(&self) -> usize {
        self.chunks
    }
    /// Values the partial buffer holds for `tokens` tokens (`tokens*hc*chunks`).
    pub fn part_values(&self, tokens: usize) -> usize {
        tokens * self.hc * self.chunks
    }

    fn launch<'a>(
        &'a self,
        xn: &'a Buf,
        gate: &'a Buf,
        out: &'a Buf,
        w: &'a Buf,
        part: &'a Buf,
        tokens: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let (gx, gy) = grid_for(self.n_embd * tokens, 256);
        (
            vec![xn, gate, out, w, part],
            Pc::new()
                .u(self.n_embd as u32)
                .u(self.hc as u32)
                .u(tokens as u32)
                .u(self.dt)
                .u(self.chunks as u32),
            (gx, gy, 1),
        )
    }

    /// `mixed` for `tokens` tokens; when the fold is enabled, `w` is
    /// `hc_*_inject.weight` and `part` receives `tokens*hc*chunks` partials.
    /// `w`/`part` must be real buffers even when unused (every declared binding
    /// is bound).
    pub fn run(
        &self,
        xn: &Buf,
        gate: &Buf,
        out: &Buf,
        w: &Buf,
        part: &Buf,
        tokens: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(xn, gate, out, w, part, tokens);
        self.pipe.run(&bufs, &pc, grid)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_timed(
        &self,
        xn: &Buf,
        gate: &Buf,
        out: &Buf,
        w: &Buf,
        part: &Buf,
        tokens: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if tokens == 0 {
            return Err(anyhow!("hc gated mean tokens must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(xn, gate, out, w, part, tokens);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

/// HC_WRITE: `out[c] = r[c] + block * 2*sigmoid(inject[c]/hc)`.
///
/// Layouts: `r`, `out` are `[n_embd, hc, T]`; `block` is `[n_embd, T]`;
/// `inject` is `[hc, T]`. `r` and `out` may be the same buffer (in-place
/// residual update).
pub struct HcWrite<'g> {
    pipe: Pipe<'g>,
    n_embd: usize,
    hc: usize,
}

impl<'g> HcWrite<'g> {
    pub fn new(gpu: &'g Gpu, n_embd: usize, hc: usize) -> Result<Self> {
        if n_embd == 0 || hc == 0 {
            return Err(anyhow!("hc write needs n_embd > 0 and hc > 0"));
        }
        if hc > 32 {
            return Err(anyhow!(
                "hc write caches the {hc} stream gates in shared memory (max 32)"
            ));
        }
        Ok(HcWrite {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/ops_hc_write.spv")),
                4,
                "hc write",
            )?,
            n_embd,
            hc,
        })
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
    pub fn hc(&self) -> usize {
        self.hc
    }

    fn launch<'a>(
        &'a self,
        r: &'a Buf,
        block: &'a Buf,
        inject: &'a Buf,
        out: &'a Buf,
        tokens: usize,
        chunks: usize,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let gx = self.n_embd.div_ceil(256).max(1) as u32;
        (
            vec![r, block, inject, out],
            Pc::new()
                .u(self.n_embd as u32)
                .u(self.hc as u32)
                .u(tokens as u32)
                .u(chunks as u32),
            (gx, tokens.max(1) as u32, 1),
        )
    }

    /// `chunks = 0` reads `inject` as `[T, hc]` final values; `chunks > 0` sums the
    /// `[T, hc, chunks]` partials that [`HcGatedMean`] wrote.
    pub fn run(
        &self,
        r: &Buf,
        block: &Buf,
        inject: &Buf,
        out: &Buf,
        tokens: usize,
        chunks: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(r, block, inject, out, tokens, chunks);
        self.pipe.run(&bufs, &pc, grid)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_timed(
        &self,
        r: &Buf,
        block: &Buf,
        inject: &Buf,
        out: &Buf,
        tokens: usize,
        chunks: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if tokens == 0 {
            return Err(anyhow!("hc write tokens must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(r, block, inject, out, tokens, chunks);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

/// The three HC projection weights of one mixer (`hc_attn_*` / `hc_ffn_*` /
/// `output_hc_*` in `docs/specs/qwen4exp-architecture.md` §6.1/§6.5).
pub struct HcWeights<'a> {
    /// `[n_embd, hc]` grouped RMS gamma (`hc_*_norm.weight`, `[10240]`).
    pub norm: &'a Buf,
    /// `[hc_lr, hc_dim]` = `hc_*_down.weight` (`[10240, 320]` ne order).
    pub down: &'a Buf,
    /// `[hc_dim, hc_lr]` = `hc_*_up.weight` (`[320, 10240]` ne order).
    pub up: &'a Buf,
    /// `[hc, hc_dim]` = `hc_*_inject.weight` (`[10240, 4]` ne order); `None`
    /// for the final mixer.
    pub inject: Option<&'a Buf>,
}

/// Per-batch HC scratch, allocated once per `T`.
pub struct HcWorkspace {
    pub tokens: usize,
    /// `[n_embd, hc, T]` the grouped RMSNorm output.
    pub xn: Buf,
    /// `[hc_lr, T]` the raw `w_down` output (pre-activation).
    pub lo_raw: Buf,
    /// `[hc_lr, T]` after `silu(x/hc)`.
    pub lo: Buf,
    /// `[hc_dim, T]` the `w_up` output (the per-stream gate logits).
    pub gate: Buf,
    /// `[n_embd, T]` the mixer output the sublayer consumes.
    pub mixed: Buf,
    /// `[T, hc, chunks]` partial `inject` dot products (`chunks = n_embd/256`,
    /// 1 when the mixer has no inject). [`HcWrite`] sums each token's `chunks`;
    /// [`HcMixer::inject_values`] does the same on the host for tests.
    pub inject: Buf,
    map_down: Buf,
    map_up: Buf,
}

impl HcWorkspace {
    /// Total device bytes this workspace holds (for the buffer budget).
    pub fn bytes(&self) -> u64 {
        [
            &self.xn,
            &self.lo_raw,
            &self.lo,
            &self.gate,
            &self.mixed,
            &self.inject,
            &self.map_down,
            &self.map_up,
        ]
        .iter()
        .map(|b| b.size)
        .sum()
    }
}

/// The hyper-connection mixer: HC_READ and HC_WRITE of
/// `docs/specs/qwen4exp-architecture.md` §4.1, implemented exactly as written.
///
/// ```text
/// HC_READ                                    R [hc, n_embd] per token
///   xn  = rmsnorm(R, eps) * w_norm            RMS over ONE stream, gamma [n_embd, hc]
///   lo  = silu( (xn @ w_down.T) / hc )        w_down [hc_dim, hc_lr] -> [hc_lr]
///   gate= lo @ w_up.T                          w_up   [hc_lr, hc_dim] -> [hc_dim]
///   gated = xn * sigmoid(gate)                 per stream
///   mixed = mean_over_streams(gated)           -> [n_embd]
///   inject= xn @ w_inject.T                    w_inject [hc_dim, hc] -> [hc]
/// HC_WRITE
///   w[c]     = 2 * sigmoid(inject[c] / hc)
///   R_out[c] = R[c] + block_out * w[c]
/// ```
///
/// The three projections are one [`crate::GemvBatch`] dispatch each, treating the
/// batch of tokens as the batch of "experts" (same weights, one activation row
/// group per token) — so `T = 1` is a decode GEMV and `T > 1` is the same
/// dispatch shape with `T` row groups. That is a *batched GEMV*, not a tiled
/// GEMM: the weights are re-read once per token, which is the measured prefill
/// cost in `docs/phase-3/04` and the reason a tiled HC GEMM is on the next list.
///
/// # Ambiguities this implementation pins (see `docs/phase-3/04` §open questions)
///
/// 1. **Gammas are used as stored.** §14 says "treat the stored value as the
///    multiplier"; whether the converter folds `(1 + w)` into `hc_*_norm`
///    (spec Q1) does not change this kernel, only which values are uploaded.
/// 2. **`lo` divides by `hc` before the SiLU** — `silu(x/hc)`, not `silu(x)/hc`.
/// 3. **`mixed` is a mean, not a sum** (`/hc`).
/// 4. **`w_inject` mixes the whole `[hc_dim]` stack**, i.e. `inject[c]` is a dot
///    product over all four streams, because the tensor is `[10240, 4]` in `ne`
///    order. The only reading consistent with that shape.
pub struct HcMixer<'g> {
    n_embd: usize,
    hc: usize,
    hc_lr: usize,
    norm: RmsNorm<'g>,
    act: Act<'g>,
    gated: HcGatedMean<'g>,
    write: HcWrite<'g>,
    down: crate::GemvBatch<'g>,
    up: crate::GemvBatch<'g>,
    /// `true` when the `inject` projection is folded into the gated-mean pass.
    fused_inject: bool,
    /// Bound as the gated-mean's inject weight when the mixer has no `inject`
    /// (every declared binding must be bound).
    no_weight: Buf,
}

impl<'g> HcMixer<'g> {
    /// Compile the pipelines of one mixer. `ty` is the projection type
    /// (`BF16` in the artifact: the `hc_*` weights are BF16, §14).
    ///
    /// With `with_inject` the fourth projection is **folded** into the gated-mean
    /// dispatch instead of being a 4-row GEMV: one dispatch fewer per HC_READ, and
    /// the 82 KiB of `w_inject` is read once and coalesced alongside `xn` instead
    /// of by a single under-filled workgroup (measured 21.85 us -> ~1 us on the RX,
    /// `docs/phase-3/06` §HC). Requires `n_embd % 256 == 0` and a BF16/F32 weight.
    pub fn new(
        gpu: &'g Gpu,
        ty: QuantType,
        n_embd: usize,
        hc: usize,
        hc_lr: usize,
        with_inject: bool,
    ) -> Result<Self> {
        if n_embd == 0 || hc == 0 || hc_lr == 0 {
            return Err(anyhow!("hc mixer needs n_embd, hc, hc_lr > 0"));
        }
        if !hc_lr.is_multiple_of(8) {
            return Err(anyhow!(
                "hc_lr must be a multiple of 8 (the batched GEMV's per-token activation stride is in vec4 units), got {hc_lr}"
            ));
        }
        let norm = RmsNorm::new(gpu, n_embd, hc)?;
        let act = Act::new(gpu, ActMode::SiluDiv)?;
        let gated = if with_inject {
            let chunks = n_embd / 256;
            HcGatedMean::with_inject(gpu, n_embd, hc, Some((ty, chunks)))?
        } else {
            HcGatedMean::new(gpu, n_embd, hc)?
        };
        let write = HcWrite::new(gpu, n_embd, hc)?;
        let down = crate::GemvBatch::new(gpu, ty, hc_lr, hc * n_embd)?;
        let up = crate::GemvBatch::new(gpu, ty, hc * n_embd, hc_lr)?;
        let no_weight = empty_buf(gpu, "hc inject weight")?;
        Ok(HcMixer {
            n_embd,
            hc,
            hc_lr,
            norm,
            act,
            gated,
            write,
            down,
            up,
            fused_inject: with_inject,
            no_weight,
        })
    }

    /// The artifact's trunk mixer: `n_embd = 2560`, `hc = 4`, `hc_lr = 320`.
    pub fn model(gpu: &'g Gpu, ty: QuantType, with_inject: bool) -> Result<Self> {
        Self::new(gpu, ty, 2560, MODEL_HC, MODEL_HC_LR, with_inject)
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
    pub fn hc(&self) -> usize {
        self.hc
    }
    pub fn hc_lr(&self) -> usize {
        self.hc_lr
    }
    pub fn has_inject(&self) -> bool {
        self.fused_inject
    }
    /// Partial sums per (token, stream) the folded `inject` writes.
    pub fn inject_chunks(&self) -> usize {
        self.gated.chunks()
    }
    /// The grouped norm stage (exposed for the bench's stage split).
    pub fn norm(&self) -> &RmsNorm<'g> {
        &self.norm
    }
    /// The bottleneck activation stage.
    pub fn act(&self) -> &Act<'g> {
        &self.act
    }
    /// The gated-mean stage.
    pub fn gated_mean(&self) -> &HcGatedMean<'g> {
        &self.gated
    }
    /// The gated-combine stage (HC_WRITE).
    pub fn combine(&self) -> &HcWrite<'g> {
        &self.write
    }

    /// Allocate the scratch for `tokens` tokens, including the two row maps.
    ///
    /// The maps are constant for a given `(weights, tokens)`: all "experts" in a
    /// batched projection share the same weight blob (base 0) and differ only in
    /// their per-token activation base, so the map is built once here and not
    /// per token. The `inject` half keeps its two buffers (a 4-element weight
    /// placeholder for the disabled case and `tokens*hc*chunks` partials) so the
    /// gated-mean dispatch always has all five bindings.
    pub fn workspace(&self, gpu: &Gpu, tokens: usize) -> Result<HcWorkspace> {
        if tokens == 0 {
            return Err(anyhow!("hc workspace needs tokens >= 1"));
        }
        let hc_dim = self.hc * self.n_embd;
        let blobs = vec![0u32; tokens];
        let chunks = self.inject_chunks();
        let maps = [
            self.down.map_for_blobs(&blobs, crate::XLayout::PerExpert),
            self.up.map_for_blobs(&blobs, crate::XLayout::PerExpert),
        ];
        let inject_vals = (tokens * self.hc * chunks.max(1)).max(4);
        // Six activation buffers + three maps, in one flat list so a failure at
        // any point frees exactly what was allocated.
        let sizes = [
            hc_dim * tokens,      // xn
            self.hc_lr * tokens,  // lo_raw
            self.hc_lr * tokens,  // lo
            hc_dim * tokens,      // gate
            self.n_embd * tokens, // mixed
            inject_vals,          // inject partials [T, hc, chunks]
            maps[0].len(),        // map_down
            maps[1].len(),        // map_up
        ];
        let mut bufs: Vec<Buf> = Vec::with_capacity(sizes.len());
        let cleanup = |bufs: &[Buf]| {
            for b in bufs {
                gpu.destroy_buf(b);
            }
        };
        for (i, n) in sizes.iter().enumerate() {
            match alloc(gpu, ((n * 4).max(8)) as u64) {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    cleanup(&bufs);
                    return Err(e.context(format!("allocating hc workspace slot {i}")));
                }
            }
        }
        for (buf, map) in bufs[6..8].iter().zip(maps.iter()) {
            if map.is_empty() {
                continue;
            }
            if let Err(e) = crate::gemv::upload_into(gpu, buf, &u32_to_bytes(map)) {
                cleanup(&bufs);
                return Err(e.context("uploading an hc row map"));
            }
        }
        let mut it = bufs.into_iter();
        let mut take = || it.next().expect("eight hc buffers were allocated");
        Ok(HcWorkspace {
            tokens,
            xn: take(),
            lo_raw: take(),
            lo: take(),
            gate: take(),
            mixed: take(),
            inject: take(),
            map_down: take(),
            map_up: take(),
        })
    }

    /// Release a workspace.
    pub fn release(&self, gpu: &Gpu, ws: HcWorkspace) {
        for b in [
            &ws.xn,
            &ws.lo_raw,
            &ws.lo,
            &ws.gate,
            &ws.mixed,
            &ws.inject,
            &ws.map_down,
            &ws.map_up,
        ] {
            gpu.destroy_buf(b);
        }
    }

    /// The `inject` scalars for each token, summed across the partials the
    /// gated-mean dispatch wrote (`[T, hc]`). The device sums them inside
    /// [`HcWrite`]; this is the same arithmetic on the host, for tests and for a
    /// debug dump.
    pub fn inject_values(&self, gpu: &Gpu, ws: &HcWorkspace) -> Result<Vec<f32>> {
        let chunks = self.inject_chunks().max(1);
        let raw = download_f32(gpu, &ws.inject, ws.tokens * self.hc * chunks)?;
        let mut out = vec![0.0f32; ws.tokens * self.hc];
        for t in 0..ws.tokens {
            for c in 0..self.hc {
                let base = (t * self.hc + c) * chunks;
                out[t * self.hc + c] = raw[base..base + chunks].iter().sum();
            }
        }
        Ok(out)
    }

    /// HC_READ for `ws.tokens` tokens: fills `ws.xn`, `ws.mixed` and (when the
    /// mixer has one) `ws.inject`. `r` is `[n_embd, hc, T]`.
    ///
    /// Five dispatches with the fold (norm, down, act, up, gated+inject) instead
    /// of six: the projection that used to be a 4-row GEMV is inside the last one.
    pub fn read(&self, ws: &HcWorkspace, r: &Buf, w: &HcWeights<'_>) -> Result<()> {
        let t = ws.tokens;
        self.norm.run(r, Some(w.norm), &ws.xn, self.hc * t)?;
        self.down.run(w.down, &ws.xn, &ws.lo_raw, &ws.map_down, t)?;
        self.act
            .run(&ws.lo_raw, &ws.lo, self.hc_lr * t, self.hc as f32)?;
        self.up.run(w.up, &ws.lo, &ws.gate, &ws.map_up, t)?;
        let winj = w.inject.unwrap_or(&self.no_weight);
        self.gated
            .run(&ws.xn, &ws.gate, &ws.mixed, winj, &ws.inject, t)?;
        Ok(())
    }

    /// Time each HC_READ stage on its own: `[norm, down, act, up, gated+inject]`.
    ///
    /// The chain total says how much a layer's mixer costs; this says *which*
    /// stage it is. With the fold there is no separate `inject` stage any more —
    /// its cost is inside `gated` (`docs/phase-3/06` §HC).
    pub fn stage_timings(
        &self,
        ws: &HcWorkspace,
        r: &Buf,
        w: &HcWeights<'_>,
        target_s: f64,
    ) -> Result<[GemvTiming; 5]> {
        let t = ws.tokens;
        let norm = self
            .norm
            .run_timed(r, Some(w.norm), &ws.xn, self.hc * t, target_s)?;
        let down = self
            .down
            .run_timed(w.down, &ws.xn, &ws.lo_raw, &ws.map_down, t, target_s)?;
        let act =
            self.act
                .run_timed(&ws.lo_raw, &ws.lo, self.hc_lr * t, self.hc as f32, target_s)?;
        let up = self
            .up
            .run_timed(w.up, &ws.lo, &ws.gate, &ws.map_up, t, target_s)?;
        let winj = w.inject.unwrap_or(&self.no_weight);
        let gated = self
            .gated
            .run_timed(&ws.xn, &ws.gate, &ws.mixed, winj, &ws.inject, t, target_s)?;
        Ok([norm, down, act, up, gated])
    }

    /// Names of [`HcMixer::stage_timings`]'s stages, in order.
    pub const STAGES: [&'static str; 5] = ["norm", "down", "act", "up", "gated"];

    /// HC_WRITE: `out[c] = r[c] + block * 2*sigmoid(inject[c]/hc)` for
    /// `ws.tokens` tokens. Reads `ws.inject` from the last [`HcMixer::read`].
    pub fn write(&self, ws: &HcWorkspace, r: &Buf, block: &Buf, out: &Buf) -> Result<()> {
        self.write
            .run(r, block, &ws.inject, out, ws.tokens, self.inject_chunks())
    }

    /// Time a whole HC_READ (`ws.tokens` tokens) as one back-to-back sequence:
    /// norm, down, activation, up, gated mean and (when present) inject, with a
    /// full barrier between stages and the queue timestamps around the whole
    /// chain. That is the latency the graph will pay for one HC_READ.
    pub fn read_timed(
        &self,
        ws: &HcWorkspace,
        r: &Buf,
        w: &HcWeights<'_>,
        target_s: f64,
    ) -> Result<GemvTiming> {
        let t = ws.tokens;
        let gpu = self.norm.pipe.gpu;
        // Warm the whole chain once (discarded), then time it.
        self.read(ws, r, w)?;

        // Bind every stage's descriptor set once; the recording below only
        // re-issues the dispatches.
        self.norm.pipe.bind(&[r, w.norm, &ws.xn])?;
        self.down.bind(w.down, &ws.xn, &ws.lo_raw, &ws.map_down)?;
        self.act.pipe.bind(&[&ws.lo_raw, &ws.lo])?;
        self.up.bind(w.up, &ws.lo, &ws.gate, &ws.map_up)?;
        let winj = w.inject.unwrap_or(&self.no_weight);
        self.gated
            .pipe
            .bind(&[&ws.xn, &ws.gate, &ws.mixed, winj, &ws.inject])?;

        let norm_pc = Pc::new()
            .u((self.hc * t) as u32)
            .u(self.n_embd as u32)
            .u(self.hc as u32)
            .f(self.norm.eps());
        let act_pc = Pc::new()
            .u((self.hc_lr * t) as u32)
            .u(ActMode::SiluDiv.code())
            .f(self.hc as f32);
        let gated_pc = Pc::new()
            .u(self.n_embd as u32)
            .u(self.hc as u32)
            .u(t as u32)
            .u(self.gated.dt)
            .u(self.inject_chunks() as u32);
        let (ggx, ggy) = grid_for(self.n_embd * t, 256);
        let (agx, agy) = grid_for(self.hc_lr * t, 256);

        let timing = gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                for _ in 0..n {
                    record(
                        d,
                        cmd,
                        &self.norm.pipe.kernel,
                        norm_pc.as_bytes(),
                        ((self.hc * t) as u32, 1, 1),
                    );
                    unsafe { Gpu::barrier_all(d, cmd) };
                    self.down.dispatch(d, cmd, self.down.rows(t));
                    unsafe { Gpu::barrier_all(d, cmd) };
                    record(
                        d,
                        cmd,
                        &self.act.pipe.kernel,
                        act_pc.as_bytes(),
                        (agx, agy, 1),
                    );
                    unsafe { Gpu::barrier_all(d, cmd) };
                    self.up.dispatch(d, cmd, self.up.rows(t));
                    unsafe { Gpu::barrier_all(d, cmd) };
                    record(
                        d,
                        cmd,
                        &self.gated.pipe.kernel,
                        gated_pc.as_bytes(),
                        (ggx, ggy, 1),
                    );
                    unsafe { Gpu::barrier_all(d, cmd) };
                }
            })
            .map_err(|e| anyhow!("hc read chain: {e}"))?;
        Ok(GemvTiming {
            per_iter_s: timing.per_iter_s,
            iters: timing.iters,
            total_s: timing.total_s,
            used_timestamps: timing.used_timestamps,
        })
    }
}

impl Drop for HcMixer<'_> {
    fn drop(&mut self) {
        // The pipelines free their own kernels through `Pipe::drop`; this is the
        // only buffer the mixer owns.
        self.norm.pipe.gpu.destroy_buf(&self.no_weight);
    }
}

// ---------------------------------------------------------------------------
// 8. Residual add / scale
// ---------------------------------------------------------------------------

/// `out = a + beta * b` (one dispatch over `n` values).
///
/// `beta = 1` is the plain residual add; `out` may alias `a` or `b`.
pub struct Add<'g> {
    pipe: Pipe<'g>,
}

impl<'g> Add<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(Add {
            pipe: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/ops_add.spv")),
                3,
                "add",
            )?,
        })
    }

    fn launch<'a>(
        &'a self,
        a: &'a Buf,
        b: &'a Buf,
        out: &'a Buf,
        n: usize,
        beta: f32,
    ) -> (Vec<&'a Buf>, Pc, (u32, u32, u32)) {
        let (gx, gy) = grid_for(n, 256);
        (vec![a, b, out], Pc::new().u(n as u32).f(beta), (gx, gy, 1))
    }

    pub fn run(&self, a: &Buf, b: &Buf, out: &Buf, n: usize, beta: f32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let (bufs, pc, grid) = self.launch(a, b, out, n, beta);
        self.pipe.run(&bufs, &pc, grid)
    }

    pub fn run_timed(
        &self,
        a: &Buf,
        b: &Buf,
        out: &Buf,
        n: usize,
        beta: f32,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if n == 0 {
            return Err(anyhow!("add n must be non-zero"));
        }
        let (bufs, pc, grid) = self.launch(a, b, out, n, beta);
        self.pipe.run_timed(&bufs, &pc, grid, target_s)
    }
}

// ---------------------------------------------------------------------------
// Embedded SPIR-V
// ---------------------------------------------------------------------------

fn gather_spv(ty: QuantType) -> &'static [u8] {
    macro_rules! spv {
        ($stem:literal) => {
            include_bytes!(concat!(env!("OUT_DIR"), "/", $stem, ".spv"))
        };
    }
    match ty {
        QuantType::Iq2Xxs => spv!("ops_gather_iq2_xxs"),
        QuantType::Iq2Xs => spv!("ops_gather_iq2_xs"),
        QuantType::Iq2S => spv!("ops_gather_iq2_s"),
        QuantType::Iq3Xxs => spv!("ops_gather_iq3_xxs"),
        QuantType::Iq3S => spv!("ops_gather_iq3_s"),
        QuantType::Iq4Nl => spv!("ops_gather_iq4_nl"),
        QuantType::Iq4Xs => spv!("ops_gather_iq4_xs"),
        QuantType::Q4K => spv!("ops_gather_q4_k"),
        QuantType::Q5K => spv!("ops_gather_q5_k"),
        QuantType::Q6K => spv!("ops_gather_q6_k"),
        QuantType::Q2_0 => spv!("ops_gather_q2_0"),
        QuantType::F16 => spv!("ops_gather_f16"),
        QuantType::Bf16 => spv!("ops_gather_bf16"),
        QuantType::F32 => spv!("ops_gather_f32"),
    }
}

/// Keep `TAIL_PAD` referenced from this module: every `alloc` here inherits it,
/// and the gather is the op that most needs it (a `vec4`/`u16` tail read one
/// element past the last row must land in mapped memory —
/// `docs/phase-3/02-expert-batching-quants.md` §4.2).
const _: u64 = TAIL_PAD;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every quant type with a decode unit must have a gather SPIR-V module, and
    /// `build.rs` must be building it: a missing entry would only show up as a
    /// link error inside the `include_bytes!` above, which is a poor error
    /// message for "you added a quant type".
    #[test]
    fn every_quant_type_has_a_gather_shader() {
        let build = include_str!("../../build.rs");
        for ty in QuantType::ALL {
            let stem = format!("ops_gather_{}", ty.shader_stem());
            assert!(
                build.contains(&format!("\"{stem}\"")),
                "build.rs does not compile shaders/{stem}.comp"
            );
            assert!(
                std::path::Path::new(&format!("shaders/{stem}.comp")).exists(),
                "shaders/{stem}.comp is missing"
            );
        }
    }

    #[test]
    fn grid_wraps_over_y_instead_of_exceeding_the_guaranteed_x_limit() {
        assert_eq!(grid_for(0, 64), (1, 1));
        assert_eq!(grid_for(64, 64), (1, 1));
        assert_eq!(grid_for(65, 64), (2, 1));
        // 65536 groups -> x = 65535, y = 2
        assert_eq!(grid_for(65536 * 64, 64), (65535, 2));
    }

    #[test]
    fn push_constants_are_word_packed_little_endian() {
        let pc = Pc::new().u(1).f(0.5).u(-1i32 as u32);
        assert_eq!(pc.as_bytes().len(), 12);
        assert_eq!(&pc.as_bytes()[0..4], &1u32.to_le_bytes());
        assert_eq!(&pc.as_bytes()[4..8], &0.5f32.to_bits().to_le_bytes());
        assert_eq!(&pc.as_bytes()[8..12], &(-1i32 as u32).to_le_bytes());
    }

    #[test]
    fn rope_table_matches_the_strata_formula() {
        // bit-for-bit against a direct transcription of
        // `build_rope_table` (src/kernels/cuda/rope.cu:49-59).
        let t = RopeTable::build(1e7, 64, 8).expect("table");
        assert_eq!(t.pairs(), 32);
        assert_eq!(t.cos_sin().len(), 8 * 32 * 2);
        for p in 0..8usize {
            for i in 0..32usize {
                let inv = 1e7f64.powf(-2.0 * i as f64 / 64.0);
                let ang = p as f64 * inv;
                let o = (p * 32 + i) * 2;
                assert_eq!(t.cos_sin()[o], ang.cos() as f32, "cos p={p} i={i}");
                assert_eq!(t.cos_sin()[o + 1], ang.sin() as f32, "sin p={p} i={i}");
            }
        }
        assert!(RopeTable::build(1e7, 63, 8).is_err(), "odd n_rot");
        assert!(RopeTable::build(1e7, 64, 0).is_err(), "empty table");
    }

    #[test]
    fn router_geometry_is_checked() {
        // The kernel's two hard limits, checked at compile time so a model change
        // cannot silently exceed them.
        assert_eq!(ROUTER_CLAMP, 2f32.powi(-14));
        const { assert!(MODEL_TOP_K <= ROUTER_MAX_K) };
        const { assert!(MODEL_N_EXPERT <= 512) };
    }
}
