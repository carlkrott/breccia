//! `breccia bench gemv` — decode-path throughput on real kernels.
//!
//! Two modes:
//!
//! * **synthetic** (default): seeded random *valid* packed blocks for a given
//!   `{type, rows, cols}`, uploaded once, then `run_timed` repeats the fused
//!   dequant+GEMV until the measurement window is long enough. Bytes-read is
//!   exactly `rows * ceil(cols/block_elems) * block_bytes`, i.e. the weight
//!   traffic of one decode step for that matrix.
//! * **real** ([`crate::real`]): the same kernel, fed straight from a real
//!   expert tensor in the 75 GB GGUF mapping (no copy of the model; the
//!   benchmarked tensor is paged in and uploaded once).
//!
//! Reported per cell: achieved GB/s, the same number as a fraction of the
//! measured DRAM bandwidth of that card (Phase-0: RX 6950 XT 445 GB/s,
//! MI50/Radeon VII 825 GB/s — `docs/phase-0/02-vulkan-raw-probe.md`), and the
//! "tokens/s if this matrix were the whole token" equivalent.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::batch::{map_to_bytes, GemvBatch, XLayout};
use crate::device::Device;
use crate::gemv::{alloc, download, upload, Accum, Gemv, GemvTiming};
use crate::quant::QuantType;

/// Measured DRAM read bandwidth per device id (vendor:device), from the
/// Phase-0.2 raw probe (`docs/phase-0/02`: RX 462 GB/s, MI50 826 GB/s; the
/// figures in the Phase-3 work order are 445 / 825).
fn reference_bandwidth_gbps(device_id: &str) -> Option<(f64, &'static str)> {
    match device_id {
        "1002:73a5" => Some((445.0, "RX 6950 XT (measured 462 GB/s Phase-0 §2)")),
        "1002:66af" => Some((825.0, "Radeon VII / MI50 (measured 826 GB/s Phase-0 §2)")),
        _ => None,
    }
}

/// One shape to benchmark.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.rows, self.cols)
    }
}

/// Benchmark options (`breccia bench gemv` CLI surface).
#[derive(Clone, Debug, Default)]
pub struct GemvBenchOptions {
    /// Types to run (empty = the model-weighted default set).
    pub types: Vec<QuantType>,
    /// Explicit shapes (empty = [`default_shapes`]).
    pub shapes: Vec<Shape>,
    /// Device by PCI address / enumeration index (empty = every discrete GPU).
    pub devices: Vec<String>,
    /// Shorter windows and fewer cells.
    pub quick: bool,
    /// Real-model mode: run against a real expert tensor instead of random blocks.
    pub real: bool,
    /// GGUF glob for real mode.
    pub real_model: Option<String>,
    /// Block (layer) index for real mode.
    pub real_layer: usize,
    /// Role for real mode: `gate` | `up` | `down`.
    pub real_role: String,
    /// Only expert `e` of that role tensor (single-expert shape), for conformance.
    pub real_expert: Option<usize>,
    /// Force a split-K factor (default: [`Gemv::recommended_splits`], i.e. 1).
    pub splits: Option<u32>,
    /// Real mode on an arbitrary tensor name instead of an expert role
    /// (e.g. `output.weight` for Q5_K, `token_embd.weight` for IQ3_S).
    pub real_tensor: Option<String>,
    /// Batched multi-expert mode: run the whole expert set in one dispatch for
    /// each requested expert count (empty = off). Shapes come from
    /// `--rows/--cols` or default to the model's 640x2560 routed-expert shape.
    pub experts: Vec<usize>,
    /// Accumulator precision of the fused dot product (default fp32).
    pub accum: Accum,
}

/// The shapes that matter for one decode step of this model
/// (`docs/specs/qwen4exp-architecture.md` §6):
/// routed-expert `[640,2560]` + `[2560,640]`, shared expert, `attn_qkv`
/// `[10240,2560]`, `ssm_out` `[2560,6144]`, `output` `[248320,2560]`,
/// and the embedding-width square as a scaling control.
pub fn default_shapes(quick: bool) -> Vec<Shape> {
    if quick {
        vec![Shape::new(640, 2560), Shape::new(2560, 640)]
    } else {
        vec![
            Shape::new(640, 2560),
            Shape::new(2560, 640),
            // the same expert repeated 10x = one layer's top-10 routed experts
            Shape::new(6400, 2560),
            Shape::new(10240, 2560),
            Shape::new(2560, 6144),
            Shape::new(2560, 2560),
            Shape::new(248320, 2560),
        ]
    }
}

/// Types to sweep when the caller does not name one, ordered by weight in the
/// model (expert tensors dominate: Q2_0 down 30 layers, IQ3_S 26, IQ2_XS 20 …).
pub fn default_types() -> Vec<QuantType> {
    vec![
        QuantType::Iq3S,
        QuantType::Q2_0,
        QuantType::Iq2Xxs,
        QuantType::Iq2Xs,
        QuantType::Iq2S,
        QuantType::Iq3Xxs,
        QuantType::Iq4Nl,
        QuantType::Iq4Xs,
        QuantType::Q4K,
        QuantType::Q5K,
        QuantType::Q6K,
        QuantType::Bf16,
        QuantType::F32,
    ]
}

/// The routed-expert shape of one expert (`docs/specs/qwen4exp-architecture.md`
/// §5.4: `n_embd = 2560`, `n_ff = 640`), used by batched mode unless the caller
/// passes `--rows/--cols`.
pub fn default_expert_shape() -> Shape {
    Shape::new(640, 2560)
}

/// Expert counts batched mode sweeps when the caller names none.
pub fn default_expert_counts(quick: bool) -> Vec<usize> {
    if quick {
        vec![1, 10]
    } else {
        vec![1, 2, 4, 8, 10, 16]
    }
}

// ---------------------------------------------------------------------------
// Report schema
// ---------------------------------------------------------------------------

pub const GEMV_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GemvReport {
    pub schema_version: u32,
    pub environment: Environment,
    pub device: crate::device::DeviceInfo,
    pub quick: bool,
    pub synthetic: Vec<Cell>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub real: Vec<RealCell>,
    /// Batched multi-expert cells (one dispatch over N experts).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch: Vec<BatchCell>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Environment {
    pub hostname: String,
    pub timestamp: String,
    pub instance_layers: Vec<String>,
    pub vulkan_validation: bool,
    pub device_open_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub ty: String,
    pub quant_id: u32,
    pub shape: Shape,
    pub bits_per_value: f64,
    pub weight_bytes: u64,
    /// Accumulator precision used (`f32` unless `--accum f16`).
    #[serde(default)]
    pub accum: String,
    /// Max |GPU - CPU reference| / max(1, |CPU|) over the first rows (a slice:
    /// 4 rows are enough to show the fp16 error, which is uniform across rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conformance_max_rel: Option<f64>,
    /// GPU seconds per GEMV (best of the measured windows).
    pub seconds_per_iter: f64,
    /// Best window, and every window, for spread checking.
    pub seconds_per_iter_all: Vec<f64>,
    pub iters: u32,
    /// Split-K factor used (1 = single pass).
    pub splits: u32,
    pub gbps: f64,
    pub reference_gbps: Option<f64>,
    pub reference_note: Option<String>,
    pub percent_of_reference: Option<f64>,
    pub tokens_per_s_equivalent: f64,
    pub used_timestamps: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RealCell {
    pub tensor: String,
    pub ty: String,
    pub quant_id: u32,
    pub shape: Shape,
    pub experts: usize,
    pub expert: Option<usize>,
    pub weight_bytes: u64,
    /// Accumulator precision used (`f32` unless `--accum f16`).
    #[serde(default)]
    pub accum: String,
    /// Absolute GGUF file byte range of the rows actually benchmarked (the
    /// `--expert` slice, or the whole tensor).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_range: Option<[u64; 2]>,
    /// True when the tensor's rows were folded to a whole kernel block
    /// (`real::usable_shape`), e.g. F16 `ple_conv1d` `[4, 10240]` -> 5120x8.
    #[serde(default)]
    pub folded_rows: bool,
    pub seconds_per_iter: f64,
    pub seconds_per_iter_all: Vec<f64>,
    pub iters: u32,
    pub splits: u32,
    pub gbps: f64,
    pub reference_gbps: Option<f64>,
    pub percent_of_reference: Option<f64>,
    pub tokens_per_s_equivalent: f64,
    pub used_timestamps: bool,
    /// Max |GPU - CPU reference| / max(1, |CPU|) over the checked rows.
    pub conformance_max_rel: Option<f64>,
    pub conformance_rows: usize,
    /// Wall time spent paging the tensor in + uploading it (one-off, not part of GB/s).
    pub upload_seconds: f64,
}

/// One batched multi-expert cell: the same expert set run as a per-expert loop
/// and as one [`crate::GemvBatch`] dispatch, in the same process/run, so the
/// before/after is apples to apples.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchCell {
    pub ty: String,
    pub quant_id: u32,
    /// Accumulator precision used (`f32` unless `--accum f16`).
    #[serde(default)]
    pub accum: String,
    /// Shape of **one** expert (`rows_per_expert x cols`).
    pub shape: Shape,
    pub experts: usize,
    /// Packed weight bytes per expert, and the whole set.
    pub blob_bytes: u64,
    pub total_bytes: u64,
    /// `shared` (one x vector for all experts, MoE gate/up) or `per-expert`
    /// (one per expert, MoE down).
    pub x_layout: String,
    /// Per-expert-loop baseline: `experts` single-expert dispatches, no split-K.
    pub loop_seconds: f64,
    pub loop_seconds_all: Vec<f64>,
    /// Same loop with the per-expert split-K factor the single-expert API
    /// recommends (`Gemv::recommended_splits`), when it is > 1.
    pub loop_split_seconds: Option<f64>,
    pub loop_split_splits: u32,
    /// One batched dispatch covering all `experts`.
    pub batched_seconds: f64,
    pub batched_seconds_all: Vec<f64>,
    /// `loop_seconds / batched_seconds`.
    pub speedup: f64,
    /// Per-expert latency of one single-expert dispatch (context for the
    /// latency-bound regime).
    pub single_expert_seconds: f64,
    pub loop_gbps: f64,
    pub batched_gbps: f64,
    pub reference_gbps: Option<f64>,
    pub percent_of_reference: Option<f64>,
    pub tokens_per_s_equivalent: f64,
    pub iters: u32,
    pub used_timestamps: bool,
    /// Real-model mode: the tensor and which experts were batched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_ids: Option<Vec<usize>>,
    /// Real-model mode: max |GPU - CPU reference| / max(1, |CPU|) on real bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conformance_max_rel: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conformance_rows: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upload_seconds: Option<f64>,
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// PCI addresses of every discrete GPU (enumeration indices when a device has
/// no PCI info) — the default device set for `breccia bench gemv`, resolved
/// before any benchmark opens a device.
pub fn discrete_devices() -> Result<Vec<String>> {
    let host = breccia_vk::Host::new().map_err(|e| anyhow!("{e}"))?;
    let devs = host.devices().map_err(|e| anyhow!("{e}"))?;
    let keys: Vec<String> = devs
        .iter()
        .filter(|d| d.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU)
        .map(|d| {
            if d.pci_address.is_empty() {
                d.index.to_string()
            } else {
                d.pci_address.clone()
            }
        })
        .collect();
    host.destroy();
    if keys.is_empty() {
        return Err(anyhow!(
            "no discrete GPU enumerated (is MESA_VK_DEVICE_SELECT hiding the GPUs?)"
        ));
    }
    Ok(keys)
}

/// Devices to run on when the caller names none.
fn default_device_selectors() -> Vec<String> {
    // Empty selector = "first discrete GPU" per crate::Device::open. The CLI
    // expands this to every discrete GPU by PCI address when it wants both.
    Vec::new()
}

pub fn run_gemv_bench(opts: &GemvBenchOptions) -> Result<GemvReport> {
    let selectors = if opts.devices.is_empty() {
        default_device_selectors()
    } else {
        opts.devices.clone()
    };
    let selector = selectors.first().map(|s| s.as_str());

    let mut warnings = Vec::new();
    if selectors.len() > 1 {
        warnings.push(format!(
            "run_gemv_bench runs one device per call; using {}",
            selectors[0]
        ));
    }

    let dev = Device::open(selector)?;
    if !dev.gpu().feats.storage_buffer16_bit_access {
        return Err(anyhow!(
            "{} does not support 16-bit storage in shader buffers, which every \
             quant kernel here requires",
            dev.info.name
        ));
    }

    let reference = match reference_bandwidth_gbps(&dev.info.device_id) {
        Some((bw, note)) => ReferenceBandwidth {
            gbps: Some(bw),
            note: Some(note.to_string()),
        },
        None => {
            warnings.push(format!(
                "no measured DRAM bandwidth on file for {} ({}); percentages omitted",
                dev.info.name, dev.info.device_id
            ));
            ReferenceBandwidth {
                gbps: None,
                note: None,
            }
        }
    };

    // Clock ramp: the RX idles at low DPM states and the first cells of a run
    // then read up to 1.6x slow (docs/PROJECT-SETUP.md hazard list). Burn a
    // bandwidth-bound F32 GEMV for a moment before measuring anything.
    if let Err(e) = clock_warmup(&dev, if opts.quick { 0.05 } else { 0.2 }) {
        report_warmup_failure(e);
    }

    let mut report = GemvReport {
        schema_version: GEMV_SCHEMA_VERSION,
        environment: Environment {
            hostname: hostname(),
            timestamp: utc_timestamp(),
            instance_layers: instance_layers(),
            vulkan_validation: std::env::var("VK_INSTANCE_LAYERS")
                .map(|v| v.contains("VK_LAYER_KHRONOS_validation"))
                .unwrap_or(false),
            device_open_seconds: dev.open_seconds,
        },
        device: dev.info.clone(),
        quick: opts.quick,
        synthetic: Vec::new(),
        real: Vec::new(),
        batch: Vec::new(),
        warnings,
    };

    if opts.real {
        if opts.experts.is_empty() {
            report.real = crate::real::run(&dev, opts)?;
        } else {
            report.batch = crate::real::run_batch(&dev, opts)?;
        }
        return Ok(report);
    }

    if !opts.experts.is_empty() {
        report.batch = run_batch_cells(&dev, opts, &reference)?;
        return Ok(report);
    }

    let types = if opts.types.is_empty() {
        default_types()
    } else {
        opts.types.clone()
    };
    let shapes = if opts.shapes.is_empty() {
        default_shapes(opts.quick)
    } else {
        opts.shapes.clone()
    };
    let window = if opts.quick { 0.05 } else { 0.25 };
    let repeats = if opts.quick { 1 } else { 3 };

    for ty in types {
        for shape in &shapes {
            let cell = run_cell(
                &dev,
                ty,
                *shape,
                window,
                repeats,
                opts.splits,
                opts.accum,
                &reference,
            )
            .with_context(|| format!("{ty} {shape}"))?;
            report.synthetic.push(cell);
        }
    }
    Ok(report)
}

fn report_warmup_failure(e: anyhow::Error) {
    eprintln!("gemv bench: clock warm-up failed (numbers may be clock-limited): {e:#}");
}

/// Bandwidth-bound F32 GEMV on a ~24 MiB matrix, repeated for `secs`.
pub(crate) fn clock_warmup(dev: &Device, secs: f64) -> Result<()> {
    let gpu = dev.gpu();
    let shape = Shape::new(2560, 3072);
    let gemv = Gemv::new(gpu, QuantType::F32, shape.rows, shape.cols)?;
    let (packed, x) = crate::real::synthetic_inputs(QuantType::F32, shape, 1);
    let w = upload(gpu, &packed)?;
    let xb = upload(gpu, &f32_to_bytes(&x))?;
    let out = alloc(gpu, (shape.rows * 4) as u64)?;
    gemv.run(&w, &xb, &out)?;
    let t = gemv.run_timed(&w, &xb, &out, secs)?;
    for b in [&w, &xb, &out] {
        gpu.destroy_buf(b);
    }
    if t.iters == 0 {
        return Err(anyhow!("warm-up ran zero iterations"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Batched multi-expert mode
// ---------------------------------------------------------------------------

/// Synthetic batched sweep: for each type and each expert count, the same
/// expert set is run as a per-expert loop and as one batched dispatch.
fn run_batch_cells(
    dev: &Device,
    opts: &GemvBenchOptions,
    reference: &ReferenceBandwidth,
) -> Result<Vec<BatchCell>> {
    let types = if opts.types.is_empty() {
        default_types()
    } else {
        opts.types.clone()
    };
    let shape = opts
        .shapes
        .first()
        .copied()
        .unwrap_or_else(default_expert_shape);
    let counts = if opts.experts.is_empty() {
        default_expert_counts(opts.quick)
    } else {
        opts.experts.clone()
    };
    let window = if opts.quick { 0.05 } else { 0.25 };
    let repeats = if opts.quick { 1 } else { 3 };

    let mut cells = Vec::new();
    for ty in types {
        for &experts in &counts {
            if experts == 0 {
                return Err(anyhow!("--experts needs at least 1"));
            }
            for layout in [XLayout::Shared, XLayout::PerExpert] {
                cells.push(
                    run_batch_cell(
                        dev, ty, shape, experts, layout, window, repeats, opts.accum, reference,
                    )
                    .with_context(|| format!("{ty} {shape} x{experts} ({layout:?})"))?,
                );
            }
        }
    }
    Ok(cells)
}

/// One batched cell: upload `experts` synthetic expert blobs into one buffer,
/// then time (a) `experts` single-expert dispatches and (b) one batched
/// dispatch over the whole set.
///
/// The loop baseline re-dispatches the first blob instead of walking all of
/// them: at this shape the whole set is 7 MB of weights (10 x 640x2560 IQ3_S),
/// i.e. Infinity-Cache resident like the single blob, so both variants read the
/// same bytes from DRAM and the difference the cell measures is dispatch
/// latency and grid occupancy — which is the effect under test.
#[allow(clippy::too_many_arguments)]
fn run_batch_cell(
    dev: &Device,
    ty: QuantType,
    shape: Shape,
    experts: usize,
    layout: XLayout,
    window: f64,
    repeats: usize,
    accum: Accum,
    reference: &ReferenceBandwidth,
) -> Result<BatchCell> {
    let gpu = dev.gpu();
    let single = Gemv::with_accum(gpu, ty, shape.rows, shape.cols, accum)?;
    let batch = GemvBatch::with_accum(gpu, ty, shape.rows, shape.cols, accum)?;

    // One synthetic blob per expert (distinct weights, distinct activations),
    // concatenated in W exactly like the M3 slot ring holds them.
    let padded = batch.padded_cols();
    let mut w_bytes = Vec::with_capacity(batch.total_bytes(experts) as usize);
    let mut x_all = Vec::with_capacity(layout.vectors(experts) * padded);
    let mut first_x = None;
    for e in 0..experts {
        let (packed, x) = crate::real::synthetic_inputs(
            ty,
            shape,
            0x5eed_1234 ^ (e as u64).wrapping_mul(0x9e37_79b9),
        );
        if e == 0 {
            first_x = Some(x.clone());
        }
        w_bytes.extend_from_slice(&packed);
        if layout == XLayout::PerExpert {
            x_all.extend_from_slice(&x);
        }
    }
    if layout == XLayout::Shared {
        x_all = first_x.expect("at least one expert");
    }

    let w = upload(gpu, &w_bytes).context("uploading the expert set")?;
    let x = upload(gpu, &f32_to_bytes(&x_all)).context("uploading x")?;
    let out = alloc(gpu, (batch.rows(experts) * 4) as u64).context("allocating output")?;
    let map_host = batch.map(experts, layout);
    let map = upload(gpu, &map_to_bytes(&map_host)).context("uploading the row map")?;

    // Loop baseline buffers: the single-expert pipeline reads expert 0's blob.
    let w0 = upload(gpu, &w_bytes[..batch.blob_bytes() as usize]).context("uploading blob 0")?;
    let x0 = upload(gpu, &f32_to_bytes(&x_all[..padded])).context("uploading x0")?;
    let out0 = alloc(gpu, (shape.rows * 4) as u64).context("allocating single output")?;

    // Warm-ups (discarded) for both paths, then three timed windows.
    single
        .run(&w0, &x0, &out0)
        .context("warm-up single-expert GEMV")?;
    batch
        .run(&w, &x, &out, &map, experts)
        .context("warm-up batched GEMV")?;
    batch
        .run_timed(&w, &x, &out, &map, experts, 0.05)
        .context("warm-up batched window")?;

    let mut single_all = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        single_all.push(single.run_timed(&w0, &x0, &out0, window)?.per_iter_s);
    }
    let single_s = best(&single_all);

    let mut loop_all = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        loop_all.push(
            single
                .run_looped_timed(&w0, &x0, &out0, experts, window)?
                .per_iter_s,
        );
    }
    let loop_s = best(&loop_all);

    // The same loop with the split-K factor the single-expert API recommends.
    let loop_splits = single.recommended_splits();
    let loop_split_s = if loop_splits > 1 {
        let part = alloc(gpu, single.partial_bytes(loop_splits)).context("allocating partials")?;
        single
            .run_split(&w0, &x0, &part, &out0, loop_splits)
            .context("warm-up split loop")?;
        let mut all = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            all.push(
                single
                    .run_split_looped_timed(&w0, &x0, &part, &out0, loop_splits, experts, window)?
                    .per_iter_s,
            );
        }
        gpu.destroy_buf(&part);
        Some(best(&all))
    } else {
        None
    };

    let mut batched_all = Vec::with_capacity(repeats);
    let mut iters = 0;
    let mut used_ts = true;
    for _ in 0..repeats {
        let t = batch.run_timed(&w, &x, &out, &map, experts, window)?;
        batched_all.push(t.per_iter_s);
        iters = t.iters;
        used_ts &= t.used_timestamps;
    }
    let batched_s = best(&batched_all);

    let total_bytes = batch.total_bytes(experts);
    let gbps = |s: f64| total_bytes as f64 / s / 1e9;
    let cell = BatchCell {
        ty: ty.name().to_string(),
        quant_id: ty.ggml_id(),
        accum: accum.name().to_string(),
        shape,
        experts,
        blob_bytes: batch.blob_bytes(),
        total_bytes,
        x_layout: match layout {
            XLayout::Shared => "shared",
            XLayout::PerExpert => "per-expert",
        }
        .to_string(),
        loop_seconds: loop_s,
        loop_seconds_all: loop_all,
        loop_split_seconds: loop_split_s,
        loop_split_splits: loop_splits,
        batched_seconds: batched_s,
        batched_seconds_all: batched_all,
        speedup: loop_s / batched_s,
        single_expert_seconds: single_s,
        loop_gbps: gbps(loop_s),
        batched_gbps: gbps(batched_s),
        reference_gbps: reference.gbps,
        percent_of_reference: reference.gbps.map(|bw| 100.0 * gbps(batched_s) / bw),
        tokens_per_s_equivalent: 1.0 / batched_s,
        iters,
        used_timestamps: used_ts,
        tensor: None,
        expert_ids: None,
        conformance_max_rel: None,
        conformance_rows: None,
        upload_seconds: None,
    };

    for b in [&w, &x, &out, &map, &w0, &x0, &out0] {
        gpu.destroy_buf(b);
    }
    Ok(cell)
}

/// Best (minimum) of the measured windows.
fn best(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn splits_for(gemv: &Gemv<'_>, forced: Option<u32>) -> u32 {
    match forced {
        Some(s) => s.clamp(1, 8),
        None => gemv.recommended_splits(),
    }
}

/// Reference bandwidth figures for one device, carried into the report.
#[derive(Clone)]
struct ReferenceBandwidth {
    gbps: Option<f64>,
    note: Option<String>,
}

#[allow(clippy::too_many_arguments)] // device, shape, windows, split-K, accumulator, reference
fn run_cell(
    dev: &Device,
    ty: QuantType,
    shape: Shape,
    window: f64,
    repeats: usize,
    opts_splits: Option<u32>,
    accum: Accum,
    reference: &ReferenceBandwidth,
) -> Result<Cell> {
    let gpu = dev.gpu();
    let gemv = Gemv::with_accum(gpu, ty, shape.rows, shape.cols, accum)?;
    let (packed, x) = crate::real::synthetic_inputs(ty, shape, 0x5eed_1234);

    let w = upload(gpu, &packed).context("uploading weights")?;
    let xb = upload(gpu, &f32_to_bytes(&x)).context("uploading x")?;
    let out = alloc(gpu, (shape.rows * 4) as u64).context("allocating output")?;
    // Small matrices cannot fill the GPU with one workgroup per 8 rows, so the
    // bench (like the engine) splits each row across several workgroups and
    // reduces. See `Gemv::recommended_splits`.
    let splits = splits_for(&gemv, opts_splits);
    let partials = if splits > 1 {
        Some(alloc(gpu, gemv.partial_bytes(splits)).context("allocating split-K partials")?)
    } else {
        None
    };

    // A discarded warm-up window, not just one dispatch: the first window of a
    // cell is 20-25 % slow (clock ramp after the upload idle), and measuring it
    // would report the wrong kernel.
    match &partials {
        Some(p) => gemv.run_split(&w, &xb, p, &out, splits),
        None => gemv.run(&w, &xb, &out),
    }
    .context("warm-up GEMV")?;
    match &partials {
        Some(p) => gemv.run_split_timed(&w, &xb, p, &out, splits, 0.05),
        None => gemv.run_timed(&w, &xb, &out, 0.05),
    }
    .context("warm-up window")?;

    let mut all = Vec::with_capacity(repeats);
    let mut iters = 0u32;
    let mut used_ts = true;
    for _ in 0..repeats {
        let t: GemvTiming = match &partials {
            Some(p) => gemv.run_split_timed(&w, &xb, p, &out, splits, window)?,
            None => gemv.run_timed(&w, &xb, &out, window)?,
        };
        all.push(t.per_iter_s);
        iters = t.iters;
        used_ts &= t.used_timestamps;
    }
    let best = all.iter().cloned().fold(f64::INFINITY, f64::min);
    let weight_bytes = gemv.weight_bytes();
    let gbps = weight_bytes as f64 / best / 1e9;

    // Accuracy of the same pipeline on the same bytes (4 rows is plenty: the
    // fp16 error is the row's running sum, not a per-row accident).
    let conformance_max_rel = {
        let got = read_out(dev, &out, shape.rows)?;
        let row_bytes = gemv.blocks_per_row() * ty.block_bytes();
        let mut worst = 0.0f64;
        for r in 0..shape.rows.min(4) {
            let cpu = crate::reference::dot_row(
                ty,
                &packed[r * row_bytes..(r + 1) * row_bytes],
                shape.cols,
                &x,
            );
            let g = f64::from(got[r]);
            worst = worst.max((g - cpu).abs() / cpu.abs().max(1.0));
        }
        Some(worst)
    };

    if let Some(p) = &partials {
        gpu.destroy_buf(p);
    }
    for b in [&w, &xb, &out] {
        gpu.destroy_buf(b);
    }

    Ok(Cell {
        ty: ty.name().to_string(),
        quant_id: ty.ggml_id(),
        shape,
        bits_per_value: ty.bits_per_value(),
        weight_bytes,
        accum: accum.name().to_string(),
        conformance_max_rel,
        seconds_per_iter: best,
        seconds_per_iter_all: all,
        iters,
        splits,
        gbps,
        reference_gbps: reference.gbps,
        reference_note: reference.note.clone(),
        percent_of_reference: reference.gbps.map(|bw| 100.0 * gbps / bw),
        tokens_per_s_equivalent: 1.0 / best,
        used_timestamps: used_ts,
    })
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// `f32` slice -> little-endian bytes (what the storage buffers hold).
pub fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

pub fn f32_from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read `out` (rows f32) back for conformance checks.
pub fn read_out(dev: &Device, out: &breccia_vk::ctx::Buf, rows: usize) -> Result<Vec<f32>> {
    let bytes = download(dev.gpu(), out, rows * 4)?;
    Ok(f32_from_bytes(&bytes))
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

/// `YYYY-MM-DDTHH:MM:SSZ` from the system clock (no chrono dependency).
fn utc_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Civil-from-days (Howard Hinnant's algorithm), valid for 1970..2400.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn instance_layers() -> Vec<String> {
    std::env::var("VK_INSTANCE_LAYERS")
        .map(|v| v.split(':').map(|s| s.to_string()).collect())
        .unwrap_or_default()
}
