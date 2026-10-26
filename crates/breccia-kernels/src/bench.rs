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

use crate::device::Device;
use crate::gemv::{alloc, download, upload, Gemv, GemvTiming};
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
        QuantType::Iq3Xxs,
        QuantType::Iq4Nl,
        QuantType::Q5K,
        QuantType::Bf16,
        QuantType::F32,
    ]
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
        warnings,
    };

    if opts.real {
        report.real = crate::real::run(&dev, opts)?;
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
            let cell = run_cell(&dev, ty, *shape, window, repeats, opts.splits, &reference)
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
fn clock_warmup(dev: &Device, secs: f64) -> Result<()> {
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

fn run_cell(
    dev: &Device,
    ty: QuantType,
    shape: Shape,
    window: f64,
    repeats: usize,
    opts_splits: Option<u32>,
    reference: &ReferenceBandwidth,
) -> Result<Cell> {
    let gpu = dev.gpu();
    let gemv = Gemv::new(gpu, ty, shape.rows, shape.cols)?;
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
