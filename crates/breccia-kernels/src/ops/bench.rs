//! `breccia bench ops` — per-op latency and effective bandwidth.
//!
//! Same protocol as the decode benchmarks ([`crate::bench`]): one device per
//! call, a discarded warm-up window (the RX idles at low DPM states, so the
//! first window of a cell reads 20-25 % slow — `docs/PROJECT-SETUP.md`), then
//! `repeats` timed windows with queue timestamps and a full barrier between
//! dispatches, reporting the best one.
//!
//! Every cell is one op at a real model shape
//! (`docs/specs/qwen4exp-architecture.md` §2), at **T = 1** (decode) and
//! **T = 256** (a prefill chunk). "Effective GB/s" is the traffic the op
//! actually moves — bytes read + bytes written, including intermediate
//! activation buffers and, for the embedding gather, the staging bytes — so a
//! small op with a wide row still reports an honest number at both batch sizes.
//!
//! One op per invocation is the safety rule (`AGENTS.md` GPU crash safety): the
//! CLI's `--op` is that selector, and the conformance suite reads the same names
//! from `BRECCIA_OP`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::device::Device;
use crate::gemv::{alloc, upload};
use crate::ops::{
    download_f32, download_u32, Act, ActMode, Add, EmbGather, HcMixer, HcWeights, RmsNorm, Rope,
    RopeTable, Router, Softmax, SwiGlu, MODEL_HC, MODEL_HC_LR, MODEL_HEAD_DIM, MODEL_N_ROT,
    MODEL_TOP_K,
};
use crate::quant::QuantType;

pub const OPS_SCHEMA_VERSION: u32 = 1;

// The model's geometry (`docs/specs/qwen4exp-architecture.md` §2).
const N_EMBD: usize = 2560;
const N_FF: usize = 640;
const N_HEAD: usize = 24;
const SSM_V_HEADS: usize = 48;
const SSM_STATE: usize = 128;
const N_EXPERT: usize = 512;
const PLE_DIM: usize = 160;
const PLE_HEADS: usize = 16;

/// Which op to benchmark (one op per process, per the GPU safety rules).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    RmsNorm,
    Rope,
    SwiGlu,
    Softmax,
    Router,
    EmbGather,
    Hc,
    Add,
}

impl OpKind {
    pub const ALL: [OpKind; 8] = [
        OpKind::RmsNorm,
        OpKind::Rope,
        OpKind::SwiGlu,
        OpKind::Softmax,
        OpKind::Router,
        OpKind::EmbGather,
        OpKind::Hc,
        OpKind::Add,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            OpKind::RmsNorm => "rmsnorm",
            OpKind::Rope => "rope",
            OpKind::SwiGlu => "swiglu",
            OpKind::Softmax => "softmax",
            OpKind::Router => "router",
            OpKind::EmbGather => "emb_gather",
            OpKind::Hc => "hc",
            OpKind::Add => "add",
        }
    }

    pub fn from_name(s: &str) -> Option<OpKind> {
        let n = s.trim().to_ascii_lowercase().replace('-', "_");
        OpKind::ALL.into_iter().find(|k| k.name() == n)
    }
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for OpKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        OpKind::from_name(s).ok_or_else(|| {
            format!(
                "unknown op {s:?}; expected one of {}",
                OpKind::ALL
                    .iter()
                    .map(|k| k.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }
}

/// Benchmark options (`breccia bench ops`).
#[derive(Clone, Debug, Default)]
pub struct OpsBenchOptions {
    /// Ops to run (empty = every op).
    pub ops: Vec<OpKind>,
    /// Device by PCI address / enumeration index (empty = the first discrete GPU).
    pub devices: Vec<String>,
    /// Shorter windows, fewer repeats.
    pub quick: bool,
    /// Batch sizes to sweep (empty = `[1, 256]`).
    pub tokens: Vec<usize>,
}

/// One measured op at one shape and batch size.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpCell {
    pub op: String,
    /// What the numbers describe, e.g. `hidden-2560`, `attn-2048`, `read-chain-bf16`.
    pub case: String,
    pub tokens: usize,
    /// Work units the op moved (see `units_name`).
    pub units: u64,
    pub units_name: String,
    /// Bytes read + written by this op — the denominator of `gbps`.
    pub bytes: u64,
    /// GPU microseconds per dispatch (best of the measured windows).
    pub us: f64,
    pub us_all: Vec<f64>,
    pub gbps: f64,
    /// GFLOP/s equivalent where "multiply-accumulate count" is meaningful.
    pub gops: f64,
    pub iters: u32,
    pub used_timestamps: bool,
}

/// One run of `breccia bench ops` on one device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpsReport {
    pub schema_version: u32,
    pub environment: Environment,
    pub device: crate::device::DeviceInfo,
    pub quick: bool,
    pub cells: Vec<OpCell>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Environment {
    pub hostname: String,
    pub timestamp: String,
    pub instance_layers: Vec<String>,
    pub vulkan_validation: bool,
    pub device_open_seconds: f64,
    pub n_embd: usize,
    pub hc: usize,
    pub hc_lr: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub n_expert: usize,
    pub top_k: usize,
    pub n_ff: usize,
    pub n_head: usize,
}

/// Run `breccia bench ops`.
pub fn run_ops_bench(opts: &OpsBenchOptions) -> Result<OpsReport> {
    let selector = opts.devices.first().map(|s| s.as_str());
    let mut warnings = Vec::new();
    if opts.devices.len() > 1 {
        warnings.push(format!(
            "run_ops_bench runs one device per call; using {}",
            opts.devices[0]
        ));
    }
    let dev = Device::open(selector)?;

    let ops = if opts.ops.is_empty() {
        OpKind::ALL.to_vec()
    } else {
        opts.ops.clone()
    };
    let tokens: Vec<usize> = if opts.tokens.is_empty() {
        vec![1, 256]
    } else {
        opts.tokens.clone()
    };
    let window = if opts.quick { 0.05 } else { 0.2 };
    let repeats = if opts.quick { 2 } else { 3 };

    let mut report = OpsReport {
        schema_version: OPS_SCHEMA_VERSION,
        environment: Environment {
            hostname: hostname(),
            timestamp: utc_timestamp(),
            instance_layers: std::env::var("VK_INSTANCE_LAYERS")
                .map(|v| v.split(':').map(|s| s.to_string()).collect())
                .unwrap_or_default(),
            vulkan_validation: std::env::var("VK_INSTANCE_LAYERS")
                .map(|v| v.contains("VK_LAYER_KHRONOS_validation"))
                .unwrap_or(false),
            device_open_seconds: dev.open_seconds,
            n_embd: N_EMBD,
            hc: MODEL_HC,
            hc_lr: MODEL_HC_LR,
            head_dim: MODEL_HEAD_DIM,
            n_rot: MODEL_N_ROT,
            n_expert: N_EXPERT,
            top_k: MODEL_TOP_K,
            n_ff: N_FF,
            n_head: N_HEAD,
        },
        device: dev.info.clone(),
        quick: opts.quick,
        cells: Vec::new(),
        warnings,
    };

    // Clock ramp: one bandwidth-bound F32 GEMV for a moment before measuring
    // (same warm-up as `bench gemv`, which measured 20-25 % on the first cell
    // without it).
    if let Err(e) = crate::bench::clock_warmup(&dev, if opts.quick { 0.05 } else { 0.2 }) {
        report.warnings.push(format!(
            "clock warm-up failed (numbers may be clock-limited): {e:#}"
        ));
    }

    for op in ops {
        let mut cells = match op {
            OpKind::RmsNorm => rmsnorm_cells(&dev, &tokens, window, repeats)?,
            OpKind::Rope => rope_cells(&dev, &tokens, window, repeats)?,
            OpKind::SwiGlu => swiglu_cells(&dev, &tokens, window, repeats)?,
            OpKind::Softmax => softmax_cells(&dev, &tokens, window, repeats)?,
            OpKind::Router => router_cells(&dev, &tokens, window, repeats)?,
            OpKind::EmbGather => emb_gather_cells(&dev, &tokens, window, repeats)?,
            OpKind::Hc => hc_cells(&dev, &tokens, window, repeats)?,
            OpKind::Add => add_cells(&dev, &tokens, window, repeats)?,
        };
        report.cells.append(&mut cells);
    }
    Ok(report)
}

fn best(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Fold one measured cell's windows into a report row.
struct Measured {
    windows: Vec<f64>,
    iters: u32,
    used_timestamps: bool,
}

impl Measured {
    fn new() -> Self {
        Measured {
            windows: Vec::new(),
            iters: 0,
            used_timestamps: true,
        }
    }
    fn push(&mut self, t: crate::gemv::GemvTiming) {
        self.windows.push(t.per_iter_s);
        self.iters = t.iters;
        self.used_timestamps &= t.used_timestamps;
    }
}

#[allow(clippy::too_many_arguments)]
fn cell(
    op: OpKind,
    case: &str,
    tokens: usize,
    units: u64,
    units_name: &str,
    bytes: u64,
    macs: u64,
    m: &Measured,
) -> OpCell {
    let per_s = best(&m.windows);
    OpCell {
        op: op.name().to_string(),
        case: case.to_string(),
        tokens,
        units,
        units_name: units_name.to_string(),
        bytes,
        us: per_s * 1e6,
        us_all: m.windows.iter().map(|s| s * 1e6).collect(),
        gbps: if per_s > 0.0 {
            bytes as f64 / per_s / 1e9
        } else {
            0.0
        },
        gops: if per_s > 0.0 {
            2.0 * macs as f64 / per_s / 1e9
        } else {
            0.0
        },
        iters: m.iters,
        used_timestamps: m.used_timestamps,
    }
}

// ---------------------------------------------------------------------------
// Per-op cells
// ---------------------------------------------------------------------------

/// RMSNorm at the model's four norm shapes.
fn rmsnorm_cells(
    dev: &Device,
    tokens: &[usize],
    window: f64,
    repeats: usize,
) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    // (case, cols, rows per token, wgroups)
    let cases: [(&str, usize, usize, usize); 4] = [
        ("hidden-2560", N_EMBD, 1, 1),
        ("head-256", MODEL_HEAD_DIM, N_HEAD, 1),
        ("ssm-128", SSM_STATE, SSM_V_HEADS, 1),
        ("hc-grouped", N_EMBD, MODEL_HC, MODEL_HC),
    ];
    let mut cells = Vec::new();
    for &t in tokens {
        for (case, cols, rows_per_t, wgroups) in cases {
            let rows = rows_per_t * t;
            let norm = RmsNorm::new(gpu, cols, wgroups)?;
            let x = crate::ops::upload_f32(gpu, &vec![0.5f32; rows * cols])?;
            let gamma = if wgroups == 0 {
                None
            } else {
                Some(crate::ops::upload_f32(gpu, &vec![1.0f32; wgroups * cols])?)
            };
            let out = alloc(gpu, (rows * cols * 4) as u64)?;
            norm.run(&x, gamma.as_ref(), &out, rows)?;
            norm.run_timed(&x, gamma.as_ref(), &out, rows, 0.05)?; // discarded
            let mut m = Measured::new();
            for _ in 0..repeats {
                m.push(norm.run_timed(&x, gamma.as_ref(), &out, rows, window)?);
            }
            let bytes = (rows * cols * 8) as u64 + (wgroups * cols * 4) as u64;
            cells.push(cell(
                OpKind::RmsNorm,
                case,
                t,
                (rows * cols) as u64,
                "elements",
                bytes,
                (rows * cols) as u64,
                &m,
            ));
            if let Some(g) = gamma {
                gpu.destroy_buf(&g);
            }
            gpu.destroy_buf(&x);
            gpu.destroy_buf(&out);
        }
    }
    Ok(cells)
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// IMRoPE on the Q and K tensors of a QSA layer, in place.
fn rope_cells(dev: &Device, tokens: &[usize], window: f64, repeats: usize) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    let table = RopeTable::model(4096)?;
    let table_buf = table.upload(gpu)?;
    let rope = Rope::new(gpu, MODEL_HEAD_DIM, MODEL_N_ROT, table.max_pos())?;
    let mut cells = Vec::new();
    for &t in tokens {
        for (case, heads) in [("q-24heads", N_HEAD), ("k-2heads", 2usize)] {
            let rows = heads * t;
            let x = crate::ops::upload_f32(gpu, &vec![0.25f32; rows * MODEL_HEAD_DIM])?;
            let pos = upload(gpu, &i32_bytes(&vec![0i32; rows]))?;
            rope.run(&x, &table_buf, &pos, &x, rows)?; // in place
            rope.run_timed(&x, &table_buf, &pos, &x, rows, 0.05)?;
            let mut m = Measured::new();
            for _ in 0..repeats {
                m.push(rope.run_timed(&x, &table_buf, &pos, &x, rows, window)?);
            }
            // read + write of the tensor, the table row, and the position word
            let bytes = (rows * MODEL_HEAD_DIM * 8) as u64
                + (rows * (MODEL_N_ROT / 2) * 8) as u64
                + (rows * 4) as u64;
            cells.push(cell(
                OpKind::Rope,
                case,
                t,
                (rows * MODEL_HEAD_DIM) as u64,
                "elements",
                bytes,
                (rows * MODEL_N_ROT) as u64,
                &m,
            ));
            gpu.destroy_buf(&x);
            gpu.destroy_buf(&pos);
        }
    }
    gpu.destroy_buf(&table_buf);
    Ok(cells)
}

/// SwiGLU over a layer's routed set (10 experts x 640) and the shared expert.
fn swiglu_cells(
    dev: &Device,
    tokens: &[usize],
    window: f64,
    repeats: usize,
) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    let gl = SwiGlu::new(gpu)?;
    let mut cells = Vec::new();
    for &t in tokens {
        for (case, n) in [
            ("routed-10x640", N_FF * MODEL_TOP_K * t),
            ("shared-640", N_FF * t),
        ] {
            let g = crate::ops::upload_f32(gpu, &vec![0.75f32; n])?;
            let u = crate::ops::upload_f32(gpu, &vec![-0.5f32; n])?;
            let o = alloc(gpu, (n * 4) as u64)?;
            gl.run(&g, &u, &o, n)?;
            gl.run_timed(&g, &u, &o, n, 0.05)?;
            let mut m = Measured::new();
            for _ in 0..repeats {
                m.push(gl.run_timed(&g, &u, &o, n, window)?);
            }
            cells.push(cell(
                OpKind::SwiGlu,
                case,
                t,
                n as u64,
                "elements",
                (n * 12) as u64,
                n as u64,
                &m,
            ));
            gpu.destroy_buf(&g);
            gpu.destroy_buf(&u);
            gpu.destroy_buf(&o);
        }
    }
    Ok(cells)
}

/// Softmax over the router's 512 logits and over attention widths.
fn softmax_cells(
    dev: &Device,
    tokens: &[usize],
    window: f64,
    repeats: usize,
) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    let sm = Softmax::new(gpu)?;
    let mut cells = Vec::new();
    for &t in tokens {
        // (case, rows per token, cols) — 2049 is the ragged selection width.
        for (case, rows_per_t, cols) in [
            ("router-512", 1usize, N_EXPERT),
            ("attn-512", N_HEAD, 512usize),
            ("attn-2048", N_HEAD, 2048usize),
            ("attn-2049", N_HEAD, 2049usize),
        ] {
            let rows = rows_per_t * t;
            let x = crate::ops::upload_f32(gpu, &vec![0.125f32; rows * cols])?;
            let o = alloc(gpu, (rows * cols * 4) as u64)?;
            sm.run(&x, &o, rows, cols)?;
            sm.run_timed(&x, &o, rows, cols, 0.05)?;
            let mut m = Measured::new();
            for _ in 0..repeats {
                m.push(sm.run_timed(&x, &o, rows, cols, window)?);
            }
            cells.push(cell(
                OpKind::Softmax,
                case,
                t,
                (rows * cols) as u64,
                "elements",
                (rows * cols * 8) as u64,
                (rows * cols) as u64,
                &m,
            ));
            gpu.destroy_buf(&x);
            gpu.destroy_buf(&o);
        }
    }
    Ok(cells)
}

/// The router: softmax(512) + top-10 + clamp renormalise, one dispatch.
fn router_cells(
    dev: &Device,
    tokens: &[usize],
    window: f64,
    repeats: usize,
) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    let r = Router::model(gpu)?;
    let mut cells = Vec::new();
    for &t in tokens {
        let logits = crate::ops::upload_f32(gpu, &vec![0.0625f32; t * N_EXPERT])?;
        let ids = alloc(gpu, r.ids_bytes(t))?;
        let w = alloc(gpu, r.weights_bytes(t))?;
        r.run(&logits, &ids, &w, t)?;
        r.run_timed(&logits, &ids, &w, t, 0.05)?;
        let mut m = Measured::new();
        for _ in 0..repeats {
            m.push(r.run_timed(&logits, &ids, &w, t, window)?);
        }
        cells.push(cell(
            OpKind::Router,
            "softmax512-top10",
            t,
            (t * N_EXPERT) as u64,
            "logits",
            (t * N_EXPERT * 4 + t * MODEL_TOP_K * 8) as u64,
            (t * N_EXPERT) as u64,
            &m,
        ));
        gpu.destroy_buf(&logits);
        gpu.destroy_buf(&ids);
        gpu.destroy_buf(&w);
    }
    Ok(cells)
}

/// Embedding gather + dequant: `token_embd` (IQ3_S) and the PLE table (IQ4_NL).
///
/// The staged path is what the model uses, so the cell's byte count includes the
/// staging bytes — for the PLE table that traffic *is* the design (90 B per row,
/// 16 rows per token, out of a 26.8 GiB table that is never resident).
fn emb_gather_cells(
    dev: &Device,
    tokens: &[usize],
    window: f64,
    repeats: usize,
) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    if !gpu.feats.storage_buffer16_bit_access {
        return Err(anyhow::anyhow!(
            "{} lacks 16-bit storage buffers, which every quant gather needs",
            dev.info.name
        ));
    }
    let mut cells = Vec::new();
    let cases: [(&str, QuantType, usize, usize); 2] = [
        ("token_embd-IQ3_S", QuantType::Iq3S, N_EMBD, 1),
        ("ple-IQ4_NL-16rows", QuantType::Iq4Nl, PLE_DIM, PLE_HEADS),
    ];
    for &t in tokens {
        for (case, ty, cols, rows_per_t) in cases {
            let batch = rows_per_t * t;
            let g = EmbGather::new(gpu, ty, cols)?;
            // A packed table holding exactly the rows the gather will read: the
            // kernel's traffic is `batch * row_bytes` either way.
            let table = vec![0u8; batch * g.row_bytes()];
            let ids = EmbGather::identity_indices(batch);
            let packed = upload(gpu, &table)?;
            let idx = upload(gpu, &crate::ops::u32_to_bytes(&ids))?;
            let out = alloc(gpu, g.out_bytes(batch))?;
            g.run(&packed, &idx, &out, batch)?;
            g.run_timed(&packed, &idx, &out, batch, 0.05)?;
            let mut m = Measured::new();
            for _ in 0..repeats {
                m.push(g.run_timed(&packed, &idx, &out, batch, window)?);
            }
            let bytes = g.stage_bytes(batch) + g.out_bytes(batch) + (batch * 4) as u64;
            cells.push(cell(
                OpKind::EmbGather,
                case,
                t,
                batch as u64,
                "rows",
                bytes,
                (batch * g.cols()) as u64,
                &m,
            ));
            gpu.destroy_buf(&packed);
            gpu.destroy_buf(&idx);
            gpu.destroy_buf(&out);
        }
    }
    Ok(cells)
}

/// BF16 bytes for `n` ones (`0x3F80`), the artifact's `hc_*` type.
fn bf16_ones(n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 2);
    for _ in 0..n {
        v.extend_from_slice(&0x3f80u16.to_le_bytes());
    }
    v
}

/// The hyper-connection mixer: the whole HC_READ chain, and its elementwise
/// stages alone so the projection share is visible.
fn hc_cells(dev: &Device, tokens: &[usize], window: f64, repeats: usize) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    let ty = QuantType::Bf16; // the artifact's hc_* type (spec §14)
    let mixer = HcMixer::model(gpu, ty, true)?;
    let hc_dim = MODEL_HC * N_EMBD;
    let norm_w = crate::ops::upload_f32(gpu, &vec![1.0f32; hc_dim])?;
    let down_w = upload(gpu, &bf16_ones(MODEL_HC_LR * hc_dim))
        .context("uploading the synthetic hc down weights")?;
    let up_w = upload(gpu, &bf16_ones(hc_dim * MODEL_HC_LR))
        .context("uploading the synthetic hc up weights")?;
    let inj_w = upload(gpu, &bf16_ones(MODEL_HC * hc_dim))
        .context("uploading the synthetic hc inject weights")?;
    let weights = HcWeights {
        norm: &norm_w,
        down: &down_w,
        up: &up_w,
        inject: Some(&inj_w),
    };
    let weight_bytes =
        ((MODEL_HC_LR * hc_dim) * 2 + (hc_dim * MODEL_HC_LR) * 2 + (MODEL_HC * hc_dim) * 2) as u64;
    let macs = ((MODEL_HC_LR * hc_dim) + (hc_dim * MODEL_HC_LR) + (MODEL_HC * hc_dim)) as u64;

    let mut cells = Vec::new();
    for &t in tokens {
        let ws = mixer.workspace(gpu, t)?;
        let r = crate::ops::upload_f32(gpu, &vec![0.125f32; hc_dim * t])?;
        let block = crate::ops::upload_f32(gpu, &vec![0.0625f32; N_EMBD * t])?;
        let out = alloc(gpu, (hc_dim * t * 4) as u64)?;

        // (a) the whole HC_READ chain
        let mut m = Measured::new();
        for _ in 0..repeats {
            m.push(mixer.read_timed(&ws, &r, &weights, window)?);
        }
        let act_bytes =
            ((hc_dim * 3 + hc_dim + MODEL_HC_LR * 2 + N_EMBD + MODEL_HC) * 4) as u64 * t as u64;
        cells.push(cell(
            OpKind::Hc,
            "read-chain-bf16",
            t,
            (hc_dim * t) as u64,
            "plan",
            weight_bytes * t as u64 + act_bytes,
            macs * t as u64,
            &m,
        ));

        // (b) the elementwise stages alone (norm + gated mean + combine)
        mixer.read(&ws, &r, &weights)?; // fill xn / gate / inject
        let mut m = Measured::new();
        for _ in 0..repeats {
            let a = mixer
                .norm()
                .run_timed(&r, Some(&norm_w), &ws.xn, MODEL_HC * t, window)?;
            let b = mixer
                .gated_mean()
                .run_timed(&ws.xn, &ws.gate, &ws.mixed, t, window)?;
            let c = mixer
                .combine()
                .run_timed(&r, &block, &ws.inject, &out, t, window)?;
            m.windows.push(a.per_iter_s + b.per_iter_s + c.per_iter_s);
            m.iters = a.iters.min(b.iters).min(c.iters);
            m.used_timestamps = a.used_timestamps && b.used_timestamps && c.used_timestamps;
        }
        cells.push(cell(
            OpKind::Hc,
            "elementwise-bundle",
            t,
            (hc_dim * t) as u64,
            "plan",
            ((hc_dim * 6) + N_EMBD * 2 + MODEL_HC) as u64 * 4 * t as u64,
            (hc_dim * 3) as u64 * t as u64,
            &m,
        ));

        // (c) per-stage breakdown, so the chain total is attributable
        let mut stage_windows: Vec<Vec<f64>> = vec![Vec::new(); 6];
        for _ in 0..repeats {
            for (i, t_stage) in mixer
                .stage_timings(&ws, &r, &weights, window)?
                .iter()
                .enumerate()
            {
                stage_windows[i].push(t_stage.per_iter_s);
            }
        }
        // (bytes, macs) per stage: the two projections read their whole weight
        // matrix per token, the rest are elementwise over the activations.
        let stage_bytes = [
            ((hc_dim * 8) as u64) * t as u64, // norm
            (MODEL_HC_LR as u64 * hc_dim as u64 * 2) * t as u64 + (hc_dim * 4) as u64 * t as u64,
            (MODEL_HC_LR * 12) as u64 * t as u64, // act
            (hc_dim as u64 * MODEL_HC_LR as u64 * 2) * t as u64 + (hc_dim * 4) as u64 * t as u64,
            (hc_dim * 8 + N_EMBD * 4) as u64 * t as u64, // gated
            (MODEL_HC as u64 * hc_dim as u64 * 2) * t as u64,
        ];
        let stage_macs = [
            (hc_dim) as u64 * t as u64,
            (MODEL_HC_LR as u64 * hc_dim as u64) * t as u64,
            (MODEL_HC_LR) as u64 * t as u64,
            (hc_dim as u64 * MODEL_HC_LR as u64) * t as u64,
            (hc_dim) as u64 * t as u64,
            (MODEL_HC as u64 * hc_dim as u64) * t as u64,
        ];
        for i in 0..6 {
            let m = Measured {
                windows: stage_windows[i].clone(),
                iters: 0,
                used_timestamps: true,
            };
            cells.push(cell(
                OpKind::Hc,
                &format!("stage-{}", HcMixer::STAGES[i]),
                t,
                1,
                "dispatch",
                stage_bytes[i],
                stage_macs[i],
                &m,
            ));
        }

        mixer.release(gpu, ws);
        gpu.destroy_buf(&r);
        gpu.destroy_buf(&block);
        gpu.destroy_buf(&out);
    }
    for b in [&norm_w, &down_w, &up_w, &inj_w] {
        gpu.destroy_buf(b);
    }
    Ok(cells)
}

/// Residual add / scale over the hidden state.
fn add_cells(dev: &Device, tokens: &[usize], window: f64, repeats: usize) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    let add = Add::new(gpu)?;
    let scale = Act::new(gpu, ActMode::Scale)?;
    let mut cells = Vec::new();
    for &t in tokens {
        let n = N_EMBD * t;
        let a = crate::ops::upload_f32(gpu, &vec![1.0f32; n])?;
        let b = crate::ops::upload_f32(gpu, &vec![0.5f32; n])?;
        let o = alloc(gpu, (n * 4) as u64)?;
        add.run(&a, &b, &o, n, 1.0)?;
        add.run_timed(&a, &b, &o, n, 1.0, 0.05)?;
        let mut m = Measured::new();
        for _ in 0..repeats {
            m.push(add.run_timed(&a, &b, &o, n, 1.0, window)?);
        }
        cells.push(cell(
            OpKind::Add,
            "residual-add",
            t,
            n as u64,
            "elements",
            (n * 12) as u64,
            n as u64,
            &m,
        ));

        scale.run(&a, &o, n, 0.5)?;
        scale.run_timed(&a, &o, n, 0.5, 0.05)?;
        let mut m = Measured::new();
        for _ in 0..repeats {
            m.push(scale.run_timed(&a, &o, n, 0.5, window)?);
        }
        cells.push(cell(
            OpKind::Add,
            "residual-scale",
            t,
            n as u64,
            "elements",
            (n * 8) as u64,
            n as u64,
            &m,
        ));
        gpu.destroy_buf(&a);
        gpu.destroy_buf(&b);
        gpu.destroy_buf(&o);
    }
    Ok(cells)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read back the router's `(ids, weights)` (used by the conformance suite).
pub fn read_route(
    gpu: &breccia_vk::ctx::Gpu,
    ids: &breccia_vk::ctx::Buf,
    weights: &breccia_vk::ctx::Buf,
    rows: usize,
    k: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    Ok((
        download_u32(gpu, ids, rows * k)?,
        download_f32(gpu, weights, rows * k)?,
    ))
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn utc_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_names_round_trip_and_are_the_cli_surface() {
        for k in OpKind::ALL {
            assert_eq!(OpKind::from_name(k.name()), Some(k), "{k}");
            assert_eq!(k.name().parse::<OpKind>(), Ok(k));
        }
        assert_eq!(OpKind::from_name("EMB-GATHER"), Some(OpKind::EmbGather));
        assert!(OpKind::from_name("gemv").is_none());
    }
}
