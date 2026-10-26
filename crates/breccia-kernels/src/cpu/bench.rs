//! `breccia bench cpu-experts` — the host expert lane's own numbers.
//!
//! Two measurements, because they answer two different questions:
//!
//! * **Per-type GB/s** — for every `(role, quant type)` the model's routed
//!   experts actually use, a GEMV over `--type-experts` real expert slices of
//!   one block, timed on the AVX2 path *and* the scalar path. This is the
//!   kernel's weight-bandwidth ceiling and the speedup the intrinsics buy.
//! * **A representative token** — 48 blocks × top-10 experts of the real MoE
//!   shape (`gate` GEMV → `up` GEMV → SwiGLU → `down` GEMV, weighted and
//!   accumulated), through the [`ExpertPool`]. That yields bytes/token,
//!   ms/token and GB/s, which is what the production bar is stated in:
//!   `docs/phase-3/07-moe-cache-ab.md` sets **17.5 GB/s** of sustained host
//!   expert reads as the parity target (20.79 tok/s baseline, 0.84 GB/token).
//!
//! Nothing is uploaded or copied: every byte read here is a page fault on the
//! real `breccia_format` mapping, and the report records the total.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use breccia_format::{Group, ShardSet};
use serde::Serialize;

use crate::cpu::expert::{moe_topk_sequential, ExpertPool, LayerPlan};
use crate::cpu::{gemv_into, path as resolve_path, row_bytes, Path};
use crate::quant::QuantType;

/// JSON schema version of the `bench cpu-experts` report.
pub const CPU_EXPERTS_SCHEMA_VERSION: u32 = 1;

/// Default model glob (read-only; `/srv/strata-data` is never written).
pub const DEFAULT_CPU_MODEL_GLOB: &str = "/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf";

/// The parity target this lane is measured against, GB/s.
///
/// `docs/phase-3/07-moe-cache-ab.md`: 20.79 tok/s × 0.84 GB/token ≈ 17.5 GB/s of
/// sustained host-side expert reads. Anything below this is slower than the
/// production llama.cpp host expert path.
pub const PARITY_BAR_GBPS: f64 = 17.5;

/// Benchmark options (`breccia bench cpu-experts` CLI surface).
#[derive(Clone, Debug)]
pub struct CpuExpertBenchOptions {
    /// Worker threads for the token pass (`0` = [`default_threads`]).
    pub threads: usize,
    /// Types to measure (empty = every type the model's experts use).
    pub types: Vec<QuantType>,
    /// Routed experts per block in the token pass (the model uses 10).
    pub topk: usize,
    /// GGUF glob.
    pub model_glob: String,
    /// Short windows / fewer experts / fewer reps.
    pub quick: bool,
    /// Token-pass repetitions (best is reported; the first is the cold one).
    pub reps: usize,
    /// Expert slices per per-type cell (0 = derived from `quick`).
    pub type_experts: usize,
    /// Measure only the first N blocks (0 = all).
    pub layers: usize,
    /// Where the JSON report goes, when the CLI asked for one.
    pub json: Option<PathBuf>,
}

impl Default for CpuExpertBenchOptions {
    fn default() -> Self {
        Self {
            threads: 0,
            types: Vec::new(),
            topk: 10,
            model_glob: DEFAULT_CPU_MODEL_GLOB.to_string(),
            quick: false,
            reps: 3,
            type_experts: 0,
            layers: 0,
            json: None,
        }
    }
}

/// Physical core count (unique `core id` per `physical id`), falling back to
/// `available_parallelism / 2`.
fn physical_cores() -> usize {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1));
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut phys = String::new();
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("physical id") {
            phys = v.trim_start_matches(['\t', ' ', ':']).trim().to_string();
        } else if let Some(v) = line.strip_prefix("core id") {
            let core = v.trim_start_matches(['\t', ' ', ':']).trim();
            seen.insert(format!("{phys}/{core}"));
        }
    }
    if seen.is_empty() {
        std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1))
    } else {
        seen.len()
    }
}

/// Default worker count: **physical cores / 2**, capped at 16.
///
/// Half the physical cores rather than all of them because the lane is a
/// bandwidth-bound reader of host memory and this machine runs the rest of the
/// engine (and the desktop) on the same package;
/// `docs/phase-3/08-cpu-expert-lane.md` §4 has the thread sweep that shows where
/// the knee is.
pub fn default_threads() -> usize {
    (physical_cores() / 2).clamp(1, 16)
}

fn cpu_model() -> String {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return "unknown".to_string();
    };
    text.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split_once(':'))
        .map_or_else(|| "unknown".to_string(), |(_, v)| v.trim().to_string())
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

#[derive(Clone, Debug, Serialize)]
pub struct CpuEnvironment {
    pub hostname: String,
    pub timestamp: String,
    pub cpu_model: String,
    pub physical_cores: usize,
    pub logical_cpus: usize,
    /// Path the measured runs actually used.
    pub path: Path,
    pub avx2_available: bool,
    pub forced_scalar: bool,
    pub threads: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CpuModelInfo {
    pub glob: String,
    pub shards: usize,
    pub total_bytes: u64,
    pub expert_bytes: u64,
    pub layers: usize,
    pub blocks_measured: usize,
    pub n_experts: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    /// `"gate:IQ2_XS" -> layers` — the per-role type histogram.
    pub role_types: BTreeMap<String, usize>,
    /// Bytes one full expert blob (`gate|up|down`) occupies.
    pub bytes_per_expert: u64,
}

/// One `(role, type)` micro cell: the same real weights on both code paths.
#[derive(Clone, Debug, Serialize)]
pub struct CpuTypeCell {
    pub ty: QuantType,
    pub role: String,
    pub block: usize,
    pub rows: usize,
    pub cols: usize,
    pub experts: usize,
    pub bytes: u64,
    pub reps: usize,
    pub avx2_ms: f64,
    pub avx2_gbps: f64,
    pub scalar_ms: f64,
    pub scalar_gbps: f64,
    /// `scalar_ms / avx2_ms` — the intrinsics' speedup on this cell.
    pub speedup: f64,
}

/// One whole-token pass over the model's MoE shape.
#[derive(Clone, Debug, Serialize)]
pub struct CpuTokenCell {
    pub label: String,
    pub path: Path,
    pub threads: usize,
    pub layers: usize,
    pub topk: usize,
    pub experts: usize,
    pub bytes: u64,
    /// First pass over the weights (page cache partially cold).
    pub cold_ms: f64,
    /// Best subsequent pass.
    pub best_ms: f64,
    pub gbps: f64,
    /// 1000 / best_ms: the decode rate this lane alone would allow.
    pub token_rate: f64,
    pub bar_gbps: f64,
    /// `gbps / bar_gbps * 100`.
    pub parity_pct: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CpuExpertReport {
    pub schema_version: u32,
    pub environment: CpuEnvironment,
    pub model: CpuModelInfo,
    pub per_type: Vec<CpuTypeCell>,
    pub token: Vec<CpuTokenCell>,
    pub notes: Vec<String>,
}

/// Run the benchmark and return the report (the CLI writes it as JSON).
pub fn run_cpu_expert_bench(opts: &CpuExpertBenchOptions) -> Result<CpuExpertReport> {
    let threads = if opts.threads == 0 {
        default_threads()
    } else {
        opts.threads
    };
    if threads == 0 {
        bail!("--threads must be ≥1");
    }
    if opts.topk == 0 {
        bail!("--topk must be ≥1");
    }
    let path = resolve_path();

    let set = ShardSet::open_glob(&opts.model_glob)
        .with_context(|| format!("opening {}", opts.model_glob))?;
    let set = Arc::new(set);
    let layout = set.layout()?;
    let mut plans: Vec<LayerPlan> = Vec::new();
    for block in layout.blocks() {
        if opts.layers > 0 && plans.len() >= opts.layers {
            break;
        }
        plans.push(LayerPlan::from_block(&set, block)?);
    }
    if plans.is_empty() {
        bail!("{}: no expert blocks", opts.model_glob);
    }
    let first = plans[0];
    let mut role_types: BTreeMap<String, usize> = BTreeMap::new();
    for p in &plans {
        *role_types.entry(format!("gate:{}", p.gate.ty)).or_default() += 1;
        *role_types.entry(format!("up:{}", p.up.ty)).or_default() += 1;
        *role_types.entry(format!("down:{}", p.down.ty)).or_default() += 1;
    }

    let model = CpuModelInfo {
        glob: opts.model_glob.clone(),
        shards: set.parts().len(),
        total_bytes: set.total_tensor_bytes(),
        expert_bytes: layout.group_bytes(Group::Experts),
        layers: layout.blocks().count(),
        blocks_measured: plans.len(),
        n_experts: first.n_experts,
        n_embd: first.n_embd,
        n_ff: first.n_ff,
        role_types,
        bytes_per_expert: first.expert_bytes() as u64,
    };

    let type_experts = if opts.type_experts > 0 {
        opts.type_experts
    } else if opts.quick {
        16
    } else {
        96
    }
    .min(first.n_experts);
    let reps = if opts.quick { 2 } else { opts.reps.max(2) };

    let per_type = measure_types(&set, &plans, &opts.types, type_experts, reps)?;
    let token = measure_token(&set, &plans, threads, opts.topk, reps, opts.quick)?;

    let mut notes = vec![
        format!(
            "weights are read straight from the breccia_format mmap (no copy, no upload); \
             {} of {} expert bytes are touched by one {}-layer top-{} token",
            byte_scale(token.iter().map(|c| c.bytes).max().unwrap_or(0)),
            byte_scale(model.expert_bytes),
            plans.len(),
            opts.topk
        ),
        format!(
            "parity bar: {PARITY_BAR_GBPS} GB/s sustained host expert reads \
             (docs/phase-3/07-moe-cache-ab.md: 20.79 tok/s baseline, 0.84 GB/token)"
        ),
        "SwiGLU (SiLU(gate)·up) runs inside the gate/up shard that owns the rows, i.e. fused \
         with that GEMV phase; the single-threaded `expert_ffn` reference path runs it as a \
         separate pass (0.57 ms/token if it were serial, and it is not on the critical path)"
            .to_string(),
        "per-type cells are single-threaded (kernel ceiling); the token cells show the pool"
            .to_string(),
    ];
    if forced_scalar_note() {
        notes.push("BRECCIA_CPU_FORCE_SCALAR is set: `path` resolved to scalar".to_string());
    }

    Ok(CpuExpertReport {
        schema_version: CPU_EXPERTS_SCHEMA_VERSION,
        environment: CpuEnvironment {
            hostname: hostname(),
            timestamp: utc_timestamp(),
            cpu_model: cpu_model(),
            physical_cores: physical_cores(),
            logical_cpus: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            path,
            avx2_available: crate::cpu::avx2_available(),
            forced_scalar: forced_scalar_note(),
            threads,
        },
        model,
        per_type,
        token,
        notes,
    })
}

fn forced_scalar_note() -> bool {
    crate::cpu::forced_scalar()
}

fn byte_scale(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.3} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else {
        format!("{bytes} B")
    }
}

/// Deterministic activation vector (seeded, normal-ish, `cols` elements).
fn activation(cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / 16_777_216.0 - 0.5
    };
    (0..cols)
        .map(|_| {
            let mut acc = 0.0;
            for _ in 0..4 {
                acc += next();
            }
            acc * 1.7
        })
        .collect()
}

fn measure_types(
    set: &ShardSet,
    plans: &[LayerPlan],
    want: &[QuantType],
    experts: usize,
    reps: usize,
) -> Result<Vec<CpuTypeCell>> {
    let mut cells = Vec::new();
    // One representative block per (role, type), in block order.
    let mut seen: Vec<(String, QuantType)> = Vec::new();
    for plan in plans {
        for role in ["gate", "up", "down"] {
            let r = plan.role(role);
            if !want.is_empty() && !want.contains(&r.ty) {
                continue;
            }
            let key = (role.to_string(), r.ty);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            cells.push(measure_one(set, plan, role, experts, reps)?);
        }
    }
    cells.sort_by(|a, b| {
        a.role
            .cmp(&b.role)
            .then(a.ty.name().cmp(b.ty.name()))
            .then(a.block.cmp(&b.block))
    });
    Ok(cells)
}

fn measure_one(
    set: &ShardSet,
    plan: &LayerPlan,
    role: &str,
    experts: usize,
    reps: usize,
) -> Result<CpuTypeCell> {
    let r = plan.role(role);
    let x = activation(r.cols, 0x5eed_1234 ^ plan.block as u64 ^ role.len() as u64);
    let mut out = vec![0f32; r.rows];
    let ids: Vec<usize> = (0..experts).map(|i| (i * 37) % plan.n_experts).collect();
    let mut run = |path: Path| -> Result<f64> {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let start = Instant::now();
            for &e in &ids {
                let slices = plan.slices(set, e)?;
                let packed = match role {
                    "gate" => slices.gate,
                    "up" => slices.up,
                    _ => slices.down,
                };
                gemv_into(path, r.ty, packed, r.rows, r.cols, &x, &mut out);
            }
            let secs = start.elapsed().as_secs_f64();
            if secs < best {
                best = secs;
            }
        }
        Ok(best)
    };
    let bytes = (r.rows * row_bytes(r.ty, r.cols) * ids.len()) as u64;
    let avx2_ms = run(Path::Avx2)? * 1e3;
    let scalar_ms = run(Path::Scalar)? * 1e3;
    Ok(CpuTypeCell {
        ty: r.ty,
        role: role.to_string(),
        block: plan.block,
        rows: r.rows,
        cols: r.cols,
        experts: ids.len(),
        bytes,
        reps,
        avx2_ms,
        avx2_gbps: gbps(bytes, avx2_ms),
        scalar_ms,
        scalar_gbps: gbps(bytes, scalar_ms),
        speedup: if avx2_ms > 0.0 {
            scalar_ms / avx2_ms
        } else {
            0.0
        },
    })
}

/// Deterministic top-k experts for a block (distinct indices).
fn pick_experts(block: usize, n_experts: usize, k: usize) -> Vec<usize> {
    let mut s = 0x2545_f491_4f6c_dd1d ^ (block as u64 + 1).wrapping_mul(0x9e37_79b9);
    let mut ids = Vec::with_capacity(k);
    while ids.len() < k.min(n_experts) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let e = (s % n_experts as u64) as usize;
        if !ids.contains(&e) {
            ids.push(e);
        }
    }
    ids
}

/// Router weights for the picked experts: positive, renormalised, deterministic.
fn pick_weights(k: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..k).map(|j| 1.0 / (j as f32 + 1.0)).collect();
    let sum: f32 = raw.iter().sum();
    raw.into_iter().map(|w| w / sum).collect()
}

fn measure_token(
    set: &Arc<ShardSet>,
    plans: &[LayerPlan],
    threads: usize,
    topk: usize,
    reps: usize,
    quick: bool,
) -> Result<Vec<CpuTokenCell>> {
    let first = plans[0];
    let k = topk.min(first.n_experts);
    let weights = pick_weights(k);
    let x = activation(first.n_embd, 0xabcdef);
    let mut cells = Vec::new();

    let pass = |label: &str, path: Path, threads: usize| -> Result<CpuTokenCell> {
        let mut out = vec![0f32; first.n_embd];
        let mut bytes = 0u64;
        let mut times = Vec::with_capacity(reps);
        let mut pool = if threads > 1 {
            Some(ExpertPool::new(Arc::clone(set), threads, path))
        } else {
            None
        };
        for _ in 0..reps {
            out.iter_mut().for_each(|v| *v = 0.0);
            bytes = 0;
            let start = Instant::now();
            for plan in plans {
                let ids = pick_experts(plan.block, plan.n_experts, k);
                bytes += if let Some(p) = pool.as_mut() {
                    p.moe_topk(plan, &ids, &weights, &x, &mut out)?
                } else {
                    moe_topk_sequential(set, path, plan, &ids, &weights, &x, &mut out)?
                } as u64;
            }
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        drop(pool);
        let cold_ms = times[0];
        let best_ms = times.iter().copied().fold(f64::INFINITY, f64::min);
        Ok(CpuTokenCell {
            label: label.to_string(),
            path,
            threads,
            layers: plans.len(),
            topk: k,
            experts: plans.len() * k,
            bytes,
            cold_ms,
            best_ms,
            gbps: gbps(bytes, best_ms),
            token_rate: if best_ms > 0.0 { 1e3 / best_ms } else { 0.0 },
            bar_gbps: PARITY_BAR_GBPS,
            parity_pct: gbps(bytes, best_ms) / PARITY_BAR_GBPS * 100.0,
        })
    };

    // One thread first: it is the kernel-level number and it warms nothing the
    // multi-threaded run then measures cold.
    cells.push(pass("avx2-1t", Path::Avx2, 1)?);
    if threads > 1 {
        cells.push(pass("avx2-pool", Path::Avx2, threads)?);
    }
    if !quick {
        cells.push(pass("scalar-1t", Path::Scalar, 1)?);
    }
    Ok(cells)
}

fn gbps(bytes: u64, ms: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        bytes as f64 / (ms * 1e6)
    }
}

/// One-line summary for the terminal (the JSON keeps everything).
pub fn print_summary(rep: &CpuExpertReport) {
    let e = &rep.environment;
    println!(
        "\nCPU expert lane — {} · {} · {} physical / {} logical cores · {} threads · path {}",
        e.hostname, e.timestamp, e.physical_cores, e.logical_cpus, e.threads, e.path
    );
    println!(
        "  model {} ({} shards, {} total, {} experts) · {} layers measured · {} experts/block · n_embd {} · n_ff {}",
        rep.model.glob,
        rep.model.shards,
        byte_scale(rep.model.total_bytes),
        byte_scale(rep.model.expert_bytes),
        rep.model.blocks_measured,
        rep.model.n_experts,
        rep.model.n_embd,
        rep.model.n_ff
    );
    println!(
        "  per-type (1 thread, {} experts/cell, best of {} reps):",
        rep.per_type.first().map_or(0, |c| c.experts),
        rep.per_type.first().map_or(0, |c| c.reps)
    );
    println!(
        "  {:<6} {:<4} {:>7} {:>6} {:>10} {:>9} {:>9} {:>8}",
        "type", "role", "rows", "cols", "bytes", "avx2 GB/s", "scalar", "speedup"
    );
    for c in &rep.per_type {
        println!(
            "  {:<6} {:<4} {:>7} {:>6} {:>10} {:>9.2} {:>9.2} {:>7.2}x",
            c.ty.name(),
            c.role,
            c.rows,
            c.cols,
            byte_scale(c.bytes),
            c.avx2_gbps,
            c.scalar_gbps,
            c.speedup
        );
    }
    println!(
        "  token passes ({} blocks × top-{} experts):",
        rep.token.first().map_or(0, |c| c.layers),
        rep.token.first().map_or(0, |c| c.topk)
    );
    println!(
        "  {:<10} {:>4} {:>10} {:>10} {:>10} {:>10} {:>9} {:>8}",
        "label", "thr", "bytes/tok", "cold ms", "best ms", "GB/s", "tok/s", "bar"
    );
    for c in &rep.token {
        println!(
            "  {:<10} {:>4} {:>10} {:>10.1} {:>10.1} {:>10.2} {:>9.1} {:>7.0}%",
            c.label,
            c.threads,
            byte_scale(c.bytes),
            c.cold_ms,
            c.best_ms,
            c.gbps,
            c.token_rate,
            c.parity_pct
        );
    }
    for note in &rep.notes {
        println!("  · {note}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_are_distinct_and_deterministic() {
        for block in 0..48 {
            let ids = pick_experts(block, 512, 10);
            assert_eq!(ids.len(), 10);
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 10, "block {block} repeated an expert");
            assert!(ids.iter().all(|&e| e < 512));
            assert_eq!(ids, pick_experts(block, 512, 10));
        }
    }

    #[test]
    fn weights_are_positive_and_normalised() {
        let w = pick_weights(10);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(w.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn default_threads_is_half_the_physical_cores_capped_at_16() {
        let t = default_threads();
        assert!((1..=16).contains(&t), "{t}");
        assert_eq!(t, (physical_cores() / 2).clamp(1, 16));
    }
}
