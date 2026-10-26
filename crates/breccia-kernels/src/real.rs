//! Real-model mode: run the fused dequant+GEMV on an actual expert tensor.
//!
//! Nothing is copied out of the model: `breccia_format::ShardSet` maps the two
//! 75 GB GGUF shards read-only and hands out `&[u8]` views of the mapping, so
//! "reading the tensor" costs the page faults for exactly the tensor being
//! benchmarked (0.24–0.47 GB for a routed-expert role) and nothing else.
//!
//! # What is multiplied
//!
//! A routed-expert role tensor is `[ne0, ne1, ne2]` = `[cols, rows, 512
//! experts]`. The kernel treats it as one tall matrix of `rows * 512` rows and
//! `cols` columns with the *same* activation vector for every expert. That is
//! exactly the weight traffic and kernel shape of one decode step for those
//! experts (a real forward pass supplies a per-expert `x`; the arithmetic is
//! identical), and it makes the measured GB/s directly comparable to the
//! synthetic cells. `--real-expert` restricts the run to one expert so the
//! result can be checked against the CPU reference on real quantised data.

use anyhow::{anyhow, Context, Result};
use breccia_format::{ExpertBlock, ShardSet, TensorView};

use crate::bench::{read_out, GemvBenchOptions, RealCell, Shape};
use crate::device::Device;
use crate::gemv::{alloc, upload, Gemv};
use crate::quant::QuantType;

/// Default model glob (read-only, `/srv/strata-data` is never written).
pub const DEFAULT_MODEL_GLOB: &str = "/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf";

/// Deterministic valid packed blocks + a padded activation vector.
///
/// Values are *valid* quant data, not arbitrary bytes: scales are fp16 in a
/// realistic range, grid/sign codes are full-range, and the activation vector
/// is seeded normal-ish data. That matters because arbitrary bytes can decode
/// to inf/NaN scales and hide bugs behind NaN comparisons.
pub fn synthetic_inputs(ty: QuantType, shape: Shape, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let qk = ty.block_elems();
    let bb = ty.block_bytes();
    let bpr = shape.cols.div_ceil(qk);
    let mut rng = XorShift64::new(seed ^ ((shape.rows as u64) << 32) ^ shape.cols as u64);
    let mut packed = vec![0u8; shape.rows * bpr * bb];

    // One template block per type with realistic scale fields, then randomise
    // the payload bytes. The scale bytes are re-randomised per block within a
    // sane range so no block is degenerate.
    for r in 0..shape.rows {
        for b in 0..bpr {
            let off = (r * bpr + b) * bb;
            let blk = &mut packed[off..off + bb];
            if matches!(ty, QuantType::Bf16 | QuantType::F32) {
                // Unquantised types have no scale field, so random payload bytes
                // would mean random exponents (values up to 3e38 and dots that
                // overflow). Generate plausible activations instead.
                let words = ty.block_elems();
                for e in 0..words {
                    let v = (rng.next_f32() - 0.5) * 8.0;
                    if ty == QuantType::Bf16 {
                        let hi = f32_to_f16_lossy_bits(v);
                        blk[2 * e..2 * e + 2].copy_from_slice(&hi.to_le_bytes());
                    } else {
                        blk[4 * e..4 * e + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                continue;
            }
            for byte in blk.iter_mut() {
                *byte = rng.next_u32() as u8;
            }
            write_scale(ty, blk, &mut rng);
        }
    }

    let padded = bpr * qk;
    let mut x = vec![0f32; padded];
    for v in x.iter_mut().take(shape.cols) {
        // ~N(0, 1) from 4 uniforms, cheap and adequate for a dot product.
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += rng.next_f32();
        }
        *v = (s - 2.0) * 1.732;
    }
    (packed, x)
}

/// BF16 = the high 16 bits of the f32 (round-to-nearest-even).
fn f32_to_f16_lossy_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// Write a plausible fp16 scale (and Q5_K min) into a block.
fn write_scale(ty: QuantType, blk: &mut [u8], rng: &mut XorShift64) {
    // 2^-6 .. 2^0, the range the real artefacts use for per-block scales.
    let d = rng.next_f32() * 0.015 + 0.0005;
    blk[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    if ty == QuantType::Q5K {
        let m = rng.next_f32() * 0.01;
        blk[2..4].copy_from_slice(&f32_to_f16(m).to_le_bytes());
    }
}

/// Round-to-nearest-even fp16 encode.
///
/// Only used to build test scales, which the generator keeps in
/// `[5e-4, 1.5e-2]` — far from the subnormal/overflow cases, which are
/// exercised directly on the decode side by `reference::tests`.
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | (u16::from(mant != 0) * 0x200);
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        return sign; // underflow to zero; outside the generator's range
    }
    let mut h = ((e as u32) << 10) | (mant >> 13);
    let round = 1u32 << 12;
    if mant & round != 0 && (mant & 0xfff != 0 || h & 1 != 0) {
        h += 1;
    }
    sign | h as u16
}

/// xorshift64* — deterministic, dependency-free test data.
struct XorShift64 {
    s: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            s: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

// ---------------------------------------------------------------------------
// Real-model runner
// ---------------------------------------------------------------------------

pub fn run(dev: &Device, opts: &GemvBenchOptions) -> Result<Vec<RealCell>> {
    let glob = opts
        .real_model
        .clone()
        .unwrap_or_else(|| DEFAULT_MODEL_GLOB.to_string());
    let shards = ShardSet::open_glob(&glob).with_context(|| {
        format!("opening the model shards {glob} (real mode needs /srv/strata-data)")
    })?;

    // Arbitrary-tensor mode: any quantised tensor in the shard set, treated as
    // a tall matrix (`ne0` = row length, everything above it = rows). This is
    // how the non-expert types (Q5_K `output.weight`, IQ3_S `token_embd.weight`)
    // get measured on real bytes.
    if let Some(name) = &opts.real_tensor {
        let tensor = shards
            .tensor(name)
            .ok_or_else(|| anyhow!("tensor {name} is not in {glob}"))?;
        return run_tensor(dev, opts, tensor, None);
    }

    let layout = shards.layout().context("model layout")?;
    let block: &ExpertBlock<'_> = layout
        .block(opts.real_layer)
        .ok_or_else(|| anyhow!("block {} does not exist in {glob}", opts.real_layer))?;
    let tensor: TensorView<'_> = match opts.real_role.as_str() {
        "gate" | "gate_exps" | "ffn_gate_exps" => block.gate,
        "up" | "up_exps" | "ffn_up_exps" => block.up,
        "down" | "down_exps" | "ffn_down_exps" => block.down,
        other => return Err(anyhow!("--real-role must be gate|up|down, got {other:?}")),
    };
    run_tensor(dev, opts, tensor, opts.real_expert)
}

/// Benchmark one real tensor (one expert's slice, or the whole thing).
fn run_tensor(
    dev: &Device,
    opts: &GemvBenchOptions,
    tensor: TensorView<'_>,
    only_expert: Option<usize>,
) -> Result<Vec<RealCell>> {
    let ty = QuantType::from_ggml(tensor.ty()).ok_or_else(|| {
        anyhow!(
            "{} is {} which has no kernel yet (implemented: {})",
            tensor.name(),
            tensor.ty(),
            QuantType::ALL
                .iter()
                .map(|q| q.name())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let cols = tensor.dims()[0] as usize;
    let rows_per_expert = tensor.dims()[1] as usize;
    let experts = tensor.dims().get(2).copied().unwrap_or(1) as usize;

    let (range, expert_sel, rows) = match only_expert {
        Some(e) => {
            if e >= experts {
                return Err(anyhow!("--real-expert {e} out of range (0..{experts})"));
            }
            let per = (tensor.n_bytes() / experts as u64) as usize;
            let start = e * per;
            (start..start + per, Some(e), rows_per_expert)
        }
        None => (
            0..tensor.n_bytes() as usize,
            None,
            rows_per_expert * experts,
        ),
    };

    // Zero-copy view of the mapping — the only place the tensor bytes are
    // touched; the page cache / mmap does the paging, nothing is copied out.
    let all = tensor.as_slice();
    let bytes = &all[range];
    let shape = Shape::new(rows, cols);

    let gpu = dev.gpu();
    let gemv = Gemv::new(gpu, ty, rows, cols)?;
    if gemv.weight_bytes() as usize != bytes.len() {
        return Err(anyhow!(
            "kernel geometry disagrees with the tensor: kernel wants {} bytes, tensor slice has {}",
            gemv.weight_bytes(),
            bytes.len()
        ));
    }

    let t_up = std::time::Instant::now();
    let w = upload(gpu, bytes).context("uploading the expert tensor")?;
    let (_, x) = synthetic_inputs(ty, shape, 0xdead_beef ^ opts.real_layer as u64);
    let xb = upload(gpu, &crate::bench::f32_to_bytes(&x)).context("uploading x")?;
    let out = alloc(gpu, (rows * 4) as u64).context("allocating output")?;
    let upload_seconds = t_up.elapsed().as_secs_f64();

    let splits = opts
        .splits
        .unwrap_or_else(|| gemv.recommended_splits())
        .clamp(1, 8);
    let partials = if splits > 1 {
        Some(alloc(gpu, gemv.partial_bytes(splits)).context("allocating split-K partials")?)
    } else {
        None
    };
    match &partials {
        Some(p) => gemv.run_split(&w, &xb, p, &out, splits),
        None => gemv.run(&w, &xb, &out),
    }
    .context("warm-up GEMV")?;
    // Discarded window: the first window after the upload idle is clock-limited.
    match &partials {
        Some(p) => gemv.run_split_timed(&w, &xb, p, &out, splits, 0.1),
        None => gemv.run_timed(&w, &xb, &out, 0.1),
    }
    .context("warm-up window")?;

    // Real tensors are 200-500 MiB, i.e. ~1 ms per pass: a 0.05 s window is
    // only ~15 iterations, too short to separate the clock ramp from the
    // kernel, so real mode uses longer windows than the synthetic sweep.
    let window = if opts.quick { 0.2 } else { 0.4 };
    let repeats = if opts.quick { 2 } else { 3 };
    let mut all_t = Vec::with_capacity(repeats);
    let mut iters = 0;
    let mut used_ts = true;
    for _ in 0..repeats {
        let t = match &partials {
            Some(p) => gemv.run_split_timed(&w, &xb, p, &out, splits, window)?,
            None => gemv.run_timed(&w, &xb, &out, window)?,
        };
        all_t.push(t.per_iter_s);
        iters = t.iters;
        used_ts &= t.used_timestamps;
    }
    let best = all_t.iter().cloned().fold(f64::INFINITY, f64::min);

    // Conformance on real quantised data (a slice, not all 300k rows).
    let gpu_out = read_out(dev, &out, rows)?;
    let check_rows = rows.min(16);
    let mut max_rel = 0.0f64;
    let row_bytes = bytes.len() / rows;
    for r in 0..check_rows {
        let cpu =
            crate::reference::dot_row(ty, &bytes[r * row_bytes..(r + 1) * row_bytes], cols, &x);
        let got = f64::from(gpu_out[r]);
        max_rel = max_rel.max((got - cpu).abs() / cpu.abs().max(1.0));
    }

    let weight_bytes = gemv.weight_bytes();
    let gbps = weight_bytes as f64 / best / 1e9;
    let (reference_gbps, _note) = reference_for(&dev.info.device_id);

    if let Some(p) = &partials {
        gpu.destroy_buf(p);
    }
    for b in [&w, &xb, &out] {
        gpu.destroy_buf(b);
    }

    Ok(vec![RealCell {
        tensor: tensor.name().to_string(),
        ty: ty.name().to_string(),
        quant_id: ty.ggml_id(),
        shape,
        experts,
        expert: expert_sel,
        weight_bytes,
        seconds_per_iter: best,
        seconds_per_iter_all: all_t,
        iters,
        splits,
        gbps,
        reference_gbps,
        percent_of_reference: reference_gbps.map(|bw| 100.0 * gbps / bw),
        tokens_per_s_equivalent: 1.0 / best,
        used_timestamps: used_ts,
        conformance_max_rel: Some(max_rel),
        conformance_rows: check_rows,
        upload_seconds,
    }])
}

fn reference_for(device_id: &str) -> (Option<f64>, Option<String>) {
    match device_id {
        "1002:73a5" => (Some(445.0), Some("RX 6950 XT, 445 GB/s reference".into())),
        "1002:66af" => (
            Some(825.0),
            Some("Radeon VII / MI50, 825 GB/s reference".into()),
        ),
        _ => (None, None),
    }
}
