//! `breccia bench ops --op gdn`: per-stage latency of one GDN sublayer.
//!
//! Weights are synthetic **BF16** at the model's own shapes (`attn_qkv`
//! 10240x2560, `attn_gate` 6144x2560, `ssm_out` 2560x6144, and the 48-wide
//! `ssm_alpha`/`ssm_beta`) because that is the storage class of this layer's
//! projections in the artifact and it gives the GEMV kernels the same bytes to
//! move per token. The activation width is the model's 2560.
//!
//! Each stage is timed with its own back-to-back chain and a barrier between
//! dispatches (`Gdn::stage_timings`), so the **layer total is the sum** of the
//! stage cells; the report multiplies it by 36 for the per-token figure.

use anyhow::{Context, Result};

use super::{Gdn, GdnProjections, GdnState, GdnWeights, MODEL_D_CONV, MODEL_H_V, VALUE_DIM};
use crate::device::Device;
use crate::gemv::Gemv;
use crate::gemv::{alloc, upload};
use crate::ops::bench::OpKind;
use crate::ops::bench::{bf16_ones, cell, Measured, OpCell};
use crate::ops::upload_f32;
use crate::proj::Proj;
use crate::quant::QuantType;

const N_EMBD: usize = 2560;

/// One GDN sublayer at `T = 1` and `T = 64`/`256`: every stage, plus the layer sum.
pub fn gdn_cells(
    dev: &Device,
    tokens: &[usize],
    window: f64,
    repeats: usize,
) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    let ty = QuantType::Bf16;
    let qkv_rows = super::CONV_CHANNELS;
    let z_rows = VALUE_DIM;
    let out_rows = N_EMBD;

    let b_w_qkv = upload(gpu, &bf16_ones(qkv_rows * N_EMBD)).context("w_qkv")?;
    let b_w_z = upload(gpu, &bf16_ones(z_rows * N_EMBD)).context("w_z")?;
    let b_w_beta = upload(gpu, &bf16_ones(MODEL_H_V * N_EMBD)).context("w_beta")?;
    let b_w_alpha = upload(gpu, &bf16_ones(MODEL_H_V * N_EMBD)).context("w_alpha")?;
    let b_w_conv = upload_f32(gpu, &vec![0.25f32; super::CONV_CHANNELS * MODEL_D_CONV])?;
    let b_dt = upload_f32(gpu, &[0.5f32; MODEL_H_V])?;
    let b_a = upload_f32(gpu, &[-0.75f32; MODEL_H_V])?;
    let b_norm = upload_f32(gpu, &[1.0f32; super::MODEL_S])?;
    let b_w_out = upload(gpu, &bf16_ones(out_rows * VALUE_DIM)).context("w_out")?;

    let g_qkv = Gemv::new(gpu, ty, qkv_rows, N_EMBD)?;
    let g_z = Gemv::new(gpu, ty, z_rows, N_EMBD)?;
    let g_beta = Gemv::new(gpu, ty, MODEL_H_V, N_EMBD)?;
    let g_alpha = Gemv::new(gpu, ty, MODEL_H_V, N_EMBD)?;
    let g_out = Gemv::new(gpu, ty, out_rows, VALUE_DIM)?;

    let gdn = Gdn::new(gpu)?;
    let state = GdnState::new(gpu)?;
    state.reset(gpu)?;
    let wg = GdnWeights {
        conv: &b_w_conv,
        dt: &b_dt,
        a: &b_a,
        ssm_norm: &b_norm,
    };
    let weight_bytes =
        ((qkv_rows * N_EMBD + z_rows * N_EMBD + 2 * MODEL_H_V * N_EMBD + out_rows * VALUE_DIM)
            as u64)
            * 2;
    let weight_macs =
        (qkv_rows * N_EMBD + z_rows * N_EMBD + 2 * MODEL_H_V * N_EMBD + out_rows * VALUE_DIM)
            as u64;
    // Per-token traffic of the non-GEMV stages (state read twice + written once,
    // conv history, the gated 6144-wide pass).
    let state_bytes = (super::MODEL_S * MODEL_H_V * super::MODEL_S * 3 * 4) as u64;

    let mut cells = Vec::new();
    for &t in tokens {
        let ws = gdn.workspace(gpu, t)?;
        let b_x = upload_f32(gpu, &vec![0.125f32; t * N_EMBD])?;
        let b_out = alloc(gpu, (t * N_EMBD * 4) as u64)?;
        // T = 1 uses the decode GEMV; a prefill chunk uses the batched form (one
        // dispatch over T token row-groups), which is the path the graph takes —
        // the same choice `tests/attn_conformance.rs` validates for both.
        let (bt_qkv, bt_z, bt_beta, bt_alpha, bt_out, maps) = if t > 1 {
            let a = crate::GemvBatch::new(gpu, ty, qkv_rows, N_EMBD)?;
            let b = crate::GemvBatch::new(gpu, ty, z_rows, N_EMBD)?;
            let c = crate::GemvBatch::new(gpu, ty, MODEL_H_V, N_EMBD)?;
            let d = crate::GemvBatch::new(gpu, ty, MODEL_H_V, N_EMBD)?;
            let e = crate::GemvBatch::new(gpu, ty, out_rows, VALUE_DIM)?;
            let maps: Vec<_> = [&a, &b, &c, &d, &e]
                .iter()
                .map(|pipe| crate::proj::batch_map_buf(gpu, pipe, t).expect("map"))
                .collect();
            (Some(a), Some(b), Some(c), Some(d), Some(e), maps)
        } else {
            (None, None, None, None, None, Vec::new())
        };
        let proj = match (&bt_qkv, &bt_z, &bt_beta, &bt_alpha, &bt_out) {
            (Some(a), Some(b), Some(c), Some(d), Some(e)) => GdnProjections {
                beta_alpha: None,
                w_beta_alpha: None,
                qkv: Proj::Batch {
                    pipe: a,
                    map: &maps[0],
                },
                w_qkv: &b_w_qkv,
                z: Proj::Batch {
                    pipe: b,
                    map: &maps[1],
                },
                w_z: &b_w_z,
                beta: Proj::Batch {
                    pipe: c,
                    map: &maps[2],
                },
                w_beta: &b_w_beta,
                alpha: Proj::Batch {
                    pipe: d,
                    map: &maps[3],
                },
                w_alpha: &b_w_alpha,
                out: Proj::Batch {
                    pipe: e,
                    map: &maps[4],
                },
                w_out: &b_w_out,
            },
            _ => GdnProjections {
                beta_alpha: None,
                w_beta_alpha: None,
                qkv: Proj::Single(&g_qkv),
                w_qkv: &b_w_qkv,
                z: Proj::Single(&g_z),
                w_z: &b_w_z,
                beta: Proj::Single(&g_beta),
                w_beta: &b_w_beta,
                alpha: Proj::Single(&g_alpha),
                w_alpha: &b_w_alpha,
                out: Proj::Single(&g_out),
                w_out: &b_w_out,
            },
        };
        // Warm every stage once so the first cell is not clock-limited.
        let _ = gdn.stage_timings(&state, &ws, &wg, &proj, &b_x, &b_out, 0.02)?;
        let mut stage_windows: Vec<Vec<f64>> = vec![Vec::new(); super::STAGES.len()];
        for _ in 0..repeats {
            for (i, tm) in gdn
                .stage_timings(&state, &ws, &wg, &proj, &b_x, &b_out, window)?
                .iter()
                .enumerate()
            {
                stage_windows[i].push(tm.per_iter_s);
            }
        }
        let mut total = 0.0;
        for (i, name) in super::STAGES.iter().enumerate() {
            let m = Measured {
                windows: stage_windows[i].clone(),
                iters: 0,
                used_timestamps: true,
            };
            let (bytes, macs) = match *name {
                "qkv-gemv" => (
                    qkv_rows as u64 * N_EMBD as u64 * 2,
                    qkv_rows as u64 * N_EMBD as u64,
                ),
                "z-gemv" => (
                    z_rows as u64 * N_EMBD as u64 * 2,
                    z_rows as u64 * N_EMBD as u64,
                ),
                "beta-gemv" | "alpha-gemv" => (
                    MODEL_H_V as u64 * N_EMBD as u64 * 2,
                    MODEL_H_V as u64 * N_EMBD as u64,
                ),
                "ssm-out-gemv" => (
                    out_rows as u64 * VALUE_DIM as u64 * 2,
                    out_rows as u64 * VALUE_DIM as u64,
                ),
                "conv+silu" => (
                    (super::CONV_CHANNELS * (MODEL_D_CONV + 1) * 4) as u64,
                    (super::CONV_CHANNELS * MODEL_D_CONV) as u64,
                ),
                "qk-l2-norm" => (
                    (2 * super::MODEL_H_K * super::MODEL_S * 2 * 4) as u64,
                    (2 * super::MODEL_H_K * super::MODEL_S) as u64,
                ),
                "beta-gate" => ((2 * MODEL_H_V * 3 * 4) as u64, (2 * MODEL_H_V) as u64),
                "delta-rule" => (
                    state_bytes,
                    (3 * super::MODEL_S * MODEL_H_V * super::MODEL_S) as u64,
                ),
                "norm+gate" => ((VALUE_DIM * 3 * 4) as u64, (VALUE_DIM * 2) as u64),
                _ => (0, 0),
            };
            total += best_of(&stage_windows[i]);
            cells.push(cell(
                OpKind::Gdn,
                &format!("stage-{name}"),
                t,
                1,
                "dispatch",
                bytes,
                macs,
                &m,
            ));
        }
        // The layer cell: the sum of the stage latencies (same dispatches, same
        // barriers — see `Gdn::stage_timings`).
        let m = Measured {
            windows: vec![total],
            iters: 0,
            used_timestamps: true,
        };
        cells.push(cell(
            OpKind::Gdn,
            "layer",
            t,
            1,
            "dispatch",
            weight_bytes + state_bytes,
            weight_macs,
            &m,
        ));
        println!(
            "gdn T={t:<3} layer {:.2} us  ({} stages, {} weights)",
            total * 1e6,
            super::STAGES.len(),
            fmt_bytes(weight_bytes)
        );
        gdn.release(gpu, ws);
        gpu.destroy_buf(&b_x);
        gpu.destroy_buf(&b_out);
        for m in &maps {
            gpu.destroy_buf(m);
        }
    }
    for b in [
        &b_w_qkv, &b_w_z, &b_w_beta, &b_w_alpha, &b_w_conv, &b_dt, &b_a, &b_norm, &b_w_out,
    ] {
        gpu.destroy_buf(b);
    }
    state.release(gpu);
    Ok(cells)
}

fn best_of(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn fmt_bytes(b: u64) -> String {
    format!("{:.1} MiB", b as f64 / (1024.0 * 1024.0))
}
