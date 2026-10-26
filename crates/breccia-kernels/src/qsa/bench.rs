//! `breccia bench ops --op qsa`: per-stage latency of one QSA sublayer.
//!
//! The cache is sized from the largest `T` in the sweep plus a small context, so
//! the same state serves `T = 1` (decode) and `T = 64`/`256` (a prefill chunk).
//! Every stage is timed with its own back-to-back chain and a barrier between
//! dispatches (`Qsa::stage_timings`), so the **layer total is the sum** of the
//! stage cells; the report multiplies it by 12 for the per-token figure.
//!
//! Weights are synthetic BF16 at the model's shapes: `attn_q` 12288x2560,
//! `attn_k`/`attn_v` 512x2560, the indexer's 512x2560 and 128x2560, and
//! `attn_output` 2560x6144.

use anyhow::{Context, Result};

use super::{
    Qsa, QsaProjections, QsaState, QsaWeights, ATTN_OUT_DIM, IDX_DIM, IDX_N_HEAD, MODEL_HEAD_DIM,
    MODEL_N_HEAD, MODEL_N_KV_HEAD, QGATE_DIM,
};
use crate::device::Device;
use crate::gemv::{alloc, upload, Gemv};
use crate::ops::bench::{bf16_ones, cell, Measured, OpCell, OpKind};
use crate::ops::upload_f32;
use crate::proj::Proj;
use crate::quant::QuantType;

const N_EMBD: usize = 2560;

pub fn qsa_cells(
    dev: &Device,
    tokens: &[usize],
    window: f64,
    repeats: usize,
) -> Result<Vec<OpCell>> {
    let gpu = dev.gpu();
    let ty = QuantType::Bf16;
    let kv_rows = MODEL_N_KV_HEAD * MODEL_HEAD_DIM;
    let idx_q_rows = IDX_N_HEAD * IDX_DIM;
    let max_t = tokens.iter().cloned().max().unwrap_or(1);
    let max_pos = (max_t + 4096).next_power_of_two();

    let b_w_q = upload(gpu, &bf16_ones(QGATE_DIM * N_EMBD)).context("w_q")?;
    let b_w_k = upload(gpu, &bf16_ones(kv_rows * N_EMBD)).context("w_k")?;
    let b_w_v = upload(gpu, &bf16_ones(kv_rows * N_EMBD)).context("w_v")?;
    let b_w_iq = upload(gpu, &bf16_ones(idx_q_rows * N_EMBD)).context("w_iq")?;
    let b_w_ik = upload(gpu, &bf16_ones(IDX_DIM * N_EMBD)).context("w_ik")?;
    let b_w_out = upload(gpu, &bf16_ones(N_EMBD * ATTN_OUT_DIM)).context("w_out")?;
    let b_q_norm = upload_f32(gpu, &[1.0f32; MODEL_HEAD_DIM])?;
    let b_k_norm = upload_f32(gpu, &[1.0f32; MODEL_HEAD_DIM])?;
    let b_iq_norm = upload_f32(gpu, &[1.0f32; IDX_DIM])?;
    let b_ik_norm = upload_f32(gpu, &[1.0f32; IDX_DIM])?;

    let g_q = Gemv::new(gpu, ty, QGATE_DIM, N_EMBD)?;
    let g_k = Gemv::new(gpu, ty, kv_rows, N_EMBD)?;
    let g_v = Gemv::new(gpu, ty, kv_rows, N_EMBD)?;
    let g_iq = Gemv::new(gpu, ty, idx_q_rows, N_EMBD)?;
    let g_ik = Gemv::new(gpu, ty, IDX_DIM, N_EMBD)?;
    let g_out = Gemv::new(gpu, ty, N_EMBD, ATTN_OUT_DIM)?;

    let qsa = Qsa::new(gpu, max_pos)?;
    let state = QsaState::new(gpu, max_t + 4096, max_pos)?;
    let wg = QsaWeights {
        q_norm: &b_q_norm,
        k_norm: &b_k_norm,
        idx_q_norm: &b_iq_norm,
        idx_k_norm: &b_ik_norm,
    };
    let weight_bytes = ((QGATE_DIM * N_EMBD
        + 2 * kv_rows * N_EMBD
        + idx_q_rows * N_EMBD
        + IDX_DIM * N_EMBD
        + N_EMBD * ATTN_OUT_DIM) as u64)
        * 2;

    let mut cells = Vec::new();
    // (tokens, pos0): the sweep, plus one decode step at the full sparse budget
    // (2051 cells) so the report has a measured long-context decode number rather
    // than only a linear extrapolation. The cache behind it is zero-filled — this
    // is a latency cell, not a correctness one.
    let mut plan: Vec<(usize, usize)> = Vec::new();
    let mut p = 0usize;
    for &t in tokens {
        plan.push((t, p));
        p += t;
    }
    if tokens.contains(&1) {
        plan.push((1, 2048));
    }
    for &(t, pos0) in &plan {
        let ws = qsa.workspace(gpu, t, pos0 + t)?;
        let b_x = upload_f32(gpu, &vec![0.125f32; t * N_EMBD])?;
        let b_out = alloc(gpu, (t * N_EMBD * 4) as u64)?;
        let (bt, maps) = if t > 1 {
            let a = crate::GemvBatch::new(gpu, ty, QGATE_DIM, N_EMBD)?;
            let b = crate::GemvBatch::new(gpu, ty, kv_rows, N_EMBD)?;
            let c = crate::GemvBatch::new(gpu, ty, kv_rows, N_EMBD)?;
            let d = crate::GemvBatch::new(gpu, ty, idx_q_rows, N_EMBD)?;
            let e = crate::GemvBatch::new(gpu, ty, IDX_DIM, N_EMBD)?;
            let f = crate::GemvBatch::new(gpu, ty, N_EMBD, ATTN_OUT_DIM)?;
            let maps: Vec<_> = [&a, &b, &c, &d, &e, &f]
                .iter()
                .map(|pipe| crate::proj::batch_map_buf(gpu, pipe, t).expect("map"))
                .collect();
            (Some((a, b, c, d, e, f)), maps)
        } else {
            (None, Vec::new())
        };
        let proj = match &bt {
            Some((a, b, c, d, e, f)) => QsaProjections {
                q: Proj::Batch {
                    pipe: a,
                    map: &maps[0],
                },
                w_q: &b_w_q,
                k: Proj::Batch {
                    pipe: b,
                    map: &maps[1],
                },
                w_k: &b_w_k,
                v: Proj::Batch {
                    pipe: c,
                    map: &maps[2],
                },
                w_v: &b_w_v,
                idx_q: Proj::Batch {
                    pipe: d,
                    map: &maps[3],
                },
                w_idx_q: &b_w_iq,
                idx_k: Proj::Batch {
                    pipe: e,
                    map: &maps[4],
                },
                w_idx_k: &b_w_ik,
                out: Proj::Batch {
                    pipe: f,
                    map: &maps[5],
                },
                w_out: &b_w_out,
            },
            None => QsaProjections {
                q: Proj::Single(&g_q),
                w_q: &b_w_q,
                k: Proj::Single(&g_k),
                w_k: &b_w_k,
                v: Proj::Single(&g_v),
                w_v: &b_w_v,
                idx_q: Proj::Single(&g_iq),
                w_idx_q: &b_w_iq,
                idx_k: Proj::Single(&g_ik),
                w_idx_k: &b_w_ik,
                out: Proj::Single(&g_out),
                w_out: &b_w_out,
            },
        };
        // Warm up, then measure each stage.
        let _ = qsa.stage_timings(&state, &ws, &wg, &proj, &b_x, &b_out, pos0, 0.02)?;
        let mut stage_windows: Vec<Vec<f64>> = vec![Vec::new(); super::STAGES.len()];
        for _ in 0..repeats {
            for (i, tm) in qsa
                .stage_timings(&state, &ws, &wg, &proj, &b_x, &b_out, pos0, window)?
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
                "q-gemv" => (
                    QGATE_DIM as u64 * N_EMBD as u64 * 2,
                    QGATE_DIM as u64 * N_EMBD as u64,
                ),
                "k-gemv" | "v-gemv" => (
                    kv_rows as u64 * N_EMBD as u64 * 2,
                    kv_rows as u64 * N_EMBD as u64,
                ),
                "idx-q-gemv" => (
                    idx_q_rows as u64 * N_EMBD as u64 * 2,
                    idx_q_rows as u64 * N_EMBD as u64,
                ),
                "idx-k-gemv" => (
                    IDX_DIM as u64 * N_EMBD as u64 * 2,
                    IDX_DIM as u64 * N_EMBD as u64,
                ),
                "out-gemv" => (
                    N_EMBD as u64 * ATTN_OUT_DIM as u64 * 2,
                    N_EMBD as u64 * ATTN_OUT_DIM as u64,
                ),
                "attn-scores+softmax" => (
                    // K reads (24 heads re-read the same 2-head cells) + the score
                    // matrix write/read plus the softmax's two extra passes.
                    (MODEL_N_HEAD as u64 * 4 * slots_hint(&ws) as u64 * MODEL_HEAD_DIM as u64 * 2
                        + 4 * MODEL_N_HEAD as u64 * slots_hint(&ws) as u64 * 4)
                        .max(1),
                    MODEL_N_HEAD as u64 * slots_hint(&ws) as u64 * MODEL_HEAD_DIM as u64,
                ),
                "attn-apply+gate" => (
                    (MODEL_N_HEAD as u64 * 4 * slots_hint(&ws) as u64 * MODEL_HEAD_DIM as u64 * 2
                        + MODEL_N_HEAD as u64 * MODEL_HEAD_DIM as u64 * 4)
                        .max(1),
                    MODEL_N_HEAD as u64 * slots_hint(&ws) as u64 * MODEL_HEAD_DIM as u64,
                ),
                _ => (0, 0),
            };
            total += stage_windows[i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            cells.push(cell(
                OpKind::Qsa,
                &format!("stage-{name}"),
                t,
                4 * slots_hint(&ws) as u64,
                "cells-considered",
                bytes,
                macs,
                &m,
            ));
        }
        let m = Measured {
            windows: vec![total],
            iters: 0,
            used_timestamps: true,
        };
        cells.push(cell(
            OpKind::Qsa,
            "layer",
            t,
            ws.n_blocks as u64,
            "blocks",
            weight_bytes,
            0,
            &m,
        ));
        println!(
            "qsa T={t:<3} pos0={pos0:<5} blocks={:<4} slots={:<5} layer {:.2} us",
            ws.n_blocks,
            ws.slots,
            total * 1e6
        );
        qsa.release(gpu, ws);
        gpu.destroy_buf(&b_x);
        gpu.destroy_buf(&b_out);
        for m in &maps {
            gpu.destroy_buf(m);
        }
    }
    for b in [
        &b_w_q, &b_w_k, &b_w_v, &b_w_iq, &b_w_ik, &b_w_out, &b_q_norm, &b_k_norm, &b_iq_norm,
        &b_ik_norm,
    ] {
        gpu.destroy_buf(b);
    }
    state.release(gpu);
    Ok(cells)
}

/// The score-matrix width the call's workspace uses (`4 * max_ids`).
fn slots_hint(ws: &super::QsaWorkspace) -> usize {
    ws.slots
}
