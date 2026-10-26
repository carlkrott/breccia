//! `f64` CPU reference for the QSA layer and its lightning indexer.
//!
//! Same rules as [`crate::gdn::cpu`]: the scalar transcription of the kernels in
//! [`super`], with the same index arithmetic and the same selection tie-break, but
//! `f64` accumulation. The only place the reference and the GPU are *meant* to
//! differ is the f16 K/V round trip: the reference keeps f32 (converted to f16
//! and back where the kernel stores f16, so the same precision loss is modelled —
//! see [`to_f16`]).
//!
//! Everything is derived from `docs/specs/qwen4exp-architecture.md` §4.3/§5.3/§10
//! and `LC/src/models/qwen4exp.cpp:738-1013`; each function names the source of
//! its formula.

use crate::ops::cpu::{rmsnorm, rope_apply};

/// `f32` -> IEEE binary16 -> `f32`, i.e. what the paged cache does to K and V.
///
/// Round-to-nearest-even, which is what `packHalf2x16`/`unpackHalf2x16` do on
/// both devices, so the reference models the cache's precision loss exactly
/// instead of hiding it. Subnormals collapse to zero: the model's K/V are O(1)
/// (`head_dim`-wide RMS-normed keys and raw values), so that branch is
/// unreachable at these magnitudes and is documented rather than silent.
pub fn to_f16(x: f32) -> f32 {
    let b = x.to_bits();
    let sign = b & 0x8000_0000;
    let exp = ((b >> 23) & 0xff) as i32 - 127;
    if (b & 0x7fff_ffff) == 0 {
        return f32::from_bits(sign);
    }
    if exp > 15 {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    if exp < -14 {
        return f32::from_bits(sign);
    }
    let m = (b & 0x007f_ffff) | 0x0080_0000;
    let shift = 13;
    let mut q = m >> shift;
    let rem = m & ((1 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    if rem > halfway || (rem == halfway && (q & 1) == 1) {
        q += 1;
    }
    let (e2, q2) = if q >= 1 << 11 {
        (exp + 1, q >> 1)
    } else {
        (exp, q)
    };
    if e2 > 15 {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    f32::from_bits(sign | (((e2 + 127) as u32) << 23) | ((q2 & 0x3ff) << 13))
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `score = (1/sqrt(idx_dim)) * sum_h relu(q_h . pooled_b)`, `-inf` for a block
/// that starts after the query's position (spec §10.3, `qwen4exp.cpp:802-811`).
pub fn indexer_score(
    q: &[f32],
    pooled: &[f32],
    n_blocks: usize,
    pos: usize,
    idx_n_head: usize,
    idx_dim: usize,
    idx_block: usize,
) -> Vec<f64> {
    let scale = 1.0 / (idx_dim as f64).sqrt();
    (0..n_blocks)
        .map(|b| {
            if b * idx_block > pos {
                return f64::NEG_INFINITY;
            }
            let mut acc = 0.0;
            for h in 0..idx_n_head {
                let mut dot = 0.0;
                for d in 0..idx_dim {
                    dot += f64::from(q[h * idx_dim + d]) * f64::from(pooled[b * idx_dim + d]);
                }
                acc += dot.max(0.0) * scale;
            }
            acc
        })
        .collect()
}

/// Top-k blocks by descending score, ties to the lower block index, then the
/// query's own block `pos / idx_block` if the selection missed it — the kernel's
/// contract (`qsa_topk.comp`, and strata's numpy-style ordering, `qsa.cu:293-400`).
///
/// The own block is what guarantees a query can attend to itself; it is appended
/// only when absent so no cell is counted twice.
pub fn topk_blocks(scores: &[f64], k: usize, own_block: Option<usize>) -> Vec<u32> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // `sort_by` with an explicit tie rule rather than a numeric compare so the
    // documented "lower index wins" is visible.
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut out: Vec<u32> = order
        .into_iter()
        .filter(|&i| scores[i].is_finite())
        .take(k)
        .map(|i| i as u32)
        .collect();
    if let Some(t) = own_block {
        if !out.contains(&(t as u32)) {
            out.push(t as u32);
        }
    }
    out
}

/// Pooled indexer keys: `pooled[b] = rope(rmsnorm(mean(raw[4b..4b+3])), pos = 4b)`.
///
/// `raw` is the whole sequence's raw keys (`[cells][idx_dim]`), which is the
/// reference's luxury; the kernel reconstructs the same values from a 3-row tail
/// plus the current chunk.
#[allow(clippy::too_many_arguments)]
pub fn pool_keys(
    raw: &[f32],
    cells: usize,
    gamma: &[f32],
    table: &[f32],
    max_pos: usize,
    idx_dim: usize,
    idx_block: usize,
    n_rot: usize,
    eps: f32,
) -> Vec<f32> {
    let n_blocks = cells / idx_block;
    let mut out = vec![0.0f32; n_blocks * idx_dim];
    for b in 0..n_blocks {
        let mut mean = vec![0.0f32; idx_dim];
        for k in 0..idx_block {
            for d in 0..idx_dim {
                mean[d] += raw[(b * idx_block + k) * idx_dim + d] / idx_block as f32;
            }
        }
        let normed = rmsnorm(&mean, 1, idx_dim, 1, Some(gamma), eps);
        let pos = (b * idx_block) as i32;
        let roped = rope_apply(&normed, 1, idx_dim, n_rot, table, max_pos, &[pos]);
        out[b * idx_dim..(b + 1) * idx_dim].copy_from_slice(&roped);
    }
    out
}

/// Dense causal attention over the **selected** cells, gated per head:
///
/// ```text
/// A[h] = softmax_cell( q[h] . k(cell) / sqrt(head_dim) )      (selected + causal only)
/// A[h] = A[h] * sigmoid(gate[h][d])
/// ```
///
/// Unselected or future cells contribute exactly zero (they are not in
/// `cells`), which is the contract the mask realises on the device (§10.4).
#[allow(clippy::too_many_arguments)]
pub fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cells: &[u32],
    pos: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    gate: &[f32],
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut out = vec![0.0f32; n_head * head_dim];
    for h in 0..n_head {
        let kvh = h * n_kv_head / n_head;
        let mut logits: Vec<f64> = Vec::with_capacity(cells.len());
        for &c in cells {
            let c = c as usize;
            if c > pos {
                continue;
            }
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += f64::from(q[h * head_dim + d])
                    * f64::from(k[c * n_kv_head * head_dim + kvh * head_dim + d]);
            }
            logits.push(dot * scale);
        }
        let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let mut acc = vec![0.0f64; head_dim];
        let mut i = 0;
        for &c in cells {
            let c = c as usize;
            if c > pos {
                continue;
            }
            let p = exps[i] / sum;
            i += 1;
            for d in 0..head_dim {
                acc[d] += p * f64::from(v[c * n_kv_head * head_dim + kvh * head_dim + d]);
            }
        }
        for d in 0..head_dim {
            out[h * head_dim + d] = (acc[d] * sigmoid(f64::from(gate[h * head_dim + d]))) as f32;
        }
    }
    out
}

/// Build the cell list from a block list plus the causal window.
pub fn cells_of(blocks: &[u32], idx_block: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(blocks.len() * idx_block);
    for &b in blocks {
        for k in 0..idx_block {
            out.push(b * idx_block as u32 + k as u32);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Whole-layer reference (projections included)
// ---------------------------------------------------------------------------

use super::{
    ATTN_OUT_DIM, IDX_BLOCK, IDX_DIM, IDX_N_HEAD, IDX_TOPK_BLOCKS, MODEL_HEAD_DIM, MODEL_N_HEAD,
    MODEL_N_KV_HEAD, MODEL_N_ROT, QGATE_DIM,
};

/// f64/`f32`-weight projection, row-major `[rows, cols]` in GGML `ne` order.
fn gemv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f64;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += f64::from(row[c]) * f64::from(x[c]);
        }
        out[r] = acc as f32;
    }
    out
}

/// Weights one QSA layer needs, all `f32` so the reference consumes exactly the
/// bytes the GPU does.
pub struct RefWeights<'a> {
    pub w_q: &'a [f32],
    pub w_k: &'a [f32],
    pub w_v: &'a [f32],
    pub w_idx_q: &'a [f32],
    pub w_idx_k: &'a [f32],
    pub w_out: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub idx_q_norm: &'a [f32],
    pub idx_k_norm: &'a [f32],
    pub table: &'a [f32],
    pub max_pos: usize,
    pub eps: f32,
}

/// Per-layer, per-sequence reference state: the K/V cache (rounded to f16 the
/// way the paged cache stores it), the raw indexer keys and the pooled keys.
pub struct RefState {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub raw: Vec<f32>,
    pub pooled: Vec<f32>,
}

impl RefState {
    pub fn new() -> Self {
        RefState {
            k: Vec::new(),
            v: Vec::new(),
            raw: Vec::new(),
            pooled: Vec::new(),
        }
    }

    pub fn cells(&self) -> usize {
        self.raw.len() / IDX_DIM
    }
}

impl Default for RefState {
    fn default() -> Self {
        Self::new()
    }
}

/// What one reference step produced, per token: the sublayer output, the
/// selected block list and the selected cells (both in the kernel's order).
pub struct RefOut {
    pub out: Vec<f32>,
    pub ids: Vec<Vec<u32>>,
    pub cells: Vec<Vec<u32>>,
    /// The gated attention output per token (`[n_head * head_dim]` each), i.e.
    /// what the `attn_output` GEMV consumes — kept so a mismatch can be
    /// attributed to attention or to the projection.
    pub attn: Vec<Vec<f32>>,
}

/// One QSA layer over `chunk` tokens placed at `pos0`.
#[allow(clippy::too_many_arguments)]
pub fn layer_step(
    state: &mut RefState,
    x: &[f32],
    chunk: usize,
    pos0: usize,
    n_embd: usize,
    w: &RefWeights<'_>,
) -> RefOut {
    let hd = MODEL_HEAD_DIM;
    const NKV: usize = MODEL_N_KV_HEAD;
    let nkv = NKV;
    let mut per_token = Vec::with_capacity(chunk);
    for t in 0..chunk {
        let xt = &x[t * n_embd..(t + 1) * n_embd];
        let q_full = gemv(w.w_q, xt, QGATE_DIM, n_embd);
        let mut k = gemv(w.w_k, xt, nkv * hd, n_embd);
        let v = gemv(w.w_v, xt, nkv * hd, n_embd);
        let idx_q_raw = gemv(w.w_idx_q, xt, IDX_N_HEAD * IDX_DIM, n_embd);
        let idx_k = gemv(w.w_idx_k, xt, IDX_DIM, n_embd);
        let pos = pos0 + t;

        // Q: per-head RMSNorm of the first half of each head's 512-block, then rope.
        let mut q = vec![0.0f32; MODEL_N_HEAD * hd];
        for h in 0..MODEL_N_HEAD {
            let base = h * 2 * hd;
            let head = &q_full[base..base + hd];
            let normed = rmsnorm(head, 1, hd, 1, Some(w.q_norm), w.eps);
            q[h * hd..(h + 1) * hd].copy_from_slice(&normed);
        }
        let q = rope_apply(
            &q,
            MODEL_N_HEAD,
            hd,
            MODEL_N_ROT,
            w.table,
            w.max_pos,
            &[pos as i32; MODEL_N_HEAD],
        );

        // K: per-kv-head RMSNorm + rope; V untouched.
        let mut kn = Vec::with_capacity(nkv * hd);
        for kh in 0..nkv {
            let normed = rmsnorm(&k[kh * hd..(kh + 1) * hd], 1, hd, 1, Some(w.k_norm), w.eps);
            kn.extend_from_slice(&normed);
        }
        k = rope_apply(
            &kn,
            nkv,
            hd,
            MODEL_N_ROT,
            w.table,
            w.max_pos,
            &[pos as i32; NKV],
        );

        // Indexer query: per-head RMSNorm (128) + rope.
        let mut iq = Vec::with_capacity(IDX_N_HEAD * IDX_DIM);
        for h in 0..IDX_N_HEAD {
            let normed = rmsnorm(
                &idx_q_raw[h * IDX_DIM..(h + 1) * IDX_DIM],
                1,
                IDX_DIM,
                1,
                Some(w.idx_q_norm),
                w.eps,
            );
            iq.extend_from_slice(&normed);
        }
        let iq = rope_apply(
            &iq,
            IDX_N_HEAD,
            IDX_DIM,
            MODEL_N_ROT,
            w.table,
            w.max_pos,
            &[pos as i32; IDX_N_HEAD],
        );

        // Cache append (f16 storage precision modelled exactly).
        for kh in 0..nkv {
            for d in 0..hd {
                state.k.push(to_f16(k[kh * hd + d]));
                state.v.push(to_f16(v[kh * hd + d]));
            }
        }
        state.raw.extend_from_slice(&idx_k);
        per_token.push((q, iq, q_full));
    }

    // Pool every block the chunk completed (from the full raw key history, which
    // is the reference's equivalent of the 3-row tail + chunk).
    let cells = state.cells();
    let n_blocks = cells / IDX_BLOCK;
    state.pooled = pool_keys(
        &state.raw,
        cells,
        w.idx_k_norm,
        w.table,
        w.max_pos,
        IDX_DIM,
        IDX_BLOCK,
        MODEL_N_ROT,
        w.eps,
    );

    let mut out = Vec::with_capacity(chunk * ATTN_OUT_DIM);
    let mut all_ids = Vec::with_capacity(chunk);
    let mut all_cells = Vec::with_capacity(chunk);
    let mut all_attn: Vec<Vec<f32>> = Vec::with_capacity(chunk);
    for (t, (q, iq, q_full)) in per_token.into_iter().enumerate() {
        let pos = pos0 + t;
        let scores = indexer_score(
            &iq,
            &state.pooled,
            n_blocks,
            pos,
            IDX_N_HEAD,
            IDX_DIM,
            IDX_BLOCK,
        );
        let ids = topk_blocks(&scores, IDX_TOPK_BLOCKS, Some(pos / IDX_BLOCK));
        let cells_sel = cells_of(&ids, IDX_BLOCK);
        // Gate = sigmoid of the second half of each head's q block.
        let mut gate = vec![0.0f32; MODEL_N_HEAD * hd];
        for h in 0..MODEL_N_HEAD {
            for d in 0..hd {
                gate[h * hd + d] = q_full[h * 2 * hd + hd + d];
            }
        }
        let a = attend(
            &q,
            &state.k,
            &state.v,
            &cells_sel,
            pos,
            MODEL_N_HEAD,
            nkv,
            hd,
            &gate,
        );
        out.extend(gemv(w.w_out, &a, n_embd, ATTN_OUT_DIM));
        all_ids.push(ids);
        all_cells.push(cells_sel);
        all_attn.push(a);
    }
    RefOut {
        out,
        ids: all_ids,
        cells: all_cells,
        attn: all_attn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_matches_ieee_half() {
        // Values K/V actually hold, against the exact binary16 grid.
        assert_eq!(to_f16(1.0), 1.0);
        assert_eq!(to_f16(-1.0), -1.0);
        assert_eq!(to_f16(0.0), 0.0);
        assert_eq!(to_f16(65504.0), 65504.0);
        assert_eq!(to_f16(0.1), 0.099_975_586f32); // 0x2E66
        assert_eq!(to_f16(std::f32::consts::PI), 3.140_625f32); // 0x4248
        assert_eq!(to_f16(-7.5), -7.5);
        // Round-to-nearest-even at an exact half: 2049 -> 2048, 2051 -> 2052.
        assert_eq!(to_f16(2049.0), 2048.0);
        assert_eq!(to_f16(2051.0), 2052.0);
        assert!(to_f16(1e6).is_infinite());
    }

    #[test]
    fn topk_breaks_ties_towards_the_lower_block() {
        let scores = vec![1.0f64, 3.0, 3.0, 2.0];
        let ids = topk_blocks(&scores, 2, None);
        assert_eq!(ids, vec![1u32, 2u32]);
        let ids = topk_blocks(&scores, 4, Some(9));
        assert_eq!(ids, vec![1u32, 2, 3, 0, 9]);
        // An own block already in the selection is not appended twice.
        let ids = topk_blocks(&scores, 3, Some(3));
        assert_eq!(ids, vec![1u32, 2, 3]);
    }

    #[test]
    fn dead_blocks_are_never_selected() {
        let scores = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 1.0];
        let ids = topk_blocks(&scores, 3, None);
        assert_eq!(ids, vec![2u32]);
    }
}
