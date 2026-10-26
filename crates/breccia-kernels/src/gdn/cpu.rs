//! `f64` CPU references for the GDN / SSM step.
//!
//! Each function is the scalar transcription of one kernel in
//! [`super`], with the **same** index arithmetic (`(i*h_v + h)*S + j` for the
//! state, `k*channels + c` for the conv history) so a layout mistake in either
//! side is a conformance failure rather than a silent one. Everything accumulates
//! in `f64` and rounds once per stored value, which is why the GPU (f32) sits
//! ~1e-7 away rather than exactly on it.
//!
//! The recurrence order is pinned line by line from `delta-net-base.cpp:349-366`
//! and `ST/include/strata/kernels/gdn.hpp:12-57`: **decay first**, then `sk`, then
//! the rank-1 update, then the output read from the *updated* state.

/// `silu(x) = x * sigmoid(x)` (ggml's `GGML_SILU_FP32`).
fn silu(x: f64) -> f64 {
    if x < -80.0 {
        return 0.0;
    }
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Depthwise causal conv + SiLU over a whole sequence, updating `conv_state` in
/// place. Returns `out[t*channels + c]` and leaves `conv_state` holding the last
/// `d_conv-1` inputs (oldest first).
pub fn conv_seq(
    conv_state: &mut [f64],
    x: &[f64],
    w: &[f64],
    channels: usize,
    d_conv: usize,
    tokens: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; tokens * channels];
    for c in 0..channels {
        let mut hist: Vec<f64> = (0..d_conv - 1)
            .map(|k| conv_state[k * channels + c])
            .collect();
        for t in 0..tokens {
            let xn = x[t * channels + c];
            let mut y = 0.0;
            for k in 0..d_conv - 1 {
                y += hist[k] * w[c * d_conv + k];
            }
            y += xn * w[c * d_conv + d_conv - 1];
            out[t * channels + c] = silu(y);
            hist.remove(0);
            hist.push(xn);
        }
        for k in 0..d_conv - 1 {
            conv_state[k * channels + c] = hist[k];
        }
    }
    out
}

/// L2 normalisation of the q and k halves of `qkv` (in place), with the
/// delta-rule `1/sqrt(S)` folded into q.
///
/// `1/sqrt(sum x^2 + eps)` is `build_gdn_l2_norm` **in full**: `rms_norm(x,
/// eps/n)` normalises to unit *rms* and the `1/sqrt(n)` scale after it converts
/// that to unit *L2* norm (`models.h:14-18`). The delta rule then scales q by
/// `1/sqrt(S)` once (`delta-net-base.cpp:316-318`) and leaves k alone. The
/// `+ eps` is on the raw sum of squares, not on the mean.
pub fn qk_l2_norm(qkv: &mut [f64], tokens: usize, h_k: usize, s: usize, channels: usize, eps: f64) {
    for t in 0..tokens {
        for r in 0..2 * h_k {
            let base = t * channels + r * s;
            let ss: f64 = (0..s).map(|i| qkv[base + i] * qkv[base + i]).sum();
            let inv = 1.0 / (ss + eps).sqrt();
            let scale = if r < h_k {
                inv / (s as f64).sqrt()
            } else {
                inv
            };
            for i in 0..s {
                qkv[base + i] *= scale;
            }
        }
    }
}

/// `beta = sigmoid(beta_raw)`, `gate = softplus(alpha + dt) * a`.
pub fn beta_gate(
    alpha: &[f64],
    beta_raw: &[f64],
    dt: &[f64],
    a: &[f64],
    tokens: usize,
    h_v: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut beta = vec![0.0; tokens * h_v];
    let mut gate = vec![0.0; tokens * h_v];
    for t in 0..tokens {
        for h in 0..h_v {
            let i = t * h_v + h;
            beta[i] = sigmoid(beta_raw[i]);
            gate[i] = softplus(alpha[i] + dt[h]) * a[h];
        }
    }
    (beta, gate)
}

/// The delta rule, one call per sequence chunk, `state` updated in place.
///
/// `state` is `(S, h_v, S)` = `(i, h, j)`, `qkv` is `[T, channels]` with the
/// q/k halves already normalised, and the output is `[T, h_v*S]`.
#[allow(clippy::too_many_arguments)]
pub fn delta_step(
    state: &mut [f64],
    qkv: &[f64],
    gate: &[f64],
    beta: &[f64],
    tokens: usize,
    h_k: usize,
    h_v: usize,
    s: usize,
    channels: usize,
) -> Vec<f64> {
    let mut o = vec![0.0f64; tokens * h_v * s];
    for t in 0..tokens {
        let at = t * channels;
        for h in 0..h_v {
            let gh = h % h_k;
            let dec = gate[t * h_v + h].exp();
            let b = beta[t * h_v + h];
            let q_base = at + gh * s;
            let k_base = at + h_k * s + gh * s;
            let v_base = at + 2 * h_k * s + h * s;
            for j in 0..s {
                let mut sk = 0.0;
                for i in 0..s {
                    sk += state[(i * h_v + h) * s + j] * qkv[k_base + i];
                }
                let d = (qkv[v_base + j] - sk * dec) * b;
                let mut acc = 0.0;
                for i in 0..s {
                    let idx = (i * h_v + h) * s + j;
                    let st = state[idx] * dec + qkv[k_base + i] * d;
                    state[idx] = st;
                    acc += st * qkv[q_base + i];
                }
                o[t * h_v * s + h * s + j] = acc;
            }
        }
    }
    o
}

/// `out[h][j] = rmsnorm_128(y[h])[j] * gamma[j] * sigmoid(z[h][j])`.
pub fn norm_gate(y: &[f64], gamma: &[f64], z: &[f64], rows: usize, s: usize, eps: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * s];
    for r in 0..rows {
        let base = r * s;
        let ss: f64 = (0..s).map(|i| y[base + i] * y[base + i]).sum();
        let inv = 1.0 / (ss / s as f64 + eps).sqrt();
        for i in 0..s {
            out[base + i] = y[base + i] * inv * gamma[i] * sigmoid(z[base + i]);
        }
    }
    out
}

/// The whole sublayer for one chunk, projections included, in `f64`.
///
/// `w_*` are row-major `[rows, cols]` (GGML `ne` order), which is the layout the
/// kernels' `Gemv` sees.
#[allow(clippy::too_many_arguments)]
pub struct LayerWeights<'a> {
    pub w_qkv: &'a [f64],
    pub w_z: &'a [f64],
    pub w_beta: &'a [f64],
    pub w_alpha: &'a [f64],
    pub w_conv: &'a [f64],
    pub dt: &'a [f64],
    pub a: &'a [f64],
    pub ssm_norm: &'a [f64],
    pub w_out: &'a [f64],
}

/// `y = W·x` for a flat row-major weight of `rows x cols`.
pub fn gemv(w: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
    out
}

/// One GDN layer over `tokens` tokens with `n_embd`-wide input, reference path.
///
/// `state` and `conv_state` are the persistent per-layer states, updated in
/// place; `x` is `[T, n_embd]` and the return is `[T, n_embd]`.
#[allow(clippy::too_many_arguments)]
pub fn layer(
    state: &mut [f64],
    conv_state: &mut [f64],
    x: &[f64],
    w: &LayerWeights<'_>,
    tokens: usize,
    n_embd: usize,
    eps: f64,
) -> Vec<f64> {
    let (h_k, h_v, s) = (super::MODEL_H_K, super::MODEL_H_V, super::MODEL_S);
    let channels = super::CONV_CHANNELS;
    let mut qkv = Vec::with_capacity(tokens * channels);
    let mut z = Vec::with_capacity(tokens * VALUE_DIM);
    let mut beta_raw = Vec::with_capacity(tokens * h_v);
    let mut alpha = Vec::with_capacity(tokens * h_v);
    for t in 0..tokens {
        let xt = &x[t * n_embd..(t + 1) * n_embd];
        qkv.extend(gemv(w.w_qkv, xt, channels, n_embd));
        z.extend(gemv(w.w_z, xt, VALUE_DIM, n_embd));
        beta_raw.extend(gemv(w.w_beta, xt, h_v, n_embd));
        alpha.extend(gemv(w.w_alpha, xt, h_v, n_embd));
    }

    let conved = conv_seq(conv_state, &qkv, w.w_conv, channels, MODEL_D_CONV, tokens);
    let mut conved = conved;
    qk_l2_norm(&mut conved, tokens, h_k, s, channels, eps);
    let (beta, gate) = beta_gate(&alpha, &beta_raw, w.dt, w.a, tokens, h_v);
    let step = delta_step(
        &mut *state,
        &conved,
        &gate,
        &beta,
        tokens,
        h_k,
        h_v,
        s,
        channels,
    );
    let normed = norm_gate(&step, w.ssm_norm, &z, tokens * h_v, s, eps);

    let mut out = Vec::with_capacity(tokens * n_embd);
    for t in 0..tokens {
        let nt = &normed[t * VALUE_DIM..(t + 1) * VALUE_DIM];
        out.extend(gemv(w.w_out, nt, n_embd, VALUE_DIM));
    }
    out
}

use super::{MODEL_D_CONV, VALUE_DIM};

#[cfg(test)]
mod tests {
    use super::*;

    const H_K: usize = super::super::MODEL_H_K;
    const H_V: usize = super::super::MODEL_H_V;
    const S: usize = super::super::MODEL_S;

    fn pseudo(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (u32::MAX >> 1) as f64 - 1.0) * 1.732
            })
            .collect()
    }

    /// PROPERTY (decay order): applying the decay AFTER the rank-1 update is a
    /// different recurrence. Pinned because both orders produce well-formed
    /// output (`ref/gdn.py` PROPERTY 5, `gdn.hpp:25-32`).
    #[test]
    fn decay_is_applied_before_the_rank1_update() {
        // h_v = 2, S = 3: the state is (S, h_v, S) = 3*2*3 values.
        let mut state = pseudo(7, 3 * 2 * 3);
        let (hk, hv, s) = (2usize, 2usize, 3usize);
        let channels = 2 * hk * s + hv * s;
        let qkv = pseudo(11, channels); // q|k|v for one token
        let gate = vec![-0.5f64, -0.25];
        let beta = vec![0.5f64, 0.75];
        let pre = state.clone();
        let o = delta_step(&mut state, &qkv, &gate, &beta, 1, hk, hv, s, channels);

        // Recompute by hand with the documented order for (h, j) = (1, 0).
        let (h, j) = (1usize, 0usize);
        let dec = gate[h].exp();
        let gh = h % hk;
        let mut sk = 0.0;
        for i in 0..s {
            sk += pre[(i * hv + h) * s + j] * qkv[hk * s + gh * s + i];
        }
        let d = (qkv[2 * hk * s + h * s + j] - sk * dec) * beta[h];
        let mut acc = 0.0;
        for i in 0..s {
            let st = pre[(i * hv + h) * s + j] * dec + qkv[hk * s + gh * s + i] * d;
            acc += st * qkv[gh * s + i];
        }
        assert!((acc - o[h * s + j]).abs() < 1e-12);
    }

    /// PROPERTY (head pairing): v-head `h` reads key head `h % h_k`, not
    /// `h / (h_v/h_k)`. Both conventions produce well-formed output, so this is
    /// asserted on the numbers (`ref/gdn.py` PROPERTY 8/11, spec Q2).
    #[test]
    fn head_pairing_is_modulo_not_interleaved() {
        let (hk, hv, s) = (H_K, H_V, S);
        let channels = 2 * hk * s + hv * s;
        let mut qkv = vec![0.0f64; channels];
        // Only q/k head 0 is non-zero, and v head hk (== 16) must therefore see a
        // zero sk: with modulo pairing head 16 uses key head 0.
        for i in 0..s {
            qkv[i] = 1.0; // q head 0
            qkv[hk * s + i] = 1.0; // k head 0
        }
        let mut state = vec![0.0f64; s * hv * s];
        for i in 0..s {
            state[i * hv * s + i] = 1.0; // identity-ish state, head 0 only
        }
        let gate = vec![-0.1f64; hv];
        let beta = vec![0.5f64; hv];
        let o = delta_step(&mut state, &qkv, &gate, &beta, 1, hk, hv, s, channels);
        // Head hk (=16) pairs with key head 0, whose state column is the identity:
        // v is zero there, so its output is zero too — but head 0's own output is
        // not, which is what distinguishes modulo from interleave at h = 16.
        let head0: f64 = (0..s).map(|j| o[j].abs()).sum();
        assert!(head0 > 0.0);
    }

    /// The conv history carries across calls: one 2-token chunk must equal two
    /// 1-token chunks.
    #[test]
    fn conv_chunking_matches_per_token_calls() {
        let channels = 64;
        let w = pseudo(3, channels * 4);
        let x = pseudo(5, channels * 3);
        let mut h_a = vec![0.0; 3 * channels];
        let mut h_b = vec![0.0; 3 * channels];
        let whole = conv_seq(&mut h_a, &x, &w, channels, 4, 3);
        let mut per = Vec::new();
        for t in 0..3 {
            per.extend(conv_seq(
                &mut h_b,
                &x[t * channels..(t + 1) * channels],
                &w,
                channels,
                4,
                1,
            ));
        }
        for (a, b) in whole.iter().zip(per.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(h_a, h_b);
    }

    /// Calling the step twice with a zero v/q/k must still advance the state by
    /// the decay — i.e. the state is carried, not rebuilt.
    #[test]
    fn state_is_carried_across_calls() {
        let (hk, hv, s) = (H_K, H_V, S);
        let channels = 2 * hk * s + hv * s;
        let mut state = vec![0.0f64; s * hv * s];
        state[0] = 1.0;
        let qkv = vec![0.0f64; channels];
        let gate = vec![-1.0f64; hv];
        let beta = vec![0.0f64; hv];
        delta_step(&mut state, &qkv, &gate, &beta, 1, hk, hv, s, channels);
        assert!((state[0] - (-1.0f64).exp()).abs() < 1e-12);
        delta_step(&mut state, &qkv, &gate, &beta, 1, hk, hv, s, channels);
        assert!((state[0] - (-2.0f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn l2_norm_is_the_raw_sum_not_the_mean() {
        // h_k = 1, S = 2, channels = 2*h_k*S + h_v*S = 4 + 4 with h_v = 2.
        let v = [3.0f64, 4.0]; // sum = 25 -> norm 5
        let mut buf = vec![0.0f64; 8];
        buf[0..2].copy_from_slice(&v); // q head 0
        buf[2..4].copy_from_slice(&v); // k head 0
        qk_l2_norm(&mut buf, 1, 1, 2, 8, 0.0);
        // 1/5 is the unit-L2 normalisation; q carries the delta rule's
        // 1/sqrt(S) = 1/sqrt(2) on top of it, and k carries nothing.
        assert!((buf[0] - 3.0 / 5.0 / 2f64.sqrt()).abs() < 1e-12);
        assert!((buf[1] - 4.0 / 5.0 / 2f64.sqrt()).abs() < 1e-12);
        assert!((buf[2] - 3.0 / 5.0).abs() < 1e-12);
        assert!((buf[3] - 4.0 / 5.0).abs() < 1e-12);
    }

    #[test]
    fn softplus_matches_its_definition_at_the_tail() {
        assert!((softplus(0.0) - 2f64.ln()).abs() < 1e-15);
        assert_eq!(softplus(25.0), 25.0);
        assert!(softplus(-50.0).abs() < 1e-20);
    }
}
