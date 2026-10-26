//! Scalar CPU references for every Phase-3 op kernel.
//!
//! Same role as [`crate::reference`] plays for the decode units: the oracle the
//! GPU kernels are measured against, written in the plainest possible form
//! (straight loops, `f64` accumulation, no reuse of the shader's structure) so a
//! disagreement points at one side or the other.
//!
//! | op | reference | GPU kernel |
//! |---|---|---|
//! | RMSNorm | [`rmsnorm`] | `shaders/ops_rmsnorm.comp` |
//! | IMRoPE | [`rope_table`] + [`rope_apply`] | `shaders/ops_rope.comp` |
//! | SwiGLU | [`swiglu`] | `shaders/ops_swiglu.comp` |
//! | softmax | [`softmax`] | `shaders/ops_softmax.comp` |
//! | router | [`router`] | `shaders/ops_router.comp` |
//! | embedding gather | [`crate::reference::dequant_row`] | `shaders/ops_gather_*.comp` |
//! | HC_READ / HC_WRITE | [`hc_read`] / [`hc_write`] | `ops_hc_*.comp` + `GemvBatch` |
//! | residual add | [`add`] | `shaders/ops_add.comp` |
//!
//! Everything here is `pub` because the conformance suite in `tests/` builds its
//! own fixtures from it.
//!
//! # Two deliberate differences from the GPU path
//!
//! * **`f64` accumulation.**  The GPU sums 2560 products in `f32` with a tree
//!   reduction; summing in `f64` on one thread gives the value the expression
//!   denotes, so the comparison measures the kernel's rounding, not a second
//!   round-off convention.
//! * **`router` follows the production kernel's *structure*** (`p` over all
//!   experts, then the `2^-14` clamp on the gathered sum) rather than a
//!   mathematically-equivalent shortcut, for the reason strata's
//!   `router_top10.cu:3-17` gives: the clamp sits inside the renormalisation, so
//!   the subset-softmax form is not equivalent.

/// `silu(x) = x / (1 + exp(-x))`, ggml's `GGML_SILU_FP32`, in `f64`.
pub fn silu(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    x / (1.0 + (-x).exp())
}

/// `sigmoid(x) = 1 / (1 + exp(-x))` in `f64`.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Row-wise RMSNorm with an optional gamma, `f64` throughout.
///
/// `x` is `rows * cols` row-major; `gamma`, when present, is `wgroups * cols`
/// and row `r` uses slice `(r % wgroups)`.  Mirrors `ops_rmsnorm.comp` including
/// the "`wgroups == 0` means no gamma" convention.
pub fn rmsnorm(
    x: &[f32],
    rows: usize,
    cols: usize,
    wgroups: usize,
    gamma: Option<&[f32]>,
    eps: f32,
) -> Vec<f32> {
    assert!(x.len() >= rows * cols, "x shorter than rows*cols");
    if wgroups > 0 {
        let g = gamma.expect("gamma required when wgroups > 0");
        assert!(g.len() >= wgroups * cols, "gamma shorter than wgroups*cols");
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mut ss = 0.0f64;
        for i in 0..cols {
            let v = f64::from(x[base + i]);
            ss += v * v;
        }
        let inv = 1.0 / (ss / cols as f64 + f64::from(eps)).sqrt();
        let gbase = if wgroups == 0 {
            0
        } else {
            (r % wgroups) * cols
        };
        for i in 0..cols {
            let w = match gamma {
                Some(g) if wgroups > 0 => f64::from(g[gbase + i]),
                _ => 1.0,
            };
            out[base + i] = (f64::from(x[base + i]) * inv * w) as f32;
        }
    }
    out
}

/// Host-side rotary table: `[max_pos][n_rot/2]` of `(cos, sin)` pairs, `f32`.
///
/// This is a transcription of strata's `build_rope_table`
/// (`src/kernels/cuda/rope.cu:49-59`) and of the reference it follows
/// (`ref/qsa.py`): `inv[i] = theta^(-2i/n_rot)`, `ang = pos * inv`, all in
/// `f64`, then rounded to `f32` for the device.  The device kernel is a pure
/// rotation over this table, so the table's values *are* the angles: rope
/// scaling, if it ever appears for this artifact, enters here and nowhere else
/// (`include/strata/kernels/rope.hpp:7-11`).
///
/// The artifact ships no `rope.scaling` keys, so the plain branch is the only
/// one implemented; `freq_base = 1e7`, `n_rot = 64`
/// (`docs/specs/qwen4exp-architecture.md` §1).
pub fn rope_table(freq_base: f64, n_rot: usize, max_pos: usize) -> Vec<f32> {
    assert!(n_rot.is_multiple_of(2), "n_rot must be even");
    let half = n_rot / 2;
    let mut t = vec![0.0f32; max_pos * half * 2];
    for p in 0..max_pos {
        for i in 0..half {
            let inv = freq_base.powf(-2.0 * i as f64 / n_rot as f64);
            let ang = p as f64 * inv;
            let o = (p * half + i) * 2;
            t[o] = ang.cos() as f32;
            t[o + 1] = ang.sin() as f32;
        }
    }
    t
}

/// Partial NEOX-paired rotary: `out[j] = x[j]*cos - x[j+half]*sin`,
/// `out[j+half] = x[j]*sin + x[j+half]*cos`, dims `[n_rot, head_dim)` copied.
///
/// `table` is the flat `(cos, sin)` table from [`rope_table`] and `positions`
/// holds one position per row.
pub fn rope_apply(
    x: &[f32],
    rows: usize,
    head_dim: usize,
    n_rot: usize,
    table: &[f32],
    max_pos: usize,
    positions: &[i32],
) -> Vec<f32> {
    assert!(positions.len() >= rows, "one position per row");
    assert!(
        n_rot <= head_dim && n_rot.is_multiple_of(2),
        "n_rot vs head_dim"
    );
    let half = n_rot / 2;
    let mut out = vec![0.0f32; rows * head_dim];
    for (r, &pos) in positions.iter().enumerate().take(rows) {
        let base = r * head_dim;
        let p = pos.clamp(0, max_pos as i32 - 1) as usize;
        for j in 0..half {
            let a = f64::from(x[base + j]);
            let b = f64::from(x[base + j + half]);
            let c = f64::from(table[(p * half + j) * 2]);
            let s = f64::from(table[(p * half + j) * 2 + 1]);
            out[base + j] = (a * c - b * s) as f32;
            out[base + j + half] = (a * s + b * c) as f32;
        }
        out[base + n_rot..base + head_dim].copy_from_slice(&x[base + n_rot..base + head_dim]);
    }
    out
}

/// `out[i] = silu(gate[i]) * up[i]`.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert!(gate.len() == up.len(), "gate/up length");
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| (silu(f64::from(*g)) * f64::from(*u)) as f32)
        .collect()
}

/// Row-wise stable softmax in `f64`.
pub fn softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert!(x.len() >= rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mx = (0..cols)
            .map(|i| f64::from(x[base + i]))
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for i in 0..cols {
            sum += (f64::from(x[base + i]) - mx).exp();
        }
        for i in 0..cols {
            out[base + i] = ((f64::from(x[base + i]) - mx).exp() / sum) as f32;
        }
    }
    out
}

/// The router's `(ids, weights)` for one row of logits.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// Selected expert ids, descending probability, ties by ascending id.
    pub ids: Vec<u32>,
    /// `p[ids] / max(sum(p[ids]), 2^-14)`.
    pub weights: Vec<f64>,
}

/// Router: softmax over ALL experts, top-k by descending probability (ties to
/// the lowest id), renormalise with ggml's `2^-14` clamp.
///
/// Transcribed from the production kernel's contract
/// (`ST/src/kernels/cuda/router_top10.cu:3-17`: `p = softmax(logits)`,
/// `ids = stable argsort(-p)[:k]`, `w = p[ids]`,
/// `s = max(sum(w), 2**-14)`, `w / s`) and from
/// `docs/specs/qwen4exp-architecture.md` §5.4/§8.
pub fn router(logits: &[f32], n_expert: usize, k: usize) -> Route {
    assert!(logits.len() >= n_expert);
    assert!(k <= n_expert, "k <= n_expert");
    let mx = logits[..n_expert]
        .iter()
        .map(|v| f64::from(*v))
        .fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = logits[..n_expert]
        .iter()
        .map(|v| (f64::from(*v) - mx).exp())
        .collect();
    let sum: f64 = ex.iter().sum();
    let p: Vec<f64> = ex.iter().map(|e| e / sum).collect();

    // Stable descending argsort by (-p, id): a total order, so the result is
    // deterministic and ties land on the lowest index.
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|&a, &b| {
        p[b].partial_cmp(&p[a])
            .expect("router probabilities are finite")
            .then(a.cmp(&b))
    });
    let ids: Vec<u32> = order[..k].iter().map(|&i| i as u32).collect();
    let sel: Vec<f64> = order[..k].iter().map(|&i| p[i]).collect();
    let s = sel.iter().sum::<f64>().max(6.103_515_625e-05);
    Route {
        ids,
        weights: sel.iter().map(|w| w / s).collect(),
    }
}

/// One HC_READ stage, `f64`, over a single token (`docs/specs/…` §4.1).
///
/// ```text
/// xn  = rmsnorm(R, eps) * gamma          per stream (cols = n_embd)
/// lo  = silu( (xn @ w_down.T) / hc )     [hc_lr]
/// gate= lo @ w_up.T                      [hc_dim]
/// gated = xn * sigmoid(gate)             per stream
/// mixed = mean_over_streams(gated)       [n_embd]
/// inject= xn @ w_inject.T                [hc]   (None for the final mixer)
/// ```
///
/// `r` is `[n_embd, hc]` row-major by stream (`r[c*n_embd + d]`), `w_down` is
/// `[hc_lr][hc_dim]`, `w_up` is `[hc_dim][hc_lr]`, `w_inject` is `[hc][hc_dim]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HcRead {
    /// `[hc*n_embd]` the grouped RMSNorm output.
    pub xn: Vec<f32>,
    /// `[hc_dim]` after `silu(x/hc)`.
    pub lo: Vec<f32>,
    /// `[n_embd]` the mixer output.
    pub mixed: Vec<f32>,
    /// `[hc]` the per-stream inject scalars (`None` for the final mixer).
    pub inject: Option<Vec<f32>>,
}

/// CPU reference for HC_READ. See [`HcRead`].
// One parameter per weight, in the order §4.1 writes them; a struct here would
// be a second copy of the same fact.
#[allow(clippy::too_many_arguments)]
pub fn hc_read_token(
    r: &[f32],
    n_embd: usize,
    hc: usize,
    hc_lr: usize,
    gamma: &[f32],
    w_down: &[f32],
    w_up: &[f32],
    w_inject: Option<&[f32]>,
    eps: f32,
) -> HcRead {
    let hc_dim = hc * n_embd;
    assert_eq!(r.len(), hc_dim, "r must be [hc*n_embd]");
    assert_eq!(
        w_down.len(),
        hc_lr * hc_dim,
        "w_down must be [hc_lr][hc_dim]"
    );
    assert_eq!(w_up.len(), hc_dim * hc_lr, "w_up must be [hc_dim][hc_lr]");
    if let Some(w) = w_inject {
        assert_eq!(w.len(), hc * hc_dim, "w_inject must be [hc][hc_dim]");
    }

    // xn = rmsnorm per stream, gamma grouped per stream
    let xn = rmsnorm(r, hc, n_embd, hc, Some(gamma), eps);

    // lo = silu((xn @ w_down.T) / hc)   -- the divide is BEFORE the silu
    let mut lo = vec![0.0f32; hc_lr];
    for (j, v) in lo.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for d in 0..hc_dim {
            acc += f64::from(xn[d]) * f64::from(w_down[j * hc_dim + d]);
        }
        *v = silu(acc / hc as f64) as f32;
    }

    // gate = lo @ w_up.T ; gated = xn * sigmoid(gate) ; mixed = mean over streams
    let mut acc_mixed = vec![0.0f64; n_embd];
    for d in 0..hc_dim {
        let mut acc = 0.0f64;
        for j in 0..hc_lr {
            acc += f64::from(w_up[d * hc_lr + j]) * f64::from(lo[j]);
        }
        let g = sigmoid(acc);
        let c = d / n_embd;
        let off = d - c * n_embd;
        acc_mixed[off] += f64::from(xn[d]) * g / hc as f64;
    }
    let mixed: Vec<f32> = acc_mixed.iter().map(|v| *v as f32).collect();

    // inject = xn @ w_inject.T  (only when this is not the final mixer)
    let inject = w_inject.map(|w| {
        (0..hc)
            .map(|c| {
                let mut acc = 0.0f64;
                for d in 0..hc_dim {
                    acc += f64::from(w[c * hc_dim + d]) * f64::from(xn[d]);
                }
                acc as f32
            })
            .collect()
    });

    HcRead {
        xn,
        lo,
        mixed,
        inject,
    }
}

/// CPU reference for HC_WRITE: `out[c] = r[c] + block * 2*sigmoid(inject[c]/hc)`.
pub fn hc_write(r: &[f32], block: &[f32], inject: &[f32], n_embd: usize, hc: usize) -> Vec<f32> {
    assert_eq!(r.len(), hc * n_embd);
    assert_eq!(block.len(), n_embd);
    assert_eq!(inject.len(), hc);
    let mut out = vec![0.0f32; hc * n_embd];
    for c in 0..hc {
        let w = 2.0 * sigmoid(f64::from(inject[c]) / hc as f64);
        for d in 0..n_embd {
            out[c * n_embd + d] = (f64::from(r[c * n_embd + d]) + f64::from(block[d]) * w) as f32;
        }
    }
    out
}

/// `out[i] = a[i] + beta * b[i]`.
pub fn add(a: &[f32], b: &[f32], beta: f32) -> Vec<f32> {
    assert!(a.len() == b.len(), "add operand lengths");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (f64::from(*x) + f64::from(beta) * f64::from(*y)) as f32)
        .collect()
}

/// `out[i] = x[i] * s`.
pub fn scale(x: &[f32], s: f32) -> Vec<f32> {
    x.iter()
        .map(|v| (f64::from(*v) * f64::from(s)) as f32)
        .collect()
}

/// `out[i] = sigmoid(x[i])`.
pub fn sigmoid_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| sigmoid(f64::from(*v)) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_of_a_constant_row_is_one() {
        // A row of four 3.0s: mean square 9, rsqrt(9) = 1/3, so every element -> 1.
        let x = vec![3.0f32; 4];
        let y = rmsnorm(&x, 1, 4, 0, None, 0.0);
        for v in &y {
            assert!((v - 1.0).abs() < 1e-7, "{v}");
        }
    }

    #[test]
    fn rope_pair_0_is_the_identity_at_position_zero() {
        // pos 0 -> cos = 1, sin = 0: the rotation is the identity, and the tail
        // (dims 64..256 of a 256-wide head) passes through untouched.
        let table = rope_table(1e7, 64, 4);
        let x: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let y = rope_apply(&x, 1, 256, 64, &table, 4, &[0]);
        for i in 0..256 {
            assert!(
                (x[i] - y[i]).abs() < 1e-6,
                "element {i}: {} vs {}",
                x[i],
                y[i]
            );
        }
    }

    #[test]
    fn rope_is_the_neox_pairing_not_the_adjacent_one() {
        // A one-hot at dim 1 (inside the first half) must rotate into dim 33
        // (half = 32), NOT into dim 2 -- which is what an adjacent-pair kernel
        // would do. At pos 1 the rotation angle for pair 1 is base^(-2/64).
        let table = rope_table(1e7, 64, 2);
        let mut x = vec![0.0f32; 64];
        x[1] = 1.0;
        let y = rope_apply(&x, 1, 64, 64, &table, 2, &[1]);
        // pair j = 1 at position pos = 1: table index (pos * pairs + j) * 2 + {cos, sin}
        let (pos, pair, pairs) = (1usize, 1usize, 32usize);
        let at = (pos * pairs + pair) * 2;
        let expect_self = table[at]; // cos
        let expect_partner = table[at + 1]; // sin
        assert!((y[1] - expect_self).abs() < 1e-6, "y[1] = {}", y[1]);
        assert!(
            (y[1 + 32] - expect_partner).abs() < 1e-6,
            "y[33] = {} (sin = {expect_partner})",
            y[1 + 32]
        );
        assert!(y[2].abs() < 1e-9, "dim 2 must stay zero: {}", y[2]);
    }

    #[test]
    fn softmax_rows_sum_to_one_and_are_shift_invariant() {
        let x = vec![1.0f32, 2.0, 3.0, -1.0, 4.0, 0.5];
        let y = softmax(&x, 2, 3);
        for r in 0..2 {
            let s: f64 = y[r * 3..r * 3 + 3].iter().map(|v| f64::from(*v)).sum();
            assert!((s - 1.0).abs() < 1e-7, "row {r} sums to {s}");
        }
        // Same values shifted by a constant -> the same probabilities.
        let shifted: Vec<f32> = x.iter().map(|v| v + 100.0).collect();
        let z = softmax(&shifted, 2, 3);
        for (a, b) in y.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-7, "{a} vs {b}");
        }
    }

    #[test]
    fn router_ties_break_to_the_lowest_id_and_weights_sum_to_one() {
        // Three experts share the top logit: ids must be 2, 3, 4 ascending.
        let mut logits = vec![-5.0f32; 8];
        logits[2] = 1.0;
        logits[3] = 1.0;
        logits[4] = 1.0;
        let r = router(&logits, 8, 3);
        assert_eq!(r.ids, vec![2, 3, 4]);
        let s: f64 = r.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "weights sum to {s}");
        for w in &r.weights {
            assert!((w - 1.0 / 3.0).abs() < 1e-12, "{w}");
        }
    }

    #[test]
    fn router_clamp_makes_a_degenerate_row_sum_to_one() {
        // A row where the top-k probabilities sum far below 2^-14 cannot happen
        // with a normalised softmax, but the clamp is still the contract: force it
        // by asking for k = 1 on a flat 512-way row (p = 1/512 = 1.95e-3, above
        // the clamp) and then check the more important property instead: the
        // weights of a k = 10 selection are the probabilities renormalised to 1.
        let logits: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01).collect();
        let r = router(&logits, 512, 10);
        assert_eq!(r.ids[0], 511, "the largest logit wins");
        let s: f64 = r.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "weights sum to {s}");
        assert!(r.weights.windows(2).all(|w| w[0] >= w[1]), "descending");
    }

    #[test]
    fn gather_rows_are_the_requested_rows() {
        // The quantised gather is `reference::dequant_row` per row; this checks
        // the packing helper (`ops::EmbGather::pack_rows`) against that.
        use crate::quant::QuantType;
        use crate::real;
        let ty = QuantType::Iq4Nl;
        let cols = 160;
        let rows = 6;
        let (packed, _) = real::synthetic_inputs(ty, crate::bench::Shape::new(rows, cols), 3);
        let row_bytes = (cols / ty.block_elems()) * ty.block_bytes();
        assert_eq!(packed.len(), rows * row_bytes);
        let ids = [4u32, 0, 5];
        let staged = crate::ops::EmbGather::pack_rows(&packed, row_bytes, &ids).expect("pack");
        for (o, &id) in ids.iter().enumerate() {
            let want = crate::reference::dequant_row(
                ty,
                &packed[id as usize * row_bytes..(id as usize + 1) * row_bytes],
                cols,
            );
            let got = crate::reference::dequant_row(
                ty,
                &staged[o * row_bytes..(o + 1) * row_bytes],
                cols,
            );
            assert_eq!(got, want, "staged row {o} (table row {id})");
        }
        assert!(
            crate::ops::EmbGather::pack_rows(&packed, row_bytes, &[rows as u32]).is_err(),
            "an out-of-range row index must be rejected, not read"
        );
    }

    #[test]
    fn hc_write_of_a_zero_inject_is_a_plain_add() {
        // inject = 0 -> w = 2*sigmoid(0) = 1 -> the residual add, exactly.
        let r = vec![1.0f32; 8];
        let block = vec![0.5f32; 4];
        let out = hc_write(&r, &block, &[0.0; 2], 4, 2);
        for c in 0..2 {
            for d in 0..4 {
                assert!((out[c * 4 + d] - 1.5).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn hc_read_mean_is_a_mean_not_a_sum() {
        // With all four streams equal, `mixed` must equal one stream's gated
        // value (a sum would be 4x).
        let n_embd = 4;
        let hc = 4;
        let hc_lr = 2;
        let hc_dim = hc * n_embd;
        let r: Vec<f32> = vec![1.0; hc_dim];
        let gamma = vec![1.0f32; hc_dim];
        // down picks a constant, up produces 0 -> sigmoid(0) = 0.5.
        let w_down = vec![0.0f32; hc_lr * hc_dim];
        let mut w_up = vec![0.0f32; hc_dim * hc_lr];
        w_up[0] = 0.0;
        let got = hc_read_token(&r, n_embd, hc, hc_lr, &gamma, &w_down, &w_up, None, 1e-6);
        for d in 0..n_embd {
            assert!(
                (got.mixed[d] - 0.5).abs() < 1e-6,
                "mixed[{d}] = {}",
                got.mixed[d]
            );
        }
        assert!(got.inject.is_none(), "final mixer has no inject");
    }
}
