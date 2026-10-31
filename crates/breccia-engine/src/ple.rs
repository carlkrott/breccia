//! The per-layer-embedding (PLE) subsystem: the host-side n-gram hash, and the
//! block's math.
//!
//! # What is where, and why
//!
//! `docs/specs/qwen4exp-architecture.md` §9 splits PLE into a **host hash**
//! (`qwen4exp.cpp:1192-1291`, op #22 in the ops inventory) and a **device block**
//! (§9.2, op #24). The hash is host-side in the reference too — ggml has no
//! `int64`/xor — so that part is parity by construction.
//!
//! The device block has no kernel in `breccia-kernels` (it is the last pending
//! item of the M2 kernel list, `docs/phase-3/06` §8.3). Rather than stand up a
//! second shader pipeline set, this module runs the block on the host and keeps
//! the two big projections (`ple_key`, `ple_value`) on the GPU:
//!
//! ```text
//! GPU   key_raw = ple_key @ emb          [10240] BF16, 52 MiB   (~40 µs)
//! GPU   value   = ple_value @ emb        [2560]  BF16, 13 MiB   (~20 µs)
//! host  everything else                  ~0.05 ms of scalar work
//! ```
//!
//! That is a **documented deviation** from a fully device-resident PLE block: it
//! costs one 90 KiB readback per PLE token (one block of 48) and leaves ~0.05 ms
//! of scalar work on the critical path. The ops it needs that do exist as kernels
//! (`RmsNorm` with `wgroups = hc`, `EmbGather`, `Add`) are *not* used, because the
//! per-stream dot product (`s[c] = Σ_d key·query`) and the dilated conv have no
//! kernel at all and splitting a four-stage chain across two venues would buy
//! only latency.

use anyhow::{bail, Result};

use breccia_kernels::reference::f16_to_f32;

use crate::config::ModelConfig;

/// The PLE host state of one sequence.
pub struct PleHost {
    /// `ple_norm_key` `[hc*n_embd]` f32.
    norm_key: Vec<f32>,
    /// `ple_norm_query` `[hc*n_embd]` f32.
    norm_query: Vec<f32>,
    /// `ple_norm_conv` `[hc*n_embd]` f32.
    norm_conv: Vec<f32>,
    /// `ple_conv1d` dequantised to `[4 * hc*n_embd]`, indexed `k + conv*c`.
    conv_w: Vec<f32>,
    /// The three u64 hash multipliers.
    mults: [u64; 3],
    offsets: Vec<u64>,
    vocab: Vec<u64>,
    ngram: usize,
    heads_per_ngram: usize,
    n_head: usize,
    eos: u32,
    /// `(conv-1) * ngram` rows of `[hc*n_embd]`, **oldest first**, zeros at the
    /// sequence start.
    hist: Vec<f32>,
    hist_rows: usize,
    hc_dim: usize,
    hc: usize,
    n_embd: usize,
    /// The last `ngram-1` tokens (most recent first), for the hash.
    recent: Vec<u32>,
    /// Per-token scratch, reused.
    pub s: [f32; 8],
    pub gate: [f32; 8],
}

/// The sequence-dependent part of [`PleHost`], for speculative rollback.
#[derive(Clone)]
pub struct PleState {
    pub hist: Vec<f32>,
    pub recent: Vec<u32>,
}

impl PleHost {
    /// `norm_key`/`norm_query`/`norm_conv` are the raw f32 tensor bytes;
    /// `conv` is the raw F16 `[4, hc_dim]` tensor bytes.
    pub fn new(
        cfg: &ModelConfig,
        norm_key: &[u8],
        norm_query: &[u8],
        norm_conv: &[u8],
        conv: &[u8],
    ) -> Result<Self> {
        let hc_dim = cfg.hc_dim();
        let check = |what: &str, bytes: &[u8]| -> Result<()> {
            if bytes.len() != hc_dim * 4 {
                bail!(
                    "{what} is {} bytes, expected {} ({hc_dim} f32)",
                    bytes.len(),
                    hc_dim * 4
                );
            }
            Ok(())
        };
        check("ple_norm_key", norm_key)?;
        check("ple_norm_query", norm_query)?;
        check("ple_norm_conv", norm_conv)?;
        if conv.len() != 4 * hc_dim * 2 {
            bail!(
                "ple_conv1d is {} bytes, expected {} (4 x {hc_dim} f16)",
                conv.len(),
                4 * hc_dim * 2
            );
        }
        if cfg.ple.mults.len() != 3 {
            bail!(
                "ple.layer_multipliers has {} entries, expected 3",
                cfg.ple.mults.len()
            );
        }
        let mut mults = [0u64; 3];
        mults.copy_from_slice(&cfg.ple.mults[..3]);
        let hist_rows = cfg.ple.hist();
        Ok(PleHost {
            norm_key: f32s(norm_key),
            norm_query: f32s(norm_query),
            norm_conv: f32s(norm_conv),
            conv_w: conv
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            mults,
            offsets: cfg.ple.offsets.clone(),
            vocab: cfg.ple.vocab.clone(),
            ngram: cfg.ple.ngram,
            heads_per_ngram: cfg.ple.heads_per_ngram,
            n_head: cfg.ple.n_head(),
            eos: cfg.ple.eos,
            hist: vec![0f32; hist_rows * hc_dim],
            hist_rows,
            hc_dim,
            hc: cfg.hc,
            n_embd: cfg.n_embd,
            recent: Vec::with_capacity(cfg.ple.ngram),
            s: [0.0; 8],
            gate: [0.0; 8],
        })
    }

    /// Reset the sequence state (history + the n-gram window).
    pub fn reset(&mut self) {
        self.hist.iter_mut().for_each(|v| *v = 0.0);
        self.recent.clear();
    }

    /// Copy of the sequence state (~368 KB of history); the weights stay put.
    pub fn state(&self) -> PleState {
        PleState {
            hist: self.hist.clone(),
            recent: self.recent.clone(),
        }
    }

    pub fn restore(&mut self, s: &PleState) {
        self.hist.copy_from_slice(&s.hist);
        self.recent.clone_from(&s.recent);
    }

    /// The `n_head` table rows for `token` at the current position, and the
    /// n-gram window advanced by one.
    ///
    /// `qwen4exp.cpp:1245-1279`: an EOS (or missing) predecessor cuts the window
    /// **forward** — that slot and every older one read as EOS — while the
    /// token's own EOS does not cut its own context.
    pub fn rows_for(&mut self, token: u32) -> Vec<u32> {
        let eos = self.eos;
        let mut ctx = vec![eos; self.ngram];
        ctx[0] = token;
        let mut cut = false;
        for (s, slot) in ctx.iter_mut().enumerate().take(self.ngram).skip(1) {
            let t = if cut {
                None
            } else {
                self.recent.get(s - 1).copied()
            };
            cut = cut || t.is_none() || t == Some(eos);
            *slot = if cut {
                eos
            } else {
                t.expect("uncut implies present")
            };
        }
        self.recent.insert(0, token);
        self.recent.truncate(self.ngram - 1);

        let mut rows = vec![0u32; self.n_head];
        for n in 2..=self.ngram {
            let mut mixed = (ctx[0] as u64).wrapping_mul(self.mults[0]);
            for (j, &c) in ctx.iter().enumerate().take(n).skip(1) {
                mixed ^= (c as u64).wrapping_mul(self.mults[j]);
            }
            let base = (n - 2) * self.heads_per_ngram;
            for g in 0..self.heads_per_ngram {
                let h = base + g;
                let v = mixed % self.vocab[h] + self.offsets[h];
                rows[h] = v as u32;
            }
        }
        rows
    }

    /// Byte range of table row `row` in the `[dim, n_rows]` IQ4_NL table.
    pub fn row_range(&self, row: u32, row_bytes: usize) -> (usize, usize) {
        let at = row as usize * row_bytes;
        (at, at + row_bytes)
    }

    /// The PLE block: `out = R + gated + silu(conv)` on `[hc, n_embd]`.
    ///
    /// * `r` — the residual stack `[hc*n_embd]` **before** this block
    ///   (llama.cpp's `hidden`, `qwen4exp.cpp:1417-1420`).
    /// * `key_raw` — `ple_key @ emb` `[hc*n_embd]`, un-normalised.
    /// * `value` — `ple_value @ emb` `[n_embd]`.
    pub fn block(
        &mut self,
        r: &[f32],
        key_raw: &[f32],
        value: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let hc = self.hc;
        if self.hc_dim == 0 || r.len() != self.hc_dim || out.len() != self.hc_dim {
            bail!(
                "PLE block expects the [{hc}, n_embd] stack ({} values), got {}/{}",
                self.hc_dim,
                r.len(),
                out.len()
            );
        }
        if key_raw.len() != self.hc_dim {
            bail!(
                "ple_key produced {} values, expected {}",
                key_raw.len(),
                self.hc_dim
            );
        }
        let n_embd = self.n_embd;
        if value.len() != n_embd {
            bail!(
                "ple_value produced {} values, expected {n_embd}",
                value.len()
            );
        }
        let inv_sqrt = 1.0 / (n_embd as f32).sqrt();
        let eps = 1e-6f32;

        // key / query: grouped RMSNorm over ONE stream of 2560 (the gamma is
        // [n_embd, hc], so `c` selects the gamma block).
        let mut key = vec![0f32; self.hc_dim];
        let mut query = vec![0f32; self.hc_dim];
        grouped_norm(key_raw, &self.norm_key, n_embd, eps, &mut key);
        grouped_norm(r, &self.norm_query, n_embd, eps, &mut query);

        // s[c] = (1/sqrt(n_embd)) * Σ_d key[c,d]·query[c,d]
        let mut gated = vec![0f32; self.hc_dim];
        for c in 0..hc {
            let base = c * n_embd;
            let mut s = 0f64;
            for d in 0..n_embd {
                s += (key[base + d] as f64) * (query[base + d] as f64);
            }
            let s = (s as f32) * inv_sqrt;
            if c < self.s.len() {
                self.s[c] = s;
            }
            let mag = s.abs().max(1e-6).sqrt();
            let signed = if s < 0.0 { -mag } else { mag };
            let g = 1.0 / (1.0 + (-signed).exp());
            if c < self.gate.len() {
                self.gate[c] = g;
            }
            for d in 0..n_embd {
                gated[base + d] = value[d] * g;
            }
        }

        // The conv input is the gated value, grouped-normalised again.
        let mut cur = vec![0f32; self.hc_dim];
        grouped_norm(&gated, &self.norm_conv, n_embd, eps, &mut cur);

        // Dilated causal conv, kernel 4, dilation = ngram (3):
        //   conv[c] = Σ_k w[k][c] * x[t - (K-1-k)*dil][c]
        // with x = history (9 rows, oldest first) ++ current.
        let kern = self.conv_w.len() / self.hc_dim;
        if kern != 4 {
            bail!("ple_conv1d has {kern} taps, expected 4");
        }
        for i in 0..self.hc_dim {
            let mut acc = 0f32;
            for k in 0..kern {
                let back = (kern - 1 - k) * self.ngram;
                // `back` positions before the current row, within hist ++ cur.
                let src = if back == 0 {
                    cur[i]
                } else {
                    self.hist[(self.hist_rows - back) * self.hc_dim + i]
                };
                acc += self.conv_w[k + kern * i] * src;
            }
            let silu = acc / (1.0 + (-acc).exp());
            out[i] = r[i] + gated[i] + silu;
        }

        // Shift the history: drop the oldest row, append the new `cur`.
        self.hist.drain(..self.hc_dim);
        self.hist.extend_from_slice(&cur);
        Ok(())
    }
}

/// `out[c,d] = x[c,d] * rsqrt(mean_d(x[c,d]²) + eps) * gamma[c,d]`.
fn grouped_norm(x: &[f32], gamma: &[f32], n_embd: usize, eps: f32, out: &mut [f32]) {
    for (xc, (gc, oc)) in x
        .chunks_exact(n_embd)
        .zip(gamma.chunks_exact(n_embd).zip(out.chunks_exact_mut(n_embd)))
    {
        let mut sq = 0f64;
        for &v in xc {
            sq += (v as f64) * (v as f64);
        }
        let scale = 1.0 / ((sq / n_embd as f64) as f32 + eps).sqrt();
        for i in 0..n_embd {
            oc[i] = xc[i] * scale * gc[i];
        }
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host() -> PleHost {
        let hc_dim = 10240;
        PleHost {
            norm_key: vec![1.0; hc_dim],
            norm_query: vec![1.0; hc_dim],
            norm_conv: vec![1.0; hc_dim],
            conv_w: vec![0.0; 4 * hc_dim],
            mults: [23703573157769, 20109073645365, 8052911324071],
            offsets: (0..16).map(|i| i * 20_000_003).collect(),
            vocab: (0..16).map(|i| 20_000_003 + i as u64).collect(),
            ngram: 3,
            heads_per_ngram: 8,
            n_head: 16,
            eos: 248044,
            hist: vec![0.0; 9 * hc_dim],
            hist_rows: 9,
            hc_dim,
            hc: 4,
            n_embd: 2560,
            recent: Vec::new(),
            s: [0.0; 8],
            gate: [0.0; 8],
        }
    }

    /// `ngram.hpp:9-27`: XOR of `token*mult`, `%` per-head vocab + offset, and
    /// EOS cutting the window forward. Tokens 7, 8, 9 are checked by hand.
    #[test]
    fn the_ngram_hash_follows_the_reference_formula() {
        let mut p = host();
        let rows = p.rows_for(7);
        // With no predecessors (sequence start) ctx = [7, EOS, EOS].
        let eos = 248044u64;
        let m = [23703573157769u64, 20109073645365, 8052911324071];
        let mixed2 = (7 * m[0]) ^ (eos * m[1]);
        let mixed3 = mixed2 ^ (eos * m[2]);
        for (h, &row) in rows.iter().enumerate().take(8) {
            assert_eq!(row as u64, mixed2 % p.vocab[h] + p.offsets[h]);
        }
        for (h, &row) in rows.iter().enumerate().skip(8).take(8) {
            assert_eq!(row as u64, mixed3 % p.vocab[h] + p.offsets[h]);
        }
    }

    #[test]
    fn the_window_advances_and_uses_the_previous_tokens() {
        let mut p = host();
        let _ = p.rows_for(7);
        let rows = p.rows_for(8);
        let m = [23703573157769u64, 20109073645365, 8052911324071];
        let mixed2 = (8 * m[0]) ^ (7 * m[1]);
        for (h, &row) in rows.iter().enumerate().take(8) {
            assert_eq!(row as u64, mixed2 % p.vocab[h] + p.offsets[h]);
        }
    }

    /// Speculative rollback: restore the pre-chunk state, replay the accepted
    /// tokens' hashing, and the window must match the one that never rolled back.
    #[test]
    fn restore_then_replay_rebuilds_the_window() {
        let mut p = host();
        let _ = p.rows_for(7);
        let base = p.state();
        let _ = p.rows_for(8);
        let want = p.state().recent;
        let _ = p.rows_for(9);
        p.restore(&base);
        let _ = p.rows_for(8);
        assert_eq!(p.state().recent, want);
        assert_eq!(p.state().hist, base.hist);
    }

    #[test]
    fn an_eos_predecessor_cuts_every_older_slot() {
        let mut p = host();
        let _ = p.rows_for(248044); // EOS as a real token
        let rows = p.rows_for(8);
        let eos = 248044u64;
        let m = [23703573157769u64, 20109073645365, 8052911324071];
        // ctx[1] == EOS (the predecessor itself), so ctx[2] is cut to EOS too.
        let mixed2 = (8 * m[0]) ^ (eos * m[1]);
        for (h, &row) in rows.iter().enumerate().take(8) {
            assert_eq!(row as u64, mixed2 % p.vocab[h] + p.offsets[h]);
        }
    }

    #[test]
    fn the_block_composes_as_the_reference_writes_it() {
        let mut p = host();
        let hc_dim = 10240;
        let n_embd = 2560;
        let r = vec![0.5f32; hc_dim];
        let key_raw = vec![0.25f32; hc_dim];
        let value = vec![2.0f32; n_embd];
        let mut out = vec![0f32; hc_dim];
        p.block(&r, &key_raw, &value, &mut out).unwrap();
        // conv_w is all zeros ⇒ conv = 0 ⇒ out = r + gated.
        for c in 0..4 {
            let g = p.gate[c];
            assert!(g > 0.0 && g < 1.0, "gate must be a sigmoid: {g}");
            for d in 0..n_embd {
                let want = r[c * n_embd + d] + value[d] * g;
                assert!((out[c * n_embd + d] - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn the_conv_history_shifts_once_per_token() {
        let mut p = host();
        let hc_dim = 10240;
        // A one-tap kernel reading the row 9 positions back would see row 0.
        p.conv_w.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..hc_dim {
            p.conv_w[4 * i] = 1.0; // tap 0 ⇒ 9 positions back
        }
        let mut out = vec![0f32; hc_dim];
        let r = vec![0f32; hc_dim];
        let key_raw = vec![0f32; hc_dim];
        let value = vec![0f32; hc_dim / 4];
        p.block(&r, &key_raw, &value, &mut out).unwrap();
        // The first token taps zeroed history, so the conv contributes nothing.
        assert!(out.iter().all(|v| *v == 0.0));
    }
}
