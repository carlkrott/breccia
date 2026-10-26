//! Two-stream decode: one sequence per card, interleaved at the **block** level.
//!
//! # The problem this solves
//!
//! A single decode stream alternates: 48 GPU phases, each followed by a host
//! expert lane. Measured (§8n) the GPU is busy 43 ms/token and the host lane 38
//! ms/token, and they are **strictly serial** — block *b+1*'s GPU work cannot be
//! recorded until the host has written block *b*'s MoE result into the mailbox,
//! because the residual stream is one dependency chain.
//!
//! The only independent work available is **another sequence**. So: two sessions,
//! each with the whole model on one card (stream A on the MI50, stream B on the
//! RX), driven block by block:
//!
//! ```text
//! submit A(0) ─┐
//! submit B(0) ─┴─▶ wait A(0) · lane A(0) · submit A(1)   ← B(0) runs on the RX
//!                 wait B(0) · lane B(0) · submit B(1)   ← A(1) runs on the MI50
//!                 wait A(1) · lane A(1) · submit A(2)   ← B(1) runs on the RX
//! ```
//!
//! Each lane hides the *other* stream's GPU phase, so a block pair costs
//! `max(lane, gpu)` instead of `lane + gpu`. Nothing is duplicated: a stream is
//! exactly [`Session`], which already owns a model, its state and its pipelines;
//! the only shared object is the host expert lane (16 threads, used serially).
//!
//! # Why one card per stream
//!
//! Blocks are serial *within* a stream, so the token costs the sum of its blocks.
//! With one placement shared by both streams, block *b* of both streams lands on
//! the same card and the two GPU phases serialise there — measured 114 ms per
//! token pair. Giving each stream its own card makes the two GPU phases genuinely
//! concurrent (§8o has both numbers).

use anyhow::{bail, Result};

use crate::graph::{Session, SessionCfg};
use crate::model::Model;
use crate::stats::LayerTrace;

/// Per-stream decode result.
#[derive(Clone, Debug, Default)]
pub struct StreamReport {
    pub tokens: Vec<u32>,
    /// Forward passes this stream ran for the decode phase.
    pub forwards: usize,
    /// Host seconds the stream's expert lane took, and the share of that which
    /// ran while **another stream's** GPU work was in flight.
    pub lane_s: f64,
    pub lane_covered_s: f64,
    /// Host seconds the stream spent waiting for its own GPU.
    pub fence_s: f64,
    pub expert_bytes: u64,
    pub last_logits: Vec<f32>,
}

/// The whole two-stream run.
#[derive(Clone, Debug, Default)]
pub struct MultiGeneration {
    pub streams: Vec<StreamReport>,
    /// Wall seconds of the decode loop (prompt feeding excluded).
    pub decode_seconds: f64,
    /// Tokens **per stream** produced in that window.
    pub steps: usize,
}

impl MultiGeneration {
    /// Tokens/second across all streams — the number that matters.
    pub fn aggregate_tokens_per_s(&self) -> f64 {
        if self.decode_seconds > 0.0 {
            self.steps as f64 * self.streams.len() as f64 / self.decode_seconds
        } else {
            0.0
        }
    }
    /// Tokens/second of one stream (every stream advances in lockstep).
    pub fn stream_tokens_per_s(&self) -> f64 {
        if self.decode_seconds > 0.0 {
            self.steps as f64 / self.decode_seconds
        } else {
            0.0
        }
    }
    /// **Latency** of one stream's token: the block-pair time, which the
    /// interleave does not shorten (it buys throughput, not latency).
    pub fn stream_ms_per_token(&self) -> f64 {
        if self.steps > 0 {
            self.decode_seconds * 1e3 / self.steps as f64
        } else {
            0.0
        }
    }
    /// Fraction of the host lane that ran under another stream's GPU work.
    pub fn overlap_efficiency(&self) -> f64 {
        let lane: f64 = self.streams.iter().map(|s| s.lane_s).sum();
        let covered: f64 = self.streams.iter().map(|s| s.lane_covered_s).sum();
        if lane > 0.0 {
            covered / lane
        } else {
            0.0
        }
    }
    pub fn lane_ms_per_step(&self) -> f64 {
        if self.steps > 0 {
            self.streams.iter().map(|s| s.lane_s).sum::<f64>() * 1e3 / self.steps as f64
        } else {
            0.0
        }
    }
    pub fn fence_ms_per_step(&self) -> f64 {
        if self.steps > 0 {
            self.streams.iter().map(|s| s.fence_s).sum::<f64>() * 1e3 / self.steps as f64
        } else {
            0.0
        }
    }
}

/// N independent sequences over N single-device sessions.
pub struct Multi<'m> {
    pub streams: Vec<Session<'m>>,
    /// Host seconds in [`Session::finish_block`] / [`Session::submit_block`]
    /// (the block-pair accounting behind the overlap number).
    pub t_finish: f64,
    pub t_submit: f64,
}

impl<'m> Multi<'m> {
    /// One session per model. Each model must be a single-device placement (that
    /// is what makes the streams' GPU work concurrent); the expert lane is built
    /// once and shared.
    pub fn new(models: &[&'m Model], cfg: &SessionCfg) -> Result<Self> {
        if models.is_empty() {
            bail!("a multi-stream session needs at least one model");
        }
        for (i, m) in models.iter().enumerate() {
            if m.devices.len() != 1 {
                bail!(
                    "stream {i} has {} devices: a stream must own exactly one, so that the \
                     streams' GPU phases are concurrent rather than queued behind each other",
                    m.devices.len()
                );
            }
        }
        let mut cfg = cfg.clone();
        let mut streams: Vec<Session<'m>> = Vec::with_capacity(models.len());
        for (i, m) in models.iter().enumerate() {
            // The first session builds the lane; every later one shares it, so
            // N streams use `threads` workers, not N*threads.
            let s = Session::new(m, &cfg)
                .map_err(|e| anyhow::anyhow!("building stream {i}'s session: {e:#}"))?;
            if cfg.pool.is_none() {
                cfg.pool = Some(std::sync::Arc::clone(s.pool()));
            }
            streams.push(s);
        }
        Ok(Multi {
            streams,
            t_finish: 0.0,
            t_submit: 0.0,
        })
    }

    pub fn len(&self) -> usize {
        self.streams.len()
    }
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// One token for **every** stream, interleaved at the block level.
    ///
    /// `pos` is the position all streams are at (they advance in lockstep, so a
    /// stream's own `pos` is the same).
    pub fn token_all(&mut self, toks: &[u32], pos: usize) -> Result<Vec<Vec<f32>>> {
        let n = self.streams.len();
        if toks.len() != n {
            bail!("{} token(s) for {n} stream(s)", toks.len());
        }
        for (sess, &t) in self.streams.iter_mut().zip(toks.iter()) {
            sess.begin_token(t, pos)?;
        }
        let n_layers = self.streams[0].model.cfg.n_layers;
        let mut no_trace: Option<&mut Vec<LayerTrace>> = None;
        let mut digests = Vec::with_capacity(n);
        for sess in self.streams.iter_mut() {
            digests.push(sess.submit_block(0, &mut no_trace)?);
        }
        for b in 0..n_layers {
            #[allow(clippy::needless_range_loop)]
            for s in 0..n {
                // Was another stream's GPU work in flight while this lane runs?
                let covered = (0..n).any(|o| o != s && self.streams[o].in_flight());
                let d = std::mem::take(&mut digests[s]);
                let t0 = std::time::Instant::now();
                self.streams[s].finish_block(b, d, &mut no_trace, covered)?;
                let t1 = std::time::Instant::now();
                if b + 1 < n_layers {
                    digests[s] = self.streams[s].submit_block(b + 1, &mut no_trace)?;
                }
                let t2 = std::time::Instant::now();
                self.t_finish += t1.duration_since(t0).as_secs_f64();
                self.t_submit += t2.duration_since(t1).as_secs_f64();
            }
        }
        // The per-stream heads interleave too: submit both, then read both.
        for sess in self.streams.iter_mut() {
            sess.submit_head()?;
        }
        let mut out = Vec::with_capacity(n);
        for sess in self.streams.iter_mut() {
            out.push(sess.finish_head()?);
        }
        Ok(out)
    }

    /// Feed the prompt to every stream, then generate `n` greedy tokens each.
    ///
    /// Decode timing covers only the forwards that produced generated tokens, so
    /// the prompt is not part of the token rate (same rule as the single-stream
    /// bench).
    pub fn generate(&mut self, prompt: &[u32], n: usize) -> Result<MultiGeneration> {
        if prompt.is_empty() {
            bail!("the prompt is empty");
        }
        let ns = self.streams.len();
        for s in 0..ns {
            self.streams[s].reset()?;
        }
        let mut logits = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            logits = self.token_all(&vec![t; ns], pos)?;
        }
        for s in 0..ns {
            self.streams[s].reset_counters()?;
        }
        crate::graph::batch_cmd_reset();
        self.t_finish = 0.0;
        self.t_submit = 0.0;
        let mut streams: Vec<StreamReport> = vec![StreamReport::default(); ns];
        let mut pos = prompt.len();
        let t0 = std::time::Instant::now();
        let mut steps = 0usize;
        #[allow(clippy::explicit_counter_loop)]
        for i in 0..n {
            let next: Vec<u32> = (0..ns)
                .map(|s| self.streams[s].sample(&logits[s]).unwrap_or(0))
                .collect();
            for (st, &t) in streams.iter_mut().zip(next.iter()) {
                st.tokens.push(t);
            }
            if i + 1 == n {
                break;
            }
            logits = self.token_all(&next, pos)?;
            pos += 1;
            steps += 1;
        }
        let decode_seconds = t0.elapsed().as_secs_f64();
        for (s, (st, sess)) in streams.iter_mut().zip(self.streams.iter()).enumerate() {
            st.forwards = sess.forwards;
            st.lane_s = sess.lane_s;
            st.lane_covered_s = sess.lane_covered_s;
            st.fence_s = sess.dev_wait_s.iter().sum();
            st.expert_bytes = sess.expert_bytes;
            st.last_logits = logits.get(s).cloned().unwrap_or_default();
        }
        Ok(MultiGeneration {
            streams,
            decode_seconds,
            steps,
        })
    }
}
