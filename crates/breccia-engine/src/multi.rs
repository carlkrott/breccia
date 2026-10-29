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
    /// Wall seconds of this stream's own prompt pass. The serial chunked path
    /// times each stream alone; token by token every stream shares the total.
    pub prefill_seconds: f64,
    pub lane_stats: Option<crate::LaneReport>,
}

/// The whole two-stream run.
#[derive(Clone, Debug, Default)]
pub struct MultiGeneration {
    pub streams: Vec<StreamReport>,
    /// Wall seconds of the decode loop (prompt feeding excluded).
    pub decode_seconds: f64,
    /// Tokens **per stream** produced in that window.
    pub steps: usize,
    /// Wall seconds of each decode step: every stream produces one token in it,
    /// so this is the per-token latency each agent felt.
    pub steps_s: Vec<f64>,
    /// Wall seconds the prompt pass took (token by token, or chunked when the
    /// session was built with a `--prefill-chunk`).
    pub prefill_seconds: f64,
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

/// `BRECCIA_BLOCK_MAJOR=1`: submit **all** streams' block *b+1* only after the
/// last stream's block-*b* lane, instead of each stream submitting its own right
/// after its lane. The lanes then read block *b*'s expert blob back to back with
/// no fresh GPU submission (and no *b+1* dense weights) in between — the shape
/// the shape that hypothesis wants (§2 of the Vulkan levers assessment). Off unless
/// the variable is exactly `1` (any other value, incl. `0`/empty, is OFF) — an
/// OFF|ON|OFF plateau script must not have its OFF arm silently run ON. The
/// schedule it changes touches no numerics: block *b*'s dependency chain
/// is unchanged, only *when* the next block is handed to the queue. Cached.
fn block_major_env_on(value: Option<&str>) -> bool {
    value == Some("1")
}

/// `BRECCIA_MULTI_PREFILL_CONCURRENT=1`: the chunked prompt pass runs the streams on
/// parallel threads instead of one after another. Off unless exactly `1`, so the
/// serial path stays the default.
fn prefill_concurrent_on(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Only the literal `1` decodes the streams in their own threads (no block lockstep).
fn decode_concurrent_on(value: Option<&str>) -> bool {
    value == Some("1")
}

fn block_major_order() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| block_major_env_on(std::env::var("BRECCIA_BLOCK_MAJOR").ok().as_deref()))
}

/// When block *b+1* is recorded, relative to the lanes of block *b*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Submit {
    /// After this stream's own lane — the default stream-major shape.
    Now,
    /// Once every stream's block-*b* lane has run.
    AtBlockEnd,
    /// Last block: there is no *b+1* to record.
    Never,
}

fn submit_plan(block_major: bool, b: usize, n_layers: usize) -> Submit {
    if b + 1 >= n_layers {
        Submit::Never
    } else if block_major {
        Submit::AtBlockEnd
    } else {
        Submit::Now
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
        let block_major = block_major_order();
        let mut no_trace: Option<&mut Vec<LayerTrace>> = None;
        let mut digests = Vec::with_capacity(n);
        for sess in self.streams.iter_mut() {
            digests.push(sess.submit_block(0, &mut no_trace)?);
        }
        for b in 0..n_layers {
            let plan = submit_plan(block_major, b, n_layers);
            #[allow(clippy::needless_range_loop)]
            for s in 0..n {
                // Was another stream's GPU work in flight while this lane runs?
                let covered = (0..n).any(|o| o != s && self.streams[o].in_flight());
                let d = std::mem::take(&mut digests[s]);
                let t0 = std::time::Instant::now();
                self.streams[s].finish_block(b, d, &mut no_trace, covered)?;
                let t1 = std::time::Instant::now();
                if plan == Submit::Now {
                    digests[s] = self.streams[s].submit_block(b + 1, &mut no_trace)?;
                    self.t_submit += t1.elapsed().as_secs_f64();
                }
                self.t_finish += t1.duration_since(t0).as_secs_f64();
            }
            // Block-major: every stream's *b+1* goes in only now, so the lanes
            // above all read block *b*'s weights with nothing new submitted.
            // shortcut: only the *submit* point moves — a batch the recorder
            // closes at SOFT_LIMIT still flushes mid-lane, and with one stream
            // there is nothing to group; upgrade if a stream's lanes still get
            // evicted by its own partial batch.
            if plan == Submit::AtBlockEnd {
                for s in 0..n {
                    let t1 = std::time::Instant::now();
                    digests[s] = self.streams[s].submit_block(b + 1, &mut no_trace)?;
                    self.t_submit += t1.elapsed().as_secs_f64();
                }
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
    ///
    /// Parity note: with a `--prefill-chunk` set the prompt pass uses `prefill`
    /// over `prefill_plan` instead of token-by-token `token_all`; the arithmetic
    /// is the same grouped-lane forward (§8s), so greedy tokens should match.
    /// That chunk-0-vs-64 parity check needs a model fixture, which the repo does
    /// not ship — it runs on a real model, e.g. via the C3 sweep arms.
    pub fn generate(&mut self, prompt: &[u32], n: usize) -> Result<MultiGeneration> {
        if prompt.is_empty() {
            bail!("the prompt is empty");
        }
        let ns = self.streams.len();
        for s in 0..ns {
            self.streams[s].reset()?;
        }
        // Coordinator cores (BRECCIA_CPU_PIN): this thread and the prefill scope
        // threads it spawns. shortcut: prefetch_rows' scoped threads inherit the
        // same mask; give them their own set if PLE faults show up on the coordinator.
        if let Some(pin) = breccia_kernels::cpu::cpu_pin().filter(|p| !p.coord.is_empty()) {
            breccia_kernels::cpu::pin_current(&pin.coord);
        }
        // Chunked prompt pass (§8s), the multi-stream twin of the single-session
        // `generate`: same tokens, same arithmetic, the grouped expert lane per
        // chunk. `0` stays token by token through `token_all`, the interleaved
        // path (chunked prefill needs a single-device session per stream, which
        // is what `Multi::new` already requires).
        let mut logits = Vec::new();
        let prefill_t = std::time::Instant::now();
        // Plan from the --prefill-chunk target: prefill_plan applies prefill_cap itself,
        // so passing the cap would plan chunks 2.25x the target and Session::prefill bails.
        let target = self.streams[0].prefill_target;
        let mut stream_prefill = vec![0.0f64; ns];
        let concurrent = prefill_concurrent_on(
            std::env::var("BRECCIA_MULTI_PREFILL_CONCURRENT")
                .ok()
                .as_deref(),
        );
        if target > 0 && concurrent {
            // One scoped thread per stream: the streams' chunks overlap instead of
            // stream 1 waiting for stream 0's whole prompt.
            let plan = crate::graph::prefill_plan(prompt.len(), target);
            let done: Vec<Result<(Vec<f32>, f64)>> = std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .streams
                    .iter_mut()
                    .map(|sess| {
                        let plan = &plan;
                        scope.spawn(move || -> Result<(Vec<f32>, f64)> {
                            let t_stream = std::time::Instant::now();
                            let mut pos = 0usize;
                            let mut lg = Vec::new();
                            for &len in plan.iter() {
                                lg = sess.prefill(&prompt[pos..pos + len], pos)?;
                                pos += len;
                            }
                            Ok((lg, t_stream.elapsed().as_secs_f64()))
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("prefill thread panicked"))
                    .collect()
            });
            logits = Vec::with_capacity(ns);
            for (s, r) in done.into_iter().enumerate() {
                let (lg, secs) = r?;
                logits.push(lg);
                stream_prefill[s] = secs;
            }
        } else if target > 0 {
            logits = vec![Vec::new(); ns];
            for (s, sess) in self.streams.iter_mut().enumerate() {
                let t_stream = std::time::Instant::now();
                let mut pos = 0usize;
                for len in crate::graph::prefill_plan(prompt.len(), target) {
                    logits[s] = sess.prefill(&prompt[pos..pos + len], pos)?;
                    pos += len;
                }
                stream_prefill[s] = t_stream.elapsed().as_secs_f64();
            }
        } else {
            for (pos, &t) in prompt.iter().enumerate() {
                logits = self.token_all(&vec![t; ns], pos)?;
            }
        }
        let prefill_seconds = prefill_t.elapsed().as_secs_f64();
        if target == 0 {
            // Token by token, every stream is fed inside the same loop.
            stream_prefill.iter_mut().for_each(|p| *p = prefill_seconds);
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
        let mut steps_s: Vec<f64> = Vec::with_capacity(n);
        let concurrent_decode = ns > 1
            && decode_concurrent_on(
                std::env::var("BRECCIA_MULTI_DECODE_CONCURRENT")
                    .ok()
                    .as_deref(),
            );
        if concurrent_decode {
            // One thread per stream, no block lockstep: the shared pool's Mutex still
            // runs one lane at a time, but each stream's fence waits, readbacks and
            // submits now overlap the other stream's lane instead of queueing behind it.
            let firsts = std::mem::take(&mut logits);
            let done: Vec<Result<(Vec<u32>, Vec<f32>, Vec<f64>)>> = std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .streams
                    .iter_mut()
                    .zip(firsts)
                    .map(|(sess, mut lg)| {
                        scope.spawn(move || -> Result<(Vec<u32>, Vec<f32>, Vec<f64>)> {
                            let mut toks = Vec::with_capacity(n);
                            let mut times = Vec::with_capacity(n);
                            let mut pos = prompt.len();
                            // The GPU-time probe accumulates per thread: base and
                            // capture on this one, or the report reads 0.
                            sess.snapshot_probe();
                            for i in 0..n {
                                let t = sess.sample(&lg).unwrap_or(0);
                                toks.push(t);
                                if i + 1 == n {
                                    break;
                                }
                                let t_step = std::time::Instant::now();
                                lg = sess.token(t, pos, None)?;
                                times.push(t_step.elapsed().as_secs_f64());
                                pos += 1;
                            }
                            sess.capture_probe();
                            Ok((toks, lg, times))
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("decode thread panicked"))
                    .collect()
            });
            for (s, r) in done.into_iter().enumerate() {
                let (toks, lg, times) = r?;
                streams[s].tokens = toks;
                logits.push(lg);
                if s == 0 {
                    steps = times.len();
                    steps_s = times;
                }
            }
        }
        #[allow(clippy::explicit_counter_loop)]
        for i in 0..if concurrent_decode { 0 } else { n } {
            let next: Vec<u32> = (0..ns)
                .map(|s| self.streams[s].sample(&logits[s]).unwrap_or(0))
                .collect();
            for (st, &t) in streams.iter_mut().zip(next.iter()) {
                st.tokens.push(t);
            }
            if i + 1 == n {
                break;
            }
            let t_step = std::time::Instant::now();
            logits = self.token_all(&next, pos)?;
            steps_s.push(t_step.elapsed().as_secs_f64());
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
            st.prefill_seconds = stream_prefill[s];
            st.lane_stats = sess.lane_report();
        }
        Ok(MultiGeneration {
            streams,
            decode_seconds,
            steps,
            steps_s,
            prefill_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{block_major_env_on, prefill_concurrent_on, submit_plan, Submit};

    /// Only the literal `1` turns the concurrent prefill on; anything else keeps the serial path.
    #[test]
    fn prefill_concurrent_env_gate_is_exactly_one() {
        assert!(prefill_concurrent_on(Some("1")));
        assert!(super::decode_concurrent_on(Some("1")));
        assert!(!super::decode_concurrent_on(None) && !super::decode_concurrent_on(Some("0")));
        for v in [None, Some("0"), Some(""), Some("true"), Some("1 ")] {
            assert!(!prefill_concurrent_on(v), "{v:?} must be OFF");
        }
    }

    /// The multi prompt pass must cover the prompt and never plan a chunk above
    /// the session's scratch cap, for every target the sweep uses.
    #[test]
    fn multi_prefill_plan_fits_session_cap() {
        for target in [64usize, 128, 256] {
            let cap = crate::graph::prefill_cap(target);
            for len in 1..=2048usize {
                let plan = crate::graph::prefill_plan(len, target);
                assert_eq!(plan.iter().sum::<usize>(), len, "len {len} target {target}");
                assert!(
                    plan.iter().all(|&c| c <= cap),
                    "len {len} target {target}: {plan:?} exceeds cap {cap}"
                );
            }
        }
    }

    /// Only the literal `1` enables: `0`, empty and anything else must read as
    /// OFF, or an OFF|ON|OFF plateau's OFF arm would silently run ON.
    #[test]
    fn block_major_env_gate_is_exactly_one() {
        assert!(block_major_env_on(Some("1")));
        assert!(!block_major_env_on(None));
        for off in ["0", "", "true", "01", "1 ", "on"] {
            assert!(!block_major_env_on(Some(off)), "{off:?} must be OFF");
        }
    }

    /// Unset (`block_major = false`) must keep the stream-major shape exactly:
    /// each stream records its own *b+1* right after its own lane.
    #[test]
    fn default_submits_after_each_stream_lane() {
        for b in 0..47 {
            assert_eq!(submit_plan(false, b, 48), Submit::Now);
        }
    }

    /// On, every block is handed in once all its lanes have run — and the last
    /// block still has nothing to record, or the loop would submit past the end.
    #[test]
    fn block_major_defers_to_the_block_end() {
        for b in 0..47 {
            assert_eq!(submit_plan(true, b, 48), Submit::AtBlockEnd);
        }
        assert_eq!(submit_plan(true, 47, 48), Submit::Never);
        assert_eq!(submit_plan(false, 47, 48), Submit::Never);
        assert_eq!(submit_plan(true, 0, 1), Submit::Never);
    }
}
