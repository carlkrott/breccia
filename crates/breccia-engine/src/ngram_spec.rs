//! Greedy generation with a **weight-free n-gram (prompt-lookup) draft**: no
//! model, no weights, no GPU work — the draft is a lookup in the tokens already
//! seen, and [`Session::verify`] (exact, greedy) decides.
//!
//! Why it exists: MTP costs a draft step (~3.5-3.9 ms after the arena fix) before
//! every verify, while a suffix match costs one hash lookup. The verify itself is
//! priced by [`crate::mtp_spec`]'s bench (T=2 1.39-1.53x decode on f2a70ed), so a
//! drafter only has to beat `V(T)/T` in *accepted tokens per round*: at T=2 one
//! accepted draft already pays (2.0 > 1.53). Copied context — the system prompt
//! and tool schemas an agent resends every turn — is exactly the case a suffix
//! match wins, and it needs no second model file.
//!
//! Shape follows llama.cpp's `--spec-type ngram*` and strata's
//! `suffix_drafter.hpp`: a hash of the last `n` tokens maps to the token that
//! followed it most recently; the draft chains that map (propose, extend, propose),
//! so a copied span yields a full-length draft. Every rejection is repaired by
//! `rollback_to`, so a wrong draft costs verify time, never correctness.

use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::graph::{prefill_plan, Session};
use crate::mtp_spec::{argmax, log_p, Accept, SpecStats, MAX_K};

/// Default look-back window, in tokens (llama.cpp `ngram-simple` uses 3).
pub const DEFAULT_N: usize = 3;

/// Visible-cell count above which the QSA indexer stops using its dense path
/// (`IDX_TOPK_BLOCKS * IDX_BLOCK` = 512 blocks of 4 cells, `breccia-kernels`
/// `qsa/mod.rs`). Below it every block is selected and a T-row verify matches
/// decode to ~1e-4 at any depth; above it the top-512 selection runs and, as
/// measured, verify at T >= 4 disagrees with decode by ~2.1 logits and T = 3
/// still drifts over ~20 greedy tokens (audit B12). Past this point the drafter
/// caps its depth at one draft; the gate still decides whether to draft at all.
pub const IDX_DENSE_CELLS: usize = 2048;

/// The suffix index over one token stream.
///
/// `map` is keyed on the last [`NgramDrafter::n`] tokens packed into a `u128`
/// (exact — at `n <= 4` no aliasing, so a hit is always a real match) and holds
/// the token that followed the most recent occurrence. Later occurrences
/// overwrite earlier ones, which is what makes the draft follow the *latest*
/// repetition rather than the first.
pub struct NgramDrafter {
    n: usize,
    map: HashMap<u128, u32>,
    /// Number of positions already indexed: `done` is the index whose state
    /// (`toks[done - n .. done]`) has been entered.
    done: usize,
}

impl NgramDrafter {
    pub fn new(n: usize) -> Self {
        NgramDrafter {
            n: n.clamp(2, 4),
            map: HashMap::new(),
            done: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Pack `toks[upto - n .. upto]` into one key. The caller guarantees `upto >= n`.
    fn state(&self, toks: &[u32], upto: usize) -> u128 {
        let mut k = 0u128;
        for &t in &toks[upto - self.n..upto] {
            k = (k << 32) | t as u128;
        }
        k
    }

    fn shift(&self, state: u128, next: u32) -> u128 {
        let mask = if self.n >= 4 {
            u128::MAX
        } else {
            (1u128 << (32 * self.n)) - 1
        };
        ((state << 32) | next as u128) & mask
    }

    /// Index every position whose successor is now known. Idempotent and
    /// incremental: pass the whole stream each round and only the new tail is
    /// hashed (`done` is the cursor).
    pub fn observe(&mut self, toks: &[u32]) {
        if toks.len() < self.n {
            return;
        }
        self.done = self.done.max(self.n);
        while self.done < toks.len() {
            let i = self.done;
            let key = self.state(toks, i);
            self.map.insert(key, toks[i]);
            self.done += 1;
        }
    }

    /// Up to `k` tokens the stream continued with after its last occurrence of
    /// the current suffix. Empty when the suffix is unseen.
    pub fn draft(&self, toks: &[u32], k: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(k);
        if toks.len() < self.n || k == 0 {
            return out;
        }
        let mut state = self.state(toks, toks.len());
        for _ in 0..k {
            let Some(&next) = self.map.get(&state) else {
                break;
            };
            out.push(next);
            state = self.shift(state, next);
        }
        out
    }
}

/// Adaptive depth gate: draft only while the measured gain pays for the verify.
///
/// The break-even is not a constant — it is `V(k+1) / decode`, and `V` depends on
/// the card, the build and the block placement (measured 1.39-1.53x at T=2,
/// 2.45x at T=4, 3.11-3.31x at T=6 on f2a70ed, RX alone). So instead of a table,
/// this measures both sides on the real stream: an EWMA of a plain decode step,
/// an EWMA of a verify of the same depth, and an EWMA of accepted drafts. A round
/// drafts iff `(1 + E[accepted]) * plain / verify > 1 + margin`; when it stops
/// paying it goes plain and probes once every `PROBE` rounds, because acceptance
/// is bursty (a copied span can appear mid-generation). This is what turns
/// "1.5x on copied context, 0.90x on free generation" into ">= 1.0 on both".
struct Gate {
    plain_ms: f64,
    verify_ms: f64,
    acc: f64,
    seen_plain: bool,
    seen_acc: bool,
    seen_verify: bool,
    picks: u64,
}

impl Gate {
    /// Probe once every this many rounds once the gain is gone.
    const PROBE: u64 = 8;
    /// Only draft when the predicted gain is at least this (times plain).
    const MARGIN: f64 = 1.03;
    const ALPHA: f64 = 0.2;

    fn new() -> Self {
        Gate {
            plain_ms: 0.0,
            verify_ms: 0.0,
            acc: 0.0,
            seen_plain: false,
            seen_acc: false,
            seen_verify: false,
            picks: 0,
        }
    }

    fn ewma(old: &mut f64, seen: &mut bool, v: f64) {
        if *seen {
            *old = (1.0 - Self::ALPHA) * *old + Self::ALPHA * v;
        } else {
            *old = v;
            *seen = true;
        }
    }

    /// `false` = take the plain step this round.
    fn want(&mut self, has_draft: bool) -> bool {
        if !has_draft {
            return false;
        }
        self.picks += 1;
        if !(self.seen_plain && self.seen_acc && self.seen_verify) {
            return true; // bootstrap: both sides have to be measured once
        }
        let gain = (1.0 + self.acc) * self.plain_ms / self.verify_ms.max(1e-9);
        gain >= Self::MARGIN || self.picks.is_multiple_of(Self::PROBE)
    }

    fn saw_plain(&mut self, ms: f64) {
        Self::ewma(&mut self.plain_ms, &mut self.seen_plain, ms);
    }

    fn saw_verify(&mut self, accepted: usize, ms: f64) {
        Self::ewma(&mut self.acc, &mut self.seen_acc, accepted as f64);
        Self::ewma(&mut self.verify_ms, &mut self.seen_verify, ms);
    }
}

/// Greedy-generate `n_gen` tokens after `prompt` with an n-gram draft of depth
/// `k` (must be 1..=[`MAX_K`]; never T=8) over a look-back window of `n` tokens
/// (clamped to 2..=4). `acc` follows [`crate::mtp_spec`], and `relax = 0` keeps
/// it exactly greedy.
pub fn generate(
    sess: &mut Session<'_>,
    prompt: &[u32],
    chunk: usize,
    n_gen: usize,
    k: usize,
    n: usize,
    acc: Accept,
) -> Result<(Vec<u32>, SpecStats)> {
    if !(1..=MAX_K).contains(&k) {
        bail!("n-gram draft depth must be 1..={MAX_K}, got {k}");
    }
    sess.spec_enable(k + 1)?;
    let mut st = SpecStats::default();
    let t0 = std::time::Instant::now();
    sess.reset()?;
    let mut logits = Vec::new();
    let mut pos = 0usize;
    for len in prefill_plan(prompt.len(), chunk) {
        logits = sess.prefill(&prompt[pos..pos + len], pos)?;
        pos += len;
    }
    st.prefill_s = t0.elapsed().as_secs_f64();
    let mut toks: Vec<u32> = prompt.to_vec();
    toks.push(argmax(&logits));
    st.nll -= log_p(&logits, toks[toks.len() - 1]);

    let mut nd = NgramDrafter::new(n);
    let t_idx = std::time::Instant::now();
    nd.observe(&toks);
    st.prompt_catchup_s = t_idx.elapsed().as_secs_f64();

    let t_dec = std::time::Instant::now();
    let gated = std::env::var("BRECCIA_NGRAM_GATE").as_deref() != Ok("0");
    let mut gate = Gate::new();
    while toks.len() - prompt.len() < n_gen {
        let n = toks.len() - 1; // toks[n] is the next token to feed
        let cur = toks[n];
        let td = std::time::Instant::now();
        // Past the QSA indexer's dense gate a multi-row verify is not token-exact
        // (B12): T=2 survived 128 greedy tokens at 4k, T=3 did not (divergence at
        // 20), and T>=4 disagrees with decode by ~2.1 logits. So cap the depth at
        // one draft there rather than emit a token plain decode would not; the
        // gate then stops drafting entirely, because T=2 does not pay on its own.
        let depth = if n + 1 > IDX_DENSE_CELLS { k.min(1) } else { k };
        let drafts = nd.draft(&toks[..=n], depth);
        let draft_s = td.elapsed().as_secs_f64();
        let offer = if gated {
            gate.want(!drafts.is_empty())
        } else {
            !drafts.is_empty()
        };
        if offer {
            st.drafted += drafts.len() as u64;
            st.draft_s += draft_s;
        }
        if !offer {
            let tp = std::time::Instant::now();
            let l = sess.token(cur, n, None)?;
            gate.saw_plain(tp.elapsed().as_secs_f64() * 1e3);
            toks.push(argmax(&l));
            st.nll -= log_p(&l, toks[toks.len() - 1]);
            st.rounds[0] += 1;
            st.tokens += 1;
            nd.observe(&toks);
            continue;
        }
        let tv = std::time::Instant::now();
        let mut vt = vec![cur];
        vt.extend_from_slice(&drafts);
        let rows = sess.verify(&vt, n)?;
        let kk = drafts.len();
        let mut a = 0;
        while a < kk {
            let d = drafts[a];
            if argmax(&rows[a]) != d {
                let lp = if acc.relax > 0.0 && d < acc.special_from {
                    log_p(&rows[a], d)
                } else {
                    f64::NEG_INFINITY
                };
                if lp.exp() < acc.relax as f64 || acc.relax == 0.0 {
                    break;
                }
                st.relaxed += 1;
                st.nll -= lp;
            } else {
                st.nll -= log_p(&rows[a], d);
            }
            a += 1;
        }
        let bonus = argmax(&rows[a]);
        st.nll -= log_p(&rows[a], bonus);
        sess.rollback_to(a)?;
        st.verify_s += tv.elapsed().as_secs_f64();
        toks.extend_from_slice(&drafts[..a]);
        toks.push(bonus);
        gate.saw_verify(a, (tv.elapsed().as_secs_f64() + draft_s) * 1e3);
        st.rounds[kk] += 1;
        st.tokens += a as u64 + 1;
        for j in 0..kk.min(a + 1) {
            st.offered[j] += 1;
            st.accepted[j] += (j < a) as u64;
        }
        nd.observe(&toks);
    }
    st.decode_s = t_dec.elapsed().as_secs_f64();
    toks.drain(..prompt.len());
    toks.truncate(n_gen);
    Ok((toks, st))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_only_what_it_can_predict() {
        let toks: Vec<u32> = vec![1, 2, 3, 4, 5];
        let mut d = NgramDrafter::new(2);
        d.observe(&toks);
        // Positions 2..5 have a known predecessor pair: three entries.
        assert_eq!(d.len(), 3);
        // The last pair (4,5) has no successor yet.
        assert!(d.draft(&toks, 1).is_empty());
    }

    #[test]
    fn drafts_the_latest_repetition_and_chains() {
        // 1 2 3 4 5 1 2 3 ... -> after the second (1,2) the stream continued 3 4 5.
        let toks: Vec<u32> = vec![1, 2, 3, 4, 5, 1, 2];
        let mut d = NgramDrafter::new(2);
        d.observe(&toks);
        let got = d.draft(&toks, 3);
        assert_eq!(got, vec![3, 4, 5]);
    }

    #[test]
    fn the_later_occurrence_wins() {
        // (1,2) is followed by 7 early and by 3 late; the draft must follow the late one.
        let toks: Vec<u32> = vec![1, 2, 7, 1, 2, 3, 1, 2];
        let mut d = NgramDrafter::new(2);
        d.observe(&toks);
        assert_eq!(d.draft(&toks, 1), vec![3]);
    }

    #[test]
    fn observe_is_incremental_and_idempotent() {
        let mut d = NgramDrafter::new(3);
        d.observe(&[10, 20, 30, 40]);
        let once = d.len();
        d.observe(&[10, 20, 30, 40]);
        assert_eq!(d.len(), once);
        d.observe(&[10, 20, 30, 40, 50]);
        assert_eq!(d.len(), once + 1);
        assert_eq!(d.draft(&[10, 20, 30, 40], 1), vec![50]);
    }

    #[test]
    fn a_short_stream_drafts_nothing() {
        let mut d = NgramDrafter::new(3);
        d.observe(&[1, 2]);
        assert!(d.is_empty());
        assert!(d.draft(&[1, 2], 5).is_empty());
    }

    #[test]
    fn the_gate_stops_drafting_when_verify_is_not_paid_for() {
        let mut g = Gate::new();
        g.saw_plain(41.0);
        g.saw_verify(0, 60.0); // T=2 verify 1.46x decode, nothing accepted
                               // 1 * 41 / 60 = 0.68 -> below the margin: plain, at most the rare probe.
        let drafted = (0..8).filter(|_| g.want(true)).count();
        assert!(drafted <= 1, "gate drafted {drafted}/8 with no gain");
    }

    #[test]
    fn the_gate_drafts_when_acceptance_pays() {
        let mut g = Gate::new();
        g.saw_plain(41.0);
        g.saw_verify(5, 130.0); // T=6 verify: 6 tokens/round = 1.89x decode
        assert!((0..8).all(|_| g.want(true)));
    }

    #[test]
    fn the_gate_never_asks_for_a_draft_it_does_not_have() {
        let mut g = Gate::new();
        assert!(!g.want(false));
    }
}
