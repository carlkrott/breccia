//! Greedy generation with MTP speculative decoding: the draft proposes `k` tokens,
//! one `Session::verify` of `k + 1` rows checks them, `rollback_to` drops the
//! rejected tail. Output is the trunk's own greedy argmax (verify decides); the
//! draft only changes how many trunk passes it takes. `k <= 3`, so T <= 4 (never 8).
//!
//! Draft cell `c` holds `(R_c, tok_{c+1})` and predicts `tok_{c+2}`; `R_c` is the
//! trunk's final residual stack after token `c`. Cells beyond the confirmed
//! prefix (the draft's own chain) are overwritten on the next round.

use anyhow::{bail, Result};

use crate::graph::{prefill_plan, Session};
use crate::mtp_draft::MtpDraft;

/// Draft-depth policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Policy {
    /// Plain greedy decode, no draft (the control arm).
    Plain,
    /// Fixed depth `k` in 1..=3.
    Fixed(usize),
    /// Adaptive depth in 0..=3.
    Gate,
}

impl Policy {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "plain" => Some(Self::Plain),
            "gate" => Some(Self::Gate),
            k => k.parse::<usize>().ok().filter(|k| (1..=3).contains(k)).map(Self::Fixed),
        }
    }
}

const MAX_K: usize = 3;

/// Depth chooser. `cost(k) = verify(k+1) + draft(k)` in ms, `tokens(k) = 1 + sum_j
/// prod_{i<=j} p_i`; picks the best tokens/ms and probes one deeper every 16 picks.
/// Priors are UNVERIFIED seeds (verify ms from the 1k mixed-prompt table); every
/// number is replaced by an EMA of what this run measures.
struct Gate {
    num: [f64; MAX_K],
    den: [f64; MAX_K],
    /// EMA ms of a whole round: index k = rounds at depth k (0 = plain decode).
    round_ms: [f64; MAX_K + 1],
    seen: [bool; MAX_K + 1],
    picks: u64,
}

impl Gate {
    fn new() -> Self {
        Gate {
            num: [1.2; MAX_K],
            den: [2.0; MAX_K],
            round_ms: [52.0, 118.0, 142.0, 172.0],
            seen: [false; MAX_K + 1],
            picks: 0,
        }
    }

    fn tokens(&self, k: usize) -> f64 {
        let (mut prod, mut t) = (1.0, 1.0);
        for j in 0..k {
            prod *= self.num[j] / self.den[j];
            t += prod;
        }
        t
    }

    fn pick(&mut self) -> usize {
        self.picks += 1;
        let best = (0..=MAX_K)
            .max_by(|&a, &b| {
                (self.tokens(a) / self.round_ms[a]).total_cmp(&(self.tokens(b) / self.round_ms[b]))
            })
            .unwrap_or(0);
        if self.picks % 16 == 0 && best < MAX_K {
            best + 1
        } else {
            best
        }
    }

    fn observe(&mut self, k: usize, accepted: usize, ms: f64) {
        let r = &mut self.round_ms[k];
        *r = if self.seen[k] { 0.8 * *r + 0.2 * ms } else { ms };
        self.seen[k] = true;
        // position j was offered iff j <= accepted (the prefix before it all passed)
        for j in 0..k.min(accepted + 1) {
            self.num[j] = self.num[j] * 0.97 + (j < accepted) as u8 as f64;
            self.den[j] = self.den[j] * 0.97 + 1.0;
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct SpecStats {
    pub rounds: [u64; MAX_K + 1],
    /// Position j+1 offered / accepted (counted only while the prefix passed).
    pub offered: [u64; MAX_K],
    pub accepted: [u64; MAX_K],
    pub tokens: u64,
    pub prefill_s: f64,
    /// Draft catch-up over the prompt (not in `decode_s`).
    pub prompt_catchup_s: f64,
    pub decode_s: f64,
    pub draft_s: f64,
    pub verify_s: f64,
}

impl std::fmt::Display for SpecStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rounds: u64 = self.rounds.iter().sum();
        write!(
            f,
            "rounds {:?} tokens/round {:.2} acc/pos",
            self.rounds,
            self.tokens as f64 / rounds.max(1) as f64
        )?;
        for j in 0..MAX_K {
            write!(f, " {}:{}/{}", j + 1, self.accepted[j], self.offered[j])?;
        }
        write!(
            f,
            " | decode {:.3} s (draft {:.3} verify {:.3}) prefill {:.2} s prompt-catchup {:.2} s",
            self.decode_s, self.draft_s, self.verify_s, self.prefill_s, self.prompt_catchup_s
        )
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = i;
        }
    }
    b as u32
}

/// Greedy-generate `n_gen` tokens after `prompt` (the first comes from the
/// prefill logits). `draft` is required unless `policy` is `Plain`.
/// Needs `sess.spec_enable(4)` for any policy but `Plain`.
pub fn generate(
    sess: &mut Session<'_>,
    mut draft: Option<&mut MtpDraft<'_>>,
    prompt: &[u32],
    chunk: usize,
    n_gen: usize,
    policy: Policy,
) -> Result<(Vec<u32>, SpecStats)> {
    if policy != Policy::Plain && draft.is_none() {
        bail!("{policy:?} needs a draft");
    }
    let mut st = SpecStats::default();
    let want_r = policy != Policy::Plain;
    let t0 = std::time::Instant::now();
    sess.reset()?;
    if let Some(d) = draft.as_deref_mut() {
        d.reset()?;
    }
    // r_store[c] = R_c for c >= r_base (positions whose draft cell is not built yet).
    let mut r_store: Vec<Vec<f32>> = Vec::new();
    let mut r_base = 0usize;
    let mut logits = Vec::new();
    let mut pos = 0usize;
    for len in prefill_plan(prompt.len(), chunk) {
        logits = sess.prefill(&prompt[pos..pos + len], pos)?;
        if want_r {
            r_store.extend(sess.final_r_slots(len)?);
        }
        pos += len;
    }
    st.prefill_s = t0.elapsed().as_secs_f64();

    let mut toks: Vec<u32> = prompt.to_vec();
    toks.push(argmax(&logits));
    let mut cells_done = 0usize; // draft cells [0, cells_done) are built from true R

    // Catch the draft up over the prompt before the decode clock starts.
    let sync = |d: &mut MtpDraft<'_>,
                r_store: &mut Vec<Vec<f32>>,
                r_base: &mut usize,
                cells_done: &mut usize,
                toks: &[u32],
                upto: usize|
     -> Result<()> {
        while *cells_done < upto {
            let c = *cells_done;
            d.catchup(&r_store[c - *r_base], toks[c + 1], c)?;
            *cells_done += 1;
        }
        // free what is built (keep R_c only for c >= cells_done)
        let drop = *cells_done - *r_base;
        r_store.drain(..drop);
        *r_base = *cells_done;
        Ok(())
    };
    if let Some(d) = draft.as_deref_mut().filter(|_| want_r) {
        let t = std::time::Instant::now();
        let n = toks.len() - 1;
        sync(d, &mut r_store, &mut r_base, &mut cells_done, &toks, n - 1)?;
        st.prompt_catchup_s = t.elapsed().as_secs_f64();
    }

    let mut gate = Gate::new();
    let t_dec = std::time::Instant::now();
    while toks.len() - prompt.len() < n_gen {
        let n = toks.len() - 1; // positions 0..n are in the trunk; toks[n] is next to feed
        let cur = toks[n];
        let k = match policy {
            Policy::Plain => 0,
            Policy::Fixed(k) => k,
            Policy::Gate => gate.pick(),
        };
        let tr = std::time::Instant::now();
        if k == 0 {
            let l = sess.token(cur, n, None)?;
            if want_r {
                r_store.push(sess.final_r_decode()?);
            }
            toks.push(argmax(&l));
            st.rounds[0] += 1;
            st.tokens += 1;
            gate.observe(0, 0, tr.elapsed().as_secs_f64() * 1e3);
            continue;
        }
        let d = draft.as_deref_mut().expect("checked above");
        let td = std::time::Instant::now();
        // cells < n-1 from true R (front only); cell n-1 = (R_{n-1}, cur) predicts d1.
        sync(d, &mut r_store, &mut r_base, &mut cells_done, &toks, n - 1)?;
        d.set_r(&r_store[n - 1 - r_base])?;
        let mut drafts = Vec::with_capacity(k);
        let mut tok = cur;
        for j in 0..k {
            let (t, _) = d.step_ex(tok, n - 1 + j, j > 0)?;
            drafts.push(t);
            tok = t;
        }
        st.draft_s += td.elapsed().as_secs_f64();
        let tv = std::time::Instant::now();
        let mut vt = vec![cur];
        vt.extend_from_slice(&drafts);
        let rows = sess.verify(&vt, n)?;
        let rs = sess.final_r_slots(k + 1)?;
        let mut a = 0;
        while a < k && argmax(&rows[a]) == drafts[a] {
            a += 1;
        }
        let bonus = argmax(&rows[a]);
        sess.rollback_to(a)?;
        st.verify_s += tv.elapsed().as_secs_f64();
        // R_{n..=n+a} are true; cell n-1 is built (the step above).
        cells_done = n;
        r_store.truncate(n - r_base);
        r_store.extend(rs.into_iter().take(a + 1));
        toks.extend_from_slice(&drafts[..a]);
        toks.push(bonus);
        st.rounds[k] += 1;
        st.tokens += a as u64 + 1;
        for j in 0..k.min(a + 1) {
            st.offered[j] += 1;
            st.accepted[j] += (j < a) as u64;
        }
        gate.observe(k, a, tr.elapsed().as_secs_f64() * 1e3);
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
    fn policy_parse() {
        assert_eq!(Policy::parse("plain"), Some(Policy::Plain));
        assert_eq!(Policy::parse("gate"), Some(Policy::Gate));
        assert_eq!(Policy::parse("2"), Some(Policy::Fixed(2)));
        assert_eq!(Policy::parse("4"), None); // never T=8
        assert_eq!(Policy::parse("0"), None);
    }

    #[test]
    fn gate_adapts_and_probes() {
        let mut g = Gate::new();
        // always-rejecting draft: k=0 must win once the estimate has decayed
        for _ in 0..200 {
            g.observe(1, 0, 118.0);
        }
        let picks: Vec<usize> = (0..32).map(|_| g.pick()).collect();
        assert_eq!(picks.iter().filter(|&&k| k == 0).count(), 30);
        assert_eq!(picks[15], 1); // the 16th pick probes one deeper
        // always-accepting draft with cheap verify: deepest wins
        let mut g = Gate::new();
        for _ in 0..200 {
            g.observe(3, 3, 100.0);
            g.observe(2, 2, 90.0);
            g.observe(1, 1, 80.0);
        }
        assert_eq!(g.pick(), 3);
    }
}
