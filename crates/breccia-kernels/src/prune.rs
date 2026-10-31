//! Opt-in routed-expert **tail skip** (`BRECCIA_EXPERT_TAU`, `BRECCIA_EXPERT_MIN_KEEP`).
//!
//! The router shader (`ops_router.comp`) is the single place that decides: after the
//! top-k it drops the experts whose normalised weight is below `tau` (keeping at least
//! `min_keep`), renormalises the kept weights to sum 1 (the model's own convention for
//! the selected set, which also keeps the routed branch balanced against the shared
//! expert) and writes the dropped slots as id [`EXPERT_NONE`] / weight 0. Because the
//! top-k comes out sorted by weight, the dropped slots are always a suffix.
//!
//! Everything downstream removes `EXPERT_NONE` slots from its dispatch lists (GPU
//! `moe_map` -> expert GEMV, `moe_fanin`; host CPU lane, chunk prefill and the merged
//! prefill bucket builder) rather than multiplying by zero.
//!
//! The rule's CPU reference is `ops::cpu::router_pruned`; its unit tests check the
//! rule, they do not exercise the GLSL.

use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicU64, Ordering};

/// Router id of a tail-skipped slot.
pub const EXPERT_NONE: u32 = u32::MAX;

/// Default `BRECCIA_EXPERT_MIN_KEEP`.
pub const DEFAULT_MIN_KEEP: u32 = 4;

/// A validated tail-skip setting. `tau > 0` always (unset / 0 means off = `None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prune {
    pub tau: f32,
    pub min_keep: u32,
}

impl Prune {
    /// Parse the two env values for a router that selects `k` experts.
    /// `tau` unset or exactly 0 is off; NaN, negative, or >= 1 is an error;
    /// `min_keep` is clamped to `1..=k`.
    pub fn parse(tau: Option<&str>, min_keep: Option<&str>, k: usize) -> Result<Option<Prune>> {
        let Some(t) = tau.map(str::trim).filter(|t| !t.is_empty()) else {
            return Ok(None);
        };
        let tau: f32 = t
            .parse()
            .map_err(|_| anyhow!("BRECCIA_EXPERT_TAU={t:?} is not a number"))?;
        if tau.is_nan() || !(0.0..1.0).contains(&tau) {
            return Err(anyhow!("BRECCIA_EXPERT_TAU={t} must be in [0, 1) (0 = off)"));
        }
        if tau == 0.0 {
            return Ok(None);
        }
        let mk = match min_keep.map(str::trim).filter(|m| !m.is_empty()) {
            None => DEFAULT_MIN_KEEP,
            Some(m) => m
                .parse::<u32>()
                .map_err(|_| anyhow!("BRECCIA_EXPERT_MIN_KEEP={m:?} is not a non-negative integer"))?,
        };
        Ok(Some(Prune { tau, min_keep: mk.clamp(1, k.max(1) as u32) }))
    }

    pub fn from_env(k: usize) -> Result<Option<Prune>> {
        Self::parse(
            std::env::var("BRECCIA_EXPERT_TAU").ok().as_deref(),
            std::env::var("BRECCIA_EXPERT_MIN_KEEP").ok().as_deref(),
            k,
        )
    }
}

static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// Count `n` skipped (token, expert) slots seen by a host reader.
pub fn note_skipped(n: usize) {
    if n > 0 {
        SKIPPED.fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// Host-observed skipped slots so far (GPU-resident decode blocks never reach the
/// host, so they are not in this number).
pub fn skipped() -> u64 {
    SKIPPED.load(Ordering::Relaxed)
}

/// Remove [`EXPERT_NONE`] slots from one token's router output and count them.
pub fn strip_none(ids: &[usize], weights: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let mut oi = Vec::with_capacity(ids.len());
    let mut ow = Vec::with_capacity(ids.len());
    for (&i, &w) in ids.iter().zip(weights) {
        if i != EXPERT_NONE as usize {
            oi.push(i);
            ow.push(w);
        }
    }
    note_skipped(ids.len() - oi.len());
    (oi, ow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rules() {
        let p = |t: Option<&str>, m: Option<&str>| Prune::parse(t, m, 10);
        assert_eq!(p(None, None).unwrap(), None);
        assert_eq!(p(Some("0"), None).unwrap(), None);
        assert_eq!(p(Some("0.0"), Some("garbage")).unwrap(), None);
        assert_eq!(p(Some("0.02"), None).unwrap(), Some(Prune { tau: 0.02, min_keep: 4 }));
        assert_eq!(p(Some("0.02"), Some("0")).unwrap().unwrap().min_keep, 1);
        assert_eq!(p(Some("0.02"), Some("99")).unwrap().unwrap().min_keep, 10);
        for bad in ["nan", "-0.1", "1", "1.5", "abc"] {
            assert!(p(Some(bad), None).is_err(), "{bad}");
        }
        assert!(p(Some("0.02"), Some("x")).is_err());
    }

    #[test]
    fn strip_none_removes_sentinels_and_counts() {
        let before = skipped();
        let n = EXPERT_NONE as usize;
        let (i, w) = strip_none(&[7, 3, n, n], &[0.6, 0.4, 0.0, 0.0]);
        assert_eq!((i, w), (vec![7, 3], vec![0.6, 0.4]));
        assert!(skipped() >= before + 2);
    }
}
