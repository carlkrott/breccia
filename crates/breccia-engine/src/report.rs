//! The JSON report `breccia bench token --json PATH` writes.
//!
//! Schema 1. Everything in here is measured by the run that writes it — the load
//! plan, the per-run decode numbers, the per-stage split, and (when it was asked
//! for) the activation digests. Nothing is transcribed from `docs/`.

use serde::{Deserialize, Serialize};

pub const TOKEN_SCHEMA_VERSION: u32 = 1;

/// One generated sequence.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunReport {
    pub run: usize,
    /// Seconds spent in the forward passes that produced generated tokens.
    pub decode_seconds: f64,
    /// Forward passes counted (generated tokens minus the last).
    pub forwards: usize,
    pub tokens_per_s: f64,
    pub ms_per_token: f64,
    /// The ids that were sent through the network, prompt + generated.
    pub prompt_ids: Vec<u32>,
    pub generated: Vec<u32>,
    /// Expert weight bytes the host lane read, divided by the forwards.
    pub expert_bytes_per_token: u64,
}

/// The whole run: load, geometry, per-run numbers.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub blocks: usize,
    pub gdn_blocks: usize,
    pub qsa_blocks: usize,
    pub vocab: usize,
    pub devices: Vec<String>,
    /// The buffer plan as printed (device → blocks, weights, state, capacity).
    pub load_plan: Vec<String>,
    pub load_seconds: f64,
    pub upload_seconds: f64,
    pub dense_bytes: u64,
    pub head_bytes: u64,
    pub session_bytes: u64,
    pub rss_bytes: u64,
    pub rss_peak_bytes: u64,
    pub expert_bytes_mapped: u64,
    pub ple_bytes_mapped: u64,
    /// `0` = greedy.
    pub temperature: f32,
    pub threads: usize,
    pub ctx_capacity: usize,
    pub runs: Vec<RunReport>,
    /// Aggregate decode rate over all runs (`forwards / Σdecode_seconds`).
    pub tokens_per_s: f64,
    pub ms_per_token: f64,
    /// Per-stage milliseconds per token, averaged over the measured runs.
    pub stages: Vec<StageRow>,
    /// The first token's logits: shape sanity, not parity.
    pub first_token: LogitSummary,
    /// Where the per-stage time goes, as a share of the decode budget.
    pub hotspots: Vec<String>,
    pub debug_layers: Vec<crate::stats::LayerTrace>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StageRow {
    pub stage: String,
    pub ms_per_token: f64,
    pub share: f64,
}

/// Summary statistics of one logits row.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LogitSummary {
    pub vocab: usize,
    pub top1: u32,
    pub top1_logit: f32,
    pub top2: u32,
    pub top2_logit: f32,
    pub top10: Vec<u32>,
    /// The `top_n` highest logits as `(id, logit)`, in descending order — the row
    /// a parity comparison is made against (`--dump-logits`).
    #[serde(default)]
    pub top: Vec<(u32, f32)>,
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub rms: f64,
    /// Distinct values / vocab — an all-equal row reads 1/vocab, ≈ 0.
    pub distinct_ratio: f64,
    pub all_finite: bool,
}

impl LogitSummary {
    /// Build the summary from one row, keeping the top 10 in `top10` and `top_n`
    /// in `top`.
    pub fn of(logits: &[f32]) -> Self {
        Self::of_top(logits, 10)
    }

    /// [`LogitSummary::of`] with an explicit `top_n`.
    pub fn of_top(logits: &[f32], top_n: usize) -> Self {
        // NaN sorts below every number, so a poisoned row still reports the
        // best finite value instead of an arbitrary index.
        let key = |i: usize| -> f64 {
            let v = logits[i];
            if v.is_nan() {
                f64::NEG_INFINITY
            } else {
                v as f64
            }
        };
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        order.sort_by(|&a, &b| {
            key(b as usize)
                .partial_cmp(&key(a as usize))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut sorted: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        sorted.sort_unstable();
        sorted.dedup();
        let mut sum = 0f64;
        let mut sq = 0f64;
        let mut finite = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in logits {
            if v.is_finite() {
                finite += 1;
                sum += v as f64;
                sq += (v as f64) * (v as f64);
                min = min.min(v);
                max = max.max(v);
            }
        }
        let f = finite.max(1) as f64;
        LogitSummary {
            vocab: logits.len(),
            top1: order.first().copied().unwrap_or(0),
            top1_logit: order.first().map_or(0.0, |&i| logits[i as usize]),
            top2: order.get(1).copied().unwrap_or(0),
            top2_logit: order.get(1).map_or(0.0, |&i| logits[i as usize]),
            top10: order.iter().take(10).copied().collect(),
            top: order
                .iter()
                .take(top_n)
                .map(|&i| (i, logits[i as usize]))
                .collect(),
            min: if finite > 0 { min } else { 0.0 },
            max: if finite > 0 { max } else { 0.0 },
            mean: sum / f,
            rms: (sq / f).sqrt(),
            distinct_ratio: sorted.len() as f64 / logits.len().max(1) as f64,
            all_finite: finite == logits.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_boring_row_is_detected() {
        let mut v = vec![1.0f32; 1000];
        let s = LogitSummary::of(&v);
        assert_eq!(s.top1, 0, "ties go to the lowest id");
        assert!((s.distinct_ratio - 0.001).abs() < 1e-9);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 1.0);
        assert!(s.all_finite);
        // A structured row is not.
        v[500] = 9.0;
        let s = LogitSummary::of(&v);
        assert_eq!(s.top1, 500);
        assert!(s.distinct_ratio > 0.001);
    }

    #[test]
    fn non_finite_logits_are_reported() {
        let s = LogitSummary::of(&[1.0, f32::NAN, 2.0]);
        assert!(!s.all_finite);
        assert_eq!(s.top1, 2);
    }
}
