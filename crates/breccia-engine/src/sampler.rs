//! Sampling from the head's logits.
//!
//! Decode parity (milestone B) is greedy, so [`Sampler::argmax`] is the path the
//! graph uses by default and the one the parity comparison exercises. The
//! temperature path exists so a later milestone does not have to re-plumb the
//! graph, and it is deterministic given a seed: same logits + same seed ⇒ same
//! token, which is what a repeatable speed run needs.
//!
//! Ties resolve to the **lowest token id**, the same rule `ops_router.comp` uses
//! for the MoE top-k and `qsa_topk.comp` for the indexer — one convention
//! everywhere, so a tie can never be the explanation for a divergence.

/// How the next token is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// `0.0` ⇒ greedy argmax (the parity path).
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens before drawing (`0` = all).
    pub top_k: usize,
    /// Deterministic draw seed; ignored when `temperature == 0`.
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling {
            temperature: 0.0,
            top_k: 0,
            seed: 0x9e37_79b9_7f4a_7c15,
        }
    }
}

impl Sampling {
    pub fn greedy() -> Self {
        Sampling::default()
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// The sampler: stateless apart from the RNG state a stochastic draw advances.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Sampler {
            state: seed | 1, // xorshift64* needs a non-zero state
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        // 24 bits, so the value is exactly representable and never reaches 1.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Index of the largest value; `None` for an empty slice. Ties go to the
    /// lowest index. NaN never wins a comparison, so a NaN-poisoned row still
    /// returns a finite maximum when one exists (and index 0 when none does —
    /// the caller is expected to have checked finiteness already).
    pub fn argmax(logits: &[f32]) -> Option<u32> {
        let mut best: Option<(u32, f32)> = None;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            match best {
                None => best = Some((i as u32, v)),
                Some((_, b)) if v > b => best = Some((i as u32, v)),
                _ => {}
            }
        }
        best.map(|(i, _)| i)
    }

    /// Draw the next token under `cfg`.
    pub fn sample(&mut self, logits: &[f32], cfg: &Sampling) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }
        if cfg.is_greedy() {
            return Self::argmax(logits);
        }
        // Temperature + optional top-k, in f64 so a long row cannot overflow.
        let inv_t = 1.0 / cfg.temperature as f64;
        let mut order: Vec<usize> = (0..logits.len()).filter(|&i| !logits[i].is_nan()).collect();
        if order.is_empty() {
            return None;
        }
        order.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        if cfg.top_k > 0 && cfg.top_k < order.len() {
            order.truncate(cfg.top_k);
        }
        let max = logits[order[0]] as f64;
        let weights: Vec<f64> = order
            .iter()
            .map(|&i| (((logits[i] as f64) - max) * inv_t).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Some(order[0] as u32);
        }
        let mut r = (self.next_f32() as f64) * total;
        for (k, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return Some(order[k] as u32);
            }
        }
        Some(*order.last().expect("non-empty order") as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_is_greedy_and_tie_breaks_low() {
        assert_eq!(Sampler::argmax(&[0.1, 2.0, -1.0, 2.0]), Some(1));
        assert_eq!(Sampler::argmax(&[5.0, 5.0, 5.0]), Some(0));
        assert_eq!(Sampler::argmax(&[]), None);
        assert_eq!(Sampler::argmax(&[f32::NAN, -3.0, f32::NAN]), Some(1));
    }

    #[test]
    fn greedy_sampling_ignores_the_rng() {
        let logits = [0.0, 1.0, 0.5];
        let cfg = Sampling::greedy();
        let mut a = Sampler::new(1);
        let mut b = Sampler::new(999);
        for _ in 0..8 {
            assert_eq!(a.sample(&logits, &cfg), Some(1));
            assert_eq!(b.sample(&logits, &cfg), Some(1));
        }
    }

    #[test]
    fn stochastic_draws_are_seeded_and_in_range() {
        let logits = [0.0, 10.0, 0.0, -1.0];
        let cfg = Sampling {
            temperature: 1.0,
            top_k: 0,
            seed: 7,
        };
        let mut a = Sampler::new(7);
        let mut b = Sampler::new(7);
        let mut seen_hot = 0;
        for _ in 0..200 {
            let x = a.sample(&logits, &cfg).unwrap();
            let y = b.sample(&logits, &cfg).unwrap();
            assert_eq!(x, y, "the same seed must draw the same token");
            assert!((x as usize) < logits.len());
            if x == 1 {
                seen_hot += 1;
            }
        }
        assert!(
            seen_hot > 190,
            "a +10 logit should dominate: {seen_hot}/200"
        );
    }

    #[test]
    fn top_k_restricts_the_support() {
        let logits = [0.0, 1.0, 2.0, 3.0];
        let cfg = Sampling {
            temperature: 1.0,
            top_k: 1,
            seed: 3,
        };
        let mut s = Sampler::new(3);
        for _ in 0..50 {
            assert_eq!(s.sample(&logits, &cfg), Some(3));
        }
    }
}
