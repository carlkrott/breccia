//! Activation checksums for `--debug-layers`.
//!
//! The purpose is *diffing*, not prettiness: a checksum that is stable across
//! runs of the same binary and changes when a single element changes is enough to
//! bisect "block 17 is the first one that differs" against a reference dump. So
//! the digest is a plain FNV-1a over the raw little-endian bits of every value
//! (order-sensitive, no float arithmetic), and the summary statistics are there to
//! say *how* two dumps differ when the digest already says *that* they do.

/// A stable, order-sensitive digest of a vector plus its shape statistics.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Checksum {
    pub n: usize,
    /// FNV-1a over the raw bits, so any single-bit change is visible.
    pub fnv: u64,
    pub finite: usize,
    pub non_finite: usize,
    pub mean: f64,
    pub rms: f64,
    pub min: f32,
    pub max: f32,
    /// The first few values, so a dump can be eyeballed without a loader.
    pub head: Vec<f32>,
}

impl Checksum {
    /// Digest `values` (keeping at most `head_n` leading samples).
    pub fn of(values: &[f32], head_n: usize) -> Checksum {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x1000_0000_01b3;
        let mut h = FNV_OFFSET;
        let mut sum = 0f64;
        let mut sq = 0f64;
        let mut finite = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in values {
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(FNV_PRIME);
            }
            if v.is_finite() {
                finite += 1;
                sum += v as f64;
                sq += (v as f64) * (v as f64);
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        let n = values.len();
        let finite_f = finite.max(1) as f64;
        Checksum {
            n,
            fnv: h,
            finite,
            non_finite: n - finite,
            mean: if finite > 0 { sum / finite_f } else { 0.0 },
            rms: if finite > 0 {
                (sq / finite_f).sqrt()
            } else {
                0.0
            },
            min: if finite > 0 { min } else { 0.0 },
            max: if finite > 0 { max } else { 0.0 },
            head: values.iter().take(head_n).copied().collect(),
        }
    }

    /// True when every element was finite.
    pub fn all_finite(&self) -> bool {
        self.non_finite == 0
    }
}

/// The per-block record a `--debug-layers` run writes.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LayerTrace {
    pub block: usize,
    /// `"gdn"` or `"qsa"`.
    pub kind: String,
    pub device: usize,
    /// The residual stack `R` `[hc, n_embd]` after the block's two HC writes.
    pub residual: Checksum,
    /// HC_READ(attn) → `mixed` `[n_embd]`.
    pub mixed_attn: Checksum,
    /// HC_READ(ffn) → `mixed` `[n_embd]`.
    pub mixed_ffn: Checksum,
    /// The token mixer's output (GDN `ssm_out` / QSA `attn_output`) `[n_embd]`.
    pub mixer_out: Checksum,
    /// The MoE FFN output `[n_embd]` (routed + shared, after the scalar gate).
    pub moe_out: Checksum,
    /// Router top-10 ids and their renormalised weights.
    pub router_ids: Vec<u32>,
    pub router_weights: Vec<f32>,
    /// PLE only: the per-stream gate and the conv output digest.
    pub ple_gate: Option<Checksum>,
    pub ple_conv: Option<Checksum>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_digest_is_order_sensitive_and_bit_exact() {
        let a = Checksum::of(&[1.0, 2.0, 3.0], 2);
        let b = Checksum::of(&[3.0, 2.0, 1.0], 2);
        assert_ne!(a.fnv, b.fnv);
        assert_eq!(a.head, vec![1.0, 2.0]);
        // -0.0 vs 0.0 differ in bits but not in value: the digest must see them.
        let z = Checksum::of(&[0.0], 1);
        let nz = Checksum::of(&[-0.0], 1);
        assert_ne!(z.fnv, nz.fnv);
    }

    #[test]
    fn statistics_ignore_non_finite_values() {
        let c = Checksum::of(&[1.0, f32::NAN, 3.0, f32::INFINITY], 8);
        assert_eq!(c.n, 4);
        assert_eq!(c.finite, 2);
        assert_eq!(c.non_finite, 2);
        assert!(!c.all_finite());
        assert_eq!(c.min, 1.0);
        assert_eq!(c.max, 3.0);
        assert!((c.mean - 2.0).abs() < 1e-6);
        assert!((c.rms - (10f64 / 2.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn an_empty_slice_is_not_a_failure() {
        let c = Checksum::of(&[], 4);
        assert_eq!(c.n, 0);
        assert!(c.all_finite());
        assert_eq!(c.mean, 0.0);
    }
}
