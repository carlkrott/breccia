//! Tier placement: which expert block runs where, from **measured** rates.
//!
//! The old rule assigned whole blocks to devices by free VRAM, which treats a
//! 5.7 TFLOP card and a 13 TFLOP card as "N GiB". This module turns the measured
//! [`TierRate`]s (`breccia profile --tiers`) into an assignment:
//!
//! * a tier's rate for a block is its expert-FFN rows/s at that block's expected
//!   users-per-call, log-interpolated between the measured points;
//! * blocks are admitted to the fastest tier in order of **gain over the host per
//!   byte**, so when VRAM binds the blocks that benefit most from a GPU get it;
//! * anything that does not fit goes to the host tier (`None`), which is what makes
//!   the same code work on one card, three cards, or none.
//!
//! There is no device name, PCI id or vendor in this file: the policy reads only
//! rates and capacities, so a new card is a new [`TierRate`] row, not a code change.

use crate::schema::TierRate;

/// A block to place.
#[derive(Clone, Copy, Debug)]
pub struct BlockDemand {
    /// Expert bytes the block needs resident on its tier.
    pub bytes: u64,
    /// Rows the tier is asked for per call (tokens routed to an expert, typically
    /// `tokens * top_k / n_expert`).
    pub users_per_call: u32,
    /// How often the block is used (from the expert-frequency profile); blocks with
    /// no profile entry use 1.0.
    pub weight: f64,
}

/// Rows/s at `users`, linearly interpolated in log2(users) between measured points.
/// Below the smallest point the smallest rate is used; above the largest, the
/// largest (the measurement points are the shape of the curve, not a cap).
pub fn rate_at(t: &TierRate, users: u32) -> f64 {
    let n = t.users.len().min(t.rows_per_s.len());
    if n == 0 {
        return 0.0;
    }
    let u = (users.max(1) as f64).log2();
    let mut pts: Vec<(f64, f64)> = (0..n)
        .map(|i| ((t.users[i].max(1) as f64).log2(), t.rows_per_s[i]))
        .collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if u <= pts[0].0 {
        return pts[0].1;
    }
    for w in pts.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if u <= x1 {
            let f = if x1 > x0 { (u - x0) / (x1 - x0) } else { 0.0 };
            return y0 + (y1 - y0) * f;
        }
    }
    pts[pts.len() - 1].1
}

/// The host tier, if the profile has one.
pub fn host_index(tiers: &[TierRate]) -> Option<usize> {
    tiers.iter().position(|t| t.kind == "host")
}

/// Assign each block to a tier index, or `None` when it does not fit any device
/// (i.e. it runs on the host tier).
///
/// Greedy by marginal gain per byte. Rates are per block (they do not depend on
/// what else is on the tier), so for a fixed capacity this is optimal: taking the
/// blocks with the largest gain per byte first cannot be beaten by leaving a
/// higher-gain block out.
pub fn place_blocks(blocks: &[BlockDemand], tiers: &[TierRate]) -> Vec<Option<usize>> {
    let host = host_index(tiers);
    let host_rate = |b: &BlockDemand| host.map_or(0.0, |h| rate_at(&tiers[h], b.users_per_call));
    let mut free: Vec<u64> = tiers.iter().map(|t| t.capacity_bytes).collect();
    let mut out = vec![None; blocks.len()];

    // Rank blocks by how much a device beats the host, per byte.
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    let gain = |i: usize| -> f64 {
        let b = &blocks[i];
        let best = tiers
            .iter()
            .enumerate()
            .filter(|(j, _)| Some(*j) != host)
            .map(|(_, t)| rate_at(t, b.users_per_call))
            .fold(0.0f64, f64::max);
        (best - host_rate(b)).max(0.0) * b.weight / b.bytes.max(1) as f64
    };
    order.sort_by(|&a, &b| {
        gain(b)
            .partial_cmp(&gain(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in order {
        let b = &blocks[i];
        // Fastest device that still has room.
        let mut best: Option<(usize, f64)> = None;
        for (j, t) in tiers.iter().enumerate() {
            if Some(j) == host || free[j] < b.bytes {
                continue;
            }
            let r = rate_at(t, b.users_per_call);
            if best.is_none_or(|(_, br)| r > br) {
                best = Some((j, r));
            }
        }
        if let Some((j, r)) = best {
            // Only if it actually beats what the host would do.
            if r > host_rate(b) {
                free[j] -= b.bytes;
                out[i] = Some(j);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier(kind: &str, cap: u64, rows: &[(u32, f64)]) -> TierRate {
        TierRate {
            kind: kind.into(),
            id: kind.into(),
            capacity_bytes: cap,
            users: rows.iter().map(|r| r.0).collect(),
            rows_per_s: rows.iter().map(|r| r.1).collect(),
            weight_bytes_per_s: 0.0,
            driver_version: String::new(),
        }
    }

    fn block(bytes: u64, users: u32) -> BlockDemand {
        BlockDemand {
            bytes,
            users_per_call: users,
            weight: 1.0,
        }
    }

    #[test]
    fn rate_interpolates_in_log_users_and_clamps() {
        let t = tier("device", 1 << 30, &[(1, 100.0), (1024, 102_400.0)]);
        assert!((rate_at(&t, 1) - 100.0).abs() < 1e-9);
        assert!((rate_at(&t, 1024) - 102_400.0).abs() < 1e-9);
        // log2 midpoint between 1 and 1024 is users=32 -> half way in log space
        assert!((rate_at(&t, 32) - 51_250.0).abs() < 1.0);
        assert!((rate_at(&t, 0) - 100.0).abs() < 1e-9);
        assert!((rate_at(&t, 1_000_000) - 102_400.0).abs() < 1e-9);
    }

    #[test]
    fn fills_the_fast_tier_by_gain_per_byte_then_spills_to_host() {
        // Fast: 2 GiB, 4x the host. Slow: 4 GiB, 2x the host. Host: 0 rate marker.
        let tiers = vec![
            tier("device", 2 << 30, &[(1, 400.0), (128, 51_200.0)]),
            tier("device", 4 << 30, &[(1, 200.0), (128, 25_600.0)]),
            tier("host", 0, &[(1, 100.0), (128, 12_800.0)]),
        ];
        // Three 1 GiB blocks: two fit the fast tier, the third goes to the slow one.
        let blocks = vec![
            block(1 << 30, 128),
            block(1 << 30, 128),
            block(1 << 30, 128),
        ];
        let p = place_blocks(&blocks, &tiers);
        assert_eq!(p, vec![Some(0), Some(0), Some(1)]);
    }

    #[test]
    fn a_tier_slower_than_the_host_is_not_used() {
        let tiers = vec![
            tier("device", 8 << 30, &[(1, 50.0), (128, 6_400.0)]),
            tier("host", 0, &[(1, 100.0), (128, 12_800.0)]),
        ];
        assert_eq!(place_blocks(&[block(1 << 30, 128)], &tiers), vec![None]);
    }

    #[test]
    fn no_device_capacity_means_the_host() {
        let tiers = vec![
            tier("device", 0, &[(1, 1000.0)]),
            tier("host", 0, &[(1, 100.0)]),
        ];
        assert_eq!(place_blocks(&[block(1, 1)], &tiers), vec![None]);
    }
}
