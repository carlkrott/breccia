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

/// Split **one block's** experts across tiers so the tiers finish together.
///
/// Each expert's cost on tier `t` is `rows_e / rate_t` (seconds); assigning experts
/// largest-first to the tier with the smallest projected completion time is the
/// classic LPT list schedule, and it is within one expert of the optimum makespan.
/// `None` means "no tier takes it" (the caller puts it on the host), and with a
/// single usable tier everything goes there.
///
/// This is the per-expert half of item 3: whole-block placement answers *where a
/// block lives*, this answers *how a block's experts are divided* inside it.
pub fn split_experts(rows_per_expert: &[u32], rates: &[f64]) -> Vec<Option<usize>> {
    let mut out = vec![None; rows_per_expert.len()];
    if rates.is_empty() || rates.iter().all(|&r| r <= 0.0) {
        return out;
    }
    let mut load = vec![0.0f64; rates.len()];
    let mut order: Vec<usize> = (0..rows_per_expert.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(rows_per_expert[i]));
    for i in order {
        let rows = rows_per_expert[i].max(1) as f64;
        let mut best: Option<(usize, f64)> = None;
        for (t, &r) in rates.iter().enumerate() {
            if r <= 0.0 {
                continue;
            }
            let finish = load[t] + rows / r;
            if best.is_none_or(|(_, b)| finish < b) {
                best = Some((t, finish));
            }
        }
        if let Some((t, _)) = best {
            load[t] += rows / rates[t];
            out[i] = Some(t);
        }
    }
    out
}

#[cfg(test)]
mod split_tests {
    use super::*;

    #[test]
    fn equal_tiers_get_equal_work() {
        let p = split_experts(&[32, 32, 32, 32], &[1.0, 1.0]);
        let a = p.iter().filter(|x| **x == Some(0)).count();
        assert_eq!(a, 2, "{p:?}");
    }

    #[test]
    fn the_faster_tier_takes_more_experts() {
        // 2x the rate: LPT gives it 3 of 4 equal experts (makespan 1.5 vs 2.0).
        let p = split_experts(&[10, 10, 10, 10], &[2.0, 1.0]);
        let fast = p.iter().filter(|x| **x == Some(0)).count();
        assert_eq!(fast, 3, "{p:?}");
    }

    #[test]
    fn a_block_s_own_work_is_balanced_not_split_by_count() {
        // One huge expert and three small ones across equal tiers: the huge one
        // takes a tier of its own, the small ones share the other.
        let p = split_experts(&[100, 5, 5, 5], &[1.0, 1.0]);
        assert_eq!(p[0], Some(0), "{p:?}");
        assert!(p[1..].iter().all(|x| *x == Some(1)), "{p:?}");
    }

    #[test]
    fn no_usable_tier_means_host() {
        assert_eq!(split_experts(&[1, 2], &[0.0, 0.0]), vec![None, None]);
        assert_eq!(split_experts(&[1, 2], &[]), vec![None, None]);
    }

    #[test]
    fn one_tier_takes_everything() {
        assert_eq!(split_experts(&[5, 6], &[1.0]), vec![Some(0), Some(0)]);
    }
}

/// Which of one block's experts the **host** should take, from the block device's
/// and the host's measured rates (the item-3 decision at the call site).
pub fn host_experts(
    rows_per_expert: &[(usize, u32)],
    device_rate: f64,
    host_rate: f64,
) -> Vec<usize> {
    let rows: Vec<u32> = rows_per_expert.iter().map(|x| x.1).collect();
    let choice = split_experts(&rows, &[device_rate, host_rate]);
    rows_per_expert
        .iter()
        .enumerate()
        .filter(|(i, _)| choice[*i] == Some(1))
        .map(|(_, (e, _))| *e)
        .collect()
}

#[cfg(test)]
mod host_experts_tests {
    use super::*;

    #[test]
    fn the_slower_host_gets_about_its_share() {
        // Device twice the host: 1 of 3 equal experts goes to the host (LPT).
        let h = host_experts(&[(0, 10), (1, 10), (2, 10)], 2.0, 1.0);
        assert_eq!(h, vec![2], "{h:?}");
    }

    #[test]
    fn a_host_with_no_measured_rate_takes_nothing() {
        assert!(host_experts(&[(0, 10), (1, 99)], 1.0, 0.0).is_empty());
    }

    #[test]
    fn a_device_with_no_rate_leaves_everything_to_the_host() {
        assert_eq!(host_experts(&[(0, 1), (1, 2)], 0.0, 1.0), vec![0, 1]);
    }
}

/// Split `blocks` contiguously across devices in proportion to their measured
/// rates, so both cards are busy for the same wall time (the faster card takes
/// more blocks). Every device with a rate gets at least one block when there are
/// enough blocks to go round. Used by `Placement`, which must produce contiguous
/// runs; the arena follows `device_of(b)` and so cannot disagree with it.
pub fn rate_counts(rates: &[f64], blocks: usize) -> Vec<usize> {
    let mut out = vec![0usize; rates.len()];
    let live: Vec<usize> = (0..rates.len()).filter(|&i| rates[i] > 0.0).collect();
    if live.is_empty() || blocks == 0 {
        return out;
    }
    let total: f64 = live.iter().map(|&i| rates[i]).sum();
    let mut assigned = 0usize;
    for (k, &i) in live.iter().enumerate() {
        let c = if k + 1 == live.len() {
            blocks - assigned
        } else {
            ((blocks as f64) * rates[i] / total).round() as usize
        }
        .min(blocks - assigned);
        out[i] = c;
        assigned += c;
    }
    // A live device with enough blocks to go round should not idle.
    if blocks >= live.len() {
        for &i in &live {
            if out[i] == 0 {
                if let Some((j, _)) = (0..out.len())
                    .filter(|&j| out[j] > 1)
                    .map(|j| (j, out[j]))
                    .max_by_key(|&(_, c)| c)
                {
                    out[j] -= 1;
                    out[i] += 1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod rate_counts_tests {
    use super::*;

    #[test]
    fn blocks_follow_the_measured_ratio() {
        assert_eq!(rate_counts(&[3.0, 1.0], 48), vec![36, 12]);
        assert_eq!(rate_counts(&[1.0, 1.0], 48), vec![24, 24]);
    }

    #[test]
    fn a_zero_rate_device_gets_no_blocks() {
        assert_eq!(rate_counts(&[2.0, 0.0], 10), vec![10, 0]);
    }

    #[test]
    fn a_live_device_is_not_left_idle() {
        // 99:1 over 4 blocks would round the slow card to 0; it keeps one.
        let c = rate_counts(&[99.0, 1.0], 4);
        assert_eq!(c.iter().sum::<usize>(), 4);
        assert!(c[1] >= 1, "{c:?}");
    }

    #[test]
    fn the_total_is_always_the_block_count() {
        for n in [1usize, 2, 3, 7, 48, 49] {
            for r in [vec![1.0f64, 1.0], vec![3.0, 1.0], vec![1.0, 2.0, 4.0]] {
                let c = rate_counts(&r, n);
                assert_eq!(c.iter().sum::<usize>(), n, "{r:?} x {n} -> {c:?}");
            }
        }
    }
}
