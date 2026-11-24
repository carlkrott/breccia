//! Stream and weight placement for multi-engine Inkling serving: one
//! [`crate::inkling_forward::InklingForward`] per device (each holds its own
//! dense weights), streams split between engines, routed-expert layers resident
//! where bytes allow and the rest on the CPU lane that all engines share.
//!
//! Pure: sizes and measured costs in, plan out. Model-agnostic: the caller
//! fills [`Sizes`] from the bound tensors, `seq_bytes` and `LoopPlan::steps`.

/// Per-engine byte inputs.
#[derive(Clone, Debug)]
pub struct Sizes {
    /// Non-expert weights plus fixed scratch, held by every engine.
    pub dense: u64,
    /// Routed-expert bytes per physical layer (0 = no routed experts).
    pub expert: Vec<u64>,
    /// Executed steps per physical layer (a looped layer counts once per pass).
    pub uses: Vec<usize>,
    /// Device bytes per active stream: `seq_bytes(ctx, serving chunk)`.
    pub seq: u64,
}

/// One device engine.
#[derive(Clone, Debug)]
pub struct Engine {
    pub label: String,
    /// Usable bytes: `DeviceBudget::budget()`, display reserve already taken off.
    pub budget: u64,
    /// GPU ms per decode step at B streams = `gpu_ms.0 + gpu_ms.1 * (B - 1)`.
    pub gpu_ms: (f64, f64),
}

/// The CPU expert lane, shared by every engine.
#[derive(Clone, Debug)]
pub struct Lane {
    /// Measured `(rows, ms per layer call)`, rows ascending.
    pub points: Vec<(usize, f64)>,
    /// Fraction of another engine's lane time that lands on this engine's step
    /// (1 = fully serialized pool, 0 = independent).
    // shortcut: 1.0 is the safe guess until a two-engine lane run is measured.
    pub share: f64,
}

impl Lane {
    /// ms per layer call at `rows`, linear between points and past the ends.
    pub fn ms(&self, rows: usize) -> f64 {
        let p = &self.points;
        if rows == 0 || p.is_empty() {
            return 0.0;
        }
        if p.len() == 1 {
            return p[0].1 * rows as f64 / p[0].0 as f64;
        }
        let i = p.iter().position(|&(r, _)| r >= rows).unwrap_or(p.len() - 1).max(1);
        let ((r0, m0), (r1, m1)) = (p[i - 1], p[i]);
        m0 + (m1 - m0) * (rows as f64 - r0 as f64) / (r1 as f64 - r0 as f64)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnginePlan {
    pub label: String,
    pub streams: usize,
    /// Expert layers kept on the device, highest use count first.
    pub resident: Vec<usize>,
    /// Expert layers on the CPU lane (pass to `set_host_experts`).
    pub host: Vec<usize>,
    /// Lane layer calls per step (sum of `uses` over `host`).
    pub host_slots: usize,
    /// Leftover bytes for `set_hot_experts` (0 when nothing is on the lane).
    pub hot_budget: u64,
    pub step_ms: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub engines: Vec<EnginePlan>,
    /// Streams that fit nowhere at this context: parked in host RAM, served by turns.
    pub parked: usize,
    pub tok_s: f64,
    pub per_stream_tok_s: f64,
}

/// Bytes, residency and lane slots of `b` streams on one engine (`None` = KV does not fit).
fn fill(s: &Sizes, e: &Engine, b: usize) -> Option<(Vec<usize>, Vec<usize>, usize, u64)> {
    let mut left = e.budget.checked_sub(s.dense)?.checked_sub(s.seq * b as u64)?;
    let mut order: Vec<usize> = (0..s.expert.len()).filter(|&l| s.expert[l] > 0).collect();
    // most executions per byte first: looped layers run once per pass
    order.sort_by(|&a, &b| (s.uses[b] * s.expert[a] as usize).cmp(&(s.uses[a] * s.expert[b] as usize)).then(a.cmp(&b)));
    let (mut res, mut host) = (Vec::new(), Vec::new());
    for l in order {
        if s.expert[l] <= left {
            left -= s.expert[l];
            res.push(l);
        } else {
            host.push(l);
        }
    }
    host.sort_unstable();
    let slots = host.iter().map(|&l| s.uses[l]).sum();
    let hot = if host.is_empty() { 0 } else { left };
    Some((res, host, slots, hot))
}

/// Plan for exactly this split of streams over `engines` (`None` if any engine's KV overflows).
fn plan_split(s: &Sizes, engines: &[Engine], lane: &Lane, split: &[usize]) -> Option<Plan> {
    let mut plans = Vec::new();
    for (e, &b) in engines.iter().zip(split) {
        let (resident, host, host_slots, hot_budget) = if b == 0 { (vec![], vec![], 0, 0) } else { fill(s, e, b)? };
        plans.push(EnginePlan {
            label: e.label.clone(),
            streams: b,
            resident,
            host,
            host_slots,
            hot_budget,
            step_ms: 0.0,
        });
    }
    let lane_ms: Vec<f64> = plans.iter().map(|p| lane.ms(p.streams) * p.host_slots as f64).collect();
    let total: f64 = lane_ms.iter().sum();
    let mut tok_s = 0.0;
    for ((p, e), own) in plans.iter_mut().zip(engines).zip(&lane_ms) {
        if p.streams == 0 {
            continue;
        }
        p.step_ms = e.gpu_ms.0 + e.gpu_ms.1 * (p.streams - 1) as f64 + own + lane.share * (total - own);
        tok_s += p.streams as f64 * 1000.0 / p.step_ms;
    }
    let n: usize = split.iter().sum();
    Some(Plan {
        engines: plans,
        parked: 0,
        tok_s,
        per_stream_tok_s: if n == 0 { 0.0 } else { tok_s / n as f64 },
    })
}

/// Every way to put `n` streams on `k` engines.
fn splits(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 1 {
        return vec![vec![n]];
    }
    (0..=n)
        .flat_map(|a| splits(n - a, k - 1).into_iter().map(move |mut r| {
            r.insert(0, a);
            r
        }))
        .collect()
}

/// Fastest split of `streams` (aggregate tok/s). If they cannot all be active,
/// the most that can are, and the rest are `parked`.
pub fn plan(s: &Sizes, engines: &[Engine], lane: &Lane, streams: usize) -> Plan {
    assert_eq!(s.expert.len(), s.uses.len(), "expert/uses length");
    for active in (0..=streams).rev() {
        let best = splits(active, engines.len())
            .iter()
            .filter_map(|sp| plan_split(s, engines, lane, sp))
            .max_by(|a, b| a.tok_s.total_cmp(&b.tok_s));
        if let Some(mut p) = best {
            p.parked = streams - active;
            return p;
        }
    }
    unreachable!("zero streams always fit")
}

/// Best of several stream counts (e.g. 4 vs 8) by aggregate tok/s.
pub fn best(s: &Sizes, engines: &[Engine], lane: &Lane, counts: &[usize]) -> Plan {
    counts
        .iter()
        .map(|&n| plan(s, engines, lane, n))
        .max_by(|a, b| a.tok_s.total_cmp(&b.tok_s))
        .expect("at least one stream count")
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    // Inkling-2-Mini shape: 24 layers, core 6..18 run 3 passes, 1.22 GiB dense,
    // 12.81 GiB experts (~547 MiB a layer).
    fn sizes(seq: u64) -> Sizes {
        Sizes {
            dense: 1249 * MIB,
            expert: vec![547 * MIB; 24],
            uses: (0..24).map(|l| if (6..18).contains(&l) { 3 } else { 1 }).collect(),
            seq,
        }
    }
    fn lane(share: f64) -> Lane {
        Lane {
            points: vec![(1, 0.685), (2, 1.257), (4, 2.24), (8, 4.135), (16, 6.836)],
            share,
        }
    }
    fn eng(label: &str, budget: u64, gpu_ms: (f64, f64)) -> Engine {
        Engine {
            label: label.into(),
            budget,
            gpu_ms,
        }
    }

    #[test]
    fn lane_interpolates() {
        let l = lane(1.0);
        assert_eq!(l.ms(0), 0.0);
        assert!((l.ms(4) - 2.24).abs() < 1e-9);
        assert!((l.ms(6) - (2.24 + 4.135) / 2.0).abs() < 1e-9);
        assert!(l.ms(32) > 6.836);
    }

    #[test]
    fn looped_layers_go_resident_first() {
        // dense + one stream + exactly 12 layers
        let s = sizes(256 * MIB);
        let e = [eng("rx", 1249 * MIB + 256 * MIB + 12 * 547 * MIB, (25.0, 5.3))];
        let p = plan(&s, &e, &lane(1.0), 1);
        assert_eq!(p.engines[0].resident, (6..18).collect::<Vec<_>>());
        assert_eq!(p.engines[0].host, (0..6).chain(18..24).collect::<Vec<_>>());
        assert_eq!(p.engines[0].host_slots, 12);
        assert_eq!(p.engines[0].hot_budget, 0);
        // 25 ms GPU + 12 x 0.685 lane
        assert!((p.engines[0].step_ms - (25.0 + 12.0 * 0.685)).abs() < 1e-9);
    }

    #[test]
    fn four_streams_fit_with_no_lane_on_big_budgets() {
        let s = sizes(GIB);
        let e = [eng("a", 40 * GIB, (25.0, 5.3)), eng("b", 40 * GIB, (25.0, 5.3))];
        let p = plan(&s, &e, &lane(1.0), 4);
        assert_eq!(p.parked, 0);
        assert!(p.engines.iter().all(|e| e.host_slots == 0 && e.hot_budget == 0));
        assert_eq!(p.engines.iter().map(|e| e.streams).sum::<usize>(), 4);
    }

    #[test]
    fn tiny_budget_puts_every_expert_on_the_lane() {
        let s = sizes(256 * MIB);
        let e = [eng("a", 1249 * MIB + 256 * MIB + 100 * MIB, (25.0, 5.3))];
        let p = plan(&s, &e, &lane(1.0), 1);
        assert!(p.engines[0].resident.is_empty());
        assert_eq!(p.engines[0].host_slots, 48);
        assert_eq!(p.engines[0].hot_budget, 100 * MIB);
    }

    #[test]
    fn too_many_long_streams_park() {
        // 150k-ish streams at 3.5 GiB on two 16 GiB cards (RX less a display reserve)
        let s = sizes(3584 * MIB);
        let e = [eng("rx", 13 * GIB, (25.0, 5.3)), eng("mi50", 15 * GIB, (40.0, 8.0))];
        let p = plan(&s, &e, &lane(1.0), 8);
        assert!(p.parked > 0);
        for (ep, eg) in p.engines.iter().zip(&e) {
            assert!(s.dense + ep.streams as u64 * s.seq <= eg.budget, "{} over budget", ep.label);
        }
        // at most floor((budget - dense) / seq) per engine
        assert_eq!(p.engines[0].streams + p.engines[1].streams, 3 + 3);
    }

    #[test]
    fn display_reserve_sends_context_to_the_bigger_budget() {
        // rx fits one long stream after its display reserve, mi50 takes the rest
        let s = sizes(3584 * MIB);
        let e = [eng("rx", 6 * GIB, (25.0, 5.3)), eng("mi50", 15 * GIB, (40.0, 8.0))];
        let p = plan(&s, &e, &lane(0.0), 4);
        assert_eq!(p.parked, 0);
        assert_eq!([p.engines[0].streams, p.engines[1].streams], [1, 3]);
    }

    #[test]
    fn shared_lane_slows_both_engines() {
        let s = sizes(512 * MIB);
        let e = [eng("rx", 8 * GIB, (25.0, 5.3)), eng("mi50", 8 * GIB, (40.0, 8.0))];
        let free = plan_split(&s, &e, &lane(0.0), &[2, 2]).unwrap();
        let shared = plan_split(&s, &e, &lane(1.0), &[2, 2]).unwrap();
        assert!(free.engines.iter().all(|e| e.host_slots > 0));
        assert!(shared.tok_s < free.tok_s);
        for (a, b) in shared.engines.iter().zip(&free.engines) {
            assert!(a.step_ms > b.step_ms);
        }
    }

    #[test]
    fn best_picks_the_faster_stream_count() {
        let s = sizes(GIB);
        let e = [eng("a", 40 * GIB, (25.0, 5.3)), eng("b", 40 * GIB, (25.0, 5.3))];
        let p = best(&s, &e, &lane(1.0), &[4, 8]);
        // no lane, so 8 streams beat 4 on aggregate
        assert_eq!(p.engines.iter().map(|e| e.streams).sum::<usize>(), 8);
        assert!(p.tok_s > plan(&s, &e, &lane(1.0), 4).tok_s);
    }
}
