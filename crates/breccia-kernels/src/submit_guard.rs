//! **Submit guard**: keep every submission well inside the driver's GPU watchdog.
//!
//! The kernel driver kills a job that runs longer than its lockup timeout (amdgpu:
//! `lockup_timeout`, 2 s by default; Windows TDR is 2 s too). On the MI50 the ring
//! reset that follows has failed into BACO and needed a cold boot, so a submit
//! that *might* run long has to be split before it is submitted, not after.
//!
//! The model is per `(device, pipeline)`: GPU ns per workgroup, the **max** ever
//! measured. A pipeline with no measurement yet is recorded **alone** in its own
//! submission, so its first reading is exact (no attribution guesswork); after that
//! a segment is cut whenever the predicted time of the next dispatch would push the
//! open submission past the budget (40 % of the timeout). A per-device correction
//! factor absorbs under-prediction (e.g. attention, whose per-workgroup cost grows
//! with context). Splitting only adds a full barrier at the cut, so results are
//! bit-identical. Timing is two timestamps per submission (the per-dispatch probe
//! costs +14.6 %; this is per submit).
//!
//! `BRECCIA_SUBMIT_BUDGET_MS=<ms>` overrides the budget, `=0` turns the guard off.

use super::RING;
use ash::vk;
use breccia_vk::ctx::Gpu;
use std::cell::RefCell;
use std::collections::HashMap;

/// The watchdog (ms) when the driver exposes none: Windows TDR and amdgpu's default.
const DEFAULT_LOCKUP_MS: f64 = 2000.0;
/// Fraction of the watchdog one submission may be predicted to use.
const BUDGET_FRACTION: f64 = 0.4;

/// The smallest positive amdgpu `lockup_timeout` entry (ms), else the default.
/// Format: `T` or `GFX,Compute,SDMA,Video`; `0`/empty = driver default, `<0` = off.
pub(crate) fn lockup_ms(sysfs: &str) -> f64 {
    sysfs
        .trim()
        .split(',')
        .filter_map(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.min(v))))
        .unwrap_or(DEFAULT_LOCKUP_MS)
}

/// Budget in ns, or `None` when the guard is off. Cached.
pub(crate) fn budget_ns() -> Option<f64> {
    static B: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        let ms = match std::env::var("BRECCIA_SUBMIT_BUDGET_MS") {
            Ok(v) => v.trim().parse::<f64>().unwrap_or(0.0),
            Err(_) => {
                let sys = std::fs::read_to_string("/sys/module/amdgpu/parameters/lockup_timeout")
                    .unwrap_or_default();
                lockup_ms(&sys) * BUDGET_FRACTION
            }
        };
        if ms <= 0.0 {
            eprintln!("submit guard: off");
            return None;
        }
        eprintln!("submit guard: {ms:.0} ms per submission");
        Some(ms * 1e6)
    })
}

#[derive(Default, Clone, Copy)]
struct Slot {
    /// Predicted GPU ns of what is recorded so far.
    pred_ns: f64,
    /// `Some((pipeline, workgroups))`: an unmeasured pipeline recorded alone.
    solo: Option<(u64, u64)>,
    /// Dispatches recorded into it.
    n: u32,
    stamped: bool,
}

struct State {
    pools: Vec<(vk::QueryPool, Option<ash::Device>)>,
    slots: Vec<Slot>,
    /// `(device, pipeline)` -> max GPU ns per workgroup.
    per_wg: HashMap<(u64, u64), f64>,
    /// Per device: measured / predicted, max with slow decay, >= 1.
    corr: HashMap<u64, f64>,
    warned: HashMap<u64, bool>,
    /// Slowest single dispatch measured: `(ns, device index)`. A guard cannot split
    /// one dispatch, so this is the number that says whether the *kernel* (chunk
    /// size) has to shrink on a slower card.
    max_solo: (f64, u32),
}

thread_local! {
    static S: RefCell<State> = RefCell::new(State {
        pools: vec![(vk::QueryPool::null(), None); RING],
        slots: vec![Slot::default(); RING],
        per_wg: HashMap::new(),
        corr: HashMap::new(),
        warned: HashMap::new(),
        max_solo: (0.0, 0),
    });
}

fn raw<H: vk::Handle>(h: H) -> u64 {
    h.as_raw()
}

/// Stamp the start of `slot`'s new command buffer (from `begin`, after its fence).
pub(crate) fn open(gpu: &Gpu, slot: usize, cmd: vk::CommandBuffer) {
    S.with(|s| {
        let mut s = s.borrow_mut();
        s.slots[slot] = Slot::default();
        if budget_ns().is_none() || !gpu.ts_ok_c {
            return;
        }
        let dev = gpu.device.handle();
        let (pool, owner) = s.pools[slot].clone();
        let pool =
            if pool != vk::QueryPool::null() && owner.as_ref().map(|d| d.handle()) == Some(dev) {
                pool
            } else {
                if let (true, Some(d)) = (pool != vk::QueryPool::null(), owner) {
                    unsafe { d.destroy_query_pool(pool, None) };
                }
                let Ok(p) = gpu.query_pool(2) else { return };
                s.pools[slot] = (p, Some(gpu.device.clone()));
                p
            };
        gpu.reset_queries(pool, 2);
        unsafe {
            gpu.device
                .cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, pool, 0)
        };
        s.slots[slot].stamped = true;
    });
}

/// Stamp the end of `slot`'s command buffer (before it is ended).
pub(crate) fn close(gpu: &Gpu, slot: usize, cmd: vk::CommandBuffer) {
    S.with(|s| {
        let s = s.borrow();
        if s.slots[slot].stamped {
            unsafe {
                gpu.device.cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    s.pools[slot].0,
                    1,
                )
            };
        }
    });
}

/// Should the open submission in `slot` be cut **before** recording this dispatch?
/// Records the dispatch's prediction into the slot as a side effect when not.
/// `count` is the dispatches already in the slot.
pub(crate) fn before_dispatch(
    gpu: &Gpu,
    slot: usize,
    count: u32,
    pipe: vk::Pipeline,
    wg: u64,
) -> bool {
    let Some(budget) = budget_ns() else {
        return false;
    };
    S.with(|s| {
        let mut s = s.borrow_mut();
        if !s.slots[slot].stamped {
            return false;
        }
        let dev = raw(gpu.device.handle());
        let key = (dev, raw(pipe));
        let est = s
            .per_wg
            .get(&key)
            .map(|b| b * wg as f64 * s.corr.get(&dev).copied().unwrap_or(1.0));
        let sl = &mut s.slots[slot];
        let cut = count > 0
            && (sl.solo.is_some() || est.is_none() || sl.pred_ns + est.unwrap_or(0.0) > budget);
        if !cut {
            sl.n += 1;
            match est {
                Some(e) => sl.pred_ns += e,
                None => sl.solo = Some((key.1, wg.max(1))),
            }
        }
        cut
    })
}

/// After a cut, the dispatch that caused it lands in the fresh slot: account it.
pub(crate) fn after_cut(gpu: &Gpu, slot: usize, pipe: vk::Pipeline, wg: u64) {
    let _ = before_dispatch(gpu, slot, 0, pipe, wg);
}

/// Fold `slot`'s measured time into the model. Only after its fence signalled.
pub(crate) fn read(gpu: &Gpu, slot: usize) {
    S.with(|s| {
        let mut s = s.borrow_mut();
        let sl = std::mem::take(&mut s.slots[slot]);
        if !sl.stamped {
            return;
        }
        let Ok(t) = gpu.timestamp_stamps(s.pools[slot].0, 2) else {
            return;
        };
        let ns = t[1].saturating_sub(t[0]) as f64 * gpu.timestamp_period_ns;
        let dev = raw(gpu.device.handle());
        if let Some((pipe, wg)) = sl.solo {
            let b = s.per_wg.entry((dev, pipe)).or_insert(0.0);
            *b = b.max(ns / wg as f64);
            if ns > s.max_solo.0 {
                s.max_solo = (ns, gpu.index);
            }
        } else if sl.pred_ns > 0.0 {
            let c = s.corr.entry(dev).or_insert(1.0);
            *c = (*c * 0.98).max(ns / sl.pred_ns).max(1.0);
        }
        if let Some(budget) = budget_ns() {
            if ns > budget * 1.25 && !s.warned.get(&dev).copied().unwrap_or(false) {
                s.warned.insert(dev, true);
                eprintln!(
                    "submit guard: device {} submission ran {:.0} ms against a {:.0} ms budget \
                     ({} dispatches, predicted {:.0} ms){}",
                    gpu.index,
                    ns / 1e6,
                    budget / 1e6,
                    sl.n,
                    sl.pred_ns / 1e6,
                    if sl.solo.is_some() {
                        " (one dispatch: the kernel itself needs tiling)"
                    } else {
                        ""
                    }
                );
            }
        }
    });
}

static CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn count_cut() {
    CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Submissions the guard has cut so far (all threads).
pub fn cuts() -> u64 {
    CUTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Free the per-slot query pools (from `shutdown`, devices still alive).
pub(crate) fn shutdown() {
    S.with(|s| {
        let mut s = s.borrow_mut();
        if cuts() > 0 || s.max_solo.0 > 0.0 {
            eprintln!(
                "submit guard: {} submissions cut, slowest single dispatch {:.1} ms (device {})",
                cuts(),
                s.max_solo.0 / 1e6,
                s.max_solo.1
            );
        }
        s.max_solo = (0.0, 0);
        for p in s.pools.iter_mut() {
            if let (pool, Some(d)) = std::mem::replace(p, (vk::QueryPool::null(), None)) {
                if pool != vk::QueryPool::null() {
                    unsafe { d.destroy_query_pool(pool, None) };
                }
            }
        }
        s.slots.iter_mut().for_each(|x| *x = Slot::default());
        // Pipelines and devices die with the session; their handles get recycled.
        s.per_wg.clear();
        s.corr.clear();
        s.warned.clear();
    });
}

#[cfg(test)]
mod tests {
    use super::lockup_ms;

    #[test]
    fn lockup_timeout_parsing() {
        assert_eq!(lockup_ms(""), 2000.0);
        assert_eq!(lockup_ms("0"), 2000.0);
        assert_eq!(lockup_ms("-1"), 2000.0);
        assert_eq!(lockup_ms("5000\n"), 5000.0);
        assert_eq!(lockup_ms("10000,60000,10000,10000"), 10000.0);
        assert_eq!(lockup_ms("0,500,-1,0"), 500.0);
    }
}
