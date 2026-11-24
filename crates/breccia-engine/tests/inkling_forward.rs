//! S11 golden parity for the `inkling2_loop` forward pass
//! (`docs/specs/inkling2-loop-port.md`). Needs a GPU and the golden dir:
//!
//!   BRECCIA_INKLING_GGUF=<breccia gguf> BRECCIA_INKLING_GOLDEN=<golden dir>
//!   BRECCIA_KERNEL_DEVICE=<pci> cargo test -p breccia-engine --test inkling_forward
//!
//! `forced`: every golden node is compared, then replaced with the golden
//! value, so each error is that node's own (teacher forcing).
//! `free`: compare only; the prompt's top-1 must be llama.cpp's.
//!
//! Placement knobs for the decode tests: `BRECCIA_INKLING_RESIDENT=dense|all`
//! keeps weights on the device, `BRECCIA_INKLING_HOST=1` runs every layer's
//! experts on the CPU lane.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use breccia_engine::inkling::{self, InklingConfig};
use breccia_engine::inkling_forward::{InklingForward, Residency};
use breccia_format::ShardSet;
use breccia_kernels::{cpu, Device};

const PROMPT: [u32; 13] = [
    371, 151451, 149879, 151194, 149884, 152717, 11, 149880, 149876, 151451, 149879, 153067, 149884,
];
const TOP1: usize = 154328;

/// `(node, slot)` -> file for every f32 node in `golden/dump` (slot -1 = global).
fn golden_index(dir: &Path) -> HashMap<(String, i32), PathBuf> {
    let mut m = HashMap::new();
    for e in std::fs::read_dir(dir).expect("golden dump dir").flatten() {
        let f = e.file_name().into_string().unwrap();
        let Some(stem) = f.strip_suffix(".f32") else {
            continue;
        };
        let stem = stem.split_once('_').map_or(stem, |(_, r)| r); // NNNN_
        let Some((node, _dims)) = stem.rsplit_once('_') else {
            continue;
        };
        let key = match node.rsplit_once('-') {
            Some((n, s)) if s.parse::<i32>().is_ok() => (n.to_string(), s.parse().unwrap()),
            _ => (node.to_string(), -1),
        };
        m.insert(key, e.path());
    }
    m
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// max |a - b| / max |b|; NaN or a size mismatch is infinite.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let (mut d, mut r) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let e = (x as f64 - y as f64).abs();
        if !e.is_finite() {
            return f64::INFINITY;
        }
        d = d.max(e);
        r = r.max((y as f64).abs());
    }
    d / r.max(1e-12)
}

struct Env {
    dev: Device,
    shards: Arc<ShardSet>,
    cfg: InklingConfig,
    golden: PathBuf,
}

fn env() -> Option<Env> {
    let gguf = std::env::var("BRECCIA_INKLING_GGUF").ok()?;
    let golden = std::env::var("BRECCIA_INKLING_GOLDEN").ok()?;
    let dev = Device::open(std::env::var("BRECCIA_KERNEL_DEVICE").ok().as_deref()).ok()?;
    let shards = Arc::new(ShardSet::open_family(&gguf).expect("open gguf"));
    let cfg = InklingConfig::from_metadata(&shards.metadata().unwrap()).expect("config");
    inkling::bind(&shards, &cfg).expect("bind");
    Some(Env {
        dev,
        shards,
        cfg,
        golden: PathBuf::from(golden),
    })
}

/// A forward with the placement the env knobs ask for.
fn placed(e: &Env) -> InklingForward<'_> {
    let mut fwd = InklingForward::new(e.dev.gpu(), &e.shards, e.cfg.clone()).unwrap();
    let r = std::env::var("BRECCIA_INKLING_RESIDENT").unwrap_or_default();
    fwd.set_residency(Residency {
        dense: r == "dense" || r == "all",
        experts: r == "all",
    });
    if std::env::var("BRECCIA_INKLING_HOST").is_ok_and(|v| v == "1") {
        let pool = cpu::ExpertPool::new(e.shards.clone(), std::thread::available_parallelism().map_or(8, |n| n.get().min(32)), cpu::path());
        // BRECCIA_INKLING_HOST_LAYERS="0,1,18..24" keeps the other layers' experts on the GPU path
        let layers: Vec<usize> = match std::env::var("BRECCIA_INKLING_HOST_LAYERS") {
            Ok(s) => s
                .split(',')
                .flat_map(|p| match p.split_once("..") {
                    Some((a, b)) => (a.trim().parse().unwrap()..b.trim().parse().unwrap()).collect::<Vec<usize>>(),
                    None => vec![p.trim().parse().unwrap()],
                })
                .collect(),
            Err(_) => (0..e.cfg.n_layers).collect(),
        };
        fwd.set_host_experts(pool, &layers).unwrap();
    }
    // BRECCIA_INKLING_HOST_MAPS=1: resident MoE builds its maps on the host (the pre-11-24 multi-row path)
    fwd.host_maps = std::env::var("BRECCIA_INKLING_HOST_MAPS").is_ok_and(|v| v == "1");
    eprintln!("placement: resident={r:?} host={:?} layers={:?} host_maps={}", std::env::var("BRECCIA_INKLING_HOST").ok(), std::env::var("BRECCIA_INKLING_HOST_LAYERS").ok(), fwd.host_maps);
    fwd
}

/// Run `prompt` against golden `dir`; `tol` = worst allowed node error, if any.
fn run(dir: &str, prompt: &[u32], force: bool, tol: Option<f64>) -> Vec<(String, i32, f64)> {
    let Some(e) = env() else {
        eprintln!(
            "skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN (+ BRECCIA_KERNEL_DEVICE)"
        );
        return Vec::new();
    };
    let golden = golden_index(&e.golden.join(dir));
    let mut fwd = InklingForward::new(e.dev.gpu(), &e.shards, e.cfg.clone()).unwrap();
    let mut seen: Vec<(String, i32, f64)> = Vec::new();
    let mut tap = |name: &str, slot: i32, v: &[f32]| -> Option<Vec<f32>> {
        let g = read_f32(golden.get(&(name.to_string(), slot))?);
        let r = rel(v, &g);
        seen.push((name.to_string(), slot, r));
        (force && g.len() == v.len()).then_some(g)
    };
    let t0 = std::time::Instant::now();
    let logits = fwd.prefill(prompt, Some(&mut tap)).unwrap();
    let secs = t0.elapsed().as_secs_f64();

    let first_bad = seen.iter().find(|x| x.2 > 2e-2);
    let mut worst = seen.clone();
    worst.sort_by(|a, b| b.2.total_cmp(&a.2));
    eprintln!(
        "{} nodes compared in {secs:.1} s (force={force})",
        seen.len()
    );
    eprintln!("first > 2e-2: {first_bad:?}");
    let mut by: std::collections::BTreeMap<&str, (f64, i32)> = Default::default();
    for x in &seen {
        let b = by.entry(&x.0).or_insert((0.0, 0));
        if x.2 > b.0 || x.2.is_nan() {
            *b = (x.2, x.1);
        }
    }
    for (n, (r, sl)) in &by {
        eprintln!("  max {n:<24} {r:.3e} (slot {sl})");
    }
    for w in worst.iter().take(5) {
        eprintln!("  {:<24} slot {:>2}  {:.3e}", w.0, w.1, w.2);
    }
    // not tapped: raw embd, kq_b (fused into attention), and the expert-order nodes
    let skip = ["embd", "kq_b", "ffn_moe_situ", "ffn_moe_weights_scaled"];
    let missed: Vec<_> = golden
        .keys()
        .filter(|k| !skip.contains(&k.0.as_str()) && !seen.iter().any(|s| s.0 == k.0 && s.1 == k.1))
        .collect();
    assert!(missed.is_empty(), "golden nodes never tapped: {missed:?}");
    assert!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    if let Some(tol) = tol {
        assert!(worst[0].2 <= tol, "worst node {:?} > {tol}", worst[0]);
    }
    let Some(gp) = golden.get(&("result_output".to_string(), -1)) else {
        return seen;
    };
    let top = |l: &[f32]| {
        let mut i: Vec<usize> = (0..l.len()).collect();
        i.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
        (i[0], i[1], l[i[0]] - l[i[1]])
    };
    let (g1, g2, gm) = top(&read_f32(gp));
    let (b1, b2, bm) = top(&logits);
    eprintln!("golden top {g1}/{g2} margin {gm:.3}; breccia top {b1}/{b2} margin {bm:.3}");
    assert_eq!(g1, TOP1, "golden top-1 changed");
    if !force {
        assert_eq!(b1, TOP1, "free-running top-1");
    }
    seen
}

fn long_prompt() -> Vec<u32> {
    let p = std::env::var("BRECCIA_INKLING_GOLDEN").unwrap_or_default();
    let s = std::fs::read_to_string(Path::new(&p).join("tokens-long.txt")).unwrap_or_default();
    s.trim()
        .trim_matches(['[', ']'])
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

// Tolerances: quantized GEMV outputs differ ~2e-2 from the CPU golden (Q8_K
// activations there, f32 here); free-running error grows over 48 slots.
#[test]
fn forced() {
    run("dump", &PROMPT, true, Some(8e-2));
}

#[test]
fn free() {
    run("dump", &PROMPT, false, None);
}

/// 1415 tokens, slots 0 and 5: SWA edge (1024), tau and the rel-bias overflow bucket.
/// Only kqv_out has all its inputs forced here (no pre-clip nodes in dump-long):
/// it must sit at the golden's f16 KV floor; Q/K/V/R carry 5 layers of drift.
#[test]
fn long_forced() {
    let seen = run("dump-long", &long_prompt(), true, Some(1e-1));
    // q6_0 KV (the default) rounds K/V harder than the golden's f16: 4.4e-2 measured.
    let floor = match breccia_kernels::qsa::VLayout::current() {
        breccia_kernels::qsa::VLayout::F16 => 5e-3,
        _ => 1e-1,
    };
    for (n, s, r) in seen.iter().filter(|x| x.0 == "kqv_out") {
        assert!(*r <= floor, "{n}-{s} {r:.3e} over the KV floor {floor:.0e}");
    }
}

#[test]
fn long_free() {
    run("dump-long", &long_prompt(), false, Some(1e-1));
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn argmax(l: &[f32]) -> usize {
    (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap()
}

/// S13: chunked prefill + one-token decode steps through `InklingSeq` must give
/// the logits of one fresh prefill of the same ids. `ctx`/`chunk` make the SWA
/// ring (window + chunk - 1 cells) wrap when `ids` is past it.
fn incremental(ids: &[u32], chunk: usize, decode: usize) {
    let Some(e) = env() else {
        eprintln!("skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN");
        return;
    };
    let mut fwd = placed(&e);
    // split reorders the f32 sums (~1e-5) and MoE routing amplifies that past 1e-3;
    // its correctness is attn_dense_paged's, this test checks the cache exactly
    fwd.set_attn_split(false);
    let t0 = std::time::Instant::now();
    let fresh = fwd.prefill(ids, None).unwrap();
    let t1 = t0.elapsed().as_secs_f64();
    let mut seq = fwd.new_seq(ids.len(), chunk).unwrap();
    assert_eq!(seq.bytes(), fwd.seq_bytes(ids.len(), chunk).unwrap(), "seq_bytes drifted from new_seq");
    let split = ids.len() - decode;
    let mut l = fwd.forward(&mut seq, &ids[..split], None).unwrap();
    for &id in &ids[split..] {
        let t = std::time::Instant::now();
        l = fwd.forward(&mut seq, &[id], None).unwrap();
        eprintln!(
            "decode step at {} in {:.2} s",
            seq.pos(),
            t.elapsed().as_secs_f64()
        );
    }
    assert_eq!(seq.pos(), ids.len());
    let d = max_abs(&l, &fresh);
    let span = fresh.iter().fold(0f32, |m, x| m.max(x.abs()));
    eprintln!(
        "{} ids, chunk {chunk}, {decode} decode steps: max |dlogit| {d:.3e} (|logit| <= {span:.2}), \
         argmax {} vs {}; fresh prefill {t1:.1} s",
        ids.len(),
        argmax(&l),
        argmax(&fresh)
    );
    assert_eq!(argmax(&l), argmax(&fresh), "incremental argmax");
    // GEMM rounding differs by chunk shape (~1e-6); a q6 cell can land a level
    // over (2.3e-3 measured at 1100/64), f16 cannot
    let tol = match breccia_kernels::qsa::VLayout::current() {
        breccia_kernels::qsa::VLayout::F16 => 1e-3,
        _ => 1e-2,
    };
    assert!(d <= tol, "incremental logits drift {d:.3e} over {tol:.0e}");
}

#[test]
fn incremental_short() {
    incremental(&PROMPT, 4, 3);
}

/// 1100 ids with chunk 64: the SWA ring (1087 cells) wraps, positions cross 1024.
/// `INC_N` / `INC_CHUNK` / `INC_DEC` override the shape for bisecting.
#[test]
fn incremental_long() {
    let ids = long_prompt();
    if ids.len() >= 1100 {
        let env = |k: &str, d: usize| std::env::var(k).map_or(d, |v| v.parse().unwrap());
        incremental(&ids[..env("INC_N", 1100)], env("INC_CHUNK", 64), env("INC_DEC", 4));
    }
}

/// S13: 32 greedy tokens through `InklingSeq` decode vs llama.cpp (`golden/greedy-32.txt`,
/// CPU build). A mismatch passes only as a near-tie (reference logit within 0.1 of the
/// top; the CPU golden quantizes activations to Q8_K), and the comparison stops there.
#[test]
fn greedy_32() {
    let Some(e) = env() else {
        eprintln!("skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN");
        return;
    };
    let s = std::fs::read_to_string(e.golden.join("greedy-32.txt")).expect("greedy-32.txt");
    let want: Vec<u32> = s
        .trim()
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect();
    let mut fwd = placed(&e);
    let mut seq = fwd.new_seq(PROMPT.len() + want.len(), 64).unwrap();
    let mut l = fwd.forward(&mut seq, &PROMPT, None).unwrap();
    let mut got = Vec::new();
    let t0 = std::time::Instant::now();
    for &w in &want {
        let top = argmax(&l) as u32;
        if top != w {
            let gap = l[top as usize] - l[w as usize];
            eprintln!("diverged at {}: {top} vs {w}, gap {gap:.4}", got.len());
            assert!(gap <= 0.1, "not a near-tie: got {got:?}");
            break;
        }
        got.push(top);
        l = fwd.forward(&mut seq, &[top], None).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "{}/{} greedy tokens match; decode {:.3} s/token",
        got.len(),
        want.len(),
        dt / got.len().max(1) as f64
    );
}

/// Resident weights are the same bytes as per-use uploads: logits bitwise equal.
#[test]
fn resident_bitwise() {
    let Some(e) = env() else {
        eprintln!("skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN");
        return;
    };
    let mut fwd = InklingForward::new(e.dev.gpu(), &e.shards, e.cfg.clone()).unwrap();
    let base = fwd.prefill(&PROMPT, None).unwrap();
    fwd.set_residency(Residency {
        dense: true,
        experts: std::env::var("BRECCIA_INKLING_RESIDENT").is_ok_and(|v| v == "all"),
    });
    for pass in 0..2 {
        let l = fwd.prefill(&PROMPT, None).unwrap();
        assert!(
            l.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()),
            "pass {pass}: resident logits differ, max {:.3e}",
            max_abs(&l, &base)
        );
    }
}

/// Batched dispatch (default) gives the same bits as one submit per op, for a
/// chunked prefill and greedy decode steps, under the env placement.
#[test]
fn batched_bitwise() {
    let Some(e) = env() else {
        eprintln!("skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN");
        return;
    };
    let mut fwd = placed(&e);
    let run = |fwd: &mut InklingForward, on: bool| -> Vec<Vec<f32>> {
        fwd.set_batching(on);
        let mut seq = fwd.new_seq(64, 8).unwrap();
        let mut out = vec![fwd.forward(&mut seq, &PROMPT, None).unwrap()];
        for _ in 0..3 {
            let l = out.last().unwrap();
            let top = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
            out.push(fwd.forward(&mut seq, &[top], None).unwrap());
        }
        out
    };
    let base = run(&mut fwd, false);
    let t0 = std::time::Instant::now();
    let got = run(&mut fwd, true);
    eprintln!("batched run {:.3} s", t0.elapsed().as_secs_f64());
    for (i, (a, b)) in got.iter().zip(&base).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "step {i}: batched logits differ, max {:.3e}",
            max_abs(a, b)
        );
    }
}

/// Measurement, not a gate: how skewed the router is per weight layer over the
/// long prompt (prefill) and 64 greedy decode tokens, and the expert bytes a
/// hot-expert pool would hold. Prints the share of picks the top 10/20/30/50%
/// of experts cover.
#[test]
#[ignore]
fn expert_skew() {
    let Some(e) = env() else {
        eprintln!("skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN");
        return;
    };
    let per: u64 = ["ffn_gate_up_exps.weight", "ffn_down_exps.weight"]
        .iter()
        .map(|s| e.shards.tensor(&format!("blk.6.{s}")).unwrap().n_bytes())
        .sum::<u64>()
        / e.cfg.n_expert as u64;
    eprintln!("bytes per expert {per} ({:.2} MiB)", per as f64 / (1 << 20) as f64);
    let prompt = long_prompt();
    let mut fwd = placed(&e);
    let mut seq = fwd.new_seq(prompt.len() + 64, 512).unwrap();
    let mut l = fwd.forward(&mut seq, &prompt, None).unwrap();
    let pre: Vec<Vec<u64>> = fwd.routed().to_vec();
    for _ in 0..64 {
        let top = argmax(&l) as u32;
        l = fwd.forward(&mut seq, &[top], None).unwrap();
    }
    let cover = |c: &[u64]| -> Vec<f64> {
        let mut v = c.to_vec();
        v.sort_unstable_by(|a, b| b.cmp(a));
        let tot = v.iter().sum::<u64>().max(1) as f64;
        [10, 20, 30, 50]
            .iter()
            .map(|p| v[..v.len() * p / 100].iter().sum::<u64>() as f64 / tot)
            .collect()
    };
    let fmt = |v: Vec<f64>| v.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>().join(" ");
    let (mut dsum, mut dtot) = (vec![0f64; 4], 0u64);
    for (layer, (all, p)) in fwd.routed().iter().zip(&pre).enumerate() {
        let dec: Vec<u64> = all.iter().zip(p).map(|(a, b)| a - b).collect();
        let n: u64 = dec.iter().sum();
        let c = cover(&dec);
        for (s, x) in dsum.iter_mut().zip(&c) {
            *s += x * n as f64;
        }
        dtot += n;
        eprintln!("layer {layer:2}: prefill {} | decode {} ({n} picks)", fmt(cover(p)), fmt(c));
    }
    eprintln!("decode weighted top10/20/30/50%: {}", fmt(dsum.iter().map(|s| s / dtot as f64).collect()));
}

/// Hot pool on top of the CPU lane: seeded from the long prompt's routed
/// counts, `BRECCIA_INKLING_HOT_MIB` (default 1024) of experts on the device.
/// Gates: two runs with the same pool are bitwise equal, and greedy decode
/// matches the CPU golden (hits run in GPU math, so not bitwise vs all-lane).
#[test]
fn hot_mixed() {
    let Some(e) = env() else {
        eprintln!("skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN");
        return;
    };
    let s = std::fs::read_to_string(e.golden.join("greedy-32.txt")).expect("greedy-32.txt");
    let want: Vec<u32> = s.trim().split(',').map(|x| x.trim().parse().unwrap()).collect();
    let mut fwd = placed(&e);
    // the lane must cover every layer the pool may pick
    if std::env::var("BRECCIA_INKLING_HOST").ok().as_deref() != Some("1") {
        eprintln!("skip: needs BRECCIA_INKLING_HOST=1");
        return;
    }
    let prompt = long_prompt();
    let mut warm = fwd.new_seq(prompt.len(), 512).unwrap();
    fwd.forward(&mut warm, &prompt, None).unwrap();
    drop(warm);
    let mib: u64 = std::env::var("BRECCIA_INKLING_HOT_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let heat = fwd.routed().to_vec();
    let held = fwd.set_hot_experts(mib << 20, &heat).unwrap();
    let per_layer: Vec<usize> = breccia_engine::inkling_forward::hot_pick(&heat, held).iter().map(Vec::len).collect();
    eprintln!("hot pool: {held} experts in {mib} MiB, per layer {per_layer:?}");

    let every: usize = std::env::var("BRECCIA_INKLING_HOT_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut runs = Vec::new();
    for pass in 0..3 {
        // passes 0/1 frozen (determinism gate), pass 2 with swaps
        let frozen = breccia_engine::inkling_forward::HotSwaps { every: 0, max: 32 };
        fwd.set_hot_swaps(if pass < 2 { frozen } else { breccia_engine::inkling_forward::HotSwaps { every, max: 32 } });
        let before = fwd.hot_stats();
        fwd.prof = [0.0; 3];
        let mut seq = fwd.new_seq(PROMPT.len() + want.len(), 64).unwrap();
        let mut l = fwd.forward(&mut seq, &PROMPT, None).unwrap();
        let (mut got, mut logits) = (Vec::new(), vec![l.clone()]);
        let t0 = std::time::Instant::now();
        for &w in &want {
            let top = argmax(&l) as u32;
            if top != w {
                let gap = l[top as usize] - l[w as usize];
                eprintln!("diverged at {}: {top} vs {w}, gap {gap:.4}", got.len());
                assert!(gap <= 0.1, "not a near-tie: got {got:?}");
                break;
            }
            got.push(top);
            l = fwd.forward(&mut seq, &[top], None).unwrap();
            logits.push(l.clone());
        }
        let dt = t0.elapsed().as_secs_f64() / got.len().max(1) as f64;
        let (h, r) = fwd.hot_stats();
        let rate = (h - before.0) as f64 / (r - before.1).max(1) as f64;
        eprintln!("pass {pass}: {}/{} greedy match; decode {dt:.3} s/token; hit rate {rate:.2}; swapped {}", got.len(), want.len(), fwd.hot_swapped());
        eprintln!("  prof (s, whole pass): cpu lane {:.2}, moe {:.2}, reload {:.2}", fwd.prof[0], fwd.prof[1], fwd.prof[2]);
        runs.push(logits);
    }
    assert!(runs[0].len() == runs[1].len() && runs[0].len() > 1 && runs[0].iter().zip(&runs[1]).all(|(a, b)| a == b), "hot pool not deterministic");
}

/// Debug: per-slot taps of the last chunk, incremental vs fresh (`INC_N`, `INC_CHUNK`).
#[test]
#[ignore]
fn incremental_taps() {
    let Some(e) = env() else { return };
    let ids = long_prompt();
    let get = |k: &str, d: usize| std::env::var(k).map_or(d, |v| v.parse().unwrap());
    let (n, chunk) = (get("INC_N", 1060), get("INC_CHUNK", 32));
    let ids = &ids[..n];
    let mut fwd = placed(&e);
    fwd.set_attn_split(false); // as incremental()
    let mut fresh: Vec<(String, i32, Vec<f32>)> = Vec::new();
    {
        let mut cap = |name: &str, slot: i32, d: &[f32]| {
            fresh.push((name.to_string(), slot, d.to_vec()));
            None
        };
        fwd.prefill(ids, Some(&mut cap)).unwrap();
    }
    let mut seq = fwd.new_seq(n, chunk).unwrap();
    let last = (n - 1) / chunk * chunk;
    let t = n - last;
    fwd.forward(&mut seq, &ids[..last], None).unwrap();
    let mut inc: Vec<(String, i32, Vec<f32>)> = Vec::new();
    {
        let mut cap = |name: &str, slot: i32, d: &[f32]| {
            inc.push((name.to_string(), slot, d.to_vec()));
            None
        };
        fwd.forward(&mut seq, &ids[last..], Some(&mut cap)).unwrap();
    }
    eprintln!("last chunk {last}..{n} ({t} tokens)");
    for (name, slot, d) in &inc {
        let Some((_, _, f)) = fresh.iter().find(|(fname, fs, _)| fname == name && fs == slot) else { continue };
        let w = d.len() / t;
        if w == 0 || f.len() != w * n {
            continue;
        }
        let tail = &f[last * w..];
        let rows: Vec<f32> = (0..t).map(|r| max_abs(&d[r * w..(r + 1) * w], &tail[r * w..(r + 1) * w])).collect();
        let m = rows.iter().cloned().fold(0f32, f32::max);
        if m > 1e-4 {
            eprintln!("  {name:<14} slot {slot:2}  max {m:.3e}  per row {:?}", rows.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>());
        }
    }
}

/// CPU-lane cost by rows per call (no GPU): T random-routed rows through
/// `moe_topk_chunk` on every layer, the planner's cost-model input for batched
/// streams and prefill. Random routing is the upper bound on distinct experts.
/// `BRECCIA_INKLING_LANE_BENCH=1` runs it; `LANE_T="1,4,8,512"`.
#[test]
fn lane_bench() {
    if std::env::var("BRECCIA_INKLING_LANE_BENCH").as_deref() != Ok("1") {
        return;
    }
    let gguf = std::env::var("BRECCIA_INKLING_GGUF").expect("BRECCIA_INKLING_GGUF");
    let shards = Arc::new(ShardSet::open_family(&gguf).expect("open gguf"));
    let cfg = InklingConfig::from_metadata(&shards.metadata().unwrap()).expect("config");
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(32));
    let mut pool = cpu::ExpertPool::new(shards.clone(), threads, cpu::path());
    let act = cpu::GluAct::Situ { beta: cfg.situ_beta, lb: cfg.situ_linear_beta };
    let plans: Vec<cpu::LayerPlan> = (0..cfg.n_layers)
        .map(|l| {
            let t = |s: &str| shards.tensor(&format!("blk.{l}.{s}")).unwrap();
            cpu::LayerPlan::merged(&shards, l, &t("ffn_gate_up_exps.weight"), &t("ffn_down_exps.weight"), act).unwrap()
        })
        .collect();
    let (ne, k, n) = (cfg.n_expert, cfg.n_expert_used, cfg.n_embd);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let ts = std::env::var("LANE_T").unwrap_or_else(|_| "1,4,8,512".into());
    // warm the page cache once over every layer
    for p in &plans {
        let ids: Vec<usize> = (0..k).collect();
        pool.moe_topk(p, &ids, &vec![0.1; k], &vec![0.01; n], &mut vec![0.0; n]).unwrap();
    }
    for t in ts.split(',').map(|s| s.trim().parse::<usize>().unwrap()) {
        let x = vec![0.01f32; t * n];
        let mut out = vec![0f32; t * n];
        let (mut secs, mut distinct, mut calls) = (0f64, 0usize, 0usize);
        for _rep in 0..3 {
            for p in &plans {
                let ids: Vec<Vec<usize>> = (0..t)
                    .map(|_| {
                        let mut v: Vec<usize> = Vec::with_capacity(k);
                        while v.len() < k {
                            let e = (rnd() % ne as u64) as usize;
                            if !v.contains(&e) {
                                v.push(e);
                            }
                        }
                        v
                    })
                    .collect();
                let w = vec![1.0 / k as f32; k];
                let routes: Vec<cpu::ChunkRoute> = ids.iter().map(|i| cpu::ChunkRoute { ids: i, weights: &w }).collect();
                let mut all: Vec<usize> = ids.concat();
                all.sort_unstable();
                all.dedup();
                distinct += all.len();
                let t0 = std::time::Instant::now();
                if t == 1 {
                    pool.moe_topk(p, &ids[0], &w, &x, &mut out).unwrap();
                } else {
                    pool.moe_topk_chunk(p, &routes, &x, &mut out).unwrap();
                }
                secs += t0.elapsed().as_secs_f64();
                calls += 1;
            }
        }
        eprintln!(
            "lane T={t:4}: {:.3} ms per layer call, {:.3} ms per row, {:.1} distinct experts ({threads} threads)",
            1e3 * secs / calls as f64,
            1e3 * secs / (calls * t) as f64,
            distinct as f64 / calls as f64
        );
    }
}

/// Seconds per `forward` call at T rows (`ROWS_T="1,8"`), after the prompt: what
/// every row-wise op costs at the row count batched decode would run.
/// `BRECCIA_INKLING_ROWS_TIMING=1` runs it.
#[test]
fn rows_timing() {
    if std::env::var("BRECCIA_INKLING_ROWS_TIMING").as_deref() != Ok("1") {
        return;
    }
    let Some(e) = env() else { return };
    let ts: Vec<usize> = std::env::var("ROWS_T")
        .unwrap_or_else(|_| "1,8".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let reps = 8;
    let mut fwd = placed(&e);
    let total = PROMPT.len() + ts.iter().map(|t| t * (reps + 1)).sum::<usize>();
    let mut seq = fwd.new_seq(total, 64).unwrap();
    fwd.forward(&mut seq, &PROMPT, None).unwrap();
    for &t in &ts {
        let ids: Vec<u32> = PROMPT.iter().cycle().take(t).copied().collect();
        fwd.forward(&mut seq, &ids, None).unwrap(); // warm this shape
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            fwd.forward(&mut seq, &ids, None).unwrap();
        }
        let s = t0.elapsed().as_secs_f64() / reps as f64;
        eprintln!("rows T={t}: {:.1} ms per forward, {:.2} ms per row", 1e3 * s, 1e3 * s / t as f64);
    }
}

/// Prints the first nodes where a stream's rows of the batched tap differ from
/// its own one-row tap (rows assumed `[streams, ..]`; other layouts show as diffs too).
fn tap_diff(got: &[(String, i32, Vec<f32>)], want: &[Vec<(String, i32, Vec<f32>)>]) {
    let b = want.len();
    let mut shown = 0;
    for (k, (name, slot, g)) in got.iter().enumerate() {
        for (i, w) in want.iter().enumerate() {
            let Some((wn, ws, wv)) = w.get(k) else { continue };
            if wn != name || ws != slot {
                eprintln!("tap order differs at {k}: {name}/{slot} vs {wn}/{ws}");
                return;
            }
            let row = if g.len() == b * wv.len() { &g[i * wv.len()..][..wv.len()] } else { &g[..wv.len().min(g.len())] };
            let d = max_abs(row, &wv[..row.len()]);
            let m = wv.iter().fold(0f32, |a, x| a.max(x.abs()));
            if d > 1e-3 * m.max(1.0) && shown < 12 {
                eprintln!("tap {k} {name} slot {slot} stream {i}: max |d| {d:.3e} (|x| <= {m:.2}), got len {} want {}", g.len(), wv.len());
                shown += 1;
            }
        }
    }
    eprintln!("tap_diff: {} nodes, {shown} diffs shown", got.len());
}

/// Batched decode: `forward_batch` over B streams (different prompt lengths, so
/// different positions per row) gives each stream the logits of its own
/// one-token `forward`, teacher-forced on the sequential greedy ids.
/// `BATCH_B` (3) streams, `BATCH_N` (6) steps; prints both paths' tokens/s.
#[test]
fn batch_decode() {
    let Some(e) = env() else {
        eprintln!("skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN");
        return;
    };
    let env = |k: &str, d: usize| std::env::var(k).map_or(d, |v| v.parse().unwrap());
    let (b, n) = (env("BATCH_B", 3), env("BATCH_N", 6));
    let prompts: Vec<&[u32]> = (0..b).map(|i| &PROMPT[..PROMPT.len() - (i * 4) % 10]).collect();
    let ctx = PROMPT.len() + n + 1;
    let mut fwd = placed(&e);

    let mut seqs: Vec<_> = (0..b).map(|_| fwd.new_seq(ctx, 64).unwrap()).collect();
    let mut last: Vec<Vec<f32>> = prompts
        .iter()
        .zip(&mut seqs)
        .map(|(p, s)| fwd.forward(s, p, None).unwrap())
        .collect();
    let (mut fed, mut want) = (Vec::new(), Vec::new());
    // BATCH_TAPS=1: per-node diff of step 0, each stream's rows vs its own forward
    let taps = env("BATCH_TAPS", 0) == 1;
    let mut want_taps: Vec<Vec<(String, i32, Vec<f32>)>> = vec![Vec::new(); b];
    // timing and submit stats cover steps 1.. (step 0 sizes staging/maps for t = B)
    let (mut t0, mut s0) = (std::time::Instant::now(), breccia_vk::ctx::submit_stats());
    for step in 0..n {
        if step == 1 {
            (t0, s0) = (std::time::Instant::now(), breccia_vk::ctx::submit_stats());
        }
        let ids: Vec<u32> = last.iter().map(|l| argmax(l) as u32).collect();
        last = ids
            .iter()
            .zip(&mut seqs)
            .zip(&mut want_taps)
            .map(|((&id, s), wt)| {
                if step > 0 || !taps {
                    return fwd.forward(s, &[id], None).unwrap();
                }
                let mut tap = |name: &str, slot: i32, v: &[f32]| -> Option<Vec<f32>> {
                    wt.push((name.to_string(), slot, v.to_vec()));
                    None
                };
                fwd.forward(s, &[id], Some(&mut tap)).unwrap()
            })
            .collect();
        fed.push(ids);
        want.push(last.clone());
    }
    let seq_s = t0.elapsed().as_secs_f64();
    let seq_st = breccia_vk::ctx::submit_stats().since(&s0);
    drop(seqs);

    let mut seqs: Vec<_> = (0..b).map(|_| fwd.new_seq(ctx, 64).unwrap()).collect();
    for (p, s) in prompts.iter().zip(&mut seqs) {
        fwd.forward(s, p, None).unwrap();
    }
    let tol = match breccia_kernels::qsa::VLayout::current() {
        breccia_kernels::qsa::VLayout::F16 => 1e-2,
        _ => 3e-2,
    };
    let mut worst = 0f32;
    let ov0 = fwd.overlap_stats();
    let (mut t0, mut s0) = (std::time::Instant::now(), breccia_vk::ctx::submit_stats());
    for (step, ids) in fed.iter().enumerate() {
        if step == 1 {
            (t0, s0) = (std::time::Instant::now(), breccia_vk::ctx::submit_stats());
        }
        let mut refs: Vec<&mut _> = seqs.iter_mut().collect();
        let got = if step == 0 && taps {
            let mut got_taps = Vec::new();
            let mut tap = |name: &str, slot: i32, v: &[f32]| -> Option<Vec<f32>> {
                got_taps.push((name.to_string(), slot, v.to_vec()));
                None
            };
            let got = fwd.forward_batch(&mut refs, ids, Some(&mut tap)).unwrap();
            tap_diff(&got_taps, &want_taps);
            got
        } else {
            fwd.forward_batch(&mut refs, ids, None).unwrap()
        };
        for (i, (g, w)) in got.iter().zip(&want[step]).enumerate() {
            let d = max_abs(g, w);
            worst = worst.max(d);
            assert_eq!(argmax(g), argmax(w), "step {step} stream {i}: argmax");
            assert!(d <= tol, "step {step} stream {i}: max |dlogit| {d:.3e} over {tol:.0e}");
        }
    }
    let bat_s = t0.elapsed().as_secs_f64();
    let bat_st = breccia_vk::ctx::submit_stats().since(&s0);
    for (s, p) in seqs.iter().zip(&prompts) {
        assert_eq!(s.pos(), p.len() + n);
    }
    let timed = n.saturating_sub(1).max(1) as f64;
    let toks = b as f64 * timed;
    eprintln!(
        "batch_decode B={b} N={n}: max |dlogit| {worst:.3e}; steps 1..: sequential {:.1} tok/s, batched {:.1} tok/s",
        toks / seq_s,
        toks / bat_s
    );
    for (name, st, wall, per) in [("sequential", seq_st, seq_s, toks), ("batched", bat_st, bat_s, timed)] {
        eprintln!(
            "  {name} per step: {:.1} ms wall, {:.1} ms fence wait, {:.1} submits, {:.1} waits",
            1e3 * wall / per,
            1e3 * st.wait_s / per,
            st.submits as f64 / per,
            st.waits as f64 / per
        );
    }
    let ov = fwd.overlap_stats();
    if ov.0 > ov0.0 {
        eprintln!(
            "  batched MoE (read-back layers): {} pairs, {} distinct experts ({:.2} pairs per expert)",
            ov.0 - ov0.0,
            ov.1 - ov0.1,
            (ov.0 - ov0.0) as f64 / (ov.1 - ov0.1) as f64
        );
    }
}
