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
        let layers: Vec<usize> = (0..e.cfg.n_layers).collect();
        fwd.set_host_experts(pool, &layers).unwrap();
    }
    eprintln!("placement: resident={r:?} host={:?}", std::env::var("BRECCIA_INKLING_HOST").ok());
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
    for (n, s, r) in seen.iter().filter(|x| x.0 == "kqv_out") {
        assert!(*r <= 5e-3, "{n}-{s} {r:.3e} over the f16 KV floor");
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
    let t0 = std::time::Instant::now();
    let fresh = fwd.prefill(ids, None).unwrap();
    let t1 = t0.elapsed().as_secs_f64();
    let mut seq = fwd.new_seq(ids.len(), chunk).unwrap();
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
    assert!(d <= 1e-3, "incremental logits drift {d:.3e}");
}

#[test]
fn incremental_short() {
    incremental(&PROMPT, 4, 3);
}

/// 1100 ids with chunk 64: the SWA ring (1087 cells) wraps, positions cross 1024.
#[test]
fn incremental_long() {
    let ids = long_prompt();
    if ids.len() >= 1100 {
        incremental(&ids[..1100], 64, 4);
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
