//! S11 golden parity for the `inkling2_loop` forward pass
//! (`docs/specs/inkling2-loop-port.md`). Needs a GPU and the golden dir:
//!
//!   BRECCIA_INKLING_GGUF=<breccia gguf> BRECCIA_INKLING_GOLDEN=<golden dir>
//!   BRECCIA_KERNEL_DEVICE=<pci> cargo test -p breccia-engine --test inkling_forward
//!
//! `forced`: every golden node is compared, then replaced with the golden
//! value, so each error is that node's own (teacher forcing).
//! `free`: compare only; the prompt's top-1 must be llama.cpp's.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use breccia_engine::inkling::{self, InklingConfig};
use breccia_engine::inkling_forward::InklingForward;
use breccia_format::ShardSet;
use breccia_kernels::Device;

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
    shards: ShardSet,
    cfg: InklingConfig,
    golden: HashMap<(String, i32), PathBuf>,
}

fn env() -> Option<Env> {
    let gguf = std::env::var("BRECCIA_INKLING_GGUF").ok()?;
    let golden = std::env::var("BRECCIA_INKLING_GOLDEN").ok()?;
    let dev = Device::open(std::env::var("BRECCIA_KERNEL_DEVICE").ok().as_deref()).ok()?;
    let shards = ShardSet::open_family(&gguf).expect("open gguf");
    let cfg = InklingConfig::from_metadata(&shards.metadata().unwrap()).expect("config");
    inkling::bind(&shards, &cfg).expect("bind");
    let golden = golden_index(&Path::new(&golden).join("dump"));
    Some(Env {
        dev,
        shards,
        cfg,
        golden,
    })
}

fn run(force: bool) {
    let Some(e) = env() else {
        eprintln!(
            "skip: set BRECCIA_INKLING_GGUF + BRECCIA_INKLING_GOLDEN (+ BRECCIA_KERNEL_DEVICE)"
        );
        return;
    };
    let mut fwd = InklingForward::new(e.dev.gpu(), &e.shards, e.cfg.clone()).unwrap();
    let mut seen: Vec<(String, i32, f64)> = Vec::new();
    let mut tap = |name: &str, slot: i32, v: &[f32]| -> Option<Vec<f32>> {
        let g = read_f32(e.golden.get(&(name.to_string(), slot))?);
        let r = rel(v, &g);
        seen.push((name.to_string(), slot, r));
        (force && g.len() == v.len()).then_some(g)
    };
    let t0 = std::time::Instant::now();
    let logits = fwd.prefill(&PROMPT, Some(&mut tap)).unwrap();
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
    let top = |l: &[f32]| {
        let mut i: Vec<usize> = (0..l.len()).collect();
        i.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
        (i[0], i[1], l[i[0]] - l[i[1]])
    };
    let gl = read_f32(&e.golden[&("result_output".to_string(), -1)]);
    let (g1, g2, gm) = top(&gl);
    let (b1, b2, bm) = top(&logits);
    eprintln!("golden top {g1}/{g2} margin {gm:.3}; breccia top {b1}/{b2} margin {bm:.3}");
    assert!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    assert_eq!(g1, TOP1, "golden top-1 changed");
    // not tapped: raw embd, kq_b (fused into attention), and the expert-order nodes
    let skip = ["embd", "kq_b", "ffn_moe_situ", "ffn_moe_weights_scaled"];
    let missed: Vec<_> = e
        .golden
        .keys()
        .filter(|k| !skip.contains(&k.0.as_str()) && !seen.iter().any(|s| s.0 == k.0 && s.1 == k.1))
        .collect();
    assert!(missed.is_empty(), "golden nodes never tapped: {missed:?}");
    if !force {
        assert_eq!(b1, TOP1, "free-running top-1");
    }
}

#[test]
fn forced() {
    run(true);
}

#[test]
fn free() {
    run(false);
}
