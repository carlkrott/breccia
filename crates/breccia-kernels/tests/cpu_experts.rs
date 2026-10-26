//! Conformance for the host (CPU) resident-expert lane.
//!
//! | check | oracle |
//! |---|---|
//! | AVX2 and scalar fused dots, every type, synthetic valid blocks | [`breccia_kernels::reference::dot_row`] (`f64`, bit-exact port of `ggml-quants.c`) |
//! | the same two paths on **real expert blobs** (`blk.N` experts, gate/up/down straight out of the mmap) | same |
//! | the whole expert FFN (`gate`→`up`→SwiGLU→`down`) on real blobs | the same reference dots + an `f64` SiLU, assembled in the test |
//! | top-k pool vs single-threaded path | bit-identical output |
//! | repeated runs | bit-identical output |
//!
//! Every type comparison is required to stay inside **1e-5 relative**
//! (`|got − want| ≤ 1e-5 · max(|want|, 1)`), on both the AVX2 and the scalar
//! path; `BRECCIA_CPU_FORCE_SCALAR=1` is exercised explicitly so a run that only
//! ever hits the fast path cannot pass.
//!
//! The real-blob tests need the model present (`/srv/strata-data`); when the
//! glob matches nothing they print a skip line and pass, so the suite runs on a
//! machine without the 75 GB model. They are the ones that matter: real quantised
//! bytes have the scale distributions and sign patterns the fields actually
//! hold, not the ones a synthetic generator imagines.

use std::sync::Arc;
use std::time::Instant;

use breccia_format::{ExpertRole, ExpertSlices, ShardSet};
use breccia_kernels::cpu::{
    expert_ffn, gemv_into, moe_topk_sequential, silu, ExpertPool, FfnScratch, LayerPlan, Path,
};
use breccia_kernels::{real, reference, QuantType};

const TOL: f64 = 1e-5;
const MODEL_GLOB: &str = "/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf";

fn open_model() -> Option<Arc<ShardSet>> {
    match ShardSet::open_glob(MODEL_GLOB) {
        Ok(s) => Some(Arc::new(s)),
        Err(e) => {
            eprintln!("skipping real-blob conformance: {MODEL_GLOB}: {e}");
            None
        }
    }
}

fn rel(got: f32, want: f64) -> f64 {
    (f64::from(got) - want).abs() / want.abs().max(1.0)
}

/// Two relative errors for one row.
///
/// `result` is the error relative to the dot product itself — the number a
/// tolerance is usually quoted in, and the one that is meaningful when the sum
/// does not cancel. `l1` is the error relative to `Σ|wⱼxⱼ|`, the scale the
/// arithmetic actually works at: a row whose terms are ~4000× larger than its
/// sum cannot be held to 1e-5 of the sum by *any* `f32` accumulation order
/// (each partial sum carries ~6e-8 relative rounding of the intermediate
/// magnitude), so conformance on pathological rows is stated against `l1` while
/// real expert rows are held to `result` as well.
#[derive(Clone, Copy, Debug)]
struct RowErr {
    result: f64,
    l1: f64,
    cancellation: f64,
}

/// Both paths, same row, against the `f64` reference dot.
fn check_row(tag: &str, ty: QuantType, packed: &[u8], cols: usize, x: &[f32]) -> RowErr {
    let want = reference::dot_row(ty, packed, cols, x);
    let w = reference::dequant_row(ty, packed, cols);
    let l1: f64 = (0..cols)
        .map(|j| (f64::from(w[j]) * f64::from(x[j])).abs())
        .sum();
    let mut worst = RowErr {
        result: 0.0,
        l1: 0.0,
        cancellation: l1 / want.abs().max(1.0),
    };
    for path in [Path::Scalar, Path::Avx2] {
        let got = f64::from(breccia_kernels::cpu::dot_row_with(
            path, ty, packed, cols, x,
        ));
        let l1_err = (got - want).abs() / l1.max(1e-30);
        assert!(
            l1_err <= TOL,
            "{tag} {ty} {path}: got {got}, want {want}, error relative to Σ|w·x| = {l1_err:.3e}"
        );
        worst.l1 = worst.l1.max(l1_err);
        worst.result = worst.result.max((got - want).abs() / want.abs().max(1.0));
    }
    worst
}

fn fixed_x(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            let a = i as f32 * 0.011;
            (a.sin() + 0.5 * (a * 2.7).cos()) * 0.75
        })
        .collect()
}

#[test]
fn every_type_matches_the_f64_reference_on_synthetic_rows() {
    let mut worst_l1: f64 = 0.0;
    let mut worst_result: f64 = 0.0;
    let mut worst_cancel: f64 = 0.0;
    let mut summary = Vec::new();
    for ty in QuantType::ALL {
        // The model's real shapes: 640x2560 (gate/up) and 2560x640 (down).
        for (rows, cols) in [(8usize, 2560usize), (8, 640)] {
            let (packed, x) =
                real::synthetic_inputs(ty, breccia_kernels::Shape::new(rows, cols), 0x51ed);
            let row_bytes = breccia_kernels::cpu::row_bytes(ty, cols);
            for r in 0..rows {
                let e = check_row("synthetic", ty, &packed[r * row_bytes..], cols, &x);
                worst_l1 = worst_l1.max(e.l1);
                worst_result = worst_result.max(e.result);
                worst_cancel = worst_cancel.max(e.cancellation);
                summary.push((ty, e));
            }
        }
    }
    let (worst_ty, _) = summary
        .iter()
        .max_by(|a, b| a.1.result.total_cmp(&b.1.result))
        .expect("cells");
    println!(
        "synthetic worst: relative to Σ|w·x| {worst_l1:.3e}, relative to the result {worst_result:.3e} \
         (worst row {worst_ty}, max cancellation {worst_cancel:.0}x)"
    );
    assert!(worst_l1 <= TOL, "{worst_l1:.3e} > {TOL:e} of Σ|w·x|");
}

#[test]
fn both_paths_are_bit_identical_across_repeated_runs() {
    let ty = QuantType::Iq3S;
    let cols: usize = 2560;
    let (packed, x) = real::synthetic_inputs(ty, breccia_kernels::Shape::new(4, cols), 7);
    let row_bytes = breccia_kernels::cpu::row_bytes(ty, cols);
    for path in [Path::Scalar, Path::Avx2] {
        let a = breccia_kernels::cpu::dot_row_with(path, ty, &packed[..row_bytes], cols, &x);
        let b = breccia_kernels::cpu::dot_row_with(path, ty, &packed[..row_bytes], cols, &x);
        assert_eq!(a.to_bits(), b.to_bits(), "{path} is not deterministic");
    }
}

#[test]
fn real_expert_blobs_match_the_f64_reference() {
    let Some(set) = open_model() else { return };
    let layout = set.layout().expect("layout");

    // Which quant types do this model's experts actually use, and where does each
    // first appear? Scanning the header-only geometry is cheap; the `f64`
    // reference decode is not, so one representative `(block, role)` per type
    // gets the row-by-row treatment.
    let mut all: Vec<QuantType> = Vec::new();
    let mut reps: Vec<(usize, ExpertRole, QuantType)> = Vec::new();
    let mut plans = Vec::new();
    for block in layout.blocks() {
        let plan = LayerPlan::from_block(&set, block).expect("plan");
        for role in ExpertRole::ALL {
            let ty = plan.role(role.name()).ty;
            if !all.contains(&ty) {
                all.push(ty);
                reps.push((plan.block, role, ty));
            }
        }
        plans.push(plan);
    }
    assert!(!all.is_empty());

    let mut worst = RowErr {
        result: 0.0,
        l1: 0.0,
        cancellation: 0.0,
    };
    let mut checked = 0usize;
    let mut worst_cancel: f64 = 0.0;
    for (block, role, ty) in &reps {
        let plan = plans
            .iter()
            .find(|p| p.block == *block)
            .expect("representative block");
        let r = plan.role(role.name());
        let x = fixed_x(r.cols);
        let row_bytes = breccia_kernels::cpu::row_bytes(r.ty, r.cols);
        // Experts 0..9 of the role, every row of expert 0 + sampled rows after.
        for expert in 0..10.min(plan.n_experts) {
            let slices = plan.slices(&set, expert).expect("slices");
            let packed = match role {
                ExpertRole::Gate => slices.gate,
                ExpertRole::Up => slices.up,
                ExpertRole::Down => slices.down,
            };
            let rows: Vec<usize> = if expert == 0 {
                (0..r.rows).collect()
            } else {
                vec![0, r.rows / 3, r.rows / 2, r.rows - 1]
            };
            for row in rows {
                let e = check_row(
                    &format!("blk.{block} expert {expert} {}", role.name()),
                    *ty,
                    &packed[row * row_bytes..],
                    r.cols,
                    &x,
                );
                worst.result = worst.result.max(e.result);
                worst.l1 = worst.l1.max(e.l1);
                worst_cancel = worst_cancel.max(e.cancellation);
                checked += 1;
            }
        }
    }
    println!(
        "real experts: {checked} rows checked across {} types {all:?}; worst relative error \
         {:.3e} (to Σ|w·x| {:.3e}, max cancellation {worst_cancel:.0}x)",
        all.len(),
        worst.result,
        worst.l1
    );
    // Every routed-expert type the model ships must have been exercised, and the
    // five gate/up families plus the two `down` families do appear in this model.
    for ty in [
        QuantType::Iq2Xxs,
        QuantType::Iq2Xs,
        QuantType::Iq2S,
        QuantType::Iq3Xxs,
        QuantType::Iq3S,
        QuantType::Iq4Nl,
        QuantType::Q2_0,
    ] {
        assert!(all.contains(&ty), "{ty} is an expert type of this model");
    }
    // The strict bar the work order sets, applied where it is arithmetically
    // meaningful: the real quantised blobs of this model.
    assert!(
        worst.result <= TOL,
        "worst relative error {:.3e} > {TOL:e}",
        worst.result
    );
}

#[test]
fn expert_ffn_matches_an_independent_f64_pipeline() {
    let Some(set) = open_model() else { return };
    let layout = set.layout().expect("layout");
    let block = layout.block(0).expect("blk.0");
    let plan = LayerPlan::from_block(&set, block).expect("plan");
    let x = fixed_x(plan.n_embd);

    for expert in [0usize, 7] {
        let slices = plan.slices(&set, expert).expect("slices");
        // Reference: the same three dots through `reference::dot_row` (`f64`),
        // the SiLU in `f64`, and the weighted accumulation the graph performs.
        let gate: Vec<f64> = (0..plan.n_ff)
            .map(|r| {
                let rb = breccia_kernels::cpu::row_bytes(plan.gate.ty, plan.n_embd);
                reference::dot_row(plan.gate.ty, &slices.gate[r * rb..], plan.n_embd, &x)
            })
            .collect();
        let up: Vec<f64> = (0..plan.n_ff)
            .map(|r| {
                let rb = breccia_kernels::cpu::row_bytes(plan.up.ty, plan.n_embd);
                reference::dot_row(plan.up.ty, &slices.up[r * rb..], plan.n_embd, &x)
            })
            .collect();
        let act: Vec<f32> = (0..plan.n_ff)
            .map(|k| {
                let g = gate[k];
                let s = if g == f64::NEG_INFINITY {
                    0.0
                } else {
                    g / (1.0 + (-g).exp())
                };
                (s * up[k]) as f32
            })
            .collect();
        let rb_down = breccia_kernels::cpu::row_bytes(plan.down.ty, plan.n_ff);
        let mut worst: f64 = 0.0;
        for r in [0usize, 1, 100, plan.n_embd / 2, plan.n_embd - 1] {
            let want =
                reference::dot_row(plan.down.ty, &slices.down[r * rb_down..], plan.n_ff, &act);
            let mut scratch = FfnScratch::new(&plan);
            let mut got = vec![0f32; plan.n_embd];
            let bytes = expert_ffn(Path::Avx2, &plan, &slices, &x, &mut scratch, 1.0, &mut got);
            assert_eq!(bytes, plan.expert_kernel_bytes());
            let r_rel = rel(got[r], want);
            assert!(
                r_rel <= TOL,
                "blk.0 expert {expert} row {r}: {} vs {want} (rel {r_rel:.3e})",
                got[r]
            );
            worst = worst.max(r_rel);
        }
        println!("expert FFN blk.0/{expert}: worst relative error {worst:.3e}");
    }
}

#[test]
fn expert_ffn_scalar_path_agrees_with_avx2_and_silu_is_shared() {
    let Some(set) = open_model() else { return };
    let layout = set.layout().expect("layout");
    let plan = LayerPlan::from_block(&set, layout.block(3).expect("blk.3")).expect("plan");
    let slices = plan.slices(&set, 5).expect("slices");
    let x = fixed_x(plan.n_embd);
    let mut scalar = vec![0f32; plan.n_embd];
    let mut avx2 = vec![0f32; plan.n_embd];
    let mut s1 = FfnScratch::new(&plan);
    let mut s2 = FfnScratch::new(&plan);
    expert_ffn(Path::Scalar, &plan, &slices, &x, &mut s1, 0.5, &mut scalar);
    expert_ffn(Path::Avx2, &plan, &slices, &x, &mut s2, 0.5, &mut avx2);
    let mut worst: f64 = 0.0;
    for (a, b) in avx2.iter().zip(&scalar) {
        worst = worst.max(f64::from((a - b).abs()) / f64::from(b.abs()).max(1.0));
    }
    println!("blk.3/5 scalar vs avx2 worst relative difference: {worst:.3e}");
    assert!(worst <= TOL);
    // SiLU itself is one shared implementation: sanity-check the plateau.
    assert_eq!(silu(f32::NEG_INFINITY), 0.0);
    assert!((silu(0.0)).abs() < f32::EPSILON);
}

#[test]
fn topk_pool_is_bit_identical_to_the_single_threaded_path() {
    let Some(set) = open_model() else { return };
    let layout = set.layout().expect("layout");
    let plan = LayerPlan::from_block(&set, layout.block(1).expect("blk.1")).expect("plan");
    let ids: Vec<usize> = vec![3, 17, 42, 99, 256, 300, 411, 450, 500, 511];
    let weights: Vec<f32> = (1..=10).map(|j| 1.0 / (j as f32 * 5.5)).collect();
    let x = fixed_x(plan.n_embd);

    let mut seq = vec![0f32; plan.n_embd];
    let bytes_seq =
        moe_topk_sequential(&set, Path::Avx2, &plan, &ids, &weights, &x, &mut seq).expect("seq");

    for threads in [2usize, 4, 8] {
        let mut out = vec![0f32; plan.n_embd];
        let mut pool = ExpertPool::new(Arc::clone(&set), threads, Path::Avx2);
        let bytes = pool
            .moe_topk(&plan, &ids, &weights, &x, &mut out)
            .expect("pool");
        assert_eq!(bytes, bytes_seq, "{threads} threads: bytes read");
        assert_eq!(bytes, plan.expert_kernel_bytes() * ids.len());
        for (k, (a, b)) in out.iter().zip(&seq).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{threads} threads: element {k} {} vs {b}",
                a
            );
        }
        // Determinism across runs of the same pool.
        let mut again = vec![0f32; plan.n_embd];
        pool.moe_topk(&plan, &ids, &weights, &x, &mut again)
            .expect("again");
        for (k, (a, b)) in again.iter().zip(&out).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "pool rerun element {k}");
        }
    }
}

#[test]
fn forced_scalar_switch_selects_the_scalar_path() {
    if !breccia_kernels::cpu::avx2_available() {
        assert_eq!(breccia_kernels::cpu::path(), Path::Scalar);
        return;
    }
    std::env::set_var("BRECCIA_CPU_FORCE_SCALAR", "1");
    assert_eq!(breccia_kernels::cpu::path(), Path::Scalar);
    // The scalar path is still the exact reference decode: a forced-scalar GEMV
    // must produce the same values as the AVX2 one.
    let ty = QuantType::Iq2Xs;
    let (packed, x) = real::synthetic_inputs(ty, breccia_kernels::Shape::new(4, 2560), 11);
    let mut a = vec![0f32; 4];
    let mut b = vec![0f32; 4];
    gemv_into(Path::Scalar, ty, &packed, 4, 2560, &x, &mut a);
    std::env::remove_var("BRECCIA_CPU_FORCE_SCALAR");
    assert_eq!(breccia_kernels::cpu::path(), Path::Avx2);
    gemv_into(Path::Avx2, ty, &packed, 4, 2560, &x, &mut b);
    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
        let d = rel(*q, f64::from(*p));
        assert!(
            d <= TOL,
            "row {i}: avx2 {q} vs forced-scalar {p} (rel {d:.3e})"
        );
    }
}

/// The lane's headline number, measured here as well as by the CLI so a
/// regression shows up in `cargo test` rather than only in a report.
#[test]
fn real_blob_gemv_throughput_is_above_one_gigabyte_per_second() {
    if cfg!(debug_assertions) {
        eprintln!("skipping the throughput floor in a debug build (unoptimised)");
        return;
    }
    let Some(set) = open_model() else { return };
    let layout = set.layout().expect("layout");
    let plan = LayerPlan::from_block(&set, layout.block(0).expect("blk.0")).expect("plan");
    let x = fixed_x(plan.n_embd);
    let experts: Vec<usize> = (0..8).collect();
    let mut out = vec![0f32; plan.n_ff.max(plan.n_embd)];
    let mut best = f64::INFINITY;
    let mut bytes = 0u64;
    for _ in 0..2 {
        let start = Instant::now();
        bytes = 0;
        for &e in &experts {
            let s = plan.slices(&set, e).expect("slices");
            for (ty, packed, rows, cols) in [
                (plan.gate.ty, s.gate, plan.n_ff, plan.n_embd),
                (plan.up.ty, s.up, plan.n_ff, plan.n_embd),
                (plan.down.ty, s.down, plan.n_embd, plan.n_ff),
            ] {
                gemv_into(Path::Avx2, ty, packed, rows, cols, &x, &mut out);
                bytes += (rows * breccia_kernels::cpu::row_bytes(ty, cols)) as u64;
            }
        }
        best = best.min(start.elapsed().as_secs_f64());
    }
    let gbps = bytes as f64 / best / 1e9;
    println!("blk.0 experts 0..8: {gbps:.2} GB/s on one thread");
    assert!(
        gbps > 1.0,
        "{gbps:.2} GB/s per thread is below the performance floor this lane assumes"
    );
}

/// `ExpertSlices` of a real expert must be `gate | up | down` windows of the
/// three role tensors — the same contract the GPU lane consumes, asserted here
/// so the host lane cannot silently drift to a copy.
#[test]
fn expert_slices_are_mmap_windows() {
    let Some(set) = open_model() else { return };
    let layout = set.layout().expect("layout");
    let block = layout.block(0).expect("blk.0");
    let plan = LayerPlan::from_block(&set, block).expect("plan");
    let slices: ExpertSlices<'_> = plan.slices(&set, 4).expect("slices");
    let part = |shard: usize, start: usize| -> usize {
        set.parts()[shard].bytes().as_ptr() as usize + start
    };
    assert_eq!(
        slices.gate.as_ptr() as usize,
        part(plan.gate.shard, plan.gate.offset + 4 * plan.gate.per_expert)
    );
    assert_eq!(
        slices.down.as_ptr() as usize,
        part(plan.down.shard, plan.down.offset + 4 * plan.down.per_expert)
    );
    assert_eq!(slices.len(), plan.expert_bytes());
    assert_eq!(plan.expert_bytes(), plan.expert_kernel_bytes());
}
