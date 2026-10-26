//! GPU ↔ CPU conformance for every kernel type (the Phase-3 gate evidence).
//!
//! For each type the kernel is compared against `reference::*` (the port that
//! `tools/refcheck.py` diffs bit-for-bit against llama.cpp's `ggml-quants.c`):
//!
//! | check | tolerance |
//! |---|---|
//! | `dequant_to_f32` vs `reference::dequant_row` | 1e-6 relative (integer codes exact, fp16 scales exact) |
//! | fused GEMV vs `reference::gemv` (f64 accumulation) | 1e-3 relative |
//! | non-finite scale handling | output must be non-finite, device must survive |
//!
//! Shapes cover the model's real cells (`640x2560`, `2560x640`, `2560x2560`),
//! `rows = 1`, and tails (`cols % block_elems != 0`) for both block families.
//!
//! Run with `cargo test -p breccia-kernels -- --nocapture`; the suite opens the
//! first discrete GPU once per test binary (see `gpu()` below) and prints which
//! device it used.

use breccia_kernels::reference;
use breccia_kernels::{bench, dequant_to_f32, real, QuantType, XLayout};
use breccia_kernels::{Device, Gemv, GemvBatch};

/// One device for the whole test binary, serialised: `ctx::Gpu` owns a single
/// command buffer per queue, so two tests must never record into it at once.
struct Fixture {
    _guard: std::sync::MutexGuard<'static, ()>,
    dev: &'static Device,
}

static DEV: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Types under test. `BRECCIA_QTYPE=IQ4_XS` (or `IQ4_XS,Q6_K`) narrows every
/// test in this file to the named types, so a new kernel path can be run **one
/// type per invocation** on the non-display GPU and a fault is attributable
/// (`AGENTS.md` → GPU crash safety).
fn types_under_test() -> Vec<QuantType> {
    match std::env::var("BRECCIA_QTYPE") {
        Ok(v) if !v.trim().is_empty() => v
            .split(',')
            .map(|t| {
                t.trim()
                    .parse::<QuantType>()
                    .unwrap_or_else(|e| panic!("BRECCIA_QTYPE: {e}"))
            })
            .collect(),
        _ => QuantType::ALL.to_vec(),
    }
}

/// Device under test: `BRECCIA_KERNEL_DEVICE` takes a PCI address
/// (`0000:0e:00.0` = Radeon VII/MI50, the safe first target) or an enumeration
/// index. Unset = the first discrete GPU.
fn device_selector() -> Option<String> {
    std::env::var("BRECCIA_KERNEL_DEVICE")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn fixture() -> Option<Fixture> {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let selector = device_selector();
    // GPU tests are opt-in by explicit device (AGENTS.md, GPU crash safety):
    // `cargo test` on a developer box must never pick a GPU implicitly.
    let Some(selector) = selector else {
        eprintln!(
            "conformance: SKIPPED — set BRECCIA_KERNEL_DEVICE=<pci|index> \
             (e.g. 0000:0e:00.0 = MI50 first; see AGENTS.md GPU crash safety)"
        );
        return None;
    };
    let dev = DEV
        .get_or_init(|| match Device::open(Some(&selector)) {
            Ok(d) => {
                println!(
                    "conformance: {} ({}), subgroup {}, {}",
                    d.info.name, d.info.device_id, d.info.subgroup_size, d.info.pci_address
                );
                Some(d)
            }
            Err(e) => {
                eprintln!("conformance: SKIPPED — no usable Vulkan device: {e}");
                None
            }
        })
        .as_ref()?;
    Some(Fixture { _guard: guard, dev })
}

macro_rules! need_gpu {
    () => {
        match fixture() {
            Some(f) => f,
            None => {
                eprintln!("skipped (no GPU)");
                return;
            }
        }
    };
}

/// Max relative error, with an absolute floor of 1.0 for near-zero results.
fn max_rel(got: &[f32], want: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (f64::from(*g) - w).abs();
        m = m.max(d / w.abs().max(1.0));
    }
    m
}

fn shapes_for(ty: QuantType) -> Vec<(usize, usize)> {
    let qk = ty.block_elems();
    let mut v = vec![(1, qk * 10), (3, qk * 4), (640, 2560), (2560, 640)];
    // Tails: one partial block, and a shape that leaves several whole blocks
    // plus a partial one (exercises the step-wise `full` boundary).
    v.push((7, qk * 3 + qk / 3 + 1));
    v.push((2, qk + 1));
    v
}

#[test]
fn dequant_matches_cpu_reference_for_every_type() {
    let fx = need_gpu!();
    let dev = fx.dev;
    for ty in types_under_test() {
        for (rows, cols) in shapes_for(ty) {
            let (packed, _) = real::synthetic_inputs(ty, bench::Shape::new(rows, cols), 0x1234);
            let got = dequant_to_f32(dev.gpu(), ty, rows, cols, &packed).expect("dequant");
            let padded = cols.div_ceil(ty.block_elems()) * ty.block_elems();
            assert_eq!(got.len(), rows * padded, "{ty} {rows}x{cols} output size");

            let row_bytes = (padded / ty.block_elems()) * ty.block_bytes();
            let mut worst = 0.0f64;
            for r in 0..rows {
                let want = reference::dequant_row(ty, &packed[r * row_bytes..], cols);
                let row = &got[r * padded..r * padded + cols];
                let want_f64: Vec<f64> = want.iter().map(|v| f64::from(*v)).collect();
                worst = worst.max(max_rel(row, &want_f64));
            }
            assert!(
                worst <= 1e-6,
                "{ty} {rows}x{cols}: dequant max relative error {worst:e} > 1e-6"
            );
            println!("dequant {ty:<8} {rows:>5}x{cols:<5} max rel {worst:e}");
        }
    }
}

#[test]
fn gemv_matches_cpu_dot_for_every_type() {
    let fx = need_gpu!();
    let dev = fx.dev;
    for ty in types_under_test() {
        for (rows, cols) in shapes_for(ty) {
            let shape = bench::Shape::new(rows, cols);
            let (packed, x) = real::synthetic_inputs(ty, shape, 0xabcd);
            let want = reference::gemv(ty, &packed, rows, cols, &x);
            let want_f64: Vec<f64> = want.iter().map(|v| f64::from(*v)).collect();

            let gpu = dev.gpu();
            let g = Gemv::new(gpu, ty, rows, cols).expect("pipeline");
            let w = breccia_kernels::gemv::upload(gpu, &packed).expect("upload w");
            let xb =
                breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x)).expect("upload x");
            let out = breccia_kernels::gemv::alloc(gpu, (rows * 4) as u64).expect("alloc out");
            g.run(&w, &xb, &out).expect("gemv");
            let got = bench::read_out(dev, &out, rows).expect("read out");
            for b in [&w, &xb, &out] {
                gpu.destroy_buf(b);
            }

            let worst = max_rel(&got, &want_f64);
            assert!(
                worst <= 1e-3,
                "{ty} {rows}x{cols}: gemv max relative error {worst:e} > 1e-3"
            );
            println!(
                "gemv    {ty:<8} {rows:>5}x{cols:<5} max rel {worst:e} (|y| up to {:.3e})",
                want_f64.iter().fold(0.0f64, |a, b| a.max(b.abs()))
            );
        }
    }
}

#[test]
fn non_finite_scales_do_not_corrupt_the_device() {
    let fx = need_gpu!();
    let dev = fx.dev;
    for ty in types_under_test() {
        // BF16/F32 have no block scale, so there is nothing to poison.
        if matches!(ty, QuantType::Bf16 | QuantType::F32) {
            continue;
        }
        // Four blocks: +inf, -inf and NaN scales, then a finite block behind
        // them.
        //
        // What this asserts is not just the arithmetic: a decode unit must never
        // turn a scale into an address. A NaN scale converted to an integer is
        // how a decode kernel becomes an out-of-bounds SSBO read, which faults
        // the GPU rather than returning a wrong number. The *exact* decode of the
        // finite block behind the poisoned ones is the observable proof that no
        // address was corrupted, and
        // `lib.rs::no_float_to_integer_conversion_in_a_decode_unit` is the static
        // half of the same guard.
        let cols = ty.block_elems() * 4;
        let (mut packed, _) = real::synthetic_inputs(ty, bench::Shape::new(1, cols), 7);
        let bb = ty.block_bytes();
        for (case, bits) in [(0usize, 0x7c00u16), (1, 0xfc00u16), (2, 0x7e00u16)] {
            for &off in real::scale_offsets(ty) {
                packed[bb * case + off..bb * case + off + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let got = dequant_to_f32(dev.gpu(), ty, 1, cols, &packed).expect("dequant with inf/nan");
        let poisoned = ty.block_elems() * 3;
        assert!(
            got[..poisoned].iter().all(|v| !v.is_finite()),
            "{ty}: inf/-inf/nan blocks must decode to non-finite values, got {:?}",
            &got[..4]
        );
        // The finite block behind them must still decode exactly, which proves
        // the kernel neither faulted nor wedged.
        let want = reference::dequant_row(ty, &packed, cols);
        for (g, w) in got[poisoned..].iter().zip(want[poisoned..].iter()) {
            assert_eq!(g, w, "{ty}: finite block after non-finite ones");
        }
        println!("nonfinite  {ty:<8} ok (+inf, -inf, nan; neighbour still exact)");
    }
}

/// Denormal and zero scale fields are finite, so they must decode to *exactly*
/// the reference value (no flush-to-zero shortcut anywhere) and must not disturb
/// the blocks around them.
#[test]
fn denormal_and_zero_scales_decode_exactly() {
    let fx = need_gpu!();
    let dev = fx.dev;
    for ty in types_under_test() {
        if matches!(ty, QuantType::Bf16 | QuantType::F32) {
            continue; // no block scale field
        }
        let cols = ty.block_elems() * 3;
        let (mut packed, _) = real::synthetic_inputs(ty, bench::Shape::new(1, cols), 0x2b2b);
        let bb = ty.block_bytes();
        // Smallest positive fp16 subnormal, then a negative one, then 0.0.
        for (case, bits) in [(0usize, 0x0001u16), (1, 0x8001u16), (2, 0x0000u16)] {
            for &off in real::scale_offsets(ty) {
                packed[bb * case + off..bb * case + off + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let got = dequant_to_f32(dev.gpu(), ty, 1, cols, &packed).expect("dequant");
        let want = reference::dequant_row(ty, &packed, cols);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "{ty}: value {i} with a subnormal/zero scale");
        }
        assert!(
            got[ty.block_elems() * 2..].iter().all(|v| *v == 0.0),
            "{ty}: d == 0 must decode to exactly 0.0"
        );
        println!("subnormal  {ty:<8} ok (2^-24, -2^-24 and 0.0 scales, exact)");
    }
}

#[test]
fn zero_scale_blocks_are_exactly_zero() {
    let fx = need_gpu!();
    let dev = fx.dev;
    for ty in types_under_test() {
        if matches!(ty, QuantType::Bf16 | QuantType::F32) {
            continue; // no block scale to zero
        }
        let cols = ty.block_elems() * 2;
        let (mut packed, _) = real::synthetic_inputs(ty, bench::Shape::new(2, cols), 11);
        let bb = ty.block_bytes();
        for b in 0..4 {
            // Zero every scale field: d == 0 alone would still leave -dmin*m
            // for the types with a minimum (Q4_K/Q5_K).
            for &off in real::scale_offsets(ty) {
                packed[b * bb + off..b * bb + off + 2].copy_from_slice(&0u16.to_le_bytes());
            }
        }
        let got = dequant_to_f32(dev.gpu(), ty, 2, cols, &packed).expect("dequant");
        assert!(
            got.iter().all(|v| *v == 0.0),
            "{ty}: d=0 blocks must decode to exactly 0.0, got {:?}",
            &got[..4]
        );
    }
}

#[test]
fn split_k_matches_the_single_pass_kernel() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let gpu = dev.gpu();
    for ty in types_under_test() {
        let shape = bench::Shape::new(37, ty.block_elems() * 5 + ty.block_elems() / 2);
        let (packed, x) = real::synthetic_inputs(ty, shape, 0x5151);
        let want = reference::gemv(ty, &packed, shape.rows, shape.cols, &x);
        let want_f64: Vec<f64> = want.iter().map(|v| f64::from(*v)).collect();

        let g = Gemv::new(gpu, ty, shape.rows, shape.cols).expect("pipeline");
        let w = breccia_kernels::gemv::upload(gpu, &packed).expect("upload w");
        let xb = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x)).expect("upload x");
        let out = breccia_kernels::gemv::alloc(gpu, (shape.rows * 4) as u64).expect("alloc out");
        for splits in [2u32, 3, 5, 8] {
            let part =
                breccia_kernels::gemv::alloc(gpu, g.partial_bytes(splits)).expect("alloc partials");
            g.run_split(&w, &xb, &part, &out, splits)
                .expect("split gemv");
            let got = bench::read_out(dev, &out, shape.rows).expect("read out");
            gpu.destroy_buf(&part);
            let worst = max_rel(&got, &want_f64);
            assert!(
                worst <= 1e-3,
                "{ty} {shape} splits={splits}: max relative error {worst:e} > 1e-3"
            );
        }
        for b in [&w, &xb, &out] {
            gpu.destroy_buf(b);
        }
        println!("split-K  {ty:<8} {shape} 1/2/3/5/8 agree with the CPU reference");
    }
}

/// The batched multi-expert dispatch must reproduce the per-expert loop exactly
/// when the experts are **scattered** in the weight buffer (the top-10 of 512
/// case: each blob is contiguous, the set is not) and each expert has its own
/// activation vector (the routed `down` projection).
#[test]
fn batched_experts_match_the_per_expert_loop() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let gpu = dev.gpu();
    let experts = 5usize;
    // Blob placement order: expert e lands in slot perm[e], so a map that used
    // expert index as the base would produce visibly wrong rows.
    let perm = [3usize, 0, 4, 1, 2];

    for ty in types_under_test() {
        let cols = ty.block_elems() * 5 + ty.block_elems() / 3; // tail path too
        let shape = bench::Shape::new(13, cols);
        let batch = GemvBatch::new(gpu, ty, shape.rows, shape.cols).expect("batch pipeline");
        let single = Gemv::new(gpu, ty, shape.rows, shape.cols).expect("single pipeline");
        let blob = batch.blob_bytes() as usize;

        let mut w_bytes = vec![0u8; blob * experts];
        let mut xs = Vec::with_capacity(experts);
        for e in 0..experts {
            let (packed, x) =
                real::synthetic_inputs(ty, shape, 0xba7c ^ (e as u64).wrapping_mul(7919));
            assert_eq!(packed.len(), blob, "{ty} blob size");
            w_bytes[perm[e] * blob..(perm[e] + 1) * blob].copy_from_slice(&packed);
            xs.push(x);
        }

        let w = breccia_kernels::gemv::upload(gpu, &w_bytes).expect("upload W");
        let out =
            breccia_kernels::gemv::alloc(gpu, (shape.rows * experts * 4) as u64).expect("out");
        let words: Vec<u32> = (0..experts)
            .map(|e| perm[e] as u32 * batch.blob_words())
            .collect();

        for layout in [XLayout::PerExpert, XLayout::Shared] {
            // Per-expert activations are concatenated; Shared uses expert 0's.
            let mut x_all = Vec::new();
            match layout {
                XLayout::PerExpert => {
                    for x in &xs {
                        x_all.extend_from_slice(x);
                    }
                }
                XLayout::Shared => x_all.extend_from_slice(&xs[0]),
            }
            let xb = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x_all)).expect("x");
            let map_host = batch.map_for_blobs(&words, layout);
            let map = breccia_kernels::gemv::upload(
                gpu,
                &breccia_kernels::batch::map_to_bytes(&map_host),
            )
            .expect("map");
            batch
                .run(&w, &xb, &out, &map, experts)
                .expect("batched gemv");
            let got = bench::read_out(dev, &out, shape.rows * experts).expect("read out");
            gpu.destroy_buf(&map);

            // (a) against the CPU reference, per expert, with that expert's x
            let mut worst = 0.0f64;
            for e in 0..experts {
                let x_e = match layout {
                    XLayout::PerExpert => &xs[e],
                    XLayout::Shared => &xs[0],
                };
                let packed_e = &w_bytes[perm[e] * blob..(perm[e] + 1) * blob];
                let want: Vec<f64> = reference::gemv(ty, packed_e, shape.rows, cols, x_e)
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect();
                let rows = &got[e * shape.rows..(e + 1) * shape.rows];
                worst = worst.max(max_rel(rows, &want));
            }
            assert!(
                worst <= 1e-3,
                "{ty} {shape} x{experts} ({layout:?}): batched vs reference max rel {worst:e}"
            );

            // (b) bit-for-bit against the single-expert kernel on the same blob
            let mut x_one = Vec::new();
            for e in 0..experts {
                let x_e = match layout {
                    XLayout::PerExpert => &xs[e],
                    XLayout::Shared => &xs[0],
                };
                let w_e = breccia_kernels::gemv::upload(
                    gpu,
                    &w_bytes[perm[e] * blob..(perm[e] + 1) * blob],
                )
                .expect("upload blob");
                let x_e_buf = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(x_e))
                    .expect("upload x");
                let o_e =
                    breccia_kernels::gemv::alloc(gpu, (shape.rows * 4) as u64).expect("alloc");
                single.run(&w_e, &x_e_buf, &o_e).expect("single gemv");
                x_one.extend(bench::read_out(dev, &o_e, shape.rows).expect("read single"));
                for b in [&w_e, &x_e_buf, &o_e] {
                    gpu.destroy_buf(b);
                }
            }
            let gpu_worst = max_rel(
                &got,
                &x_one.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            );
            assert!(
                gpu_worst <= 1e-6,
                "{ty} {shape} x{experts} ({layout:?}): batched vs single-expert loop {gpu_worst:e}"
            );
            println!(
                "batch   {ty:<8} {shape:>9} x{experts} {:<10} reference {worst:e} · single-expert {gpu_worst:e}",
                format!("{layout:?}")
            );
            gpu.destroy_buf(&xb);
        }
        for b in [&w, &out] {
            gpu.destroy_buf(b);
        }
    }
}

/// A dispatch larger than the guaranteed `maxComputeWorkGroupCount[0]` (65535)
/// must wrap over `grid.y` instead of silently dropping the tail rows. 4.19 M
/// Q2_0 blocks = 75 MiB of weights, i.e. 65536 workgroups for the dequantiser.
#[test]
fn large_dispatch_wraps_over_grid_y() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let ty = QuantType::Q2_0;
    let shape = bench::Shape::new(8192, 32768); // 512 blocks/row -> 4_194_304 blocks
    let (packed, _) = real::synthetic_inputs(ty, shape, 0x9001);
    let got = dequant_to_f32(dev.gpu(), ty, shape.rows, shape.cols, &packed).expect("dequant");
    assert_eq!(got.len(), shape.rows * shape.cols);
    let row_bytes = (shape.cols / ty.block_elems()) * ty.block_bytes();
    for r in [0usize, 1, shape.rows / 2, shape.rows - 1] {
        let want = reference::dequant_row(ty, &packed[r * row_bytes..], shape.cols);
        let row = &got[r * shape.cols..(r + 1) * shape.cols];
        let want_f64: Vec<f64> = want.iter().map(|v| f64::from(*v)).collect();
        let worst = max_rel(row, &want_f64);
        assert!(worst <= 1e-6, "row {r}: max rel {worst:e}");
    }
    println!(
        "grid.y wrap ok: {} blocks over {} workgroups",
        4_194_304u64,
        4_194_304u64 / 64
    );
}
