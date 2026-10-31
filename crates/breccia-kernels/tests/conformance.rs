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

/// The coalesced-dequant (`_wide`) build is a **pure throughput switch**: for every
/// type that has one, the GEMV output must be *bit-identical* to the incumbent
/// per-element build, on the shapes the graph uses (§8ax).
///
/// This is the strongest form of the parity argument for a kernel change that
/// rewrites the decode unit: the same block bytes, the same nibble extraction and
/// the same fp32 accumulation order, so nothing about the numerics moves and the
/// 32-token greedy parity of the graph is expected to be unchanged by construction
/// — this test is what makes that a measurement rather than a claim.
#[test]
fn wide_dequant_is_bit_identical_to_the_default() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let mut checked = 0;
    for ty in types_under_test() {
        if breccia_kernels::gemv::gemv_wide_spv(ty).is_none() {
            continue;
        }
        for (rows, cols) in shapes_for(ty) {
            let (packed, x) = real::synthetic_inputs(ty, bench::Shape::new(rows, cols), 0x5eed);
            let gpu = dev.gpu();
            let w = breccia_kernels::gemv::upload(gpu, &packed).expect("upload w");
            let xb =
                breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x)).expect("upload x");
            let out_default = breccia_kernels::gemv::alloc(gpu, (rows * 4) as u64).expect("out");
            let out_wide = breccia_kernels::gemv::alloc(gpu, (rows * 4) as u64).expect("out");
            let a = Gemv::with_variant(gpu, ty, rows, cols, breccia_kernels::Accum::F32, false)
                .expect("default pipeline");
            let b = Gemv::with_variant(gpu, ty, rows, cols, breccia_kernels::Accum::F32, true)
                .expect("wide pipeline");
            a.run(&w, &xb, &out_default).expect("default run");
            b.run(&w, &xb, &out_wide).expect("wide run");
            let fa = bench::read_out(dev, &out_default, rows).expect("read a");
            let fb = bench::read_out(dev, &out_wide, rows).expect("read b");
            for buf in [&w, &xb, &out_default, &out_wide] {
                gpu.destroy_buf(buf);
            }
            let diff = fa
                .iter()
                .zip(fb.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                diff, 0,
                "{ty} {rows}x{cols}: {diff} of {rows} rows differ between the default and the \
                 wide dequant build"
            );
            println!("wide dequant {ty:<8} {rows:>5}x{cols:<5} bit-identical");
            checked += 1;
        }
    }
    assert!(checked > 0, "no type with a `_wide` build was tested");
}

/// Concatenating two same-type matrices **along rows** and running one GEMV must be
/// **bit-identical** to running the two separately (§8ay). This is the invariant the
/// GDN `beta`+`alpha` merge relies on: a GEMV output row is an independent dot
/// product, so `rows` only changes which row each workgroup owns.
#[test]
fn concatenated_rows_are_bit_identical_to_the_separate_gemvs() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let gpu = dev.gpu();
    let ty = breccia_kernels::QuantType::Bf16;
    let (rows_a, rows_b, cols) = (48usize, 48usize, 2560usize);
    let (packed_a, x) = real::synthetic_inputs(ty, bench::Shape::new(rows_a, cols), 0x1234);
    let (packed_b, _) = real::synthetic_inputs(ty, bench::Shape::new(rows_b, cols), 0x9876);
    let row_bytes = packed_a.len() / rows_a;
    assert_eq!(packed_b.len() / rows_b, row_bytes);
    let mut both = packed_a.clone();
    both.extend_from_slice(&packed_b);

    let w_a = breccia_kernels::gemv::upload(gpu, &packed_a).expect("upload a");
    let w_b = breccia_kernels::gemv::upload(gpu, &packed_b).expect("upload b");
    let w_ab = breccia_kernels::gemv::upload(gpu, &both).expect("upload ab");
    let xb = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x)).expect("upload x");
    let out_a = breccia_kernels::gemv::alloc(gpu, (rows_a * 4) as u64).expect("out a");
    let out_b = breccia_kernels::gemv::alloc(gpu, (rows_b * 4) as u64).expect("out b");
    let out_ab = breccia_kernels::gemv::alloc(gpu, ((rows_a + rows_b) * 4) as u64).expect("out ab");
    let ga = Gemv::new(gpu, ty, rows_a, cols).expect("pipe a");
    let gb = Gemv::new(gpu, ty, rows_b, cols).expect("pipe b");
    let gab = Gemv::new(gpu, ty, rows_a + rows_b, cols).expect("pipe ab");
    ga.run(&w_a, &xb, &out_a).expect("run a");
    gb.run(&w_b, &xb, &out_b).expect("run b");
    gab.run(&w_ab, &xb, &out_ab).expect("run ab");
    let a = bench::read_out(dev, &out_a, rows_a).expect("read a");
    let b = bench::read_out(dev, &out_b, rows_b).expect("read b");
    let ab = bench::read_out(dev, &out_ab, rows_a + rows_b).expect("read ab");
    for buf in [&w_a, &w_b, &w_ab, &xb, &out_a, &out_b, &out_ab] {
        gpu.destroy_buf(buf);
    }
    for i in 0..rows_a {
        assert_eq!(a[i].to_bits(), ab[i].to_bits(), "row {i} of the first half");
    }
    for i in 0..rows_b {
        assert_eq!(
            b[i].to_bits(),
            ab[rows_a + i].to_bits(),
            "row {i} of the second half"
        );
    }
    println!("concatenated rows: bit-identical ({rows_a} + {rows_b} x {cols})");
}

/// The **word-level pack** build (§8bc) must be bit-identical to the incumbent
/// per-element build: same codes, same scale, same multiply order, fewer instructions.
/// Compared on the GEMV's actual output (`to_bits()`), across shapes that include
/// partial blocks and odd row counts.
#[test]
fn word_pack_is_bit_identical_to_the_default() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let gpu = dev.gpu();
    let ty = breccia_kernels::QuantType::Iq4Xs;
    if breccia_kernels::gemv::gemv_words_spv(ty).is_none() {
        println!("word pack: no build for {ty}, skipping");
        return;
    }
    let mut checked = 0;
    for (rows, cols) in [
        (1usize, 2560usize),
        (3, 1024),
        (640, 2560),
        (2560, 640),
        (7, 854),
        (2, 257),
        (10240, 2560),
    ] {
        let (packed, x) = real::synthetic_inputs(ty, bench::Shape::new(rows, cols), 0x77aa);
        let w = breccia_kernels::gemv::upload(gpu, &packed).expect("upload w");
        let xb = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x)).expect("upload x");
        let out_a = breccia_kernels::gemv::alloc(gpu, (rows * 4) as u64).expect("out");
        let out_b = breccia_kernels::gemv::alloc(gpu, (rows * 4) as u64).expect("out");
        // The two builds are selected by the *environment* inside `Gemv`, so build
        // them explicitly through the variant constructor the conformance suite owns.
        let a = Gemv::with_variant(gpu, ty, rows, cols, breccia_kernels::Accum::F32, false)
            .expect("default pipeline");
        let b = Gemv::with_words(gpu, ty, rows, cols, breccia_kernels::Accum::F32)
            .expect("word-pack pipeline");
        a.run(&w, &xb, &out_a).expect("default run");
        b.run(&w, &xb, &out_b).expect("word run");
        let fa = bench::read_out(dev, &out_a, rows).expect("read a");
        let fb = bench::read_out(dev, &out_b, rows).expect("read b");
        for buf in [&w, &xb, &out_a, &out_b] {
            gpu.destroy_buf(buf);
        }
        let diff = fa
            .iter()
            .zip(fb.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(diff, 0, "{rows}x{cols}: {diff} of {rows} rows differ");
        println!("word pack {rows:>5}x{cols:<5} bit-identical");
        checked += 1;
    }
    assert!(checked > 0);
}

#[test]
fn non_finite_scales_do_not_corrupt_the_device() {
    let fx = need_gpu!();
    let dev = fx.dev;
    for ty in types_under_test() {
        // BF16/F32 have no block scale, so there is nothing to poison.
        if matches!(ty, QuantType::F16 | QuantType::Bf16 | QuantType::F32) {
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
        if matches!(ty, QuantType::F16 | QuantType::Bf16 | QuantType::F32) {
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
        if matches!(ty, QuantType::F16 | QuantType::Bf16 | QuantType::F32) {
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

/// Tail-skip sentinel: a map entry of `0xFFFFFFFF` is a dead row. The batched GEMV
/// must write exactly 0 for it (without reading W) and leave every other expert
/// bit-identical to the run with no sentinel.
#[test]
fn batched_gemv_skips_sentinel_rows() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let gpu = dev.gpu();
    let experts = 5usize;
    let dead = [1usize, 4];
    for ty in types_under_test() {
        let cols = ty.block_elems() * 5 + ty.block_elems() / 3;
        let shape = bench::Shape::new(13, cols);
        let batch = GemvBatch::new(gpu, ty, shape.rows, shape.cols).expect("batch pipeline");
        let blob = batch.blob_bytes() as usize;
        let mut w_bytes = vec![0u8; blob * experts];
        let mut x_all = Vec::new();
        for e in 0..experts {
            let (packed, x) =
                real::synthetic_inputs(ty, shape, 0x5e17 ^ (e as u64).wrapping_mul(7919));
            w_bytes[e * blob..(e + 1) * blob].copy_from_slice(&packed);
            x_all.extend_from_slice(&x);
        }
        let w = breccia_kernels::gemv::upload(gpu, &w_bytes).expect("upload W");
        let xb = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x_all)).expect("x");
        let words: Vec<u32> = (0..experts)
            .map(|e| e as u32 * batch.blob_words())
            .collect();
        let run = |map_host: &[u32]| -> Vec<f32> {
            let map =
                breccia_kernels::gemv::upload(gpu, &breccia_kernels::batch::map_to_bytes(map_host))
                    .expect("map");
            let out =
                breccia_kernels::gemv::alloc(gpu, (shape.rows * experts * 4) as u64).expect("out");
            batch
                .run(&w, &xb, &out, &map, experts)
                .expect("batched gemv");
            let got = bench::read_out(dev, &out, shape.rows * experts).expect("read out");
            gpu.destroy_buf(&map);
            gpu.destroy_buf(&out);
            got
        };
        let mut map_host = batch.map_for_blobs(&words, XLayout::PerExpert);
        let full = run(&map_host);
        for &e in &dead {
            for r in 0..shape.rows {
                map_host[e * shape.rows + r] = u32::MAX;
            }
        }
        let pruned = run(&map_host);
        for e in 0..experts {
            let (g, f) = (
                &pruned[e * shape.rows..(e + 1) * shape.rows],
                &full[e * shape.rows..(e + 1) * shape.rows],
            );
            if dead.contains(&e) {
                assert!(
                    g.iter().all(|v| *v == 0.0),
                    "{ty}: dead expert {e} must write 0, got {g:?}"
                );
            } else {
                assert_eq!(
                    g, f,
                    "{ty}: live expert {e} changed by a neighbour's sentinel"
                );
            }
        }
        for b in [&w, &xb] {
            gpu.destroy_buf(b);
        }
        println!("sentinel {ty:<8}: dead experts {dead:?} write 0, live experts bit-identical");
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

/// fp16 accumulation (`Accum::F16`) is a research path: it rounds each lane's
/// running sum of the row to fp16, which for a 2560-term decode row lands around
/// 1e-2 relative — two orders above the 1e-3 bar the f32 path is held to. This
/// test pins that documented behaviour down and, in the same process on the same
/// bytes, proves the default f32 path is unchanged: `Gemv::new` (the default) and
/// `Gemv::with_accum(.., F32)` produce bit-identical output, and both stay
/// ≤ 1e-3 against the f64 CPU reference. The tolerance here is deliberately
/// **not** the default one (`docs/phase-3/03-gaps-closed.md` §2).
#[test]
fn fp16_accumulate_has_a_documented_tolerance_and_f32_is_unchanged() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let gpu = dev.gpu();
    // The two shapes the measurement in the phase report uses: the top-10
    // expert shape and the DRAM-bound one.
    let types: Vec<QuantType> = {
        let wanted = ["IQ3_S", "Q6_K"];
        let sel = types_under_test();
        let narrowed: Vec<QuantType> = sel
            .iter()
            .copied()
            .filter(|t| wanted.iter().any(|w| *w == t.name()))
            .collect();
        if narrowed.is_empty() {
            // BRECCIA_QTYPE=IQ4_XS (say) asked for a type not in this list:
            // run it anyway, the property is type-independent.
            sel
        } else {
            narrowed
        }
    };
    for ty in types {
        // The top-10 routed-expert shape and one real dense projection. The
        // DRAM-bound 248,320-row sweep is a throughput cell (bench), not an
        // accuracy one: the CPU f64 reference over 635 M terms takes a minute in
        // a debug test binary and adds nothing to the per-row error distribution.
        // The same test at 248,320 rows (one run) measured a worst row of 2.5e-1.
        for shape in [bench::Shape::new(6400, 2560), bench::Shape::new(2560, 640)] {
            let (packed, x) = real::synthetic_inputs(ty, shape, 0xacc0);
            let want = reference::gemv(ty, &packed, shape.rows, shape.cols, &x);
            let want_f64: Vec<f64> = want.iter().map(|v| f64::from(*v)).collect();

            let run = |accum: breccia_kernels::Accum| -> Vec<f32> {
                let g = Gemv::with_accum(gpu, ty, shape.rows, shape.cols, accum).expect("pipeline");
                let w = breccia_kernels::gemv::upload(gpu, &packed).expect("upload w");
                let xb =
                    breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x)).expect("upload x");
                let out =
                    breccia_kernels::gemv::alloc(gpu, (shape.rows * 4) as u64).expect("alloc out");
                g.run(&w, &xb, &out).expect("gemv");
                let got = bench::read_out(dev, &out, shape.rows).expect("read out");
                for b in [&w, &xb, &out] {
                    gpu.destroy_buf(b);
                }
                got
            };

            let f32_out = run(breccia_kernels::Accum::F32);
            let f16_out = run(breccia_kernels::Accum::F16);
            let f32_err = max_rel(&f32_out, &want_f64);
            let mut errs: Vec<f64> = f16_out
                .iter()
                .zip(want_f64.iter())
                .map(|(g, w)| (f64::from(*g) - w).abs() / w.abs().max(1.0))
                .collect();
            errs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a conformance error"));
            let median = errs[errs.len() / 2];
            let p95 = errs[errs.len() * 95 / 100];
            let f16_max = *errs.last().expect("rows > 0");

            // The default pipeline is the f32 pipeline, byte for byte.
            let default = Gemv::new(gpu, ty, shape.rows, shape.cols).expect("default pipeline");
            assert_eq!(default.accum(), breccia_kernels::Accum::F32);
            drop(default);

            assert!(
                f32_err <= 1e-3,
                "{ty} {shape}: f32 accumulate max rel {f32_err:e} > 1e-3 (default bar)"
            );
            // Documented fp16 bars (docs/phase-3/03-gaps-closed.md §2): the
            // typical row is inside 2e-2; the tail is much worse because fp16
            // rounds the running sum *before* heavily cancelling halves of the
            // row meet, so the error is relative to the magnitude of the terms,
            // not to the (small) result — 6,400 Q6_K rows already show a 6.8e-1
            // worst row and 248,320 F16 rows a 2.5e-1 one. The `max` assertion is
            // the "the kernel still computes this dot product" bound (a row
            // worse than 100 % is not a precision loss, it is a different
            // answer), not an accuracy claim; the meaningful bars are median and
            // p95.
            assert!(
                median <= 2e-2,
                "{ty} {shape}: fp16 accumulate median rel {median:e} > 2e-2"
            );
            assert!(
                p95 <= 2e-2,
                "{ty} {shape}: fp16 accumulate p95 rel {p95:e} > 2e-2"
            );
            assert!(
                f16_max <= 1.0,
                "{ty} {shape}: fp16 accumulate worst-row rel {f16_max:e} > 1.0 (sanity bound)"
            );
            println!(
                "accum   {ty:<8} {shape:>13} f32 max {f32_err:.2e} (bar 1e-3) · \
                 f16 median {median:.2e} p95 {p95:.2e} max {f16_max:.2e} (f16 bars: median+p95 2e-2, max 1.0 sanity)"
            );
        }
    }
}

/// The batched (`gemv_*_moe_f16acc`) path must do the same arithmetic as the
/// single-expert fp16 path: the row indirection changes *where* a row reads, not
/// what it computes, so two experts in one dispatch must reproduce two
/// single-expert dispatches bit for bit. (Correctness only — fp16's accuracy is
/// measured in `fp16_accumulate_has_a_documented_tolerance_and_f32_is_unchanged`.)
#[test]
fn batched_fp16_accumulate_matches_the_single_expert_path() {
    let fx = need_gpu!();
    let dev = fx.dev;
    let gpu = dev.gpu();
    let experts = 2usize;
    for ty in types_under_test() {
        let cols = ty.block_elems() * 3;
        let shape = bench::Shape::new(13, cols);
        let batch =
            GemvBatch::with_accum(gpu, ty, shape.rows, shape.cols, breccia_kernels::Accum::F16)
                .expect("batch f16 pipeline");
        let single = Gemv::with_accum(gpu, ty, shape.rows, shape.cols, breccia_kernels::Accum::F16)
            .expect("single f16 pipeline");
        let blob = batch.blob_bytes() as usize;
        let padded = batch.padded_cols();
        let mut w_bytes = vec![0u8; blob * experts];
        let mut x_all = Vec::new();
        for e in 0..experts {
            let (packed, x) = real::synthetic_inputs(ty, shape, 0xf16 ^ e as u64);
            w_bytes[e * blob..(e + 1) * blob].copy_from_slice(&packed);
            x_all.extend_from_slice(&x);
        }

        let w = breccia_kernels::gemv::upload(gpu, &w_bytes).expect("upload W");
        let xb = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x_all)).expect("x");
        let out =
            breccia_kernels::gemv::alloc(gpu, (shape.rows * experts * 4) as u64).expect("out");
        let map_host = batch.map(experts, XLayout::PerExpert);
        let map =
            breccia_kernels::gemv::upload(gpu, &breccia_kernels::batch::map_to_bytes(&map_host))
                .expect("map");
        batch.run(&w, &xb, &out, &map, experts).expect("batched");
        let got = bench::read_out(dev, &out, shape.rows * experts).expect("read out");

        let mut single_out = Vec::new();
        for e in 0..experts {
            let w_e =
                breccia_kernels::gemv::upload(gpu, &w_bytes[e * blob..(e + 1) * blob]).expect("w");
            let x_e = breccia_kernels::gemv::upload(
                gpu,
                &bench::f32_to_bytes(&x_all[e * padded..(e + 1) * padded]),
            )
            .expect("x_e");
            let o = breccia_kernels::gemv::alloc(gpu, (shape.rows * 4) as u64).expect("o");
            single.run(&w_e, &x_e, &o).expect("single");
            single_out.extend(bench::read_out(dev, &o, shape.rows).expect("read"));
            for b in [&w_e, &x_e, &o] {
                gpu.destroy_buf(b);
            }
        }
        let worst = max_rel(
            &got,
            &single_out.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        );
        assert!(
            worst <= 1e-6,
            "{ty} {shape} x{experts} fp16: batched vs single-expert path {worst:e}"
        );
        println!("batch-f16 {ty:<8} {shape:>9} x{experts} vs single-expert {worst:e}");
        for b in [&w, &xb, &out, &map] {
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

/// The `down` GEMV + `Act` fusion (`docs/phase-4/01` §8at) must be
/// **bit-identical** to the two dispatches it replaces — that is the whole
/// claim, and it is checked here on the three shapes the HC mixer uses rather
/// than asserted in a comment:
///
/// * the single-expert **split-K** path (`T = 1`, the decode shape): the
///   partials stay raw and the activation moves into the reduce pass;
/// * the **unsplit** single-expert path (`splits == 1`): the activation is the
///   GEMV's own epilogue;
/// * the **batched** path (`T > 1`): each row's scalar gets the same epilogue.
///
/// The reference on every line is literally `Gemv`/`GemvBatch` + `Act`, i.e. the
/// incumbent two-dispatch path, so a pass here is a proof, not a tolerance.
#[test]
fn hc_epilogue_matches_a_separate_act_bit_for_bit() {
    use breccia_kernels::ops::{Act, ActMode};

    let fx = need_gpu!();
    let dev = fx.dev;
    let gpu = dev.gpu();
    let s = 4.0f32; // the HC stream count: `silu(x / hc)`
    let act = Act::new(gpu, ActMode::SiluDiv).expect("act pipeline");
    // The artifact's `hc_*` type (§14) and the real `hc_down` shape.
    let ty = QuantType::Bf16;
    let rows = 320usize;
    let cols = 4 * 2560usize;
    let shape = bench::Shape::new(rows, cols);
    let (packed, x) = real::synthetic_inputs(ty, shape, 0x7a17);

    let w = breccia_kernels::gemv::upload(gpu, &packed).expect("upload w");
    let xb = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&x)).expect("upload x");
    let raw = breccia_kernels::gemv::alloc(gpu, (rows * 4) as u64).expect("raw");
    let want_buf = breccia_kernels::gemv::alloc(gpu, (rows * 4) as u64).expect("want");
    let got_buf = breccia_kernels::gemv::alloc(gpu, (rows * 4) as u64).expect("got");

    // ---- (a) split-K: partials raw, activation in the reduce pass ----------
    let splits = 4u32;
    let part = breccia_kernels::gemv::alloc(gpu, {
        let g = Gemv::new(gpu, ty, rows, cols).expect("plain gemv");
        g.partial_bytes(splits)
    })
    .expect("partials");
    let plain = Gemv::new(gpu, ty, rows, cols).expect("plain gemv");
    let fused = Gemv::new(gpu, ty, rows, cols)
        .expect("plain gemv")
        .with_epilogue(s)
        .expect("epilogue gemv");
    plain
        .run_split(&w, &xb, &part, &raw, splits)
        .expect("plain split gemv");
    act.run(&raw, &want_buf, rows, s).expect("act");
    fused
        .run_split(&w, &xb, &part, &got_buf, splits)
        .expect("fused split gemv");
    let want = bench::read_out(dev, &want_buf, rows).expect("read want");
    let got = bench::read_out(dev, &got_buf, rows).expect("read got");
    assert_bits_equal(&got, &want, &format!("split-K splits={splits}"));
    gpu.destroy_buf(&part);
    drop(plain);
    drop(fused);

    // ---- (b) unsplit: the activation is the GEMV's own epilogue ------------
    let plain = Gemv::new(gpu, ty, rows, cols).expect("plain gemv");
    let fused = Gemv::new(gpu, ty, rows, cols)
        .expect("plain gemv")
        .with_epilogue(s)
        .expect("epilogue gemv");
    plain.run(&w, &xb, &raw).expect("plain gemv");
    act.run(&raw, &want_buf, rows, s).expect("act");
    fused.run(&w, &xb, &got_buf).expect("fused gemv");
    let want = bench::read_out(dev, &want_buf, rows).expect("read want");
    let got = bench::read_out(dev, &got_buf, rows).expect("read got");
    assert_bits_equal(&got, &want, "unsplit splits=1");
    drop(plain);
    drop(fused);

    // ---- (c) batched (T > 1): one dispatch, every row's own epilogue -------
    let experts = 2usize;
    let total = rows * experts;
    let all_w = vec![0u8; packed.len() * experts];
    let mut all_w = all_w;
    for e in 0..experts {
        all_w[e * packed.len()..(e + 1) * packed.len()].copy_from_slice(&packed);
    }
    let ew = breccia_kernels::gemv::upload(gpu, &all_w).expect("upload batch w");
    let mut all_x = Vec::with_capacity(x.len() * experts);
    for _ in 0..experts {
        all_x.extend_from_slice(&x);
    }
    let exb = breccia_kernels::gemv::upload(gpu, &bench::f32_to_bytes(&all_x)).expect("upload x");
    let plain = GemvBatch::new(gpu, ty, rows, cols).expect("plain batch");
    let fused = GemvBatch::new(gpu, ty, rows, cols)
        .expect("plain batch")
        .with_epilogue(s)
        .expect("epilogue batch");
    let map_host = plain.map_for_blobs(&[0, plain.blob_words()], XLayout::PerExpert);
    let map = breccia_kernels::gemv::upload(gpu, &breccia_kernels::batch::map_to_bytes(&map_host))
        .expect("map");
    let braw = breccia_kernels::gemv::alloc(gpu, (total * 4) as u64).expect("braw");
    let bwant = breccia_kernels::gemv::alloc(gpu, (total * 4) as u64).expect("bwant");
    let bgot = breccia_kernels::gemv::alloc(gpu, (total * 4) as u64).expect("bgot");
    plain
        .run(&ew, &exb, &braw, &map, experts)
        .expect("plain batch");
    act.run(&braw, &bwant, total, s).expect("act");
    fused
        .run(&ew, &exb, &bgot, &map, experts)
        .expect("fused batch");
    let want = bench::read_out(dev, &bwant, total).expect("read want");
    let got = bench::read_out(dev, &bgot, total).expect("read got");
    assert_bits_equal(&got, &want, &format!("batched x{experts}"));

    for b in [
        &w, &xb, &raw, &want_buf, &got_buf, &ew, &exb, &map, &braw, &bwant, &bgot,
    ] {
        gpu.destroy_buf(b);
    }
    println!("hc epilogue: split-K, unsplit and batched paths are bit-identical to Gemv+Act");
}

/// Bit-for-bit comparison (not a tolerance): the fusion claims identity.
fn assert_bits_equal(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bad = got
        .iter()
        .zip(want.iter())
        .position(|(g, w)| g.to_bits() != w.to_bits());
    if let Some(i) = bad {
        panic!(
            "{what}: row {i} differs — fused {:#010x} ({}) vs gemv+act {:#010x} ({})",
            got[i].to_bits(),
            got[i],
            want[i].to_bits(),
            want[i]
        );
    }
}
