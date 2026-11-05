//! The mixed-float-dot GEMM (`gemm::DotGemm`): correctness against a CPU reference
//! and timing against the tiled BF16 build.
//!
//! The kernel body is assembly (`shaders/gemm_f16_dot.spvasm`) because
//! `OpFDot2MixAcc32VALVE` is unreachable from GLSL — measured, the driver emits
//! `v_fma_mix_f32` pairs for the equivalent GLSL (audit §13).
//!
//! `BRECCIA_KERNEL_DEVICE=<pci|index>` selects the device.

use breccia_kernels::gemm::{DotGemm, GemmMany};
use breccia_kernels::gemv::{alloc, upload};
use breccia_kernels::QuantType;
use breccia_vk::ctx::{Gpu, Host};

fn open(selector: &str) -> Option<(Host, Gpu)> {
    let host = Host::new().ok()?;
    let devs = host.devices().ok()?;
    let want = selector.trim();
    let idx = devs
        .iter()
        .position(|d| d.pci_address == want || d.index.to_string() == want)
        .unwrap_or(0);
    let gpu = Gpu::create(&host, &devs[idx], 1).ok()?;
    Some((host, gpu))
}

/// f16 bits of `v`, the same rounding the kernel's `OpFConvert` applies.
fn half(v: f32) -> u16 {
    // Narrow round-trip: f32 -> f16 (round to nearest even) -> bits.
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        return sign; // underflow to zero is close enough for a parity test
    }
    if exp >= 31 {
        return sign | 0x7c00; // overflow to inf
    }
    sign | ((exp as u16) << 10) | man as u16
}

fn to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32 / 1024.0;
    if exp == 0 {
        sign * man * 2f32.powi(-14)
    } else {
        sign * (1.0 + man) * 2f32.powi(exp - 15)
    }
}

#[test]
fn dot_gemm_matches_the_cpu_reference() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("dot_gemm: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        eprintln!("dot_gemm: no device for {selector}");
        return;
    };
    if !DotGemm::supported(&gpu) {
        eprintln!("dot_gemm: SKIPPED — device does not advertise the extension");
        return;
    }
    let (rows, cols, users) = (64usize, 256usize, 4usize);
    // Deterministic values, small enough that f16 holds them exactly (multiples of
    // 1/64 below 8), so the reference is the kernel's arithmetic, not a rounding
    // argument.
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 20) & 0xff) as f32 / 64.0 - 2.0
    };
    let wf: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let xf: Vec<f32> = (0..users * cols).map(|_| next()).collect();
    // Weights as f16 pairs (two per u32 word, low half first).
    let mut wbytes = Vec::new();
    for pair in wf.chunks(2) {
        let lo = half(pair[0]) as u32;
        let hi = half(pair[1]) as u32;
        wbytes.extend_from_slice(&(lo | (hi << 16)).to_le_bytes());
    }
    let mut xbytes = Vec::new();
    for v in &xf {
        xbytes.extend_from_slice(&v.to_le_bytes());
    }
    let w = upload(&gpu, &wbytes).expect("w");
    let x = upload(&gpu, &xbytes).expect("x");
    let o = alloc(&gpu, (rows * users * 4) as u64).expect("o");

    let g = DotGemm::new(&gpu).expect("pipeline");
    g.run(&w, &x, &o, rows, cols, users).expect("dispatch");
    let got = breccia_kernels::ops::download_f32(&gpu, &o, rows * users).expect("readback");

    let mut worst = 0f32;
    for u in 0..users {
        for r in 0..rows {
            let mut acc = 0f32;
            for k in 0..cols {
                acc += to_f32(half(wf[r * cols + k])) * to_f32(half(xf[u * cols + k]));
            }
            let d = (got[u * rows + r] - acc).abs();
            worst = worst.max(d);
        }
    }
    println!("dot_gemm: {rows}x{cols} x {users} users, worst |err| vs CPU {worst:.3e}");
    assert!(worst < 1e-3, "mixed-dot GEMM off by {worst:.3e}");
    gpu.destroy_buf(&w);
    gpu.destroy_buf(&x);
    gpu.destroy_buf(&o);
}

/// Timing against the tiled BF16 build (same 2 bytes per weight, same shapes).
#[test]
fn dot_gemm_vs_the_tiled_bf16_build() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("dot_gemm_speed: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        return;
    };
    if !DotGemm::supported(&gpu) {
        eprintln!("dot_gemm_speed: SKIPPED — no extension");
        return;
    }
    let g = DotGemm::new(&gpu).expect("naive pipeline");
    let gt = DotGemm::new_tiled(&gpu).expect("tiled pipeline");
    println!(
        "dot_gemm: {:>10} {:>6} {:>9} {:>9}",
        "shape", "users", "dot us", "bf16 us"
    );
    for (rows, cols, shape) in [(640usize, 2560usize, "gate"), (2560, 640, "down")] {
        let gemm = GemmMany::new(&gpu, QuantType::Bf16).expect("bf16 gemm");
        for users in [128usize, 1024] {
            let wbytes = gemm.weight_bytes(rows, cols);
            let w = alloc(&gpu, wbytes).expect("w");
            let x = alloc(&gpu, (users * cols * 4) as u64).expect("x");
            let o = alloc(&gpu, (users * rows * 4) as u64).expect("o");
            upload(&gpu, &vec![0u8; wbytes as usize]).expect("wz");
            upload(&gpu, &vec![0u8; users * cols * 4]).expect("xz");
            let mut t_dot = f64::INFINITY;
            let mut t_tiled = f64::INFINITY;
            let mut t_bf16 = f64::INFINITY;
            for pass in 0..4 {
                let t0 = std::time::Instant::now();
                g.run(&w, &x, &o, rows, cols, users).expect("dot");
                if pass > 0 {
                    t_dot = t_dot.min(t0.elapsed().as_secs_f64() * 1e6);
                }
                let tt = std::time::Instant::now();
                gt.run_tiled(&w, &x, &o, rows, cols, users)
                    .expect("tiled dot");
                if pass > 0 {
                    t_tiled = t_tiled.min(tt.elapsed().as_secs_f64() * 1e6);
                }
                let t1 = std::time::Instant::now();
                gemm.run_at(&w, &x, &o, rows, cols, users, 0).expect("bf16");
                if pass > 0 {
                    t_bf16 = t_bf16.min(t1.elapsed().as_secs_f64() * 1e6);
                }
            }
            println!(
                "dot_gemm: {:>10} {users:>6} {t_dot:>9.1} {t_tiled:>9.1} {t_bf16:>9.1}  \
                 (tiled vs bf16 {:.2}x)",
                shape,
                t_bf16 / t_tiled
            );
            gpu.destroy_buf(&w);
            gpu.destroy_buf(&x);
            gpu.destroy_buf(&o);
        }
    }
}

/// The tiled body must agree with the naive one element for element (same
/// arithmetic, different schedule), and both with the CPU reference.
#[test]
fn tiled_dot_gemm_matches_the_naive_body() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("tiled_dot: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        return;
    };
    if !DotGemm::supported(&gpu) {
        eprintln!("tiled_dot: SKIPPED — no extension");
        return;
    }
    // Deliberately not tile-aligned: rows 67 > 64, users 11 > 8, cols 96 (>BK),
    // so the clamps are exercised.
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (rows, cols, users) = (
        envn("BRECCIA_DOT_ROWS", 67),
        envn("BRECCIA_DOT_COLS", 96),
        envn("BRECCIA_DOT_USERS", 11),
    );
    let mut seed = 0xdead_beefu32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 20) & 0xff) as f32 / 64.0 - 2.0
    };
    let wf: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let xf: Vec<f32> = (0..users * cols).map(|_| next()).collect();
    let mut wbytes = Vec::new();
    for pair in wf.chunks(2) {
        let lo = half(pair[0]) as u32;
        let hi = half(pair[1]) as u32;
        wbytes.extend_from_slice(&(lo | (hi << 16)).to_le_bytes());
    }
    let mut xbytes = Vec::new();
    for v in &xf {
        xbytes.extend_from_slice(&v.to_le_bytes());
    }
    let w = upload(&gpu, &wbytes).expect("w");
    let x = upload(&gpu, &xbytes).expect("x");
    let a = alloc(&gpu, (rows * users * 4) as u64).expect("a");
    let b = alloc(&gpu, (rows * users * 4) as u64).expect("b");
    DotGemm::new(&gpu)
        .expect("naive")
        .run(&w, &x, &a, rows, cols, users)
        .expect("naive dispatch");
    DotGemm::new_tiled(&gpu)
        .expect("tiled")
        .run_tiled(&w, &x, &b, rows, cols, users)
        .expect("tiled dispatch");
    let va = breccia_kernels::ops::download_f32(&gpu, &a, rows * users).expect("a");
    let vb = breccia_kernels::ops::download_f32(&gpu, &b, rows * users).expect("b");
    let worst = va
        .iter()
        .zip(&vb)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let mut worst_cpu = 0f32;
    for u in 0..users {
        for r in 0..rows {
            let mut acc = 0f32;
            for k in 0..cols {
                acc += to_f32(half(wf[r * cols + k])) * to_f32(half(xf[u * cols + k]));
            }
            worst_cpu = worst_cpu.max((vb[u * rows + r] - acc).abs());
        }
    }
    println!(
        "tiled_dot: {rows}x{cols} x {users} users — tiled vs naive {worst:.3e}, \
         tiled vs CPU {worst_cpu:.3e}"
    );
    assert!(worst < 1e-4 && worst_cpu < 1e-3, "tiled body diverges");
    for v in 0..DotGemm::MT_TILES.len() {
        DotGemm::new_mt(&gpu, v)
            .expect("micro-tiled")
            .run_tiled(&w, &x, &b, rows, cols, users)
            .expect("micro-tiled dispatch");
        let vm = breccia_kernels::ops::download_f32(&gpu, &b, rows * users).expect("b");
        let worst_mt = va
            .iter()
            .zip(&vm)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        println!(
            "tiled_dot: micro-tile {:?} vs naive {worst_mt:.3e}",
            DotGemm::MT_TILES[v]
        );
        assert!(worst_mt < 1e-4, "micro-tiled body {v} diverges");
    }
    gpu.destroy_buf(&w);
    gpu.destroy_buf(&x);
    gpu.destroy_buf(&a);
    gpu.destroy_buf(&b);
}

/// **Selection path**: a calibrated `dot` cell plus a caller that declares f16
/// weights must make `GemmMany::run_at` dispatch the mixed-dot body — and the
/// result must match a CPU reference. Without the declaration it must stay on the
/// tile build, so a quantised tensor can never be read as f16.
#[test]
fn a_calibrated_dot_cell_routes_run_at_to_the_dot_body() {
    use breccia_kernels::gemm::{set_tile_table, GemmMany, TileCell};
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("dot_select: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        return;
    };
    if !DotGemm::supported(&gpu) {
        eprintln!("dot_select: SKIPPED — no extension");
        return;
    }
    let (rows, cols, users) = (64usize, 64usize, 8usize);
    let mut seed = 0x5eed_1234u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 20) & 0xff) as f32 / 64.0 - 2.0
    };
    let wf: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let xf: Vec<f32> = (0..users * cols).map(|_| next()).collect();
    let mut wbytes = Vec::new();
    for pair in wf.chunks(2) {
        let lo = half(pair[0]) as u32;
        let hi = half(pair[1]) as u32;
        wbytes.extend_from_slice(&(lo | (hi << 16)).to_le_bytes());
    }
    let mut xbytes = Vec::new();
    for v in &xf {
        xbytes.extend_from_slice(&v.to_le_bytes());
    }
    let w = upload(&gpu, &wbytes).expect("w");
    let x = upload(&gpu, &xbytes).expect("x");
    let o = alloc(&gpu, (rows * users * 4) as u64).expect("o");

    // The dot wins at this cell (calibrated), and the caller declares f16 weights.
    set_tile_table(vec![TileCell {
        ty: "BF16".into(),
        rows,
        users,
        tile: "dot".into(),
        us: 1.0,
        ..Default::default()
    }]);
    let mut gemm = GemmMany::new(&gpu, QuantType::Bf16).expect("gemm");
    assert!(!gemm.dot_available(), "not declared yet");
    gemm.set_dot_weights(true);
    assert!(gemm.dot_available(), "declared f16 weights");
    gemm.run_at(&w, &x, &o, rows, cols, users, 0)
        .expect("run_at");
    let got = breccia_kernels::ops::download_f32(&gpu, &o, rows * users).expect("readback");

    let mut worst = 0f32;
    for u in 0..users {
        for r in 0..rows {
            let mut acc = 0f32;
            for k in 0..cols {
                acc += to_f32(half(wf[r * cols + k])) * to_f32(half(xf[u * cols + k]));
            }
            worst = worst.max((got[u * rows + r] - acc).abs());
        }
    }
    println!("dot_select: run_at via the calibrated dot cell, worst |err| {worst:.3e}");
    assert!(worst < 1e-3, "the routed dot body is wrong: {worst:.3e}");
    set_tile_table(Vec::new());
    gpu.destroy_buf(&w);
    gpu.destroy_buf(&x);
    gpu.destroy_buf(&o);
}

/// **The prepass**: a BF16 weight matrix declared eligible (`set_dot_weights`) and a
/// calibrated `dot` cell must run the mixed-dot body on an f16-pair copy of those
/// weights, and match a CPU reference that does the same BF16 -> f16 conversion.
/// Values are multiples of 1/64 in [-2, 2), i.e. exact in both formats, so the
/// reference is the kernel's arithmetic rather than a rounding argument.
#[test]
fn a_bf16_matrix_runs_on_the_dot_after_the_f16_prepass() {
    use breccia_kernels::gemm::{set_tile_table, GemmMany, TileCell};
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("dot_prepass: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        return;
    };
    if !DotGemm::supported(&gpu) {
        eprintln!("dot_prepass: SKIPPED — no extension");
        return;
    }
    println!(
        "dot_prepass: {:>10} {:>6} {:>10} {:>10}  ratio",
        "shape", "users", "dot us", "bf16 us"
    );
    for (rows, cols, users, name) in [
        (640usize, 2560usize, 128usize, "gate"),
        (640, 2560, 1024, "gate"),
        (2560, 640, 128, "down"),
        (2560, 640, 1024, "down"),
    ] {
        let mut seed = 0x51ed_2a37u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 20) & 0xff) as f32 / 64.0 - 2.0
        };
        let wf: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
        let xf: Vec<f32> = (0..users * cols).map(|_| next()).collect();
        // BF16 storage (top 16 bits), which is what the engine uploads.
        let mut wbytes = Vec::with_capacity(rows * cols * 2);
        for v in &wf {
            wbytes.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        let mut xbytes = Vec::with_capacity(users * cols * 4);
        for v in &xf {
            xbytes.extend_from_slice(&v.to_le_bytes());
        }
        let w = upload(&gpu, &wbytes).expect("w");
        let x = upload(&gpu, &xbytes).expect("x");
        let o = alloc(&gpu, (rows * users * 4) as u64).expect("o");
        let o2 = alloc(&gpu, (rows * users * 4) as u64).expect("o2");

        // Calibration says this cell wants the dot.
        set_tile_table(vec![TileCell {
            ty: "BF16".into(),
            rows,
            users,
            tile: "dot".into(),
            us: 1.0,
            ..Default::default()
        }]);
        let mut dot = GemmMany::new(&gpu, QuantType::Bf16).expect("gemm");
        // The weights are BF16: the prepass converts and caches them.
        dot.set_dot_bf16(true);
        let plain = GemmMany::new(&gpu, QuantType::Bf16).expect("plain gemm");

        dot.run_at(&w, &x, &o, rows, cols, users, 0)
            .expect("dot run");
        plain
            .run_at(&w, &x, &o2, rows, cols, users, 0)
            .expect("bf16 run");
        let got = breccia_kernels::ops::download_f32(&gpu, &o, rows * users).expect("read");
        let want = breccia_kernels::ops::download_f32(&gpu, &o2, rows * users).expect("read2");

        let mut worst_cpu = 0f32;
        let mut worst_bf16 = 0f32;
        // Sample the first 4 users and 16 rows: the kernel is uniform over both, so
        // a full O(rows*cols*users) CPU reference would cost seconds for nothing.
        for u in 0..users.min(4) {
            for r in 0..rows.min(16) {
                let mut acc = 0f32;
                for k in 0..cols {
                    acc += to_f32(half(wf[r * cols + k])) * to_f32(half(xf[u * cols + k]));
                }
                worst_cpu = worst_cpu.max((got[u * rows + r] - acc).abs());
                worst_bf16 = worst_bf16.max((got[u * rows + r] - want[u * rows + r]).abs());
            }
        }
        // Timing: best of 4 for each path.
        let mut t_dot = f64::INFINITY;
        let mut t_bf16 = f64::INFINITY;
        for pass in 0..5 {
            let t0 = std::time::Instant::now();
            dot.run_at(&w, &x, &o, rows, cols, users, 0).expect("dot");
            if pass > 0 {
                t_dot = t_dot.min(t0.elapsed().as_secs_f64() * 1e6);
            }
            let t1 = std::time::Instant::now();
            plain
                .run_at(&w, &x, &o2, rows, cols, users, 0)
                .expect("bf16");
            if pass > 0 {
                t_bf16 = t_bf16.min(t1.elapsed().as_secs_f64() * 1e6);
            }
        }
        println!(
            "dot_prepass: {name:>10} {users:>6} {t_dot:>10.1} {t_bf16:>10.1}  {:.2}x  \
             |err vs CPU| {worst_cpu:.2e}, vs BF16 {worst_bf16:.2e}",
            t_bf16 / t_dot
        );
        assert!(worst_cpu < 1e-2, "{name}: dot off CPU by {worst_cpu:.3e}");
        assert!(
            worst_bf16 < 5e-2,
            "{name}: dot vs BF16 path by {worst_bf16:.3e}"
        );
        set_tile_table(Vec::new());
        gpu.destroy_buf(&w);
        gpu.destroy_buf(&x);
        gpu.destroy_buf(&o);
        gpu.destroy_buf(&o2);
    }
}

/// Dense-shape speed probe (`BRECCIA_DOT_BIG=1`): the tiled f16 dot vs the
/// quantised tile build on the model's largest dense projections.
#[test]
fn dot_gemm_dense_shape_speed() {
    let (Ok(selector), Ok(_)) = (
        std::env::var("BRECCIA_KERNEL_DEVICE"),
        std::env::var("BRECCIA_DOT_BIG"),
    ) else {
        eprintln!("dot_big: SKIPPED — set BRECCIA_KERNEL_DEVICE and BRECCIA_DOT_BIG");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        return;
    };
    if !DotGemm::supported(&gpu) {
        return;
    }
    let mut dots = vec![(
        "tile64x8".to_string(),
        DotGemm::new_tiled(&gpu).expect("tiled pipeline"),
    )];
    for (v, t) in DotGemm::MT_TILES.iter().enumerate() {
        dots.push((
            format!("mt{}x{}", t.0, t.1),
            DotGemm::new_mt(&gpu, v).expect("micro-tiled"),
        ));
    }
    let q = GemmMany::new(&gpu, QuantType::Iq3S).expect("iq3_s gemm");
    // No 2560x640: IQ3_S needs cols % 256.
    let grid: [(usize, usize, &[usize]); 3] = [
        (12288, 2560, &[256, 1024, 4096]),
        (2560, 4096, &[256, 1024, 4096]),
        (640, 2560, &[64, 128, 256, 512]),
    ];
    for (rows, cols, us) in grid {
        for &users in us {
            let w = alloc(&gpu, (rows * cols * 2) as u64).expect("w");
            let wq = alloc(&gpu, q.weight_bytes(rows, cols)).expect("wq");
            let x = alloc(&gpu, (users * cols * 4) as u64).expect("x");
            let o = alloc(&gpu, (users * rows * 4) as u64).expect("o");
            let gf = 2.0 * (rows * cols * users) as f64 / 1e9;
            let time = |f: &dyn Fn()| {
                (0..4)
                    .map(|_| {
                        let t0 = std::time::Instant::now();
                        f();
                        t0.elapsed().as_secs_f64() * 1e3
                    })
                    .skip(1)
                    .fold(f64::INFINITY, f64::min)
            };
            let tq = time(&|| q.run_at(&wq, &x, &o, rows, cols, users, 0).expect("q"));
            let mut line = format!(
                "dot_big: {rows}x{cols} u={users}: iq3_s {tq:.2} ms ({:.1} TF/s)",
                gf / tq
            );
            for (name, g) in &dots {
                let td = time(&|| g.run_tiled(&w, &x, &o, rows, cols, users).expect("dot"));
                line += &format!("  {name} {td:.2} ms ({:.1})", gf / td);
            }
            let tdq = time(&|| q.dequant_f16(&wq, &w, rows, cols).expect("deq"));
            let tdd = time(&|| {
                q.run_dot_dequant(&wq, &x, &o, rows, cols, users, &w)
                    .expect("dd")
            });
            line += &format!(
                "  deq16 {tdq:.2} ms  deq16+mt {tdd:.2} ms ({:.1})",
                gf / tdd
            );
            println!("{line}");
            for b in [&w, &wq, &x, &o] {
                gpu.destroy_buf(b);
            }
        }
    }
}

/// **Inside a dispatch batch** the dequant + dot route must record, not submit: a
/// submit+wait dot runs before the batch's recorded dequant has filled the scratch
/// (the engine's 1k prefill gave flat logits, top-1 3.9 vs 19.6, until it recorded).
/// The scratch starts zeroed, so a dot that runs early returns all zeros.
#[test]
fn the_dequant_dot_route_records_inside_a_dispatch_batch() {
    use breccia_kernels::dispatch_batch;
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("dot_batch: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        return;
    };
    if !DotGemm::supported(&gpu) {
        eprintln!("dot_batch: SKIPPED — no extension");
        return;
    }
    let (rows, cols, users) = (256usize, 512usize, 64usize);
    let q = GemmMany::new(&gpu, QuantType::Iq4Nl).expect("iq4_nl gemm");
    let mut seed = 0x0bad_5eedu32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 24) as u8
    };
    // IQ4_NL block: f16 scale (1.0 / 64) + 16 bytes of 4-bit codebook indices.
    let mut wq = Vec::new();
    for _ in 0..rows * cols / 32 {
        wq.extend_from_slice(&half(1.0 / 64.0).to_le_bytes());
        wq.extend((0..16).map(|_| next()));
    }
    let xs: Vec<u8> = (0..users * cols)
        .flat_map(|_| ((next() as f32 - 128.0) / 64.0).to_le_bytes())
        .collect();
    let wq = upload(&gpu, &wq).expect("wq");
    let x = upload(&gpu, &xs).expect("x");
    let o_ref = alloc(&gpu, (rows * users * 4) as u64).expect("o_ref");
    let o = alloc(&gpu, (rows * users * 4) as u64).expect("o");
    // Garbage-filled (0x5a5a = f16 203.25, finite), so a dequant that skips cells shows up.
    let scratch_ref = upload(&gpu, &vec![0x5au8; rows * cols * 2]).expect("scratch");
    q.run_dot_dequant(&wq, &x, &o_ref, rows, cols, users, &scratch_ref)
        .expect("reference");
    let want = breccia_kernels::ops::download_f32(&gpu, &o_ref, rows * users).expect("ref");
    assert!(want.iter().any(|v| *v != 0.0), "reference is all zeros");
    // The dequant must fill the whole scratch: dot route vs the plain quant GEMM.
    let o_plain = alloc(&gpu, (rows * users * 4) as u64).expect("o_plain");
    q.run(&wq, &x, &o_plain, rows, cols, users).expect("plain");
    let plain = breccia_kernels::ops::download_f32(&gpu, &o_plain, rows * users).expect("plain");
    let scale = plain.iter().fold(0f32, |m, v| m.max(v.abs()));
    let off = want.iter().zip(&plain).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("dot_batch: dot route vs plain worst |diff| {off:.3e} (scale {scale:.3e})");
    // `f32::max` drops NaN, so check finiteness separately.
    assert!(want.iter().all(|v| v.is_finite()), "dot route output has non-finite values");
    assert!(off <= 1e-2 * scale, "dot route vs plain GEMM: {off:.3e} of {scale:.3e}");
    gpu.destroy_buf(&o_plain);

    // A second matrix through the same scratch: its dequant must wait for the
    // first dot to finish reading (write-after-read on the shared scratch).
    let mut wq2 = Vec::new();
    for _ in 0..rows * cols / 32 {
        wq2.extend_from_slice(&half(1.0 / 32.0).to_le_bytes());
        wq2.extend((0..16).map(|_| next()));
    }
    let wq2 = upload(&gpu, &wq2).expect("wq2");
    q.run_dot_dequant(&wq2, &x, &o_ref, rows, cols, users, &scratch_ref)
        .expect("reference 2");
    let want2 = breccia_kernels::ops::download_f32(&gpu, &o_ref, rows * users).expect("ref2");
    let o2 = alloc(&gpu, (rows * users * 4) as u64).expect("o2");

    let scratch = upload(&gpu, &vec![0u8; rows * cols * 2]).expect("zeroed scratch");
    dispatch_batch::begin(&gpu).expect("begin");
    q.run_dot_dequant(&wq, &x, &o, rows, cols, users, &scratch)
        .expect("batched");
    q.run_dot_dequant(&wq2, &x, &o2, rows, cols, users, &scratch)
        .expect("batched 2");
    dispatch_batch::end(&gpu).expect("end");
    let got = breccia_kernels::ops::download_f32(&gpu, &o, rows * users).expect("got");
    let got2 = breccia_kernels::ops::download_f32(&gpu, &o2, rows * users).expect("got2");
    let worst = got
        .iter()
        .zip(&want)
        .chain(got2.iter().zip(&want2))
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("dot_batch: batched vs direct worst |diff| {worst:.3e}");
    assert!(
        worst == 0.0,
        "the batched dequant + dot differs: {worst:.3e}"
    );
    for b in [&wq, &wq2, &x, &o_ref, &o, &o2, &scratch_ref, &scratch] {
        gpu.destroy_buf(b);
    }
}

fn h2f(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as i32, (h & 1023) as f32);
    let v = if e == 0 { m * 2f32.powi(-24) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
    if s == 1 { -v } else { v }
}

/// Int8-dot micro-tiled GEMM (`I8DotGemm`, prefill-int8 experiment): exact vs a
/// CPU int reference on the GPU-quantised X, error vs fp32, then (with
/// `BRECCIA_DOT_BIG=1`) speed incl. the X quantise against the f16 mt builds.
#[test]
fn i8_dot_gemm_matches_and_speed() {
    use breccia_kernels::gemm::I8DotGemm;
    use breccia_kernels::gemv::download;
    use breccia_kernels::ops::download_f32;
    use breccia_kernels::Q8Quantize;
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("i8dot: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        return;
    };
    let quant = Q8Quantize::new(&gpu).expect("q8 quantise");
    let i8s: Vec<I8DotGemm> = (0..4).map(|v| I8DotGemm::new_mt(&gpu, v).expect("i8 mt")).collect();
    let mut seed = 0x1234_5678u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };

    // Correctness: rows/users not tile multiples.
    let (rows, cols, users) = (200usize, 512usize, 70usize);
    let wf: Vec<f32> = (0..rows * cols).map(|_| rnd()).collect();
    let xf: Vec<f32> = (0..users * cols).map(|_| rnd()).collect();
    let (mut wq, mut wd) = (vec![0i8; rows * cols], vec![0u16; rows * cols / 32]);
    for g in 0..rows * cols / 32 {
        let s = &wf[g * 32..g * 32 + 32];
        let amax = s.iter().fold(0f32, |a, v| a.max(v.abs()));
        wd[g] = half(amax / 127.0);
        let d = h2f(wd[g]);
        for (i, v) in s.iter().enumerate() {
            wq[g * 32 + i] = (v / d).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let w_b = upload(&gpu, &wq.iter().map(|&v| v as u8).collect::<Vec<u8>>()).expect("w");
    let wd_b = upload(&gpu, &wd.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>()).expect("wd");
    let x_b = upload(&gpu, &bytes(&xf)).expect("x");
    let xq_b = alloc(&gpu, (users * cols) as u64).expect("xq");
    let xd_b = alloc(&gpu, (users * cols / 32 * 2) as u64).expect("xd");
    let o_b = alloc(&gpu, (users * rows * 4) as u64).expect("o");
    quant.run(&x_b, &xq_b, &xd_b, users * cols).expect("quantise");
    let xq: Vec<i8> = download(&gpu, &xq_b, users * cols).expect("xq").into_iter().map(|b| b as i8).collect();
    let xd: Vec<u16> = download(&gpu, &xd_b, users * cols / 32 * 2)
        .expect("xd")
        .chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let (mut want, mut exact) = (vec![0f32; users * rows], vec![0f32; users * rows]);
    for u in 0..users {
        for r in 0..rows {
            let mut acc = 0f64;
            let mut f = 0f64;
            for g in 0..cols / 32 {
                let mut ia = 0i32;
                for i in 0..32 {
                    let k = g * 32 + i;
                    ia += wq[r * cols + k] as i32 * xq[u * cols + k] as i32;
                    f += wf[r * cols + k] as f64 * xf[u * cols + k] as f64;
                }
                acc += ia as f64 * h2f(wd[r * cols / 32 + g]) as f64 * h2f(xd[u * cols / 32 + g]) as f64;
            }
            want[u * rows + r] = acc as f32;
            exact[u * rows + r] = f as f32;
        }
    }
    let scale = exact.iter().fold(0f32, |a, v| a.max(v.abs()));
    for (v, g) in i8s.iter().enumerate() {
        g.run(&w_b, &wd_b, &xq_b, &xd_b, &o_b, rows, cols, users).expect("run");
        let got = download_f32(&gpu, &o_b, users * rows).expect("o");
        let int_err = got.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / scale;
        let q_err = got.iter().zip(&exact).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / scale;
        println!("i8dot: variant {v}: |err vs int ref| {int_err:.2e}, vs fp32 {q_err:.2e} (max-normalised)");
        assert!(int_err < 1e-5, "variant {v}: off the int reference by {int_err:.3e}");
    }
    for b in [&w_b, &wd_b, &x_b, &xq_b, &xd_b, &o_b] {
        gpu.destroy_buf(b);
    }

    if std::env::var("BRECCIA_DOT_BIG").is_err() || !DotGemm::supported(&gpu) {
        return;
    }
    let f16s: Vec<(String, DotGemm)> = DotGemm::MT_TILES
        .iter()
        .enumerate()
        .map(|(v, t)| (format!("mt{}x{}", t.0, t.1), DotGemm::new_mt(&gpu, v).expect("f16 mt")))
        .collect();
    let grid: [(usize, usize, &[usize]); 3] = [
        (12288, 2560, &[256, 1024, 4096]),
        (2560, 4096, &[256, 1024, 2048]), // 4096x4096 exceeds one q8-quantise dispatch
        (640, 2560, &[64, 128, 256, 512]),
    ];
    for (rows, cols, us) in grid {
        for &users in us {
            let w = alloc(&gpu, (rows * cols * 2) as u64).expect("w");
            let w8 = alloc(&gpu, (rows * cols) as u64).expect("w8");
            let wd8 = alloc(&gpu, (rows * cols / 16) as u64).expect("wd8");
            let x = alloc(&gpu, (users * cols * 4) as u64).expect("x");
            let xq = alloc(&gpu, (users * cols) as u64).expect("xq");
            let xd = alloc(&gpu, (users * cols / 16) as u64).expect("xd");
            let o = alloc(&gpu, (users * rows * 4) as u64).expect("o");
            let gf = 2.0 * (rows * cols * users) as f64 / 1e9;
            let time = |f: &dyn Fn()| {
                (0..5)
                    .map(|_| {
                        let t0 = std::time::Instant::now();
                        f();
                        t0.elapsed().as_secs_f64() * 1e3
                    })
                    .skip(1)
                    .fold(f64::INFINITY, f64::min)
            };
            let tq = time(&|| quant.run(&x, &xq, &xd, users * cols).expect("q"));
            let mut line = format!("i8big: {rows}x{cols} u={users}: quant {tq:.2} ms |");
            let mut best16 = f64::INFINITY;
            for (name, g) in &f16s {
                let t = time(&|| g.run_tiled(&w, &x, &o, rows, cols, users).expect("f16"));
                best16 = best16.min(t);
                line += &format!(" f16 {name} {t:.2} ({:.1})", gf / t);
            }
            line += " |";
            let mut best8 = f64::INFINITY;
            for (v, g) in i8s.iter().enumerate() {
                let t = time(&|| g.run(&w8, &wd8, &xq, &xd, &o, rows, cols, users).expect("i8"));
                best8 = best8.min(t);
                let tt = DotGemm::MT_TILES[v];
                line += &format!(" i8 mt{}x{} {t:.2} ({:.1})", tt.0, tt.1, gf / t);
            }
            line += &format!(
                " | best f16 {best16:.2} vs i8+quant {:.2} ms = {:.2}x",
                best8 + tq,
                best16 / (best8 + tq)
            );
            println!("{line}");
            for b in [&w, &w8, &wd8, &x, &xq, &xd, &o] {
                gpu.destroy_buf(b);
            }
        }
    }
}
