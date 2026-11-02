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
