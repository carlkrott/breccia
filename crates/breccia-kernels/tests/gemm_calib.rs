//! GEMM calibration harness: every quant type the multi-user GEMM supports, at the
//! two shapes that matter, per device, per build.
//!
//! Why: `tuned()` (`gemm.rs:132`) is a compiled-in table whose doc says
//! "wave64 = Vega20 (MI50), wave32 = RDNA2 (RX)" — but RADV reports subgroup size
//! **64** on the RX, so the RX has always taken the Vega20 row and `ldsx` has
//! always been on (a verdict measured on the MI50 only). This harness produces the
//! numbers that table should contain, without needing a model file: the GEMM's cost
//! depends on the shape and the type, not on the weight values, so the weights are
//! synthetic bytes of the right size and `spirv-val`/parity is irrelevant here.
//!
//! Run it under each build condition and compare:
//!
//! ```text
//! BRECCIA_KERNEL_DEVICE=0000:03:00.0 cargo test --release -p breccia-kernels \
//!     --test gemm_calib -- --nocapture                  # natural width, ldsx auto
//! BRECCIA_SUBGROUP=32 ...                                 # forced wave32
//! BRECCIA_GEMM_LDSX=0 ... / BRECCIA_GEMM_LDSX=1 ...       # LDS-staged X on/off
//! ```
//!
//! Output is one CSV-ish line per (type, m), so the runs can be diffed:
//! `type m avg_us gbps variant_rpl variant_umax`.

use breccia_kernels::gemm::GemmMany;
use breccia_kernels::gemv::alloc;
use breccia_kernels::QuantType;
use breccia_vk::ctx::{Gpu, Host};

/// Shapes that matter: expert gate/up (640x2560) and down (2560x640). Prefill runs
/// m at the chunk size, decode at 1-8; the body's cost tracks `m` for weight reads.
const SHAPES: [(usize, usize, &str); 2] = [(640, 2560, "gate"), (2560, 640, "down")];
const MS: [usize; 3] = [8, 128, 1024];

const TYPES: [QuantType; 12] = [
    QuantType::Iq2Xxs,
    QuantType::Iq2Xs,
    QuantType::Iq2S,
    QuantType::Iq3Xxs,
    QuantType::Iq3S,
    QuantType::Iq4Nl,
    QuantType::Iq4Xs,
    QuantType::Q2_0,
    QuantType::Q4K,
    QuantType::Q5K,
    QuantType::Q6K,
    QuantType::Bf16,
];

#[test]
fn gemm_calibration() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("gemm_calib: SKIPPED — set BRECCIA_KERNEL_DEVICE=<pci|index>");
        return;
    };
    // The Host owns the instance: it must outlive the device, or the device's
    // instance handle dangles (this crashed with SIGSEGV the first time).
    let host = Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let idx = devs
        .iter()
        .position(|d| d.pci_address == selector.trim() || d.index.to_string() == selector.trim())
        .unwrap_or(0);
    let gpu = Gpu::create(&host, &devs[idx], 1).expect("device");
    let cond = format!(
        "wg={} ldsx={}",
        match gpu.forced_subgroup() {
            Some(n) => n.to_string(),
            None => "natural".to_string(),
        },
        std::env::var("BRECCIA_GEMM_LDSX").unwrap_or_else(|_| "auto".into())
    );
    println!("# device {selector} {cond}");
    println!("type,m,shape,avg_us,variant_rpl,variant_umax");

    for (rows, cols, shape) in SHAPES {
        let only = std::env::var("BRECCIA_CALIB_TYPE").unwrap_or_default();
        for ty in TYPES {
            if !only.is_empty() && !ty.name().to_lowercase().contains(&only.to_lowercase()) {
                continue;
            }
            eprintln!("calib: {} {} {}x{} ...", ty.name(), shape, rows, cols);
            let mut gemm = match GemmMany::new(&gpu, ty) {
                Ok(g) => g,
                Err(e) => {
                    println!("{},{},{},SKIP({})", ty.name(), 0, shape, e);
                    continue;
                }
            };
            // The table's variant for this (type, width, ldsx) is what `new` already
            // instal`led; report it so the numbers are attributable.
            eprintln!("  new ok");
            let v = gemm.variant_for(rows, MS[MS.len() - 1]);
            eprintln!("  variant {:?}", v);
            let wbytes = gemm.weight_bytes(rows, cols);
            let w = match alloc(&gpu, wbytes) {
                Ok(b) => b,
                Err(e) => {
                    println!("{},{},{},ALLOC({})", ty.name(), 0, shape, e);
                    continue;
                }
            };
            let x = alloc(&gpu, (MS[MS.len() - 1] * cols * 4) as u64).expect("x");
            let out = alloc(&gpu, (MS[MS.len() - 1] * rows * 4) as u64).expect("out");
            // Zero-fill: uninitialised device memory can hold denormals/NaN patterns
            // that some i-quant paths do not expect, and the timing does not depend
            // on the values (the cost is the same weight decode either way).
            let zero_w = vec![0u8; wbytes as usize];
            let zero_x = vec![0u8; MS[MS.len() - 1] * cols * 4];
            eprintln!("  alloc ok w={wbytes}");
            breccia_kernels::gemv::upload_into(&gpu, &w, &zero_w).expect("zero w");
            breccia_kernels::gemv::upload_into(&gpu, &x, &zero_x).expect("zero x");
            for m in MS {
                // best of 5, after a warm run
                let mut best = f64::INFINITY;
                for pass in 0..6 {
                    let t = std::time::Instant::now();
                    // `run_at` submits and waits (the crate's synchronous contract),
                    // so the host clock brackets the kernel.
                    if gemm.run_at(&w, &x, &out, rows, cols, m, 0).is_err() {
                        break;
                    }
                    let dt = t.elapsed().as_secs_f64() * 1e6;
                    if pass > 0 {
                        best = best.min(dt);
                    }
                }
                let gbps = (wbytes as f64) / (best * 1e-6) / 1e9;
                println!("{},{},{},{:.1},{},{}", ty.name(), m, shape, best, v.0, v.1);
                eprintln!(
                    "  {:>7} m={:<5} {:<5} {:8.1} us  weight {:6.1} GB/s",
                    ty.name(),
                    m,
                    shape,
                    best,
                    gbps
                );
            }
            gpu.destroy_buf(&w);
            gpu.destroy_buf(&x);
            gpu.destroy_buf(&out);
            let _ = &mut gemm;
        }
    }
}

/// Variant sweep: the blocked builds a type can actually use are (1,8), (2,8),
/// (4,8), (1,16), (2,16) plus the plain fallback, so the whole tuning space is
/// small enough to measure exhaustively. Reports the best variant per (type, m)
/// for the current device and build condition, which is what `tuned()` should
/// contain instead of its hand-written rows.
///
/// `BRECCIA_CALIB_VARIANTS=1` selects it.
#[test]
fn gemm_variant_sweep() {
    if std::env::var("BRECCIA_CALIB_VARIANTS").is_err() {
        eprintln!("gemm_variants: SKIPPED — set BRECCIA_CALIB_VARIANTS=1");
        return;
    }
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("gemm_variants: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let host = Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let idx = devs
        .iter()
        .position(|d| d.pci_address == selector.trim() || d.index.to_string() == selector.trim())
        .unwrap_or(0);
    let gpu = Gpu::create(&host, &devs[idx], 1).expect("device");
    let wg = match gpu.forced_subgroup() {
        Some(n) => n.to_string(),
        None => format!("{}", devs[idx].subgroup_size),
    };
    println!("# variants device {selector} wg={wg}");
    println!("type,rows,shape,m,variant,rpl,umax,us");

    const VARIANTS: [(usize, usize); 5] = [(1, 8), (2, 8), (4, 8), (1, 16), (2, 16)];
    let ms = [8usize, 32, 128, 1024];
    for (rows, cols, shape) in SHAPES {
        for ty in TYPES {
            let Ok(mut gemm) = GemmMany::new(&gpu, ty) else {
                continue;
            };
            let wbytes = gemm.weight_bytes(rows, cols);
            let (Ok(w), Ok(x), Ok(out)) = (
                alloc(&gpu, wbytes),
                alloc(&gpu, (ms[ms.len() - 1] * cols * 4) as u64),
                alloc(&gpu, (ms[ms.len() - 1] * rows * 4) as u64),
            ) else {
                continue;
            };
            breccia_kernels::gemv::upload_into(&gpu, &w, &vec![0u8; wbytes as usize]).expect("w");
            breccia_kernels::gemv::upload_into(&gpu, &x, &vec![0u8; ms[ms.len() - 1] * cols * 4])
                .expect("x");
            for m in ms {
                let mut best = (f64::INFINITY, -1i32, 0usize, 0usize);
                for (rpl, umax) in VARIANTS {
                    gemm.set_variant(rpl, umax);
                    let mut t_best = f64::INFINITY;
                    for pass in 0..4 {
                        let t = std::time::Instant::now();
                        if gemm.run_at(&w, &x, &out, rows, cols, m, 0).is_err() {
                            break;
                        }
                        let dt = t.elapsed().as_secs_f64() * 1e6;
                        if pass > 0 {
                            t_best = t_best.min(dt);
                        }
                    }
                    if t_best < best.0 {
                        // what the dispatcher actually used (may have fallen back)
                        let v = gemm.variant_for(rows, m);
                        best = (t_best, 0, v.0, v.1);
                    }
                }
                println!(
                    "{},{},{},{},{},{},{},{:.1}",
                    ty.name(),
                    rows,
                    shape,
                    m,
                    if best.2 == 1 && best.3 == 8 {
                        "plain"
                    } else {
                        "blocked"
                    },
                    best.2,
                    best.3,
                    best.0
                );
                eprintln!(
                    "  {:>7} {:<5} m={:<5} best {:.1} us  variant ({},{})",
                    ty.name(),
                    shape,
                    m,
                    best.0,
                    best.2,
                    best.3
                );
            }
            gpu.destroy_buf(&w);
            gpu.destroy_buf(&x);
            gpu.destroy_buf(&out);
        }
    }
}

/// Tile sweep: which of the four tiled builds wins for each (shape, users) cell.
/// This is the matrix `tile_default()` should be a function of — today it returns
/// (4,4) for every type and shape, and the measured optimum moves with both.
///
/// `BRECCIA_CALIB_TILES=1` selects it.
#[test]
fn gemm_tile_sweep() {
    if std::env::var("BRECCIA_CALIB_TILES").is_err() {
        eprintln!("gemm_tiles: SKIPPED — set BRECCIA_CALIB_TILES=1");
        return;
    }
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("gemm_tiles: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let host = Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let idx = devs
        .iter()
        .position(|d| d.pci_address == selector.trim() || d.index.to_string() == selector.trim())
        .unwrap_or(0);
    let gpu = Gpu::create(&host, &devs[idx], 1).expect("device");
    println!("# tiles device {selector}");
    println!("type,rows,shape,users,tile,us");

    const TLI: [breccia_kernels::gemm::Tile; 4] = [(4, 2), (4, 4), (8, 4), (8, 8)];
    let users_list = [64usize, 96, 128, 192, 256, 512, 1024];
    // One quantised type and BF16: the tile optimum is a launch-geometry property,
    // and these two cover the fastest and slowest inner loops.
    for ty in [QuantType::Iq3S, QuantType::Bf16] {
        eprintln!("tiles: type {} ...", ty.name());
        for (rows, cols, shape) in SHAPES {
            let mut gemm = match GemmMany::new(&gpu, ty) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("tiles: {} new failed: {e}", ty.name());
                    continue;
                }
            };
            let wbytes = gemm.weight_bytes(rows, cols);
            let max_users = users_list[users_list.len() - 1];
            let (Ok(w), Ok(x), Ok(out)) = (
                alloc(&gpu, wbytes),
                alloc(&gpu, (max_users * cols * 4) as u64),
                alloc(&gpu, (max_users * rows * 4) as u64),
            ) else {
                continue;
            };
            breccia_kernels::gemv::upload_into(&gpu, &w, &vec![0u8; wbytes as usize]).expect("w");
            breccia_kernels::gemv::upload_into(&gpu, &x, &vec![0u8; max_users * cols * 4])
                .expect("x");
            for users in users_list {
                for t in TLI {
                    // Public path: `run_at` picks the tiled build for users >= 64,
                    // which is exactly what production does.
                    gemm.set_tile(Some(t));
                    let mut best = f64::INFINITY;
                    for pass in 0..4 {
                        let t0 = std::time::Instant::now();
                        if gemm.run_at(&w, &x, &out, rows, cols, users, 0).is_err() {
                            break;
                        }
                        let dt = t0.elapsed().as_secs_f64() * 1e6;
                        if pass > 0 {
                            best = best.min(dt);
                        }
                    }
                    println!(
                        "{},{},{},{},{}x{},{:.1}",
                        ty.name(),
                        rows,
                        shape,
                        users,
                        t.0,
                        t.1,
                        best
                    );
                    eprintln!(
                        "  {:>6} {:<5} users={:<5} tile {}x{} {:8.1} us",
                        ty.name(),
                        shape,
                        users,
                        t.0,
                        t.1,
                        best
                    );
                }
            }
            gemm.set_tile(None);
            gpu.destroy_buf(&w);
            gpu.destroy_buf(&x);
            gpu.destroy_buf(&out);
        }
    }
}
