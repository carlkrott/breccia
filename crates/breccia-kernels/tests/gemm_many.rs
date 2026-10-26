//! Conformance + throughput for the **multi-user** fused dequant + mat-vector
//! (`GemmMany`, `docs/phase-4/01-forward-pass.md` §8v).
//!
//! | check | oracle |
//! |---|---|
//! | `m` users in one dispatch vs `m` single-user GEMV dispatches on the same expert | the existing kernel, per user |
//! | `m = 1` | **bit-identical** to the single-user GEMV (same decodes, same accumulation order, same reduction) |
//! | throughput at `m = 1, 4, 8` | weights read x m / GPU time, GB/s and GFLOP/s |
//!
//! GPU tests are opt-in by device (`BRECCIA_KERNEL_DEVICE`, AGENTS.md). The MGEMV
//! path is new shader code: run it on the MI50 first.

use std::sync::Arc;
use std::sync::Mutex;

use ash::vk;
use breccia_format::{ExpertRole, ShardSet};
use breccia_kernels::gemm::{GemmMany, USERS_MAX};
use breccia_kernels::gemv::{alloc, Gemv};
use breccia_kernels::{cpu::LayerPlan, QuantType};
use breccia_vk::ctx::{Gpu, Host};

const MODEL_GLOB: &str = "/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf";
static LOCK: Mutex<()> = Mutex::new(());

struct Dev {
    gpu: Gpu,
    _host: Host,
}

fn open(selector: &str) -> Option<Dev> {
    let host = Host::new().ok()?;
    let devs = host.devices().ok()?;
    let want = selector.trim();
    let idx = devs
        .iter()
        .position(|d| d.pci_address == want || d.index.to_string() == want)
        .unwrap_or(0);
    let gpu = Gpu::create(&host, &devs[idx], 1).ok()?;
    Some(Dev { gpu, _host: host })
}

#[test]
fn multi_user_gemm_matches_the_single_user_gemv() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!(
            "gemm_many: SKIPPED - set BRECCIA_KERNEL_DEVICE=<pci|index> \
             (0000:0e:00.0 = MI50 first)"
        );
        return;
    };
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = open(&selector) else {
        eprintln!("gemm_many: no device for {selector}");
        return;
    };
    let set = match ShardSet::open_glob(MODEL_GLOB) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("gemm_many: skipping, no model: {e}");
            return;
        }
    };
    let layout = set.layout().expect("layout");
    // Every (block, role) whose type has a multi-user kernel: the delivered set
    // covers `down` (IQ4_NL, Q2_0) and `gate`/`up` (IQ2_*, IQ3_*).
    let mut cells: Vec<(usize, ExpertRole, QuantType)> = Vec::new();
    for block in layout.blocks() {
        let plan = LayerPlan::from_block(&set, block).expect("plan");
        for role in [ExpertRole::Down, ExpertRole::Gate] {
            let ty = plan.role(role.name()).ty;
            if matches!(
                ty,
                QuantType::Iq4Nl
                    | QuantType::Q2_0
                    | QuantType::Iq2Xxs
                    | QuantType::Iq2Xs
                    | QuantType::Iq2S
                    | QuantType::Iq3Xxs
                    | QuantType::Iq3S
            ) && !cells.iter().any(|c| c.1 == role && c.2 == ty)
            {
                cells.push((plan.block, role, ty));
            }
        }
        if cells.len() >= 6 {
            break;
        }
    }
    if cells.is_empty() {
        eprintln!("gemm_many: no supported role/type in this artifact");
        return;
    }
    let gpu = &dev.gpu;
    for (block_no, role, ty) in cells {
        let plan =
            LayerPlan::from_block(&set, layout.block(block_no).expect("block")).expect("plan");
        let (rows, cols) = if role == ExpertRole::Down {
            (plan.n_embd, plan.n_ff)
        } else {
            (plan.n_ff, plan.n_embd)
        };
        let gemm = GemmMany::new(gpu, ty).expect("gemm pipeline");
        let gemv = Gemv::new(gpu, ty, rows, cols).expect("gemv pipeline");
        let expert = 7usize;
        let slices = plan.slices(&set, expert).expect("slices");
        let down = match role {
            ExpertRole::Down => slices.down,
            ExpertRole::Gate => slices.gate,
            ExpertRole::Up => slices.up,
        };
        let pad = gemm.padded_cols(cols);
        let w = breccia_kernels::gemv::upload(gpu, down).expect("upload weights");
        let mut seed = 0x1234_5678u32;
        let mut xs: Vec<f32> = Vec::with_capacity(USERS_MAX * pad);
        for _ in 0..USERS_MAX * pad {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            xs.push(((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5);
        }
        let x = breccia_kernels::ops::upload_f32(gpu, &xs).expect("upload x");
        let out_gemm = alloc(gpu, (USERS_MAX * rows * 4) as u64).expect("out buffer");
        let out_ref = alloc(gpu, (rows * 4) as u64).expect("ref buffer");

        let mut worst: f64 = 0.0;
        let mut same_bits_all = true;
        for m in [1usize, 2, 4, 8] {
            gemm.run(&w, &x, &out_gemm, rows, cols, m).expect("gemm");
            let got =
                breccia_kernels::ops::download_f32(gpu, &out_gemm, m * rows).expect("download");
            for u in 0..m {
                let xu = breccia_kernels::ops::upload_f32(gpu, &xs[u * pad..(u + 1) * pad])
                    .expect("upload x_u");
                gemv.run(&w, &xu, &out_ref).expect("gemv");
                let want =
                    breccia_kernels::ops::download_f32(gpu, &out_ref, rows).expect("download");
                for r in 0..rows {
                    let a = got[u * rows + r];
                    let b = want[r];
                    if a.to_bits() != b.to_bits() {
                        same_bits_all = false;
                    }
                    worst = worst.max(((a - b).abs() as f64) / (b.abs() as f64).max(1.0));
                }
                gpu.destroy_buf(&xu);
            }
            let t = gemm
                .run_timed(&w, &x, &out_gemm, rows, cols, m, 0.2)
                .expect("timed");
            let bytes = gemm.weight_bytes(rows, cols) as f64;
            let flops = 2.0 * m as f64 * rows as f64 * cols as f64;
            println!(
                "{} {} m = {m}: {:.3} ms/dispatch, weights {:.1} GB/s, {:.0} GFLOP/s",
                ty.name(),
                role.name(),
                t.per_iter_s * 1e3,
                bytes / t.per_iter_s / 1e9,
                flops / t.per_iter_s / 1e9,
            );
        }
        println!(
            "{} {}: worst relative {worst:.3e}, bit-identical for every m: {same_bits_all}",
            ty.name(),
            role.name()
        );
        assert!(
            worst <= 1e-6,
            "{} {}: multi-user GEMM differs by {worst:.3e}",
            ty.name(),
            role.name()
        );
        assert!(
            same_bits_all,
            "{} {}: m = 1 must be bit-identical",
            ty.name(),
            role.name()
        );
        gpu.destroy_buf(&out_ref);
        gpu.destroy_buf(&out_gemm);
        gpu.destroy_buf(&x);
        gpu.destroy_buf(&w);
    }
    drop(guard);
    let _ = vk::WHOLE_SIZE;
}
