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
    // covers `down` (IQ4_NL, Q2_0) and `gate`/`up` (IQ2_*, IQ3_*, and since §8aa
    // also IQ4_XS, Q4_K, Q5_K, Q6_K — the shared expert's types).
    // AGENTS.md: one quant type per invocation. `BRECCIA_GEMM_TYPE` narrows the
    // sweep further; the default walks only a few cells because the 2026-10-26
    // reboot correlated with many new dispatch shapes in a single process.
    let only: Option<QuantType> = std::env::var("BRECCIA_GEMM_TYPE").ok().and_then(|s| {
        match s.trim().to_ascii_uppercase().as_str() {
            "IQ4_NL" => Some(QuantType::Iq4Nl),
            "Q2_0" => Some(QuantType::Q2_0),
            "IQ2_XXS" => Some(QuantType::Iq2Xxs),
            "IQ2_XS" => Some(QuantType::Iq2Xs),
            "IQ2_S" => Some(QuantType::Iq2S),
            "IQ3_XXS" => Some(QuantType::Iq3Xxs),
            "IQ3_S" => Some(QuantType::Iq3S),
            _ => None,
        }
    });
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
                    | QuantType::Iq4Xs
                    | QuantType::Q4K
                    | QuantType::Q5K
                    | QuantType::Q6K
            ) && only.is_none_or(|o| o == ty)
                && !cells.iter().any(|c| c.1 == role && c.2 == ty)
            {
                cells.push((plan.block, role, ty));
            }
        }
        if cells.len() >= 4 {
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

/// The chunked-prefill fan-in against `MoeCombine` (single token, must match) and
/// against a CPU reference for a multi-token chunk.
#[test]
fn fan_in_matches_moe_combine_and_the_cpu_reference() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("fan_in: SKIPPED - set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = open(&selector) else {
        eprintln!("fan_in: no device for {selector}");
        return;
    };
    let gpu = &dev.gpu;
    let fan = breccia_kernels::MoeFanIn::new(gpu).expect("fan-in pipeline");
    let combine = breccia_kernels::MoeCombine::new(gpu).expect("combine pipeline");
    let (k, cols) = (4usize, 256usize);

    let mut seed = 0x2468_ace0u32;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };

    // ---- one token: the fan-in must equal MoeCombine ----
    let down = rnd(k * cols); // [k, cols], the same layout MoeCombine takes
    let wts = rnd(k);
    let shared = rnd(cols);
    let gate = rnd(1);
    let slots: Vec<u32> = (0..k as u32).collect();
    let down_b = breccia_kernels::ops::upload_f32(gpu, &down).unwrap();
    let w_b = breccia_kernels::ops::upload_f32(gpu, &wts).unwrap();
    let sh_b = breccia_kernels::ops::upload_f32(gpu, &shared).unwrap();
    let g_b = breccia_kernels::ops::upload_f32(gpu, &gate).unwrap();
    let s_b =
        breccia_kernels::gemv::upload(gpu, &breccia_kernels::ops::u32_to_bytes(&slots)).unwrap();
    let out_fan = breccia_kernels::gemv::alloc(gpu, (cols * 4) as u64).unwrap();
    let out_cmb = breccia_kernels::gemv::alloc(gpu, (cols * 4) as u64).unwrap();
    fan.run(&down_b, &s_b, &w_b, &sh_b, &g_b, &out_fan, 1, k, cols)
        .expect("fan-in");
    combine
        .run(&down_b, &w_b, &sh_b, &g_b, &out_cmb, k, cols)
        .expect("combine");
    let a = breccia_kernels::ops::download_f32(gpu, &out_fan, cols).unwrap();
    let b = breccia_kernels::ops::download_f32(gpu, &out_cmb, cols).unwrap();
    let mut worst = 0f64;
    let mut same = true;
    for i in 0..cols {
        if a[i].to_bits() != b[i].to_bits() {
            same = false;
        }
        worst = worst.max(((a[i] - b[i]).abs() as f64) / (b[i].abs() as f64).max(1.0));
    }
    println!("fan-in vs MoeCombine (1 token): worst {worst:.3e}, bit-identical {same}");

    // ---- a 3-token chunk with overlapping slots: CPU reference ----
    let tokens = 3usize;
    let slots_t: Vec<u32> = vec![0, 2, 5, 7, 1, 3, 4, 6, 2, 3, 5, 8];
    let n_slots = 9usize;
    let scratch = rnd(n_slots * cols);
    let wts_t = rnd(tokens * k);
    let shared_t = rnd(tokens * cols);
    let gate_t = rnd(tokens);
    let sc_b = breccia_kernels::ops::upload_f32(gpu, &scratch).unwrap();
    let sl_b =
        breccia_kernels::gemv::upload(gpu, &breccia_kernels::ops::u32_to_bytes(&slots_t)).unwrap();
    let wt_b = breccia_kernels::ops::upload_f32(gpu, &wts_t).unwrap();
    let sht_b = breccia_kernels::ops::upload_f32(gpu, &shared_t).unwrap();
    let gt_b = breccia_kernels::ops::upload_f32(gpu, &gate_t).unwrap();
    let out_t = breccia_kernels::gemv::alloc(gpu, (tokens * cols * 4) as u64).unwrap();
    fan.run(&sc_b, &sl_b, &wt_b, &sht_b, &gt_b, &out_t, tokens, k, cols)
        .expect("fan-in chunk");
    let got = breccia_kernels::ops::download_f32(gpu, &out_t, tokens * cols).unwrap();
    let mut worst_t = 0f64;
    for t in 0..tokens {
        for c in 0..cols {
            let mut want = 0f64;
            for j in 0..k {
                let slot = slots_t[t * k + j] as usize;
                want += f64::from(wts_t[t * k + j]) * f64::from(scratch[slot * cols + c]);
            }
            let sig = 1.0 / (1.0 + (-f64::from(gate_t[t])).exp());
            want += f64::from(shared_t[t * cols + c]) * sig;
            let gotv = f64::from(got[t * cols + c]);
            worst_t = worst_t.max((gotv - want).abs() / want.abs().max(1.0));
        }
    }
    println!("fan-in vs f64 reference (3 tokens, 12 pairs, 9 slots): worst {worst_t:.3e}");
    assert!(
        worst <= 1e-6,
        "fan-in differs from MoeCombine by {worst:.3e}"
    );
    assert!(worst_t <= 1e-5, "fan-in differs from f64 by {worst_t:.3e}");
    drop(guard);
}

/// **Dense** (shared-expert) conformance for the multi-user GEMMs, one type per
/// invocation.
///
/// The routed-expert test above walks `LayerPlan`s, and this artifact's routed
/// experts never use `IQ4_XS`/`Q4_K`/`Q5_K`/`Q6_K` — those four appear on the
/// **shared** expert (`ffn_{gate,up,down}_shexp`), which has no plan. So the four
/// kernels §8aa added need this fixture: take the real dense tensor, run the
/// multi-user GEMM for `m = USERS_MAX`, and compare every user against a
/// single-user `Gemv` on the same weights.
///
/// `BRECCIA_GEMM_TYPE` selects exactly one type (AGENTS.md: one quant type per
/// invocation, so a fault is attributable). On 2026-10-26 a run of these four
/// shaders on the **display** GPU (the MI50) hard-reset the machine; the RX is the
/// headless card and goes first.
#[test]
fn dense_shared_expert_gemm_matches_the_gemv() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("dense_gemm: SKIPPED - set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Ok(want) = std::env::var("BRECCIA_GEMM_TYPE") else {
        eprintln!("dense_gemm: SKIPPED - set BRECCIA_GEMM_TYPE=IQ4_XS|Q4_K|Q5_K|Q6_K|BF16");
        return;
    };
    let ty = match want.trim().to_ascii_uppercase().as_str() {
        "IQ4_XS" => QuantType::Iq4Xs,
        "Q4_K" => QuantType::Q4K,
        "Q5_K" => QuantType::Q5K,
        "Q6_K" => QuantType::Q6K,
        "BF16" => QuantType::Bf16,
        other => {
            eprintln!("dense_gemm: {other} is not a dense GEMM type");
            return;
        }
    };
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = open(&selector) else {
        eprintln!("dense_gemm: no device for {selector}");
        return;
    };
    let set = match ShardSet::open_glob(MODEL_GLOB) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("dense_gemm: no model: {e}");
            return;
        }
    };
    // Find the first shared-expert tensor of the requested type.
    let mut found: Option<(String, usize, usize)> = None;
    'outer: for b in 0..48 {
        // BF16 lives in the QSA indexer projections, not the shared expert.
        // `BRECCIA_GEMM_TENSOR` picks one role instead (the router shapes).
        let names = match std::env::var("BRECCIA_GEMM_TENSOR") {
            Ok(n) => vec![n],
            Err(_) => ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "indexer.q_proj"]
                .map(String::from)
                .to_vec(),
        };
        for name in &names {
            let name = format!("blk.{b}.{name}.weight");
            let Some(t) = set.tensor(&name) else { continue };
            if QuantType::from_ggml(t.ty()) != Some(ty) {
                continue;
            }
            found = Some((name, t.ne1() as usize, t.ne0() as usize));
            break 'outer;
        }
    }
    let Some((name, rows, cols)) = found else {
        eprintln!("dense_gemm: this artifact has no {want} shared-expert tensor");
        return;
    };
    println!("dense_gemm: {name} [{rows} x {cols}] {}", ty.name());
    let t = set.tensor(&name).expect("tensor");
    let gpu = &dev.gpu;
    let w = breccia_kernels::gemv::upload(gpu, t.as_slice()).expect("upload");
    let m = USERS_MAX;
    let xs: Vec<Vec<f32>> = (0..m)
        .map(|u| {
            (0..cols)
                .map(|i| (((i * 7 + u * 13) % 29) as f32 - 14.0) * 0.011)
                .collect()
        })
        .collect();
    let flat: Vec<f32> = xs.iter().flatten().copied().collect();
    let x = breccia_kernels::ops::upload_f32(gpu, &flat).expect("x");
    let out_many = alloc(gpu, (m * rows * 4) as u64).expect("out_many");
    let many = GemmMany::new(gpu, ty).expect("gemm pipeline");
    many.run(&w, &x, &out_many, rows, cols, m).expect("gemm");
    let got = breccia_kernels::ops::download_f32(gpu, &out_many, m * rows).expect("down");
    let single = Gemv::new(gpu, ty, rows, cols).expect("gemv");
    let out_ref = alloc(gpu, (rows * 4) as u64).expect("out_ref");
    let mut worst = 0f64;
    let mut same_m1 = true;
    for u in 0..m {
        let xu = breccia_kernels::ops::upload_f32(gpu, &xs[u]).expect("xu");
        single.run(&w, &xu, &out_ref).expect("gemv run");
        let want = breccia_kernels::ops::download_f32(gpu, &out_ref, rows).expect("down ref");
        for r in 0..rows {
            let (a, b) = (got[u * rows + r], want[r]);
            let rel = if b.abs() > 1e-6 {
                ((a - b).abs() / b.abs()) as f64
            } else {
                (a - b).abs() as f64
            };
            worst = worst.max(rel);
            if u == 0 && a.to_bits() != b.to_bits() {
                same_m1 = false;
            }
        }
        gpu.destroy_buf(&xu);
    }
    println!(
        "dense_gemm: {} worst relative {worst:.3e}, m = 1 bit-identical: {same_m1}",
        ty.name()
    );
    assert!(
        worst <= 1e-6,
        "{}: multi-user differs by {worst:.3e}",
        ty.name()
    );
    assert!(same_m1, "{}: m = 1 must be bit-identical", ty.name());
    // Price the chunk-level shared expert (§8ab): one dispatch serves USERS_MAX
    // tokens, so a `t`-token chunk costs `t/m` dispatches per role instead of `t`.
    let timing = many
        .run_timed(&w, &x, &out_many, rows, cols, m, 0.002)
        .expect("timed");
    println!(
        "dense_gemm: {} m = {m}: {:.3} ms/dispatch -> {:.3} ms/token/role",
        ty.name(),
        timing.per_iter_s * 1e3,
        timing.per_iter_s * 1e3 / m as f64
    );
    gpu.destroy_buf(&out_ref);
    gpu.destroy_buf(&out_many);
    gpu.destroy_buf(&x);
    gpu.destroy_buf(&w);
    let _ = vk::WHOLE_SIZE;
}
