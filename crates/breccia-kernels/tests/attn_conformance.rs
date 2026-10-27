//! GPU ↔ CPU conformance for the GDN/SSM step and the QSA attention + indexer.
//!
//! What each test holds to what:
//!
//! | test | bar | why |
//! |---|---|---|
//! | GDN 32-token sequence, decode chunks (T=1) | 1e-5 rel | 19-step pipeline in f32 vs f64 |
//! | GDN 32-token sequence, prefill chunks (T=8) | 1e-5 rel | the T>1 path is the same kernel |
//! | GDN state bit-stability | **bit-identical** | two downloads agree, and a zero-token call changes nothing |
//! | QSA 64 tokens x 2 layers | 1e-5 rel | full sublayer; the reference rounds K/V to f16 exactly as the cache does, so the bar measures arithmetic, not storage |
//! | QSA indexer pooling (chunks straddling blocks) | 1e-5 rel | mean -> RMS -> rope at the block's first cell |
//! | QSA top-k at 2062 cells | **id lists exactly equal** | the selection budget (2051 cells) and the tie rule |
//! | QSA sparse attention at 2062 cells | 1e-5 rel | only the selected cells contribute |
//!
//! # Running it
//!
//! GPU tests are opt-in by device (`AGENTS.md` GPU crash safety) and `BRECCIA_OP`
//! narrows the suite to **one sublayer per process**:
//!
//! ```bash
//! timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
//!     BRECCIA_KERNEL_DEVICE=0000:0e:00.0 BRECCIA_OP=gdn \
//!     cargo test -p breccia-kernels --test attn_conformance -- --test-threads=1 --nocapture
//! ```
//!
//! The projection width in these tests is 256, not the model's 2560: the GEMV is
//! conformance-tested on the real shapes in `tests/conformance.rs`, and the
//! GDN/QSA kernels see only the *sublayer* geometry (S=128, h_k=16, h_v=48,
//! channels=10240, 24 heads x 256, indexer 4x128, 2051-cell budget), which is the
//! model's own. A smaller hidden width keeps the f64 reference inside a test
//! budget without weakening a single kernel claim.

use breccia_kernels::gdn::{self, Gdn, GdnProjections, GdnState, GdnWeights};
use breccia_kernels::proj::Proj;
use breccia_kernels::qsa::{self, Qsa, QsaProjections, QsaState, QsaWeights};
use breccia_kernels::{gemv, Device, Gemv, GemvBatch, QuantType};

// ---------------------------------------------------------------------------
// Fixture (one device per process, one sublayer at a time)
// ---------------------------------------------------------------------------

struct Fixture {
    _guard: std::sync::MutexGuard<'static, ()>,
    dev: &'static Device,
}

static DEV: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn runs(op: &str) -> bool {
    match std::env::var("BRECCIA_OP") {
        Ok(v) if !v.trim().is_empty() => v.split(',').any(|s| s.trim().eq_ignore_ascii_case(op)),
        _ => true,
    }
}

fn fixture() -> Option<Fixture> {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let selector = match std::env::var("BRECCIA_KERNEL_DEVICE") {
        Ok(s) if !s.trim().is_empty() => s,
        _ => {
            eprintln!(
                "attention conformance: SKIPPED — set BRECCIA_KERNEL_DEVICE=<pci|index> \
                 (0000:0e:00.0 = MI50 first; AGENTS.md GPU crash safety)"
            );
            return None;
        }
    };
    let dev = DEV
        .get_or_init(|| match Device::open(Some(&selector)) {
            Ok(d) => {
                println!(
                    "attention conformance: {} ({}), subgroup {}, {}",
                    d.info.name, d.info.device_id, d.info.subgroup_size, d.info.pci_address
                );
                Some(d)
            }
            Err(e) => {
                eprintln!("attention conformance: SKIPPED — no usable device: {e}");
                None
            }
        })
        .as_ref()?;
    Some(Fixture { _guard: guard, dev })
}

macro_rules! need_gpu {
    ($op:literal) => {{
        if !runs($op) {
            eprintln!("skipped by BRECCIA_OP ({} not selected)", $op);
            return;
        }
        match fixture() {
            Some(f) => f,
            None => return,
        }
    }};
}

/// Max relative error with an absolute floor of 1.0 (same measure as the other
/// conformance suites, so the numbers are comparable).
fn max_rel(got: &[f32], want: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        m = m.max((f64::from(*g) - w).abs() / w.abs().max(1.0));
    }
    m
}

fn normalish(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut acc = 0.0f32;
        for _ in 0..4 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            acc += ((s >> 33) as f32) / (u32::MAX >> 1) as f32;
        }
        out.push((acc - 1.0) * 1.732);
    }
    out
}

fn as_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

fn write_f32(gpu: &breccia_vk::ctx::Gpu, dst: &breccia_vk::ctx::Buf, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    breccia_kernels::gemv::upload_into(gpu, dst, &bytes).expect("upload_into");
}

fn read_f32(gpu: &breccia_vk::ctx::Gpu, src: &breccia_vk::ctx::Buf, n: usize) -> Vec<f32> {
    let raw = breccia_kernels::gemv::download(gpu, src, n * 4).expect("download");
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn upload(gpu: &breccia_vk::ctx::Gpu, v: &[f32]) -> breccia_vk::ctx::Buf {
    breccia_kernels::ops::upload_f32(gpu, v).expect("upload")
}

// ---------------------------------------------------------------------------
// GDN
// ---------------------------------------------------------------------------

const N_EMBD: usize = 256; // hidden width of these tests (kernel geometry is the model's)

struct GdnRefFixture {
    w_qkv: Vec<f32>,
    w_z: Vec<f32>,
    w_beta: Vec<f32>,
    w_alpha: Vec<f32>,
    w_conv: Vec<f32>,
    dt: Vec<f32>,
    a: Vec<f32>,
    ssm_norm: Vec<f32>,
    w_out: Vec<f32>,
}

fn gdn_weights() -> GdnRefFixture {
    let ch = gdn::CONV_CHANNELS;
    GdnRefFixture {
        w_qkv: normalish(0xA1, ch * N_EMBD),
        w_z: normalish(0xA2, gdn::VALUE_DIM * N_EMBD),
        w_beta: normalish(0xA3, gdn::MODEL_H_V * N_EMBD),
        w_alpha: normalish(0xA4, gdn::MODEL_H_V * N_EMBD),
        w_conv: normalish(0xA5, ch * gdn::MODEL_D_CONV),
        // `ssm_dt` lives in [-8, 2.53] in the artifact (§14).
        dt: normalish(0xA6, gdn::MODEL_H_V)
            .into_iter()
            .map(|v| 2.5f32 * (v / 3.0).tanh())
            .collect(),
        // `ssm_a = -exp(A_log)` is negative for all 48 heads (§14).
        a: normalish(0xA7, gdn::MODEL_H_V)
            .into_iter()
            .map(|v| -(v.abs() + 0.1))
            .collect(),
        ssm_norm: normalish(0xA8, gdn::MODEL_S)
            .into_iter()
            .map(|v| v.abs() + 0.25)
            .collect(),
        w_out: normalish(0xA9, N_EMBD * gdn::VALUE_DIM),
    }
}

/// One 32-token sequence in chunks of `chunk` tokens, GPU vs the f64 reference.
fn gdn_sequence(chunk: usize, label: &str) {
    let fx = match fixture() {
        Some(f) => f,
        None => return,
    };
    let gpu = fx.dev.gpu();
    let w = gdn_weights();
    let tokens = 32usize;

    let b_w_qkv = upload(gpu, &w.w_qkv);
    let b_w_z = upload(gpu, &w.w_z);
    let b_w_beta = upload(gpu, &w.w_beta);
    let b_w_alpha = upload(gpu, &w.w_alpha);
    let b_w_conv = upload(gpu, &w.w_conv);
    let b_dt = upload(gpu, &w.dt);
    let b_a = upload(gpu, &w.a);
    let b_ssm_norm = upload(gpu, &w.ssm_norm);
    let b_w_out = upload(gpu, &w.w_out);

    let g_qkv = Gemv::new(gpu, QuantType::F32, gdn::CONV_CHANNELS, N_EMBD).expect("qkv gemv");
    let g_z = Gemv::new(gpu, QuantType::F32, gdn::VALUE_DIM, N_EMBD).expect("z gemv");
    let g_beta = Gemv::new(gpu, QuantType::F32, gdn::MODEL_H_V, N_EMBD).expect("beta gemv");
    let g_alpha = Gemv::new(gpu, QuantType::F32, gdn::MODEL_H_V, N_EMBD).expect("alpha gemv");
    let g_out = Gemv::new(gpu, QuantType::F32, N_EMBD, gdn::VALUE_DIM).expect("out gemv");

    // Batched forms for the T > 1 path.
    let bt_qkv = GemvBatch::new(gpu, QuantType::F32, gdn::CONV_CHANNELS, N_EMBD).expect("b qkv");
    let bt_z = GemvBatch::new(gpu, QuantType::F32, gdn::VALUE_DIM, N_EMBD).expect("b z");
    let bt_beta = GemvBatch::new(gpu, QuantType::F32, gdn::MODEL_H_V, N_EMBD).expect("b beta");
    let bt_alpha = GemvBatch::new(gpu, QuantType::F32, gdn::MODEL_H_V, N_EMBD).expect("b alpha");
    let bt_out = GemvBatch::new(gpu, QuantType::F32, N_EMBD, gdn::VALUE_DIM).expect("b out");
    let map_qkv = breccia_kernels::proj::batch_map_buf(gpu, &bt_qkv, chunk).expect("map");
    let map_z = breccia_kernels::proj::batch_map_buf(gpu, &bt_z, chunk).expect("map");
    let map_beta = breccia_kernels::proj::batch_map_buf(gpu, &bt_beta, chunk).expect("map");
    let map_alpha = breccia_kernels::proj::batch_map_buf(gpu, &bt_alpha, chunk).expect("map");
    let map_out = breccia_kernels::proj::batch_map_buf(gpu, &bt_out, chunk).expect("map");

    // The reference comparison pins the **two-dispatch** `beta|alpha` shape, so the
    // merged layout is one variable at a time (its bit-identity to this shape is
    // asserted below and in `conformance::concatenated_rows_*`).
    let gdn = Gdn::with_merge(gpu, false).expect("gdn pipelines");
    let state = GdnState::new(gpu).expect("gdn state");
    state.reset(gpu).expect("reset");
    let ws = gdn.workspace(gpu, chunk).expect("gdn workspace");
    let wg = GdnWeights {
        conv: &b_w_conv,
        dt: &b_dt,
        a: &b_a,
        ssm_norm: &b_ssm_norm,
    };

    let mut ref_state = vec![0.0f64; gdn::MODEL_S * gdn::MODEL_H_V * gdn::MODEL_S];
    let mut ref_conv = vec![0.0f64; (gdn::MODEL_D_CONV - 1) * gdn::CONV_CHANNELS];
    let ref_w = gdn::cpu::LayerWeights {
        w_qkv: &as_f64(&w.w_qkv),
        w_z: &as_f64(&w.w_z),
        w_beta: &as_f64(&w.w_beta),
        w_alpha: &as_f64(&w.w_alpha),
        w_conv: &as_f64(&w.w_conv),
        dt: &as_f64(&w.dt),
        a: &as_f64(&w.a),
        ssm_norm: &as_f64(&w.ssm_norm),
        w_out: &as_f64(&w.w_out),
    };

    let x_all = normalish(0x1234, tokens * N_EMBD);
    let mut worst = 0.0f64;
    for (c, start) in (0..tokens).step_by(chunk).enumerate() {
        let n = chunk.min(tokens - start);
        let x_chunk = &x_all[start * N_EMBD..(start + n) * N_EMBD];
        let b_x = upload(gpu, x_chunk);
        let b_out = gemv::alloc(gpu, (n * N_EMBD * 4) as u64).expect("out");

        let (qkv, z, beta, alpha, out) = if n == 1 {
            (
                Proj::Single(&g_qkv),
                Proj::Single(&g_z),
                Proj::Single(&g_beta),
                Proj::Single(&g_alpha),
                Proj::Single(&g_out),
            )
        } else {
            (
                Proj::Batch {
                    pipe: &bt_qkv,
                    map: &map_qkv,
                },
                Proj::Batch {
                    pipe: &bt_z,
                    map: &map_z,
                },
                Proj::Batch {
                    pipe: &bt_beta,
                    map: &map_beta,
                },
                Proj::Batch {
                    pipe: &bt_alpha,
                    map: &map_alpha,
                },
                Proj::Batch {
                    pipe: &bt_out,
                    map: &map_out,
                },
            )
        };
        let proj = GdnProjections {
            qkv,
            w_qkv: &b_w_qkv,
            z,
            w_z: &b_w_z,
            beta,
            w_beta: &b_w_beta,
            alpha,
            w_alpha: &b_w_alpha,
            // The merged `beta|alpha` path is exercised by the graph's A/B and the
            // bit-identity test in `conformance`; this suite pins the two-dispatch
            // shape so the reference comparison stays one variable per run.
            beta_alpha: None,
            w_beta_alpha: None,
            out,
            w_out: &b_w_out,
        };
        // ---- the merged `beta|alpha` layout must agree bit for bit (§8ay) ----
        //
        // The merge runs `ssm_beta` and `ssm_alpha` (both `[48, n_embd]`, same type,
        // same input row) as **one** 96-row GEMV and reads both halves out of one
        // buffer. A GEMV row is an independent dot product, so this must be
        // bit-identical to the two-dispatch shape; `beta_gate` is compared too,
        // because that is where the row base is applied.
        if n == 1 {
            let beta: Vec<f32> = (0..48).map(|i| 0.25 + i as f32 * 0.011).collect();
            let alpha: Vec<f32> = (0..48).map(|i| -0.75 + i as f32 * 0.017).collect();
            let mut merged_in = beta.clone();
            merged_in.extend_from_slice(&alpha);

            let ws_m = gdn.workspace(gpu, 1).expect("merged workspace");
            write_f32(gpu, &ws_m.beta_raw, &beta);
            write_f32(gpu, &ws_m.alpha, &alpha);
            write_f32(gpu, &ws_m.beta_alpha, &merged_in);
            let ws_s = gdn.workspace(gpu, 1).expect("separate workspace");
            write_f32(gpu, &ws_s.beta_raw, &beta);
            write_f32(gpu, &ws_s.alpha, &alpha);
            write_f32(gpu, &ws_s.beta_alpha, &merged_in);

            let g_merged = Gdn::with_merge(gpu, true).expect("merged pipelines");
            let g_sep = Gdn::with_merge(gpu, false).expect("separate pipelines");
            g_merged.beta_gate(&ws_m, &wg).expect("beta_gate merged");
            g_sep.beta_gate(&ws_s, &wg).expect("beta_gate separate");
            let b1 = read_f32(gpu, &ws_m.beta, 48);
            let g1 = read_f32(gpu, &ws_m.gate, 48);
            let b2 = read_f32(gpu, &ws_s.beta, 48);
            let g2 = read_f32(gpu, &ws_s.gate, 48);
            for i in 0..48 {
                assert_eq!(b1[i].to_bits(), b2[i].to_bits(), "beta[{i}]");
                assert_eq!(g1[i].to_bits(), g2[i].to_bits(), "gate[{i}]");
            }
            println!("merged beta|alpha: beta and gate bit-identical to the two-buffer path");
            gdn.release(gpu, ws_m);
            gdn.release(gpu, ws_s);
        }

        // The workspace is sized for `chunk`; the last chunk may be shorter.
        if n != chunk {
            // Run the shorter tail with its own workspace (the kernels take the
            // token count as a push constant, the buffers only need to be large
            // enough, so the same workspace would work — but `ws.tokens` is what
            // the runner uses, hence a fresh one for clarity).
            let ws_tail = gdn.workspace(gpu, n).expect("tail workspace");
            gdn.forward(&state, &ws_tail, &wg, &proj, &b_x, &b_out)
                .expect("gdn forward");
            let got = breccia_kernels::ops::download_f32(gpu, &b_out, n * N_EMBD).expect("dl");
            let want = gdn::cpu::layer(
                &mut ref_state,
                &mut ref_conv,
                &as_f64(x_chunk),
                &ref_w,
                n,
                N_EMBD,
                f64::from(breccia_kernels::ops::MODEL_RMS_EPS),
            );
            worst = worst.max(max_rel(&got, &want));
            gdn.release(gpu, ws_tail);
        } else {
            gdn.forward(&state, &ws, &wg, &proj, &b_x, &b_out)
                .expect("gdn forward");
            let got = breccia_kernels::ops::download_f32(gpu, &b_out, n * N_EMBD).expect("dl");
            let want = gdn::cpu::layer(
                &mut ref_state,
                &mut ref_conv,
                &as_f64(x_chunk),
                &ref_w,
                n,
                N_EMBD,
                f64::from(breccia_kernels::ops::MODEL_RMS_EPS),
            );
            worst = worst.max(max_rel(&got, &want));
        }
        gpu.destroy_buf(&b_x);
        gpu.destroy_buf(&b_out);
        let _ = c;
    }

    // The state is the only thing that survives; compare it too, so a decay that
    // is "almost right" cannot hide behind the next chunk's output.
    let got_state = breccia_kernels::ops::download_f32(
        gpu,
        &state.state,
        gdn::MODEL_S * gdn::MODEL_H_V * gdn::MODEL_S,
    )
    .expect("state");
    let state_rel = max_rel(&got_state, &ref_state);
    assert!(
        worst <= 1e-5,
        "{label}: output max rel {worst:e} (bar 1e-5)"
    );
    assert!(
        state_rel <= 1e-5,
        "{label}: state max rel {state_rel:e} (bar 1e-5)"
    );
    println!("gdn {label:<22} out {worst:.2e}  state {state_rel:.2e}  (bar 1e-5)");

    // Bit-stability: two downloads agree, and a zero-token call (a no-op that
    // still runs the kernels) leaves every state byte alone.
    let again = breccia_kernels::ops::download_f32(
        gpu,
        &state.state,
        gdn::MODEL_S * gdn::MODEL_H_V * gdn::MODEL_S,
    )
    .expect("state");
    assert_eq!(
        got_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{label}: two state downloads differ"
    );
    let conv_before = breccia_kernels::ops::download_f32(
        gpu,
        &state.conv,
        (gdn::MODEL_D_CONV - 1) * gdn::CONV_CHANNELS,
    )
    .expect("conv");
    gdn.conv(&state, &ws.qkv, &b_w_conv, 0).expect("no-op conv");
    gdn.qk_l2_norm(&ws.qkv, 0).expect("no-op l2");
    gdn.delta_step(&state, &ws, &ws.step).expect("no-op delta");
    let state_after = breccia_kernels::ops::download_f32(
        gpu,
        &state.state,
        gdn::MODEL_S * gdn::MODEL_H_V * gdn::MODEL_S,
    )
    .expect("state");
    let conv_after = breccia_kernels::ops::download_f32(
        gpu,
        &state.conv,
        (gdn::MODEL_D_CONV - 1) * gdn::CONV_CHANNELS,
    )
    .expect("conv");
    assert_eq!(
        got_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        state_after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{label}: a no-op call changed the recurrent state"
    );
    assert_eq!(
        conv_before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        conv_after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{label}: a no-op call changed the conv history"
    );
    println!("gdn {label:<22} state bit-stable across a no-op call");

    gdn.release(gpu, ws);
    state.release(gpu);
    for b in [
        &b_w_qkv,
        &b_w_z,
        &b_w_beta,
        &b_w_alpha,
        &b_w_conv,
        &b_dt,
        &b_a,
        &b_ssm_norm,
        &b_w_out,
    ] {
        gpu.destroy_buf(b);
    }
    for b in [&map_qkv, &map_z, &map_beta, &map_alpha, &map_out] {
        gpu.destroy_buf(b);
    }
}

#[test]
fn gdn_decode_sequence_matches_the_cpu_reference() {
    need_gpu!("gdn");
    gdn_sequence(1, "decode T=1 x32");
}

#[test]
fn gdn_prefill_sequence_matches_the_cpu_reference() {
    need_gpu!("gdn");
    gdn_sequence(8, "prefill T=8 x4");
}

// ---------------------------------------------------------------------------
// QSA
// ---------------------------------------------------------------------------

struct QsaRefFixture {
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_idx_q: Vec<f32>,
    w_idx_k: Vec<f32>,
    w_out: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    idx_q_norm: Vec<f32>,
    idx_k_norm: Vec<f32>,
}

fn qsa_weights() -> QsaRefFixture {
    let hd = qsa::MODEL_HEAD_DIM;
    QsaRefFixture {
        w_q: normalish(0xB1, qsa::QGATE_DIM * N_EMBD),
        w_k: normalish(0xB2, qsa::MODEL_N_KV_HEAD * hd * N_EMBD),
        w_v: normalish(0xB3, qsa::MODEL_N_KV_HEAD * hd * N_EMBD),
        w_idx_q: normalish(0xB4, qsa::IDX_N_HEAD * qsa::IDX_DIM * N_EMBD),
        w_idx_k: normalish(0xB5, qsa::IDX_DIM * N_EMBD),
        w_out: normalish(0xB6, N_EMBD * qsa::ATTN_OUT_DIM),
        q_norm: normalish(0xB7, hd)
            .into_iter()
            .map(|v| v.abs() + 0.5)
            .collect(),
        k_norm: normalish(0xB8, hd)
            .into_iter()
            .map(|v| v.abs() + 0.5)
            .collect(),
        idx_q_norm: normalish(0xB9, qsa::IDX_DIM)
            .into_iter()
            .map(|v| v.abs() + 0.5)
            .collect(),
        idx_k_norm: normalish(0xBA, qsa::IDX_DIM)
            .into_iter()
            .map(|v| v.abs() + 0.5)
            .collect(),
    }
}

/// 64 tokens through two independent QSA layers, GPU vs the f64 reference, at
/// both chunk sizes: `T = 1` (the decode `Gemv` path) and `T = 8` (the batched
/// prefill path). The two must agree with each other as well as with the
/// reference, which is what pins "chunking does not change the answer".
fn qsa_sequence(chunk: usize, label: &str) {
    let fx = match fixture() {
        Some(f) => f,
        None => return,
    };
    let gpu = fx.dev.gpu();
    let w = qsa_weights();
    let tokens = 64usize;
    let max_pos = 4096usize;

    let b_w_q = upload(gpu, &w.w_q);
    let b_w_k = upload(gpu, &w.w_k);
    let b_w_v = upload(gpu, &w.w_v);
    let b_w_idx_q = upload(gpu, &w.w_idx_q);
    let b_w_idx_k = upload(gpu, &w.w_idx_k);
    let b_w_out = upload(gpu, &w.w_out);
    let b_q_norm = upload(gpu, &w.q_norm);
    let b_k_norm = upload(gpu, &w.k_norm);
    let b_idx_q_norm = upload(gpu, &w.idx_q_norm);
    let b_idx_k_norm = upload(gpu, &w.idx_k_norm);

    let hd_kv = qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM;
    let g_q = Gemv::new(gpu, QuantType::F32, qsa::QGATE_DIM, N_EMBD).expect("q");
    let g_k = Gemv::new(gpu, QuantType::F32, hd_kv, N_EMBD).expect("k");
    let g_v = Gemv::new(gpu, QuantType::F32, hd_kv, N_EMBD).expect("v");
    let g_iq = Gemv::new(gpu, QuantType::F32, qsa::IDX_N_HEAD * qsa::IDX_DIM, N_EMBD).expect("iq");
    let g_ik = Gemv::new(gpu, QuantType::F32, qsa::IDX_DIM, N_EMBD).expect("ik");
    let g_out = Gemv::new(gpu, QuantType::F32, N_EMBD, qsa::ATTN_OUT_DIM).expect("out");

    let bt_q = GemvBatch::new(gpu, QuantType::F32, qsa::QGATE_DIM, N_EMBD).expect("bq");
    let bt_k = GemvBatch::new(gpu, QuantType::F32, hd_kv, N_EMBD).expect("bk");
    let bt_v = GemvBatch::new(gpu, QuantType::F32, hd_kv, N_EMBD).expect("bv");
    let bt_iq =
        GemvBatch::new(gpu, QuantType::F32, qsa::IDX_N_HEAD * qsa::IDX_DIM, N_EMBD).expect("biq");
    let bt_ik = GemvBatch::new(gpu, QuantType::F32, qsa::IDX_DIM, N_EMBD).expect("bik");
    let bt_out = GemvBatch::new(gpu, QuantType::F32, N_EMBD, qsa::ATTN_OUT_DIM).expect("bout");
    let maps: Vec<breccia_vk::ctx::Buf> = if chunk > 1 {
        [&bt_q, &bt_k, &bt_v, &bt_iq, &bt_ik, &bt_out]
            .iter()
            .map(|p| breccia_kernels::proj::batch_map_buf(gpu, p, chunk).expect("map"))
            .collect()
    } else {
        Vec::new()
    };

    let qsa = Qsa::new(gpu, max_pos).expect("qsa pipelines");
    let table = qsa.table().cos_sin().to_vec();
    let wg = QsaWeights {
        q_norm: &b_q_norm,
        k_norm: &b_k_norm,
        idx_q_norm: &b_idx_q_norm,
        idx_k_norm: &b_idx_k_norm,
    };
    let ref_w = qsa::cpu::RefWeights {
        w_q: &w.w_q,
        w_k: &w.w_k,
        w_v: &w.w_v,
        w_idx_q: &w.w_idx_q,
        w_idx_k: &w.w_idx_k,
        w_out: &w.w_out,
        q_norm: &w.q_norm,
        k_norm: &w.k_norm,
        idx_q_norm: &w.idx_q_norm,
        idx_k_norm: &w.idx_k_norm,
        table: &table,
        max_pos,
        eps: breccia_kernels::ops::MODEL_RMS_EPS,
    };

    let mut states: Vec<QsaState> = (0..2)
        .map(|_| QsaState::new(gpu, max_pos, max_pos).expect("qsa state"))
        .collect();
    let mut ref_states: Vec<qsa::cpu::RefState> =
        (0..2).map(|_| qsa::cpu::RefState::new()).collect();

    let mut input = normalish(0xC0, tokens * N_EMBD);
    let mut worst = 0.0f64;
    let mut worst_a = 0.0f64;
    let mut bad_ids = 0usize;
    for layer in 0..2 {
        let mut next_input = vec![0.0f32; tokens * N_EMBD];
        for start in (0..tokens).step_by(chunk) {
            let n = chunk.min(tokens - start);
            let ws = qsa.workspace(gpu, n, start + n).expect("qsa workspace");
            let b_x = upload(gpu, &input[start * N_EMBD..(start + n) * N_EMBD]);
            let b_out = gemv::alloc(gpu, (n * N_EMBD * 4) as u64).expect("out");
            let proj = if n == 1 {
                QsaProjections {
                    q: Proj::Single(&g_q),
                    w_q: &b_w_q,
                    k: Proj::Single(&g_k),
                    w_k: &b_w_k,
                    v: Proj::Single(&g_v),
                    w_v: &b_w_v,
                    idx_q: Proj::Single(&g_iq),
                    w_idx_q: &b_w_idx_q,
                    idx_k: Proj::Single(&g_ik),
                    w_idx_k: &b_w_idx_k,
                    out: Proj::Single(&g_out),
                    w_out: &b_w_out,
                }
            } else {
                QsaProjections {
                    q: Proj::Batch {
                        pipe: &bt_q,
                        map: &maps[0],
                    },
                    w_q: &b_w_q,
                    k: Proj::Batch {
                        pipe: &bt_k,
                        map: &maps[1],
                    },
                    w_k: &b_w_k,
                    v: Proj::Batch {
                        pipe: &bt_v,
                        map: &maps[2],
                    },
                    w_v: &b_w_v,
                    idx_q: Proj::Batch {
                        pipe: &bt_iq,
                        map: &maps[3],
                    },
                    w_idx_q: &b_w_idx_q,
                    idx_k: Proj::Batch {
                        pipe: &bt_ik,
                        map: &maps[4],
                    },
                    w_idx_k: &b_w_idx_k,
                    out: Proj::Batch {
                        pipe: &bt_out,
                        map: &maps[5],
                    },
                    w_out: &b_w_out,
                }
            };
            qsa.forward(&states[layer], &ws, &wg, &proj, &b_x, &b_out, start)
                .expect("qsa forward");
            let got = breccia_kernels::ops::download_f32(gpu, &b_out, n * N_EMBD).expect("dl");
            let want = qsa::cpu::layer_step(
                &mut ref_states[layer],
                &input[start * N_EMBD..(start + n) * N_EMBD],
                n,
                start,
                N_EMBD,
                &ref_w,
            );
            worst = worst.max(max_rel(&got, &as_f64(&want.out)));
            // Attribute a mismatch: attention output before the projection.
            let got_a =
                breccia_kernels::ops::download_f32(gpu, &ws.attn_out, n * qsa::ATTN_OUT_DIM)
                    .expect("attn out");
            for t in 0..n {
                let a = &got_a[t * qsa::ATTN_OUT_DIM..(t + 1) * qsa::ATTN_OUT_DIM];
                // Relative to the vector's own scale, and informational only: this
                // test's unnormalised random inputs put V at O(1e3) by layer 1, so a
                // 1e-7 absolute probability difference (the f32-vs-f64 softmax) shows
                // up here as ~1e-4 on A and is diluted to ~1e-7 in the projection.
                // The bar that matters is the sublayer output (`worst`) and the ids.
                let scale = want.attn[t]
                    .iter()
                    .fold(0.0f64, |m, v| m.max(f64::from(*v).abs()));
                let mut d = 0.0f64;
                for (g, w) in a.iter().zip(want.attn[t].iter()) {
                    d = d.max((f64::from(*g) - f64::from(*w)).abs());
                }
                worst_a = worst_a.max(d / scale.max(1e-30));
            }
            next_input[start * N_EMBD..(start + n) * N_EMBD].copy_from_slice(&got);

            // Selection id sets: exact (order too — both are descending score,
            // ties to the lower block, tail last).
            let got_ids =
                breccia_kernels::ops::download_u32(gpu, &ws.ids, n * ws.max_ids).expect("ids");
            let got_counts = breccia_kernels::ops::download_u32(gpu, &ws.counts, n).expect("cnt");
            for t in 0..n {
                let count = got_counts[t] as usize;
                let got_row: Vec<u32> = got_ids[t * ws.max_ids..t * ws.max_ids + count].to_vec();
                if got_row != want.ids[t] {
                    eprintln!(
                        "layer {layer} token {}: got {got_row:?} want {:?}",
                        start + t,
                        want.ids[t]
                    );
                    bad_ids += 1;
                }
            }

            qsa.release(gpu, ws);
            gpu.destroy_buf(&b_x);
            gpu.destroy_buf(&b_out);
        }
        input = next_input;
    }

    assert_eq!(bad_ids, 0, "{label}: selection differs at {bad_ids} tokens");
    assert!(
        worst <= 1e-5,
        "{label}: qsa 2 layers output max rel {worst:e} attn {worst_a:e} (bar 1e-5)"
    );
    println!(
        "qsa 2 layers 64 tok {label:<14} out {worst:.2e} attn {worst_a:.2e}  ids exact  (bar 1e-5)"
    );

    for s in states.drain(..) {
        s.release(gpu);
    }
    for b in [
        &b_w_q,
        &b_w_k,
        &b_w_v,
        &b_w_idx_q,
        &b_w_idx_k,
        &b_w_out,
        &b_q_norm,
        &b_k_norm,
        &b_idx_q_norm,
        &b_idx_k_norm,
    ] {
        gpu.destroy_buf(b);
    }
    for m in &maps {
        gpu.destroy_buf(m);
    }
}

#[test]
fn qsa_two_layers_match_the_cpu_reference() {
    need_gpu!("qsa");
    qsa_sequence(1, "decode T=1");
    qsa_sequence(8, "prefill T=8");
}

/// Pooling across chunk boundaries (the tail-source path) vs the f64 reference.
#[test]
fn qsa_indexer_pooling_matches_the_cpu_reference() {
    let fx = need_gpu!("qsa");
    let gpu = fx.dev.gpu();
    let max_pos = 4096usize;
    let tokens = 64usize;
    let chunk = 5usize; // deliberately not a multiple of idx_block
    let gamma = normalish(0xBA, qsa::IDX_DIM)
        .into_iter()
        .map(|v| v.abs() + 0.5)
        .collect::<Vec<f32>>();
    let b_gamma = upload(gpu, &gamma);

    let qsa = Qsa::new(gpu, max_pos).expect("qsa");
    let table = qsa.table().cos_sin().to_vec();
    let state = QsaState::new(gpu, tokens + 8, max_pos).expect("state");
    let ws = qsa.workspace(gpu, chunk, 1).expect("ws");
    let raw = normalish(0xD1, tokens * qsa::IDX_DIM);

    for start in (0..tokens).step_by(chunk) {
        let n = chunk.min(tokens - start);
        let b_raw = upload(gpu, &raw[start * qsa::IDX_DIM..(start + n) * qsa::IDX_DIM]);
        // `idx_pool` reads `ws.idx_k`; upload the chunk there and run the pool
        // then the tail update, exactly as `forward` orders them.
        breccia_kernels::gemv::upload_into(
            gpu,
            &ws.idx_k,
            &breccia_kernels::ops::f32_to_bytes(
                &raw[start * qsa::IDX_DIM..(start + n) * qsa::IDX_DIM],
            ),
        )
        .expect("upload raw");
        qsa.idx_pool(&state, &ws, &b_gamma, start).expect("pool");
        qsa.idx_tail(&state, &ws, start).expect("tail");
        gpu.destroy_buf(&b_raw);
    }

    let n_blocks = tokens / qsa::IDX_BLOCK;
    let got = breccia_kernels::ops::download_f32(gpu, &state.idx_pooled, n_blocks * qsa::IDX_DIM)
        .expect("pooled");
    let want = qsa::cpu::pool_keys(
        &as_f64(&raw).iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        tokens,
        &gamma,
        &table,
        max_pos,
        qsa::IDX_DIM,
        qsa::IDX_BLOCK,
        qsa::MODEL_N_ROT,
        breccia_kernels::ops::MODEL_RMS_EPS,
    );
    let rel = max_rel(&got, &as_f64(&want));
    assert!(rel <= 1e-5, "indexer pooling max rel {rel:e} (bar 1e-5)");
    println!("qsa indexer pooling  {rel:.2e}  (bar 1e-5, {n_blocks} blocks, chunk {chunk})");

    qsa.release(gpu, ws);
    state.release(gpu);
    gpu.destroy_buf(&b_gamma);
}

/// The sparse path: 2062 cached cells (> 2051) so the selection drops blocks,
/// with the id lists compared exactly and attention over the selected cells.
#[test]
fn qsa_sparse_selection_and_attention_match_at_2062_cells() {
    let fx = need_gpu!("qsa");
    let gpu = fx.dev.gpu();
    let max_pos = 4096usize;
    let cells = 2062usize; // 515 whole blocks + 2 tail cells
    let n_blocks = cells / qsa::IDX_BLOCK;
    let qsa = Qsa::new(gpu, max_pos).expect("qsa");
    let state = QsaState::new(gpu, cells, max_pos).expect("state");

    // Synthetic indexer queries and pooled keys: this test is about the
    // selection + attention contract, and the projections that feed them are
    // covered by the layer test above.
    let idx_q = normalish(0xE1, qsa::IDX_N_HEAD * qsa::IDX_DIM);
    let pooled = normalish(0xE2, (n_blocks + 1) * qsa::IDX_DIM);
    breccia_kernels::gemv::upload_into(
        gpu,
        &state.idx_pooled,
        &breccia_kernels::ops::f32_to_bytes(&pooled),
    )
    .expect("pooled");

    // K/V: fill the paged cache in 256-cell chunks (so the append path is the one
    // the graph uses), K normalised + rotated is irrelevant here — what is
    // compared is attention over whatever the cache holds.
    let k_all = normalish(0xE3, cells * qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM);
    let v_all = normalish(0xE4, cells * qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM);
    let fill_chunk = 256usize;
    let mut fill_ws = qsa.workspace(gpu, fill_chunk, fill_chunk).expect("fill ws");
    for start in (0..cells).step_by(fill_chunk) {
        let n = fill_chunk.min(cells - start);
        if fill_ws.tokens != n {
            // `kv_append` appends exactly `ws.tokens` cells, so a short tail chunk
            // needs a workspace of its own length (otherwise it would append
            // uninitialised rows under the right-of-way cells).
            qsa.release(gpu, fill_ws);
            fill_ws = qsa.workspace(gpu, n, n).expect("fill ws");
        }
        let rows = n * qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM;
        breccia_kernels::gemv::upload_into(
            gpu,
            &fill_ws.k_cur,
            &breccia_kernels::ops::f32_to_bytes(
                &k_all[start * qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM..][..rows],
            ),
        )
        .expect("k");
        breccia_kernels::gemv::upload_into(
            gpu,
            &fill_ws.v_cur,
            &breccia_kernels::ops::f32_to_bytes(
                &v_all[start * qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM..][..rows],
            ),
        )
        .expect("v");
        qsa.kv_append(&state, &fill_ws, start).expect("append");
    }

    // One query token at the end of the cache.
    let n = 1usize;
    let pos0 = cells - 1;
    let ws = qsa.workspace(gpu, n, cells).expect("ws");
    breccia_kernels::gemv::upload_into(gpu, &ws.idx_q, &breccia_kernels::ops::f32_to_bytes(&idx_q))
        .expect("idx q");
    breccia_kernels::gemv::upload_into(
        gpu,
        &ws.q,
        &breccia_kernels::ops::f32_to_bytes(&normalish(
            0xE5,
            qsa::MODEL_N_HEAD * qsa::MODEL_HEAD_DIM,
        )),
    )
    .expect("q");
    // The gate half of `attn_q` decides the output scale; give it a non-trivial
    // value so the gate is exercised.
    let gate = normalish(0xE6, qsa::MODEL_N_HEAD * 2 * qsa::MODEL_HEAD_DIM);
    breccia_kernels::gemv::upload_into(gpu, &ws.q_full, &breccia_kernels::ops::f32_to_bytes(&gate))
        .expect("q_full");

    qsa.idx_score(&state, &ws, pos0).expect("score");
    qsa.topk(&ws, pos0).expect("topk");
    let got_ids = breccia_kernels::ops::download_u32(gpu, &ws.ids, ws.max_ids).expect("ids");
    let got_counts = breccia_kernels::ops::download_u32(gpu, &ws.counts, 1).expect("counts");

    // Reference: score the same pooled keys, select, then attend.
    let scores = qsa::cpu::indexer_score(
        &idx_q,
        &pooled,
        n_blocks,
        pos0,
        qsa::IDX_N_HEAD,
        qsa::IDX_DIM,
        qsa::IDX_BLOCK,
    );
    let want_ids =
        qsa::cpu::topk_blocks(&scores, qsa::IDX_TOPK_BLOCKS, Some(pos0 / qsa::IDX_BLOCK));
    let count = got_counts[0] as usize;
    assert_eq!(count, want_ids.len(), "selection length");
    let got_row: Vec<u32> = got_ids[..count].to_vec();
    assert_eq!(
        got_row, want_ids,
        "selection order/set differs (descending score, ties to the lower block, tail last)"
    );
    println!(
        "qsa sparse  blocks={n_blocks} cells={cells} selected={count} blocks ({} cells)",
        count * qsa::IDX_BLOCK
    );

    // Attention over exactly those cells.
    qsa.attend(&state, &ws, pos0).expect("attend");
    qsa.apply(&state, &ws, pos0).expect("apply");
    let got = breccia_kernels::ops::download_f32(
        gpu,
        &ws.attn_out,
        qsa::MODEL_N_HEAD * qsa::MODEL_HEAD_DIM,
    )
    .expect("out");

    let q = normalish(0xE5, qsa::MODEL_N_HEAD * qsa::MODEL_HEAD_DIM);
    let kf: Vec<f32> = k_all.iter().map(|&x| qsa::cpu::to_f16(x)).collect();
    let vf: Vec<f32> = v_all.iter().map(|&x| qsa::cpu::to_f16(x)).collect();
    let mut gate_vals = vec![0.0f32; qsa::MODEL_N_HEAD * qsa::MODEL_HEAD_DIM];
    for h in 0..qsa::MODEL_N_HEAD {
        for d in 0..qsa::MODEL_HEAD_DIM {
            gate_vals[h * qsa::MODEL_HEAD_DIM + d] =
                gate[h * 2 * qsa::MODEL_HEAD_DIM + qsa::MODEL_HEAD_DIM + d];
        }
    }
    let cells_sel = qsa::cpu::cells_of(&want_ids, qsa::IDX_BLOCK);
    let cells_of = cells_sel.clone();
    let want = qsa::cpu::attend(
        &q,
        &kf,
        &vf,
        &cells_sel,
        pos0,
        qsa::MODEL_N_HEAD,
        qsa::MODEL_N_KV_HEAD,
        qsa::MODEL_HEAD_DIM,
        &gate_vals,
    );
    // Debug: compare the GPU's probability matrix with the reference's for head 0.
    {
        let probs = breccia_kernels::ops::download_f32(gpu, &ws.attn, qsa::MODEL_N_HEAD * ws.slots)
            .expect("probs");
        let row = &probs[0..ws.slots];
        let sum: f32 = row.iter().sum();
        let mut nz = row.iter().filter(|p| **p != 0.0).count();
        eprintln!(
            "gpu head0 prob sum {sum:.7} nonzero {nz} slots {}",
            ws.slots
        );
        let row23 = &probs[23 * ws.slots..24 * ws.slots];
        eprintln!("gpu head23 sum {:.7}", row23.iter().sum::<f32>());
        nz = 0;
        for (i, cell) in cells_of.iter().enumerate() {
            if *cell as usize > pos0 {
                nz += 1;
            }
            let _ = i;
        }
        eprintln!("reference valid cells {nz} of {}", cells_sel.len());
        // Reference probabilities for head 0, same masking rule.
        let mut logits: Vec<(usize, f64)> = Vec::new();
        for (slot, cell) in cells_sel.iter().enumerate() {
            let c = *cell as usize;
            if c > pos0 {
                continue;
            }
            let mut dot = 0.0f64;
            for d in 0..qsa::MODEL_HEAD_DIM {
                dot += f64::from(q[d])
                    * f64::from(kf[c * qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM + d]);
            }
            logits.push((slot, dot / 16.0));
        }
        let mx = logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f64::NEG_INFINITY, f64::max);
        let den: f64 = logits.iter().map(|(_, l)| (l - mx).exp()).sum();
        let mut cp: Vec<(usize, f64)> = logits
            .iter()
            .map(|(s, l)| (*s, (l - mx).exp() / den))
            .collect();
        cp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut gp: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        gp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("cpu top10 {:?}", &cp[..10]);
        eprintln!("gpu top10 {:?}", &gp[..10]);
        let tail_mass: f32 = row[2048..].iter().sum();
        let cpu_tail: f64 = cp.iter().filter(|(s, _)| *s >= 2048).map(|(_, p)| p).sum();
        eprintln!("tail-slot mass gpu {tail_mass:.6} cpu {cpu_tail:.6}");
        // Weighted sum of V using the GPU's own probabilities: if this reproduces
        // the kernel output, the apply phase is consistent and any remaining
        // difference is in the logits.
        let mut acc = vec![0.0f64; qsa::MODEL_HEAD_DIM];
        for (slot, cell) in cells_sel.iter().enumerate() {
            let c = *cell as usize;
            if c > pos0 {
                continue;
            }
            let p = f64::from(row[slot]);
            for d in 0..qsa::MODEL_HEAD_DIM {
                acc[d] += p * f64::from(vf[c * qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM + d]);
            }
        }
        let mut g = vec![0.0f32; qsa::MODEL_HEAD_DIM];
        for d in 0..qsa::MODEL_HEAD_DIM {
            g[d] = (acc[d] * (1.0 / (1.0 + (-f64::from(gate_vals[d])).exp()))) as f32;
        }
        eprintln!(
            "recomputed-from-gpu-probs head0 {:?} vs kernel {:?}",
            &g[..4],
            &got[..4]
        );
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
    // Attribute the result: the probabilities the kernel softmaxed must match the
    // reference's to f32 (this is what catches a wrong cell, scale or mask), and
    // the V accumulation is then pure f32 arithmetic.
    {
        let probs = breccia_kernels::ops::download_f32(gpu, &ws.attn, 24 * ws.slots).expect("p");
        let mut logits: Vec<(usize, f64)> = Vec::new();
        for (slot, cell) in cells_sel.iter().enumerate() {
            let c = *cell as usize;
            if c > pos0 {
                continue;
            }
            let mut dot = 0.0f64;
            for d in 0..qsa::MODEL_HEAD_DIM {
                dot += f64::from(q[d])
                    * f64::from(kf[c * qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM + d]);
            }
            logits.push((slot, dot / 16.0));
        }
        let mx = logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f64::NEG_INFINITY, f64::max);
        let den: f64 = logits.iter().map(|(_, l)| (l - mx).exp()).sum();
        let mut maxdp = 0.0f64;
        let mut nbad = 0usize;
        for (slot, l) in &logits {
            let cp = (l - mx).exp() / den;
            let dp = (cp - f64::from(probs[*slot])).abs();
            maxdp = maxdp.max(dp);
            if dp > 1e-7 {
                nbad += 1;
            }
        }
        assert!(
            maxdp <= 1e-6,
            "attention probabilities differ: max |dp| {maxdp:e} over {nbad} slots"
        );
        println!(
            "qsa sparse probs    max |dp| {maxdp:.2e} over {} cells (bar 1e-6)",
            logits.len()
        );
    }
    let rel = max_rel(&got, &as_f64(&want));
    assert!(rel <= 1e-5, "sparse attention max rel {rel:e} (bar 1e-5)");
    println!("qsa sparse attention {rel:.2e}  (bar 1e-5)");

    qsa.release(gpu, ws);
    qsa.release(gpu, fill_ws);
    state.release(gpu);
}

/// The KV page layout and its f16 precision, checked element-for-element: the
/// physical address of a cell's K/V is
/// `((page * n_kv_head + kv_head) * PAGE_CELLS + cell_in_page) * head_dim + d`,
/// with `page = page_table[cell / PAGE_CELLS]`, and the stored value is the
/// round-to-nearest-even f16 of the f32 input (which is what `packHalf2x16`
/// does, and what the reference in `qsa::cpu::to_f16` models).
#[test]
fn qsa_kv_cache_layout_and_f16_precision() {
    let fx = need_gpu!("qsa");
    let gpu = fx.dev.gpu();
    let cells = 40usize; // three pages: 0..15, 16..31, 32..39
    let qsa = Qsa::new(gpu, 64).expect("qsa");
    let state = QsaState::new(gpu, cells, 64).expect("state");

    // One token per append, positions 5..37, so the first token lands mid-page
    // and the page boundary (cell 16) is crossed by the sequence.
    let start = 5usize;
    let n_tok = 33usize;
    let ws = qsa.workspace(gpu, 1, start + n_tok).expect("ws");
    let dim = 2 * qsa::MODEL_HEAD_DIM; // both kv heads of one cell
    let mut expect_k = vec![0.0f32; (start + n_tok) * dim];
    let mut expect_v = vec![0.0f32; (start + n_tok) * dim];
    for t in 0..n_tok {
        let cell = start + t;
        let k: Vec<f32> = (0..dim)
            .map(|i| ((cell * 97 + i * 13) as f32 * 0.017).sin() * 3.0)
            .collect();
        let v: Vec<f32> = (0..dim)
            .map(|i| ((cell * 61 + i * 7) as f32 * 0.031).cos() * 2.0)
            .collect();
        expect_k[cell * dim..(cell + 1) * dim].copy_from_slice(&k);
        expect_v[cell * dim..(cell + 1) * dim].copy_from_slice(&v);
        breccia_kernels::gemv::upload_into(gpu, &ws.k_cur, &breccia_kernels::ops::f32_to_bytes(&k))
            .expect("k");
        breccia_kernels::gemv::upload_into(gpu, &ws.v_cur, &breccia_kernels::ops::f32_to_bytes(&v))
            .expect("v");
        qsa.kv_append(&state, &ws, cell).expect("append");
    }

    let words = (state.capacity_cells / qsa::PAGE_CELLS
        * qsa::MODEL_N_KV_HEAD
        * qsa::PAGE_CELLS
        * qsa::MODEL_HEAD_DIM)
        / 2;
    let k_words = breccia_kernels::ops::download_u32(gpu, &state.k_pool, words).expect("k_pool");
    let v_words = breccia_kernels::ops::download_u32(gpu, &state.v_pool, words).expect("v_pool");
    // The `u16` half a value must occupy: the f16 encoding of the f32 (which is
    // already on the f16 grid because the reference rounded it).
    let f16_bits = |x: f32| -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let e = (((b >> 23) & 0xff) as i32 - 127 + 15) as u16;
        let m = ((b >> 13) & 0x3ff) as u16;
        sign | (e << 10) | m
    };
    let half_of = |w: u32, odd: bool| -> u16 {
        if odd {
            (w >> 16) as u16
        } else {
            (w & 0xffff) as u16
        }
    };

    let mut checked = 0usize;
    let mut mismatches = 0usize;
    for cell in start..start + n_tok {
        for kvh in 0..qsa::MODEL_N_KV_HEAD {
            for d in 0..qsa::MODEL_HEAD_DIM {
                let page = cell / qsa::PAGE_CELLS;
                let in_page = cell % qsa::PAGE_CELLS;
                let elem = ((page * qsa::MODEL_N_KV_HEAD + kvh) * qsa::PAGE_CELLS + in_page)
                    * qsa::MODEL_HEAD_DIM
                    + d;
                let word = elem / 2;
                let odd = elem % 2 == 1;
                let want_k = f16_bits(qsa::cpu::to_f16(
                    expect_k[cell * dim + kvh * qsa::MODEL_HEAD_DIM + d],
                ));
                let want_v = f16_bits(qsa::cpu::to_f16(
                    expect_v[cell * dim + kvh * qsa::MODEL_HEAD_DIM + d],
                ));
                let got_k = half_of(k_words[word], odd);
                let got_v = half_of(v_words[word], odd);
                if got_k != want_k || got_v != want_v {
                    if mismatches < 6 {
                        let in_k = expect_k[cell * dim + kvh * qsa::MODEL_HEAD_DIM + d];
                        let in_v = expect_v[cell * dim + kvh * qsa::MODEL_HEAD_DIM + d];
                        eprintln!(
                            "mismatch cell {cell} kvh {kvh} d {d}: K in {:08x} got {got_k:04x} \
                             want {want_k:04x} | V in {:08x} got {got_v:04x} want {want_v:04x}",
                            in_k.to_bits(),
                            in_v.to_bits()
                        );
                    }
                    mismatches += 1;
                }
                checked += 1;
            }
        }
    }
    println!(
        "qsa kv layout+f16   {checked} K/V elements, {mismatches} differ from to_f16 \
         ({n_tok} cells, {} pages)",
        state.capacity_cells / qsa::PAGE_CELLS
    );
    assert_eq!(mismatches, 0, "f16 encoding differs from qsa::cpu::to_f16");

    qsa.release(gpu, ws);
    state.release(gpu);
}

// ---------------------------------------------------------------------------
// q8_0 V cache (`BRECCIA_KV_Q8V`, docs/phase-4/01-forward-pass.md §8bg)
// ---------------------------------------------------------------------------

/// One attention pass over a filled 64-cell cache with the given V layout.
///
/// The block selection is uploaded instead of computed (`topk` is covered above);
/// what this exercises is exactly the pair this round added — `qsa_kv_append`'s
/// q8_0 store and `qsa_attn_apply`'s in-register dequant — plus the pool sizing
/// that feeds them.
#[allow(clippy::too_many_arguments)]
fn qsa_v_attention(
    gpu: &breccia_vk::ctx::Gpu,
    qsa: &Qsa,
    layout: qsa::VLayout,
    max_pos: usize,
    cells: usize,
    ids: &[u32],
    k_all: &[f32],
    v_all: &[f32],
    q: &[f32],
    q_full: &[f32],
) -> (Vec<f32>, Vec<f32>, u64) {
    let state = QsaState::with_layout(gpu, cells, max_pos, layout).expect("state");
    assert_eq!(state.v_layout, layout);
    let pages = state.capacity_cells / qsa::PAGE_CELLS;
    let want_bytes = match layout {
        qsa::VLayout::F16 => qsa::sizes::f16_pool_bytes(pages),
        qsa::VLayout::Q8_0 => qsa::sizes::q8v_pool_bytes(pages),
    };
    assert!(
        state.v_pool.size >= want_bytes,
        "v_pool is {} B for {layout:?}, the layout needs {want_bytes} B",
        state.v_pool.size
    );

    // Fill every cell through the append path the graph uses.
    let fill = qsa.workspace(gpu, cells, cells).expect("fill ws");
    write_f32(gpu, &fill.k_cur, k_all);
    write_f32(gpu, &fill.v_cur, v_all);
    qsa.kv_append(&state, &fill, 0).expect("append");
    qsa.release(gpu, fill);

    // One query token at the last cell, over the uploaded block ids.
    let ws = qsa.workspace(gpu, 1, cells).expect("ws");
    assert_eq!(ws.max_ids, cells / qsa::IDX_BLOCK + 1);
    write_f32(gpu, &ws.q, q);
    write_f32(gpu, &ws.q_full, q_full);
    let mut id_row = vec![0u32; ws.max_ids];
    id_row[..ids.len()].copy_from_slice(ids);
    breccia_kernels::gemv::upload_into(gpu, &ws.ids, &breccia_kernels::ops::u32_to_bytes(&id_row))
        .expect("ids");
    breccia_kernels::gemv::upload_into(
        gpu,
        &ws.counts,
        &breccia_kernels::ops::u32_to_bytes(&[ids.len() as u32]),
    )
    .expect("counts");

    let pos0 = cells - 1;
    qsa.attend(&state, &ws, pos0).expect("attend");
    qsa.apply(&state, &ws, pos0).expect("apply");
    let out = read_f32(gpu, &ws.attn_out, qsa::MODEL_N_HEAD * qsa::MODEL_HEAD_DIM);
    let probs = read_f32(gpu, &ws.attn, qsa::MODEL_N_HEAD * ws.slots);
    let pool_bytes = state.v_pool.size;
    qsa.release(gpu, ws);
    state.release(gpu);
    (out, probs, pool_bytes)
}

/// The q8_0 **V** path, held to two things at once:
///
/// 1. the CPU reference **with the same quantisation** (`qsa::cpu::to_q8_0`), at the
///    1e-5 bar the f16 path is held to — so the layout (288 B rows, 9-word blocks,
///    sign-extended i8 lanes) is right, not merely close;
/// 2. its distance from the f16 path, reported as a number: q8_0 is lossy and this
///    is where the loss is measured at the kernel level.
#[test]
fn qsa_v_q8_0_matches_the_reference_and_the_f16_path() {
    let fx = need_gpu!("q8v");
    let gpu = fx.dev.gpu();
    let max_pos = 4096usize;
    let cells = 64usize;
    let ids = [0u32, 5, 9, 15];
    let qsa = Qsa::new(gpu, max_pos).expect("qsa");

    let dim = qsa::MODEL_N_KV_HEAD * qsa::MODEL_HEAD_DIM;
    let k_all = normalish(0xF1, cells * dim);
    let v_all = normalish(0xF2, cells * dim);
    let q = normalish(0xF3, qsa::MODEL_N_HEAD * qsa::MODEL_HEAD_DIM);
    let q_full = normalish(0xF4, qsa::MODEL_N_HEAD * 2 * qsa::MODEL_HEAD_DIM);

    let (out_f16, _, bytes_f16) = qsa_v_attention(
        gpu,
        &qsa,
        qsa::VLayout::F16,
        max_pos,
        cells,
        &ids,
        &k_all,
        &v_all,
        &q,
        &q_full,
    );
    let (out_q8, _, bytes_q8) = qsa_v_attention(
        gpu,
        &qsa,
        qsa::VLayout::Q8_0,
        max_pos,
        cells,
        &ids,
        &k_all,
        &v_all,
        &q,
        &q_full,
    );

    // Gate half of the projection, as `attend`'s reference takes it.
    let mut gate = vec![0.0f32; qsa::MODEL_N_HEAD * qsa::MODEL_HEAD_DIM];
    for h in 0..qsa::MODEL_N_HEAD {
        for d in 0..qsa::MODEL_HEAD_DIM {
            gate[h * qsa::MODEL_HEAD_DIM + d] =
                q_full[h * 2 * qsa::MODEL_HEAD_DIM + qsa::MODEL_HEAD_DIM + d];
        }
    }
    let cells_sel = qsa::cpu::cells_of(&ids, qsa::IDX_BLOCK);
    let kf: Vec<f32> = k_all.iter().map(|&x| qsa::cpu::to_f16(x)).collect();
    let vf: Vec<f32> = v_all.iter().map(|&x| qsa::cpu::to_f16(x)).collect();
    let vq = qsa::cpu::to_q8_0(&v_all);
    let want_f16 = qsa::cpu::attend(
        &q,
        &kf,
        &vf,
        &cells_sel,
        cells - 1,
        qsa::MODEL_N_HEAD,
        qsa::MODEL_N_KV_HEAD,
        qsa::MODEL_HEAD_DIM,
        &gate,
    );
    let want_q8 = qsa::cpu::attend(
        &q,
        &kf,
        &vq,
        &cells_sel,
        cells - 1,
        qsa::MODEL_N_HEAD,
        qsa::MODEL_N_KV_HEAD,
        qsa::MODEL_HEAD_DIM,
        &gate,
    );

    let rel_f16 = max_rel(&out_f16, &as_f64(&want_f16));
    let rel_q8 = max_rel(&out_q8, &as_f64(&want_q8));
    assert!(
        rel_f16 <= 1e-5,
        "f16 attention max rel {rel_f16:e} (bar 1e-5)"
    );
    assert!(
        rel_q8 <= 1e-5,
        "q8_0 attention max rel {rel_q8:e} (bar 1e-5) — the pool layout or the \
         in-register dequant does not match qsa::cpu::to_q8_0"
    );
    // q8_0 vs f16 on the same input: the loss this round is allowed to have.
    let rel_vs_f16 = max_rel(&out_q8, &as_f64(&out_f16));
    let max_abs = out_q8
        .iter()
        .zip(&out_f16)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "qsa v-q8_0          ref {rel_q8:.2e} (f16 path {rel_f16:.2e}), vs f16 {rel_vs_f16:.2e} \
         rel / {max_abs:.2e} abs; pool {bytes_f16} -> {bytes_q8} B",
    );
}
