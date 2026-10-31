//! GPU ↔ CPU conformance for the Phase-3 op kernels (`crates/breccia-kernels/src/ops`).
//!
//! Every op is held to a documented number against its `f64` CPU reference in
//! [`breccia_kernels::ops::cpu`]:
//!
//! | op | bar | why |
//! |---|---|---|
//! | RMSNorm | 1e-5 rel | a 2560-term sum of squares in f32 |
//! | IMRoPE | 1e-5 rel | pure f32 rotation over an fp32 table |
//! | SwiGLU | 1e-5 rel | one transcendental + one multiply |
//! | Softmax | 1e-5 rel | two f32 reductions per row |
//! | Router | **ids exactly equal**, weights 1e-5 rel | the production kernel accumulates in f64, this one cannot (see `ops::Router`) |
//! | EmbGather | 1e-6 rel | the decode units are bit-exact; the gather adds no arithmetic |
//! | HC mixer | 1e-5 rel on `mixed`/`inject`, 1e-5 on HC_WRITE | a chain of RMSNorm + two GEMVs + sigmoid |
//! | Add / Act | 1e-6 rel | one add / one transcendental |
//!
//! # Running it
//!
//! GPU tests are opt-in by explicit device (`AGENTS.md` GPU crash safety) and
//! `BRECCIA_OP` narrows the suite to **one op per invocation** so a fault is
//! attributable:
//!
//! ```bash
//! # the safe card first (Radeon VII / MI50, no display):
//! timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
//!     BRECCIA_KERNEL_DEVICE=0000:0e:00.0 BRECCIA_OP=router \
//!     cargo test -p breccia-kernels --test ops_conformance -- --test-threads=1 --nocapture
//! ```
//!
//! Shapes are the model's own (`docs/specs/qwen4exp-architecture.md` §2):
//! 2560 hidden, 512 router, 640 ffn, 320 hc rank, head_dim 256, ssm 128/6144,
//! plus tails and odd widths where an op claims to handle them.

use breccia_kernels::ops::{
    self, cpu, Act, ActMode, Add, EmbGather, HcMixer, HcWeights, RmsNorm, Rope, RopeTable, Router,
    Softmax, SwiGlu,
};
use breccia_kernels::{Device, QuantType};

/// One device for the whole test binary, serialised: `ctx::Gpu` owns a single
/// command buffer per queue.
struct Fixture {
    _guard: std::sync::MutexGuard<'static, ()>,
    dev: &'static Device,
}

static DEV: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn device_selector() -> Option<String> {
    std::env::var("BRECCIA_KERNEL_DEVICE")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// `BRECCIA_OP=rmsnorm` (or a comma list) narrows every test to that op.
fn runs(op: &str) -> bool {
    match std::env::var("BRECCIA_OP") {
        Ok(v) if !v.trim().is_empty() => v.split(',').any(|s| s.trim().eq_ignore_ascii_case(op)),
        _ => true,
    }
}

fn fixture() -> Option<Fixture> {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(selector) = device_selector() else {
        eprintln!(
            "ops conformance: SKIPPED — set BRECCIA_KERNEL_DEVICE=<pci|index> \
             (0000:0e:00.0 = MI50 first; see AGENTS.md GPU crash safety)"
        );
        return None;
    };
    let dev = DEV
        .get_or_init(|| match Device::open(Some(&selector)) {
            Ok(d) => {
                println!(
                    "ops conformance: {} ({}), subgroup {}, {}",
                    d.info.name, d.info.device_id, d.info.subgroup_size, d.info.pci_address
                );
                Some(d)
            }
            Err(e) => {
                eprintln!("ops conformance: SKIPPED — no usable Vulkan device: {e}");
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

/// Max relative error with an absolute floor of 1.0 for near-zero values (the
/// same measure the decode conformance uses, so numbers are comparable).
fn max_rel(got: &[f32], want: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        m = m.max((f64::from(*g) - w).abs() / w.abs().max(1.0));
    }
    m
}

fn as_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

/// Deterministic value in ~N(0, 1) from a small LCG (no dev-dependency).
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

// ---------------------------------------------------------------------------

#[test]
fn rmsnorm_matches_the_cpu_reference() {
    let fx = need_gpu!("rmsnorm");
    let gpu = fx.dev.gpu();
    // (label, cols, rows, wgroups) — the model's four norm shapes plus tails.
    let cases: [(&str, usize, usize, usize); 7] = [
        ("hidden 2560", 2560, 4, 1),
        ("head 256", 256, 24, 1),
        ("head 256 (T=5 heads)", 256, 120, 1),
        ("ssm 128", 128, 48, 1),
        ("hc grouped 2560x4", 2560, 4, 4),
        ("hc grouped T=3", 2560, 12, 4),
        ("odd tail 257", 257, 3, 1),
    ];
    for (label, cols, rows, wgroups) in cases {
        // Two gamma cases: present, and absent (the L2-norm form).
        for gamma in [true, false] {
            let wgroups = if gamma { wgroups } else { 0 };
            let x = normalish(0x1234_5678 ^ cols as u64, rows * cols);
            let g = if gamma {
                Some(normalish(0xfeed, wgroups * cols))
            } else {
                None
            };
            let want = cpu::rmsnorm(&x, rows, cols, wgroups, g.as_deref(), ops::MODEL_RMS_EPS);

            let norm = RmsNorm::new(gpu, cols, wgroups).expect("pipeline");
            let xb = ops::upload_f32(gpu, &x).expect("x");
            let gb = g.as_ref().map(|g| ops::upload_f32(gpu, g).expect("gamma"));
            let ob = breccia_kernels::gemv::alloc(gpu, (rows * cols * 4) as u64).expect("out");
            norm.run(&xb, gb.as_ref(), &ob, rows).expect("rmsnorm");
            let got = ops::download_f32(gpu, &ob, rows * cols).expect("read");
            let rel = max_rel(&got, &as_f64(&want));
            assert!(
                rel <= 1e-5,
                "rmsnorm {label} cols={cols} rows={rows} wgroups={wgroups}: {rel:e} > 1e-5"
            );
            println!("rmsnorm  {label:<22} rows={rows:<4} gamma={gamma:<5} max rel {rel:.2e}");
            for b in [Some(&xb), gb.as_ref(), Some(&ob)].into_iter().flatten() {
                gpu.destroy_buf(b);
            }
        }
    }
}

#[test]
fn rope_matches_the_cpu_reference() {
    let fx = need_gpu!("rope");
    let gpu = fx.dev.gpu();
    let table = RopeTable::model(1024).expect("table");
    let tb = table.upload(gpu).expect("upload table");
    // (label, head_dim, n_rot, heads, tokens)
    let cases: [(&str, usize, usize, usize, usize); 6] = [
        ("q 24 heads", 256, 64, 24, 1),
        ("k 2 heads", 256, 64, 2, 1),
        ("q 24 heads T=3", 256, 64, 24, 3),
        ("indexer q", 128, 64, 4, 2),
        ("pooled key", 128, 64, 1, 5),
        ("full-width rot", 64, 64, 4, 2),
    ];
    for (label, head_dim, n_rot, heads, tokens) in cases {
        let rows = heads * tokens;
        let x = normalish(0xbeef ^ head_dim as u64, rows * head_dim);
        // Positions: token-major, so row r sees pos r / heads; include a
        // non-zero start to exercise the table lookup.
        let positions: Vec<i32> = (0..rows).map(|r| 7 + (r / heads) as i32).collect();
        let want = cpu::rope_apply(
            &x,
            rows,
            head_dim,
            n_rot,
            table.cos_sin(),
            table.max_pos(),
            &positions,
        );

        let rope = Rope::new(gpu, head_dim, n_rot, table.max_pos()).expect("pipeline");
        let xb = ops::upload_f32(gpu, &x).expect("x");
        let pb = breccia_kernels::gemv::upload(gpu, &i32_bytes(&positions)).expect("pos");
        let ob = breccia_kernels::gemv::alloc(gpu, (rows * head_dim * 4) as u64).expect("out");
        rope.run(&xb, &tb, &pb, &ob, rows).expect("rope");
        let got = ops::download_f32(gpu, &ob, rows * head_dim).expect("read");
        let rel = max_rel(&got, &as_f64(&want));
        assert!(rel <= 1e-5, "rope {label}: {rel:e} > 1e-5");

        // In place must give the same answer (the kernel claims it is safe).
        let ib = ops::upload_f32(gpu, &x).expect("in-place copy");
        rope.run(&ib, &tb, &pb, &ib, rows).expect("rope in place");
        let inplace = ops::download_f32(gpu, &ib, rows * head_dim).expect("read in place");
        let inplace_rel = max_rel(&inplace, &as_f64(&got));
        assert!(
            inplace_rel <= 1e-6,
            "rope {label}: in-place differs from out-of-place by {inplace_rel:e}"
        );
        println!(
            "rope     {label:<22} rows={rows:<4} max rel {rel:.2e} (in-place {inplace_rel:.1e})"
        );
        for b in [&xb, &pb, &ob, &ib] {
            gpu.destroy_buf(b);
        }
    }
    gpu.destroy_buf(&tb);
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[test]
fn swiglu_matches_the_cpu_reference() {
    let fx = need_gpu!("swiglu");
    let gpu = fx.dev.gpu();
    let gl = SwiGlu::new(gpu).expect("pipeline");
    // The routed set (10 x 640), the shared expert (640), a T=4 batch, and an
    // odd tail.
    for (label, n) in [
        ("routed 10x640", 6400usize),
        ("shared 640", 640),
        ("routed T=4", 6400 * 4),
        ("odd tail 641", 641),
    ] {
        // Include the saturating tails: silu(-inf) must be 0, silu(+big) = x.
        let mut gate = normalish(0x5eed, n);
        if n > 8 {
            gate[0] = -120.0;
            gate[1] = 120.0;
            gate[2] = 0.0;
            gate[3] = -0.0;
        }
        let up = normalish(0xabcd, n);
        let want = cpu::swiglu(&gate, &up);
        let gb = ops::upload_f32(gpu, &gate).expect("gate");
        let ub = ops::upload_f32(gpu, &up).expect("up");
        let ob = breccia_kernels::gemv::alloc(gpu, (n * 4) as u64).expect("out");
        gl.run(&gb, &ub, &ob, n).expect("swiglu");
        let got = ops::download_f32(gpu, &ob, n).expect("read");
        let rel = max_rel(&got, &as_f64(&want));
        assert!(rel <= 1e-5, "swiglu {label}: {rel:e} > 1e-5");
        println!("swiglu   {label:<22} n={n:<6} max rel {rel:.2e}");
        for b in [&gb, &ub, &ob] {
            gpu.destroy_buf(b);
        }
    }
}

#[test]
fn softmax_matches_the_cpu_reference() {
    let fx = need_gpu!("softmax");
    let gpu = fx.dev.gpu();
    let sm = Softmax::new(gpu).expect("pipeline");
    // (label, rows, cols): router 512, the attention widths (512..2048) with a
    // ragged tail, and a single row.
    for (label, rows, cols) in [
        ("router 512 x T=1", 1usize, 512usize),
        ("router 512 x T=8", 8, 512),
        ("attn 512", 24, 512),
        ("attn 1024", 24, 1024),
        ("attn 2048", 24, 2048),
        ("attn tail 2049", 3, 2049),
        ("attn tail 1000", 3, 1000),
    ] {
        // Logits wide enough that exp() alone would overflow f32 without the
        // max subtraction.
        let x: Vec<f32> = normalish(0x77 ^ cols as u64, rows * cols)
            .into_iter()
            .map(|v| v * 20.0)
            .collect();
        let want = cpu::softmax(&x, rows, cols);
        let xb = ops::upload_f32(gpu, &x).expect("x");
        let ob = breccia_kernels::gemv::alloc(gpu, (rows * cols * 4) as u64).expect("out");
        sm.run(&xb, &ob, rows, cols).expect("softmax");
        let got = ops::download_f32(gpu, &ob, rows * cols).expect("read");
        let rel = max_rel(&got, &as_f64(&want));
        assert!(rel <= 1e-5, "softmax {label}: {rel:e} > 1e-5");
        // Rows must sum to 1 in f32 as a property of the op itself.
        for r in 0..rows {
            let s: f64 = got[r * cols..(r + 1) * cols]
                .iter()
                .map(|v| f64::from(*v))
                .sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "softmax {label} row {r} sums to {s}"
            );
        }
        println!("softmax  {label:<22} rows={rows:<3} cols={cols:<5} max rel {rel:.2e}");
        gpu.destroy_buf(&xb);
        gpu.destroy_buf(&ob);
    }
}

#[test]
fn router_ids_are_exact_and_weights_within_a_documented_bar() {
    let fx = need_gpu!("router");
    let gpu = fx.dev.gpu();
    let r = Router::model(gpu).expect("pipeline");
    assert_eq!(r.n_expert(), 512);
    assert_eq!(r.k(), 10);

    for (label, rows, scale) in [
        ("random T=1", 1usize, 12.0f32),
        ("random T=16", 16, 12.0),
        ("narrow logits", 4, 0.5),
    ] {
        let logits: Vec<f32> = normalish(0x9e37 ^ rows as u64, rows * 512)
            .into_iter()
            .map(|v| v * scale)
            .collect();
        let lb = ops::upload_f32(gpu, &logits).expect("logits");
        let idb = breccia_kernels::gemv::alloc(gpu, r.ids_bytes(rows)).expect("ids");
        let wb = breccia_kernels::gemv::alloc(gpu, r.weights_bytes(rows)).expect("weights");
        r.run(&lb, &idb, &wb, rows).expect("router");
        let (ids, weights) = ops::bench::read_route(gpu, &idb, &wb, rows, 10).expect("read");

        for t in 0..rows {
            let want = cpu::router(&logits[t * 512..(t + 1) * 512], 512, 10);
            let got_ids: Vec<u32> = ids[t * 10..(t + 1) * 10].to_vec();
            assert_eq!(
                got_ids, want.ids,
                "router {label} row {t}: id sets differ (gpu {got_ids:?} vs cpu {:?})",
                want.ids
            );
            let rel = max_rel(&weights[t * 10..(t + 1) * 10], &want.weights);
            assert!(
                rel <= 1e-5,
                "router {label} row {t}: weight max rel {rel:e} > 1e-5"
            );
            // The renormalised weights always sum to 1 (the clamp never bites
            // for a real softmax, but the contract is the division by max(s, 2^-14)).
            let s: f32 = weights[t * 10..(t + 1) * 10].iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "router {label} row {t} weights sum to {s}"
            );
            println!(
                "router   {label:<16} row {t:<2} ids {:?} weight max rel {rel:.2e}",
                got_ids
            );
        }
        gpu.destroy_buf(&lb);
        gpu.destroy_buf(&idb);
        gpu.destroy_buf(&wb);
    }

    // Exact ties must resolve to the lowest id, on device as on the host. Six
    // experts share the top logit; the kernel's argmax keeps the lowest index.
    let mut logits = vec![-9.0f32; 512];
    for e in [37usize, 5, 300, 12, 411, 88] {
        logits[e] = 3.0;
    }
    let lb = ops::upload_f32(gpu, &logits).expect("logits");
    let idb = breccia_kernels::gemv::alloc(gpu, r.ids_bytes(1)).expect("ids");
    let wb = breccia_kernels::gemv::alloc(gpu, r.weights_bytes(1)).expect("weights");
    r.run(&lb, &idb, &wb, 1).expect("router");
    let (ids, _) = ops::bench::read_route(gpu, &idb, &wb, 1, 10).expect("read");
    let want = cpu::router(&logits, 512, 10);
    assert_eq!(
        ids, want.ids,
        "router tie-break: gpu {ids:?} vs cpu {:?}",
        want.ids
    );
    assert_eq!(
        &ids[..6],
        &[5u32, 12, 37, 88, 300, 411],
        "six tied experts must come out in ascending id order"
    );
    println!(
        "router   tie-break 6 tied experts -> {:?} (ascending id)",
        &ids[..6]
    );
    for b in [&lb, &idb, &wb] {
        gpu.destroy_buf(b);
    }
}

#[test]
fn embedding_gather_matches_the_dequant_reference() {
    let fx = need_gpu!("emb_gather");
    let gpu = fx.dev.gpu();
    if !gpu.feats.storage_buffer16_bit_access {
        eprintln!("skipped: device lacks 16-bit storage buffers");
        return;
    }
    // The two real embedding tables (type, cols) plus a tail shape, with the
    // gather taken from real quantised bytes.
    for (label, ty, cols, table_rows) in [
        ("token_embd IQ3_S", QuantType::Iq3S, 2560usize, 64usize),
        ("PLE IQ4_NL", QuantType::Iq4Nl, 160, 512),
        ("tail IQ4_XS", QuantType::Iq4Xs, 160 + 8, 32),
    ] {
        let g = EmbGather::new(gpu, ty, cols).expect("pipeline");
        assert_eq!(
            g.row_bytes(),
            (cols.div_ceil(ty.block_elems())) * ty.block_bytes()
        );
        let (table, _) = breccia_kernels::real::synthetic_inputs(
            ty,
            breccia_kernels::Shape::new(table_rows, cols),
            0x1234_5678,
        );
        // A batch that is out of order and repeated, like a token-id list.
        let ids: Vec<u32> = (0..table_rows as u32)
            .rev()
            .take(6)
            .chain([3u32, 3, 0])
            .collect();
        let batch = ids.len();

        // Reference: dequantise each requested row from the mapped bytes.
        let staged = EmbGather::pack_rows(&table, g.row_bytes(), &ids).expect("pack rows");
        let mut want = Vec::with_capacity(batch * g.padded_cols());
        for r in 0..batch {
            let row = &staged[r * g.row_bytes()..(r + 1) * g.row_bytes()];
            let mut v = breccia_kernels::reference::dequant_row(ty, row, cols);
            // pad to a whole block, zero-filled (the caller only reads `cols`)
            v.resize(g.padded_cols(), 0.0);
            want.extend_from_slice(&v);
        }

        let packed = breccia_kernels::gemv::upload(gpu, &staged).expect("upload rows");
        let idx = breccia_kernels::gemv::upload(
            gpu,
            &ops::u32_to_bytes(&EmbGather::identity_indices(batch)),
        )
        .expect("index list");
        let out = breccia_kernels::gemv::alloc(gpu, g.out_bytes(batch)).expect("out");
        g.run(&packed, &idx, &out, batch).expect("gather");
        let got = ops::download_f32(gpu, &out, batch * g.padded_cols()).expect("read");
        let rel = max_rel(&got, &as_f64(&want));
        assert!(rel <= 1e-6, "gather {label}: {rel:e} > 1e-6");

        // The device-side indirection must address the same rows when the whole
        // table is resident: upload it all and pass the ids as the index list.
        let full = breccia_kernels::gemv::upload(gpu, &table).expect("upload table");
        let out2 = breccia_kernels::gemv::alloc(gpu, g.out_bytes(batch)).expect("out2");
        let idx2 = breccia_kernels::gemv::upload(gpu, &ops::u32_to_bytes(&ids)).expect("ids");
        g.run(&full, &idx2, &out2, batch).expect("resident gather");
        let got2 = ops::download_f32(gpu, &out2, batch * g.padded_cols()).expect("read2");
        let same = max_rel(&got2, &as_f64(&got));
        assert!(
            same <= 1e-6,
            "gather {label}: resident path differs from the staged path by {same:e}"
        );
        println!(
            "gather   {label:<20} batch={batch:<3} row_bytes={:<5} max rel {rel:.2e} (resident {same:.1e})",
            g.row_bytes()
        );
        for b in [&packed, &idx, &out, &full, &idx2, &out2] {
            gpu.destroy_buf(b);
        }
    }
}

/// The fused `up` + gated-mean variant (§8az) must agree with the two-dispatch
/// shape: `mixed` **bit-identical** (same four-term order, same divide), `gate`
/// bit-identical, and `inject` within rounding (its partial *chunking* changes from
/// one partial per 256 elements to one per workgroup, so the summation grouping —
/// and only the grouping — differs).
#[test]
fn hc_gated_fusion_matches_the_separate_path() {
    let fx = need_gpu!("hc");
    let gpu = fx.dev.gpu();
    let n_embd = 2560usize;
    let hc = 4usize;
    let hc_lr = 320usize;
    let hc_dim = hc * n_embd;
    // bf16 = the top 16 bits of an f32; both paths get the same bytes.
    let bf16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let gamma = normalish(0x1111, hc_dim)
        .into_iter()
        .map(|v| v.abs() + 0.5)
        .collect::<Vec<f32>>();
    let w_down = normalish(0x2222, hc_lr * hc_dim);
    let w_up = normalish(0x3333, hc_dim * hc_lr);
    let w_inject = normalish(0x4444, hc * hc_dim);
    let r: Vec<f32> = normalish(0x5555, hc_dim);

    let gbuf = ops::upload_f32(gpu, &gamma).expect("gamma");
    let dbuf = breccia_kernels::gemv::upload(gpu, &bf16(&w_down)).expect("down");
    let ubuf = breccia_kernels::gemv::upload(gpu, &bf16(&w_up)).expect("up");
    let ibuf = breccia_kernels::gemv::upload(gpu, &bf16(&w_inject)).expect("inject");
    let rbuf = ops::upload_f32(gpu, &r).expect("r");
    let w = HcWeights {
        norm: &gbuf,
        down: &dbuf,
        up: &ubuf,
        inject: Some(&ibuf),
    };

    let mut got = Vec::new();

    for fuse in [false, true] {
        let mixer = HcMixer::with_fusion(gpu, QuantType::Bf16, n_embd, hc, hc_lr, true, fuse)
            .expect("mixer");
        assert_eq!(mixer.fused_gated(), fuse, "fusion switch");
        let ws = mixer.workspace(gpu, 1).expect("workspace");
        mixer.read(&ws, &rbuf, &w).expect("hc read");
        let mixed = ops::download_f32(gpu, &ws.mixed, n_embd).expect("mixed");
        let gate = ops::download_f32(gpu, &ws.gate, hc_dim).expect("gate");
        // `HcWrite` turns the partials into the four `2*sigmoid(inject/hc)` weights;
        // the raw partials are what each path writes, so sum them the same way.
        let chunks = mixer.inject_chunks();
        let part = ops::download_f32(gpu, &ws.inject, hc * chunks).expect("inject");
        let mut inject = Vec::with_capacity(hc);
        for c in 0..hc {
            let mut acc = 0.0f32;
            for k in 0..chunks {
                acc += part[c * chunks + k];
            }
            inject.push(acc);
        }

        got.push((mixed, gate, inject, chunks));
        mixer.release(gpu, ws);
    }
    let (mixed_a, gate_a, inj_a, chunks_a) = &got[0];
    let (mixed_b, gate_b, inj_b, chunks_b) = &got[1];
    assert_eq!(
        *chunks_a,
        n_embd / 256,
        "the separate path keeps its chunking"
    );
    assert_eq!(
        *chunks_b,
        n_embd / 2,
        "the fused path is one chunk per workgroup"
    );
    let mut mixed_diff = 0;
    for i in 0..n_embd {
        if mixed_a[i].to_bits() != mixed_b[i].to_bits() {
            mixed_diff += 1;
        }
    }
    assert_eq!(
        mixed_diff, 0,
        "mixed must be bit-identical ({mixed_diff} differ)"
    );
    let mut gate_diff = 0;
    for i in 0..hc_dim {
        if gate_a[i].to_bits() != gate_b[i].to_bits() {
            gate_diff += 1;
        }
    }
    assert_eq!(
        gate_diff, 0,
        "gate must be bit-identical ({gate_diff} differ)"
    );
    let inj_rel = max_rel(inj_b, &as_f64(inj_a));
    assert!(
        inj_rel <= 1e-6,
        "inject max rel {inj_rel:e} exceeds the 1e-6 contract"
    );
    println!(
        "hc fusion: mixed/gate bit-identical, inject max rel {inj_rel:e} \
         (chunks {} -> {})",
        chunks_a, chunks_b
    );
}

/// The opt-in `up` GEMM + gated-mean epilogue (`BRECCIA_FUSE_HC=1`, `T >= 64`) must
/// give the same `mixed` bit for bit and the same `inject` / `write` result to
/// rounding as the separate `up` + `gated` dispatches. `T` values off a multiple of
/// 64 exercise the dead-user lanes of the last tile row. `gate` is not written on
/// the fused path, so it is not compared.
#[test]
fn hc_tile_gated_fusion_matches_the_separate_path() {
    let fx = need_gpu!("hc");
    let gpu = fx.dev.gpu();
    let (n_embd, hc, hc_lr) = (2560usize, 4usize, 320usize);
    let hc_dim = hc * n_embd;
    let bf16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let gamma = normalish(0x1111, hc_dim)
        .into_iter()
        .map(|v| v.abs() + 0.5)
        .collect::<Vec<f32>>();
    let gbuf = ops::upload_f32(gpu, &gamma).expect("gamma");
    let dbuf = breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x2222, hc_lr * hc_dim)))
        .expect("down");
    let ubuf = breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x3333, hc_dim * hc_lr)))
        .expect("up");
    let ibuf = breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x4444, hc * hc_dim)))
        .expect("inject");
    let w = HcWeights {
        norm: &gbuf,
        down: &dbuf,
        up: &ubuf,
        inject: Some(&ibuf),
    };
    let sep = HcMixer::with_fusion_hc(gpu, QuantType::Bf16, n_embd, hc, hc_lr, true, true, false)
        .expect("separate mixer");
    let tiled = HcMixer::with_fusion_hc(gpu, QuantType::Bf16, n_embd, hc, hc_lr, true, true, true)
        .expect("tile-fused mixer");

    for tokens in [64usize, 100, 130] {
        assert!(!sep.tile_fused(tokens), "T={tokens}: flag off must not fuse");
        assert!(tiled.tile_fused(tokens), "T={tokens}: flag on must fuse");
        assert_eq!(sep.read_chunks(tokens), n_embd / 256);
        assert_eq!(tiled.read_chunks(tokens), n_embd / 16);
        let rbuf = ops::upload_f32(gpu, &normalish(0x5555, hc_dim * tokens)).expect("r");
        let bbuf = ops::upload_f32(gpu, &normalish(0x6666, n_embd * tokens)).expect("block");
        let mut got = Vec::new();
        for m in [&sep, &tiled] {
            let obuf =
                breccia_kernels::gemv::alloc(gpu, (hc_dim * tokens * 4) as u64).expect("out");
            let ws = m.workspace(gpu, tokens).expect("workspace");
            m.read(&ws, &rbuf, &w).expect("read");
            let mixed = ops::download_f32(gpu, &ws.mixed, n_embd * tokens).expect("mixed");
            let inject = m.inject_values(gpu, &ws).expect("inject");
            m.write(&ws, &rbuf, &bbuf, &obuf).expect("write");
            let out = ops::download_f32(gpu, &obuf, hc_dim * tokens).expect("out");
            m.release(gpu, ws);
            gpu.destroy_buf(&obuf);
            got.push((mixed, inject, out));
        }
        let mixed_diff = (0..n_embd * tokens)
            .filter(|&i| got[0].0[i].to_bits() != got[1].0[i].to_bits())
            .count();
        assert_eq!(mixed_diff, 0, "T={tokens}: mixed must be bit-identical");
        let inj_rel = max_rel(&got[1].1, &as_f64(&got[0].1));
        let out_rel = max_rel(&got[1].2, &as_f64(&got[0].2));
        println!("hc tile-gated T={tokens}: inject {inj_rel:.2e} write {out_rel:.2e}");
        assert!(inj_rel <= 1e-5, "T={tokens}: inject max rel {inj_rel:e}");
        assert!(out_rel <= 1e-5, "T={tokens}: write max rel {out_rel:e}");
        gpu.destroy_buf(&rbuf);
        gpu.destroy_buf(&bbuf);
    }
    for b in [&gbuf, &dbuf, &ubuf, &ibuf] {
        gpu.destroy_buf(b);
    }
}

/// A fused BF16 mixer at `T > 1` runs the separate gated-mean layout (the fusion
/// is `T = 1` only), so `write` must sum that layout's partial count: a T-token
/// read + write must match T single-token runs (the batched prefill HC chain).
#[test]
fn hc_fused_mixer_batched_matches_single_tokens() {
    let fx = need_gpu!("hc");
    let gpu = fx.dev.gpu();
    // `BRECCIA_HC_T` overrides the token count (the engine's chunk is 512).
    let tokens = std::env::var("BRECCIA_HC_T").ok().and_then(|v| v.parse().ok()).unwrap_or(3usize);
    let (n_embd, hc, hc_lr) = (2560usize, 4usize, 320usize);
    let hc_dim = hc * n_embd;
    let bf16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let gamma = normalish(0x1111, hc_dim)
        .into_iter()
        .map(|v| v.abs() + 0.5)
        .collect::<Vec<f32>>();
    let gbuf = ops::upload_f32(gpu, &gamma).expect("gamma");
    let dbuf = breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x2222, hc_lr * hc_dim)))
        .expect("down");
    let ubuf = breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x3333, hc_dim * hc_lr)))
        .expect("up");
    let ibuf = breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x4444, hc * hc_dim)))
        .expect("inject");
    let w = HcWeights {
        norm: &gbuf,
        down: &dbuf,
        up: &ubuf,
        inject: Some(&ibuf),
    };
    let r = normalish(0x5555, hc_dim * tokens);
    let block = normalish(0x6666, n_embd * tokens);
    let mixer = HcMixer::with_fusion(gpu, QuantType::Bf16, n_embd, hc, hc_lr, true, true)
        .expect("mixer");
    assert!(mixer.fused_gated(), "fusion must be on for this test");

    // Batched.
    let rbuf = ops::upload_f32(gpu, &r).expect("r");
    let bbuf = ops::upload_f32(gpu, &block).expect("block");
    let obuf = breccia_kernels::gemv::alloc(gpu, (hc_dim * tokens * 4) as u64).expect("out");
    let ws = mixer.workspace(gpu, tokens).expect("workspace");
    mixer.read(&ws, &rbuf, &w).expect("read");
    let got_mixed = ops::download_f32(gpu, &ws.mixed, n_embd * tokens).expect("mixed");
    mixer.write(&ws, &rbuf, &bbuf, &obuf).expect("write");
    let got_out = ops::download_f32(gpu, &obuf, hc_dim * tokens).expect("out");
    mixer.release(gpu, ws);

    // One token at a time (the fused path).
    let mut want_mixed = Vec::new();
    let mut want_out = Vec::new();
    for t in 0..tokens {
        let rb = ops::upload_f32(gpu, &r[t * hc_dim..(t + 1) * hc_dim]).expect("r1");
        let bb = ops::upload_f32(gpu, &block[t * n_embd..(t + 1) * n_embd]).expect("b1");
        let ob = breccia_kernels::gemv::alloc(gpu, (hc_dim * 4) as u64).expect("o1");
        let ws = mixer.workspace(gpu, 1).expect("ws1");
        mixer.read(&ws, &rb, &w).expect("read1");
        want_mixed.extend(ops::download_f32(gpu, &ws.mixed, n_embd).expect("m1"));
        mixer.write(&ws, &rb, &bb, &ob).expect("write1");
        want_out.extend(ops::download_f32(gpu, &ob, hc_dim).expect("o1"));
        mixer.release(gpu, ws);
        for b in [&rb, &bb, &ob] {
            gpu.destroy_buf(b);
        }
    }
    let mixed_rel = max_rel(&got_mixed, &as_f64(&want_mixed));
    let out_rel = max_rel(&got_out, &as_f64(&want_out));
    println!("hc fused batched T={tokens}: mixed {mixed_rel:.2e} write {out_rel:.2e}");
    assert!(mixed_rel <= 1e-5, "mixed max rel {mixed_rel:e}");
    assert!(out_rel <= 1e-5, "write max rel {out_rel:e}");
    for b in [&gbuf, &dbuf, &ubuf, &ibuf, &rbuf, &bbuf, &obuf] {
        gpu.destroy_buf(b);
    }
}

#[test]
fn hc_mixer_matches_the_cpu_reference() {
    let fx = need_gpu!("hc");
    let gpu = fx.dev.gpu();
    let n_embd = 2560usize;
    let hc = 4usize;
    let hc_lr = 320usize;
    let hc_dim = hc * n_embd;

    for &with_inject in &[true, false] {
        for tokens in [1usize, 2] {
            let mixer =
                HcMixer::new(gpu, QuantType::F32, n_embd, hc, hc_lr, with_inject).expect("mixer");

            // Weights: F32 so the CPU reference can consume the same bytes the
            // GPU does (the artifact ships BF16; the algebra is identical and
            // the BF16 decode is covered by the GEMV conformance suite).
            let gamma = normalish(0x1111, hc_dim)
                .into_iter()
                .map(|v| v.abs() + 0.5)
                .collect::<Vec<f32>>();
            let w_down = normalish(0x2222, hc_lr * hc_dim);
            let w_up = normalish(0x3333, hc_dim * hc_lr);
            let w_inject = normalish(0x4444, hc * hc_dim);
            let r: Vec<f32> = normalish(0x5555, hc_dim * tokens);

            let gbuf = ops::upload_f32(gpu, &gamma).expect("gamma");
            let dbuf = ops::upload_f32(gpu, &w_down).expect("down");
            let ubuf = ops::upload_f32(gpu, &w_up).expect("up");
            let ibuf = ops::upload_f32(gpu, &w_inject).expect("inject");
            let rbuf = ops::upload_f32(gpu, &r).expect("r");
            let ws = mixer.workspace(gpu, tokens).expect("workspace");
            let w = HcWeights {
                norm: &gbuf,
                down: &dbuf,
                up: &ubuf,
                inject: if with_inject { Some(&ibuf) } else { None },
            };
            mixer.read(&ws, &rbuf, &w).expect("hc read");

            let got_mixed = ops::download_f32(gpu, &ws.mixed, n_embd * tokens).expect("mixed");

            let mut want_mixed = Vec::with_capacity(n_embd * tokens);
            let mut want_inject = Vec::with_capacity(hc * tokens);
            for t in 0..tokens {
                let rt = &r[t * hc_dim..(t + 1) * hc_dim];
                let ref_read = cpu::hc_read_token(
                    rt,
                    n_embd,
                    hc,
                    hc_lr,
                    &gamma,
                    &w_down,
                    &w_up,
                    if with_inject { Some(&w_inject) } else { None },
                    ops::MODEL_RMS_EPS,
                );
                want_mixed.extend_from_slice(&ref_read.mixed);
                if let Some(inj) = ref_read.inject {
                    want_inject.extend_from_slice(&inj);
                }
            }
            let mixed_rel = max_rel(&got_mixed, &as_f64(&want_mixed));
            assert!(
                mixed_rel <= 1e-5,
                "hc read (inject={with_inject}, T={tokens}): mixed max rel {mixed_rel:e}"
            );
            let mut inject_rel = 0.0;
            if with_inject {
                // The `inject` projection is folded into the gated-mean dispatch,
                // so the workspace holds one partial per 256-wide chunk; the host
                // helper sums them exactly as `HcWrite` does on the device.
                let got_inject = mixer.inject_values(gpu, &ws).expect("inject");
                inject_rel = max_rel(&got_inject, &as_f64(&want_inject));
                assert!(
                    inject_rel <= 1e-5,
                    "hc read (T={tokens}): inject max rel {inject_rel:e} got {got_inject:?} want {want_inject:?}"
                );
            }

            // HC_WRITE: R_out[c] = R[c] + block * 2*sigmoid(inject[c]/hc)
            let block = normalish(0x6666, n_embd * tokens);
            let bbuf = ops::upload_f32(gpu, &block).expect("block");
            let obuf =
                breccia_kernels::gemv::alloc(gpu, (hc_dim * tokens * 4) as u64).expect("out");
            mixer.write(&ws, &rbuf, &bbuf, &obuf).expect("hc write");
            let got_write = ops::download_f32(gpu, &obuf, hc_dim * tokens).expect("write");
            let mut want_write = Vec::with_capacity(hc_dim * tokens);
            for t in 0..tokens {
                let rt = &r[t * hc_dim..(t + 1) * hc_dim];
                let inj = if with_inject {
                    &want_inject[t * hc..(t + 1) * hc]
                } else {
                    // The final mixer never runs HC_WRITE; with no inject the
                    // kernel would scale by 2*sigmoid(0/4) = 1, so the
                    // reference for the unbounded case is a plain add.
                    &vec![0.0f32; hc][..]
                };
                want_write.extend_from_slice(&cpu::hc_write(
                    rt,
                    &block[t * n_embd..(t + 1) * n_embd],
                    inj,
                    n_embd,
                    hc,
                ));
            }
            let write_rel = max_rel(&got_write, &as_f64(&want_write));
            assert!(
                write_rel <= 1e-5,
                "hc write (inject={with_inject}, T={tokens}): max rel {write_rel:e}"
            );
            println!(
                "hc       inject={with_inject:<5} T={tokens:<2} mixed {mixed_rel:.2e} \
                 inject {inject_rel:.2e} write {write_rel:.2e}"
            );

            mixer.release(gpu, ws);
            for b in [&gbuf, &dbuf, &ubuf, &ibuf, &rbuf, &bbuf, &obuf] {
                gpu.destroy_buf(b);
            }
        }
    }
}

#[test]
fn add_act_and_scale_match_the_cpu_reference() {
    let fx = need_gpu!("add");
    let gpu = fx.dev.gpu();
    let add = Add::new(gpu).expect("add pipeline");
    for (label, n) in [
        ("hidden 2560", 2560usize),
        ("hidden T=5", 2560 * 5),
        ("tail 257", 257),
    ] {
        let a = normalish(0x111, n);
        let b = normalish(0x222, n);
        for beta in [1.0f32, -0.75] {
            let want = cpu::add(&a, &b, beta);
            let ab = ops::upload_f32(gpu, &a).expect("a");
            let bb = ops::upload_f32(gpu, &b).expect("b");
            let ob = breccia_kernels::gemv::alloc(gpu, (n * 4) as u64).expect("out");
            add.run(&ab, &bb, &ob, n, beta).expect("add");
            let got = ops::download_f32(gpu, &ob, n).expect("read");
            let rel = max_rel(&got, &as_f64(&want));
            assert!(rel <= 1e-6, "add {label} beta={beta}: {rel:e} > 1e-6");
            println!("add      {label:<14} beta={beta:<6} max rel {rel:.2e}");
            for x in [&ab, &bb, &ob] {
                gpu.destroy_buf(x);
            }
        }
    }

    // Act: all five modes at 320 values (the HC bottleneck) and 10240 (a stream).
    for (mode, s) in [
        (ActMode::Silu, 0.0f32),
        (ActMode::SiluDiv, 4.0),
        (ActMode::Sigmoid, 0.0),
        (ActMode::Scale, 1.0 / (128.0f32).sqrt()),
        (ActMode::SiluScale, 0.125),
    ] {
        let act = Act::new(gpu, mode).expect("act pipeline");
        for n in [320usize, 10240] {
            let x = normalish(0x333 ^ n as u64, n);
            let want: Vec<f32> = match mode {
                ActMode::Silu => x.iter().map(|v| cpu::silu(f64::from(*v)) as f32).collect(),
                ActMode::SiluDiv => x
                    .iter()
                    .map(|v| cpu::silu(f64::from(*v) / f64::from(s)) as f32)
                    .collect(),
                ActMode::Sigmoid => cpu::sigmoid_vec(&x),
                ActMode::Scale => cpu::scale(&x, s),
                ActMode::SiluScale => x
                    .iter()
                    .map(|v| (cpu::silu(f64::from(*v)) * f64::from(s)) as f32)
                    .collect(),
            };
            let xb = ops::upload_f32(gpu, &x).expect("x");
            let ob = breccia_kernels::gemv::alloc(gpu, (n * 4) as u64).expect("out");
            act.run(&xb, &ob, n, s).expect("act");
            let got = ops::download_f32(gpu, &ob, n).expect("read");
            let rel = max_rel(&got, &as_f64(&want));
            assert!(rel <= 1e-5, "act {} n={n}: {rel:e} > 1e-5", mode.name());
            println!(
                "act      {:<12} n={n:<6} s={s:<8.4} max rel {rel:.2e}",
                mode.name()
            );
            gpu.destroy_buf(&xb);
            gpu.destroy_buf(&ob);
        }
    }
}

/// The op kernels must never read outside their bound storage buffers: every
/// buffer ends exactly at the last byte the kernel uses plus `TAIL_PAD`, and a
/// kernel that runs off the end is a UTCL2 permission fault (process death, and
/// a frozen desktop on a display card — `docs/phase-3/02` §4).
///
/// This test is the cheapest way to exercise that: one-element and one-row
/// shapes, where every strided loop runs exactly once and every guard
/// (`i >= n`, `row >= rows`, `blk >= total`) is on its boundary.
#[test]
fn degenerate_shapes_do_not_fault() {
    let fx = need_gpu!("add");
    let gpu = fx.dev.gpu();

    // denominator ops with n = 1
    let one = ops::upload_f32(gpu, &[1.0]).expect("one");
    let out1 = breccia_kernels::gemv::alloc(gpu, 4).expect("out1");
    Add::new(gpu)
        .expect("add")
        .run(&one, &one, &out1, 1, 1.0)
        .expect("add 1");
    Act::new(gpu, ActMode::Sigmoid)
        .expect("act")
        .run(&one, &out1, 1, 0.0)
        .expect("act 1");
    SwiGlu::new(gpu)
        .expect("swiglu")
        .run(&one, &one, &out1, 1)
        .expect("swiglu 1");
    Softmax::new(gpu)
        .expect("softmax")
        .run(&one, &out1, 1, 1)
        .expect("softmax 1x1");
    RmsNorm::new(gpu, 1, 0)
        .expect("rmsnorm")
        .run(&one, None, &out1, 1)
        .expect("rmsnorm 1x1");

    // a 1-row gather of each embedding type
    for (ty, cols) in [(QuantType::Iq3S, 2560usize), (QuantType::Iq4Nl, 160)] {
        let g = EmbGather::new(gpu, ty, cols).expect("gather");
        let row = vec![0u8; g.row_bytes()];
        let p = breccia_kernels::gemv::upload(gpu, &row).expect("row");
        let idx = breccia_kernels::gemv::upload(gpu, &ops::u32_to_bytes(&[0])).expect("idx");
        let o = breccia_kernels::gemv::alloc(gpu, g.out_bytes(1)).expect("out");
        g.run(&p, &idx, &o, 1).expect("gather 1 row");
        for b in [&p, &idx, &o] {
            gpu.destroy_buf(b);
        }
    }

    // a 1-token router and a 1-token HC_READ
    let r = Router::model(gpu).expect("router");
    let logits = ops::upload_f32(gpu, &vec![0.0f32; 512]).expect("logits");
    let ids = breccia_kernels::gemv::alloc(gpu, r.ids_bytes(1)).expect("ids");
    let w = breccia_kernels::gemv::alloc(gpu, r.weights_bytes(1)).expect("weights");
    r.run(&logits, &ids, &w, 1).expect("router 1 token");
    // all-equal logits: the top-10 must be 0..10 in order.
    let (got_ids, _) = ops::bench::read_route(gpu, &ids, &w, 1, 10).expect("read");
    assert_eq!(
        got_ids,
        (0..10u32).collect::<Vec<_>>(),
        "flat logits -> ids 0..10"
    );

    gpu.destroy_buf(&one);
    gpu.destroy_buf(&out1);
    gpu.destroy_buf(&logits);
    gpu.destroy_buf(&ids);
    gpu.destroy_buf(&w);
    println!("degenerate shapes: 1-element / 1-row / flat-logits cases are clean");
}

/// `ops::cpu` (the oracle) needs no GPU; this keeps the non-GPU path exercised
/// in the same binary so `cargo test` covers something even without a device.
#[test]
fn cpu_reference_self_checks() {
    let x = normalish(0x999, 8);
    let y = cpu::rmsnorm(&x, 1, 8, 0, None, 1e-6);
    assert!(y.iter().all(|v| v.is_finite()));
    let s = cpu::softmax(&x, 1, 8);
    let sum: f64 = s.iter().map(|v| f64::from(*v)).sum();
    assert!((sum - 1.0).abs() < 1e-7, "softmax sums to {sum}");
    let route = cpu::router(&normalish(0xaaa, 512), 512, 10);
    assert_eq!(route.ids.len(), 10);
    assert_eq!(route.weights.len(), 10);
}
