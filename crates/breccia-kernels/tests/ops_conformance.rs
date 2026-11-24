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
//! | Router | **ids exactly equal** (Inkling sigmoid: id sets), weights 1e-5 rel | the production kernel accumulates in f64, this one cannot (see `ops::Router`) |
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
    RowClip, Softmax, SwiGlu,
};
use breccia_kernels::ops::{DepthMix, Glu, GluMode};
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
    let ubuf =
        breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x3333, hc_dim * hc_lr))).expect("up");
    let ibuf =
        breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x4444, hc * hc_dim))).expect("inject");
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
        assert!(
            !sep.tile_fused(tokens),
            "T={tokens}: flag off must not fuse"
        );
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
    let tokens = std::env::var("BRECCIA_HC_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3usize);
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
    let ubuf =
        breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x3333, hc_dim * hc_lr))).expect("up");
    let ibuf =
        breccia_kernels::gemv::upload(gpu, &bf16(&normalish(0x4444, hc * hc_dim))).expect("inject");
    let w = HcWeights {
        norm: &gbuf,
        down: &dbuf,
        up: &ubuf,
        inject: Some(&ibuf),
    };
    let r = normalish(0x5555, hc_dim * tokens);
    let block = normalish(0x6666, n_embd * tokens);
    let mixer =
        HcMixer::with_fusion(gpu, QuantType::Bf16, n_embd, hc, hc_lr, true, true).expect("mixer");
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

    // Act: all six modes at 320 values (the HC bottleneck) and 10240 (a stream).
    for (mode, s) in [
        (ActMode::Silu, 0.0f32),
        (ActMode::SiluDiv, 4.0),
        (ActMode::Sigmoid, 0.0),
        (ActMode::Scale, 1.0 / (128.0f32).sqrt()),
        (ActMode::SiluScale, 0.125),
        (ActMode::TanhCap, 1.0),
        (ActMode::TanhCap, 30.0),
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
                ActMode::TanhCap => cpu::tanh_cap(&x, s),
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

/// Inkling S4 (`docs/specs/inkling2-loop-port.md`): tanh/softcap and the row
/// clip with channel scale and tau, against the f64 CPU reference; then, with
/// `BRECCIA_INKLING_GOLDEN=<golden dir>`, against llama.cpp's own Q/K/R clip
/// outputs for all 48 loop slots (input = the golden's pre-clip tensor).
/// A golden dump (`ffn_in-3_2304x13.f32`, with or without a `SEQ_` prefix) or
/// weight file (`blk.3.exp_probs_b.bias_128.f32`) under `golden/dir`, as raw words.
fn golden_f32(golden: &std::path::Path, dir: &str, stem: &str) -> Vec<f32> {
    let d = golden.join(dir);
    let hit = std::fs::read_dir(&d)
        .expect("golden dir")
        .filter_map(|e| e.ok().map(|e| e.file_name().into_string().unwrap()))
        .find(|f| {
            f.starts_with(stem)
                || f.split_once('_')
                    .is_some_and(|(_, rest)| rest.starts_with(stem))
        })
        .unwrap_or_else(|| panic!("{stem} not in {}", d.display()));
    std::fs::read(d.join(hit))
        .expect("read")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// A route's (ids, weights) sorted by id, for order-free comparison.
fn by_id<T: Copy>(ids: &[u32], w: &[T]) -> (Vec<u32>, Vec<T>) {
    let mut p: Vec<(u32, T)> = ids.iter().copied().zip(w.iter().copied()).collect();
    p.sort_by_key(|x| x.0);
    p.into_iter().unzip()
}

/// Inkling's router (S5): sigmoid gate, selection-only bias, `sum + 1e-20`
/// renorm, `* 8`. The id SET is contractual and each id's weight is within 1e-5;
/// the order is not: f32 sigmoid saturates, so near-equal scores can swap (seen
/// on random data), and the expert sum does not depend on it.
#[test]
fn inkling_router() {
    let fx = need_gpu!("inkling_router");
    let gpu = fx.dev.gpu();
    let (n, k) = (128usize, 8usize);
    let r = Router::sigmoid(gpu, n, k, 1e-20, 8.0).expect("router");
    let check =
        |logits: &[f32], bias: Option<&[f32]>, rows: usize, label: &str| -> (Vec<u32>, Vec<f32>) {
            let lb = ops::upload_f32(gpu, logits).expect("logits");
            let bb = bias.map(|b| ops::upload_f32(gpu, b).expect("bias"));
            let idb = breccia_kernels::gemv::alloc(gpu, r.ids_bytes(rows)).expect("ids");
            let wb = breccia_kernels::gemv::alloc(gpu, r.weights_bytes(rows)).expect("weights");
            match &bb {
                Some(b) => r.run_biased(&lb, b, &idb, &wb, rows),
                None => r.run(&lb, &idb, &wb, rows),
            }
            .expect("router");
            let (ids, w) = ops::bench::read_route(gpu, &idb, &wb, rows, k).expect("read");
            let mut worst = 0.0f64;
            for t in 0..rows {
                let want = cpu::router_sigmoid(&logits[t * n..(t + 1) * n], bias, n, k, 1e-20, 8.0);
                let (gi, gw) = by_id(&ids[t * k..(t + 1) * k], &w[t * k..(t + 1) * k]);
                let (wi, ww) = by_id(&want.ids, &want.weights);
                assert_eq!(gi, wi, "router {label} row {t} id sets");
                worst = worst.max(max_rel(&gw, &ww));
            }
            assert!(
                worst <= 1e-5,
                "router {label}: weight max rel {worst:e} > 1e-5"
            );
            println!("inkling router {label:<18} rows={rows:<3} weight max rel {worst:.2e}");
            for b in [Some(&lb), bb.as_ref(), Some(&idb), Some(&wb)]
                .into_iter()
                .flatten()
            {
                gpu.destroy_buf(b);
            }
            (ids, w)
        };

    // Random logits, with and without a bias of the model's range (-0.76..0.07).
    let rows = 37;
    let logits: Vec<f32> = normalish(0x5e1, rows * n)
        .into_iter()
        .map(|v| v * 3.0)
        .collect();
    let bias: Vec<f32> = normalish(0xb1a5, n)
        .into_iter()
        .map(|v| v * 0.3 - 0.2)
        .collect();
    check(&logits, None, rows, "random");
    check(&logits, Some(&bias), rows, "random + bias");
    // The bias must decide selection but not weights: a big bias on expert 100
    // with a very negative logit gets it picked at a near-zero weight.
    let mut l1 = vec![0.0f32; n];
    l1[100] = -20.0;
    let mut b1 = vec![0.0f32; n];
    b1[100] = 5.0;
    let (ids, w) = check(&l1, Some(&b1), 1, "bias steers only");
    assert_eq!(
        ids,
        [100, 0, 1, 2, 3, 4, 5, 6],
        "bias picks 100, then ties by id"
    );
    assert!(
        w[0] < 1e-7,
        "biased pick keeps its unbiased weight: {}",
        w[0]
    );

    let Ok(golden) = std::env::var("BRECCIA_INKLING_GOLDEN") else {
        eprintln!("inkling router golden parity: SKIPPED (set BRECCIA_INKLING_GOLDEN)");
        return;
    };
    let golden = std::path::PathBuf::from(golden);
    let tok = 13usize;
    let mut worst = 0.0f64;
    for slot in 0..48usize {
        let phys = match slot {
            0..=23 => slot,
            _ => 6 + (slot - 24) % 12,
        };
        let logits = golden_f32(&golden, "dump", &format!("ffn_moe_logits-{slot}_"));
        let order = golden_f32(&golden, "dump", &format!("ffn_moe_argsort-{slot}_"));
        let want_w = golden_f32(&golden, "dump", &format!("ffn_moe_weights_scaled-{slot}_"));
        let bias = golden_f32(&golden, "weights", &format!("blk.{phys}.exp_probs_b.bias_"));
        assert_eq!(
            (logits.len(), order.len(), want_w.len()),
            (n * tok, n * tok, k * tok)
        );
        let lb = ops::upload_f32(gpu, &logits).expect("logits");
        let bb = ops::upload_f32(gpu, &bias).expect("bias");
        let idb = breccia_kernels::gemv::alloc(gpu, r.ids_bytes(tok)).expect("ids");
        let wb = breccia_kernels::gemv::alloc(gpu, r.weights_bytes(tok)).expect("weights");
        r.run_biased(&lb, &bb, &idb, &wb, tok).expect("router");
        let (ids, w) = ops::bench::read_route(gpu, &idb, &wb, tok, k).expect("read");
        for t in 0..tok {
            // ggml's argsort is i32; the dump stores the raw words.
            let want: Vec<u32> = order[t * n..t * n + k]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let (gi, gw) = by_id(&ids[t * k..(t + 1) * k], &w[t * k..(t + 1) * k]);
            let (wi, ww) = by_id(&want, &want_w[t * k..(t + 1) * k]);
            assert_eq!(gi, wi, "golden id sets slot {slot} token {t}");
            let rel = max_rel(&gw, &as_f64(&ww));
            assert!(
                rel <= 1e-5,
                "golden weights slot {slot} token {t}: {rel:e} > 1e-5"
            );
            worst = worst.max(rel);
        }
        for b in [&lb, &bb, &idb, &wb] {
            gpu.destroy_buf(b);
        }
    }
    println!("golden ffn_moe ids + weights_scaled, 48 slots: worst max rel {worst:.2e}");
}

/// Inkling S6: SITU over split and fused `[gate | up]` rows against the f64
/// reference, `GluMode::SwiGlu` bit-identical to [`SwiGlu`]; then, with
/// `BRECCIA_INKLING_GOLDEN`, golden `ffn_moe_gate_up` (dump-s6) -> `ffn_moe_situ`
/// for all 48 slots.
#[test]
fn inkling_situ() {
    let fx = need_gpu!("inkling_situ");
    let gpu = fx.dev.gpu();
    let (beta, lb) = (4.0f32, 25.0f32);
    let situ = Glu::new(gpu, GluMode::Situ { beta, lb }).expect("situ");
    let read = |b: &breccia_vk::ctx::Buf, n: usize| ops::download_f32(gpu, b, n).expect("read");
    let (rows, half) = (37usize, 1152usize);
    // Wide range: saturates both tanh caps (|g| >> 4, |u| >> 25).
    let gu: Vec<f32> = normalish(0x51, rows * 2 * half)
        .into_iter()
        .map(|v| v * 12.0)
        .collect();
    let (mut g, mut u) = (Vec::new(), Vec::new());
    for r in gu.chunks_exact(2 * half) {
        g.extend_from_slice(&r[..half]);
        u.extend_from_slice(&r[half..]);
    }
    let n = rows * half;
    let want = as_f64(&cpu::situ(&g, &u, beta, lb));
    let (gub, gb, ub) = (
        ops::upload_f32(gpu, &gu).expect("gu"),
        ops::upload_f32(gpu, &g).expect("g"),
        ops::upload_f32(gpu, &u).expect("u"),
    );
    let ob = breccia_kernels::gemv::alloc(gpu, (n * 4) as u64).expect("out");
    situ.run_fused(&gub, &ob, rows, half).expect("fused");
    let fused = max_rel(&read(&ob, n), &want);
    situ.run(&gb, &ub, &ob, n).expect("split");
    let split = max_rel(&read(&ob, n), &want);
    // lb = 0: plain up.
    let nolb = Glu::new(gpu, GluMode::Situ { beta, lb: 0.0 }).expect("situ lb0");
    nolb.run(&gb, &ub, &ob, n).expect("lb0");
    let plain = max_rel(&read(&ob, n), &as_f64(&cpu::situ(&g, &u, beta, 0.0)));
    for (label, e) in [("fused", fused), ("split", split), ("lb=0", plain)] {
        println!("inkling situ {label:<6} n={n} max rel {e:.2e}");
        assert!(e <= 1e-5, "situ {label}: {e:e} > 1e-5");
    }
    // SwiGlu mode == the SwiGlu op, bit for bit (same silu helper).
    let sw = Glu::new(gpu, GluMode::SwiGlu).expect("glu swiglu");
    sw.run_fused(&gub, &ob, rows, half).expect("glu swiglu");
    let a = read(&ob, n);
    SwiGlu::new(gpu)
        .expect("swiglu")
        .run(&gb, &ub, &ob, n)
        .expect("swiglu");
    assert_eq!(a, read(&ob, n), "GluMode::SwiGlu fused == SwiGlu");
    for b in [&gub, &gb, &ub, &ob] {
        gpu.destroy_buf(b);
    }

    let Ok(golden) = std::env::var("BRECCIA_INKLING_GOLDEN") else {
        eprintln!("inkling situ golden parity: SKIPPED (set BRECCIA_INKLING_GOLDEN)");
        return;
    };
    let golden = std::path::PathBuf::from(golden);
    let rows = 8 * 13; // experts used x tokens
    let mut worst = 0.0f64;
    for slot in 0..48usize {
        let gu = golden_f32(&golden, "dump-s6", &format!("ffn_moe_gate_up-{slot}_"));
        let want = golden_f32(&golden, "dump", &format!("ffn_moe_situ-{slot}_"));
        assert_eq!((gu.len(), want.len()), (rows * 2 * half, rows * half));
        let gub = ops::upload_f32(gpu, &gu).expect("gu");
        let ob = breccia_kernels::gemv::alloc(gpu, (rows * half * 4) as u64).expect("out");
        situ.run_fused(&gub, &ob, rows, half).expect("situ");
        let e = max_rel(&read(&ob, rows * half), &as_f64(&want));
        assert!(e <= 1e-5, "golden situ slot {slot}: {e:e} > 1e-5");
        worst = worst.max(e);
        for b in [&gub, &ob] {
            gpu.destroy_buf(b);
        }
    }
    println!("golden ffn_moe_situ from fused gate_up, 48 slots: worst max rel {worst:.2e}");
}

/// Inkling S7: the depth mix (AttnRes inputs, FFN/output residual mix, loop
/// message bank) against the f64 reference; then, with `BRECCIA_INKLING_GOLDEN`,
/// a replay of llama.cpp's block bookkeeping over the golden layer inputs:
/// `attn_res_mix`, `ffn_res_mix`, `loop_msgs` for every slot and `out_res_mix`.
#[test]
fn inkling_depth_mix() {
    let fx = need_gpu!("inkling_depth_mix");
    let gpu = fx.dev.gpu();
    let read = |b: &breccia_vk::ctx::Buf, n: usize| ops::download_f32(gpu, b, n).expect("read");
    // Runs one mix; `blocks` oldest first (stack order), `cur` last.
    let mix = |dm: &DepthMix,
               n: usize,
               blocks: &[&[f32]],
               cur: &[f32],
               query: &[f32],
               nq: usize,
               pad: usize,
               first: usize| {
        let tokens = cur.len() / n;
        let stride = tokens * n + pad;
        let mut st = vec![0f32; (first + blocks.len()).max(1) * stride];
        for (i, b) in blocks.iter().enumerate() {
            st[(first + i) * stride..][..tokens * n].copy_from_slice(b);
        }
        let (sb, cb, qb) = (
            ops::upload_f32(gpu, &st).expect("stack"),
            ops::upload_f32(gpu, cur).expect("cur"),
            ops::upload_f32(gpu, query).expect("query"),
        );
        let ob = breccia_kernels::gemv::alloc(gpu, (tokens * nq * n * 4) as u64).expect("out");
        dm.run(&sb, first, blocks.len(), stride, &cb, &qb, nq, &ob, tokens)
            .expect("depth mix");
        let got = read(&ob, tokens * nq * n);
        for b in [&sb, &cb, &qb, &ob] {
            gpu.destroy_buf(b);
        }
        let mut order: Vec<&[f32]> = blocks.iter().rev().copied().collect();
        order.push(cur);
        let want = cpu::depth_mix(&order, query, nq, n, 1e-6);
        (got, want)
    };
    for (n, tokens) in [(2304usize, 5usize), (1000, 3)] {
        let dm = DepthMix::new(gpu, n, 1e-6).expect("depth mix");
        for (count, nq) in [(0usize, 1usize), (2, 3), (4, 4), (7, 1)] {
            let blocks: Vec<Vec<f32>> = (0..count)
                .map(|i| {
                    normalish(0xb10c + i as u64, tokens * n)
                        .into_iter()
                        .map(|v| v * (1.0 + i as f32))
                        .collect()
                })
                .collect();
            let refs: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
            let cur = normalish(0xc0, tokens * n);
            // Model scale: real AttnRes / message queries have std ~0.2-0.55.
            let query: Vec<f32> = normalish(0x9e, nq * n)
                .into_iter()
                .map(|v| v * 0.5)
                .collect();
            let (got, want) = mix(&dm, n, &refs, &cur, &query, nq, 37, 1);
            let e = max_rel(&got, &as_f64(&want));
            println!("depth mix n={n:<4} T={tokens} blocks={count} nq={nq}: max rel {e:.2e}");
            assert!(e <= 1e-5, "depth mix n={n} blocks={count} nq={nq}: {e:e}");
        }
    }

    let Ok(golden) = std::env::var("BRECCIA_INKLING_GOLDEN") else {
        eprintln!("inkling depth mix golden parity: SKIPPED (set BRECCIA_INKLING_GOLDEN)");
        return;
    };
    let golden = std::path::PathBuf::from(golden);
    let n = 2304usize;
    let dm = DepthMix::new(gpu, n, 1e-6).expect("depth mix");
    let dump = |name: &str| golden_f32(&golden, "dump", &format!("{name}_"));
    let weight = |name: &str| golden_f32(&golden, "weights", &format!("{name}_"));
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    let mut check = |label: String, blocks: &[String], cur: &[f32], query: &[f32]| {
        let nq = query.len() / n;
        let bv: Vec<Vec<f32>> = blocks.iter().map(|b| dump(b)).collect();
        let refs: Vec<&[f32]> = bv.iter().map(|b| b.as_slice()).collect();
        let (got, _) = mix(&dm, n, &refs, cur, query, nq, 0, 0);
        let want = dump(&label);
        let e = max_rel(&got, &as_f64(&want));
        assert!(
            e <= 1e-5,
            "golden {label}: {e:e} > 1e-5 (blocks {blocks:?})"
        );
        worst = worst.max(e);
        checked += 1;
    };
    // Execution order: prelude + core pass 0, loop passes 1 and 2, post.
    let phys = |slot: usize| {
        if slot < 24 {
            slot
        } else {
            6 + (slot - 24) % 12
        }
    };
    let mut blocks: Vec<String> = Vec::new();
    let mut cur = "inp_norm".to_string();
    let mut bank: (Vec<String>, String) = (Vec::new(), String::new());
    for slot in (0..18).chain(24..48).chain(18..24) {
        let ip = phys(slot);
        if slot == 24 || slot == 36 {
            // The bank: the previous pass's blocks after the first core block, and its output.
            bank = (blocks[2..].to_vec(), cur.clone());
            blocks = vec!["inp_norm".to_string()];
            cur = "l_out-5".to_string();
        }
        if slot >= 24 {
            let q = weight(&format!("blk.{ip}.loop_msg_query.weight"));
            check(format!("loop_msgs-{slot}"), &bank.0, &dump(&bank.1), &q);
        }
        if ip > 0 {
            let q = weight(&format!("blk.{ip}.attn_res_query.weight"));
            check(format!("attn_res_mix-{slot}"), &blocks, &dump(&cur), &q);
        }
        if ip % 6 == 0 {
            blocks.push(cur.clone());
        }
        let q = weight(&format!("blk.{ip}.ffn_res_query.weight"));
        check(
            format!("ffn_res_mix-{slot}"),
            &blocks,
            &dump(&format!("ffn_inp-{slot}")),
            &q,
        );
        cur = format!("l_out-{slot}");
    }
    check(
        "out_res_mix".to_string(),
        &blocks,
        &dump(&cur),
        &weight("output_res_query.weight"),
    );
    println!("golden depth mix ({checked} mixes: attn_res, ffn_res, loop_msgs, out_res): worst max rel {worst:.2e}");
}

#[test]
fn inkling_tanh_cap_and_row_clip() {
    let fx = need_gpu!("inkling");
    let gpu = fx.dev.gpu();

    // tanh_cap: both branches, the 0.25 seam, saturation, and +-inf-ish inputs.
    let mut x = normalish(0x515, 4096);
    x.iter_mut().skip(2048).for_each(|v| *v *= 40.0);
    x.extend_from_slice(&[
        0.0,
        0.2499999,
        0.25,
        -0.25,
        9.0,
        -9.0,
        1e4,
        -1e4,
        1e-30,
        f32::MAX,
        -f32::MAX,
    ]);
    let n = x.len();
    let xb = ops::upload_f32(gpu, &x).expect("x");
    let ob = breccia_kernels::gemv::alloc(gpu, (n * 4) as u64).expect("out");
    for s in [1.0f32, 30.0, 4.0, 25.0] {
        let act = Act::new(gpu, ActMode::TanhCap).expect("act");
        act.run(&xb, &ob, n, s).expect("tanh_cap");
        let got = ops::download_f32(gpu, &ob, n).expect("read");
        let rel = max_rel(&got, &as_f64(&cpu::tanh_cap(&x, s)));
        assert!(rel <= 1e-6, "tanh_cap s={s}: {rel:e} > 1e-6");
        println!("tanh_cap s={s:<5} n={n} max rel {rel:.2e}");
    }
    gpu.destroy_buf(&xb);
    gpu.destroy_buf(&ob);

    // degenerate: 1 col, 1 row, tau on at pos 0 (tau = 1, clip to thr).
    {
        let x = ops::upload_f32(gpu, &[50.0]).expect("x");
        let p = breccia_kernels::gemv::upload(gpu, &i32_bytes(&[0])).expect("pos");
        let o = breccia_kernels::gemv::alloc(gpu, 4).expect("out");
        RowClip::new(gpu, 1, 1, 8.0, Some((8192.0, 0.1)))
            .expect("row clip 1x1")
            .run(&x, None, Some(&p), &o, 1)
            .expect("row clip 1x1 run");
        let got = ops::download_f32(gpu, &o, 1).expect("read")[0];
        assert!((got - 8.0).abs() < 1e-5, "1x1 clip: {got}");
        for b in [&x, &p, &o] {
            gpu.destroy_buf(b);
        }
    }

    // row clip vs CPU: q (128x24, scale, tau), k (128x8), r (16x24, scale, tau).
    // Positions straddle the tau floor (8192) and reach 1M.
    let pos: Vec<i32> = vec![0, 8190, 8191, 8192, 20000, 131071, 1_000_000];
    let t = pos.len();
    let pb = breccia_kernels::gemv::upload(gpu, &i32_bytes(&pos)).expect("pos");
    for (label, cols, heads, thr, scaled, tau) in [
        ("q", 128usize, 24usize, 22.627417f32, true, true),
        ("k", 128, 8, 22.627417, false, false),
        ("r", 16, 24, 8.0, true, true),
    ] {
        // spread 0.5..8 so some rows clip and some do not
        let mut x = normalish(0x900 ^ cols as u64, cols * heads * t);
        for (i, v) in x.iter_mut().enumerate() {
            *v *= 0.5 + (i / cols % 16) as f32 * 0.5;
        }
        let sc: Vec<f32> = normalish(0x901, cols * heads)
            .iter()
            .map(|v| 1.0 + 0.3 * v)
            .collect();
        let tau_p = tau.then_some((8192.0f32, 0.1f32));
        let want = cpu::row_clip(
            &x,
            cols,
            heads,
            scaled.then_some(&sc[..]),
            thr,
            tau_p.map(|(f, a)| (&pos[..], f, a)),
        );
        let xb = ops::upload_f32(gpu, &x).expect("x");
        let sb = ops::upload_f32(gpu, &sc).expect("scale");
        let ob = breccia_kernels::gemv::alloc(gpu, (x.len() * 4) as u64).expect("out");
        let clip = RowClip::new(gpu, cols, heads, thr, tau_p).expect("row clip");
        clip.run(&xb, scaled.then_some(&sb), tau.then_some(&pb), &ob, t)
            .expect("run");
        let got = ops::download_f32(gpu, &ob, x.len()).expect("read");
        let rel = max_rel(&got, &as_f64(&want));
        assert!(rel <= 1e-5, "row clip {label}: {rel:e} > 1e-5");
        println!("row_clip {label} {cols}x{heads}x{t} max rel {rel:.2e}");
        // in place (out aliases x), as the engine will call it
        clip.run(&xb, scaled.then_some(&sb), tau.then_some(&pb), &xb, t)
            .expect("run in place");
        let got = ops::download_f32(gpu, &xb, x.len()).expect("read");
        let rel = max_rel(&got, &as_f64(&want));
        assert!(rel <= 1e-5, "row clip {label} in place: {rel:e} > 1e-5");
        for b in [&xb, &sb, &ob] {
            gpu.destroy_buf(b);
        }
    }

    let Ok(golden) = std::env::var("BRECCIA_INKLING_GOLDEN") else {
        eprintln!("inkling golden parity: SKIPPED (set BRECCIA_INKLING_GOLDEN)");
        gpu.destroy_buf(&pb);
        return;
    };
    let golden = std::path::PathBuf::from(golden);
    let find = |dir: &str, stem: &str| golden_f32(&golden, dir, stem);
    let tok = 13usize; // golden prompt, positions 0..12 (tau = 1 below 8192)
    let gpos: Vec<i32> = (0..tok as i32).collect();
    let gpb = breccia_kernels::gemv::upload(gpu, &i32_bytes(&gpos)).expect("pos");
    let pipes = [
        RowClip::new(gpu, 128, 24, 22.627417, None).expect("q swa"),
        RowClip::new(gpu, 128, 24, 22.627417, Some((8192.0, 0.1))).expect("q full"),
        RowClip::new(gpu, 128, 8, 22.627417, None).expect("k"),
        RowClip::new(gpu, 16, 24, 8.0, None).expect("r swa"),
        RowClip::new(gpu, 16, 24, 8.0, Some((8192.0, 0.1))).expect("r full"),
    ];
    let mut worst = 0.0f64;
    for slot in 0..48usize {
        let phys = match slot {
            0..=23 => slot,
            _ => 6 + (slot - 24) % 12,
        };
        let full = phys % 6 == 5;
        for (pre, post, scale, pipe) in [
            (
                "Qproj",
                "Qcur",
                Some("attn_q_clip"),
                &pipes[usize::from(full)],
            ),
            ("Kconv", "Kcur", None, &pipes[2]),
            (
                "Rproj",
                "Rcur",
                Some("attn_rel_clip"),
                &pipes[3 + usize::from(full)],
            ),
        ] {
            let x = find("dump-s4", &format!("{pre}-{slot}_"));
            let want = find(
                if post == "Rcur" { "dump-s4" } else { "dump" },
                &format!("{post}-{slot}_"),
            );
            let sc = scale.map(|w| find("weights", &format!("blk.{phys}.{w}.weight_")));
            let xb = ops::upload_f32(gpu, &x).expect("x");
            let sb = sc.as_ref().map(|s| ops::upload_f32(gpu, s).expect("scale"));
            let ob = breccia_kernels::gemv::alloc(gpu, (x.len() * 4) as u64).expect("out");
            pipe.run(
                &xb,
                sb.as_ref(),
                full.then_some(&gpb).filter(|_| post != "Kcur"),
                &ob,
                tok,
            )
            .expect("run");
            let got = ops::download_f32(gpu, &ob, x.len()).expect("read");
            let rel = max_rel(&got, &as_f64(&want));
            assert!(rel <= 1e-5, "golden {post}-{slot}: {rel:e} > 1e-5");
            worst = worst.max(rel);
            gpu.destroy_buf(&xb);
            gpu.destroy_buf(&ob);
            if let Some(b) = sb {
                gpu.destroy_buf(&b);
            }
        }
    }
    println!("golden Qcur/Kcur/Rcur, 48 slots: worst max rel {worst:.2e}");
    gpu.destroy_buf(&gpb);
    gpu.destroy_buf(&pb);
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
