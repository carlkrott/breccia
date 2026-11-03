//! §6 offset binds: `GemmMany::run_at_off` / `run_pair_at_off` (X and O bound at a
//! byte offset inside a bigger buffer, through a per-batch descriptor set) must be
//! **bit-identical** to `run_at` / `run_pair_at` on standalone buffers. Same kernel
//! choice either way: both take a non-zero weight base, so neither takes the dot path.
//!
//! Opt-in by device (`BRECCIA_KERNEL_DEVICE`, AGENTS.md).

use std::sync::Arc;

use breccia_format::ShardSet;
use breccia_kernels::batch_cmd;
use breccia_kernels::cpu::LayerPlan;
use breccia_kernels::gemm::GemmMany;
use breccia_kernels::gemv::{alloc, upload};
use breccia_kernels::ops::{download_f32, upload_f32};
use breccia_vk::ctx::{Gpu, Host};

const MODEL_GLOB: &str = "/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf";
const W_PREFIX: usize = 4096; // bytes of junk before the expert blob: base_words = 2048

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn offset_binds_match_standalone_buffers() {
    let Ok(sel) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("gemm_off: SKIPPED - set BRECCIA_KERNEL_DEVICE=<pci|index>");
        return;
    };
    let host = Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let idx = devs
        .iter()
        .position(|d| d.pci_address == sel.trim() || d.index.to_string() == sel.trim())
        .unwrap_or(0);
    let gpu: Gpu = Gpu::create(&host, &devs[idx], 1).expect("gpu");
    let set = match ShardSet::open_glob(MODEL_GLOB) {
        Ok(s) => Arc::new(s),
        Err(e) => return eprintln!("gemm_off: skipping, no model: {e}"),
    };
    let layout = set.layout().expect("layout");
    // Block 1 (IQ2 host lane, no pair shader) and block 28 (IQ3_S: has the gate+up pair).
    for blk in [1usize, 28] {
        let plan = LayerPlan::from_block(&set, layout.block(blk).expect("block")).expect("plan");
        let sl = plan.slices(&set, 7).expect("slices");
        let (n_embd, n_ff) = (plan.n_embd, plan.n_ff);
        let gate = GemmMany::new(&gpu, plan.role("gate").ty).expect("gate gemm");
        let down = GemmMany::new(&gpu, plan.role("down").ty).expect("down gemm");

        let with_prefix = |parts: &[&[u8]]| {
            let mut v = vec![0xA5u8; W_PREFIX];
            for p in parts {
                v.extend_from_slice(p);
            }
            upload(&gpu, &v).expect("upload w")
        };
        let wg = with_prefix(&[sl.gate]);
        let wp = with_prefix(&[sl.gate, sl.up]);
        let wd = with_prefix(&[sl.down]);
        let base = (W_PREFIX / 2) as u32;

        let mut seed = 0x9e37_79b9u32;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32) - 0.5
                })
                .collect()
        };

        for users in [4usize, 256] {
            // (rows, cols, gemm, weights, pair)
            let cases: [(&str, usize, usize, &GemmMany, &breccia_vk::ctx::Buf, bool); 3] = [
                ("gate", n_ff, n_embd, &gate, &wg, false),
                ("pair", n_ff, n_embd, &gate, &wp, true),
                ("down", n_embd, n_ff, &down, &wd, false),
            ];
            for (name, rows, cols, g, w, pair) in cases {
                if pair && !g.has_pair() {
                    continue;
                }
                let pad = g.padded_cols(cols);
                let xs = rnd(users * pad);
                // Big buffers: X at slot row 3, O at slot row 5 (256-aligned like the engine's).
                let x_off = (3 * pad * 4).next_multiple_of(256);
                let o_off = (5 * rows * 4).next_multiple_of(256);
                let mut big = vec![0f32; x_off / 4];
                big.extend_from_slice(&xs);
                big.extend(std::iter::repeat_n(0f32, 512));
                let xb = upload_f32(&gpu, &big).expect("x big");
                let xp = upload_f32(&gpu, &xs).expect("x plain");
                let ob =
                    upload_f32(&gpu, &vec![0f32; o_off / 4 + users * rows + 1024]).expect("o big");
                let op = alloc(&gpu, (users * rows * 4) as u64).expect("o plain");

                if pair {
                    g.run_pair_at(w, &xp, &op, rows, cols, users, base)
                        .expect("ref");
                } else {
                    g.run_at(w, &xp, &op, rows, cols, users, base).expect("ref");
                }
                batch_cmd::begin(&gpu).expect("begin");
                let offs = (x_off as u64, o_off as u64);
                let r = if pair {
                    g.run_pair_at_off(w, &xb, &ob, rows, cols, users, base, offs)
                } else {
                    g.run_at_off(w, &xb, &ob, rows, cols, users, base, offs)
                };
                let e = batch_cmd::end(&gpu);
                r.expect("off");
                e.expect("end");

                let want = download_f32(&gpu, &op, users * rows).expect("dl ref");
                let all = download_f32(&gpu, &ob, (o_off / 4) + users * rows + 16).expect("dl off");
                let got = &all[o_off / 4..o_off / 4 + users * rows];
                assert!(
                    all[..o_off / 4].iter().all(|v| *v == 0.0),
                    "{name} m={users}: wrote before o_off"
                );
                assert_eq!(
                    bits(got),
                    bits(&want),
                    "{name} m={users}: offset bind differs"
                );
                println!("gemm_off blk {blk} {name} m={users}: bit-identical ({rows}x{cols})");
                for b in [&xb, &xp, &ob, &op] {
                    gpu.destroy_buf(b);
                }
            }
        }
        for b in [&wg, &wp, &wd] {
            gpu.destroy_buf(b);
        }
    }
}
