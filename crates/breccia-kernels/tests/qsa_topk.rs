//! `qsa_topk_rank.comp` selection is exact at long-context block counts, and cheap.
//!
//! The rank kernel prunes two ways: a block below `THR[t]` (a proven lower bound on
//! the `want`-th largest score, `qsa_topk_thr.comp`) can never be selected, and a
//! block whose partial rank has reached `want` can never be selected either. Both
//! are only safe if the emitted list is identical to the CPU reference's, which this
//! checks at the block counts where the kernel is actually used (16k blocks = a
//! 64k-token cache), including the cases a prune is most likely to get wrong: exact
//! ties, a boundary made entirely of ties (index order decides), dead blocks, and a
//! block count that is not a multiple of the workgroup width.
//!
//! `BRECCIA_KERNEL_DEVICE=<pci|index>` selects the device. Runs in well under a
//! second, so it is the fast loop for selection changes — no prefill required.
//!
//! Measured cost (RX 6950 XT, batched): 63 us at 1024 blocks, 221 us at 4096, 859 us
//! at 16384 — **linear** in blocks, not quadratic in compares. The score-row sweeps
//! are latency-bound, which is why pruning compares changed the time by nothing and
//! why the next step is compaction (fewer blocks ranked), not a cheaper compare.

use breccia_kernels::gemv::upload_into;
use breccia_kernels::ops::download_u32;
use breccia_kernels::qsa::{Qsa, QsaWorkspace};
use breccia_vk::ctx::{Gpu, Host};

const DEAD: f32 = -1.0e29;

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

fn check(gpu: &Gpu, qsa: &Qsa<'_>, ws: &QsaWorkspace, scores: &[f32], pos0: usize, what: &str) {
    let mut bytes = Vec::with_capacity(scores.len() * 4);
    for s in scores {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    upload_into(gpu, &ws.idx_scores, &bytes).expect("upload scores");
    qsa.topk(ws, pos0).expect("topk");
    let got = download_u32(gpu, &ws.ids, ws.max_ids).expect("ids");
    let count = download_u32(gpu, &ws.counts, 1).expect("count")[0] as usize;
    let as_f64: Vec<f64> = scores.iter().map(|&v| v as f64).collect();
    let own = pos0 / 4;
    let want = breccia_kernels::qsa::cpu::topk_blocks(
        &as_f64,
        ws.topk_blocks,
        if own < ws.n_blocks { Some(own) } else { None },
    );
    assert_eq!(
        count,
        want.len().min(ws.max_ids),
        "{what}: count {} vs reference {}",
        count,
        want.len()
    );
    assert_eq!(
        &got[..count],
        &want[..count],
        "{what}: selection differs from the CPU reference"
    );

    // Dispatch cost at this block count: a handful of passes over the score row,
    // not the prefill that an end-to-end run would need.
    // Batched: one submission for all passes, which is how the engine records a
    // token. The unbatched number includes a fence wait per dispatch, i.e. submit
    // overhead rather than kernel time.
    const REPS: u32 = 20;
    breccia_kernels::dispatch_batch::begin(gpu).expect("begin");
    for _ in 0..REPS {
        qsa.topk(ws, pos0).expect("topk timing");
    }
    let t0 = std::time::Instant::now();
    breccia_kernels::dispatch_batch::end(gpu).expect("end");
    let us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;
    println!(
        "qsa_topk: {:<22} n_blocks {:>6} count {:>4}  exact  {us:>7.1} us",
        what, ws.n_blocks, count
    );
}

fn run_for_cells(gpu: &Gpu, qsa: &Qsa<'_>, cells: usize) {
    let ws = qsa.workspace(gpu, 1, cells).expect("workspace");
    let n = ws.n_blocks;
    let mut seed = 0x9e37_79b9u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 8) & 0xffff) as f32 / 65536.0
    };

    // Distinct values: the ordinary case.
    let distinct: Vec<f32> = (0..n).map(|_| next()).collect();
    check(gpu, qsa, &ws, &distinct, 0, "distinct");

    // A boundary made entirely of ties: only `want` of these equal scores fit, so
    // the lower-index rule alone decides the emitted order.
    let mut tied = vec![0.0f32; n];
    for v in tied.iter_mut().take(100) {
        *v = 1.0;
    }
    check(gpu, qsa, &ws, &tied, 0, "ties-at-the-boundary");

    // Dead blocks interspersed: they must never be selected, and they must not
    // disturb the pruning of the live ones.
    let mut dead = distinct.clone();
    for (i, v) in dead.iter_mut().enumerate() {
        if i % 3 == 0 {
            *v = DEAD;
        }
    }
    check(gpu, qsa, &ws, &dead, 0, "dead-blocks");

    // Every score equal: rank(b) == b, so exactly the first `want` blocks win.
    check(gpu, qsa, &ws, &vec![0.5f32; n], 0, "all-equal");

    // A non-zero pos0 exercises the own-block append path.
    check(gpu, qsa, &ws, &distinct, 4 * (n - 3), "own-block-appended");
}

#[test]
fn topk_rank_pruning_is_exact_at_long_context() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("qsa_topk: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let Some((_host, gpu)) = open(&selector) else {
        eprintln!("qsa_topk: no device for {selector}");
        return;
    };
    let qsa = Qsa::new(&gpu, 1 << 17).expect("qsa pipelines");
    // 65536 cells = 16384 blocks (a 64k-token cache), then a count that is not a
    // multiple of the 256-wide rank workgroup.
    for cells in [65536usize, 65521] {
        run_for_cells(&gpu, &qsa, cells);
    }
}
