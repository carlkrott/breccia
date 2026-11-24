//! Device-side MoE indirection: the map `MoeMap` builds must equal the map
//! `GemvBatch::map_for_blobs` builds **byte for byte**, and the device combine
//! must equal the host expression.
//!
//! These are unit-level checks (synthetic blobs, fixed ids and activations), not
//! end-to-end: the first version of `moe_map.comp` wrote one base per *expert*
//! instead of one per *row*, which every end-to-end check reported as "wrong but
//! plausible". A byte diff says which row is wrong instead.
//!
//! ```text
//! timeout 180 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
//!   BRECCIA_KERNEL_DEVICE=0000:0e:00.0 BRECCIA_OP=moe \
//!   cargo test -p breccia-kernels --test moe_indirection -- --nocapture
//! ```

use anyhow::Result;
use breccia_kernels::ops::f32_to_bytes;
use breccia_kernels::{Device, GemvBatch, MoeCombine, MoeMap, QuantType, XLayout};

fn device() -> Option<Device> {
    if std::env::var("BRECCIA_OP").ok().as_deref() != Some("moe") {
        println!("skipping: set BRECCIA_OP=moe");
        return None;
    }
    let sel = std::env::var("BRECCIA_KERNEL_DEVICE").ok();
    Some(Device::open(sel.as_deref()).expect("device"))
}

/// The device map must be identical to `map_for_blobs`, for both layouts and for
/// types with different block geometries (a Q2_0 down role and an IQ2_XS gate).
#[test]
fn device_map_equals_the_host_map() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let gpu = dev.gpu();
    let map_pipe = MoeMap::new(gpu)?;
    let ids: Vec<u32> = vec![7, 0, 511, 3, 129, 64, 255, 1, 400, 42];
    let ids_buf = breccia_kernels::gemv::upload(gpu, &breccia_kernels::ops::u32_to_bytes(&ids))?;

    for (ty, per_expert) in [
        (QuantType::Iq2Xs, false),
        (QuantType::Iq4Nl, true),
        (QuantType::Q2_0, true),
    ] {
        let (rows_per_expert, cols) = if per_expert { (2560, 640) } else { (640, 2560) };
        let batch = GemvBatch::new(gpu, ty, rows_per_expert, cols)?;
        let rows = batch.rows(ids.len());
        let map = breccia_kernels::gemv::alloc(gpu, batch.map_bytes(ids.len()))?;

        // The arena offset used by the engine is 0 for a one-block arena; use a
        // non-zero one here so an off-by-one in `base_words` cannot hide.
        let base = 4_096u32;
        map_pipe.run(&ids_buf, &map, &batch, ids.len(), base, per_expert)?;

        let blobs: Vec<u32> = ids
            .iter()
            .map(|&id| base + id * batch.blob_words())
            .collect();
        let layout = if per_expert {
            XLayout::PerExpert
        } else {
            XLayout::Shared
        };
        let want = batch.map_for_blobs(&blobs, layout);
        let got = breccia_kernels::ops::download_u32(gpu, &map, rows * 2)?;

        assert_eq!(got.len(), want.len(), "{ty}: map length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g,
                w,
                "{ty} ({layout:?}) map[{i}] differs (row {} of {}): device {g} vs host {w}",
                i.min(rows) % rows_per_expert,
                rows_per_expert
            );
        }
        println!(
            "map {ty:>8} {layout:?}: {} entries byte-identical (blob_words {})",
            got.len(),
            batch.blob_words()
        );
        gpu.destroy_buf(&map);
    }
    gpu.destroy_buf(&ids_buf);
    Ok(())
}

/// `run_grouped`: `tokens x k` ids, id `j` reads activation row `j / k` — the host
/// map Inkling builds for a multi-token gate/up batch.
#[test]
fn grouped_map_equals_the_host_map() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let gpu = dev.gpu();
    let map_pipe = MoeMap::new(gpu)?;
    let (k, tokens) = (4usize, 3usize);
    let ids: Vec<u32> = vec![7, 0, 127, 3, 9, 64, 100, 1, 40, 42, 5, 6];
    let ids_buf = breccia_kernels::gemv::upload(gpu, &breccia_kernels::ops::u32_to_bytes(&ids))?;
    let batch = GemvBatch::new(gpu, QuantType::Iq4Nl, 640, 2560)?;
    let rows = batch.rows(ids.len());
    let map = breccia_kernels::gemv::alloc(gpu, batch.map_bytes(ids.len()))?;
    map_pipe.run_grouped(&ids_buf, &map, &batch, ids.len(), k)?;
    let blobs: Vec<u32> = ids.iter().map(|&id| id * batch.blob_words()).collect();
    let mut want = batch.map_for_blobs(&blobs, XLayout::PerExpert);
    let stride = (batch.padded_cols() / 4) as u32;
    for p in 0..tokens * k {
        for r in 0..640 {
            want[rows + p * 640 + r] = stride * (p / k) as u32;
        }
    }
    let got = breccia_kernels::ops::download_u32(gpu, &map, rows * 2)?;
    assert_eq!(got, want[..rows * 2], "grouped map");
    println!("grouped map: {} entries identical", got.len());
    gpu.destroy_buf(&map);
    gpu.destroy_buf(&ids_buf);
    Ok(())
}

/// `MoeCombine` must equal `Σ w_j·in_j + shared·sigmoid(g)` computed on the host.
#[test]
fn device_combine_equals_the_host_expression() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let gpu = dev.gpu();
    let combine = MoeCombine::new(gpu)?;
    let k = 10usize;
    let cols = 2560usize;
    let down: Vec<f32> = (0..k * cols)
        .map(|i| ((i % 97) as f32) * 1e-3 - 0.5)
        .collect();
    let weights: Vec<f32> = (0..k).map(|j| 0.02 + j as f32 * 1e-3).collect();
    let shared: Vec<f32> = (0..cols).map(|i| ((i % 31) as f32) * 1e-2).collect();
    let gate = vec![0.75f32];

    let up = |v: &[f32]| -> Result<breccia_vk::ctx::Buf> {
        breccia_kernels::gemv::upload(gpu, &f32_to_bytes(v))
    };
    let b_down = up(&down)?;
    let b_w = up(&weights)?;
    let b_sh = up(&shared)?;
    let b_g = up(&gate)?;
    let b_out = breccia_kernels::gemv::alloc(gpu, cols as u64 * 4)?;
    combine.run(&b_down, &b_w, &b_sh, &b_g, &b_out, k, cols)?;
    let got = breccia_kernels::ops::download_f32(gpu, &b_out, cols)?;

    let g = 1.0f64 / (1.0 + (-(gate[0] as f64)).exp());
    let mut worst = 0f64;
    for i in 0..cols {
        let mut acc = 0f64;
        for j in 0..k {
            acc += (weights[j] as f64) * (down[j * cols + i] as f64);
        }
        let want = acc + (shared[i] as f64) * g;
        worst = worst.max((want - got[i] as f64).abs());
    }
    assert!(
        worst < 1e-5,
        "combine differs from the host expression: {worst:e}"
    );
    println!("combine: worst |Δ| vs host = {worst:.3e}");

    for b in [&b_down, &b_w, &b_sh, &b_g, &b_out] {
        gpu.destroy_buf(b);
    }
    Ok(())
}

/// Tier 2: `run_pairs` = the host map with each pair's activation row taken from the pair.
#[test]
fn device_pair_map_equals_the_host_map() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let gpu = dev.gpu();
    let map_pipe = MoeMap::new(gpu)?;
    let pairs: Vec<(u32, u32)> = vec![(7, 2), (0, 0), (511, 3), (3, 1), (129, 2)];
    let flat: Vec<u32> = pairs.iter().flat_map(|&(s, x)| [s, x]).collect();
    let ids_buf = breccia_kernels::gemv::upload(gpu, &breccia_kernels::ops::u32_to_bytes(&flat))?;
    for (rows_per_expert, cols) in [(640, 2560), (2560, 640)] {
        let batch = GemvBatch::new(gpu, QuantType::Iq2Xs, rows_per_expert, cols)?;
        let rows = batch.rows(pairs.len());
        let map = breccia_kernels::gemv::alloc(gpu, batch.map_bytes(pairs.len()))?;
        map_pipe.run_pairs(&ids_buf, &map, &batch, pairs.len())?;
        let blobs: Vec<u32> = pairs.iter().map(|&(s, _)| s * batch.blob_words()).collect();
        let mut want = batch.map_for_blobs(&blobs, XLayout::PerExpert);
        let st = (batch.padded_cols() / 4) as u32;
        for i in 0..rows {
            want[rows + i] = st * pairs[i / rows_per_expert].1;
        }
        let got = breccia_kernels::ops::download_u32(gpu, &map, rows * 2)?;
        assert_eq!(got, want, "pair map {rows_per_expert}x{cols}");
        gpu.destroy_buf(&map);
    }
    gpu.destroy_buf(&ids_buf);
    Ok(())
}
