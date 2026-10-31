//! Device-side expert bucketing (`BRECCIA_GPU_DISPATCH`): the lists `MoeBucket` builds
//! must equal the host's `pairs.sort_unstable()` lists **exactly** (slot map, per-slot
//! token list, per-expert offsets), and `MoeGatherRows` must reproduce the host's row
//! gather bit for bit. Equal lists + equal gather inputs = bit-identical logits, since
//! the fan-in only reads the slot map.
//!
//! Never run in CI (needs a GPU):
//!
//! ```text
//! timeout 180 env BRECCIA_KERNEL_DEVICE=0000:0e:00.0 BRECCIA_OP=moe \
//!   cargo test --release -p breccia-kernels --test moe_bucket -- --nocapture
//! ```

use anyhow::Result;
use breccia_kernels::ops::{download_f32, download_u32, f32_to_bytes, u32_to_bytes};
use breccia_kernels::{bucket_host, groups_from_offsets, Device, MoeBucket, MoeGatherRows};

fn device() -> Option<Device> {
    if std::env::var("BRECCIA_OP").ok().as_deref() != Some("moe") {
        println!("skipping: set BRECCIA_OP=moe");
        return None;
    }
    let sel = std::env::var("BRECCIA_KERNEL_DEVICE").ok();
    Some(Device::open(sel.as_deref()).expect("device"))
}

/// Router-like ids: `k` distinct experts per token (xorshift, fixed seed).
fn routing(tokens: usize, k: usize, n_expert: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut ids = Vec::with_capacity(tokens * k);
    for _ in 0..tokens {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < k {
            let e = (next() % n_expert as u64) as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
    }
    ids
}

#[test]
fn device_lists_equal_the_host_lists() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let gpu = dev.gpu();
    let bucket = MoeBucket::new(gpu)?;
    let gather = MoeGatherRows::new(gpu)?;
    let (k, ne, cols) = (10usize, 512usize, 2560usize);
    // Skewed routing too (one hot expert in every token): the worst case for a group.
    for (t, seed, hot) in [
        (1usize, 3u64, false),
        (37, 5, false),
        (512, 7, false),
        (1024, 11, false),
        (1024, 13, true),
    ] {
        let mut ids = routing(t, k, ne, seed);
        if hot {
            for row in ids.chunks_mut(k) {
                if !row.contains(&5) {
                    row[0] = 5;
                }
            }
        }
        let (slots, groups, distinct) = bucket_host(&ids, k, ne, 1024)?;
        let users: Vec<u32> = groups
            .iter()
            .flat_map(|g| g.3.iter().map(|&u| u as u32))
            .collect();

        let b_ids = breccia_kernels::gemv::upload(gpu, &u32_to_bytes(&ids))?;
        let b_slots = breccia_kernels::gemv::alloc(gpu, (t * k * 4) as u64)?;
        let b_tok = breccia_kernels::gemv::alloc(gpu, (t * k * 4) as u64)?;
        let b_off = breccia_kernels::gemv::alloc(gpu, ((ne + 2) * 4) as u64)?;
        bucket.run(&b_ids, &b_slots, &b_tok, &b_off, t, k, ne)?;
        let got_slots = download_u32(gpu, &b_slots, t * k)?;
        let got_tok = download_u32(gpu, &b_tok, t * k)?;
        let off = download_u32(gpu, &b_off, ne + 2)?;

        assert_eq!(
            got_slots, slots,
            "t={t}: slot map differs from the host sort"
        );
        assert_eq!(
            got_tok, users,
            "t={t}: token list differs from the host groups"
        );
        assert_eq!(off[ne] as usize, t * k, "t={t}: pair count");
        assert_eq!(off[ne + 1], 0, "t={t}: out-of-range count");
        let (dg, dd) = groups_from_offsets(&off[..=ne], 1024);
        let strip = |g: &[breccia_kernels::BucketGroup]| {
            g.iter().map(|g| (g.0, g.1, g.2)).collect::<Vec<_>>()
        };
        assert_eq!(dd, distinct, "t={t}: distinct experts");
        assert_eq!(
            strip(&dg),
            strip(&groups),
            "t={t}: groups from the device offsets"
        );

        // The row gather, group by group, against the host's copy of the same rows.
        let src: Vec<f32> = (0..t * cols)
            .map(|i| ((i * 31 % 1009) as f32) * 1e-3 - 0.4)
            .collect();
        let b_src = breccia_kernels::gemv::upload(gpu, &f32_to_bytes(&src))?;
        let mu_max = groups.iter().map(|g| g.2).max().unwrap_or(1);
        let b_dst = breccia_kernels::gemv::alloc(gpu, (mu_max * cols * 4) as u64)?;
        for (e, base, mu, us) in groups.iter().take(8) {
            gather.run(&b_src, &b_tok, &b_dst, cols, *base, *mu)?;
            let got = download_f32(gpu, &b_dst, mu * cols)?;
            for (u, &tk) in us.iter().enumerate() {
                assert_eq!(
                    got[u * cols..(u + 1) * cols],
                    src[tk * cols..(tk + 1) * cols],
                    "t={t} expert {e} user {u}: gathered row differs"
                );
            }
        }
        println!(
            "t={t:>5} hot={hot}: slots/tok/off identical, {distinct} experts, {} groups",
            groups.len()
        );
        for b in [&b_ids, &b_slots, &b_tok, &b_off, &b_src, &b_dst] {
            gpu.destroy_buf(b);
        }
    }
    Ok(())
}
