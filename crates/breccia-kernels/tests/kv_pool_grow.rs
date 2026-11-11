//! `KvPool::grown` (serve P0.5): the old pool's bytes land at the same offsets of the
//! bigger pool (page numbers stay valid) and the new tail is zero.
//! `BRECCIA_KERNEL_DEVICE=<pci|index>` selects the device.

use breccia_kernels::gemv::{download, upload_into};
use breccia_kernels::qsa::{KvPool, PAGE_CELLS};
use breccia_vk::ctx::{Gpu, Host};

fn open(selector: &str) -> Option<(Host, Gpu)> {
    let host = Host::new().ok()?;
    let devs = host.devices().ok()?;
    let want = selector.trim();
    let idx = devs.iter().position(|d| d.pci_address == want || d.index.to_string() == want).unwrap_or(0);
    let gpu = Gpu::create(&host, &devs[idx], 1).ok()?;
    Some((host, gpu))
}

#[test]
fn grown_keeps_pages_and_zeroes_the_tail() {
    let Some((_host, gpu)) = open(&std::env::var("BRECCIA_KERNEL_DEVICE").unwrap_or_default()) else {
        eprintln!("no Vulkan device, skipped");
        return;
    };
    let old = KvPool::new(&gpu, 4 * PAGE_CELLS).expect("pool");
    let pattern = |n: u64, salt: u32| -> Vec<u8> { (0..n as u32 / 4).flat_map(|i| (i.wrapping_mul(2654435761) ^ salt).to_le_bytes()).collect() };
    let (pk, pv) = (pattern(old.k.size, 1), pattern(old.v.size, 2));
    upload_into(&gpu, &old.k, &pk).expect("upload k");
    upload_into(&gpu, &old.v, &pv).expect("upload v");
    let new = old.grown(&gpu, 10 * PAGE_CELLS).expect("grow");
    assert_eq!((old.pages, new.pages), (4, 10));
    for (buf, pat) in [(&new.k, &pk), (&new.v, &pv)] {
        let got = download(&gpu, buf, buf.size as usize).expect("download");
        assert_eq!(&got[..pat.len()], &pat[..], "old pages moved");
        assert!(got[pat.len()..].iter().all(|&b| b == 0), "tail not zeroed");
    }
    old.release(&gpu);
    new.release(&gpu);
}

/// Grow stall at 8 x 200k scale, one QSA layer (x12 for the model):
/// `BRECCIA_KV_HOST=1 cargo test --release --test kv_pool_grow -- --ignored --nocapture`.
#[test]
#[ignore]
fn grow_time_200k() {
    let Some((_host, gpu)) = open(&std::env::var("BRECCIA_KERNEL_DEVICE").unwrap_or_default()) else { return };
    let t = std::time::Instant::now();
    let old = KvPool::new(&gpu, 4 * 204800).expect("pool");
    let t_new = t.elapsed();
    let t = std::time::Instant::now();
    let new = old.grown(&gpu, 8 * 204800).expect("grow");
    let t_grow = t.elapsed();
    let t = std::time::Instant::now();
    old.release(&gpu);
    eprintln!(
        "per layer: new(4x200k, {} MiB) {:.3} s, grown(8x200k, {} MiB) {:.3} s, release old {:.3} s",
        (old.k.size + old.v.size) >> 20, t_new.as_secs_f64(), (new.k.size + new.v.size) >> 20, t_grow.as_secs_f64(), t.elapsed().as_secs_f64()
    );
    new.release(&gpu);
}
