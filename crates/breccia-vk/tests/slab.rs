//! Slab sub-allocation: small buffers share blocks, mapped ones get disjoint
//! pointers, and every block is freed once its last buffer goes (any order).
use ash::vk;
use breccia_vk::ctx::{Gpu, Host, SLAB_BYTES};

#[test]
fn slabs_share_blocks_and_free_in_any_order() {
    let Ok(sel) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("slab: SKIPPED - set BRECCIA_KERNEL_DEVICE=<pci|index>");
        return;
    };
    let host = Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let idx = devs
        .iter()
        .position(|d| d.pci_address == sel.trim() || d.index.to_string() == sel.trim())
        .unwrap_or(0);
    let gpu: Gpu = Gpu::create(&host, &devs[idx], 1).expect("gpu");
    let dl = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
    // Device-only: enough 300 KiB buffers for >= 3 blocks.
    let n = (3 * SLAB_BYTES / (300 << 10) + 1) as usize;
    let mut bufs: Vec<_> = (0..n)
        .map(|_| gpu.buf((300 << 10) - 7, usage, dl, dl).expect("device buf"))
        .collect();
    let blocks = gpu.slab_count();
    assert!(blocks >= 3, "{n} small buffers used {blocks} blocks");
    // Mapped: 12 KiB each, all written before any is checked.
    let mapped: Vec<_> = (0..2000u32)
        .map(|i| {
            let (b, p) = gpu
                .buf_mapped(12 << 10, usage, hv, dl | hv)
                .expect("mapped buf");
            assert!(!p.is_null());
            unsafe { std::ptr::write_bytes(p, i as u8, 12 << 10) };
            (b, p, i as u8)
        })
        .collect();
    for (_, p, v) in &mapped {
        let s = unsafe { std::slice::from_raw_parts(*p, 12 << 10) };
        assert!(s.iter().all(|x| x == v), "mapped buffers overlap");
    }
    assert!(gpu.slab_count() > blocks);
    // Free in a scrambled order.
    bufs.extend(mapped.into_iter().map(|(b, _, _)| b));
    let len = bufs.len();
    for k in 0..len {
        let b = bufs.swap_remove((k * 7919) % bufs.len());
        gpu.destroy_buf(&b);
    }
    assert_eq!(gpu.slab_count(), 0, "a block outlived its buffers");
}
