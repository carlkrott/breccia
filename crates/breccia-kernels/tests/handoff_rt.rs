//! G3: what one hot-expert hand-off to a helper card costs (option 7 break-even
//! is 0.07-0.16 ms). Ignored: needs a GPU.
//! `BRECCIA_TEST_DEVICE=0000:0e:00.0 cargo test --release -p breccia-kernels --test handoff_rt -- --ignored --nocapture`
//!
//! Prints medians of: an empty submit+wait; one submit with the activation
//! copy in (10 KiB) and the expert outputs copy out (8 x 2560 f32, 80 KiB),
//! host write/read included (host-cached readback); and one synchronous f32 gemv (640 x 2560).

use breccia_kernels::gemv::{alloc, staging, upload, Gemv};
use breccia_kernels::QuantType;
use breccia_vk::ctx::{Gpu, Host, Qc};
use std::time::Instant;

fn median_us(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2] * 1e6
}

#[test]
#[ignore]
fn handoff_round_trip() {
    let want = std::env::var("BRECCIA_TEST_DEVICE").unwrap_or_default();
    let host = Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let idx = devs.iter().position(|d| d.pci_address == want.trim()).unwrap_or(0);
    let gpu = Gpu::create(&host, &devs[idx], 1).expect("device");
    let (n_in, n_out) = (2560 * 4usize, 8 * 2560 * 4usize);
    let stage_in = staging(&gpu, n_in as u64).unwrap();
    // Host-cached readback, as download_regions does (uncached reads run at ~0.3 GiB/s).
    let hv = ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT;
    let stage_out = gpu
        .buf(n_out as u64, ash::vk::BufferUsageFlags::TRANSFER_DST, hv, hv | ash::vk::MemoryPropertyFlags::HOST_CACHED)
        .unwrap();
    let (x, y) = (alloc(&gpu, n_in as u64).unwrap(), alloc(&gpu, n_out as u64).unwrap());
    let act = vec![1u8; n_in];
    let mut back = vec![0u8; n_out];
    let n = 500;

    let time = |f: &mut dyn FnMut()| -> f64 {
        for _ in 0..20 {
            f();
        }
        median_us((0..n).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).collect())
    };
    let empty = time(&mut || {
        gpu.wall_run(Qc::C, |_, _| {}).unwrap();
    });
    let copies = time(&mut || {
        gpu.write_host(&stage_in, &act).unwrap();
        gpu.wall_run(Qc::C, |d, cmd| unsafe {
            Gpu::barrier_all(d, cmd);
            d.cmd_copy_buffer(cmd, stage_in.buf, x.buf, &[ash::vk::BufferCopy { src_offset: 0, dst_offset: 0, size: n_in as u64 }]);
            Gpu::barrier_all(d, cmd);
            d.cmd_copy_buffer(cmd, y.buf, stage_out.buf, &[ash::vk::BufferCopy { src_offset: 0, dst_offset: 0, size: n_out as u64 }]);
            Gpu::barrier_all(d, cmd);
        })
        .unwrap();
        gpu.read_host(&stage_out, &mut back).unwrap();
    });
    let (rows, cols) = (640, 2560);
    let w = upload(&gpu, &vec![0u8; rows * cols * 4]).unwrap();
    let out = alloc(&gpu, (rows * 4) as u64).unwrap();
    let g = Gemv::new(&gpu, QuantType::F32, rows, cols).unwrap();
    let gemv = time(&mut || g.run(&w, &x, &out).unwrap());
    eprintln!(
        "G3 {}: empty submit {empty:.1} us | copy in 10 KiB + out 80 KiB, one submit {copies:.1} us | f32 gemv 640x2560 sync {gemv:.1} us | hand-off floor ~ {:.1} us (copies + gemv - empty)",
        devs[idx].pci_address,
        copies + gemv - empty
    );
    for b in [&stage_in, &stage_out, &x, &y, &w, &out] {
        gpu.destroy_buf(b);
    }
}
