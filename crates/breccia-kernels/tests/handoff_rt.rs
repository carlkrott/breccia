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

/// Tier-2 gate: the helper's real work for one layer of a T-row window. `E` distinct
/// IQ3_XXS experts (gate/up 640x2560, down 2560x640) applied to `T` rows each (E*T
/// expert-row pairs sharing E weight blobs), recorded as one segment, plus the
/// hand-off copies (T x 2560 f32 in, T x 2560 f32 partial sums out, one submit).
/// Tier 2 pays only if this beats the host lane's time for the same E experts x T rows.
/// `BRECCIA_TEST_DEVICE=0000:0e:00.0 cargo test --release -p breccia-kernels --test handoff_rt tier2 -- --ignored --nocapture`
#[test]
#[ignore]
fn tier2_helper_layer() {
    use breccia_kernels::batch::map_to_bytes;
    use breccia_kernels::{dispatch_batch, GemvBatch, XLayout};
    let want = std::env::var("BRECCIA_TEST_DEVICE").unwrap_or_default();
    let host = Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let idx = devs.iter().position(|d| d.pci_address == want.trim()).unwrap_or(0);
    let gpu = Gpu::create(&host, &devs[idx], 1).expect("device");
    let ty = QuantType::Iq3Xxs;
    let gu = GemvBatch::new(&gpu, ty, 640, 2560).unwrap();
    let dn = GemvBatch::new(&gpu, ty, 2560, 640).unwrap();
    let emax = 16usize;
    // Zero blocks are valid IQ3_XXS (scale 0): the timing is the bytes and the grid.
    let wg = upload(&gpu, &vec![0u8; (gu.blob_bytes() as usize) * emax]).unwrap();
    let wu = upload(&gpu, &vec![0u8; (gu.blob_bytes() as usize) * emax]).unwrap();
    let wd = upload(&gpu, &vec![0u8; (dn.blob_bytes() as usize) * emax]).unwrap();
    let n = 200;
    let time = |f: &mut dyn FnMut()| -> f64 {
        for _ in 0..10 {
            f();
        }
        median_us((0..n).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).collect())
    };
    let hv = ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT;
    for t in [1usize, 4, 8] {
        let io = (t * 2560 * 4) as u64;
        let stage_in = staging(&gpu, io).unwrap();
        let stage_out = gpu
            .buf(io, ash::vk::BufferUsageFlags::TRANSFER_DST, hv, hv | ash::vk::MemoryPropertyFlags::HOST_CACHED)
            .unwrap();
        let (xin, xout) = (alloc(&gpu, io).unwrap(), alloc(&gpu, io).unwrap());
        let (act, mut back) = (vec![1u8; io as usize], vec![0u8; io as usize]);
        let copies = time(&mut || {
            gpu.write_host(&stage_in, &act).unwrap();
            gpu.wall_run(Qc::C, |d, cmd| unsafe {
                Gpu::barrier_all(d, cmd);
                d.cmd_copy_buffer(cmd, stage_in.buf, xin.buf, &[ash::vk::BufferCopy { src_offset: 0, dst_offset: 0, size: io }]);
                Gpu::barrier_all(d, cmd);
                d.cmd_copy_buffer(cmd, xout.buf, stage_out.buf, &[ash::vk::BufferCopy { src_offset: 0, dst_offset: 0, size: io }]);
                Gpu::barrier_all(d, cmd);
            })
            .unwrap();
            gpu.read_host(&stage_out, &mut back).unwrap();
        });
        for e in [1usize, 2, 4, 8, 16] {
            let pairs = e * t;
            // pair p = (expert p % e, row p / e): rows share their expert's blob.
            let words = |b: &GemvBatch| (0..pairs).map(|p| (p % e) as u32 * b.blob_words()).collect::<Vec<_>>();
            let mg = upload(&gpu, &map_to_bytes(&gu.map_for_blobs(&words(&gu), XLayout::PerExpert))).unwrap();
            let md = upload(&gpu, &map_to_bytes(&dn.map_for_blobs(&words(&dn), XLayout::PerExpert))).unwrap();
            let x = alloc(&gpu, gu.x_bytes(pairs, XLayout::PerExpert)).unwrap();
            let (g, u) = (alloc(&gpu, (pairs * 640 * 4) as u64).unwrap(), alloc(&gpu, (pairs * 640 * 4) as u64).unwrap());
            let d = alloc(&gpu, (pairs * 2560 * 4) as u64).unwrap();
            let work = time(&mut || {
                dispatch_batch::begin(&gpu).unwrap();
                gu.run(&wg, &x, &g, &mg, pairs).unwrap();
                gu.run(&wu, &x, &u, &mg, pairs).unwrap();
                dn.run(&wd, &g, &d, &md, pairs).unwrap();
                dispatch_batch::end_submit(&gpu, None).unwrap();
                dispatch_batch::wait_all(&[&gpu]).unwrap();
            });
            eprintln!(
                "TIER2 {} T={t} E={e:>2}: experts segment {work:7.1} us | copies {copies:6.1} us | layer total ~ {:7.1} us | weight MiB {:.1}",
                devs[idx].pci_address,
                work + copies,
                (2 * gu.blob_bytes() + dn.blob_bytes()) as f64 * e as f64 / (1 << 20) as f64
            );
            for b in [&mg, &md, &x, &g, &u, &d] {
                gpu.destroy_buf(b);
            }
        }
        for b in [&stage_in, &stage_out, &xin, &xout] {
            gpu.destroy_buf(b);
        }
    }
    for b in [&wg, &wu, &wd] {
        gpu.destroy_buf(b);
    }
}
