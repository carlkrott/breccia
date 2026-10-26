//! Where a *synchronous* dispatch's host time goes.
//!
//! Every kernel entry point in `breccia-kernels` records one dispatch into a
//! command buffer, submits it and waits on a fresh fence (`ops::Pipe::run` →
//! `Gpu::wall_run`). The phase-3 reports all quote **GPU** time from queue
//! timestamps, which is why they measure a 1.5–4.5 µs dispatch floor; the graph
//! pays the *host* round trip instead, once per call, ~1 500 times per token.
//! This example measures that round trip, an allocation, and a 10 KiB transfer,
//! so the forward-pass report can attribute its overhead instead of guessing.
//!
//! ```text
//! timeout 180 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
//!   ./target/release/examples/dispatch_overhead 0000:0e:00.0
//! ```

use anyhow::Result;
use breccia_kernels::ops::{Add, RmsNorm};
use breccia_kernels::{Device, QuantType};
use breccia_vk::ctx::Buf;

fn main() -> Result<()> {
    let sel = std::env::args().nth(1);
    let dev = Device::open(sel.as_deref())?;
    let gpu = dev.gpu();
    println!("device {} [{}]", dev.info.name, dev.info.pci_address);

    // --- 1. a two-buffer elementwise kernel, one submit+wait per call ---------
    let add = Add::new(gpu)?;
    let n = 2560usize;
    let a = breccia_kernels::gemv::alloc(gpu, n as u64 * 4)?;
    let b = breccia_kernels::gemv::alloc(gpu, n as u64 * 4)?;
    let o = breccia_kernels::gemv::alloc(gpu, n as u64 * 4)?;
    let reps = 200;
    // warm
    for _ in 0..5 {
        add.run(&a, &b, &o, n, 1.0)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        add.run(&a, &b, &o, n, 1.0)?;
    }
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "one 2560-wide add       {:>8.2} us/call  (submit + fence wait + 1 dispatch)",
        per * 1e6
    );

    // --- 2. a real projection, same protocol ---------------------------------
    let gemv = breccia_kernels::Gemv::new(gpu, QuantType::Iq4Nl, 2560, 640)?;
    let w = breccia_kernels::gemv::alloc(gpu, gemv.weight_bytes())?;
    let x = breccia_kernels::gemv::alloc(gpu, 640 * 4)?;
    let y = breccia_kernels::gemv::alloc(gpu, 2560 * 4)?;
    for _ in 0..3 {
        gemv.run(&w, &x, &y)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        gemv.run(&w, &x, &y)?;
    }
    let per_g = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "2560x640 IQ4_NL GEMV    {:>8.2} us/call  (1.15 MiB read)",
        per_g * 1e6
    );

    // --- 3. a normalising op (one workgroup per row) -------------------------
    let rms = RmsNorm::new(gpu, 128, 1)?;
    let g = breccia_kernels::gemv::alloc(gpu, 128 * 4)?;
    let xi = breccia_kernels::gemv::alloc(gpu, 48 * 128 * 4)?;
    let xo = breccia_kernels::gemv::alloc(gpu, 48 * 128 * 4)?;
    let src = breccia_kernels::gemv::alloc(gpu, 48 << 10)?;
    for _ in 0..3 {
        rms.run(&xi, Some(&g), &xo, 48)?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        rms.run(&xi, Some(&g), &xo, 48)?;
    }
    let per_r = t0.elapsed().as_secs_f64() / reps as f64;
    println!("rmsnorm 48x128          {:>8.2} us/call", per_r * 1e6);

    // --- 4. a 40 KiB staging readback ---------------------------------------
    let mut st = breccia_engine::xfer::Staging::new(gpu, 64 << 10)?;
    for _ in 0..3 {
        st.read_batch(gpu, &[(&src, 0, 40 << 10)])?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        st.read_batch(gpu, &[(&src, 0, 40 << 10)])?;
    }
    let per_rb = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "40 KiB device→host      {:>8.2} us/round trip",
        per_rb * 1e6
    );

    // --- 5. alloc + free of one 40 KiB device-local buffer -------------------
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let b: Buf = breccia_kernels::gemv::alloc(gpu, 40 << 10)?;
        gpu.destroy_buf(&b);
    }
    let per_a = t0.elapsed().as_secs_f64() / reps as f64;
    println!("alloc+free 40 KiB       {:>8.2} us/call", per_a * 1e6);

    for b in [&a, &b, &o, &w, &x, &y, &g, &xi, &xo, &src] {
        gpu.destroy_buf(b);
    }
    st.release(gpu);
    println!(
        "\nper token the graph issues ~1 500 synchronous dispatches, ~130\n\
         allocations (a QSA workspace per layer per token) and ~200 transfers:\n\
         {:.1} ms + {:.1} ms + {:.1} ms of pure overhead at these rates.",
        1500.0 * per * 1e3,
        130.0 * per_a * 1e3,
        200.0 * per_rb * 1e3
    );
    Ok(())
}
