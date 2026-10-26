//! Batching conformance: a batch must produce **bit-identical** results to the
//! synchronous path, and must collapse N dispatches into one submission.
//!
//! Opt-in by explicit device, like every GPU test here:
//!
//! ```text
//! timeout 180 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
//!   BRECCIA_KERNEL_DEVICE=0000:0e:00.0 BRECCIA_OP=batch \
//!   cargo test -p breccia-kernels --test batching -- --nocapture
//! ```

use anyhow::Result;
use breccia_kernels::batch_cmd;
use breccia_kernels::ops::{Act, ActMode, Add, RmsNorm};
use breccia_kernels::{Device, Gemv, QuantType};

fn device() -> Option<Device> {
    if std::env::var("BRECCIA_OP").ok().as_deref() != Some("batch") {
        println!("skipping: set BRECCIA_OP=batch");
        return None;
    }
    let sel = std::env::var("BRECCIA_KERNEL_DEVICE").ok();
    Some(Device::open(sel.as_deref()).expect("device"))
}

/// `n` dependent dispatches of three different pipelines, batched, must equal
/// the same sequence run synchronously element for element.
#[test]
fn a_batch_is_bit_identical_to_the_synchronous_path() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let gpu = dev.gpu();
    let n = 2560usize;
    let add = Add::new(gpu)?;
    let scale = Act::new(gpu, ActMode::Silu)?;
    let rms = RmsNorm::new(gpu, n, 1)?;

    let a = breccia_kernels::gemv::upload(
        gpu,
        &breccia_kernels::ops::f32_to_bytes(
            &(0..n).map(|i| (i as f32) * 1e-3 - 1.0).collect::<Vec<_>>(),
        ),
    )?;
    let mut tmp =
        breccia_kernels::gemv::upload(gpu, &breccia_kernels::ops::f32_to_bytes(&vec![0.5f32; n]))?;
    let gamma =
        breccia_kernels::gemv::upload(gpu, &breccia_kernels::ops::f32_to_bytes(&vec![1.0f32; n]))?;
    let sync_out = breccia_kernels::gemv::alloc(gpu, n as u64 * 4)?;
    let batch_out = breccia_kernels::gemv::alloc(gpu, n as u64 * 4)?;

    // reference: 6 dependent dispatches, each its own submit+fence
    add.run(&a, &tmp, &sync_out, n, 1.0)?;
    scale.run(&sync_out, &sync_out, n, 0.0)?;
    rms.run(&sync_out, Some(&gamma), &sync_out, 1)?;
    add.run(&sync_out, &a, &sync_out, n, -0.5)?;
    scale.run(&sync_out, &sync_out, n, 0.0)?;
    rms.run(&sync_out, Some(&gamma), &sync_out, 1)?;

    // the same six, recorded into one command buffer
    batch_cmd::begin(gpu)?;
    add.run(&a, &tmp, &batch_out, n, 1.0)?;
    scale.run(&batch_out, &batch_out, n, 0.0)?;
    rms.run(&batch_out, Some(&gamma), &batch_out, 1)?;
    add.run(&batch_out, &a, &batch_out, n, -0.5)?;
    scale.run(&batch_out, &batch_out, n, 0.0)?;
    rms.run(&batch_out, Some(&gamma), &batch_out, 1)?;
    assert_eq!(batch_cmd::recorded(), 6, "six dispatches must be recorded");
    batch_cmd::end(gpu)?;
    assert!(!batch_cmd::is_open());

    let s = breccia_kernels::ops::download_f32(gpu, &sync_out, n)?;
    let b = breccia_kernels::ops::download_f32(gpu, &batch_out, n)?;
    let mut worst = 0f32;
    for (x, y) in s.iter().zip(b.iter()) {
        worst = worst.max((x - y).abs());
    }
    assert!(
        worst == 0.0,
        "a batch must be bit-identical, worst |Δ| = {worst:e}"
    );
    println!(
        "batch of 6 dependent dispatches: bit-identical ({})",
        dev.info.name
    );

    // a GEMV inside a batch, against the synchronous result
    let gemv = Gemv::new(gpu, QuantType::Bf16, 64, 2560)?;
    let wbuf = breccia_kernels::gemv::upload(gpu, &vec![0x3cu8; (64 * 10 * 2) * 16])?;
    let x = breccia_kernels::gemv::upload(
        gpu,
        &breccia_kernels::ops::f32_to_bytes(&vec![0.25f32; 2560]),
    )?;
    let o_sync = breccia_kernels::gemv::alloc(gpu, 64 * 4)?;
    let o_batch = breccia_kernels::gemv::alloc(gpu, 64 * 4)?;
    gemv.run(&wbuf, &x, &o_sync)?;
    batch_cmd::begin(gpu)?;
    gemv.run(&wbuf, &x, &o_batch)?;
    batch_cmd::end(gpu)?;
    let a1 = breccia_kernels::ops::download_f32(gpu, &o_sync, 64)?;
    let a2 = breccia_kernels::ops::download_f32(gpu, &o_batch, 64)?;
    assert_eq!(a1, a2, "a batched GEMV must match the synchronous one");
    println!("batched GEMV: bit-identical");

    for b in [
        &a, &tmp, &gamma, &sync_out, &batch_out, &wbuf, &x, &o_sync, &o_batch,
    ] {
        gpu.destroy_buf(b);
    }
    let _ = &mut tmp;
    Ok(())
}
