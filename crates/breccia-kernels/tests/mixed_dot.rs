//! Price `OpFDot2MixAcc32VALVE` (`SPV_VALVE_mixed_float_dot_product`) against an
//! f32 `Fma` on this hardware.
//!
//! Why this exists: glslang/glslc 16.6 cannot emit the VALVE extension (probed:
//! "extension not supported"), but SPIRV-Tools carries the opcode and RADV
//! advertises the extension on every device here. So the only way to answer "is
//! `v_dot2_f32_f16` actually 2x the fp32 FMA rate on this silicon" — the gate for
//! using packed f16 dots with an **fp32 accumulator** in the GEMM bodies — is to
//! assemble the kernel by hand and time it.
//!
//! The two kernels are generated here (not committed as blobs): identical shape,
//! 8 independent accumulators (throughput, not latency), `OPS` instructions each.
//! One instruction is 2 f16 MACs into f32, the other 1 f32 MAC, so *equal time*
//! means 2x MAC throughput for the dot.
//!
//! `BRECCIA_KERNEL_DEVICE=<pci|index>` selects the device; skips without it or
//! without `spirv-as` on PATH.
//!
//! # Status: the harness is proven, the *timing* is not yet valid
//!
//! What this already establishes (2026-11-01, RX 6950 XT):
//! - `spirv-as` assembles the instruction and `spirv-val` accepts the module when
//!   `OpExtension "SPV_VALVE_mixed_float_dot_product"` and the capability are
//!   present (the extension string has no `shader_` in it).
//! - **RADV accepts and runs `OpFDot2MixAcc32VALVE`** — a pipeline is created and a
//!   dispatch completes, so the extension is implemented, not advertised-and-stubbed.
//!   The device extension must be *enabled* at device creation; using the capability
//!   without that is undefined behaviour.
//! - A missing `OpDecorate %S Block` on the SSBO struct makes RADV **segfault**
//!   instead of reporting VUID-StandaloneSpirv-OpTypeRuntimeArray-04680.
//!
//! What is NOT valid yet: the timing. The accumulator chains are affine
//! (`acc = fma(mul, c, acc)` repeated), so LLVM strength-reduces them and the
//! measured time does not scale with `BRECCIA_MIXED_DOT_OPS` (8 -> 256 ops is
//! 0.037 -> 0.038 ms). The fix is a non-affine operand — e.g. feed each step's
//! f16 pair from the previous accumulator, or load it from memory per step — and
//! then re-check that time scales linearly in OPS before believing any ratio.

use std::process::Command;

use ash::vk;
use breccia_vk::ctx::Qc;

const ACC: usize = 8;
fn ops() -> usize {
    std::env::var("BRECCIA_MIXED_DOT_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}
const WORKGROUPS: u32 = 32768;
const GROUP: u32 = 64;

fn emit(kind: &str) -> String {
    let dot = kind == "dot";
    let mut s = String::from("OpCapability Shader\n");
    if dot {
        s.push_str("OpCapability DotProductFloat16AccFloat32VALVE\n");
        s.push_str("OpExtension \"SPV_VALVE_mixed_float_dot_product\"\n");
    } else {
        s.push_str("%glsl = OpExtInstImport \"GLSL.std.450\"\n");
    }
    s.push_str("OpMemoryModel Logical GLSL450\n");
    s.push_str("OpEntryPoint GLCompute %main \"main\" %data %gid\n");
    s.push_str(&format!("OpExecutionMode %main LocalSize {GROUP} 1 1\n"));
    s.push_str("OpDecorate %data DescriptorSet 0\nOpDecorate %data Binding 0\n");
    s.push_str(
        "OpDecorate %arr ArrayStride 4\nOpDecorate %S Block\nOpMemberDecorate %S 0 Offset 0\n",
    );
    s.push_str("OpDecorate %gid BuiltIn GlobalInvocationId\n");
    s.push_str("%void = OpTypeVoid\n%fn = OpTypeFunction %void\n%f32 = OpTypeFloat 32\n");
    s.push_str("%u32 = OpTypeInt 32 0\n%v3u32 = OpTypeVector %u32 3\n");
    s.push_str("%u32_0 = OpConstant %u32 0\n%f0 = OpConstant %f32 0.0\n");
    s.push_str("%f1 = OpConstant %f32 1.0009765625\n%f2 = OpConstant %f32 1.001953125\n");
    s.push_str("%arr = OpTypeRuntimeArray %f32\n%S = OpTypeStruct %arr\n");
    s.push_str("%ptr = OpTypePointer StorageBuffer %S\n");
    s.push_str("%ptr_f32 = OpTypePointer StorageBuffer %f32\n");
    s.push_str("%ptr_v3u32 = OpTypePointer Input %v3u32\n");
    s.push_str("%data = OpVariable %ptr StorageBuffer\n%gid = OpVariable %ptr_v3u32 Input\n");
    if dot {
        s.push_str("%half = OpTypeFloat 16\n%v2half = OpTypeVector %half 2\n");
        s.push_str("%h3 = OpConstant %half 3.0\n%h4 = OpConstant %half 4.0\n");

        s.push_str("%vb = OpConstantComposite %v2half %h3 %h4\n");
    }
    s.push_str("%main = OpFunction %void None %fn\n%entry = OpLabel\n");
    s.push_str("%g = OpLoad %v3u32 %gid\n%gx = OpCompositeExtract %u32 %g 0\n");
    // Operands derived from the invocation id: with constants the whole chain
    // folds away at compile time (measured: 1e9 "instructions" in 0.04 ms).
    s.push_str("%fx = OpConvertUToF %f32 %gx\n%mul = OpFMul %f32 %fx %f1\n");
    if dot {
        s.push_str("%hx = OpFConvert %half %fx\n");
        s.push_str("%va = OpCompositeConstruct %v2half %hx %hx\n");
    }
    let mut cur: Vec<String> = (0..ACC)
        .map(|a| {
            s.push_str(&format!("%init{a} = OpCopyObject %f32 %f0\n"));
            format!("%init{a}")
        })
        .collect();
    let mut n = 0usize;
    for _ in 0..ops() {
        for a in 0..ACC {
            let new = format!("%d{n}");
            n += 1;
            if dot {
                s.push_str(&format!(
                    "{new} = OpFDot2MixAcc32VALVE %f32 %va %vb {}\n",
                    cur[a]
                ));
            } else {
                s.push_str(&format!(
                    "{new} = OpExtInst %f32 %glsl Fma %mul %f2 {}\n",
                    cur[a]
                ));
            }
            cur[a] = new;
        }
    }
    let mut acc = cur[0].clone();
    for (a, c) in cur.iter().enumerate().skip(1) {
        s.push_str(&format!("%s{a} = OpFAdd %f32 {acc} {c}\n"));
        acc = format!("%s{a}");
    }
    s.push_str(&format!(
        "%idx = OpAccessChain %ptr_f32 %data %u32_0 %gx\nOpStore %idx {acc}\n"
    ));
    s.push_str("OpReturn\nOpFunctionEnd\n");
    s
}

fn assemble(kind: &str, dir: &std::path::Path) -> Option<Vec<u8>> {
    let asm = dir.join(format!("{kind}.spvasm"));
    let spv = dir.join(format!("{kind}.spv"));
    std::fs::write(&asm, emit(kind)).ok()?;
    let ok = Command::new("spirv-as")
        .arg(&asm)
        .arg("-o")
        .arg(&spv)
        .status()
        .ok()?
        .success();
    if !ok {
        return None;
    }
    std::fs::read(&spv).ok()
}

#[test]
fn mixed_float_dot_vs_f32_fma() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("mixed-dot: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let host = breccia_vk::ctx::Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let want = selector.trim();
    let idx = devs
        .iter()
        .position(|d| d.pci_address == want || d.index.to_string() == want)
        .unwrap_or(0);
    let dev = breccia_vk::ctx::Gpu::create(&host, &devs[idx], 1).expect("device");
    let gpu = &dev;
    let dir = std::env::temp_dir().join("breccia-mixed-dot");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let only = std::env::var("BRECCIA_MIXED_DOT_ONLY").unwrap_or_default();
    let (Some(dot_spv), Some(fma_spv)) = (assemble("dot", &dir), assemble("fma", &dir)) else {
        eprintln!("mixed-dot: SKIPPED — spirv-as missing or rejected the module");
        return;
    };

    // 64 lanes x WORKGROUPS invocations, one output float each.
    let out =
        breccia_kernels::gemv::alloc(gpu, (WORKGROUPS as u64) * GROUP as u64 * 4).expect("alloc");

    eprintln!("mixed-dot: buffers allocated, assembling done");
    let mut times = Vec::new();
    for (name, spv) in [("dot", &dot_spv), ("fma", &fma_spv)] {
        if !only.is_empty() && !only.split(',').any(|k| k == name) {
            continue;
        }
        let k = match gpu.kernel(spv, 1) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("mixed-dot: {name} pipeline REJECTED by the driver: {e}");
                return;
            }
        };
        eprintln!("mixed-dot: {name} pipeline created; dispatching");
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            gpu.bind_kernel(&k, &[out.buf]).expect("bind");
            let t = gpu
                .wall_run(Qc::C, |d, cmd| unsafe {
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipe);
                    d.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        k.layout,
                        0,
                        &[k.set],
                        &[],
                    );
                    d.cmd_dispatch(cmd, WORKGROUPS, 1, 1);
                })
                .expect("dispatch");
            eprintln!("mixed-dot: {name} dispatch ok in {t:.4} s");
            best = best.min(t);
        }
        let instr = (WORKGROUPS as f64) * (GROUP as f64) * (ops() * ACC) as f64;
        times.push((name, best, instr / best / 1e9));
        println!(
            "mixed-dot {name:>3}: {:>8.3} ms for {:.3}e9 instructions -> {:.1} G instr/s",
            best * 1e3,
            instr / 1e9,
            instr / best / 1e9
        );
    }
    let (dot_name, dot_s, dot_g) = times[0];
    let (fma_name, fma_s, fma_g) = times[1];
    let mac_ratio = (dot_g * 2.0) / fma_g;
    println!(
        "mixed-dot: {dot_name} {dot_g:.1} G instr/s ({:.1} G MAC/s, {dot_s:.3} ms) vs \
         {fma_name} {fma_g:.1} G instr/s ({fma_g:.1} G MAC/s, {fma_s:.3} ms) -> \
         MAC ratio {mac_ratio:.2}x",
        dot_g * 2.0,
    );
    gpu.destroy_buf(&out);
}
