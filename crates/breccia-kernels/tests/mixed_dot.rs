//! Price `OpFDot2MixAcc32VALVE` (`SPV_VALVE_mixed_float_dot_product`: two f16 MACs
//! into an f32 accumulator) against an f32 `Fma` on this hardware.
//!
//! Why this exists: glslang/glslc 16.6 cannot emit the extension (probed:
//! "extension not supported"), but SPIRV-Tools carries the opcode and RADV
//! advertises the extension on every device here, so hand assembly is the only way
//! to answer "is `v_dot2_f32_f16` actually 2x the fp32 rate" — the gate for using
//! packed f16 dots with an fp32 accumulator in the GEMM bodies.
//!
//! # Result (2026-11-01)
//!
//! Both kernels at ~peak instruction rate, work linear in the trip count (4x trips
//! = 4.0x work), dispatch overhead (~30 us) measured at 0 trips and subtracted:
//!
//! | device | f32 fma | mixed dot | MAC ratio |
//! |---|---|---|---|
//! | RX 6950 XT (gfx1030) | 13.19 T instr/s | 11.16 T instr/s (22.33 T MAC/s) | **1.69x** |
//! | Pro VII / MI50 (gfx906) | 6.23 T instr/s | 5.55 T instr/s (11.10 T MAC/s) | **1.78x** |
//!
//! So `v_dot2_f32_f16` is a real ~1.7-1.8x MAC lever on both cards with an **f32
//! accumulator** — i.e. the f16 speedup without the precision loss that made
//! `Accum::F16` unusable. GPU compute for the GEMM family is ~75 % of prefill, so
//! Amdahl caps the prefill win at ~1.45x before any other change.
//!
//! # Design (each rule is the fix for a measured trap)
//!
//! * **Runtime-bounded loop** (`OpLoopMerge`, trip count from a push constant).
//!   An unrolled straight-line chain gets folded: with a constant trip count the
//!   compiler proves the result and deletes the body, which reads as an
//!   impossibly fast kernel. A trip count the compiler cannot see keeps the work.
//! * **Non-zero seed and operands derived from the invocation id.** A chain seeded
//!   at 0 with `va = h(acc)` is a fixed point (`h(0) = 0`) and is deleted outright;
//!   the read-back was all zeros until the seed came from a runtime value.
//! * **One instruction per step in both kernels**, so the comparison is rate-for-rate:
//!   `fma` does 1 MAC, `dot` does 2, and equal time then means 2x MAC throughput.
//! * The SSBO struct needs `OpDecorate %S Block`, and the dot kernel needs
//!   `OpCapability Float16` as well as the VALVE capability; RADV **segfaults**
//!   rather than reporting either violation.
//! * The device extension must be enabled at creation (`ctx.rs` does it when
//!   advertised) — otherwise using the capability is undefined behaviour.
//!
//! `BRECCIA_KERNEL_DEVICE=<pci|index>` selects the device; `BRECCIA_MIXED_DOT_TRIPS`
//! sets the trip count (default 256). Skips without a device or without `spirv-as`.

use std::process::Command;

use ash::vk;
use breccia_vk::ctx::Qc;

const ACC: usize = 8;
const WORKGROUPS: u32 = 8192;
const GROUP: u32 = 64;

fn trips() -> u32 {
    std::env::var("BRECCIA_MIXED_DOT_TRIPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256)
}

/// One kernel. `kind` is "dot" or "fma".
fn emit(kind: &str) -> String {
    let dot = kind == "dot";
    let mut s = String::from("OpCapability Shader\n");
    if dot {
        s.push_str("OpCapability DotProductFloat16AccFloat32VALVE\n");
        s.push_str("OpCapability Float16\n");
        s.push_str("OpExtension \"SPV_VALVE_mixed_float_dot_product\"\n");
    } else {
        s.push_str("%glsl = OpExtInstImport \"GLSL.std.450\"\n");
    }
    s.push_str("OpMemoryModel Logical GLSL450\n");
    s.push_str("OpEntryPoint GLCompute %main \"main\" %pcv %data %gid\n");
    s.push_str(&format!("OpExecutionMode %main LocalSize {GROUP} 1 1\n"));
    s.push_str("OpDecorate %pc Block\nOpMemberDecorate %pc 0 Offset 0\n");
    s.push_str("OpDecorate %data DescriptorSet 0\nOpDecorate %data Binding 0\n");
    s.push_str("OpDecorate %arr ArrayStride 4\nOpDecorate %S Block\n");
    s.push_str("OpMemberDecorate %S 0 Offset 0\n");
    s.push_str("OpDecorate %gid BuiltIn GlobalInvocationId\n");
    s.push_str("%void = OpTypeVoid\n%fn = OpTypeFunction %void\n%f32 = OpTypeFloat 32\n");
    s.push_str("%u32 = OpTypeInt 32 0\n%bool = OpTypeBool\n%v3u32 = OpTypeVector %u32 3\n");
    s.push_str("%u32_0 = OpConstant %u32 0\n%u32_1 = OpConstant %u32 1\n");
    s.push_str("%f0 = OpConstant %f32 0.0\n%f1 = OpConstant %f32 1.0009765625\n");
    s.push_str("%f2 = OpConstant %f32 1.001953125\n");
    s.push_str("%arr = OpTypeRuntimeArray %f32\n%S = OpTypeStruct %arr\n");
    s.push_str("%pc = OpTypeStruct %u32\n%ptr_pc = OpTypePointer PushConstant %pc\n");
    s.push_str("%ptr_pc_u32 = OpTypePointer PushConstant %u32\n");
    s.push_str("%ptr = OpTypePointer StorageBuffer %S\n");
    s.push_str("%ptr_f32 = OpTypePointer StorageBuffer %f32\n");
    s.push_str("%ptr_v3u32 = OpTypePointer Input %v3u32\n");
    s.push_str("%pcv = OpVariable %ptr_pc PushConstant\n");
    s.push_str("%data = OpVariable %ptr StorageBuffer\n%gid = OpVariable %ptr_v3u32 Input\n");
    if dot {
        s.push_str("%half = OpTypeFloat 16\n%v2half = OpTypeVector %half 2\n");
        s.push_str("%h3 = OpConstant %half 3.0\n%h4 = OpConstant %half 4.0\n");
        s.push_str("%vb = OpConstantComposite %v2half %h3 %h4\n");
    }

    // prologue
    s.push_str("%main = OpFunction %void None %fn\n%entry = OpLabel\n");
    s.push_str("%nptr = OpAccessChain %ptr_pc_u32 %pcv %u32_0\n");
    s.push_str("%n = OpLoad %u32 %nptr\n");
    s.push_str("%g = OpLoad %v3u32 %gid\n%gx = OpCompositeExtract %u32 %g 0\n");
    s.push_str("%gx1 = OpIAdd %u32 %gx %u32_1\n%fx = OpConvertUToF %f32 %gx1\n");
    s.push_str("%mul = OpFMul %f32 %fx %f1\n");
    if dot {
        // operand derived from the invocation id, loop-invariant, non-zero seed
        s.push_str("%hx = OpFConvert %half %fx\n");
        s.push_str("%va = OpCompositeConstruct %v2half %hx %hx\n");
    }
    for a in 0..ACC {
        s.push_str(&format!("%seed{a} = OpFAdd %f32 %fx %f0\n"));
    }
    s.push_str("OpBranch %header\n");

    // header: phis for the trip counter and every accumulator. The back-edge parent
    // is `%continue`; the body's results dominate it, which is what makes the phi
    // legal.
    s.push_str("%header = OpLabel\n");
    s.push_str("%i = OpPhi %u32 %u32_0 %entry %inext %continue\n");
    for a in 0..ACC {
        s.push_str(&format!(
            " %acc{a} = OpPhi %f32 %seed{a} %entry %b{a} %continue\n"
        ));
    }
    // OpLoopMerge must be the second-to-last instruction of the header
    s.push_str("%done = OpUGreaterThanEqual %bool %i %n\n");
    s.push_str("OpLoopMerge %merge %continue None\n");
    s.push_str("OpBranchConditional %done %merge %body\n");

    // body: one op per accumulator
    s.push_str("%body = OpLabel\n");
    for a in 0..ACC {
        let new = format!("%b{a}");
        let src = format!("%acc{a}");
        if dot {
            s.push_str(&format!(
                "{new} = OpFDot2MixAcc32VALVE %f32 %va %vb {src}\n"
            ));
        } else {
            s.push_str(&format!(
                "{new} = OpExtInst %f32 %glsl Fma {src} %mul %f2\n"
            ));
        }
    }
    s.push_str("OpBranch %continue\n");

    // continue: increment the counter and branch back (the header phis take the
    // body's results with %continue as their parent).
    s.push_str("%continue = OpLabel\n");
    s.push_str("%inext = OpIAdd %u32 %i %u32_1\n");
    s.push_str("OpBranch %header\n");

    // merge: reduce and store
    s.push_str("%merge = OpLabel\n");
    let mut acc = "%acc0".to_string();
    for a in 1..ACC {
        s.push_str(&format!("%m{a} = OpFAdd %f32 {acc} %acc{a}\n"));
        acc = format!("%m{a}");
    }
    s.push_str("%idx = OpAccessChain %ptr_f32 %data %u32_0 %gx\n");
    s.push_str(&format!("OpStore %idx {acc}\n"));
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
    // Validate as Vulkan too: the assembler accepts things the driver segfaults on.
    let valid = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&spv)
        .status()
        .ok()?
        .success();
    if !valid {
        eprintln!("mixed-dot: {kind} module rejected by spirv-val (Vulkan)");
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
    let gpu = breccia_vk::ctx::Gpu::create(&host, &devs[idx], 1).expect("device");

    let dir = std::env::temp_dir().join("breccia-mixed-dot");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (Some(dot_spv), Some(fma_spv)) = (assemble("dot", &dir), assemble("fma", &dir)) else {
        eprintln!("mixed-dot: SKIPPED — spirv-as missing or module rejected");
        return;
    };

    let out =
        breccia_kernels::gemv::alloc(&gpu, (WORKGROUPS as u64) * GROUP as u64 * 4).expect("alloc");

    let mut res = Vec::new();
    for (name, spv) in [("fma", &fma_spv), ("dot", &dot_spv)] {
        let k = match gpu.kernel(spv, 1) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("mixed-dot: {name} pipeline REJECTED by the driver: {e}");
                return;
            }
        };
        // Linearity check first: if the time does not scale with the trip count the
        // body is still being folded and every ratio below is meaningless.
        let mut ms = Vec::new();
        for t in [0, trips(), trips() * 4] {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                gpu.bind_kernel(&k, &[out.buf]).expect("bind");
                let pc = t.to_le_bytes();
                let secs = gpu
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
                        d.cmd_push_constants(cmd, k.layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
                        d.cmd_dispatch(cmd, WORKGROUPS, 1, 1);
                    })
                    .expect("dispatch");
                best = best.min(secs);
            }
            ms.push(best * 1e3);
        }
        // 4x the trips must cost ~4x the *work* (absolute time includes the fixed
        // dispatch overhead, which is why the absolute ratio looks sub-linear).
        let scale = (ms[2] - ms[0]) / (ms[1] - ms[0]);
        // The dispatch itself costs ~30 us at this grid, so the per-trip work is the
        // slope, not the absolute time: fit it from the 0-trip and 4x arms.
        let work_ms = ms[2] - ms[0];
        let per_trip = work_ms / (4.0 * trips() as f64);
        println!(
            "mixed-dot {name:>3}: trips 0/{:<5}/{:<5} -> {:.4}/{:.4}/{:.4} ms \
             (overhead {:.4} ms, {:.5} ms per trip, work x{scale:.2} for 4x)",
            trips(),
            trips() * 4,
            ms[0],
            ms[1],
            ms[2],
            ms[0],
            per_trip
        );
        let instr = (WORKGROUPS as f64) * (GROUP as f64) * (4.0 * trips() as f64) * (ACC as f64);
        res.push((name, work_ms, instr / (work_ms * 1e-3) / 1e12, scale));
    }

    // read back: proves the body actually ran
    for (name, _, _, _) in &res {
        let _ = name;
    }
    let vals = breccia_kernels::ops::download_f32(&gpu, &out, 4).expect("readback");
    println!("mixed-dot: first outputs {vals:?}");

    let (fma_name, fma_ms, fma_g, fma_scale) = (res[0].0, res[0].1, res[0].2, res[0].3);
    let (dot_name, dot_ms, dot_g, dot_scale) = (res[1].0, res[1].1, res[1].2, res[1].3);
    println!(
        "mixed-dot: {fma_name} {fma_g:.2} T instr/s ({fma_g:.2} T MAC/s) | \
         {dot_name} {dot_g:.2} T instr/s ({:.2} T MAC/s) | MAC ratio {:.2}x",
        dot_g * 2.0,
        (dot_g * 2.0) / fma_g
    );
    if !(3.4..=4.6).contains(&fma_scale) || !(3.4..=4.6).contains(&dot_scale) {
        println!(
            "mixed-dot: NOT VALID — work does not scale linearly (x{fma_scale:.2}/x{dot_scale:.2} \
             for 4x trips); no ratio is claimed"
        );
    }
    let _ = (fma_ms, dot_ms);
    gpu.destroy_buf(&out);
}

/// Answer (measured, gfx1030, RADV): **no**. With runtime `f16vec2` operands the
/// driver emits two `v_fma_mix_f32` (f16 x f16 -> f32, one per half) per dot and a
/// `v_add_f32` — i.e. fp32 rate, no 2x. With constant operands it folds the whole
/// thing to a single `v_fmac_f32`. So the 1.7x in `mixed_float_dot_vs_f32_fma`
/// requires the explicit `OpFDot2MixAcc32VALVE` instruction; a plain GLSL shader
/// cannot reach it. Porting it into a GEMM therefore means hand-written SPIR-V or
/// a SPIR-V post-pass, not a shader edit.
///
/// Does the driver emit `v_dot2_f32_f16` from ordinary GLSL? Compiles the plain
/// `f16vec2` dot pattern `acc += float(a.x)*float(b.x) + float(a.y)*float(b.y)`
/// and creates the pipeline; run with `RADV_DEBUG=asm` and grep the dump for
/// `v_dot2` to see the ISA. This decides whether the GEMM port can be a shader
/// change or has to be hand-assembled SPIR-V.
#[test]
fn glsl_dot_isa() {
    let Ok(selector) = std::env::var("BRECCIA_KERNEL_DEVICE") else {
        eprintln!("glsl_dot_isa: SKIPPED — set BRECCIA_KERNEL_DEVICE");
        return;
    };
    let src = r#"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Out { float o[]; };
layout(std430, binding = 1) readonly buffer In { uint w[]; };
layout(push_constant) uniform P { uint n; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    float fx = float(gid + 1u);
    // runtime operands: constants let the compiler fold the whole dot away
    f16vec2 a = unpackFloat2x16(w[gid * 2u]);
    f16vec2 b = unpackFloat2x16(w[gid * 2u + 1u]);
    float acc0 = fx, acc1 = fx, acc2 = fx, acc3 = fx;
    for (uint i = 0u; i < n; ++i) {
        acc0 += float(a.x) * float(b.x) + float(a.y) * float(b.y);
        acc1 += float(a.x) * float(b.x) + float(a.y) * float(b.y);
        acc2 += float(a.x) * float(b.x) + float(a.y) * float(b.y);
        acc3 += float(a.x) * float(b.x) + float(a.y) * float(b.y);
    }
    o[gid] = acc0 + acc1 + acc2 + acc3;
}
"#;
    let dir = std::env::temp_dir().join("breccia-mixed-dot");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let comp = dir.join("isa.comp");
    let spv = dir.join("isa.spv");
    std::fs::write(&comp, src).expect("write glsl");
    let ok = Command::new("glslc")
        .arg("--target-env=vulkan1.3")
        .arg("-O")
        .arg(&comp)
        .arg("-o")
        .arg(&spv)
        .status()
        .ok()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("glsl_dot_isa: glslc failed");
        return;
    }
    let bytes = std::fs::read(&spv).expect("spv");
    let host = breccia_vk::ctx::Host::new().expect("host");
    let devs = host.devices().expect("devices");
    let idx = devs
        .iter()
        .position(|d| d.pci_address == selector.trim() || d.index.to_string() == selector.trim())
        .unwrap_or(0);
    let gpu = breccia_vk::ctx::Gpu::create(&host, &devs[idx], 1).expect("device");
    match gpu.kernel(&bytes, 1) {
        Ok(_) => {
            println!("glsl_dot_isa: pipeline created (check the RADV_DEBUG=asm dump for v_dot2)")
        }
        Err(e) => println!("glsl_dot_isa: pipeline REJECTED: {e}"),
    }
}
