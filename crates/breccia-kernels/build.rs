//! Compile the GLSL compute kernels to SPIR-V at build time and validate them.
//!
//! Same toolchain and pattern as `crates/breccia-vk/build.rs`
//! (docs/PROJECT-SETUP.md): `glslc -O --target-env=vulkan1.3` -> `spirv-val`.
//! Fails the build loudly if either tool is missing or rejects a shader, and
//! `include_str!` in the crate embeds the validated SPIR-V.

use std::path::PathBuf;
use std::process::Command;

/// (shader stem, SPIR-V env var) — one `<stem>.comp` in `shaders/`.
const SHADERS: &[&str] = &[
    "gemv_iq2_xxs",
    "gemv_iq2_xs",
    "gemv_iq3_xxs",
    "gemv_iq3_s",
    "gemv_iq4_nl",
    "gemv_q5_k",
    "gemv_q2_0",
    "gemv_bf16",
    "gemv_f32",
    "gemv_reduce",
    "dequant_iq2_xxs",
    "dequant_iq2_xs",
    "dequant_iq3_xxs",
    "dequant_iq3_s",
    "dequant_iq4_nl",
    "dequant_q5_k",
    "dequant_q2_0",
    "dequant_bf16",
    "dequant_f32",
];

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".to_string());

    for name in SHADERS {
        let src = format!("shaders/{name}.comp");
        let spv = out_dir.join(format!("{name}.spv"));

        let compile = Command::new(&glslc)
            .args([
                "-O",
                "--target-env=vulkan1.3",
                // `#include` support for the shared block decode / body files.
                "-I",
                "shaders",
            ])
            .arg(&src)
            .arg("-o")
            .arg(&spv)
            .output()
            .unwrap_or_else(|e| panic!("failed to launch `{glslc}` ({e}); is shaderc installed?"));
        if !compile.status.success() {
            panic!(
                "glslc failed for {src}:\n{}",
                String::from_utf8_lossy(&compile.stderr)
            );
        }

        let validate = Command::new("spirv-val")
            .arg(&spv)
            .output()
            .unwrap_or_else(|e| panic!("failed to launch `spirv-val` ({e})"));
        if !validate.status.success() {
            panic!(
                "spirv-val rejected {src}:\n{}",
                String::from_utf8_lossy(&validate.stderr)
            );
        }
    }
}
