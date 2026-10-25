//! Compile GLSL compute shaders to SPIR-V at build time and validate them.
//!
//! Uses the system shader toolchain documented in docs/PROJECT-SETUP.md:
//!   glslc -O --target-env=vulkan1.3  ->  spirv-val
//!
//! Fails the build loudly if either tool is missing or rejects a shader.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".to_string());

    let shaders = ["bandwidth", "fma_fp32", "fma_fp16"];
    for name in shaders {
        let src = format!("shaders/{name}.comp");
        let spv = out_dir.join(format!("{name}.spv"));

        let compile = Command::new(&glslc)
            .args(["-O", "--target-env=vulkan1.3",])
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
