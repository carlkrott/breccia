//! `breccia-kernels` — Phase 3 M2 kernels: quant block decode and the fused
//! dequant + matrix-vector multiply that carries batch-1 (decode) inference.
//!
//! Vulkan is the primary backend (2026-10-26 decision), so every kernel here is
//! a GLSL compute shader compiled to SPIR-V by `build.rs` (`glslc` + `spirv-val`,
//! the same toolchain as `breccia-vk`) and driven through `breccia_vk::ctx`'s
//! instance/device/queue plumbing — this crate does not create a Vulkan device
//! of its own.
//!
//! # Pieces
//!
//! | Item | What it does |
//! |---|---|
//! | [`QuantType`] | the nine types with a kernel, in kernel (block) geometry |
//! | [`Gemv`] | device-independent `{type, rows, cols}` + compiled pipeline; [`Gemv::run`] dispatches one decode+GEMV |
//! | [`dequant_to_f32`] | standalone block dequantiser (same decode path, used by conformance tests and by anything that needs f32 weights) |
//! | [`reference`] | scalar CPU port of `ggml-quants.c` (decode + dot) used as the conformance oracle |
//! | [`bench`] | `breccia bench gemv`: throughput vs bytes-read, plus a real-model mode that runs the kernel on an expert tensor straight out of the GGUF mapping |
//!
//! # Shape contract
//!
//! Weights are packed blocks with `rows` rows of `cols` values; a row stores
//! `ceil(cols / block_elems)` whole blocks (`breccia-format` rejects
//! non-block-aligned tensors, so the tail path exists for tests and for future
//! paged weights, not for the shipped model). The activation vector `x` is
//! padded by the host to a multiple of 8 elements with zeros.
//!
//! # Attribution
//!
//! Decode semantics are ported from llama.cpp `ggml-quants.c` and the GPU
//! structuring from `ggml-vulkan` (both MIT, commit 6050b46 — the qwen4exp
//! fork, which is the one that defines the fork-only `Q2_0` block). The shaders
//! carry per-file attribution headers; [`tools/gen_tables.py`] regenerates the
//! shared quant tables from `ggml-common.h`.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod bench;
pub mod device;
pub mod gemv;
pub mod quant;
pub mod real;
pub mod reference;
pub(crate) mod tables;

pub use bench::{run_gemv_bench, GemvBenchOptions, GemvReport, Shape};
pub use device::Device;
pub use gemv::{dequant_to_f32, Gemv, GemvTiming};
pub use quant::QuantType;

#[cfg(test)]
mod tests {
    /// The shader reads the quant tables from a storage buffer with hard-coded
    /// word offsets; the host fills that buffer from `tables::GPU_TABLE_WORDS`.
    /// If either side drifts, every IQ decode silently reads the wrong grid —
    /// so the two are compared here, at build time, from the shader source.
    #[test]
    fn shader_table_offsets_match_the_host_layout() {
        let glsl = include_str!("../shaders/blocks.glsl");
        let checks = [
            ("TBL_SIGNS", crate::tables::TBL_SIGNS),
            ("TBL_IQ2XXS_LO", crate::tables::TBL_IQ2XXS_LO),
            ("TBL_IQ2XXS_HI", crate::tables::TBL_IQ2XXS_HI),
            ("TBL_IQ2XS_LO", crate::tables::TBL_IQ2XS_LO),
            ("TBL_IQ2XS_HI", crate::tables::TBL_IQ2XS_HI),
            ("TBL_IQ3XXS_GRID", crate::tables::TBL_IQ3XXS_GRID),
            ("TBL_IQ3S_GRID", crate::tables::TBL_IQ3S_GRID),
            ("TBL_IQ4NL_VALUES", crate::tables::TBL_IQ4NL_VALUES),
        ];
        for (name, want) in checks {
            let needle = format!("#define {name} ");
            let line = glsl
                .lines()
                .find(|l| l.trim_start().starts_with(&needle))
                .unwrap_or_else(|| panic!("shaders/blocks.glsl lacks `{needle}`"));
            let value = line
                .trim()
                .trim_start_matches(&needle)
                .trim_end_matches('u');
            assert_eq!(
                value.parse::<usize>().expect("numeric offset"),
                want,
                "{name} in shaders/blocks.glsl disagrees with tables::{name}"
            );
        }
        let words = crate::tables::GPU_TABLE_WORDS_LEN;
        assert_eq!(
            words,
            crate::tables::TBL_IQ4NL_VALUES + 16,
            "table buffer length must end after the IQ4_NL values"
        );
    }
}
