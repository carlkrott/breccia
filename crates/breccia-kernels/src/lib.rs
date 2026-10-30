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
//! | [`QuantType`] | the fourteen types with a kernel, in kernel (block) geometry |
//! | [`Gemv`] | device-independent `{type, rows, cols}` + compiled pipeline; [`Gemv::run`] dispatches one decode+GEMV |
//! | [`Accum`] | fp32 (default) or fp16 dot-product accumulation; fp16 is a measured research path, not recommended |
//! | [`dequant_to_f32`] | standalone block dequantiser (same decode path, used by conformance tests and by anything that needs f32 weights) |
//! | [`GemvBatch`] | one dispatch over N routed experts' blobs (row indirection + per-expert `x`), the MoE decode shape |
//! | [`ops`] | the elementwise/utility ops the graph needs around the GEMVs: RMSNorm, IMRoPE, SwiGLU, softmax, router top-k, quantised gather, HC mixer, residual add |
//! | [`gdn`] | the GDN/SSM sublayer (36 layers): conv+SiLU, q/k L2 norm, beta/gate, delta-rule recurrence with in-place state, norm+gate |
//! | [`qsa`] | the QSA sublayer (12 layers): per-head q norm, paged f16 KV, lightning indexer, top-k selection, masked attention with the output gate |
//! | [`proj`] | one projection, two dispatch shapes (decode `Gemv` / batched prefill `GemvBatch`) |
//! | [`reference`] | scalar CPU port of `ggml-quants.c` (decode + dot) used as the conformance oracle |
//! | [`cpu`] | the **host resident-expert lane**: AVX2/scalar fused decode+dot, the expert FFN (SwiGLU + GEMV), the per-token top-k thread pool, and `breccia bench cpu-experts` |
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

pub mod batch;
pub mod bench;
pub mod cpu;
pub mod device;
pub mod dispatch_batch;
pub mod gdn;
pub mod gemm;
pub mod gemv;
pub mod moe;
pub mod ops;
pub mod proj;
pub mod qsa;
pub mod quant;
pub mod real;
pub mod reference;
pub(crate) mod tables;

pub use batch::{GemvBatch, XLayout};

/// Hash of every compiled SPIR-V module in this build (see `build.rs`).
pub const SPIRV_HASH: &str = env!("BRECCIA_SPIRV_HASH");
pub use bench::{run_gemv_bench, BatchCell, GemvBenchOptions, GemvReport, Shape};
pub use device::Device;
pub use dispatch_batch as batch_cmd;
pub use gdn::{Gdn, GdnProjections, GdnState, GdnWeights, GdnWorkspace};
pub use gemv::{dequant_to_f32, Accum, Gemv, GemvTiming};
pub use moe::{MoeCombine, MoeFanIn, MoeMap};
pub use ops::bench::{run_ops_bench, OpKind, OpsBenchOptions, OpsReport};
pub use ops::{HcMixer, HcWeights, HcWorkspace, Q8Quantize, RmsNorm, Rope, RopeTable, Router};
pub use proj::Proj;
pub use qsa::{Qsa, QsaProjections, QsaState, QsaWeights, QsaWorkspace};
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
            ("TBL_IQ2S_LO", crate::tables::TBL_IQ2S_LO),
            ("TBL_IQ2S_HI", crate::tables::TBL_IQ2S_HI),
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
        // The buffer must end exactly after the last table (the IQ2_S grid is
        // the newest one: 1024 u64 entries split into lo/hi u32 words).
        assert_eq!(
            crate::tables::GPU_TABLE_WORDS_LEN,
            crate::tables::TBL_IQ2S_HI + 1024,
            "table buffer length must end after the IQ2_S grid"
        );
    }

    /// Every table accessor in `blocks.glsl` must mask its index to its own
    /// table's length, and the masked range must fit the uploaded buffer.
    ///
    /// This is the static half of the "a bad index must never become an
    /// out-of-bounds SSBO read" guard: an OOB read of a storage buffer is a GPU
    /// page fault (process death, and on a display card a frozen machine), not a
    /// wrong number. The indices are packet bits today, so the masks are
    /// belt-and-braces — but they are what makes that statement checkable.
    #[test]
    fn table_accessors_are_bounded() {
        use crate::tables as t;
        let glsl = include_str!("../shaders/blocks.glsl");
        // (accessor, host base, mask in the shader, entries in the table)
        let checks: [(&str, usize, u32, u32); 10] = [
            ("sign_code", t::TBL_SIGNS, 127, 128),
            ("grid_iq2xxs_lo", t::TBL_IQ2XXS_LO, 255, 256),
            ("grid_iq2xxs_hi", t::TBL_IQ2XXS_HI, 255, 256),
            ("grid_iq2xs_lo", t::TBL_IQ2XS_LO, 511, 512),
            ("grid_iq2xs_hi", t::TBL_IQ2XS_HI, 511, 512),
            ("grid_iq3xxs", t::TBL_IQ3XXS_GRID, 255, 256),
            ("grid_iq3s", t::TBL_IQ3S_GRID, 511, 512),
            ("grid_iq2s_lo", t::TBL_IQ2S_LO, 1023, 1024),
            ("grid_iq2s_hi", t::TBL_IQ2S_HI, 1023, 1024),
            ("iq4nl_value", t::TBL_IQ4NL_VALUES, 15, 16),
        ];
        for (name, base, mask, entries) in checks {
            let needle = format!("{name}(uint ");
            let line = glsl
                .lines()
                .find(|l| l.contains(&needle) && l.trim_start().starts_with(['u', 'f', 'i']))
                .unwrap_or_else(|| panic!("shaders/blocks.glsl lacks the `{name}` accessor"));
            assert!(
                line.contains(&format!("& {mask}u")),
                "{name} must mask its index to {} entries, found: {}",
                entries,
                line.trim()
            );
            assert_eq!(mask + 1, entries, "{name} mask vs table size");
            assert!(
                base + entries as usize <= t::GPU_TABLE_WORDS_LEN,
                "{name}: table [{base}, +{entries}) runs past the uploaded buffer ({})",
                t::GPU_TABLE_WORDS_LEN
            );
        }
    }

    /// No decode unit may turn a float into an index: the scale is data, and a
    /// NaN scale converted to an integer is exactly how a decode kernel becomes
    /// an out-of-bounds memory read. The scale may only ever be multiplied into
    /// a value (`docs/phase-3/02-expert-batching-quants.md` §4.3).
    #[test]
    fn no_float_to_integer_conversion_in_a_decode_unit() {
        let glsl = include_str!("../shaders/blocks.glsl");
        // Scale-carrying float identifiers used by the decode units.
        let floats = ["d", "db", "dl", "dm", "dmin", "sc ", "dv"];
        let mut offenders = Vec::new();
        for (n, line) in glsl.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("").trim();
            if code.is_empty() || code.starts_with('*') {
                continue;
            }
            // `int(x)` / `uint(x)` where x is one of the scale values.
            for f in floats {
                let f = f.trim();
                for cast in ["int(", "uint("] {
                    let pat = format!("{cast}{f})");
                    let pat2 = format!("{cast}{f} ");
                    if code.contains(&pat) || code.contains(&pat2) {
                        offenders.push(format!("line {}: {}", n + 1, code));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "float->integer conversion of a scale value (would make a NaN scale \
             an out-of-bounds index): {offenders:#?}"
        );
    }
}
