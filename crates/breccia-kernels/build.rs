//! Compile the GLSL compute kernels to SPIR-V at build time and validate them.
//!
//! Same toolchain and pattern as `crates/breccia-vk/build.rs`
//! (docs/PROJECT-SETUP.md): `glslc -O --target-env=vulkan3` -> `spirv-val`.
//! Fails the build loudly if either tool is missing or rejects a shader, and
//! `include_str!` in the crate embeds the validated SPIR-V.
//!
//! Every GEMV shader is compiled **twice**: once as-is (fp32 accumulation, the
//! default) and once with `-DBRECCIA_F16_ACCUM` (fp16 accumulation, selected by
//! `breccia_kernels::Accum::F16`). The variant is a separate pipeline rather
//! than a runtime push-constant branch so the default f32 pipeline is
//! bit-identical to what the earlier phase reports measured
//! (`docs/phase-3/03-gaps-closed.md`).

use std::path::{Path, PathBuf};
use std::process::Command;

/// (shader stem, SPIR-V env var) — one `<stem>.comp` in `shaders/`.
const SHADERS: &[&str] = &[
    "gemv_iq2_xxs",
    "gemv_iq2_xs",
    "gemv_iq2_s",
    "gemv_iq3_xxs",
    "gemv_iq3_s",
    "gemv_iq4_nl",
    "gemv_iq4_xs",
    "gemv_q4_k",
    "gemv_q5_k",
    "gemv_q6_k",
    "gemv_q2_0",
    "gemv_f16",
    "gemv_bf16",
    "gemv_f32",
    // multi-user (chunked-prefill) fused dequant + mat-vector: one expert, m users
    "gemm_iq4_nl",
    "gemm_q2_0",
    "gemm_iq2_xxs",
    "gemm_iq2_xs",
    "gemm_iq2_s",
    "gemm_iq3_xxs",
    "gemm_iq3_s",
    "gemm_iq3_s_q8",
    "gemm_iq3_s_pair",
    "gemm_iq4_xs",
    "gemm_q4_k",
    "gemm_q5_k",
    "gemm_bf16",
    "gemm_q6_k",
    "moe_fanin",
    // batched multi-expert variants (one dispatch over N experts' blobs)
    "gemv_iq2_xxs_moe",
    "gemv_iq2_xs_moe",
    "gemv_iq2_s_moe",
    "gemv_iq3_xxs_moe",
    "gemv_iq3_s_moe",
    "gemv_iq4_nl_moe",
    "gemv_iq4_xs_moe",
    "gemv_q4_k_moe",
    "gemv_q5_k_moe",
    "gemv_q6_k_moe",
    "gemv_q2_0_moe",
    "gemv_f16_moe",
    "gemv_bf16_moe",
    "gemv_f32_moe",
    "gemv_reduce",
    "dequant_iq2_xxs",
    "dequant_iq2_xs",
    "dequant_iq2_s",
    "dequant_iq3_xxs",
    "dequant_iq3_s",
    "dequant_iq4_nl",
    "dequant_iq4_xs",
    "dequant_q4_k",
    "dequant_q5_k",
    "dequant_q6_k",
    "dequant_q2_0",
    "dequant_f16",
    "dequant_bf16",
    "dequant_f32",
    // Phase-3 op kernels (`docs/phase-3/04-op-kernels.md`): elementwise /
    // reduction / gather ops the forward-pass graph needs around the GEMVs.
    "ops_rmsnorm",
    "ops_rope",
    "ops_swiglu",
    "ops_softmax",
    "ops_router",
    "ops_act",
    "ops_add",
    "ops_hc_gated_mean",
    "ops_hc_write",
    // quantised row gather + dequant, one wrapper per type (same decode units as
    // `dequant_<type>.comp`; `src/ops::EmbGather` selects the stem).
    "ops_gather_iq2_xxs",
    "ops_gather_iq2_xs",
    "ops_gather_iq2_s",
    "ops_gather_iq3_xxs",
    "ops_gather_iq3_s",
    "ops_gather_iq4_nl",
    "ops_gather_iq4_xs",
    "ops_gather_q4_k",
    "ops_gather_q5_k",
    "ops_gather_q6_k",
    "ops_gather_q2_0",
    "ops_gather_f16",
    "ops_gather_bf16",
    "ops_gather_f32",
    // GDN / SSM step (`src/gdn`, docs/phase-3/06-attention-gdn.md).
    "gdn_conv",
    "gdn_qk_l2",
    "gdn_gate",
    "gdn_step",
    "gdn_norm_gate",
    // QSA attention + lightning indexer (`src/qsa`, docs/phase-3/06).
    "qsa_q_norm",
    "qsa_kv_append",
    "qsa_idx_tail",
    "qsa_idx_pool",
    "qsa_idx_score",
    "qsa_topk",
    "qsa_attn_scores",
    "qsa_attn_apply",
    // int8-dot decode path (§8bb): the q8_1 quantiser and the IQ3_S dp4a GEMV.
    "ops_q8_quantize",
    "gemv_iq3s_q8",
    // Host-free MoE: device-side expert indirection + the routed combine.
    "moe_map",
    "moe_combine",
];

/// GEMV stems that get the `silu_div` epilogue build (`_hcact`), i.e. the ones
/// the HC mixer's `down` projection can be constructed with: the artifact ships
/// BF16 `hc_*` weights (§14) and the conformance suite uses F32, in both the
/// batched (multi-token) and the single-expert split-K (one token) shape.
/// `breccia_kernels::gemv::has_hc_epilogue` mirrors this list.
const EPILOGUE_SHADERS: &[&str] = &["gemv_bf16", "gemv_bf16_moe", "gemv_f32", "gemv_f32_moe"];

/// GEMV stems that get the coalesced-dequant build (`_wide`, §8ax): the quantised
/// types whose decode unit has a per-element byte-load loop. Listed in the order
/// they matter for one token of this model — `Q5_K` is the head (4.25 ms/token),
/// `IQ4_XS` the shared expert, `IQ3_S`/`IQ2_*` the GDN projections and the routed
/// experts. The fp16/F32/BF16 types already saturate the memory path (98-103 % of
/// DRAM) and have no byte-load loop, so they are not built.
/// GEMV stems with a **word-level** decode build (`_words`, §8bc): four codes per
/// instruction and the codebook in LDS instead of per-element extracts, selects and
/// SSBO lookups. Bit-identical to the incumbent unit by construction.
const WORD_SHADERS: &[&str] = &["gemv_iq4_xs"];

const WIDE_SHADERS: &[&str] = &[
    // The head (Q5_K, 248320x2560, 4.25 ms/token = the worst stage in §8av).
    "gemv_q5_k",
    // Added as their decode units are converted (one at a time, each with its own
    // bit-identity check): IQ4_XS is the shared expert, IQ3_S/IQ2_* the GDN
    // projections and the routed experts.
    "gemv_iq4_xs",
];

/// `(rows per lane group, users per decode)` builds of every plain GEMM; keep in
/// step with `blocked_spv!` in `src/gemm.rs`.
const GEMM_VARIANTS: &[(u32, u32)] = &[(2, 8), (4, 8), (1, 16), (2, 16)];

fn main() {
    // **Every** shader and include, file by file. `rerun-if-changed=shaders` (the
    // directory) only fires when a file is *added or removed*, so editing a
    // `.comp` or `.glsl` in place silently kept the previous SPIR-V — a stale
    // shader that looks like a kernel bug (§8ay: `gdn_gate` ignored a new push
    // constant until this was fixed).
    for dir in ["shaders", "shaders/tests"] {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("comp") | Some("glsl") | Some("spvasm")
            ) {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
    }
    println!("cargo:rerun-if-env-changed=GLSLC");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".to_string());

    // The hand-written `OpSDot` fallback module (§8bb): assembled and validated with
    // the same discipline as the compiled shaders.
    assemble(&out_dir, "shaders/intdot_probe.spvasm");

    for name in SHADERS {
        compile(&glslc, &out_dir, name, name, &[]);
        // fp16-accumulate variant of every fused GEMV (not the reduce pass: it
        // sums fp32 partials by contract).
        if name.starts_with("gemv_") && *name != "gemv_reduce" {
            compile(
                &glslc,
                &out_dir,
                name,
                &format!("{name}_f16acc"),
                &["BRECCIA_F16_ACCUM"],
            );
        }
        // The `down`-GEMV + `Act` epilogue build (HC mixer, `docs/phase-4/01`
        // §8at): the same shader with `silu(acc / s)` applied to the row value
        // just before it is stored. A separate pipeline rather than a runtime
        // branch, so the default pipeline keeps byte-identical SPIR-V (the
        // sanctioned shape for a wired kernel change, `AGENTS.md`).
        // The fused HC `up` + gated-mean build (§8az): the GEMV with the
        // column-blocked row mapping and the gated-mean/inject epilogue. A
        // separate pipeline, so the default build keeps byte-identical SPIR-V.
        // Register-blocked GEMM builds (§12 item 1): `RPL` rows per lane group
        // and/or `UMAX` users per decode, separate pipelines so the default
        // SPIR-V stays byte-identical.
        if name.starts_with("gemm_") && !name.ends_with("_q8") && !name.ends_with("_pair") {
            for (rpl, umax) in GEMM_VARIANTS {
                let mut defs = Vec::new();
                if *rpl > 1 {
                    defs.push(format!("BRECCIA_GEMM_RPL={rpl}u"));
                }
                if *umax != 8 {
                    defs.push(format!("BRECCIA_GEMM_UMAX={umax}u"));
                }
                let defs: Vec<&str> = defs.iter().map(String::as_str).collect();
                compile(&glslc, &out_dir, name, &format!("{name}_r{rpl}u{umax}"), &defs);
            }
        }
        if *name == "gemv_bf16" {
            compile(
                &glslc,
                &out_dir,
                name,
                "gemv_bf16_hcgate",
                &["BRECCIA_HC_GATED_EPILOGUE"],
            );
        }
        // The **coalesced-dequant** build (§8ax): the same GEMV with the
        // per-element byte loads in the decode unit replaced by one wide load per
        // half-unit. Over the whole block the arithmetic and the values are
        // identical (bit-identical conformance), so this is a *wired* change in the
        // sanctioned sense — a separate pipeline rather than a runtime branch, so
        // the default build keeps byte-identical SPIR-V and `BRECCIA_GEMV_WIDE`
        // is a one-flag A/B and revert.
        // The word-level-pack build (§8bc): `& 0x0F0F0F0F`-style extraction and an
        // LDS codebook, i.e. the same arithmetic in fewer instructions.
        if WORD_SHADERS.contains(name) {
            compile(
                &glslc,
                &out_dir,
                name,
                &format!("{name}_words"),
                &["BRECCIA_WORD_PACK"],
            );
        }
        if WIDE_SHADERS.contains(name) {
            compile(
                &glslc,
                &out_dir,
                name,
                &format!("{name}_wide"),
                &["BRECCIA_WIDE_DEQUANT"],
            );
        }
        if EPILOGUE_SHADERS.contains(name) || *name == "gemv_reduce" {
            compile(
                &glslc,
                &out_dir,
                name,
                &format!("{name}_hcact"),
                &["BRECCIA_GEMV_EPILOGUE_SILU_DIV"],
            );
        }
        // The q8_0 V-cache build (§8bg): a *wired* variant — same bindings, same
        // push constants, one `-D` — so the f16 pipelines stay byte-identical
        // SPIR-V and `BRECCIA_KV_Q8V` is a one-flag A/B and revert.
        if KV_Q8V_SHADERS.contains(name) {
            compile(
                &glslc,
                &out_dir,
                name,
                &format!("{name}_q8v"),
                &["BRECCIA_KV_Q8V"],
            );
        }
    }

    spirv_hash(&out_dir);
}

/// FNV-1a 64 over every module in `OUT_DIR`, in file-name order, exported as
/// `BRECCIA_SPIRV_HASH`: the shader half of a `breccia profile --prefill` verdict's
/// staleness key (a shader edit re-asks the per-GPU question; a Rust-only rebuild
/// does not).
fn spirv_hash(out_dir: &Path) {
    let mut names: Vec<PathBuf> = std::fs::read_dir(out_dir)
        .expect("OUT_DIR readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "spv"))
        .collect();
    names.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in &names {
        for b in p.file_name().unwrap().as_encoded_bytes().iter().chain(&std::fs::read(p).expect("a compiled module")) {
            h = (h ^ *b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    println!("cargo:rustc-env=BRECCIA_SPIRV_HASH={h:016x}");
}

/// QSA stems that get the q8_0 **V**-cache build (`BRECCIA_KV_Q8V`, §8bg): the
/// append (quantise on write) and the apply (dequantise in registers inside the
/// accumulate loop). The scores kernel never reads V, so it has no variant.
const KV_Q8V_SHADERS: &[&str] = &["qsa_kv_append", "qsa_attn_apply"];

/// Assemble a hand-written `.spvasm` module with `spirv-as` and validate it, the
/// same two-step the compiled shaders go through (`AGENTS.md`: malformed SPIR-V must
/// abort the host process, never reach the GPU).
fn assemble(out_dir: &Path, src: &str) {
    let spv = out_dir.join(format!(
        "{}.spv",
        Path::new(src)
            .file_stem()
            .expect("a file stem")
            .to_string_lossy()
    ));
    let asm = Command::new("spirv-as")
        .args(["--target-env", "vulkan1.3", src, "-o"])
        .arg(&spv)
        .output()
        .unwrap_or_else(|e| panic!("failed to launch `spirv-as` ({e}); is spirv-tools installed?"));
    if !asm.status.success() {
        panic!(
            "spirv-as failed for {src}:\n{}",
            String::from_utf8_lossy(&asm.stderr)
        );
    }
    let st = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .arg(&spv)
        .output()
        .unwrap_or_else(|e| panic!("failed to launch `spirv-val` ({e})"));
    if !st.status.success() {
        panic!(
            "spirv-val rejected {src}:\n{}",
            String::from_utf8_lossy(&st.stderr)
        );
    }
    println!("cargo:rerun-if-changed={src}");
}

/// `glslc -O ... <src>.comp -o <out>.spv` followed by `spirv-val`.
fn compile(glslc: &str, out_dir: &Path, src_stem: &str, out_stem: &str, defines: &[&str]) {
    let src = format!("shaders/{src_stem}.comp");
    let spv = out_dir.join(format!("{out_stem}.spv"));

    let mut cmd = Command::new(glslc);
    cmd.args([
        "-O",
        "--target-env=vulkan1.3",
        // `#include` support for the shared block decode / body files.
        "-I",
        "shaders",
    ]);
    for d in defines {
        cmd.arg(format!("-D{d}"));
    }
    let compile = cmd
        .arg(&src)
        .arg("-o")
        .arg(&spv)
        .output()
        .unwrap_or_else(|e| panic!("failed to launch `{glslc}` ({e}); is shaderc installed?"));
    if !compile.status.success() {
        panic!(
            "glslc failed for {src} (defines {defines:?}):\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );
    }

    let validate = Command::new("spirv-val")
        .arg(&spv)
        .output()
        .unwrap_or_else(|e| panic!("failed to launch `spirv-val` ({e})"));
    if !validate.status.success() {
        panic!(
            "spirv-val rejected {src} (defines {defines:?}):\n{}",
            String::from_utf8_lossy(&validate.stderr)
        );
    }
}
