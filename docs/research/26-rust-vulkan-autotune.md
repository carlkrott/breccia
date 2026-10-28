# 26 — Rust/SPIR-V and mobile-Vulkan autotune: what reaches Breccia

Round 26, 2026-10-28. Scope: a per-device autotuned workgroup/tile table for the IQ3_XXS, IQ4_XS and Q6_K GEMV and for `gdn_step`, and bit-parity-safe fusion, checked against Rust and Vulkan compute stacks (Burn, CubeCL, MNN, TNN, rust-gpu, krnl, strix-rt). Standing rules: no GPU work, no model loads, no benchmarks, no commits.

Citation convention: a bare path is in the base tree (bbffb06), and line numbers are base-tree lines unless the cite is `branch@SHA`. `branch@SHA` names a commit that is not in the base tree. Upstream projects are cited as org/repo @ SHA (fetched 2026-10-28). Every speed or capacity number is quoted from a repo doc or commit (cited), re-derived here and marked **derived**, or marked **UNVERIFIED** when it needs a GPU run.

## 0. Verdict

Ranked cards. Each has a CPU gate (done or not) and a GPU gate (not run). Acceptance numbers are UNVERIFIED until measured in-repo.

1. **Card A — gdn_step: adopt and measure the 32-lane shared-memory reduction (`7272338`, off base).** Derived state traffic falls from 339.7 MB to 226.5 MB per token (×2/3), and workgroups rise from 48 to 1,536. The floor at 825 GB/s is 275 µs, against 2,768 µs measured on MI50 (`docs/BRIEF.md:279`). The reduction order changes, so the result is **not bit-identical**. One accept rule: MI50 gdn-scan ≤ 2,516 µs and the accuracy contract passes, including 32/32 greedy tokens.
2. **Card C — gate+up+SwiGLU fusion, adopt and measure from history (`ed7ebd3`, a WIP commit).** The fused shader exists only on `wip/four-gemms-crash-20261026`, in a commit that crashed the MI50 and is marked DO NOT MERGE. Its per-block sum, accumulator split and butterfly match the production GEMV body (derived from source, not run). Its epilogue does not match production `breccia_silu`, so a one-line fix is needed before the parity gate. The measured 3.1×/5.3× is Q4_K on RX only (research 06 §2.2). The whole-token size is about 1.6% of drain (derived; premise contested, C12).
3. **Card B — per-device tuned ROWS_PER_WG through specialization constants, measured rather than ALU-pruned.** The expected gain is small (UNVERIFIED). With LANES fixed at 32, changing ROWS_PER_WG leaves each row's arithmetic unchanged (bit-identical, derived). HC variants need ROWS_PER_WG to be a multiple of `hc`. Geometry alone has not moved quantised GEMV in this repo (doc 06 flat within 5%; `9f1ad74` flat in M el/ms across bpw).

Fallback, demoted: the base kernel at 64 threads per workgroup, which stays bit-identical. Use it only if Card A fails the accuracy contract. Its ceiling is 1.25× on MI50 (derived from CU occupancy, UNVERIFIED).

Two corrections to the plan going in. (i) The round-12 ALU/DRAM classification and its IQ4_XS constant are stale, so they cannot gate Card B (see §5, C5). (ii) Four commits I first treated as branch-only are already in the base tree: `6bbcc6a`, `9f1ad74`, `2883d0a`, `c4f4292`. The IQ4_XS baseline is therefore the word-packed kernel, not the per-element kernel that round-12's constants describe.

## 1. What the base tree has

| Item | Base status | Lane |
|---|---|---|
| gdn_step: one thread per state column, serial 128-term reduction over `i`, state read twice per token | `crates/breccia-kernels/shaders/gdn_step.comp:43, 66–73`; `crates/breccia-kernels/src/gdn/mod.rs:409, 430` (`grid_for(MODEL_H_V*MODEL_S, 128)` = 48 WGs) | MI50, RX |
| IQ4_XS word-level nibble extract, 16-entry codebook in LDS, on by default, bit-identical (`6bbcc6a`) | `crates/breccia-kernels/src/gemv.rs:1080–1093` (`word_pack()`, `gemv_words_spv`; `BRECCIA_NO_GEMV_WORDS=1` reverts) | MI50 (2.02×), RX (1.38×) |
| Wide (coalesced-dequant) decode for Q5_K and IQ4_XS (`9f1ad74`, bit-identical) | `crates/breccia-kernels/src/gemv.rs:1096–1107`; `crates/breccia-kernels/build.rs:134, 236` | MI50, RX |
| Int8-dot IQ3_S path (`2883d0a`, `c4f4292`), off by default, accuracy fails | `crates/breccia-kernels/src/gemv.rs:103–112` (`BRECCIA_GEMV_DP4A`); `crates/breccia-kernels/shaders/gemv_iq3s_q8.comp:8, 18, 93` | MI50, RX (not usable as is) |
| ROWS_PER_WG 8 and LANES 32 in the GEMV body | Host: `crates/breccia-kernels/src/gemv.rs:887` (`pub const ROWS_PER_WG: usize = 8`), used by `grid_for` at `:465, 582, 637, 656, 844` and `wgs = rows.div_ceil(ROWS_PER_WG)` at `:522`. Shader: `crates/breccia-kernels/shaders/gemv_body.glsl:58` (`local_size_x = 256`), `:60` (`LANES 32u`), `:61` (`ROWS_PER_WG 8u`). Separate definitions that Card B does not touch: `gemm_body.glsl:40` (`ROWS_PER_WG 8u`), `ops_gather.glsl:61` (`ROWS_PER_WG (256u / LANES)`, derived from LANES) | MI50, RX |
| Split-K factor, CU count hard-coded to 60 | `crates/breccia-kernels/src/gemv.rs:502–503` (`recommended_splits` → `recommended_splits_for(60)`), `:516–531`; callers `crates/breccia-kernels/src/real.rs:440`, `crates/breccia-kernels/src/bench.rs:610, 692`, `crates/breccia-kernels/src/ops/mod.rs:1775` (`down_splits`), `:1833, 2018, 2024, 2056` | MI50 value used on RX |
| Gate and up share a quant type, enforced at pack time | `crates/breccia-format/src/pack.rs:635` (type check); `:147` (`gate_up_type`) | MI50, RX |
| SwiGLU as a separate op | `crates/breccia-kernels/shaders/ops_swiglu.comp:45` (`breccia_silu(G[i]) * U[i]`) | MI50, RX |
| GEMV epilogue `silu(acc / s)` | `crates/breccia-kernels/shaders/gemv_body.glsl:264–270` (`BRECCIA_GEMV_EPILOGUE_SILU_DIV`, line 270 `breccia_silu(facc / pc.ep)`) | MI50, RX |
| HC gated-mean fused into the up GEMV (`d64b51b`, −96 dispatches). A different fusion from gate+up+SwiGLU | `crates/breccia-kernels/src/ops/mod.rs:1798` (`fused_gated`); `crates/breccia-kernels/shaders/gemv_body.glsl:93–148` (`BRECCIA_HC_GATED_EPILOGUE`) | MI50, RX |
| No specialization constants, no `VkSpecializationInfo`, no `RequiredSubgroupSize` in `crates/` (grep) | — | — |
| Pipeline cache is null | `crates/breccia-vk/src/ctx.rs:1203` | all |
| No fused gate+up dispatch in base (grep). The WIP fused shader is `wip/four-gemms-crash-20261026@ed7ebd3`, not in base | — | — |
| Per-device measurements stored in the hardware profile | `crates/breccia-profile/src/schema.rs:19` (`HwProfile`), `:34` (`DeviceEntry`); `crates/breccia-profile/src/backend.rs:36` (`DeviceMeasurements`) | all |

## 2. Ranked cards

### Card A — gdn_step 32-lane shared-memory reduction (`7272338`)

**Technique.** Each workgroup owns 4 state columns and 32 lanes per column. Each lane holds 4 state rows in registers (`i = lane + 32r`). Per token there are two reductions through shared memory (`part[LANES][COLS]`, `LANES` 32, `COLS` 4; declared at `gdn_step.comp` line 95 in that commit), each summed by lane 0 in fixed order, with barriers. There are no subgroup operations, so the result does not depend on wave size. The state is written once at the end of the token.

**Source.** `7272338` (`crates/breccia-kernels/shaders/gdn_step.comp`, 174 lines at that commit; the base grid call is `crates/breccia-kernels/src/gdn/mod.rs:409, 430`, and the branch grid change to 1,536 workgroups is derived from 6,144 / 4 and was not re-read on the branch). The commit is contained in `bundle-a2-synthetic`, `bundle-b2-decode`, `claude/kq8-part2`, `claude/kq8-validator-fix`, `feat/block-major-order` and `feat/gdn-clustered-add`. It is not an ancestor of the base tree.

**Breccia has it?** Not in base. The base kernel is the serial one in §1.

**Numbers.** All derived except where marked.

| Quantity | Base | `7272338` | Note |
|---|---|---|---|
| Workgroups | 6,144 cols / 128 = **48** | 6,144 cols / 4 = **1,536** | derived |
| Workgroups per CU (MI50, 60 CUs) | 0.8 (48 CUs busy) | 25.6 | derived; assumes one WG per CU placement. Base check: 48 WG × 2 wave64 = 96 waves, 1.6 per CU, which is the figure `docs/BRIEF.md:279` gives (derived) |
| State bytes per layer | 3,145,728 (786,432 f32 × 4) | same | derived |
| Passes over state per layer | 3 (read twice, write once) | 2 (state kept in registers, written once) | derived from source as read |
| State traffic per token (36 layers) | 339.7 MB | 226.5 MB (×2/3) | derived |
| Floor at 825 GB/s (MI50) | 411.8 µs | 274.5 µs (→ 275) | derived |
| Measured MI50 gdn-scan | 2,768 µs (`docs/BRIEF.md:279`) | not found in a limited grep of branch docs. The reverted subgroup kernel's 1,547–1,920 µs is at `docs/phase-4/01-forward-pass.md:5777`, a different kernel with wrong logits | — |
| Distance from floor | 10.1× (2,768 / 275) | — | derived |
| Accuracy | baseline | state rel err MI50 1.02e-6, RX 7.54e-6 against the 1e-5 bar (RX at 75% of bar). Commit-reported conformance-green on state error only; greedy parity not claimed | not re-run here |

The base-tree figure of 119 GB/s (`docs/BRIEF.md:279`) is 14% of 825 GB/s. From 339.7 MB and 2.768 ms the figure is 122.7 GB/s, or 14.9% (derived; see C6).

**CPU gates, done here.** `glslc -O --target-env=vulkan1.3` and `spirv-val` pass with rc=0 for both the base and the `7272338` `gdn_step` (`scratch-26/glslc-probe/`). `spirv-dis` shows no `GroupNonUniform` ops (wave-size independent) and 4 `OpControlBarrier`, which matches 2 barriers per reduction, 2 reductions per token. The per-column sum does not depend on `COLS`, so `COLS` is a candidate autotune axis that should be bit-identical (derived from source; not run).

**GPU gates, not run.** Both cards, under the owner's GPU window (O8).
- Conformance: state relative error ≤ 1e-5 (kernel bar) on MI50 and RX. The commit reports MI50 1.02e-6 and RX 7.54e-6. Re-run both.
- Accuracy contract: per-stage relative L2 ≤ 2e-3, top-1 agreement, logit delta ≤ 0.02, PPL +0.3%, and 32/32 greedy tokens against the reference.
- Timing, one rule: accept if MI50 gdn-scan ≤ **2,516 µs** (≥ 1.1× over 2,768; 2,768 / 1.1 = 2,516.4, derived) **and** the accuracy contract passes, including 32/32 greedy. Otherwise reject. For any result, report the distance from the 275 µs floor (derived). The 1,547 µs figure at `docs/phase-4/01-forward-pass.md:5777` belongs to the reverted subgroup kernel, which gave wrong logits. It is not an accept threshold.
- Root cause: the reverted subgroup variant gave wrong logits (§8bi, `docs/phase-4/01-forward-pass.md:5770`, `:5777`). `docs/phase-4/01-forward-pass.md:5786–5789` ranks a Wave64 `subgroupShuffleXor` reduction as the most likely of three unproven suspects. Card A uses no subgroup ops. The base GEMV body already runs a 16→1 `subgroupShuffleXor` butterfly on both devices (`crates/breccia-kernels/shaders/gemv_body.glsl:257–258`), so the primitive is in production use on Wave64. A passing conformance run does not prove that the cause is gone. The 32/32 greedy run is the check that matters.

**Open items.** (1) The clustered A/B arm (`95dfa2b@feat/gdn-clustered-add`) assumes Wave32, is unguarded on `gl_SubgroupSize`, and is unmeasured (O3). (2) Lane 0 runs a serial 32-term sum twice per token. That is a new serial chain; a fixed-order tree reduction is a possible follow-up (UNVERIFIED). (3) `openIE-dev/strix-rt @ 7b76f1634c94` README:115 says its grouped GEMMs (more than 1,000 workgroups on 40 CUs, a different GPU) leave no idle tail, so a persistent queue has nothing to recover. The 1,536-workgroup case here is not measured.

**Lanes.** MI50 is the primary gate. RX is a secondary gate because its state error is 75% of the bar. CPU: none.

**Demoted fallback.** The base kernel's geometry at 64 threads per workgroup keeps base arithmetic, so it stays bit-identical. It gives 96 workgroups. Under the occupancy assumption it uses 60 CUs instead of 48, so its ceiling is 60/48 = **1.25×** on MI50 (derived, UNVERIFIED). The traffic stays at 339.7 MB. Use it only if Card A fails the accuracy contract.

### Card B — per-device tuned ROWS_PER_WG through specialization constants

**Technique.** A table keyed on (device fingerprint, SPIR-V hash, candidate-list checksum) whose value is the winning `ROWS_PER_WG`. LANES stays 32 (see the bit-parity rule below). The table lives with the device measurements, not in a new cache. CubeCL's persistent cache stores a key with a checksum and reads back only the fastest index (`tune_cache.rs`, lines 48–53 and 58); whether its checksum covers the candidate list is UNVERIFIED.

**Primary sources.**
- `tracel-ai/cubecl @ a5e814313c60`: `crates/cubecl-runtime/src/tune/tune_cache.rs` (`PersistentCacheKey{key, checksum}` at lines 48–53; line 58 says only `fastest_index` is read back); `crates/cubecl-runtime/src/config/autotune.rs:161–177` (`AutotuneLevel`); `tuner.rs`, `sampler.rs`, `patience.rs`.
- `alibaba/MNN @ 35a20ca0995b`: `source/backend/vulkan/buffer/execution/glsl/gemv_dequant_int3.comp` (`constant_id` for ACTIVATION, `local_size_{x,y,z}_id`, `subgroupAdd` at line 110).
- Doc 06 §5.1 item 2 (line 255) records that ncnn keys its pipeline cache on (SPIR-V, specializations, local size, subgroup size).

**Breccia has it?** No. There are no specialization constants (grep). The probes below show the form works.

**CPU probes (done).** `scratch-26/glslc-probe/lsid.comp` (`local_size_x_id`, validates rc=0). `arrsize_specconst.comp` (`shared float s[ROWS]` with `ROWS` as a spec constant; disassembly shows `OpSpecConstant %uint 8` and a spec-sized array; glslc and spirv-val rc=0, and rc=0 at vulkan1.2). `arrsize_fixedmax.comp` (fixed `shared float s[32]` with a spec-constant index; rc=0).

**Two forms.**
- (a) One SPIR-V module, specialized at pipeline creation through `VkSpecializationInfo`. No new SPIR-V per candidate. Needs the fixed-max or spec-sized array form, which the probes show works.
- (b) Per-variant SPIR-V, as the build already does for `_wide` and `_words`. This keeps today's build pattern but multiplies the shader count by the candidate count.

Form (a) fits the table. Form (b) is the fallback if (a) breaks the existing `build.rs` flow.

**Search rule.** No ALU-law prune (C5). Measure every candidate in the cell's set at the production shape, against the default `ROWS_PER_WG 8`. Keep a non-default cell only if it beats the default by more than 5% (the flatness band from doc 06). GB/s is reported as context only, not as a stop rule. Burn's roofline stop rule is not borrowed.

**Candidates (UNVERIFIED).**
- Plain GEMVs: MI50 {4, 8}; RX {2, 4, 8}.
- HC variant (`BRECCIA_HC_GATED_EPILOGUE`, `gemv_bf16_hcgate`, `build.rs:209`): ROWS_PER_WG must be a multiple of `hc`, because `per = ROWS_PER_WG / hc` (`gemv_body.glsl:111, 169`) and `hc` defaults to 4 (`crates/breccia-engine/src/config.rs:405`). The HC set is therefore {4, 8, 16} on both devices. {2} gives `per = 0` and is illegal on RX.
- Doc 12 lines 452–453 cite llama.cpp's per-architecture values as 4/8 on GCN and 2/4 elsewhere. That conflicts with doc 06 (C4), so the set is measured, not inherited.

**Types.** IQ3_XXS, IQ4_XS (on the word-packed build) and Q6_K for the GEMV cells. For `gdn_step` the axis is COLS (columns per workgroup): the `7272338` shape uses 4, and the base uses 128 (derived from its grid). Lanes: MI50 and RX. CPU: none.

**Bit-parity rule (derived from source, not run).**
- Plain path: `row = wg * ROWS_PER_WG + slot` (`gemv_body.glsl:178`). Each row's 32 lanes, block order and butterfly are unchanged when only ROWS_PER_WG changes, so the output should be bit-identical.
- HC epilogue: `per` sets the partial chunk each workgroup injects (`gemv_body.glsl:165–169`), so the summation grouping changes. The accuracy contract applies, not bit parity.
- LANES: changing it changes the per-row order. Card B keeps LANES = 32. Any LANES candidate needs the accuracy contract.
- `local_size_x` is LANES × ROWS_PER_WG = 256 today, so the spec constant must cover both, and the host constant (`gemv.rs:887`, with `grid_for` and `wgs` at 465–844 and 522) must read the same per-device value. `gemm_body.glsl:40` and `ops_gather.glsl:61` are separate definitions and must not be specialized by accident.

**Cell 2 — split-K CU count.** `gemv.rs:502–503` hard-codes 60 CUs. For the RX (80 CUs, derived target 80 × 4 = 320), a 640-row shape (80 workgroups) should get 4 splits, and the code gives 3 (derived: 60 × 4 = 240, ceil(240 / 80) = 3). The rule is wrong for RX at 80–106 workgroups. But `9f1ad74` measured split-K as never paying on the head shape (−2% and −6% at s=2 and s=4; router and qkv flat), and the doc-comment numbers in `gemv.rs` (+7% to +27%) come from a different shape (640×2560). Measure on the down-projection shapes (`crates/breccia-kernels/src/ops/mod.rs:1775`) before any change. Split-K changes summation order, so it is gated by the accuracy contract, not bit parity.

**Expected gain (UNVERIFIED).** Small. The gains that landed in this repo came from access pattern and lookup depth (`6bbcc6a`, `9f1ad74`), not from workgroup geometry.

**GPU gate.** Per-cell timing of every candidate at the production shape, against the default. Keep a non-default cell only if it beats the default by more than 5% **and** its output is bit-identical (plain path) or passes the accuracy contract (HC path).

### Card C — gate+up+SwiGLU fusion, adopt and measure from history

**Technique.** One workgroup computes the gate and up dot products for one output row of one expert from one activation vector, reduces both inside the dispatch, and writes `silu(gate) * up`. The unfused path is separate gate, up and SwiGLU dispatches. Research 06 §2.2 measured 3 dispatches at 15,633 ns (m=1) on RX, against 4,690–5,018 ns for the fused single dispatch.

**Primary sources.**
- `wip/four-gemms-crash-20261026@ed7ebd3`: `harnesses/vk-gemm-sweep/shaders/gateup_swiglu.comp` (129 lines). The commit adds six files under `harnesses/vk-gemm-sweep/`. Its subject says the four multi-user GEMMs crashed the MI50. Its body says a GPU reset hit the MI50 during `gemm_many` and marks the round DO NOT MERGE until the four shaders pass conformance on RX (headless) with per-kernel isolation and a watchdog. The commit does not isolate this shader as the cause.
- `docs/research/06-vulkan-gemm-shapes-and-ncnn.md:134–142` (§2.2): this shader's structure measured on RX, Q4_K only. m=1: 3.1–3.3×; m=8: 5.3–5.5×. The mechanism given is dependent-chain latency (5.35 µs dependent against 2.79 µs back-to-back).
- `docs/plan/PDF-GAP-ANALYSIS.md:41` (row 17, "MEASURED, NOT WIRED — 3.1× (m=1) / 5.3× (m=8)"). It cites "research 06; §8ad", but the measurement is research 06 §2.2. §8ad starts at `docs/phase-4/01-forward-pass.md:2479`, and lines 2479–2530 do not carry the 3.1× or 5.3× figures (grep, derived). The figures are repeated at 2587–2588, in §8ae. The PDF-GAP cite should point to research 06 §2.2.
- `openIE-dev/strix-rt @ 7b76f1634c94` README:116: fused gate+up+SwiGLU with identical numerics, energy-neutral (11.18 against 11.17 J/token). That refutes a byte-saving argument, not the dispatch-count argument.
- `alibaba/MNN @ 35a20ca0995b`: `source/backend/vulkan/buffer/execution/VulkanFusedProj.cpp` (QKV and `act_silu_mul` fused, explicit buffer barriers).

**Breccia has it?** Not in base. Three different things in base must not be confused with it: the SwiGLU epilogue on one GEMV (`gemv_body.glsl:264–270`); the HC gated-mean fused into the up GEMV (`d64b51b`, `crates/breccia-kernels/src/ops/mod.rs:1798`); and the type precondition (`pack.rs:635`). Whether the production expert path already applies the epilogue was not traced.

**Shader facts (read from `ed7ebd3`).**
- Q4_K only: `DEQUANT_UNIT` is `dequant_q4_k_unit` (`crates/breccia-kernels/shaders/blocks.glsl:333`), `QK 256`, `BLOCK_WORDS 72`. IQ3_XXS and IQ4_XS gate/up need their own dequant units.
- Two matrix bases in push constants (`base`, `rows_up`); bindings inline (`BRECCIA_OWN_BINDINGS`). `LSX` and `ROWS` come from build defines in the harness, not from the file.
- The header says it targets the shared expert. Whether the production shared-expert gate/up type is Q4_K was not traced.
- Reduction, in WIP file lines: per-block sum `dot(v0..3, x0) + dot(v4..7, x1)` (83–84, 98–99); even and odd block steps go to two accumulators (110–116); 16→1 `subgroupShuffleXor` butterfly (121–124). Derived: XOR offsets ≤ 16 stay inside aligned 32-lane groups, so the reduction is valid for subgroup size ≥ 32. It is not guarded on `gl_SubgroupSize`.

**Bit parity, by source comparison (derived, not run).**
- *Matches.* The per-block sum, the even/odd accumulator split and the 16→1 butterfly are the same operations in the same order as the production GEMV body (`crates/breccia-kernels/shaders/gemv_body.glsl:212–258`: per-block `s = dot + dot`, `acc0`/`acc1` at 241 and 243, `facc = acc0 + acc1` at 255, butterfly at 257–258).
- *Does not match.* The epilogue. WIP line 127 is `(fg / (1.0 + exp(-fg))) * fu`. Production SwiGLU is `breccia_silu(G[i]) * U[i]` (`ops_swiglu.comp:45`). `breccia_silu` is `x * breccia_sigmoid(x)` with a 0.0 clamp for x < -80 (`crates/breccia-kernels/shaders/ops_math.glsl:30–36`). `breccia_sigmoid` is `1/(1+exp(-x))` for x ≥ 0 and `e/(1+e)` with `e = exp(x)` for x < 0 (`ops_math.glsl:21–27`). The WIP divides once and does not sign-split, so it rounds differently from production across the range, not only in the tail.
- *Fix before the parity gate.* Call `breccia_silu(fg) * fu`. With that, the Q4_K fused path should be bit-identical to the unfused path (derived, not run).

**Shuffle exposure.** Card C uses `subgroupShuffleXor` on Wave64 (RX default, doc 06 lines 104–108). The reverted gdn variant (§8bi) gave wrong logits. `docs/phase-4/01-forward-pass.md:5786–5789` ranks a Wave64 shuffle reduction as the most likely of three unproven suspects. The production GEMV already uses the same butterfly on both devices, so the primitive alone is not disproven. The parity run must run on RX at its default subgroup size of 64, explicitly.

**Size (derived from the doc's premise).** `docs/phase-4/01-forward-pass.md:2587–2588` (§8ae, which starts at line 2544) says fusion "halves one of ~31 dispatches per block", giving 1/31 × 1/2 = **1.6%** of drain. That premise conflicts with the single fused dispatch in research 06 (C12). If the fused path removes two dispatches per expert, the size must be recomputed. The forward-pass table at `docs/phase-4/01-forward-pass.md:2643` lists the same fusion at "~1.6 % of the drain", "with the barrier work". The barrier cost is about 1.1 µs plus 0.45 µs × bpr, blocks per row (round-10 bullet in `docs/research/README.md`, not re-read here).

**Caveat from `6bbcc6a`.** The roughly 35 µs per call that the engine pays outside the kernel is a fence-waited submit (research 08: 33.2 µs against 8.6 µs queued). Fusion removes dispatches inside a submit, not submits. Card C's gain is the dispatch and barrier term only. Do not credit it with the 35 µs.

**Strix caveat.** Energy-neutral (11.18 against 11.17 J/token). That refutes the byte-saving argument, not the dispatch-count argument.

**Gate, in order.**
1. Crash gate (the WIP's own condition): RX headless first, per-kernel isolation, watchdog. No MI50 run until RX passes.
2. Parity: bit-compare fused against unfused gate, up and output for Q4_K on synthetic inputs, after the `breccia_silu` fix. Exact equality is the target. If it is not exact, the accuracy contract applies.
3. 32/32 greedy tokens for the type that ships.
4. Token time: accept if the gain exceeds run-to-run spread. Report it against the derived 1.6%.
5. Op-level re-measure at the production shape on RX. The 3.1×/5.3× figure is Q4_K on RX only. IQ3_XXS and IQ4_XS are UNVERIFIED.

**Lanes.** RX first (gate 1). MI50 only after RX passes. CPU: none.

## 3. Rejected

- **TNN** (`Tencent/TNN @ f0cb08129a05`): no Vulkan backend. `source/tnn/device/` holds arm, cpu, cuda, huawei_npu, metal, opencl, rknpu, x86.
- **krnl**: a Rust GPU library. No LLM-specific technique found in a narrow search.
- **rust-gpu** (`Rust-GPU/rust-gpu @ 60d1e2541ea5`): a compiler, no LLM technique. Breccia already compiles GLSL through glslc.
- **MNN `gemv_dequant_int3.comp`**: int3 weights packed as 2-bit plus 1-bit planes (`Wp[]` layout comment), not the IQ3_XXS codebook (`crates/breccia-kernels/shaders/blocks.glsl:58, 78`). Its `subgroupAdd` and `constant_id` pattern is kept for Card B.
- **MNN int8 cooperative-matrix conv1x1** (`VulkanConv1x1CoopA8.cpp`): doc 22 settled cooperative matrix as negative.
- **MNN `EagerBufferAllocator`** (`source/core/BufferAllocator.cpp`): doc 22 — a fixed slot table beats a general suballocator.
- **Burn fusion trace** (`tracel-ai/burn @ 5856e0735828`, `crates/burn-cubecl-fusion/src/engine/trace/fuser.rs`): generic elementwise fusion. Breccia's epilogues are hand-written.
- **CubeCL quant scheme** (UE4M3 block scales; quant files found by code search at `tracel-ai/cubecl @ a5e814313c60` under `crates/cubecl-common/src/quant/` and `crates/cubecl-std/src/quant/`, not opened): GGUF block formats are fixed, so the scheme does not transfer as is.
- **Burn matmul tile tuning**: dense GEMM, not decode. Its roofline stop rule is not borrowed; Card B measures every candidate.
- **strix hybrid tiles**: +0.3%, noise (README:114).
- **strix persistent work queue**: equal or 0.5% slower (README:115).
- **Breccia's own int8-dot path** (`2883d0a`, `c4f4292`): measured 1.18× on MI50 and 1.11× on RX, with per-element error 6.21e-3 against the 1e-3 contract. The suspect is the q8_1 activation quantiser (UNVERIFIED). Off by default. Reopen only with a different activation quantiser. strix's own integer-dot run hit the same wall (README:113: per-32 activation scales failed parity; per-8 passed but lost on energy).

## 4. Watch list

- **strix Q6_K quad** (`GEMVQ_Q4P=71`, README:117): −7 W GPU power, +16% J/token, 15% slower on a quiet box, load-issue bound. Relevant to Card B's Q6_K cell only.
- **MNN `attention_decode_q1_subgroup.comp`** (and `_hd128`): Breccia attention is flat at 2,048 cells, and the QSA indexer is the cost (doc 18).

## 5. Conflicts with existing docs

- **C1.** `docs/BRIEF.md:279` says "DIAGNOSED, fix open", and `:342` says the "wrong-but-fast 32-lane kernel was reverted". `7272338` is a different 32-lane kernel, commit-reported conformance-green on state error only (greedy parity not claimed), on six branches. Update the BRIEF status once an owner picks the branch (O2).
- **C2.** Wave size. `docs/BRIEF.md:16` says RX is Wave32; `docs/phase-0/gate-A-report.md:32` says Wave32 via `VK_EXT_subgroup_size_control` on RX only, and `docs/phase-4/01-forward-pass.md:5801` gives an occupancy of 2.4 wave32 on the RX. Doc 06 lines 104–108 measured RX default 64 (equal to required 64) and forcing 32 as 5–8% slower. The base tree requests no required size. `95dfa2b` assumes Wave32.
- **C3.** Doc 18:75 and `docs/research/README.md` round-18 bullet say "no precedent" for a gated fold. `strix-rt README:116` reports fused gate+up+SwiGLU with identical numerics.
- **C4.** Doc 12:452–453 says ROWS_PER_WG 8 is right on the MI50 and "2× high for RDNA2". Doc 06 says geometry is flat within 5%. Card B measures the candidate set rather than inheriting either.
- **C5.** The round-12 ALU/DRAM law uses constants 8.0·bpw (MI50) and 53·bpw (RX), which round 25 flags as not following from doc 12's stated rates (`docs/research/25-exl3-trellis-and-kv-quant-context.md:19–27`). Round 26 adds two contradictions. `6bbcc6a` says the IQ4_XS word build raised the static count from 123 to 144 instructions per unit and was still 2× faster, so the limit is dependent lookup depth, not I_unit. `2883d0a` measured 1.18× on MI50 against round-12's predicted 1.87× for IQ3_S. The round-12 IQ4_XS value of 104 is stale and is not used here.
- **C6.** `docs/BRIEF.md:279` gives 119 GB/s and 14%. From 339.7 MB and 2.768 ms the derived figures are 122.7 GB/s and 14.9%. Small discrepancy; the BRIEF number is kept as cited.
- **C7.** Round-12's X1 (q8_1 activation card) against `c4f4292` and strix integer-dot. Per-32 scales fail parity. X1 is not adopted.
- **C8.** `gemv.rs` doc comment (+7% to +27% from doc 01-dequant-gemv §4.2 at 640×2560) against `9f1ad74` (split-K never paid on the head shape). Shapes differ, so this is scope rather than contradiction, but the `recommended_splits` policy has not been checked at other shapes.
- **C9.** `docs/plan/PDF-GAP-ANALYSIS.md:41` (3.1×/5.3×, op level, research 06 §2.2) against `docs/phase-4/01-forward-pass.md:2587–2588` (~1.6% of drain, token level). Scope, not contradiction.
- **C10.** `9f1ad74` finds quantised GEMV flat in M el/ms across bpw (on the head; the flat figures are in the commit message). Round 25 (`docs/research/25-exl3-trellis-and-kv-quant-context.md:37`) finds IQ3_XXS ALU-bound on both cards under doc 12's own rates, and DRAM-bound on RX only under the brief's 53·bpw constant. No RX measurement of the flat-rate claim was found in the docs read here.
- **C11.** The `7272338` comment's "7–10× off floor" against 2,768 / 275 = 10.1× (derived). Consistent.
- **C12.** Dispatch count per expert. `docs/phase-4/01-forward-pass.md:2587–2588` says fusion gives 2 dispatches per expert instead of 3. Research 06 §2.2 and the WIP shader (one dispatch) give 1 fused dispatch. The size in Card C depends on which count is right (O5).

## 6. Owner decisions

- **O1.** Accept Card A's non-bit-identical reduction under the 1e-5 kernel bar and the 32/32 greedy gate, or keep the 64-thread bit-identical fallback (≤ 1.25× on MI50, derived).
- **O2.** Choose the canonical branch for `7272338` (six branches contain it) and update `docs/BRIEF.md:279` and `:342` to match.
- **O3.** Keep or delete the `95dfa2b` clustered A/B arm (Wave32 assumption, unguarded, unmeasured).
- **O4.** Table form: specialization constants (form a) or per-variant SPIR-V (form b), and where to store it (`DeviceMeasurements` in `backend.rs:36`, or a sibling of `DeviceEntry` in `schema.rs:34`).
- **O5.** Card C: adopt the WIP from `ed7ebd3` under the crash gate (RX first, isolation, watchdog). Decide whether an RX Q4_K result can stand in for IQ3_XXS and IQ4_XS. Decide the fused dispatch count (C12) before the 1.6% size is used.
- **O6.** Change `recommended_splits` only after a down-shape measurement, given `9f1ad74`.
- **O7.** Retire the round-12 ALU/DRAM law as a gate, or re-derive it from measured rates first (C5). This also affects round 25's bound claims.
- **O8.** GPU window for the gate runs. Not the subagent's call.

## 7. Not checked

- No GPU runs, no model loads, no Breccia builds, no cargo tests.
- `7272338` conformance not re-run. Its GPU timing was not found in a limited grep of branch docs. The reverted subgroup kernel's timing is at `docs/phase-4/01-forward-pass.md:5777` (§8bi header at 5770), a different kernel.
- `ed7ebd3` not run. The commit attributes the MI50 reset to the four-GEMM round during `gemm_many`; this shader's share of it is not isolated.
- Card C parity was compared by reading source (`gemv_body.glsl:212–258` against WIP lines 66–124). It is not run. The WIP's Wave64 behaviour is not tested. The production shared-expert gate/up type was not traced.
- `95dfa2b` GPU parity not measured.
- Lane-0 synchronisation correctness reasoned by hand. The barrier lowering to `OpControlBarrier` with workgroup-memory semantics is assumed, not checked.
- Burn, CubeCL and MNN read from flat scratch copies, not full trees. The CubeCL quant scheme path is unresolved.
- Docs 22, 24 and 25 read in part. strix read only from README and `docs/prefill.md` excerpts.
- rust-gpu, wgpu and krnl searches were narrow.
- Ancestry checked: the base tree contains `6bbcc6a`, `9f1ad74`, `2883d0a` and `c4f4292`, and does not contain `7272338` or `ed7ebd3`. The branch `feat/r18-moe-gateup-fused` is fully merged into the base (bbffb06, checked in the research worktree) and has no gate+up or fused file in its tree.
