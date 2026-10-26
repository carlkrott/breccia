# Gate A — Feasibility report (Phase 0 conclusion)

**Decision: hybrid HIP+Vulkan (as pre-authorized), with architecture corrections from measurement. HIP is the M2 reference compute backend; Vulkan stays actively viable and owns the RX-specific path. The plan's dual-GPU *layer-pipeline* assumption is overturned and must be re-proven in Phase 2.**

Date: 2026-10-26 · Evidence: `docs/phase-0/01`–`05` · Raw: `logs/`, `harnesses/vk-probe/results/`

## 1. The evidence chain

| # | Experiment | Result |
|---|---|---|
| 01 | Production baseline (llama.cpp HIP, dual GPU, exact service flags) | **20.79 tok/s** decode, 112.7 tok/s prefill — the number to beat |
| 03 | Single-device llama-bench matrix, busy%-attributed | Vulkan ≥ HIP on RX decode (355 vs 275 tg/s); MI50 favors HIP decode (261 vs 189); **`MESA_VK_DEVICE_SELECT=1002:164e!` had hidden both discrete GPUs from all Vulkan processes** (root cause of the old "Vulkan = 2.56 tok/s" *if* that test ran filtered — it did not; see 05) |
| 05 | Full-model Vulkan, dual GPU, byte-identical flags/placement | **2.40 tok/s** (8.7× slower), prefill only −17%, zero kernel warnings — reproduces the historical number on the real GPUs |
| 05b | Full-model Vulkan, **RX-only**, all 13,577 MiB GPU weights on one card | **16.6 tok/s decode, 123 tok/s prefill** — 80% of HIP-dual decode and the *best prefill of any config tested* |
| 02 | Raw RADV probe (both cards, validation-clean, dual-run ≤0.15% variance) | RX: 462 GB/s DRAM (80%), **26.7 TFLOPS FP32 (97%)**, 28.7 GB/s PCIe (91%), **100% compute/transfer overlap**; MI50: 826 GB/s HBM (81%), 6.8 TFLOPS, **PCIe ≈2.9 GB/s (gen3×4), ~0% overlap**; **no `VK_KHR_cooperative_matrix`** on either card |
| 04 | Host memory/tier check | mlock 8 MiB default, MemAvailable 7–9 GB under load, io_uring OK, 673 GB free SSD tier @ ~350 MB/s O_DIRECT |

### What this resolves

1. **RADV is not the bottleneck.** Raw Vulkan reaches 80–97% of hardware limits on both cards; single-device full-model Vulkan is production-class (16.6 tok/s, best prefill measured). The 8.7× full-model collapse is isolated to **llama.cpp's dual-device Vulkan path** (same split shape runs fine on HIP at 20.79). Remaining suspicion set: cross-device reduction/sync per op-batch, KV residency across two VkDevices, or the host-mediated boundary in ggml-vulkan — to be profiled with `GGML_VK_PERF_LOGGER` if/when we touch that path.
2. **The plan's central hardware assumption was wrong in a measurable way:** the MI50's host link is **gen3 ×4 behind the chipset (~2.9 GB/s/dir, transfers serialize with compute)**. The PDF's design (MI50 = "high-bandwidth expert *streaming* vault" fed over PCIe) is invalidated — streaming to the MI50 costs 10× and cannot overlap. The RX is the only good streaming citizen (28 GB/s, DMA fully hidden under compute, 128 MiB Infinity Cache = 3.9× bandwidth lever for hot working sets).
3. **Environment hazards now codified** (`docs/PROJECT-SETUP.md`): session-wide `MESA_VK_DEVICE_SELECT` iGPU pinning, kfd/group permissions for HIP, triple-visibility double-mask, llama-bench `-d` non-isolation + backend mislabeling, DPM clock variance, gpu-manager idle eviction.

## 2. Backend & architecture decision (M2 binding)

| Concern | Decision | Why |
|---|---|---|
| Reference compute backend for M2 kernels | **HIP** (ROCm, both cards) | Only dual-GPU config >20 tok/s today; MI50 decode faster under HIP (261 vs 189); MFMA matrix cores available (RADV lacks cooperative_matrix) |
| Vulkan's role | **Keep** — RX-primary path + portability floor | Single-device 16.6 tok/s + best prefill; raw 97%/80% of peak; one API for both archs; probe proves Phase-2 transfer design (timeline semaphores, distinct transfer queue, zero-copy host import) implementable |
| GPU roles (measurement-derived) | **RX 6950 XT**: streaming/expert-paging home + prefill + drafter · **MI50**: static resident weights + bandwidth-bound decode (no PCIe streaming to it) | RX: 28 GB/s + 100% overlap + Infinity Cache. MI50: 2.9 GB/s + 0% overlap, but 826 GB/s HBM |
| Dual-GPU strategy | **Expert-parallel / disjoint placement by default** (no per-token cross-device traffic). A layer-pipeline split is allowed **only after Phase 2 proves** its handoff overhead on the custom path | llama.cpp's dual-device Vulkan collapses 6.9× vs its own single-device; the same shape is fine on HIP — implementation-fragile |
| GEMM strategy | Subgroup/FMA kernels (no cooperative matrix on RADV); RX FP16 measured 42.6 TFLOPS achievable; Wave32 via `VK_EXT_subgroup_size_control` on RX only, MI50 locked Wave64 → specialization constants | 02 §2.4, §3.2 |

## 3. M2 kernel work order (from `docs/specs/qwen4exp-architecture.md`)

1. **Decode GEMV + dequant** for the per-layer quant mix actually present (IQ2_XXS/IQ3_S/IQ4_NL/Q2_0, 14 types) — HIP reference kernels first, RX-Vulkan parity second. *This is the 20→85 tok/s lever.*
2. **Attention decode**: q4_0-KV flash attention + lightning-indexer scoring (12 QSA layers) + GDN recurrent step (36 layers, state `(S,h_v,S)`, conv history) — GDN is ctx-free and tiny (~110 MiB/seq), which is why concurrency is affordable.
3. **PLE gather path**: 27 GB lazy CPU_Mapped table — measure host-side gather cost per token first (present in all configs, so not the collapse cause, but a fixed tax on absolute speed).
4. **MoE routing**: softmax-512 → top-10 with `2⁻¹⁴` clamp, BF16 router activation (8.1e-3 logit error otherwise — spec §5), shared-expert fusion.
5. Cross-device/expert-tiling only after Phase 2's DMA/overlap harness reports (RX overlap proven 100% at raw level; MI50 target excluded from streaming by 02).

## 4. Gate checklist

- [x] Baseline scorecard recorded (01)
- [x] Backend risk empirically retired — pure-Vulkan-vs-hybrid decided with a localization experiment (03, 05, 05b)
- [x] Artifact/quant inventory complete (03 spec) — includes fork-only Q2_0=type 42, byte-exact pack/MTP formats, no RCO parser exists
- [x] Forward-pass spec complete (04 spec) — 36 GDN + 12 QSA layers derived, ops with file:line in both engines, 12 open questions
- [x] Host RAM/Tier-2 sizing grounded (04)
- [x] Reference engines cloned + assessed (manifest, `docs/reference-engines/01-02`)
- [ ] **Open (carried to Phase 1):** exact root cause of ggml-vulkan dual-device collapse (profile only if that path is pursued); HIP-vs-Vulkan parity harness for M2 kernel validation; DPM clock pinning for reproducible benches

**→ Gate A: PASS. Proceed to Phase 1 (workspace + `breccia-profile`).**
