# Breccia — gated phase plan

Source of truth for scope. Each phase ends in a gate: the gate report lives in `docs/phase-<N>/gate-*.md` and records the measurements that decided continuation, pivot, or stop. Scope was agreed with the user: **full arc, gated phases**; **hybrid HIP+Vulkan** is the sanctioned pivot if RADV measurements fail; success = **~85 tok/s solo decode AND 4–6 concurrent agent slots** in the 32 GB VRAM + ~30 GB host-RAM envelope; tailnet (M6) deferred.

## Phase 0 — Baselines & risk retirement

- **0.1 Baseline capture** — fresh solo/parallel decode + prefill numbers for llama.cpp (HIP) and Strata; this is the scorecard every later gate beats.
- **0.2 Backend probe (the decisive experiment)** — raw VRAM bandwidth, GEMM/GEMV throughput, PCIe H2D/D2H, and compute/transfer overlap measured on both cards under **Vulkan (RADV) and HIP**. Context: llama.cpp on Vulkan previously measured 2.56 tok/s vs 23 tok/s HIP on this workload.
- **0.3 Artifact/quant inventory** — GSQ-RCO/IQ3_XXS layout, Strata pack format, MTP runtime artifacts → `docs/specs/model-artifacts-inventory.md`.
- **0.4 qwen4exp forward-pass spec** — the ops inventory for M2 → `docs/specs/qwen4exp-architecture.md`.
- **0.5 Host-RAM reality check** — mmap/pinned-buffer behavior under a 28–30 GB Tier-2 cap (host has 58 GB with swap history; a 30 GB cap previously thrashed).
- **0.6 Reference engine assessment** — clones in `Reference Projects/`, manifest + per-engine working-ways/feature notes.

**Gate A** → feasibility report: pure-Vulkan vs hybrid verdict + M2 kernel work order.

**Gate A: PASSED 2026-10-26** → `docs/phase-0/gate-A-report.md`. Verdict: hybrid — HIP is the M2 reference compute backend; Vulkan viable and RX-primary (single-device full model 16.6 tok/s, best prefill measured); MI50 excluded from PCIe streaming (gen3×4 ~2.9 GB/s, ~0% overlap); expert-parallel placement by default; layer-pipeline split only after Phase-2 proof. Environment hazards codified in `docs/PROJECT-SETUP.md`.

## Phase 1 — Workspace + `breccia-profile` (M1a)

Cargo workspace (`breccia-cli`, `breccia-profile`, `breccia-vk`, `breccia-kernels`, `breccia-format`, `breccia-cache`, `breccia-draft`, `breccia-server`). `breccia-profile`: device discovery (heaps, queue families, subgroup 32/64, extension probe), CPU/RAM/NVMe probe, calibration micro-benchmarks, `hw_profile.json` + BLAKE3 fingerprint, `--reprofile` / `--dry-run-plan` / `--override-role`.

**Gate B** → printed partition plan for the IQ3_XXS shards that matches measured reality.

## Phase 2 — `breccia-vk` async harness (M1b)

Dual `VkDevice`, `gpu-allocator` pools, transfer-queue DMA rings + timeline semaphores, host-pinned staging, io_uring reader; overlap benchmark proving streaming with zero compute stalls. (Under the hybrid pivot this crate narrows to transfer orchestration + RDNA compute.)

**Gate C** → transfer-throughput + overlap report.

## Phase 3 — Loader + kernels + single-token forward pass (M2)

Zero-copy GGUF loader splitting dense / GSQ-RCO experts / PLE tables (reuse Strata pack); dequant kernels (IQ3_XXS, Q8_0, IQ4_NL) with LDS lookups; GDN/SSM, QSA attention, hyper-connections, indexer, PLE per `docs/specs/qwen4exp-architecture.md`.

**Gate D** → logits parity vs llama.cpp, clean decode across the heterogeneous split.

## Phase 4 — LRU expert cache + route prefetch (M3)

~10–12k hot experts across both GPUs, pinned-RAM Tier-2 ring, NVMe miss tier, route predictor (L+1), route-tail skipping.

**Gate E** → solo decode from ~23 → 45–55 tok/s, zero CPU layer fallback.

## Phase 5 — Speculative drafting (M4)

MTP head on the RX (compute role), batched verification on MI50, n-gram copy-drafter. Known trap (own logs): naive MTP collapsed decode 10–12 → 1.6 tok/s by doubling expert reads — drafting must ride the expert cache.

**Gate F** → acceptance-weighted effective decode **≥ 85 tok/s**.

## Phase 6 — Tiered KV + parallel agent server (M5)

PagedAttention (FP16 tail + compressed body, K at higher bits than V), radix-tree prefix cache with disk persistence, agent state machine with tool-wait eviction to NVMe, OpenAI-compatible server.

**Gate G (acceptance)** → 4–6 concurrent agents at usable speed without OOM/thrash, solo ≥ 85 tok/s.

## Standing rules

- `qwen-dual` / `strata-rx` are gpu-manager-owned; benchmark windows restore prior state and are logged in the phase report. `d1-mi50` stop/restart is pre-authorized.
- llama.cpp and Strata stay living baselines; every gate compares against a fresh measurement.
- Disk: 121 GB free — artifacts stay in `/srv/strata-data` (read-only); no model copies.
