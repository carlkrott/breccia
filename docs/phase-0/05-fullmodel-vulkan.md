# Phase 0.2b — Full-model Vulkan confirmation (Gate A input)

The decisive full-model comparison: same model, same flags, same memory placement, different backend.

## Setup

- Binary: `build-vulkan/bin/llama-server` vs the production `build-hip-q5kv` (same source, build 1b70350c9)
- Flags: **identical to `qwen-dual-serve.sh`** (`-ngl 99 -ts 9,15`, same `-ot` expert map with device names swapped `ROCm0/1 → Vulkan0/1`, `-cmoe --moe-cache-mib 0 -lzm on -lm none --no-host`, `-c 200000 -np 1 -ctk/-ctv q4_0 -fa on`, `-ub 256`, `-t 16 -tb 16`), port 8162, scratch slot-save dir
- Environment: `MESA_VK_DEVICE_SELECT` stripped + `GGML_VK_VISIBLE_DEVICES=1,2` (see 03 — otherwise the run silently lands on the iGPU)
- **Verified identical placement at load**: `Vulkan0 = 9215.72 MiB / Vulkan1 = 4361.47 MiB / CPU 24973 + CPU_REPACK 6300 + CPU_Mapped(lazy PLE) 27465` — byte-for-byte the same split as the HIP shutdown log (`ROCm0 9215 / ROCm1 4361 / Host 52438`)
- No warnings, no unsupported-op messages, no fallbacks in the entire server log (grep count = 0)

## Results (server `print_timing` ground truth)

| Metric | Vulkan (RADV) | HIP (ROCm) | Ratio |
|---|---|---|---|
| **Decode, 256 tok** | **2.40 tok/s** (416.2 ms/tok) | **20.79 tok/s** (48.1 ms/tok) | **0.115× (8.7× slower)** |
| Decode runs (client wall) | 127.8 / 122.1 / 106.9 s | 13.2 / 12.5 / 12.6 s | — |
| TTFT | 1.2–2.2 s | 0.26–1.1 s | comparable |
| **Prefill, 1001 tok** | **93.36 tok/s** | **112.72 tok/s** | 0.83× (17% slower) |
| Load-time placement | identical | identical | — |

Raw: `docs/phase-0/baseline-vulkan-fullmodel-20261025.json`, `logs/vulkan-fullmodel-*.log`.

## Reading

1. **This reproduces the historical data point** ("Vulkan 2.56 tok/s vs ROCm 23", `qwen-dual-serve.sh` header, 2026-10-25) *on the real GPUs with the filter removed* — the old number was not an iGPU artifact. The full-model Vulkan pathology is real.
2. **It is decode-specific**: prefill only loses 17%, decode collapses 8.7×. Prefill (large GEMMs) and decode (GEMV + attention + sync) behave very differently — consistent with the 0.3B micro result where Vulkan decode *won* on the RX (355 vs 275 tg/s) when weights were fully GPU-resident.
3. Combined with 03, the contradiction is informative: **RADV raw device performance is fine; llama.cpp's ggml-vulkan full-model decode path on this workload is not.** Candidates for the pathology (to be separated in Phase 1+): decode GEMV/dequant paths for the per-layer mixed quants (IQ2_XXS/IQ4_NL/Q2_0), q4_0-KV flash-attention on RADV, per-step host sync in the dual-device layer split, and the lazy-read CPU_Mapped PLE path under the Vulkan backend.
4. Micro-benchmark verdict still stands (03): single-device, GPU-resident weights → Vulkan ≥ HIP on the RX for decode.

## Gate A implication

- **Production engine for the target workload: HIP remains the reference backend today** (only path at 20+ tok/s with this model/flags).
- The "pure Vulkan" path fails its full-model gate **in llama.cpp's implementation** — but the raw-probe data (02, pending) determines whether this is a *driver/hardware* verdict or an *implementation* verdict. If RADV raw bandwidth/compute/overlap are healthy, breccia's custom Vulkan kernels remain viable (as planned for M2+); if they're also pathological, the hybrid pivot is mandatory.
- The `MESA_VK_DEVICE_SELECT` finding (03) must be treated as an operational hazard for **any** Vulkan tooling on this machine — including future breccia runs — and fixed at the session/service level.
