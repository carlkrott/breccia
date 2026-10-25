# Phase 0.2 — Backend matrix: HIP vs Vulkan (llama-bench)

Single-device llama-bench measurements across both backends and all three GPUs, with per-GPU busy% attribution. **Contains the session's most important finding: the "Vulkan is unusably slow" belief on this machine was an environment artifact.**

## Environment

| Item | Value |
|---|---|
| Binary | fork `llama.cpp-qwen4exp-mmap` build 1b70350c9 (11606): `build-hip-q5kv` (ROCm) + `build-vulkan` (RADV/Mesa 26.2.4) |
| Model | `embeddinggemma-2-Q8_0.gguf` (280 MiB, 0.3B dense) — chosen to isolate backend behavior from model complexity |
| Args | `-p 512 -n 128 -t 16 -r 3` |
| Host | idle otherwise; d1 services stopped for the window |
| Harness | `harnesses/bench_attrib.sh` (pins single device via env + samples `/sys/class/drm/card*/device/gpu_busy_percent`) |

Device indices: HIP — `CUDA_VISIBLE_DEVICES` 0=RX 6950 XT (gfx1030/W32), 1=Radeon VII/MI50 (gfx906/W64), 2=iGPU. ggml-Vulkan enumeration (unfiltered) — 0=iGPU, 1=RX, 2=MI50.

## Results (single device, attributed by busy% peaks)

| Device | HIP pp512 | HIP tg128 | Vulkan pp512 | Vulkan tg128 |
|---|---|---|---|---|
| **RX 6950 XT** | 28,480 ± 6,207 | 274.7 ± 1.5 | **24,340 ± 112** | **355.4 ± 6.9** |
| **MI50 (Radeon VII)** | 11,972 ± 990 | **261.1 ± 0.8** | **14,526 ± 354** | 188.6 ± 0.3 |
| iGPU (Raphael, 2 CU) | (CPU fallback) | (CPU fallback) | 875 ± 7 | 66.6 ± 1.3 |
| CPU (7950X, 16T) | 2,917 ± 112 | 73.7 ± 3.5 | — | — |

Raw logs: `logs/llama-bench-pinned-*.txt`, `logs/llama-bench-attributed-*.txt`, banners `logs/banner-*.txt`.

### Reading the table

1. **Vulkan is competitive on both discrete GPUs — not 10× worse.** On the RX, Vulkan *beats* HIP on decode (355 vs 275 tg/s, +30%) and is 15% behind on prefill. On the MI50, Vulkan leads prefill (+21%) but trails decode (189 vs 261 tg/s, −28%).
2. Per-GPU Vulkan profiles differ: RX decode favors RADV's Wave32 path; MI50 decode favors HIP's Wave64 path — consistent with the plan's asymmetric role assignment (compute role → RX, bandwidth role → MI50).
3. The iGPU (2 CU) runs Vulkan at **875 pp / 67 tg — slower than the 16-thread CPU** (2,917 / 74). Any run landing there is effectively crippled.

## The `MESA_VK_DEVICE_SELECT` artifact (session-critical)

The session environment carried **`MESA_VK_DEVICE_SELECT=1002:164e!`** (1002:164e = Raphael iGPU, `!` = exclusive). Effect: MESA's device-select layer hid both discrete GPUs from **every Vulkan process** in this session:

- `vulkaninfo` initially showed 3 GPUs (pre-filter state), later 1 — the var appeared in the opencode server environment between restarts.
- llama-bench Vulkan: `ggml_vulkan: Found 1 Vulkan devices: 0 = ... RAPHAEL_MENDOCINO` → all early "Vulkan" numbers (875 pp / 67 tg) were **iGPU numbers**.
- `sudo` (env_reset) loses the var → root sees all 3. `env -u MESA_VK_DEVICE_SELECT` restores them.
- **Historical implication:** the qwen-dual-serve.sh log entry "Vulkan tested at 2.56 tok/s vs ROCm 23 tok/s" and the plan's premise that Vulkan is dead on this pair must be re-validated — if that test ran with this filter (or another iGPU-only condition), it measured a 2-CU iGPU, not the RX/MI50. *Action: re-run the full-model Vulkan test in a filtered-free environment before Gate A closes.*

Provenance: not set in `/etc/profile.d`, `/etc/environment`, `~/.bashrc`, `~/.config/environment.d`, or systemd user env — inherited from the opencode server process across restarts (likely imported from a gpu-manager/tool session). Treat as **session-contaminating env**: strip it in all Vulkan measurement commands (harness does this).

## Measurement pitfalls discovered (encoded in the harness)

1. **`llama-bench -d N` does not isolate a device.** With multiple devices visible, layer-split stays active across all of them: `-d 0/1/2` all produced statistically identical results (≈7.4k pp / 119 tg) with *both* discrete cards busy. Single-device selection requires visibility env (`CUDA_VISIBLE_DEVICES` / `GGML_VK_VISIBLE_DEVICES`).
2. **Triple visibility masking double-masks.** Setting `CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1` makes HIP init fail outright ("no ROCm-capable device is detected") because CUDA→HIP masking applies twice and index 1 of a 1-device set is empty. Set **only** `CUDA_VISIBLE_DEVICES`. (Symptom seen as CPU-fallback silently labelled `ROCm`.)
3. **The fork's llama-bench mislabels the backend column** — it prints `ROCm` even when HIP init failed and the run silently fell back to CPU (confirmed: identical numbers to CPU control, no card busy). Never trust the label; attribute by busy% or banner.
4. Multi-device-without-pinning halves throughput: unmasked HIP runs (all 3 devices) gave ≈14.6k pp / 174 tg vs 28.5k / 275 pinned — the idle iGPU and cross-device sync drag the split down. The production `qwen-dual` service's `-ts 9,15` two-GPU split is unaffected by the iGPU (it uses `--device ROCm0,ROCm1`), but any future engine must **exclude the iGPU explicitly**.
5. GPU DPM states move results: an MI50 left at `high` perf state (1801 MHz after service teardown) produced 273–293 tg runs that dropped to 174 tg after idle — calibration benchmarks must pin clocks or warm up consistently (a `breccia-profile` requirement).

## Implication for Gate A (preliminary)

At the microbenchmark level on a 0.3B model, **RADV Vulkan is in the same class as HIP on this heterogeneous pair** — the earlier premise for abandoning Vulkan is unproven. Pending: the vk-probe raw bandwidth/DMA/overlap numbers (`docs/phase-0/02-*`), then the full-model confirmation. The hybrid HIP+Vulkan pivot remains available: HIP for MI50 decode, Vulkan/RADV for RX — matching each card's stronger path above.
