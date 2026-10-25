# Phase 0.1 — Baseline: llama.cpp HIP (qwen-dual)

Fresh measurements of the current production configuration. This is the scorecard every later gate beats.

## Configuration under test

| Item | Value |
|---|---|
| Service | `qwen-dual.service` (owned by gpu-manager; started for this window) |
| Binary | `/home/korphaus/llama.cpp-qwen4exp-mmap/build-hip-q5kv/bin/llama-server` |
| Backend | ROCm/HIP, `--device ROCm0,ROCm1 -ngl 99 -ts 9,15` + per-block `-ot` expert map |
| Model | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` shard 1 (43.8 GiB GGUF, mmap) |
| Context | `-c 200000 -np 1`, KV `q4_0/q4_0`, `-fa on`, `-kvu` |
| Extras | `-cmoe --moe-cache-mib 0 -lzm on --no-host -ub 256 --fit off --no-op-offload` |
| Host | 7950X 32T, 58 GB RAM (zram swap active), both GPUs idle otherwise |
| Date | 2026-10-25 22:48 GMT |

## Command

```bash
python3 harnesses/bench_decode.py --url http://127.0.0.1:8160 \
  --out docs/phase-0/baseline-hip-q5kv-20261025.json \
  --runs 3 --gen-tokens 256 --tag hip-q5kv-iq3xxs-solo-np1
```

(raw JSON incl. `/metrics` scrape: `docs/phase-0/baseline-hip-q5kv-20261025.json`; server-side `print_timing` lines are ground truth below)

## Results

| Metric | Result |
|---|---|
| **Decode (256 tok × 3 runs)** | **20.92 / 20.82 / 20.64 tok/s** (mean **20.79 tok/s**, server `eval time`) |
| Client-observed wall | 12.45–13.24 s per 256 tok (incl. TTFT 0.26 s warm) |
| **Prefill (1001 tok)** | **112.72 tok/s** (8.88 s, server `prompt eval time`) |
| Prefill (48 tok, small prompt) | 45.79 tok/s — fixed overhead dominates short prompts |
| TTFT | 0.26 s warm, 1.11 s first request after load |

An earlier run before service restart measured 20.37 tok/s (same flags) — variance ±0.5 tok/s.

## Context / historical comparison (from `/home/korphaus/projects/qwen-dual/results.md`)

| Run | Decode | Notes |
|---|---|---|
| llama.cpp Q2_0 dual-GPU solo | 23.6 tok/s | Q2_0 is ~10% faster than IQ3_XXS here |
| llama.cpp conc-4 | 43–45 tok/s aggregate | after `-tb 16` fix, 4×100k config (superseded) |
| Strata solo / batch 8 | 23 / 47 tok/s | RX-only, pack-resident experts |
| llama.cpp Vulkan (same fork) | **2.56 tok/s** | historical — re-measured in 0.2 |
| MTP draft on RX (naive) | 10–12 → 1.6 tok/s | doubled expert reads — Gate F trap |

**Baseline to beat: 20.8 tok/s solo decode / 112.7 tok/s prefill.** Target: 85 tok/s solo + 4–6 agents.

## Operational findings (this window)

1. **gpu-manager's monitor loop runs even while `gpu-manager.service` is inactive** (separate python PIDs). It evicted `qwen-dual` mid-window: *"stopped qwen-dual (idle): MemAvailable 3002 MiB < 4096"*. It only evicts **idle** units, so benchmarks with requests in flight are safe; expect restarts-of-convenience to be reaped if left idle under 4 GB available.
2. Distro `/usr/bin/llama-server` (libggml-hip) fails `ggml_cuda_init: no ROCm-capable device is detected` — the working ROCm baseline is the fork's `build-hip-q5kv`.
3. Host memory is genuinely tight: with the model loaded, `MemAvailable` sits near 7–9 GB and zram swap usage is 26–34 GB. Tier-2 sizing (plan 0.5) must respect this.
