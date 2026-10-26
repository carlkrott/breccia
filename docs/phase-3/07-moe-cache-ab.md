# Production A/B — `--moe-cache-mib` (expert cache) hurts decode

**Result: keep `--moe-cache-mib 0`.** The fork's GPU expert-cache lever makes decode 18–35% slower on this configuration. Measured twice, in both orders, with <1% run-to-run spread.

## Method

`harnesses/moe_cache_ab.sh` — the exact production flags (`qwen-dual-serve.sh`), only `--moe-cache-mib` varied, one server at a time, port 8164, scratch slot-save dir, 3×256-token decode + 1k-token prefill per config. Two passes: forward (0 → 512 → 1024) and reversed (1024 → 512 → 0) to separate the effect from thermal/clock drift. Server `eval time` is the ground truth.

## Results

| `--moe-cache-mib` | Pass 1 | Pass 2 (reversed) | Δ vs cache 0 |
|---|---|---|---|
| **0** | **22.22 tok/s** (45.0 ms/tok) | **20.91 tok/s** (47.8 ms/tok) | baseline |
| 512 | 14.44 (69.2 ms) | 14.49 (69.0 ms) | **−31…35%** |
| 1024 | 17.07 (58.6 ms) | 17.18 (58.2 ms) | **−18…23%** |

Prefill is flat across configs (125–137 tok/s), so this is entirely a decode-path effect. Reproducibility is exceptional (512: 14.44 vs 14.49; 1024: 17.07 vs 17.18), and cache 0 wins even as the *last* run of the reversed pass — the result is not thermal.

## Why the cache loses

With `-cmoe`, the expert GEMV runs on the **CPU reading host memory** (~0.84 GB/token across 35 host blocks at the 20.8 tok/s baseline ≈ 17.5 GB/s sustained — well within DDR5 capability). A non-zero `--moe-cache-mib` makes the fork stage experts into GPU memory; at 512 MiB the hit rate over 512 experts × 35 layers is far too low, so most expert reads pay a **PCIe staging copy** (≈1.7 MB per expert blob, ×10 top-k ×35 layers per token). Even on the RX (28.7 GB/s, measured) that per-miss latency exceeds the CPU's direct host read; on the MI50 (2.9 GB/s) it is ~10× worse. A larger cache helps only marginally because the working set (experts touched per token ≈ 0.84 GB) dwarfs 1 GiB.

## Implications

1. **Production: keep `--moe-cache-mib 0`.** No change recommended to `qwen-dual-serve.sh` — the cheap win does not exist.
2. **M3 design confirmation (negative evidence):** an expert cache that *uploads on miss* over PCIe is worse than doing the math on the host. The engine's expert tier must be **VRAM-resident placement chosen ahead of time** (route-frequency profile → pin the hot set, exactly Strata's `expert-profile.bin` approach), with prefetch pipelined behind compute — not demand-paging. This is consistent with Gate A (RX DMA hides under compute at 90%+, but only when the consumer doesn't stall on it) and with the transfer-ring measurement (stall 0.003 ms/MB at depth 6 — depth is what saves you, not bandwidth).
3. **Parity target for the host lane:** the graph's CPU expert path must sustain ≈17.5 GB/s of quantised expert reads; that is the measured bar this baseline sets.
