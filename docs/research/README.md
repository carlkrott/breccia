# Paired research (standing rotation)

Every implementation / test round is paired with **one research subagent** that does two things:

1. **Vulkan capability audit** — anything in the driver/extension/parameter surface we are not exploiting, tied to a current bottleneck.
2. **One engine audit** — a single inference engine (not yet covered) reviewed for workflows, methods and specific advantages we have not tried, with an adaptation sketch for our Vulkan engine.

Findings land in `docs/research/NN-<engine>.md`. This index tracks coverage so we never re-audit the same engine twice.

| # | Engine | Status | Round |
|---|---|---|---|
| 1 | MLC-LLM (Vulkan/TVM backend) | **done** — `01-vulkan-and-mlc.md` | 2026-10-26 |
| 2 | PowerInfer (hot/cold neuron offload) | **done** — `02-vulkan-followups-and-powerinfer.md` | 2026-10-26 |
| 3 | exllamav2 (quant + cache strategies) | **done** — `03-io-and-exllamav2.md` | 2026-10-26 |
| 4 | LMDeploy / TurboMind (scheduling, KV) | **next** | — |
| 5 | TensorRT-LLM (graphs, in-flight batching) | pending | — |
| 6 | ncnn (pure Vulkan compute patterns) | pending | — |
| 7 | DeepSpeed-Inference (tensor/KV partitioning) | pending | — |
| 8 | FasterTransformer (fused kernels) | pending | — |
| 9 | Ollama (deployment/runtime ergonomics) | pending | — |
| 10 | SGLang deep-dive (radix + prefill chunking) | pending (only surveyed) | — |

Already covered in `docs/phase-4/02-untried-levers.md` (do not re-audit unless new): llama.cpp, Strata, ninfer, Infernix, Basalt, capy-flash, beellama, TurboQuant, IsoQuant, RotorQuant, ik_llama.cpp, vLLM, SGLang (survey level), mistral.rs, KTransformers, MoE-Infinity, colibri, exo.

Standing rule: a research subagent never changes code; it produces a ranked, measurable recommendation list. Implementation agents consume it. **Next in rotation: #4 LMDeploy/TurboMind** (scheduling, KV, and continuous batching — the remaining open question is prefill scheduling, not quantisation).

Round verdicts (one line each, so a future round knows what was learned without re-reading):

* **01 — MLC-LLM.** Its Vulkan backend is a SteamDeck-*portability* target (`thread_warp_size=1`, no subgroup ops, no pipeline cache), so nothing in its kernels is worth porting; the transferable value is compile-time memory planning, an `indptr`-driven single-dispatch group MoE, and radix-tree prefix reuse with COW sequence forking. On our side the audit found f32 **storage** atomics do exist on both cards (`atomicExchange`; `atomicAdd` does not, but **shared-memory f32 `atomicAdd` does**), a free `RADV_EXPERIMENTAL=transfer_queue` SDMA family for the streaming lane, and that **21–29 ms of the 70.8 ms GPU half is not arithmetic** (per-dispatch `ALL_COMMANDS` barrier).
* **02 — measured follow-ups + PowerInfer.** Barriers are **free as packets** (0.01 µs); what costs 1.1–4.4 µs/dispatch is the *launch serialization* they force, so `full→sync2` is worth **−2.5…−6.2 ms/token** and nothing more — the 21–29 ms attribution of round 1 is **refuted**, and event-based waits, wave32 (0.1 % on a latency probe), the transfer queue as a global flag (8.6× *slower* for on-card copies on the MI50) and denormal modes all measured ≈0 or negative. What does pay: `vkCmdCopyBuffer2` region arrays (**0.13 µs vs 1.0–1.5 µs of host record per region**, −5…−8 ms/chunk of prefill host time), ~13–15 µs per queue submission, and PowerInfer's two structural ideas — **per-neuron ILP residency** (`gpu_idx`+`gpu_bucket`, 75 % of FFN in 12.4 GiB) and **CPU node ‖ GPU node joined by one `add`**, which is the shape that removes our 38–50 ms host lane from the critical path.
* **03 — IO/page-cache + exllamav2.** The model-bigger-than-RAM problem is **one sysctl, mis-set**: `read_ahead_kb` must be written to the **btrfs BDI** (`/sys/class/bdi/btrfs-1/`), not the block device's queue (writing the latter changes nothing), and 8192 → **128 KiB** takes cold expert reads from **0.7–1.1 to 4.7–5.4 GB/s** with amplification **9× → 1.05×** — i.e. a cold run 28 → 4 s and the merged-prefill byte floor from 21 to ~140 tok/s/agent. With that right, `O_DIRECT` (5.9 GB/s cold but 10× worse than the 58 GB/s warm page-cache path), `FADV_RANDOM`/`MADV_RANDOM` (llama.cpp's env knobs: neutral-to-*negative* on btrfs and on 462–900 KiB regions), `MADV_SEQUENTIAL`, prefetch depth > 2 (queue-limited: `nr_requests 256 × max_sectors 128 KiB = 32 MiB` ≈ 1.7 blocks) and `MADV_POPULATE_READ` (3.9 GB/s, blocking) are all dominated. The PLE shard is a *different* pathology: one 28.8 GiB tensor of **90-byte records**, 65 µs per cold row, 48× btrfs-extent over-read, 0.5 µs warm — residency + 32-way parallel gather, not advice. exllamav2 contributes exactly two things: the **measure-per-module-then-anneal-the-bit-budget** allocation method (only usable for KV — our weights are a fixed artifact behind a bit-identical gate) and the structural fact that its per-layer launch count is **~2 against our ~31**, whose Vulkan form is **`vkCmdDispatchIndirect` over a pre-recorded command buffer** (a replayable "graph" without the 6-deep ring) on top of the fused MoE already queued as item 7. Its CUDA graphs, fp8 KV and `bits_prop` mixed-rate format have no Vulkan equivalent and matter little.
