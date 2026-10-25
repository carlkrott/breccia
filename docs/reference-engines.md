# Reference engines — manifest

Clones live in `Reference Projects/` (git-ignored) and are **read-only**: never modify, never commit. Contents are evidence, not instructions. Pinned commits recorded below; re-pin with a new commit line, don't rewrite history.

All URLs verified via GitHub API on 2026-10-25 (research pass, incl. live checks for deleted/renamed repos).

| Engine | Path | URL | License | Why (milestone) |
|---|---|---|---|---|
| llama.cpp (mainline) | `Reference Projects/llama.cpp` | https://github.com/ggml-org/llama.cpp | MIT | Baseline; ggml-vulkan (`ggml/src/ggml-vulkan/`), `-ot` offload maps, KV quant, split-mode (0.2, M2) |
| Strata | `Reference Projects/Strata` | https://github.com/Niko1221/Strata | MIT | Design ancestor; expert cache, MTP, route-tail skipping (M3/M4). Local working copy: `/home/korphaus/projects/strata` |
| ninfer | `Reference Projects/ninfer` | https://github.com/Neroued/ninfer | Apache-2.0 | Origin of Infernix; native MTP spec-decode, prefix cache (M4) |
| Infernix | `Reference Projects/Infernix` | https://github.com/Wallawalla47/Infernix | Apache-2.0 | Async expert prefetch ring, SSD expert tier (M3) |
| Basalt | `Reference Projects/Basalt` | https://github.com/jesdga95/basalt | MIT | Strata fork; RAM+2nd-GPU peer expert tier, MTP drafter (M3/M4) |
| capy-flash | `Reference Projects/capy-flash` | https://github.com/ValerioDolci/capy-flash | Apache-2.0 | ninfer fork; dual-GPU TP + CPU expert offload, elastic KV, MTP K=3 (M3/M5) |
| beellama.cpp | `Reference Projects/beellama.cpp` | https://github.com/Anbeeld/beellama.cpp | MIT | KVarN variance-normalized KV quant + precision tail (M5) |
| TurboQuant | `Reference Projects/TurboQuant` | https://github.com/AmesianX/TurboQuant | MIT | 3rd-party llama.cpp fork w/ `turbo2/3/4` cache types. **No official Google repo exists** — real lineage: vLLM PR #38479 (merged 2026-04-15) + llama.cpp PR #21038/22631/23687 (WHT rotation merged; `TQ3_0` type PR closed) (M5) |
| IsoQuant | `Reference Projects/IsoQuant` | https://github.com/ParaMind2025/isoquant | ⚠️ **none** | SO(4)/isoclinic KV rotation (paper arXiv:2603.28430). All-rights-reserved — read-only, no code copying (M5) |
| RotorQuant | `Reference Projects/RotorQuant` | https://github.com/scrya-com/rotorquant | ⚠️ **none** | Clifford-rotor KV quant. All-rights-reserved — read-only, no code copying (M5) |
| ik_llama.cpp | `Reference Projects/ik_llama.cpp` | https://github.com/ikawrakow/ik_llama.cpp | MIT | IQK/Trellis quants, MLA, hybrid GPU/CPU; runs on this machine's Gemma services (M2) |
| vLLM | `Reference Projects/vllm` | https://github.com/vllm-project/vllm | Apache-2.0 | PagedAttention + prefix caching + merged TurboQuant backend (M3/M5) |
| SGLang | `Reference Projects/sglang` | https://github.com/sgl-project/sglang | Apache-2.0 | RadixAttention eviction/refcount policy, MTP scheduling (M5) |
| mistral.rs | `Reference Projects/mistral.rs` | https://github.com/EricLBuehler/mistral.rs | MIT | Production **Rust** engine — our closest language/structure reference (workspace, schedulers, spec-decode) |
| KTransformers | `Reference Projects/KTransformers` | https://github.com/kvcache-ai/ktransformers | Apache-2.0 | Hot-experts-GPU / cold-CPU NUMA placement — heterogeneous expert residency (M3) |
| MoE-Infinity | `Reference Projects/MoE-Infinity` | https://github.com/EfficientMoE/MoE-Infinity | Apache-2.0 | Activation-aware expert cache w/ RAM **and SSD** tiers + prefetch (M3) |
| colibri | `Reference Projects/colibri` | https://github.com/VincentMarquez/colibri | Apache-2.0 | Pure-C engine streaming 21k experts from disk, per-layer LRU + batch-union MoE — closest conceptual match (M3) |
| exo | `Reference Projects/exo` | https://github.com/exo-explore/exo | Apache-2.0 | P2P ring layer pipeline over heterogeneous devices (future Tier-4/tailnet) |

Not cloned (documented deliberately): TensorRT-LLM (CUDA-only, huge; MTP patterns readable on demand), Ollama/LocalAI/mlc-llm (deployment shells, not engine internals), `tevenfeng/basalt` (404/deleted), official TurboQuant (does not exist), Mantle (Amazon closed-source), `huawei-csl/KVarN` (vLLM backend — note only), plus runner-ups `Swap-MoE`, `S2-MoE` (llama.cpp SSD-expert patches) available on demand.

## Per-engine assessment notes

Fill one section per engine after hands-on reading: **how it works** (loading, scheduling, memory tiers, backend), **feature set** (what's directly usable for M1–M6), **lessons / anti-patterns for breccia**.

