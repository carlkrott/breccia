# Paired research (standing rotation)

Every implementation / test round is paired with **one research subagent** that does two things:

1. **Vulkan capability audit** — anything in the driver/extension/parameter surface we are not exploiting, tied to a current bottleneck.
2. **One engine audit** — a single inference engine (not yet covered) reviewed for workflows, methods and specific advantages we have not tried, with an adaptation sketch for our Vulkan engine.

Findings land in `docs/research/NN-<engine>.md`. This index tracks coverage so we never re-audit the same engine twice.

| # | Engine | Status | Round |
|---|---|---|---|
| 1 | MLC-LLM (Vulkan/TVM backend) | in flight | 2026-10-26 |
| 2 | PowerInfer (hot/cold neuron offload) | pending | — |
| 3 | exllamav2 (quant + cache strategies) | pending | — |
| 4 | LMDeploy / TurboMind (scheduling, KV) | pending | — |
| 5 | TensorRT-LLM (graphs, in-flight batching) | pending | — |
| 6 | ncnn (pure Vulkan compute patterns) | pending | — |
| 7 | DeepSpeed-Inference (tensor/KV partitioning) | pending | — |
| 8 | FasterTransformer (fused kernels) | pending | — |
| 9 | Ollama (deployment/runtime ergonomics) | pending | — |
| 10 | SGLang deep-dive (radix + prefill chunking) | pending (only surveyed) | — |

Already covered in `docs/phase-4/02-untried-levers.md` (do not re-audit unless new): llama.cpp, Strata, ninfer, Infernix, Basalt, capy-flash, beellama, TurboQuant, IsoQuant, RotorQuant, ik_llama.cpp, vLLM, SGLang (survey level), mistral.rs, KTransformers, MoE-Infinity, colibri, exo.

Standing rule: a research subagent never changes code; it produces a ranked, measurable recommendation list. Implementation agents consume it.
