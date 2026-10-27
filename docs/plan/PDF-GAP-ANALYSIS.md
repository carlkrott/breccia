# Design-plan coverage audit — Gemini plan vs. built engine

**Created 2026-10-27.** Purpose: the original design conversation (`Optimizing Llama.cpp for
Heterogeneous AMD GPUs.pdf`, 19 pages, 2026-10-25) is the scope source. This file maps **every
concrete proposal in it** to the current state of the engine, so work is neither repeated nor
silently dropped. Status is verified against the crates and the phase reports, not from memory.

Legend: **BUILT** (in the tree, measured) · **PARTIAL** (a piece exists, the proposal does not) ·
**OPEN** (not started) · **DEFERRED** (user decision) · **REJECTED** (measured, and it lost).

## 1. Foundations (probe → profile → solver)

| # | Plan proposal | Status | Evidence |
|---|---|---|---|
| 1 | Rust workspace: `ash` + `gpu-allocator` + tokio, per-crate benchable modules | **BUILT** | `crates/` (11 crates incl. `breccia-engine`), `docs/plan/BRECCIA-PLAN.md` |
| 2 | Hardware discovery: Vulkan devices/heaps/extensions/wave width, CPU/cache/NUMA, storage | **BUILT** | `breccia-profile`; `docs/phase-1/profile-report.md` |
| 3 | Calibration micro-benchmarks: VRAM GB/s, TFLOPS, PCIe per queue family, P2P, storage IOPS | **BUILT** (P2P + storage IOPS probed, not swept) | `docs/phase-0/02-vulkan-raw-probe.md` |
| 4 | Cost-model solver by operational intensity (compute vs bandwidth roles) | **PARTIAL** — roles are measured and assigned, but placement is block-affinity + explicit split, not graph-solved | `docs/phase-0/gate-A-report.md` |
| 5 | CLI surface `--reprofile` / `--auto-partition` / `--dry-run-plan` / `--override-role` | **BUILT** for profile/dry-run/override; `--auto-partition` is the default path | `breccia-flags`, 52 tests |
| 6 | Persistent `hw_profile.json` + BLAKE3 fingerprint + invalidation | **BUILT** | `docs/phase-1/profile-report.md` |
| 7 | Accept the existing HIP launch flags verbatim (`ROCm0/1` → devices by PCI identity) | **BUILT** | Gate B report; `breccia serve --dry-run-plan` |

## 2. Streaming, tiering, expert cache

| # | Plan proposal | Status | Evidence |
|---|---|---|---|
| 8 | Dedicated transfer queue + timeline semaphores + DMA ring | **BUILT** | Phase 2: RX 28.25–28.71 GB/s, 100 % overlap; MI50 2.8 GB/s, ~0 % |
| 9 | Host-pinned staging (Tier 2) | **BUILT** (staging ring); pinned-pool sizing blocked by `ulimit -l` 8 MiB → needs `LimitMEMLOCK=infinity` | `docs/phase-0/04-host-memory.md` |
| 10 | `VK_EXT_external_memory_host` zero-copy host weights | **PARTIAL** — capability probe + verified import exists (`breccia-vk/src/transfer.rs:probe_host_memory`), never used to execute host-resident experts | research round 9 pricing it |
| 11 | NVMe direct tier: `io_uring` + `O_DIRECT` 4 KiB random reads for cold experts / PLE blocks | **OPEN** — io_uring only probed; PLE reads go through page cache; btrfs read-ahead cut to 128 KiB instead | `docs/REPRODUCE.md` |
| 12 | Dynamic **LRU expert cache** (~10–12 k hot experts across both GPUs) | **OPEN** — residency is per *block* (9 → 13 blocks measured), not per *expert*; +4 blocks = −7.5 % of the token, full coverage needs ~35 GiB vs 16 GiB/card (§8ag) | not built |
| 13 | **Layer L+1 route predictor** + prefetch one layer early | **IN PROGRESS** (this round; priced ~38 ms/token in §8af/§8ag) | — |
| 14 | **Route-tail skipping** (Strata: skip lowest-weight top-k expert below ε when not resident) | **IN PROGRESS** (this round, behind a flag, with the quality curve) | — |
| 15 | Inter-GPU **layer-pipeline split** with activations over PCIe | **REJECTED** — MI50 is PCIe gen3×4 ≈ 2.9 GB/s with ~0 % overlap; expert-parallel placement replaced it | Gate A report |

## 3. Kernels & attention

| # | Plan proposal | Status | Evidence |
|---|---|---|---|
| 16 | Dequant + GEMV/GEMM for the model's quant types (IQ3_XXS/IQ3_S/IQ4_XS/Q4_K/Q5_K/Q6_K/…) with LDS codebooks | **BUILT** — 14 types, 271–470 GB/s on the RX, exact dequant, bit-identical GEMM/GEMV | Phase 3; §8aa |
| 17 | **Fused gate+up+SwiGLU** (kill intermediate round-trips) | **MEASURED, NOT WIRED** — 3.1× (m=1) / 5.3× (m=8), 2 dispatches per expert | research 06; §8ad |
| 18 | Fused RMSNorm + QKV, fused PLE + RMSNorm + QKV | **OPEN** | — |
| 19 | Register-tile dequantizers with non-uniform subgroup ops | **PARTIAL** — register-tile dequant is done; subgroup-uniform/ballot tricks are not used | research 06 |
| 20 | PagedAttention with block tables (16/32-token pages) | **BUILT** (paged f16 KV, QSA + indexer + GDN) | Phase 3/4 |
| 21 | **Quantized KV** (Q8_0/Q4_0, or 4-bit rotated; K at higher bits than V) | **OPEN** — f16 only. This is the item that gates 4–6 agents (plan Gate G) | research round 9 pricing it |
| 22 | BeeLlama precision tail (recent N tokens f16, older compressed) | **OPEN** (depends on 21) | — |
| 23 | TurboQuant WHT / RotorQuant Givens rotations for KV | **OPEN** (depends on 21) | — |
| 24 | Unified attention descriptor: sliding window / strided sparse / read-only injection in one shader | **PARTIAL** — Qwen's QSA + indexer are implemented natively; the generic mask descriptor is not | `docs/specs/qwen4exp-architecture.md` |
| 25 | Engram static n-gram tables (hash → memory rows) | **N/A for this model** — the equivalent is Qwen's PLE n-gram table, which *is* implemented; the *read-only static injection* idea is not | Phase 3 PLE work |

## 4. Drafting, agents, server

| # | Plan proposal | Status | Evidence |
|---|---|---|---|
| 26 | MTP head / DFlash2 / n-gram copy-drafter + batched tree verification | **OPEN** — earlier attempt measured **0.60×** (doubled expert reads); plan's own warning: drafting must ride the expert cache | session history; plan Phase 5 note |
| 27 | Continuous batching, iteration-level scheduling, agent state machine (tool-wait eviction) | **OPEN** — `breccia-server` is a stub; the bench harness streams but does not schedule | — |
| 28 | Radix-tree prefix sharing across agents | **OPEN** — no `radix` anywhere in `crates/` | — |
| 29 | Disk-persisted prefix cache | **OPEN** | — |
| 30 | S-LoRA multi-adapter swapping | **OPEN** (no adapters in use yet) | — |
| 31 | VRAM governor with Nominal/Pressure/Critical watermarks | **PARTIAL** — global policy is gpu-manager's; the engine sizes arenas and refuses, it does not evict | — |
| 32 | Observability: cache-hit ratios, DMA saturation, per-device util, exporter | **PARTIAL** — `BRECCIA_TRACE_*` stage timers + the calibrated timestamp recipe; no exporter | `docs/research/07` |
| 33 | OpenAI/Anthropic-compatible server + `/v1/decide` | **OPEN** | — |

## 5. Model/hardware agnosticism (the plan's Pillar 4)

| # | Plan proposal | Status | Evidence |
|---|---|---|---|
| 34 | Virtual Tensor Container: format-agnostic descriptors, zero-copy hybrid loading (GGUF + SafeTensors in one DAG), on-the-fly swizzle shaders | **OPEN** — a zero-copy GGUF loader + pack format exist, but there is no abstraction layer over formats | — |
| 35 | Parametric shader library decoupled from the op graph | **PARTIAL** — dequant units are shared over one `gemm_body.glsl`, but geometry lives in the shader, not in spec constants | research 06 |
| 36 | Declarative compute graph (acyclic op list) instead of a hand-written forward pass | **PARTIAL** — `graph.rs` is hand-written with a chunk planner | — |
| 37 | SPIR-V **specialization constants** for wave size (64 on MI50, 32 on RX) instead of hardcoding | **OPEN** — one binary, driver default wave64; no `VK_EXT_subgroup_size_control` path | **IN PROGRESS** this round |

## 6. Mesh

| # | Plan proposal | Status |
|---|---|---|
| 38 | Tailnet mesh: QUIC transport, node manifests, Tier-4 remote KV leases, dead-man fallback, remote expert sharding | **DEFERRED** (explicit user decision: local engine first) |

---

## Ranked remaining queue (by measured value, not by plan order)

| Rank | Item | Measured / priced value | Why here |
|---|---|---|---|
| 1 | **Intra-stream lane↔GPU overlap** (route predictor + concurrent CPU/both-GPU expert split) | single stream 84.9 ms → ≈48 ms at best (**11.8 → ~20.8 tok/s**); lane 47.5 ms + GPU 37.5 ms are strictly serial at 1 stream | the only item that moves the user's headline metric |
| 2 | **Route-tail skipping** | cuts 799 MiB/token of expert traffic directly | reduces both PCIe floor (27.8 ms) and lane compute |
| 3 | **Quantized KV (#21)** + precision tail | unlocks 4–6 agent slots at 32K–64K (plan Gate G) | gates the whole multi-agent goal |
| 4 | **Radix prefix sharing (#28/29)** | agentic workloads share a huge system prompt; re-prefill is the dominant cost today (6.0–6.45 tok/s/agent) | biggest real-world agent win after the lane |
| 5 | **Fused gate+up+SwiGLU (#17)** | 3.1×/5.3× per expert role, 2 dispatches instead of 3 | cheap, measured, purely additive |
| 6 | **Per-resource barrier tracking** | ~4.5 ms/token at 1 100 → ~400 barriers | only structural launch-cost lever left |
| 7 | **`external_memory_host` zero-copy experts (#10)** | unknown — research round 9 measuring the crossover | could remove the CPU lane entirely |
| 8 | **NVMe direct tier (#11)** | unknown — round 9 measuring 4 KiB random IOPS | matters only for a cold cache; page cache is currently fine |
| 9 | **VTC + spec constants (#34/37)** | portability, not speed | user asked for agnosticism; bounded version is in this round |
| 10 | **MTP/speculative (#26)** | 0.60× today; needs 1–4 first | plan's own trap: must ride the expert cache |
| 11 | **Server + continuous batching (#27/33)** | product surface; needs 3–4 first | — |

## Deliberate deviations from the plan (and why)

- **Layer-pipeline split rejected** (Gate A): MI50's gen3×4 link makes activation shipping the bottleneck.
  Expert-parallel placement + device-per-stream is the implemented alternative.
- **GPU-resident coverage is capped** (§8ag): the plan's "~10–12 k hot experts in VRAM" needs ~35 GiB;
  per-expert LRU (#12) is the VRAM-efficient form of that idea and is still open — the block-granular
  version we swept cannot reach it.
- **The plan's "20 → 45–55 tok/s via expert cache"** assumed the lane would not dominate. Measurement
  says the lane is the constraint, which is why #1 and #2 outrank #12.
- **MTP before the cache** was tried and lost 40 %; it is ordered after #1–#4.
