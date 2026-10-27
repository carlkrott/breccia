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
| 12 | Dynamic **LRU expert cache** (~10–12 k hot experts across both GPUs) | **PARTIAL, high value** — static per-block coverage to **30 blocks** is the largest single-stream win so far: 11.82 → **16.94 tok/s** (59.0 ms), lane 47.5 → 13.5 ms, expert bytes 799 → 301 MiB/token (§8ak). Per-**expert** LRU is still open; it is the only way past the ~44 GiB full-coverage wall | §8ak ladder |
| 13 | **Layer L+1 route predictor** + prefetch one layer early | **MEASURED, rescoped** — block L+1's prelude consumes the residual that L's lane produces, so the lane is a *serial section* inside a stream with no GPU work to hide under it. What's left to hide is the lane's **data** (301 MiB ≈ 13.5 ms), not its compute (§8ak). Applies to prefill and to the 18 still-hosted blocks | §8ak |
| 14 | **Route-tail skipping** (Strata: skip lowest-weight top-k expert below ε when not resident) | **OPEN** — not attempted; it now matters *less* on the decode path (only 18 blocks are host-side) and more as a way to shrink the 301 MiB/token that remains | — |
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
| 1 | **`wait-gpu` drain at decode** (33.4 ms of a 58 ms token): attribute it (execution vs overhead), then kernel fusions (plan #18) and/or pre-recorded CB replay | needs ≈ −10 ms to reach 20.79 tok/s on one stream | the lane is now only 13.5 ms and is a serial section; the drain is 57 % of the token |
| 2 | **Head placement** (`head` 7.1 ms sits on the MI50) | ~−4 ms/token, config-level | nearly free |
| 3 | **Coverage past 30 blocks** (per-expert LRU, #12) | full block coverage needs ~44 GiB; per-expert is the only route | the ladder's gain was linear in *blocks freed*, not in bytes |
| 4 | **Quantized KV (#21)** + precision tail | unlocks 4–6 agent slots at 32K–64K (plan Gate G) | gates the whole multi-agent goal |
| 5 | **Radix prefix sharing (#28/29)** | agentic workloads share a huge system prompt; re-prefill dominates (6.0–6.45 tok/s/agent) | biggest real-world agent win |
| 6 | **Route-tail skipping (#14)** | shrinks the remaining 301 MiB/token | now a second-order lever on decode, first-order on prefill |
| 7 | **`external_memory_host` zero-copy experts (#10)** | unknown — research round 9 measuring the crossover | could remove the CPU lane for the 18 hosted blocks |
| 8 | **NVMe direct tier (#11)** | unknown — round 9 measuring 4 KiB random IOPS | matters only for a cold cache |
| 9 | **VTC + spec constants (#34/37)** | portability, not speed | user asked for agnosticism; adaptive coverage (this round) is the first piece |
| 10 | **MTP/speculative (#26)** | 0.60× when tried; needs the cache and the drain fixed first | plan's own trap: must ride the expert cache |
| 11 | **Server + continuous batching (#27/33)** | product surface; needs 3–5 first | — |

## Deliberate deviations from the plan (and why)

- **Layer-pipeline split rejected** (Gate A): MI50's gen3×4 link makes activation shipping the bottleneck.
  Expert-parallel placement + device-per-stream is the implemented alternative.
- **GPU-resident coverage is capped** (§8ag): the plan's "~10–12 k hot experts in VRAM" needs ~35 GiB;
  per-expert LRU (#12) is the VRAM-efficient form of that idea and is still open — the block-granular
  version we swept cannot reach it.
- **The plan's "20 → 45–55 tok/s via expert cache"** was substantially right after all, and one earlier
  reading of ours was wrong: §8ag concluded "coverage is capped and is not the lever" from a 4-block step
  (9 → 13 = −7.5 %). The ladder in §8ak (9 → 16 → 26 → 30 blocks) gives **+43 % on one stream**, because
  the lane only pays for blocks that are *not* resident. The lesson is about step size, not about the idea.
  What the earlier round did establish correctly is the *wall*: full block coverage needs ~44 GiB against
  32 GiB of VRAM, so the plan's "~10–12 k hot experts in VRAM" needs per-**expert** residency (#12) to go
  further.
- **The route predictor (#13) does not hide decode lane compute**: block L+1's prelude consumes the
  residual that block L's lane produces, so at decode the lane is a serial section. It hides the lane's
  *data* (13.5 ms at 301 MiB/token) and it still applies to prefill, where T tokens share one lane pass.
- **MTP before the cache** was tried and lost 40 %; it is ordered after #1–#4.
