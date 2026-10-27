# 04 — Vulkan launch-count, priced — plus LMDeploy / TurboMind

Paired research round 4. Date: 2026-10-26. **Read-only**: no engine code was touched. Two parts:

* **Part A** — the launch-count gap (`~31` dispatches per block against exllamav2's `~2`) priced on
  both cards with a purpose-built microbench, and four candidate mechanisms (indirect dispatch,
  pre-recorded command buffers, `VK_EXT_device_generated_commands`, fused MoE) ranked by measured cost.
* **Part B** — audit of **LMDeploy / TurboMind** (engine #4 in the rotation), focused on in-flight
  batching, KV management, attention-kernel shapes and launch amortisation.

Current state this is scored against (this round): 2-stream steady-state decode **13.3–14.6 tok/s
aggregate** (`docs/phase-4/01` §8x; the 20.3 gate is being reconciled), prefill **6.45 tok/s/agent**
(2×300 warm), GPU half **1 509 dispatches/token, ~31/block**, recording **0.32 ms/token**.

---

## Part A — what a dispatch actually costs on these two RADV cards

### A.0 Headline

**The launch count is not the problem; one barrier per dispatch is.** Measured marginal cost of a
dispatch on the RX 6950 XT:

| | RX 6950 XT | Radeon VII (MI50) |
|---|---:|---:|
| dispatch, **no** barrier between dispatches | **0.007 µs** | **0.031 µs** |
| dispatch, compute→compute barrier after each | **0.650 µs** | **2.867 µs** |
| dispatch, `vkCmdDispatchIndirect` + barrier | 1.418 µs | 4.510 µs |

A dispatch that does not have to drain the previous one costs *seven nanoseconds* of GPU timeline on
the RX (3 024 chained dispatches in 20.6 µs). The 1 509 dispatches/token therefore cost **0.98 ms/token
on the RX and 4.32 ms on the MI50** — i.e. **1.4 % / 6.1 % of the 70.8 ms GPU half**, and almost all of
it is the barrier, not the launch. This is the quantitative form of round 2's conclusion ("barriers are
free as packets; the *serialisation* they force is what costs") and it is why `dispatch_batch::record()`
emitting exactly one `barrier_compute_on` per dispatch (`crates/breccia-kernels/src/dispatch_batch.rs:756-763`)
means **barriers/token == dispatches/token** today.

### A.1 Method

Standalone probe built *outside* the tree (`<SCRATCH>/launchprobe`, a copy of
`harnesses/vk-probe` plus a `--launch` mode), RADV/mesa 26.2.4, Vulkan 1.4.354, headless compute, one
device per invocation, trivial shader (wg 256, one dependent store) unless stated. Barrier is a
`vkCmdPipelineBarrier` with `COMPUTE_SHADER → COMPUTE_SHADER` and `SHADER_READ|WRITE` masks (the
`sync2` shape the engine emits). Timings: host = `Instant` around `vkResetCommandBuffer`→`vkEndCommandBuffer`;
GPU = `vkCmdWriteTimestamp` brackets, median of 8–40 repetitions.

```
timeout 120 ./target/release/vk-probe --device 0 --launch    # RX 6950 XT  (0000:03:00.0)
timeout 150 ./target/release/vk-probe --device 1 --launch    # Radeon VII  (0000:0e:00.0), run first
```

### A.2 Host-side recording cost (no submit)

| arm | RX | MI50 |
|---|---:|---:|
| `vkCmdDispatch` + barrier | 44.8 ns | 46.2 ns |
| `vkCmdDispatch`, no barrier | 13.4 ns | 13.2 ns |
| `vkCmdDispatchIndirect` + barrier | 46.3 ns | 49.0 ns |
| engine-shaped record (bind set + push constants + barrier + dispatch) | 67.9 ns | 68.5 ns |
| `vkUpdateDescriptorSets` (1 set, 1 binding) | 19.6 ns | 24.3 ns |
| `vkCmdExecuteCommands` of a 32-dispatch secondary | 473.6 ns | 340.2 ns |

Engine-shaped recording of 1 509 dispatches is therefore **≈ 0.10 ms/token**, against the engine's
measured 0.32 ms — the 0.22 ms difference is `vkAllocateDescriptorSets` per dispatch plus per-dispatch
`Vec` churn in `dispatch_batch::record`, not command recording. **Recording is already 4× cheaper than
the engine thinks**, and even the full 0.32 ms is 0.4 % of a ~75 ms step.

Submission is not a factor either: with 128 dispatches per submit, 16 submits in flight, throughput was
**100.7 µs/submit on the RX against 88.7 µs of GPU time** (378.8 vs 364.6 on the MI50) — submits
overlap GPU execution and cost ~12–14 µs of host each, hidden behind the fence.

### A.3 The four candidate mechanisms, priced

**(1) `vkCmdDispatchIndirect`** — legal, works, and **costs more than it saves**. Geometry lives in a
host-visible `INDIRECT_BUFFER` (frozen per token, zero host cost), but the marginal GPU cost per
dispatch is **+0.77 µs on the RX and +1.65 µs on the MI50** over a direct dispatch (1.418 vs 0.650;
4.510 vs 2.867). RADV has to serialise the geometry fetch before the launch. For 1 509 dispatches that
is **+1.16 ms/token (RX) / +2.49 ms (MI50)** — strictly negative. Its only benefit is that geometry stops
being re-derived on the host, which is worth ≤0.3 ms.

**(2) Pre-recorded command buffers + `vkCmdExecuteCommands`** — replay is free on the GPU and saves
only host time. 128 dispatches as 4 pre-recorded 32-dispatch secondaries, primary re-recording only the
`executeCommands` calls:

| | RX gpu / host | MI50 gpu / host |
|---|---:|---:|
| re-record the primary every step | 88.7 / 41.5 µs | 364.6 / 51.1 µs |
| replay pre-recorded secondaries | 89.2 / 36.0 µs | 357.0 / 41.8 µs |

(GPU difference is noise; "host" includes a ~34 µs fence create/destroy, so the real recording delta is
~8 µs → ~1.5 µs per 128 dispatches.) Mechanism confirmed: **RADV re-submits secondaries with only a
descriptor-set update** (19.6 ns per set, no re-record), so a per-slot descriptor-set ring + frozen
secondary CBs is valid. Ceiling: 0.32 ms → ~0.02 ms/token, **−0.3 ms/token, 0.4 % of a step**, and it does
not remove the 6-deep ring's *barrier* constraint, only its *geometry* constraint.

**(3) `VK_EXT_device_generated_commands`** — **present on both cards, and the wrong tool here.** Evidence
(`vulkaninfo`, both devices): `deviceGeneratedCommands = true`,
`maxIndirectCommandsTokenCount = 128`, `maxIndirectSequenceCount = 1 048 576`,
`supportedIndirectCommandsShaderStagesPipelineBinding = { COMPUTE }` (only), and the shader-stage list
includes `COMPUTE` on both gfx1030 and gfx906. So the compute-only path exists in mesa 26.2.4. But:
(a) every DGCE token is an *indirect* dispatch, so it inherits the +0.77/+1.65 µs measured above before
it saves anything; (b) its entire saving is the host recording cost, whose ceiling is 0.32 ms/token;
(c) `maxIndirectCommandsTokenCount = 128` means ≥12 layout switches per token anyway (le 12.5 µs);
(d) the workspace's binding crate is `ash 0.38.0+1.3.281`, which **has no
`VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`/`…PropertiesEXT` types at all** — adoption means an
ash bump plus hand-rolled `pNext` structs. Verdict: **do not pursue**; net expected effect negative.
This supersedes round 1's "lever 6 (deferred)" — it is now priced out, not just deferred.

**(4) Fused single-dispatch MoE** — the only launch-side item that is positive, but for structural rather
than launch reasons. Model facts (`docs/specs/qwen4exp-architecture.md:93`): `n_ff = 640` per expert,
`n_expert = 512`, `k = 10`, expert total **1.665 MiB** (gate+up+down). A fully fused per-(expert, row)
workgroup is cheap to build:

| constraint | number | consequence |
|---|---|---|
| hidden row in LDS (f32, `n_embd = 2560`) | 10.0 KiB | 4 rows = 40 KiB |
| SwiGLU intermediate (`n_ff = 640`) | 2.56 KiB | 4 rows = 10.2 KiB |
| LDS per workgroup on gfx10 (64 KiB) | 64 KiB | **≤4 rows/workgroup** with f32 staging, 8 if the row is re-read |
| weights read per (expert,row) | 1.665 MiB | unchanged — fusion saves launches, **not bytes** |
| gate/up as one GEMM (`[2·n_ff, n_embd]` concatenated, LMDeploy/vLLM shape) | 1 dispatch | halves the launch count of the pair |
| grid | `ceil(active_experts × rows / rows_per_wg)` | row map from a device buffer, **direct** dispatch |

Priced: 31 → ~8 launches/block removes 23 × 48 = 1 104 dispatches/token = **−0.72 ms (RX) / −3.2 ms
(MI50) per token**, i.e. **1–4 %**. Its real value is (i) it is the precondition for per-expert residency
(untried-levers item 7) and (ii) it removes the gate/up intermediate's VRAM round trip (~5 KiB/row, noise)
and one dependency per expert.

### A.4 Projection table (per token, 1 509 dispatches, 2 streams, ~75 ms step)

| lever | dispatch/barrier change | RX | MI50 | verdict |
|---|---|---:|---:|---|
| indirect dispatch for all | 0 → +0.77/1.65 µs each | +1.16 ms | +2.49 ms | **negative** |
| DGCE (compute) | host −0.3 ms, +indirect cost | −0.3 +1.16 | −0.3 +2.49 | **negative** |
| secondary-CB replay | host 0.32 → 0.02 ms | −0.30 ms | −0.30 ms | marginal, free |
| **barrier pruning** (1 509 → ~400) | −1 109 barriers | **−0.72 ms** | **−3.18 ms** | **best/cheapest** |
| fused MoE (31 → 8/block) | −1 104 dispatches | −0.72 ms | −3.18 ms | best structural |
| both | −2 000 boundaries | −1.3 ms | −5.7 ms | 1.7–7.6 % |

Sanity anchor from the engine's own history: `dispatch_batch.rs:758-762` records `barrier_all` at
4.4 µs (MI50) / 5.2 µs (RX) and compute-scoped `sync2` at 2.7 / 1.1 µs. My independent measurement gives
2.87 / 0.65 µs for a bare boundary — the same order, with the RX 1.7× cheaper than the engine's number.
**The MI50's boundary is 4.4× the RX's**, which is a placement rule in itself: *many small dependent
dispatches belong on the RX; the MI50 should get few, fat ones* (it is the same card whose transfer queue
and PCIe gen3×4 already make it the wrong home for fine-grained work).

### A.5 Best option and the minimal experiment

**Best: barrier pruning, then fused MoE. Neither needs indirect dispatch, DGCE, or a graph analogue.**

Minimal experiment (one flag, one bench, bit-identical by construction — moving a barrier cannot change
arithmetic):

1. `--barrier-group on|off` (default off) in `crates/breccia-kernels/src/dispatch_batch.rs`: `record()`
   takes a `Barrier::After`/`Barrier::Defer` argument; producer dispatches whose outputs are consumed only
   by a later join (the 10 expert GEMVs, which write distinct slots) take `Defer`, and the join takes one
   `After` (a `memoryBarrier` covering all deferred producers).
2. Instrument: `recorded()` already counts dispatches; add a `barriers()` counter, and print
   dispatches/token, barriers/token, `record_seconds()`, `wait_seconds()` in the existing step summary.
3. Bench (short form, per AGENTS.md): 2 streams × 64-token prefix, 8 decode steps, `--warm 1`, A/B/A in
   one window on **dual**, alternating `--barrier-group off/on/off`. Gate: bit-identical logits, and the
   *prediction* is −0.7 ms (RX-side work) to −3.2 ms (MI50-side work) of fence time per token, visible
   directly in `wait_seconds()` and in the step time.
4. Sweep one card at a time for attribution: MI50 first (`BRECCIA_KERNEL_DEVICE=0000:0e:00.0`), where the
   effect is 4.4× larger and therefore unambiguous even inside a noisy window.

A negative result here would be informative too: it would mean the barriers are not actually redundant,
i.e. the deferred producers do alias, and the only way down is fewer dispatches (fused MoE).

---

## Part B — LMDeploy / TurboMind audit

Clone: `git clone --depth 1 https://github.com/InternLM/lmdeploy` → `Reference Projects/lmdeploy`
(24 MB, HEAD `7ef6731`, 2026). Two backends in one repo: **TurboMind** (`src/turbomind/**`, C++/CUDA,
the original persistent-batching engine) and a **PyTorch engine** (`lmdeploy/pytorch/**`, 637 Python
files, now the default path, with its own scheduler, cache engine and CUDA-graph runner). The PyTorch
side is where the scheduling/cache design lives today; TurboMind is where the kernel craft lives.

### B.1 In-flight batching / how the GPU is kept fed

`lmdeploy/pytorch/engine/engine_loop.py:520-568` — one `main_loop` per engine, and the shape is
**prefetch-before-wait**:

```
loop:
  if next_running is None:  forward_inputs, next_running = await send_next_inputs()   # schedule
  activate_seqs(next_running)
  forward_inputs, next_running = await _main_loop_get_outputs(...)
      └─ _prefetch_next_inputs()          # schedule token t+1 while t is still on the GPU
         out = await executor.get_output_async()   # only *then* wait
```

The host never sits idle on the GPU: the next batch's inputs (including the KV block tables and the
attention metadata) are built *while the current forward executes*. That is the same overlap principle
as our lane‖GPU split, but applied to the **scheduler** rather than to the experts.

Two scheduling rules worth stealing for a 2–6 stream agent workload:

* **`prefill_interval` forces fairness** (`engine/inputs_maker.py:137`, used at
  `engine/inputs_maker.py:1322`): after *N* (default 16) consecutive decode rounds a prefill turn is
  forced even if the prefill queue is small. Decode ITL is bounded by construction instead of by luck.
* **Prefill is chunked** (`max_prefill_token_num`, `paging/prefill_scheduler.py`), with long-context
  chunks *reserved* rather than retried (`reserve_long_context_chunk`, `paging/scheduler.py:186`), and a
  backoff path when KV is exhausted (`_wait_for_schedulable_prefill`, `engine_loop.py:443-457`).

Unlike exllamav2 (single-user-first, round 3), this *is* a multi-request scheduler — the first engine in
the rotation whose scheduling we can compare to our own 2–6 stream case on equal terms.

### B.2 KV cache management

* **Paged blocks with a logical/physical split**: `paging/block_manager/base_block_manager.py:40-63`
  (a `np.arange` free-list, allocate/append/free by block id) and a second-level physical allocator
  (`:93-119`) so block ids are stable while storage can move. `block_size` and `kernel_block_size` are
  separate (`inputs_maker.py:155-156`) — the *kernel* tile can differ from the allocation granularity.
* **Swap in/out maps per step** rather than in-place migration: `SchedulerOutput(swap_in_map,
  swap_out_map, ...)` (`paging/scheduler.py:44-60`), plus a configurable **eviction helper**
  (`paging/scheduler.py:136`, `try_make_capacity_for` at `:296` and `:364`) that admits work by evicting
  lower-value blocks. This is the only engine in the rotation with *KV over-subscription* (KV larger than
  VRAM, spilled and brought back) as a first-class path.
* **Prefix caching with state checkpoints**: `enable_prefix_caching`,
  `prefix_cache_decode_state_interval`, plus a *state* cache (`states_shapes`, `engine/inputs_maker.py:160`)
  — they checkpoint SSM/GDN state, not just KV blocks. Directly relevant to us at ≥2 k context (Phase 6).
* **Cache engine as a plan/allocate/realize pipeline** (`engine/cache_engine/README.md`): operators
  declare a `BlockCacheRequest` (shape/dtype/contiguity), a `BlockCachePlan` assigns each one a row, and
  `CacheEngine` owns views/streams/events/migration. Compile-time memory planning (the MLC lesson from
  round 1) done properly.

### B.3 Attention kernel design

* Decoding is a **split-K flash-decoding kernel with a heuristic split count**:
  `attention/utils.h:7` `GetSplitCount(max_split_cnt, grid_size, max_active_ctas, sm_count, max_wave_cnt,
  alpha, beta)`, called with `max_wave_cnt = 4` for decoding (`decoding_template.h:50`) and `8` for the
  prefill/context template (`attention_template.h:45`); the split dimension is folded into the grid
  (`cta_map.h:104` `get_grid_shape(kv_head_num, batch_size, split_count, cta_per_q_group)`), and if
  `split_cnt > 1` a **separate reduce kernel** finishes it (`decoding_template.h:71-78`).
* `max_split_k` is workspace-derived, not a constant:
  `min(max(1, kMaxWorkspaceTokens / token_num), max_kv_splits)` (`unified_attention_layer.cc:577`).
* **One templated kernel body for both phases** — `AttentionUniversal<Arch, Mainloop, CacheIteratorFactory,
  CtaMap, Causal>` (`attention/attention_universal.h:23`), with `DecodingCtaMap` vs context `CtaMap`
  choosing the shape. Rotating attention *within one kernel family* is how they avoid a second kernel
  family per phase.
* The cache iterator dequantises **through the page table** (`CreateCacheIterFactory`, quantisation in
  `attention/quantization.cuh`) — paged access is in the inner loop, not a separate gather kernel.

### B.4 How launches are amortised

| mechanism | where | what it buys |
|---|---|---|
| **full CUDA graph for decode** (whole model, one graph) | `backends/cuda/graph_runner/full_graph.py:1-216` | 1 graph launch per decode step |
| **piecewise CUDA graph with token buckets** for prefill | `graph_runner/piecewise.py:1-807`, `standard.py`, `graph_runner/README.md:80-200` | fixed geometry per bucket (`token axis` padded to a bucket) so one plan replays for any request of that length |
| **static buffers + stable addresses** instead of re-recording | `graph_runner/README.md:145-200` (`frame_inputs`, bridge storage) | "only buffer contents vary" — the same invariant our `vkCmdDispatchIndirect` plan was trying to buy |
| **grouped-MoE single launch** via a device row map | `kernels/cuda/moe/fused_moe.py:112-181` (`packed = expert_ids * BLOCK_R + lane`), grid `(cdiv(M,BLOCK_M)*cdiv(N,BLOCK_N), E)` at `:312` | all experts in *one* launch, gate+up as one GEMM (`n = 2·inter`) |
| **persistent tile scheduler** | `src/turbomind/kernels/gemm/scheduler.cuh:75` `TileScheduler` | CTAs pull tiles from a device-side counter instead of `grid = tiles` |
| **kernel-selection cache** instead of per-shape tuning | `gemm/dispatch_cache.h:16-27` (`LowerBound`/`Find` over `GemmDesc`) | zero host cost per call for a shape already seen |
| fused epilogues/activations | `activation.cu` (`silu_and_mul`), `v4_swiglu_quant`, `apply_rotary_emb.py`, `rms_norm.py` | 1 launch instead of 2–3 |

Their answer to "31 launches per layer" is exactly the answer round 3 predicted, and it is worth noting
that **the graph buys the recording and the *geometry*, not the barrier**: decode is one graph *and*
every dependency inside it is still a dependency. What a graph removes on CUDA — per-node launch latency
— RADV removes for free in a command buffer anyway (Part A: 7 ns/dispatch unchained).

### B.5 Explicitly CUDA-only, no Vulkan equivalent

* CUDA graphs themselves, `torch.cuda.CUDAGraph` capture, graph-private memory pools, stream capture.
  Vulkan analogue = a pre-recorded command buffer; **we measured it and it is worth 0.3 ms/token at
  most** (Part A.3/2).
* `deep_gemm` (`m_grouped_bf16_gemm_nt_masked`), sm90 TMA/cluster/GMMA (`gemm/tma.cu`,
  `gmma_*.h`, `mainloop_sm90*`), `cutlass`/`cute` usage: Hopper-class. Nothing portable; our equivalent
  question is closed by Part A.
* `torch.compile` inside `full_graph.py` — no Vulkan analogue and no need.
* Triton kernels (`fused_moe.py` is Triton): the *structure* (row map, grouped grid) transfers, the code
  does not. Ours would be GLSL/SPIR-V.
* fp8 (`e4m3`/`fp4`) paths, DeepEP/DeepGEMM intranode dispatch, `symm_mem_allgather`, Mooncake KV
  transfer, DP/TP/EP collectives — all CUDA/NCCL-specific and all irrelevant to a single-node 2-stream box.
* `cudaMemcpyAsync` inside `copy_cache.py` / `fill_kv_cache.py` — we already have the Vulkan transfer
  queue measured (round 2: fine on the RX, 8.6× worse as a global flag on the MI50).

### B.6 What maps onto our four bottlenecks

| our bottleneck | LMDeploy offers | verdict |
|---|---|---|
| (a) host expert lane 72–75 % of prefill, 39 ms/token decode | nothing kernel-level; their expert weights are device-resident. **But** the prefetch-before-wait loop (`engine_loop.py:520-568`) is the *scheduling* form of our lane‖GPU overlap | **adaptable structure** (B.7 #1) |
| (b) model > RAM | nothing — no SSD tier, no residency policy; KV spill is the only memory-pressure path | no adaptation (our `docs/research/03` Part A stands) |
| (c) 1 509 dispatches / 70.8 ms GPU half | grouped-MoE single launch, gate+up as one GEMM, persistent tile scheduler, bucket geometry | **adaptable, priced at 1–4 %** (Part A) |
| (d) per-(block,expert) residency | none (no frequency ranking) | no adaptation; our PowerInfer plan is ahead |
| KV / streams | paged blocks, swap maps, eviction helper, **prefix + state checkpointing**, `kernel_block_size ≠ block_size` | **the most transferable thing in this audit** (Phase 6) |

### B.7 Ranked adaptations (LMDeploy → Breccia)

| # | adaptation | maps to | expected effect | effort |
|---|---|---|---|---|
| 1 | **Barrier pruning + grouped producer dispatches** (Part A.5) — LMDeploy's grouped-MoE grid is the same idea at kernel granularity | (c) | **−0.7 ms (RX) … −3.2 ms (MI50)/token**, bit-identical, one flag | S |
| 2 | **`kernel_block_size ≠ block_size` + prefix/state checkpoints** for KV: keep our token-aligned quant blocks (round 3), add an explicit decode-state checkpoint interval so a 2–6 stream agent turn can resume without re-prefilling its system prompt (`enable_prefix_caching`, `prefix_cache_decode_state_interval`) | prefill (×2–3 on agents 2..N of a shared prefix), Phase 6 | 0 at 300 tok today; **×2–3 prefill** on shared prefixes, the cheapest of untried-levers 5 | M |
| 3 | **Prefetch-before-wait scheduling**: build token *t+1*'s batch (descriptors, row map, push constants) while *t*'s submits are in flight, and make the ring depth a function of host-record time / GPU-step time (currently fixed at 6, `ctx.rs:344`) rather than a constant | (a), (c) | removes the 0.32 ms recording from the critical path *and* decouples ring depth from geometry; ~0.3 ms/token + more overlap | S–M |
| 4 | **Fused single-dispatch MoE** (§A.3/4): gate+up as one GEMM, one workgroup per (expert, row), grid from a device row map | (c) → (d) | −0.7…−3.2 ms/token at 31→8, and it is the precondition for residency | M |
| 5 | **KV over-subscription with swap maps** (`swap_in_map`/`swap_out_map`, eviction helper): only when 6 streams × ≥2 k context exceeds VRAM | streams, Phase 6 | 0 today; unblocks the 6-stream case at long context | M |
| 6 | **Split-K flash decoding with a workspace-derived split count** (`GetSplitCount`, `kMaxWorkspaceTokens/token_num`) | decode attention when KV is long | ~0 at 300 tok (KV is 24 KiB/token); matters ≥2 k | M |

Deliberately not ranked: their CUDA graphs (Part A prices ours at 0.3 ms), their MoE Triton kernels as
code, their distributed/EP stack, and their sampling kernels.

---

## Top-3 next experiments (each with its bench)

1. **Barrier pruning** — `--barrier-group on|off` in `dispatch_batch.rs` (+ a `barriers()` counter printed
   in the step summary). Bench: `2 streams × 64-prefix, 8 steps, --warm 1`, A/B/A in one window on dual,
   then isolated on `0000:0e:00.0` where the effect is 4.4× larger. Gate: bit-identical logits; predict
   −0.7…−3.2 ms/token in `wait_seconds()`.
2. **Frozen-geometry recording** (direct dispatch, *not* indirect) — record the batch geometry once for a
   *max* grid with an in-shader `rows ≤ count` mask read from a device buffer, so one recorded command
   buffer serves every token; ring 6 → 2. Bench: `recorded()`/`record_seconds()` per token vs the 0.32 ms
   baseline, plus step time with ring depth 2/3/6 (`--ring N`). Gate: bit-identical; predict −0.3 ms and
   a strictly better overlap at deeper pipelining. (Explicitly **not** `vkCmdDispatchIndirect`: +0.77 µs
   per dispatch measured.)
3. **Fused single-dispatch MoE for m ≤ 4** — one workgroup per (expert, row), hidden row (10 KiB) +
   intermediate (2.6 KiB) in LDS, gate and up read as one `[2·n_ff, n_embd]` block. Bench: existing
   per-block MoE timing (2.0 ms → target ≤1.2 ms/block), dispatches/block counter, and the 32/32 parity
   suite at chunk 8/16/32/128. Gate: bit-identical; predict −0.7…−3.2 ms/token from the launch side alone,
   plus the residency crossover.

---

## Surprises / corrections to earlier rounds

1. **Round 3's #1 recommendation is refuted in its stated form.** "Pre-recorded command buffer +
   `vkCmdDispatchIndirect`" would *lose* 1.2–2.5 ms/token, because indirect dispatch is **1.1–1.6× the
   cost of a direct one** on these cards (1.418 vs 0.650 µs RX; 4.510 vs 2.867 µs MI50). The reusable half
   — pre-recorded CBs — is real but worth 0.3 ms.
2. **A dispatch without a barrier is ~7 ns on the RX.** The entire launch-count question is a *barrier*
   question. That also explains why exllamav2's ~2 launches/layer is worth less to us than it looked:
   their kernels are big and chained, ours are small and barriered.
3. **Recording is 4× cheaper than the engine's own measurement** (68 ns vs ~212 ns per dispatch): the
   difference is a `vkAllocateDescriptorSets` per dispatch, not command recording. A pool-allocated
   descriptor-set ring would remove it without any replay machinery.
4. **The MI50's dependency boundary is 4.4× the RX's** (2.87 vs 0.65 µs) — a placement rule, not just a
   performance fact: dependent fine-grained work belongs on the RX.
5. **`VK_EXT_device_generated_commands` is fully present on both cards with COMPUTE pipeline binding**
   (mesa 26.2.4) yet still not worth using: its ceiling is the host recording cost it replaces, and every
   token it emits is an indirect dispatch. Listed as "deferred" in round 1; now priced and closed.
6. **LMDeploy's answer to the launch gap is the geometry, not the graph** — their decode graph exists to
   freeze shapes (token buckets, stable addresses) so replay is possible at all. Their *scheduler* is the
   more interesting half: prefetch-before-wait, a forced prefill every ≤16 decode rounds, and KV
   over-subscription with swap maps. First engine audited that is genuinely multi-request-native.
7. **Prefill's per-group cost dwarfs everything here**: 429–530 µs per *streamed* expert group against
   67–84 µs resident (`docs/phase-4/01` §8v). Launch+barrier is 1–4 % of that group. Whatever fixes
   prefill, it is not launch count.

## Appendix — probe provenance

* Harness: copy of `harnesses/vk-probe` + `src/launch.rs` + `shaders/launch_noop.comp`,
  `shaders/launch_row.comp`, built at `<SCRATCH>/launchprobe` (never inside the repo,
  so the tree stays read-only). Arms: R1–R6 record cost, G1/G2/G4/G5 GPU cost, E1/E2 replay, S1/S2 submit.
* Devices by PCI identity: RX 6950 XT `0000:03:00.0` (Vulkan index 0), Radeon VII `0000:0e:00.0`
  (Vulkan index 1, always run first), iGPU `0000:75:00.0` unused.
* All runs wrapped in `timeout 90`/`120`, one device per invocation, `gpu-manager`, `qwen-dual.service`,
  `strata-rx.service` and `d1-mi50.service` **inactive before and after** (state as found). No
  `amdgpu … page fault` / `Illegal opcode` / ring timeout in `journalctl -b` after the runs.
* Numbers above are medians of 8–40 repetitions; repeated runs agreed within 1 % on the arms quoted
  (the 3 024-dispatch arm on the MI50 showed one 40 % outlier run and is excluded from the slope).
