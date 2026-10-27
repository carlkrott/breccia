# 13 — TensorRT-LLM `batch_manager`: admission, eviction and fairness for Breccia's multi-stream allocator

Round 13. Closes README row 15 (deferred three times). Paired with the Vulkan capability half of this round (Section A).
Engine audited: **TensorRT-LLM** at commit `0a03c3ed` (repo-relative paths below), read-only. Nothing was built or run on a GPU. Prices are derived from figures already in the tree. Anything that needs the card is marked `UNVERIFIED - requires the GPU qualification card`.

**Label note.** README row 15 says the multi-stream allocator is needed before "D1–D5" land. In `10-dispatch-reduction-and-exllamav2.md` §A2, D1–D5 are *dispatch fusions* (D1 gate+up+SwiGLU, D2 expert-indexed MoE, D3 projection bundle, D4 elementwise bundles, D5 hc_read/hc_write collapse). This document reads the row as "the multi-stream allocator that the D1–D5 fusions need before they go multi-stream".

**Headline.** Four findings decide the design.

1. **Recompute is not an eviction option at long context.** TRT-LLM's pause moves generated tokens back into the prompt and re-prefills (`cpp/include/tensorrt_llm/batch_manager/llmRequest.h:705-760`). At the measured prefill rate of 6.0–6.45 tok/s per agent (`docs/BRIEF.md` line 80), a 32K re-prefill takes 85–91 min and a 100K re-prefill takes 4.3–4.6 h. Moving a 100K stream's QSA state over the T1↔T2 link takes 66–112 ms. Breccia eviction must be swap-or-keep.
2. **The admission problem is a static reservation, not the scheduler.** `QsaState::new(gpu, ctx_capacity, …)` (`crates/breccia-engine/src/graph.rs:965`) reserves `ctx_capacity` cells per QSA layer when a Session is built. The CLI default is 256 (`crates/breccia-cli/src/main.rs:281-282`). A stream's T1 cost therefore does not depend on its prompt. Sizing to `min(prompt + max_new, cap)` is the cheapest change in the tree.
3. **Fairness must replace LIFO.** TRT-LLM's MaxUtilization victim is the last started request in list order (`capacityScheduler.cpp:609-628`). For agents, the victim should be a stream waiting on a tool call. `Multi` also steps every stream at one shared position (`crates/breccia-engine/src/multi.rs:225-250`), so no stream can be admitted mid-flight. That has to change before any slot table works.
4. **Prefix reuse is creditable only at a GDN snapshot boundary.** TRT-LLM's hybrid cache commits blocks only up to the last snapshot (`tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager_v2.py:1108-1144`). A content-hash hit on QSA pages alone cannot credit a prefix.

## Standing constraints

- Tiers: T1 = VRAM (implemented), T2 = pinned host RAM (~30 GB, planned), T3 = shared NVMe capped at 100 GiB in its own directory (planned). Weights never go to NVMe. Only T1 is implemented today.
- **T3 must never share gpu-manager's 4d `kv-slots/` directory.** That directory's `prune()` deletes `*.bin`.
- Standing verdicts this round does not contradict: quantised GEMV is instruction-issue bound (round 12); recording is 66 ns per dispatch and submit count is not the lever (rounds 08, 11); the GPU can execute host-resident experts at 22–29 GB/s (round 09); the RX is pinned at 500 MHz under `power_dpm_force_performance_level = auto` (round 08); the lead lever is legs, led by the hyper-connection projector D5 (round 10); a command-buffer boundary costs about 300 ns.

## A. Vulkan capability check (scheduling-relevant)

**Supersession.** Round 12's instruction law supersedes the cold-load-chain suggestion in README line 34 (`§X3`: `bpr` cold-load chain, `UNROLL`/software prefetch) for this round's scope. A cold-load chain is a stall, which makes it a slot-efficiency term (round 12: rate = K / I_unit), not a byte term. It is not a scheduling input, so §X3 is not re-run here.

| Capability | State on our cards | Scheduling meaning | Source |
|---|---|---|---|
| `VK_KHR_timeline_semaphore` | Required; the device test asserts it on every device | Completion of a swap without a fence per step (phase-4 fence cost: 14 ms at 4 streams) | `crates/breccia-vk/src/lib.rs:88-89`; `docs/phase-4/02-untried-levers.md` line 14 |
| Second queue in the compute family | Current default (`XferMode::SameFamilySecondQueue`) | A swap copy can run beside decode without the transfer-only family | `docs/research/01-vulkan-and-mlc.md:243-247` |
| Transfer-only family (`RADV_EXPERIMENTAL=transfer_queue`) | Two queues measured, not used | **Not for T1↔T2 by default.** On-card copies were 8.6× slower (423 → 49 GB/s on the MI50), and the knob is process-wide | `docs/research/02-vulkan-followups-and-powerinfer.md:178,185,576` |
| `VK_EXT_external_memory_host` | Listed and enabled | T2 is host memory the GPU reads directly, so a swap-in can be a shader read instead of a copy | `docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md:38`; round 09 (22–29 GB/s) |
| `VK_EXT_memory_budget` | Advertised and enabled; `Gpu::mem_budget()` returns `heapBudget − heapUsage` for this process | Live admission headroom exists. Today it is read only at load, by the arena | `crates/breccia-vk/src/ctx.rs:447-449, 987`; `crates/breccia-engine/src/arena.rs:104-113` |
| `VK_KHR_performance_query` | Advertised, but `vkCmdBeginQuery` segfaults | No per-copy hardware counters; time with timestamps | `docs/research/07-attribution-and-ollama.md` |
| `vkCmdCopyBuffer2` region arrays | 110–140 ns per region (RX), 300–360 ns (MI50), nearly independent of size | A swap must be one contiguous region per stream class, never a region array of 16-cell pages | `docs/research/05-ple-io-and-tensorrt-llm.md` lines 25–26 |

**Decision number: does a T1↔T2 copy overlap decode or stall it?**

η = (t_decode_alone + t_copy_alone − t_decode_with_copy) / t_copy_alone

Measurement: decode at the 4-stream shape (162 ms/step reference, phase-4 §8p), with one copy in flight on a second compute-family queue, completion on a timeline semaphore. Copies are one 100K stream's QSA state (1.8 GiB as V q8_0, 2.3 GiB as f16) and one GDN state (112 MiB), both directions.

- η ≥ 0.8: the overlap is real. Swaps run in the background and eviction may happen at any step.
- η ≤ 0.2: the copy serialises with decode. Each swap adds its full copy time to the step clock, so the scheduler must budget swaps between steps.
- Also measure the shared link. The 22–29 GB/s figure is a link ceiling, so swap traffic and host-expert streaming compete for it. Record the decode-step change from expert streaming alone.

No value is claimed here. `UNVERIFIED - requires the GPU qualification card`.

**Live admission headroom.** Read `Gpu::mem_budget()` at each admission decision, not once at load. Start with the two margins the arena already uses: `display_reserve` 768 MiB and `headroom` 256 MiB (`crates/breccia-engine/src/arena.rs:82-89`). These are starting values for admission, not measured ones.

## B. TRT-LLM policy anatomy

All paths are repo-relative to TRT-LLM at commit `0a03c3ed`.

| # | Concern | Mechanism | Where | Breccia note |
|---|---|---|---|---|
| B1 | Policy enum | MAX_UTILIZATION=0, GUARANTEED_NO_EVICT=1, STATIC_BATCH=2 | `cpp/include/tensorrt_llm/executor/types.h:217-231` | The V2 Python scheduler forces MAX_UTILIZATION (`tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py:207-212`) |
| B2 | Admission, MaxUtilization | Pack requests while blocks fit, using "now" accounting | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:480-637` | Keep "now" accounting |
| B3 | Victim, MaxUtilization | Reverse scan for the last started request in list order (LIFO). Remove with `schedulingRemoveSequence` (617), record in `pausedRequests` (625) | `capacityScheduler.cpp:609-628`; `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py:2186-2197` | LIFO picks the newest stream, which has the least sunk work. Not fair for agents |
| B4 | GuaranteedNoEvict | Worst-case reservation (prompt + maxNew) per started request; head-of-line `break` | `capacityScheduler.cpp:229-468` (reservation 289-303, admission check 427, break 461, empty paused set 467) | The static ctx reservation is this policy's worst case with no scheduler behind it |
| B5 | StaticBatch | Admit only when nothing is active | `capacityScheduler.cpp:317` | Not useful here |
| B6 | Block accounting | "Needed now" with reuse credit, versus "remaining to completion" | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:3403-3490` (credit 3438-3453); `:3492-3600` | Copy the split. Admission uses "now". Worst case only for guaranteed streams |
| B7 | Pause | `LlmRequest::pause(maxInputLen)` moves generated tokens into the prompt and resets to CONTEXT_INIT | `cpp/include/tensorrt_llm/batch_manager/llmRequest.h:705-760` | Recompute. Not viable (C4) |
| B8 | Eviction | LRU over priorities 0–100 (default 35); cold-tier offload minimum 30 | `cpp/tensorrt_llm/batch_manager/evictionPolicy.cpp:31-35`, `:141-162` (getFreeBlock), `:157` (canOffload), `:169-200` (releaseBlock); `evictionPolicy.h:71-129`; `common.h:36` | Per-block priority. Breccia's unit is the stream (C3) |
| B9 | Retention | Priority per token range, optional duration | `cpp/include/tensorrt_llm/executor/executor.h:519-593` | Maps to "pin GDN, evict QSA pages" |
| B10 | Chunk policies | FCFS=0, EQUAL_PROGRESS=1, FORCE_CHUNK=2 | `cpp/include/tensorrt_llm/executor/types.h:235-248`; `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp:141-199` (equal), `:202-262` (FCFS), `:265-325` (force chunk), `:335-353` (dispatch) | EQUAL_PROGRESS is the prefill-fairness analogue (C4) |
| B11 | Snapshot points | Expected points per request; read only by FORCE_CHUNK | `cpp/include/tensorrt_llm/batch_manager/llmRequest.h:960-968`; member `:1533` | The boundary rule (C5) |
| B12 | Hybrid state, legacy | Fixed slot pool keyed by sequence slot, size `max_batch_size + 1` (+1 with attention-DP) | `tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager/common.py:307-318`; `legacy.py:1903`, `:2296` | A fixed pool outside the block count |
| B13 | Hybrid state, V2 | SSM state lives as blocks in the same pool. Free blocks = max over attention and SSM page indices. SSM capacity = `max_resident × fixed_rules + capacity // interval`. Pool sized from a typical-request ratio | `tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager_v2.py:436-437, 447-462, 835-856, 965-1007` | The Python scheduler counts state through that ratio. Whether an SSM term reaches the C++ capacity path is `UNVERIFIED` (not traced) |
| B14 | Snapshot rule | Periodic multiples of the interval up to prompt length, plus offsets from start and end (prompt end = offset 0). Empty when block reuse is off | `mamba_cache_manager/common.py:271-294, 489-527`; `legacy.py:1269-1290` | GDN state at boundaries |
| B15 | Reuse commit | `commit_limit = min(max(snapshot points), prompt_len)`; commit only at a boundary | `mamba_cache_manager_v2.py:1108-1144, 1159-1195` | A QSA hash alone is not enough (C5) |
| B16 | Config | `periodic_snapshot_interval` defaults to 0 (prototype); the per-conversation policy disables it | `tensorrt_llm/llmapi/llm_args.py:4647-4680, 5363-5371` | Snapshot reuse is a prototype upstream as well |
| B17 | Recompute gate | `enable_recompute_pause = not is_disagg` | `tensorrt_llm/_torch/pyexecutor/_util.py:4311` | — |
| B18 | Python defaults | Default policy GUARANTEED_NO_EVICT; MaxUtilizationPolicy victim scan | `scheduler/scheduler.py:2455-2466` (default), `:2077-2210` | — |
| B19 | V2 eviction | `_try_evict_for_gen` (backward scan, suspend, retry) `scheduler_v2.py:2068`; `_try_recompute_pause_for_gen` `:2109-2196`; `_is_evictable` `:2007`; recompute candidates `:2021` | `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py` | The recompute path is the one not to copy (E) |
| B20 | V2 ordering | `_sort_requests` sorts by LoRA task id only | `scheduler_v2.py:2201` | No arrival-time or fairness key at all |

## C. Adaptation sketch: 4–6 slots, per-stream GDN state, QSA pages

### C1. Per-stream T1 classes (derived)

- **GDN** (not paged): 3 MiB state + 120 KiB conv per block (`crates/breccia-kernels/src/gdn/mod.rs:60-100`). 36 GDN blocks give **112.2 MiB per stream**, fixed.
- **QSA KV**: 12 QSA blocks, 2 KV heads, head_dim 256, 16-cell pages (`crates/breccia-kernels/src/qsa/mod.rs:69-89`). f16 is 24 KiB per token (K and V). V q8_0 is 18.75 KiB per token (`qsa/mod.rs:106-132, 169-184`). A page is 384 KiB (f16) or 300 KiB (V q8_0) across the 12 layers.
- **Indexer pooled keys**: f32, 128 wide, one row per 4-cell block (`qsa/mod.rs:37-40, 146`). That is about 128 B per cell per QSA layer, or **1.5 KiB per token** over 12 layers, plus a fixed 3-row raw tail per layer.

| Context | GDN | QSA KV, V q8_0 | QSA KV, f16 | Indexer | Total, V q8_0 | Total, f16 |
|---|---|---|---|---|---|---|
| 32K | 112 MiB | 600 MiB | 768 MiB | 48 MiB | 760 MiB | 928 MiB |
| 100K | 112 MiB | 1 831 MiB | 2 344 MiB | 147 MiB | 2 090 MiB | 2 603 MiB |

**Cross-check.** The tree measures 764 MiB (V q8_0) and 932 MiB (f16) per agent at 32K (`docs/BRIEF.md` lines 169 and 282). The derivation lands 4 MiB under both. A per-token indexer estimate of 384 B would leave about 40 MiB unexplained, so the f32 figure from the source is the one that closes the gap.

**Reservation today.** With `ctx_capacity` = 100K, every stream holds the full 100K QSA reservation whatever its prompt. Four streams hold 8.2–10.2 GiB before any token is decoded. The per-layer workspace at `ctx_capacity` (`graph.rs:986`) comes on top and is not sized here.

### C2. Admission test

A stream is admitted to a T1 slot only if

`live_free ≥ GDN (112.2 MiB) + ⌈prompt/16⌉ × page + 1.5 KiB × prompt + first-chunk workspace + reserve`

- `live_free` = `Gpu::mem_budget()` (heapBudget − heapUsage), read at decision time.
- `reserve` = `display_reserve` + `headroom` (768 + 256 MiB, starting values).
- Growth is charged per chunk as it happens, not reserved to the maximum. This is the difference from GuaranteedNoEvict, which reserves prompt + maxNew (B4). Per-chunk growth needs a pool that can grow (rank 1 risk).

### C3. Eviction unit and fairness rule

**Units.**
- The whole stream is the unit for GDN state and for the indexer pooled keys. Both are read on every decode step: the GDN recurrence touches each GDN block's state per token, and the indexer scores the pooled keys in each QSA block every step (`indexer_score`, `crates/breccia-kernels/src/qsa/cpu.rs:95`, is the CPU reference; the GPU scoring shader was not re-read this round, `UNVERIFIED`). Evicting only one of them gains nothing.
- The 16-cell page is the unit for QSA KV. A decode step reads only the indexer's top `IDX_TOP_K = 2048` cells per QSA layer (`qsa/mod.rs:83`). With the selected pages in T2, that is 48 MiB per step (12 layers × 2048 cells × 2 KiB), or 1.7–2.3 ms per stream at 22–29 GB/s. The cost is bounded and does not grow with context once context exceeds 2048. The gather pattern (2 KiB segments) is `UNVERIFIED`: round 09 measured coalesced reads only.

**Victim cost** = bytes not already in T2 ÷ link rate. Keep-resident cost = T1 bytes held.

**Fairness rule (replaces LIFO).**
1. Victims are streams waiting on a tool call, longest idle first. This is the tool-wait eviction of `docs/plan/BRECCIA-PLAN.md` Phase 6.
2. Otherwise, the stream with the oldest last-token time.
3. Never a stream mid-prefill chunk, never a stream with a copy in flight, never the stream whose step is running.
4. Progress: every resident stream takes one decode step per round (round-robin). One prefill chunk is one unit per round (C4).

### C4. Prefill fairness and the 100K price

Recompute is never a victim option.

| Path | 32K | 100K |
|---|---|---|
| Re-prefill at the measured 6.0–6.45 tok/s per agent (BRIEF line 80) | 85–91 min | 4.3–4.6 h |
| Re-prefill if the 13 tok/s NVMe-streaming bound (BRIEF line 284, measured at 32K) held at 100K | — | ~2.1 h |
| Swap one way, QSA state at 22–29 GB/s (V q8_0 / f16) | 26–38 / 30–43 ms | 66–87 / 85–112 ms |
| Swap one way, GDN state | 4.1–5.3 ms | 4.1–5.3 ms |

The gap is about 10^5 at 100K. Swap-or-keep is the only option at 32K and above. The 6 tok/s figure is the chunk-64 prefill rate, so the re-prefill cost is a measured anchor, not a model.

**Prefill fairness.** A 100K admission at 6 tok/s holds one stream's prefill for hours, so prefill must be chunked and interleaved, never run to completion. TRT-LLM's EQUAL_PROGRESS (`microBatchScheduler.cpp:141-199`) advances every prefilling request by one chunk unit per round. FCFS (`:202-262`) would starve decode. Breccia's `Multi` feeds the same token to every stream at the same position during prompt processing (`multi.rs:234`), so prefill is also lockstep and has to be decoupled with decode (D2).

### C5. Prefix reuse at GDN snapshot boundaries

- TRT-LLM commits only up to the last snapshot (B15). For Breccia, a page-chain hash identifies a prefix: each page hash is `hash(previous page hash, its 16 token ids)`. A prefix is creditable only where a GDN snapshot exists at that page index.
- A snapshot is a 112.2 MiB GDN copy to T2: 4.1–5.3 ms at 22–29 GB/s.
- Interval I in cells sets the snapshot count: 100K / I. I = 16K gives 6 snapshots, or 673 MiB of T2 per cached 100K prefix, before its QSA pages. I = 4K gives about 24 snapshots, or 2.6 GiB. Choose I against reuse granularity and the T2 budget.
- A restored state must match a non-restored run bit for bit. Check it with the engine's `Checksum` before any reuse path is trusted.

## D. Ranked recommendations

Ranked by the value of the precondition they unlock, then by cost. Each item gives its effect on TPS, PP (prefill), context, cache and transfer.

**1. Size T1 reservations to need, and gate admission on the live budget.**
- Effect: TPS about 0. The aggregate is set by the GPU (24.74 tok/s at 4 streams, 25.03 at 6; phase-4 §8p). PP 0. Context: a 100K-capable Session no longer costs 2.0–2.6 GiB per stream at a 2K prompt. Cache: frees T1 for more streams. Transfer 0.
- Measure: T1 bytes per stream (`VK_EXT_memory_budget` delta across `Session::new`) at ctx 256, 4K, 32K and 100K, before and after. Count the streams admitted at 32K prompts before the first allocation failure.
- Risk: a growing pool reallocates. Recorded command buffers must not hold stale buffer handles (`UNVERIFIED`: check the bind path in `graph.rs`).
- Touched: `crates/breccia-engine/src/graph.rs` (965, 986), `crates/breccia-kernels/src/qsa/mod.rs` (`QsaState::new`), `crates/breccia-engine/src/multi.rs` (admission), `crates/breccia-cli/src/main.rs` (ctx default).

**2. Decouple per-stream position in `Multi`.**
- Effect: admission mid-flight and per-stream progress. TPS unchanged if the GEMV still shares dispatches across streams (`UNVERIFIED`). PP 0. Context 0. Cache 0.
- Measure: 4-stream decode with staggered starts. Aggregate must stay at the 24.74 tok/s lockstep reference (162 ms/step, phase-4 §8p) within noise.
- Risk: the lockstep shared position (`multi.rs:225-250`) may be a batching assumption. Per-stream positions could split dispatches.
- Touched: `crates/breccia-engine/src/multi.rs` (`token_all`, `generate`).

**3. Swap-or-keep at stream granularity, on a second compute-family queue, gated by η (Section A).**
- Effect: a tool-waiting stream can leave T1, and more than T1 streams can exist. TPS 0 if η ≥ 0.8. Otherwise each swap adds 66–112 ms to a step that takes 162 ms at 4 streams. PP: avoids a 4.3–4.6 h re-prefill at 100K. Transfer: 66–112 ms per 100K stream, one way.
- Measure: η on the 4-stream shape (Section A plan). Swap bytes per stream class. Checksum after a round trip.
- Risk: the link is shared with host-resident experts. Copy throughput on the MI50 is `UNVERIFIED`. Do not use the transfer-only family (8.6× slower on-card).
- Touched: `crates/breccia-engine/src/xfer.rs` (`Staging` does one round trip per direction per layer today; a swap needs one region per stream class), `crates/breccia-vk/src/transfer.rs:120` (`TransferRing`, the timeline-semaphore ring).

**4. Victim policy: tool-waiting first, then oldest token. No LIFO, no recompute.**
- Effect: TPS 0. Context 0. Fairness for agent workloads. Cache: keeps the state of tool-waiting streams alive in T2 rather than re-prefilling it.
- Measure: a deterministic simulation with no GPU, over a recorded agent trace. Compare p95 resume latency and swaps per hour against LIFO.
- Risk: the policy is only as good as the tool-wait signal. A stream wrongly marked idle gets evicted and pays a swap.
- Touched: the slot table exists only on the `serve-v1` branch (`crates/breccia-server/src/batcher.rs`: one pending job per free slot, `run()` lines 139–160, no eviction today). It is a branch, so read it with `git show serve-v1:…`.

**5. Snapshot-boundary prefix reuse (GDN snapshot every I cells, plus a page-chain hash).**
- Effect: PP: a cached 100K prefix saves 4.3–4.6 h of re-prefill. TPS 0. Context: none directly. Cache: 673 MiB of T2 per cached 100K prefix at I = 16K, plus its pages. Transfer: 4–5 ms per snapshot.
- Measure: hit rate and PP saved on an agent trace with repeated system prompts and histories. Checksum equality after restore.
- Risk: TRT-LLM ships this as a prototype (B16). GDN restore correctness is the whole risk.
- Touched: a new module in `crates/breccia-engine/src/` (no cache exists today); `crates/breccia-kernels/src/gdn/mod.rs` (GDN state save and restore).

**6. Host-resident QSA pages for the selected cells, with the indexer pooled keys pinned in T1.**
- Effect: context: T1 KV at 100K falls from 1.8 GiB (V q8_0) to the pooled keys (147 MiB) plus a working set. TPS: each stream pays 1.7–2.3 ms per step on the shared link. For 4 streams, if the gathers serialise, that is 7–9 ms of 162 ms (about 4–6%). PP 0. Transfer: 48 MiB per stream per step.
- Measure: a gather kernel that reads 2048 selected cells × 12 layers from pinned host memory via `VK_EXT_external_memory_host`, compared with the coalesced 22–29 GB/s figure. The gather granularity is the open question.
- Risk: 2 KiB segments may sit below the efficient transaction size. The indexer scan reads all pooled keys each step (`UNVERIFIED`), so the pooled keys must stay in T1.
- Touched: `crates/breccia-kernels/src/qsa/mod.rs` (page layout, `idx_pooled`), and the QSA attention shader.

## E. What not to copy

- **Pause as recompute** (`llmRequest.h:705-760`; V2 `_try_recompute_pause_for_gen`, `scheduler_v2.py:2109-2196`). At 100K it costs 4.3–4.6 h per pause.
- **LIFO victim** (`capacityScheduler.cpp:609-628`; `scheduler.py:2186-2197`). The newest stream has the least sunk work, which is the wrong fairness signal for agents.
- **Worst-case reservation as the default** (GuaranteedNoEvict, prompt + maxNew). It holds what a stream may never use. Use "now" accounting (B6).
- **`_sort_requests` by LoRA id** (`scheduler_v2.py:2201`). There is no fairness key.
- **The SSM pool-ratio heuristic** (`mamba_cache_manager_v2.py:462`). Breccia's per-stream GDN size is exact (112.2 MiB), so no ratio is needed.
- **The CUDA runtime layer**: streams and events, graph capture, pinned `cudaHostAlloc`, NCCL, and a paged block allocator on `cudaMalloc` pools. The Vulkan equivalents are timeline semaphores, `VK_EXT_external_memory_host`, and the boundary costs in the standing constraints.
- **`kFORCE_CHUNK` as written** (`microBatchScheduler.cpp:265-325`). It advances to snapshot points tied to block reuse. Copy the boundary idea (C5), not the chunk policy.
- **Snapshot machinery as proven.** `periodic_snapshot_interval` defaults to 0 (`llm_args.py:4647-4680`).
- **"Streams ≤ card count" as a hard rule.** `Multi::new` checks only the per-model device count (`multi.rs:139-147`). One card per stream is a performance choice, not an enforced invariant. Phase-4 ran 4 streams at 2 per card and 6 at 3 per card, all on the two cards (`docs/phase-4/01-forward-pass.md` lines 1058, 1312).

## Not done here

- No GPU run. η, the gather granularity, the copy-queue count and the shared-link share are all `UNVERIFIED - requires the GPU qualification card`.
- The D1–D5 dispatch fusions are not re-scoped. This document fixes only the allocator they need.
- The SSM accounting in the C++ capacity path (B13) was traced at the V2 level only.
- The indexer scoring kernel (the per-step pooled-key scan) was not re-read.
- No Breccia code was changed. The `serve-v1` branch was read, not merged.

## Sources

TRT-LLM at commit `0a03c3ed` (repo-relative): as cited in Sections B and C.

Breccia (repo-relative): `crates/breccia-engine/src/{graph.rs, multi.rs, arena.rs, xfer.rs, model.rs}`; `crates/breccia-kernels/src/{qsa/mod.rs, gdn/mod.rs}`; `crates/breccia-vk/src/{ctx.rs, lib.rs}`; `crates/breccia-cli/src/main.rs`; `crates/breccia-server/src/batcher.rs` (on `serve-v1`); `docs/BRIEF.md`; `docs/plan/BRECCIA-PLAN.md`; `docs/phase-4/01-forward-pass.md`; `docs/phase-4/02-untried-levers.md`; `docs/research/01`, `02`, `05`, `07`, `09`, `10`, `12` (files named by their README rows).
