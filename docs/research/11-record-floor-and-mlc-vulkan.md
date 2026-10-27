# 11 — The achievable recording floor (measured), and MLC-LLM's Vulkan backend

Round 11. Paired with the implementation round that owns the single-stream lane.
Engine audited: **MLC-LLM** (`Reference Projects/mlc-llm` @ `8978ea9`, 2026-10-07 — already cloned in
round 01; the `3rdparty/tvm` submodule is checked out, so `src/backend/vulkan/` is readable).

**Headline.** The host-side floor for recording our dispatch shape is **66 ns/dispatch**, and the
whole of `dispatch_batch::record()` — replicated call-for-call, including our Rust bookkeeping — is
**266 ns**. The 12.7 ms/token "record" budget divides to 6.9 µs/dispatch, which is **25× the entire
measured path and 100× the floor**, so it cannot be dispatch recording: it is a counter-window
artefact (§A6). Recording at the floor costs **0.12 ms/token**; at today's path, **0.49 ms/token**.
The two things worth fixing are the five `Instant::now()` pairs inside the hot path (165 of the 202 ns
of our own overhead) and the `(u64, Vec<Buffer>)` hash key — together worth ~0.4 ms/token, and
**nothing about this changes the token's critical path**, which is `wait-gpu 32.4 + lane 12.8 +
head 7.0`.

All measurements are the RX 6950 XT (`0000:03:00.0`, headless), one process at a time, `timeout`
wrapped, `power_dpm_force_performance_level` verified `high` in every run header, and the MI50
(`0000:0e:00.0`, the **display** card) was never touched. The harness records only; it submits once at
the end as a liveness check. Scratch code:
`/var/tmp/users/korphaus/tmp/opencode/research-harnesses/r11-record-floor/` (not in the repo).

---

# Part A — the recording floor

## A1. Method

`r11-record-floor` measures **only the recording loop**: N compute dispatches into one command buffer,
no submits, no waits, one pre-bound pipeline, one pre-written descriptor set, 32-byte push constants,
a trivial shader (`local_size_x 256`, one workgroup dispatched). Each cell is best-of-3 passes ×
best-of-N record loops; the first record of a pipeline is discarded (a cold dispatch is 5× slow, `06`).
The *difference* between adjacent rows is the marginal price of one call, which is why the rows are
cumulative rather than independent.

```
cd /var/tmp/users/korphaus/tmp/opencode/research-harnesses/r11-record-floor
cargo build --release
timeout 90 ./target/release/r11-record-floor caps                    # device + capability readout
timeout 120 env R11_DBUF=1 ./target/release/r11-record-floor record   # the floor table
timeout 90  ./target/release/r11-record-floor cbfrag                  # CB-boundary cost
timeout 60  ./target/release/r11-record-floor hostcost                # host-cost decomposition
timeout 120 ./target/release/r11-record-floor mt                      # same work, 1/4/8/16 threads
# validation pass (numbers unused): the only VUIDs are the deliberate omissions of the
# one-call-removed arms (08117 / 08600 / 08601); every composed arm is VU-clean.
timeout 90 env R11_DBUF=1 VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ./target/release/r11-record-floor record --validate
```

Header on every run: `power_dpm_force_performance_level: ["high" × 6]`, device
`AMD Radeon RX 6950 XT (RADV NAVI21)`, API 1.4.354, ash 0.38 (`DeviceFnV1a_3` table, same loader path
as `breccia-vk`).

## A2. The floor table (RX 6950 XT, RADV, no validation layer)

| what is recorded | N=100 | N=1 000 | N=2 000 | marginal |
|---|---:|---:|---:|---:|
| bare `vkCmdDispatch` | 13.6 | 13.4 | 14.0 | **+13.6** |
| + `vkCmdBindPipeline` | 16.0 | 15.8 | 15.9 | +2.3 |
| + `vkCmdBindDescriptorSets` (1 set, 2 SSBO) | 26.8 | 26.3 | 26.5 | +10.9 |
| + `vkCmdPushConstants` (32 B) | 37.0 | 37.4 | 37.4 | +10.2 |
| + `vkCmdPipelineBarrier2` sync2 COMPUTE→COMPUTE | **65.6** | **65.7** | **67.1** | **+28.6** |
| same, but v1 `ALL_COMMANDS` barrier | 92.6 | 92.8 | 93.4 | +27.3 |
| **engine shape via raw `PFN_` pointers** (no ash) | 67.3 | 67.4 | 64.9 | −0.8 (0) |
| **engine shape + our `record()` host replica** | **269.0** | **266.1** | **264.7** | **+202** |
| `vkCmdPushDescriptorSet` (2 SSBO, legal layout) | 150.2 | 150.4 | 151.2 | — |
| `vkCmdSetDescriptorBufferOffsetsEXT` (1 buffer) | 71.8 | 72.1 | 72.2 | — |
| bindless: pipe + push + barrier + dispatch (no descriptor call) | 54.8 | 54.6 | 54.9 | — |
| **old shape**: allocate + `vkUpdateDescriptorSets` per dispatch | 235.0 | 234.2 | 233.6 | — |

`ns per recorded dispatch`; N is dispatches per command buffer, identical at every N → the cost is
per call, not per command buffer.

Command-buffer boundaries (`cbfrag`): an empty `reset + begin + end` is **204 ns**, and laying the
same N dispatches out across **51 command buffers** instead of 1 costs **+7.0…13.2 ns/dispatch** —
i.e. 51 boundaries per token are **10 µs/token**, not a factor.

Host-cost decomposition of the +202 ns (`hostcost`, plus the same work under 16 threads, `mt`):

| piece | ns/dispatch |
|---|---:|
| 5 × (`Instant::now()` + `elapsed()`), the engine's profiler | **165** |
| `std::env::var_os` ×2 (`track_disabled()`, called twice per dispatch) | 60–78 |
| `HashMap<(u64, Vec<u64>), _>` probe with the key `Vec` allocated fresh | 18 |
| thread-local `DIRTY` push + linear `contains` | 2 |
| same alloc+hash probe at 4 / 8 / 16 threads | 8.5 / 3.7 / 2.5 (**it does not degrade**) |

A raw `Instant::now()` is ~8 ns; `elapsed()` is a second read, which is why the five-pair shape is
165 ns and not 80.

## A3. What the decomposition actually says

1. **A dispatch costs 13.6 ns to record.** Everything else is what we hang off it. 1 843 dispatches at
   the floor = **25 µs** of the token; the barrier is the single most expensive call we make.
2. **`vkCmdPipelineBarrier2` is 28.6 ns to record** — 2× a dispatch, but *cheap*; there is nothing to
   win by recording it differently. (Its *GPU-timeline* price is the 1.1 µs + 0.45·bpr law of round
   10, which this round does not touch.)
3. **v1 `ALL_COMMANDS` records 27 ns/dip more than sync2** — 92.9 vs 65.7. That is a host-lane
   difference of 0.05 ms/token at 1.8 k, i.e. nothing like round 10's "1.45× wall" claim. It is
   consistent with `06a`: the barrier *form* changes nothing on the GPU timeline and ~40 % on the host
   lane. If `BRECCIA_BARRIER=full` is still the default anywhere, that is the whole of its effect.
4. **ash's function table is free** (row 7 = row 5). Hypothesis (a) is refuted outright, not inferred.
5. **Push descriptors are *worse* on RADV, not better.** With a legal
   `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR` layout, a 2-write
   `vkCmdPushDescriptorSet` is **+85 ns** over `vkCmdBindDescriptorSets` (150 vs 26.6 ns for the bind).
   ncnn's `cmd_push_descriptor_set_with_template` pattern is cheap *in ncnn* because the blob is
   pre-built and memcpy'd; the raw call is not cheap here. **Do not port ncnn's push-descriptor
   shape.** (Round 10 §A4 listed push descriptors as an option; this retires it.)
6. **`VK_EXT_descriptor_buffer` offsets are +5.4 ns** over a set bind (72.0 vs 26.6 = one extra call),
   so a descriptor-buffer *rewrite* per dispatch is not free either — its value is that a **bindless
   base index needs no descriptor call at all**: the `bindless` row is **54.8 ns**, i.e. **−10.9 ns**
   vs `bind_descriptor_sets` and **−11 ns** vs the engine's 65.7. At 1 843 dispatches that is
   **−20 µs/token**. Round 10's verdict stands and is now measured: **D0 is infrastructure, not a win.**
7. **The old per-dispatch set allocation was the real cost it was blamed for**: 235 vs 66 ns = **+168
   ns/dispatch**, which is exactly `04`'s 68 → 212 ns finding reproduced. It is already fixed in the
   current code, so this row is a regression guard.

## A4. The 20× gap, ranked by elimination

| hypothesis | verdict | evidence |
|---|---|---|
| (a) ash loader trampolines / no device-function table | **refuted** | raw `PFN_` pointers record the same shape 0.8 ns *slower*, inside noise (§A2 row 7) |
| (b) `vkAllocateDescriptorSets` / `vkUpdateDescriptorSets` on the path | **already removed; would be +168 ns if present** | §A2 last row; the code caches per `(pipeline, buffer-set)` (`dispatch_batch.rs:886`) |
| (c) a barrier per dispatch | **real but small** | +28.6 ns to record, and the per-resource tracker already prunes; the GPU-timeline half is round 10's problem, not this one |
| (d) our own Rust alloc/format/hash per dispatch | **real, and it is the largest term we control: +202 ns**, of which **165 is the profiler itself** | §A2 replica row, §A3 host table; does not degrade under 16 threads |
| (e) RADV's `vkCmdPipelineBarrier2` being expensive to *record* | **refuted** | 28.6 ns, the same as v1 within 1.5 ns, 2× a bare dispatch |
| (f) 51 command buffers/token, reset and re-recorded | **refuted** | 204 ns per boundary = 10 µs/token total; +7…13 ns/dispatch (§A2 cbfrag) |

So the ranking, largest first: **(d) our own in-path bookkeeping (202 ns, 76 % of the path) → (c)
barrier record cost (29 ns) → (b) descriptor binds (11 ns) → everything else ≈ 0.** And all of it
sums to 266 ns, which is 25× short of 6.9 µs.

## A5. How the reference engines keep it cheap — validated against the floor

* **ggml-vulkan** (`ggml-vulkan-common.h:265-330`, `ggml_vk_dispatch_pipeline`): per dispatch it does
  `descriptor_set_idx++` into a **pre-allocated descriptor-set ring indexed by the dispatch ordinal in
  the command buffer**, compares the new `DescriptorBufferInfo`s against what that slot held last graph
  and **skips `vkUpdateDescriptorSets` entirely when they match** (the steady-state decode case), then
  `pushConstants` + `bindPipeline` + `bindDescriptorSets` + `dispatch`. **No allocation, no hash map,
  no timers, no env lookups, and no barrier inside the dispatch helper.** That is our floor shape
  (65.7 ns) plus a 2-element struct compare — call it ~70 ns. Everything we do above that is our own.
* **TVM/MLC** (`3rdparty/tvm/src/backend/vulkan/runtime/vulkan_wrapped_func.cc:76-100`): on RADV
  `UseImmediate()` is true (the descriptor-update-template fn pointers exist), so the path is
  `bindPipeline` + **`vkCmdPushDescriptorSetWithTemplateKHR`** + `pushConstants` + `dispatch` + one
  memory barrier. One descriptor call, pre-built template, no set cache. On this driver that call costs
  85 ns more than a bind (§A3.5) — i.e. **TVM's Vulkan record path is ~150 ns/dispatch, twice our
  floor, and it is the shape ncnn/MLC chose for portability, not for RADV.**
* **ncnn** (`src/command.cpp:1386-1440`) builds a flat byte blob and calls
  `vkCmdPushDescriptorSetWithTemplate` — the same conclusion: cheap *because* the template blob is
  pre-computed, not because pushing is cheap.
* **Vulkan best-practice guidance** ("cache descriptor sets, avoid per-draw allocation, prefer a
  device-function table") is all confirmed by the table — but the guidance stops at ~66 ns, which is
  where we already are once the profiler is removed. **No external pattern closes a 6.9 µs gap,
  because there is no 6.9 µs gap in the recording path.**

## A6. The 6.9 µs is an accounting artefact — and it is the thing to fix first

`record_seconds()` is `RECORD_NS`, a thread-local bumped only inside `record()`
(`dispatch_batch.rs:1023`). The printed figure is `record_seconds() * 1e3 / runs_n`
(`breccia-cli/src/main.rs:2718`) — divided by **the number of runs**, not by tokens. Meanwhile:

* `RECORD_NS` is reset **after the prompt** of each run (`graph.rs:3280`), so the numerator covers the
  run's `--gen` decode tokens.
* `sess.dispatches` (`graph.rs:1025/1079`) is **never reset** — not after the prompt, not between runs;
  it is cumulative from process start, prompt included. Grep confirms: the only resets are
  `Census.dispatches` (the stage array) and `pf_merged_dispatches`.
* The per-stage profile is divided by the same `sess.dispatches` (`main.rs:2735`), so it inherits the
  same mismatch.
* And the parts of the budget do not sum to the token: `12.7 + 32.4 + 12.8 + 7.0 + 1.2 + 1.1 = 67.2 ms`
  against **56.9 ms/token of wall**. Recording happens *while* earlier submits drain (that is what the
  6-slot ring is for), so `record` and `wait-gpu` overlap by construction — treating them as additive
  double-counts.

**What the numbers can actually support.** Numerator and denominator are from different windows, so
the ratio is not a per-dispatch price. Two independent checks bound it:

1. The whole `record()` body measures **266 ns** (§A2) — every Vulkan call, every thread-local, the
   hash key, and the profiler. There is no code path in `record()` that can reach 6.9 µs; the gap is
   25× the measured total, not 25× a missing sub-phase.
2. `12.7 ms ÷ 266 ns = 47 700` record calls. At a plausible 614–1 843 dispatches/token that is
   26–78 tokens — i.e. the numerator is a **multi-token** quantity (a `--gen 25…32` run would produce
   exactly this), while the printed line labels it "ms/token".

**Fix (before any further recording work, ~10 lines, no GPU):** reset `RECORD_NS` **and**
`sess.dispatches` **and** `sess.submits` at the same point (`batch_cmd_reset()` already exists and is
called at `graph.rs:423`), and divide by tokens (`forwards`), not `runs_n`. Then the profile line
becomes a real per-dispatch price and the "22 % of the token" claim either survives or dies on data.

## A7. What to change, ranked (all independent, all flaggable)

| # | change | expected | cost | risk |
|---|---|---|---|---|
| R1 | **Fix the counter window** (§A6): reset record/dispatch/submit together, divide by tokens | re-prices the whole budget; likely deletes the "recording is 22 %" workstream | 10 lines | none |
| R2 | **Remove the in-path profiler**: compile the 5 `Instant::now()` pairs behind a `cfg`/`OnceLock` flag (`BRECCIA_RECORD_PROFILE`), keep the counters | **−165 ns/dispatch = −0.30 ms/token** | XS | none (measurement-only code) |
| R3 | **ggml-style set identity**: replace `HashMap<(u64, Vec<vk::Buffer>), DescriptorSet>` with a `Vec`-indexed ring keyed by the dispatch ordinal + a 5-element `buffer`-handle compare | **−20 ns/dispatch now**; unlocks D7 by making the set stable for the life of a CB | S | low |
| R4 | **Hoist `track_disabled()`** into a `OnceLock<bool>` (it is 2 `env::var_os` per dispatch, and it is also read per dispatch in the barrier decision) | −60…80 ns/dispatch | XS | none |
| R5 | Do **not** port push descriptors (§A3.5); keep `bind_descriptor_sets` until the bindless/dbuf scheme lands | — | — | — |
| R6 | Barrier form: normalise on sync2 everywhere; v1 costs +27 ns/dispatch to record for zero GPU benefit | −0.05 ms/token | XS | none |

R2+R3+R4 take the path from 266 → **~70 ns**, i.e. **0.49 → 0.13 ms/token**. That is the whole of the
"recording" line. It is worth doing *because it is nearly free*, not because it is 22 % of the token.

## A8. Top-3 experiments for the next round

**X1 — make the counter honest, then re-price (R1, no GPU-safety risk, do it first).** Patch the reset
points, print `record ns / dispatch` and `record ms / token` from the same window, and re-run the
standard short form. Success = a record line ≤0.5 ms/token; if it is still >5 ms, the gap is real and
`BRECCIA_RECORD_PROFILE=1` prints exactly which of the 8 phases — that is then a *single* measurement,
not a hunt.

**X2 — R2+R3+R4 in one round (three different files, one flag each).** Expected 266 → ~70 ns/dispatch.
Verify with the harness's replica row as the reference and the engine's own corrected profile. Gate:
bit-identical 32/32 parity (these are pure host-side changes; any parity change is a bug).

**X3 — the thing that actually costs 32 ms: one GPU-timeline experiment, not a recording one.**
Re-run the `bpr=10` dependent-chain ladder (`06a` protocol) *interleaved* with the engine's real
expert GEMV, with `UNROLL=2` and a software-prefetch variant of the block loop, on the RX. The open
question is whether the 0.45 µs × `bpr` term is a cold-load chain that a pipelined block loop can hide
(round 10 §A1's decision rule says it can, and `06` measured −9 % for `UNROLL=2`). This is the only
place on the box where 30 ms of the 57 ms token can move today.

---

# Part B — MLC-LLM's Vulkan backend

## B1. What the "Vulkan backend" actually is

`mlc-llm` contains **no Vulkan code of its own**. The backend is three things in `3rdparty/tvm`:
`src/backend/vulkan/runtime/` (device, stream, buffer, wrapped-func — the dispatch layer),
`src/backend/vulkan/codegen/` (SPIR-V emitter for TIR), and `src/runtime/vm/paged_kv_cache.cc` (the
paged KV cache as a VM builtin). Model-side, `mlc_llm/op/*.py` holds TIR kernels (`moe_matmul`,
`attention`, `batch_matmul`) and `mlc_llm/quantization/` the group-quant formats.

The target is **`vulkan:generic` and nothing else** (`support/auto_target.py:539`):
`thread_warp_size: 1`, `max_threads_per_block: 256`, `max_shared_memory_per_block: 32768`. There is no
AMD- or RADV-specific target entry. So the Vulkan path is WebGPU/SteamDeck portability, which is why
round 01's verdict ("nothing in its kernels is worth porting") holds — **this round does not revise
that; it audits the runtime layer instead**, which round 01 did not cover.

## B2. Per-dispatch recording (the part that maps onto our 1 843)

`vulkan_wrapped_func.cc:76-100` (immediate path — taken whenever `VK_KHR_descriptor_update_template` +
push descriptors are present, which is *always* on RADV): `bindPipeline` →
**`vkCmdPushDescriptorSetWithTemplateKHR`** (pre-built template, one memcpy) → `pushConstants` →
`dispatch` → one `VkMemoryBarrier` `SHADER_WRITE|READ → TRANSFER|SHADER_WRITE|READ`, stages
`COMPUTE → TRANSFER|COMPUTE`.

The deferred path (`:120-190`) is the non-immediate fallback: per-kernel descriptor set, re-written only
when the buffer arguments change, launches queued as `std::function`s and flushed on a descriptor
conflict.

**Against our floor:** the immediate path is 5 recorded calls, ours is 6, and the one-call difference is
the push-descriptor call which measures **+85 ns** here. So their record path is ~150 ns vs our 66 ns
floor — **MLC is not a recording model to copy**; ggml is (§A5). The transferable detail is negative:
their barrier is unconditional per dispatch, exactly the shape our per-resource tracker already
improves on.

## B3. Command buffers and submissions per token

One `VulkanStream` per (device, CPU thread), one command buffer, `ONE_TIME_SUBMIT`, begun in the
constructor. `Synchronize()` = `vkEndCommandBuffer` → `vkQueueSubmit` → **a busy
`vkWaitForFences` loop with a ~1 s timeout** → `vkResetCommandBuffer` → begin again
(`vulkan_stream.cc:146-186`). Flushes happen on a descriptor-set conflict or an explicit sync.

**This is the anti-pattern we must not copy**: every flush is a full host↔device round trip, and the
conflict rule (`LaunchDeferred`'s `any_of` over that set's queued tokens) is O(n) per launch, so a long
CB degrades quadratically before it flushes. Our 6-slot ring with fence-waited slots and a
no-wait `end_submit` is strictly better and already measured (`08`). **Nothing to take.**

## B4. KV cache layout and paging

`src/runtime/vm/paged_kv_cache.cc` (3 336 lines) + `tvm/relax/frontend/nn/llm/kv_cache.py`.

* **Paged, page size = `--page-size` (default 16 tokens)**, `global_block_pool_` of fixed-size blocks,
  per-sequence `last_block_idx` + `seq_length`, page table `int32` per layer.
* **A forward pass is "begin → append → attention per depth → end".** `BeginForward(seq_ids,
  append_lengths)` builds, from the *page table only*, a **block arc by depth**:
  `GetBlockIdsOnDepth` + `GetChunkedBlockIds` coalesce contiguous blocks at the same depth
  (`kPagedKVCacheMaxBlockDepth` ≈ 5), and each depth group becomes **one attention call** with its own
  kernel choice. Token-tree/spec-decode masks ride the same structure.
* **Slot mapping is computed on device** by a TIR kernel when `NeedKernelBeginForward()` (i.e. the
  attention kernels want per-token slot ids), otherwise on the host into pre-allocated
  `HostMemoryVector`s (`commit_copy_src/dst_pos_in_page_table_host_`) — the "we make them class members
  to avoid repetitive allocation time in BeginForward" comment at `:266` is the same lesson as our
  `SETS` cache.
* **`append_before_attn_`** decides whether the current token's KV append precedes attention
  (`!support_sliding_window && use_decode_kernel_.back()`); when it is false the kernel reads the cache
  *and* the in-flight token separately. That single boolean is the whole "fusion vs ordering" question
  we solved with a barrier, and it tells you MLC pays it as a kernel variant instead.
* **No quantised KV in the Vulkan path** — the cache is `dtype` (f16/bf16/f32) only.
* `ForkSequence` / `PopNBack` / `CommitAcceptedTokenTreeNodes` give prefix reuse and spec-decode
  rollback; round 01 already extracted the radix/COW idea.

## B5. Quantisation

* Group quant, **group_size 32** for the shipped `q4f16_1`/`q4f16_0`/`q3f16_1` (`q4f16_0` = 32 with
  `KN` layout, `q4f16_1` = 32 with `NK`; int3 variants use 40), int4 packed into `uint32`, one f16
  scale per group, model dtype f16/bf16 (`quantization.py:60-140`). Everything is **weights-only**:
  activations stay f16.
* Dequantisation is **fused into the matmul** by compiler passes
  (`compiler_pass/fuse_dequantize_matmul.py`, `fuse_dequantize_take.py`,
  `fuse_dequantize_transpose.py`), i.e. the same "decode in the kernel epilogue" shape we use — no
  materialised f16 weight buffer anywhere.
* `moe_matmul.gemv` (`op/moe_matmul.py:13-130`): `thread_binding(experts_per_tok, thread="blockIdx.y")`
  — **one dispatch for all selected experts at decode**, indexing the weight tensor with
  `indptr[0, e]` *on device*.
* `moe_matmul.group_gemm` (`:386-480`): `BLK_M/N/K = 8/128/32`, `TX/TY = 8/32`, **`CTA_COUNT = 1024`
  persistent blocks** walking a flattened `(expert, tile)` space whose segment boundaries come from
  `indptr` (`sum[1] = ceil(indptr[1]-indptr[0], BLK_M) * tiles_per_row`, then `sum[1] += …` per expert
  in a `while (cur_e < Ne)` loop). **This is the D2 algorithm, in full, and it needs no host map.**

## B6. Prefill vs decode

**One compiled graph serves both.** Sequence length is symbolic; `prefill_chunk_size` is a compile-time
bound. The differences at run time are (a) `is_decode_request_ = all(append_lengths == 1)` and
`use_decode_kernel_` chosen **per depth**, (b) `append_before_attn_`, (c) prefill uses the ragged
(paged, variable-length) attention kernel. Additionally
`compiler_pass/low_batch_specialization.py` emits **three variants** of every low-batch GEMM (plain,
`LowBatchGEMV(2)`, `LowBatchGEMV(4)` — buckets `[2, 4]`) and picks at dispatch time, so decode at
B ∈ {1,2,4} does not pay prefill's tile shape.

That is a *cheaper* answer to the same problem our D-list attacks with fusion: **freeze the geometry,
specialise at compile time by bucket, select at run time.** It does not remove dispatches, which is why
it is not a substitute for D1–D5 — but it is the right shape for our multi-stream goal: one graph at
`(n_streams, q_max)` and a bucket per `n_streams`.

## B7. Multi-adapter

There is **no heterogeneous or layer-split multi-GPU** in MLC. `cpp/multi_gpu` is TP (shard weights) and
PP (`pipeline_parallel_rewrite.py`, `layer_partition`) with NCCL/disaggregated-KV transports; the
`disaggregation`/`nvshmem` code is CUDA-only and the KV-cache builtin's `enable_disaggregation` flag
gates C++ paths that do not exist for Vulkan. **Nothing transfers to RX + MI50 with a host lane.** Our
placement problem (48 blocks across two unequal cards, with a CPU lane) is not a problem MLC has.

## B8. Ranked adaptations

| # | adaptation | value | cost | risk |
|---|---|---|---|---|
| 1 | **`indptr`-driven persistent-CTA group GEMM** (`moe_matmul.py:386-480`) as the concrete algorithm for **D2**: fixed grid of ~#CU×waves blocks, tile space flattened per expert from a device-side `indptr`, one dispatch for all experts at both decode (`grid.y = k`) and prefill (persistent loop) | High — it *is* D2, with a worked tile schedule (`8/128/32`, `8×32` threads) | M | medium (silent wrong-map risk; gate on digests) |
| 2 | **Compile-time low-batch buckets** (`low_batch_specialization.py`): 3 variants of each GEMV for B ∈ {1,2,4}, selected at dispatch — the cheapest possible preparation for the 4–6 stream goal, and it needs no new kernel | High for multi-stream, and it *de-risks* the multi-stream round | S (build-time + a dispatch table) | low |
| 3 | **`append_before_attn` as an explicit per-layer capability** rather than "barrier or not": when a layer can read the in-flight token, the KV append disappears from the critical path | Medium | S | low |
| 4 | **Per-depth block coalescing** from the page table (group blocks by depth, one attention call per group) — relevant only if/when we page KV | Low now (our KV is per-block contiguous) | M | low |
| 5 | Do **not** copy the stream/flush model (§B3), the push-descriptor record path (§B2), the generic `thread_warp_size=1` target (§B1), or the CUDA-only multi-GPU stack (§B7) | — | — | — |

## B9. What does not transfer

The whole TVM apparatus: Relax VM dispatch, `T.view`/`T.call_pure_packed` builtins,
`vm.builtin.paged_attention_kv_cache_create` with its 15-argument function table, symbolic-shape
specialisation (we compile nothing — our shapes are fixed per config), the tree-attention mask
machinery for spec decode, and `tirx`. Read `moe_matmul.py` for the tile loop and
`paged_kv_cache.cc:1096-1200` for the depth/coalescing logic; ignore the rest.

---

## Sources

* New measurements (this round): `research-harnesses/r11-record-floor/` — `src/main.rs` (12 record
  combos, cbfrag, hostcost, mt), full commands in §A1.
* Our code: `crates/breccia-kernels/src/dispatch_batch.rs` (`record()` at `:840-1035`, `track_disabled`
  at `:316`, `SETS` at `:100`, `RECORD_NS` at `:1023`), `crates/breccia-cli/src/main.rs:2718-2760`
  (the record/profile print), `crates/breccia-engine/src/graph.rs:1025/1079` (dispatch counter),
  `:3274-3282` (the post-prompt reset), `crates/breccia-vk/src/ctx.rs:110-130` (the loader `Entry` is
  kept in `Host` — correct).
* Reference engines: `llama.cpp-ggml-vulkan/ggml/src/ggml-vulkan/ggml-vulkan-common.h:265-330`;
  `mlc-llm/3rdparty/tvm/src/backend/vulkan/runtime/{vulkan_wrapped_func.cc,vulkan_stream.cc}`,
  `src/runtime/vm/paged_kv_cache.cc:1096-1200`, `python/tvm/relax/frontend/nn/llm/kv_cache.py:559-736`;
  `mlc-llm/python/mlc_llm/{quantization/quantization.py,op/moe_matmul.py,support/auto_target.py}`,
  `python/mlc_llm/compiler_pass/low_batch_specialization.py`; `ncnn/src/command.cpp:1386-1440`.
* Prior rounds: `04` (the 68 → 212 ns descriptor-allocation delta, reproduced as +168 ns), `06`/`06a`
  (cold-first-record, barrier forms), `08` (CB boundaries, submit economics), `10` (A0 denominator
  trap — the same failure mode recurs in §A6).

## Harness trap worth recording

`ash::Entry` owns the `libloading::Library`; **dropping it `dlclose`s `libvulkan.so` and every stored
`Instance`/`Device` function pointer dangles.** Symptom: a perfectly valid `vkGetPhysicalDeviceProperties`
segfaults minutes later, with a backtrace that lands in an unmapped page. `breccia-vk` gets this right
(keeps `entry` in `Host`); a fresh probe must too (`std::mem::forget(entry)` or store it).
