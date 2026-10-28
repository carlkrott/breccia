# 38 — CPU-GPU-I/O pipelined scheduling for hybrid MoE decode and prefill

Round 38. Docs only: no engine code changed, no build, no cargo, no GPU work, and the model was not loaded. Base commit 9ac402e. Every gain below is UNVERIFIED until the GPU qualification card runs. Line numbers are at the base. Derived numbers are labelled "derived" and use only the brief's figures and numbers already pinned in docs 32, 35 and 37.

## 0. Verdict

- Two mechanisms transfer. MoE-Lightning's per-(layer, micro-batch) task graph keeps I/O and compute on separate resources, and FlexGen's zig-zag loop issues weight load, cache load and cache store on separate CUDA streams with one sync per step. Breccia already has the stream-level form of this (two cards, block-level interleave). What it lacks is readiness-driven ordering inside a step, and overlap across streams in prefill.
- The largest pinned gap is the fence order. `token_all` always finishes stream 0 before stream 1 (multi.rs:192-207), with no readiness check. The brief's fence waits sum to 69.8 ms/step (derived). Card A (ready-first order, flag OFF) addresses this without new threads and keeps each stream's arithmetic unchanged, but it needs a non-blocking segment test that does not exist yet (section 3).
- The expert lane is not a queue. `moe_topk` buckets shards statically (`i % threads`) and runs two barrier-separated phases per call (expert.rs:425, 463). Card C (dynamic shard claim) and card D (fixed per-worker buffers) are the pinned options. Neither has a measured gain.
- Prefill is serial per stream in the CLI agents driver (main.rs:2120-2128). `Multi::generate` instead feeds the prompt token by token in lockstep (multi.rs:234-236). Card E (concurrent chunk prefill across cards) and card F (ragged start, no padding) are the prefill cards.
- Two corrections to working assumptions. (1) The lane's default thread count is min(logical CPUs, 32) (graph.rs:4155-4165; 8 if the Linux CPU table cannot be read), which is 32 on this host. The comments saying 16 (graph.rs:441, multi.rs:25) are stale. (2) The CLI pads shorter prompts by repeating their last token to the common length (main.rs:2109-2118). Padding therefore enters each stream's context and changes its output, so any parity gate must use the padded prompt as its reference.
- Nothing here is a measured speedup. The only numbers are derived bounds (section 5). The 1.47-1.67x step bound uses the brief's RX 6950 XT busy figure (60-68 %). Doc 32 gives 53-62 ms per step for the same card, which changes the binding term (section 5). That conflict is unresolved, so the bound is uncertain.

## 1. Scope, pins and method

| Source | Pin | How read |
|---|---|---|
| MoE-Lightning | arXiv 2411.11217 v1 (18 Nov 2024) | summary of the arXiv HTML page; no verbatim text; no code linked on the abstract page |
| Fiddler | arXiv 2402.07033 v3 (1 May 2025); repository efeslab/fiddler at 227715bf (2024-04-28) | abstract summary; code read in src/fiddler/mixtral.py |
| HybriMoE | arXiv 2504.05897 v1 (8 Apr 2025); repository PKU-SEC-Lab/HybriMoE at 508a85b9 (2025-12-15) | abstract summary; code read in ktransformers/operators/experts.py |
| FlexGen | arXiv 2303.06865 v2 (12 Jun 2023); repository FMInference/FlexGen at 004ffef8 (2024-10-28) | the PDF text was unreadable through the fetch tool; schedule taken from flexllmgen/flex_opt.py |
| llama.cpp | PR #6017, merge commit f30ea47a (2024-03-13) | survey only; constants read at HEAD |
| Breccia | base 9ac402e | read directly |

Method: read-only. Fetched code was saved to scratch-38/src (untracked) and read, never executed.

Card t_c99b13ce is not in the repository (a search of docs/ finds no reference), so its content is taken from the brief's description: "1-stream per-card measure, ready-first fence order, then per-device threads and a lane queue". Section 7 relates the cards to it.

## 2. Per-engine scheduling mechanism

### 2.1 MoE-Lightning (arXiv 2411.11217 v1)

- Decode only. Each step loops over layers i and micro-batches j. A prologue starts CPU attention for the first two micro-batches. Per (i, j) the tasks are LoadH (I/O, CPU to GPU), PostAttn and PreAttn (GPU), OffloadQKV (I/O, GPU to CPU), CPUAttn (CPU), and the paged-weight transfers W_PintoG and W_CtoPin.
- Opposite-direction transfers overlap, and same-direction transfers are serialised. This is the summary's paraphrase of section 4.1, UNVERIFIED as verbatim.
- Paged weights: the page count equals the micro-batch count, and the pages are double-buffered (appendix A.1, summary).
- Sizing uses a Hierarchical Roofline Model: P = min(P_peak, B_peak · I, B_peak^{j,i} · I_j) (Eq. 7), and per-layer T = max(comm, T_cpu, T_gpu) (Eq. 12). Requests are sorted by length and packed greedily into the micro-batch with the fewest tokens (appendix A.2).
- No prefill schedule. MoE-Lightning runs attention on the CPU. Breccia runs GPU phases with a host expert lane for the routed experts (multi.rs:1-9).

### 2.2 Fiddler (arXiv 2402.07033 v3; efeslab/fiddler at 227715bf)

- Per-layer expert placement by exhaustive search: all 2^8 CPU/GPU assignments of the eight experts. Expert i costs tokens × latency_cpu on the CPU and latency_gpu on the GPU (0 if resident). The code keeps the assignment with the least total cost (src/fiddler/mixtral.py).
- The code's objective is the sum of the assigned costs. The comment says to minimise max(sum CPU, sum GPU). The code does not match its comment.
- latency_cpu = 7 and latency_gpu = 70 are constants with a TODO, not measured at runtime.
- The branch labelled "prefill stage with offloading" is not gated on the decode flag, so the subset search also runs in decode.
- The all-GPU baseline (cpu_offload == 0) copies each non-resident expert to the GPU when it is used. No pipelined copy was found in that path.
- Speedups from the abstract summary: 1.26x single batch, 1.30x long prefill, 11.57x beam search. UNVERIFIED against the PDF.

### 2.3 HybriMoE (arXiv 2504.05897 v1; PKU-SEC-Lab/HybriMoE at 508a85b9)

- Prefetch: `KExpertsCache.prefetch_expert` (experts.py:1057) prefetches the top-k experts of the next layer on a prefetch stream. The choice comes from `KScoreAwareCache.get_experts_to_prefetch` (line 662), an exponential moving average of routing scores with alpha 0.5 and p = 12. `wait_prefetch` (line 1076) blocks on it.
- Per-layer CPU/GPU split: `calc_experts` (line 1128) returns all experts unchanged when not generating, and otherwise runs HSS, a greedy assignment. HSS sorts loads and unloads by id_counts descending, sends the largest to the GPU queue and the smallest to the CPU queue, and compares projected totals.
- Cost constants: cpu = 1 × tokens + 0.79, gpu = 0.5 × tokens, move = 0.41791011235117914. The code comments say "need to be tested". HSS runs in decode only.
- In generate mode id_counts is [1] × expert_num, so the sort carries no information in decode (code read).
- Speedups from the abstract summary: 1.33x prefill and 1.70x decode against a leading hybrid framework. UNVERIFIED.

### 2.4 FlexGen (arXiv 2303.06865 v2; FMInference/FlexGen at 004ffef8)

- Zig-zag loop nest over token i, layer j and micro-batch k. The overlapped multi-batch loop (generation_loop_overlap_multi_batch, flexllmgen/flex_opt.py) runs, per step: load_weight(i, j+1, k), load_cache(i, j, k+1), store_hidden(i, j, k-1), load_hidden(i, j, k+1), compute_layer(i, j, k), store_cache(i, j, k-1), then one sync. A prologue loads weight (0, 0, k) and hidden (0, 0, 0).
- Three CUDA streams: load_weight_stream, load_cache_stream and store_cache_stream (flex_opt.py:618-620). Where compute is issued was not read this round.
- The non-overlapped loop (generation_loop_normal, about line 913) runs load_weight with overlap=False before each layer. It is the baseline.
- UNVERIFIED: whether load_weight(i, j+1, k) reloads for every k. The paper's zig-zag text (weight reuse across micro-batches) could not be read, and the load_weight code path was not read this round.

### 2.5 llama.cpp pipeline parallelism (survey only)

- PR #6017 (merge f30ea47a, 2024-03-13) added the multi-copy scheduler. At HEAD, GGML_SCHED_MAX_COPIES defaults to 4 (ggml-backend.cpp:911-912), `sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1` (ggml-backend.cpp:1973), and the CMake default is "4" (ggml/CMakeLists.txt:188).
- n_copies copies of the graph inputs let the next micro-batch's transfers start while the previous one computes. The copy and compute internals beyond these constants were not read this round.
- n_ubatch is the compute micro-batch size, separate from the logical batch, and prompt processing is chunked by it. The clamp rules between n_batch and n_ubatch were not read this round (UNVERIFIED here).
- Per-expert CPU/GPU placement was not checked against the scheduler source this round (UNVERIFIED).

## 3. Breccia at the base: pinned facts

- `token_all` (multi.rs:178-218). Every stream runs begin_token. All streams submit block 0. Then for each block b, for each stream s in the fixed order 0..n: compute `covered` (is another stream in flight), finish_block(b, s), and submit_block(b+1, s). Heads are submitted for all streams, then read. The order over s is fixed (multi.rs:192-207). There is no readiness check, and there is one host thread (no per-device thread).
- Waits. Every decode segment is submitted with no timeline signal: `batch_cmd::end_submit(gpu, None)` at graph.rs:1158 and 1212, and dispatch_batch.rs:1042. Completion is a Vulkan fence. finish_block waits through `batch_cmd::wait` (graph.rs:3021; dispatch_batch.rs:1011), which calls `Gpu::wait_fence` (crates/breccia-vk/src/ctx.rs:1551). No non-blocking fence query exists in breccia-vk or dispatch_batch. `Timeline::wait_host` (ctx.rs:409) and `Timeline::counter` (ctx.rs:423) exist, and the dispatch_batch.rs header describes a timeline marker for the two-segment path, but no caller passes a timeline signal today. GPU-resident (placed) blocks defer their wait (`defer_placed_waits`, graph.rs:98, read at graph.rs:3014-3016; disabled by BRECCIA_NO_DEFER_PLACED). Their segment is collected by flush, by wait_all, or by the ring's back-pressure in `begin`. Whether that back-pressure time is added to `dev_wait_s` was not read (UNVERIFIED). `fence_s` is the sum of `dev_wait_s` (multi.rs:270). Doc 36 reports that two-stream `--gpu-experts auto` runs zero resident blocks, so for that setting the defer path does not apply (UNVERIFIED for the brief's bench).
- Lane. The expert pool is shared by all sessions through an `Arc` (multi.rs:150-158) and locked for each lane call (`self.pool.lock()` in finish_block, graph.rs ~3120-3145). `ExpertPool` (crates/breccia-kernels/src/cpu/expert.rs:301) starts `threads` workers, each with its own job channel and one shared done channel (expert.rs:389-401). `moe_topk` (expert.rs:463) runs phase 1 (gate, up, SwiGLU) and then phase 2 (down) as two barriers. `run_phase` (expert.rs:425) puts shard i into bucket `i % threads`, sends one batch message per non-empty bucket, and then receives one done message per busy bucket. `choose_shards` packs shards with MIN_SHARD_ROWS = 16, MAX_SHARDS = 4 and PACK_TOLERANCE = 1.15 (expert.rs:363-367).
- Thread count. `default_threads` returns logical CPUs clamped to [1, 32], or 8 when the Linux CPU table cannot be read (graph.rs:4155-4165). The comments at graph.rs:441 and multi.rs:25 say 16 threads, which is stale for a 32-logical host. The comment at graph.rs:4153-4154 says the recorded §8q gate was run with `--threads 32`. Which count the current bench uses is UNVERIFIED (it depends on the bench invocation).
- Lane accounting. In finish_block, `lane_covered_s` is credited only when `covered` (graph.rs:3142). In prefill_block the covered term is `self.lane_s.min(0.0)` (graph.rs:2462), which is always 0 for a non-negative lane time. Prefill lanes therefore never count as covered.
- Expert slot ring. STREAM_SLOTS = 8 (graph.rs:271) sizes the decode stream mailbox (graph.rs:889) and the prefill staging. The merged prefill path walks distinct experts in slot order and drains (submit_batch and flush) when the slots wrap (graph.rs:2715-2750). It streams each distinct expert once per chunk.
- Prefill. `Session::prefill` (graph.rs:2102) is one blocking call, single device only. `prefill_plan` splits a length into ceil(len / cap) even chunks, with cap from prefill_cap(target).
- CLI agents driver. Prompts are padded to a common length by repeating each prompt's last id (main.rs:2109-2118). The chunked prefill loop (main.rs:2120-2128) runs stream by stream: each stream runs all of its chunks before the next stream starts. The non-chunked path runs token_all once per position (main.rs:2130-2133).
- Multi::generate (multi.rs:225-236). The prompt is fed token by token through token_all with the same token to every stream (`vec![t; ns]`). It is not chunked, and the prompt work is duplicated across streams. Every stream gets the same tokens, so under greedy decode they produce the same output. This path is a bench shape, not an agent workload.
- Settled negatives (brief): direct GPU-to-GPU copies and layer split across cards. Multi::new requires exactly one device per stream (multi.rs:139-147).

Brief claim check: "chunked prefill in generate runs the streams serially" holds for the CLI agents driver (main.rs:2120-2128), not for Multi::generate.

## 4. Mapping onto token_all

### 4.1 Ready-first fence order

- Current: fixed stream order (multi.rs:192-207). Each fence wait is on the single host thread, so while the host waits on stream 0 it cannot run stream 1's lane, even if stream 1's segment has finished.
- Engines: Fiddler and HybriMoE have no multi-device fence order. MoE-Lightning's Algorithm 1 is a fixed software-pipelined order over (layer, micro-batch), with I/O and compute on separate resources; it is not a readiness-driven order, so it is an analogue for overlap only. FlexGen's single sync per step is the coarse form.
- What transfers: a non-blocking completion test answers "is this stream's segment done" without blocking. Breccia has none for a fence-only segment (section 3). Two ways to add one: (a) a fence-status query (`vkGetFenceStatus`) on the segment's stored fence, which leaves submission unchanged; or (b) pass `Some((timeline, value))` to `end_submit` at graph.rs:1158 and 1212 and poll `Timeline::counter` (ctx.rs:423), which adds a timeline signal to every decode submit. Ready-first changes only the host order of finish calls, so each stream's block sequence is unchanged.
- Derived: the brief's fence waits sum to 47.9 + 21.9 = 69.8 ms per step (both streams, summed). How much of that ready-first order can remove is UNVERIFIED. It depends on how often the RX 6950 XT segment completes before the MI50 segment.

### 4.2 Per-device driver threads

- Current: one host thread. Lanes are serialised by design (multi.rs:1-33 module comment, graph.rs:439-441).
- Engines: HybriMoE's prefetch stream and FlexGen's three CUDA streams give device-level overlap, but each uses one host thread issuing into streams.
- What transfers, with a limit: the expert pool is a single Mutex, so two driver threads still run lanes one at a time. A second thread only hides one stream's fence wait under the other stream's lane. Card A gets that on one thread.
- Derived: a driver thread per stream on top of the pool workers oversubscribes a 32-logical-CPU host during a lane, because 32 pool workers already fill the CPUs. Build it only if card A leaves a fence residual.

### 4.3 Shared lane work queue as a fixed-slot ring

- Current: there is no queue. Each moe_topk call builds shard vectors, copies x into an Arc, allocates act, and clones an Arc per shard (expert.rs:463 onward). Each phase sends at most one message per worker.
- Derived upper bound: messages per step ≤ 48 layers × 2 streams × 2 phases × 32 workers = 6,144. The 48 layers come from the 48 GPU phases in multi.rs:5. The messages are already batched, so any saving lies in allocation and dispatch overhead, which was not measured here.
- Engines: MoE-Lightning's paged weights behind a page table (n pages, double-buffered) are the fixed-slot ring analogue. FlexGen's per-stream load and store slots play the same role.
- What transfers: keep one batch message per worker per phase. The fixed-slot ring is per-worker preallocated buffers (card D), not a new shared queue. This fits the brief's preference for few large fixed slots over many small copies. The research 05 figure (300-360 ns per vkCmdCopyBuffer2 region on MI50) concerns Vulkan copy regions, not channel messages, so it is an analogy only.

### 4.4 Micro-batching streams

- Current: one token per stream per step. Each stream owns a full model on one card (multi.rs:139-147).
- Cross-stream micro-batching would need both streams' weights on one card's VRAM. That reintroduces the serialisation that one card per stream removed (multi.rs:27-33, the 114 ms per token pair measurement). Not carded.
- Within-stream micro-batching is the prefill chunk. MoE-Lightning's micro-batch j maps to a prefill chunk, its paged weights map to the 8-slot expert ring, and its reuse of a layer's weights across micro-batches maps to "stream each distinct expert once per chunk" (graph.rs:2715-2750).
- Derived: streamed-expert bytes per prompt token scale as 1/chunk length if the distinct-expert count per chunk saturates. The brief has no prefill routing trace, so this is UNVERIFIED.

### 4.5 Concurrent chunked prefill across devices

- Current: serial (main.rs:2120-2128). While stream 0 (MI50) prefills, the RX 6950 XT idles, and vice versa.
- Engines: FlexGen overlaps load, compute and store of different micro-batches within one loop, which is the closest analogue. Fiddler's prefill branch is single device.
- What transfers: a submit/finish split of Session::prefill (graph.rs:2102), which is currently one blocking call. The CLI pads prompts to a common length, so both streams have identical chunk plans and chunk c can run on both cards together (main.rs:2109-2128).
- Derived ceiling: serial T_MI50 + T_RX versus concurrent max(T_MI50, T_RX), so at most 2x on the prefill phase only. That holds only if host RAM refaults under the 30 GiB cap and the host-to-device link do not contend. The brief gives no prefill timings, so there is no gain number.

### 4.6 Lockstep vs block-major Infinity-Cache ordering

- Current decode: block-major across streams and lockstep in time. For each block b, both streams run block b (multi.rs:192-207). Each stream's own blocks run in order.
- Infinity Cache: the RX 6950 XT has a 128 MiB Infinity Cache. This is a public specification figure and is UNVERIFIED in this round.
- Derived: the streamed set is 34,589,900,800 B over 39 blocks (doc 37), which is 886,920,533 B per block (0.83 GiB). That is about 6.6 times the 128 MiB cache, so one block's experts cannot stay resident across the streams. Block-major ordering therefore gives no reuse window for the streamed set. The RX 6950 XT's resident weights are not sized in the brief, so their reuse is UNVERIFIED.
- Not carded. Skewing stream 1 by k blocks changes only when each stream's GPU phases meet the other's lane. Each stream's own chain is unchanged, so card A covers the same effect.

## 5. Derived arithmetic

Brief inputs: decode step 136 ms (midpoint of the brief's 131-141 ms), lane 42 ms per step summed over streams, MI50 busy 48-58 %, RX 6950 XT busy 60-68 %, fence 47.9 ms per step (MI50) and 21.9 ms per step (RX 6950 XT).

| Quantity | Calculation | Result (derived) |
|---|---|---|
| Lane duty | 42 / 136 | 31 % (the brief rounds to 32 %) |
| Fence sum | 47.9 + 21.9 | 69.8 ms per step |
| Host time accounted | lane 42 + fence 69.8 | 111.8 of 136 ms; residual ≈ 24 ms for submit, encode and other host work (t_submit and t_finish are timed but not reported here) |
| MI50 busy | 0.48-0.58 × 136 | 65-79 ms |
| RX 6950 XT busy | 0.60-0.68 × 136 | 81.6-92.5 ms |
| Step bound, GPU work unchanged | max(RX 6950 XT 81.6-92.5, MI50 65-79, lane 42) | 81.6-92.5 ms, so 136 / (81.6-92.5) = 1.47-1.67x |
| Host floor, if the residual is host work | lane 42 + residual 24 | 66 ms (derived; assumes the residual runs on the one host thread) |
| Step bound with doc 32's RX 6950 XT figure | max(RX 6950 XT 53-62, MI50 65-79, host floor 66) | 66-79 ms, so 136 / (66-79) = 1.72-2.06x (derived bound) |
| Aggregate baseline | 2 streams / 0.136 s | 14.7 tokens/s |

The step bound is the best case when GPU work and lane are perfectly overlapped. It is not a speedup claim, and gains are UNVERIFIED.

Conflict to resolve: doc 32 reports RX 6950 XT busy 53-62 ms per step, which is 39-46 % of 136 ms. That disagrees with the brief's 60-68 %, so the two windows or busy definitions differ. Until this is reconciled, the bound above is uncertain. With doc 32's figure, the RX 6950 XT is no longer the binding GPU constraint and the MI50 bound (65-79 ms) would be. The host floor (66 ms, derived) then binds alongside it, giving the 1.72-2.06x row above.

Prefill ceiling: 2x on the prefill phase at most (section 4.5).
Padding cost: positions saved = sum over streams of (common − len_s). Prompt lengths are not in the brief, so this is UNVERIFIED.
Messages per step: at most 6,144 (section 4.3).
Expert ring: one drain per 8 distinct experts per prefill chunk (from STREAM_SLOTS = 8).
Streamed bytes per block: 886,920,533 B (section 4.6).

## 6. Ranked flag-OFF cards

Ranking is expected value per unit of risk. Card A is first: it targets the largest measured host wait, needs no new threads or memory, and keeps each stream's arithmetic unchanged. Every card defaults to OFF, so the flag-OFF path is the current code.

Gate definitions used on every card: per-card GPU busy % comes from the GPU-time probe (`BRECCIA_GPU_TIME`; flag read at main.rs:1715, table at main.rs:2836). Lane duty is lane_ms_per_step divided by the step time. Refaults are the delta of workingset_refault_file in the cgroup's memory.stat under MemoryMax=30G. Greedy parity means the tokens and last_logits of each stream are bit-identical to flag-OFF on the same padded prompts.

**Card A. Ready-first fence order in token_all**
- Change: in multi.rs:192-207, replace the fixed stream loop with a readiness pass. First add a non-blocking segment test (section 4.1). For each block, poll the streams with a segment in flight. Finish the first stream whose segment is complete. If none is, block on the stream submitted earliest (`batch_cmd::wait`, graph.rs:3021). Each stream still runs its blocks in order, so its arithmetic is unchanged. A poll must not clear the slot's fence or its GPU-time stamps; only `batch_cmd::wait` does that (dispatch_batch.rs:1011-1040).
- Flag: BRECCIA_READY_FIRST.
- Gates: per-card GPU busy %; lane duty; tok/s (aggregate and per stream); greedy parity per stream (identical tokens and last_logits to flag-OFF on the same padded prompts); refaults under MemoryMax=30G. Also fence_ms_per_step per card, split into streamed-block and placed-block fence (placed waits are deferred, section 3), and the fraction of blocks where completion order differs from index order. No segment holds a timeline value today, so the poll primitive is the first step of this card.
- Slots: no buffers or copies are added. The change adds a non-blocking poll and reorders which finished segment is waited on, so the fixed `STREAM_SLOTS = 8` design (graph.rs:271) is unchanged. This follows the research 05 preference for few large fixed slots over many small copies.

Gain: UNVERIFIED - requires the GPU qualification card

**Card B. Per-stream driver threads (refines the per-device threads item of t_c99b13ce)**
- Change: one host thread per stream calling submit_block and finish_block. The lane stays behind the shared Mutex (multi.rs:150-158). Build only if card A leaves a measured fence residual.
- Thread-local constraints (dispatch_batch.rs:66 onward; code read, not run). The batch state is per OS thread: OPEN, FENCES, DEVICES, RECORD_NS, WAIT_NS and the probe state are all `thread_local!`. (1) A segment must be submitted and waited on by the same thread. On any other thread its slot holds a null fence (ensure_slots, dispatch_batch.rs:188), so `wait` returns Ok(0.0) without waiting (dispatch_batch.rs:1011-1016). (2) `batch_cmd_reset` (graph.rs:446) calls `reset_record_seconds` (dispatch_batch.rs:803), which clears only the calling thread's RECORD_NS, RECORDED and WAIT_NS. Under card B the reset runs on every driver thread, and the busy % gate merges WAIT_NS and the probe GPU-time counters across threads. (3) A Session must be created on, or moved to, its driver thread, which needs Session: Send (section 8).
- Flag: BRECCIA_MULTI_THREADS.
- Gates: per-card GPU busy %; lane duty; tok/s (aggregate and per stream); greedy parity per stream; refaults under MemoryMax=30G. Also overlap_efficiency (lane_covered / lane) must not fall below the card A value, and pool workers plus drivers must not exceed the logical CPU count.

Gain: UNVERIFIED - requires the GPU qualification card

**Card C. Dynamic shard claim in the expert pool**
- Change: in run_phase (expert.rs:425), replace the static `i % threads` buckets (expert.rs:430) with a shared atomic next-shard index that workers claim from. Keep the seq-indexed result slots and the two phase barriers. Bit-identity: phase 2 already sums on the calling thread in seq order (expert.rs:539-547), and each shard's values depend only on its inputs, not on which worker ran it. Output is therefore bit-identical if the shard plans from choose_shards (expert.rs:371) are unchanged. This is a code-read argument, not a test. Add per-worker busy counters for the gate.
- Flag: BRECCIA_LANE_CLAIM.
- Gates: per-card GPU busy %; lane duty; tok/s (aggregate and per stream); greedy parity per stream; refaults under MemoryMax=30G. Also lane milliseconds per call (mean and p95) and the max/min per-worker busy ratio. The lane is about 31 % duty and largely covered, so only lane time on the critical path turns into tok/s.

Gain: UNVERIFIED - requires the GPU qualification card

**Card E. Concurrent chunked prefill across cards**
- Change: split Session::prefill (graph.rs:2102) into submit and finish halves. In the CLI loop (main.rs:2120-2128), run chunk c for every stream and then finish them all, instead of stream by stream. Padded prompts give identical chunk plans, so chunk c aligns across streams.
- Flag: BRECCIA_PREFILL_CONCURRENT.
- Gates: per-card GPU busy % during prefill; lane duty; tok/s (aggregate and per stream); greedy parity per stream; refaults under MemoryMax=30G, which matter most here because both streams read experts in the same window. Also prompt wall time.
- Derived ceiling: 2x on the prefill phase only (section 5).
- Slots: the change adds no copies. Each stream keeps its existing fixed `STREAM_SLOTS = 8` ring (graph.rs:271), and the submit/finish split only reorders work, which follows the research 05 preference for few large fixed slots over many small copies.

Gain: UNVERIFIED - requires the GPU qualification card

**Card F. Ragged start without padding**
- Change: keep each prompt at its own length. token_all takes a per-stream position, and a stream starts decoding when its own prompt ends. Shorter prompts stop being conditioned on repeated padding (main.rs:2109-2118). This is a behaviour change, not a timing change.
- Flag: BRECCIA_RAGGED_PREFILL.
- Gates: per-card GPU busy %; lane duty; tok/s (aggregate and per stream); refaults under MemoryMax=30G. Parity is against the unpadded single-stream greedy reference, not against flag-OFF. Also positions saved, the sum over streams of (common − len_s) (derived).

Gain: UNVERIFIED - requires the GPU qualification card

**Card D. Fixed per-worker buffers in the expert pool**
- Change: preallocate per-worker shard buffers and the act and out scratch in ExpertPool (expert.rs:301, 463), instead of the per-call Vec, the x copy into an Arc, and the per-shard Arc clones. Keep one message per worker per phase.
- Flag: BRECCIA_LANE_SLOTS.
- Gates: per-card GPU busy %; lane duty; tok/s (aggregate and per stream); greedy parity per stream (bit-identical); refaults under MemoryMax=30G. Also lane milliseconds per call (mean and p95) and allocations per step.
- The saving is bounded by allocation time, which was not measured.
- Slots: card D keeps the fixed-slot design. It preallocates one buffer per worker in ExpertPool (expert.rs:301) and replaces per-call Vec and Arc allocations rather than adding small copies, which is the research 05 preference.

Gain: UNVERIFIED - requires the GPU qualification card

Not carded, with reasons: block-major Infinity-Cache ordering (the cache is about 6.6 times smaller than one streamed block, derived); stream skew by k blocks (covered by card A); cross-stream micro-batching on one card (reintroduces the serialisation that one card per stream removed).

Measurement fixes, not flag cards, to ship with card A: correct the prefill covered term (graph.rs:2462 adds min(lane_s, 0)), and correct the thread-count comments (graph.rs:441, multi.rs:25). Neither changes outputs.

## 7. Relation to card t_c99b13ce

The card text is not in the repository. The relation below uses the brief's summary of it.

- The "1-stream per-card measure" is the baseline every gate here uses. This doc does not duplicate it.
- The "ready-first fence order" is refined by card A: the mechanism (a non-blocking segment test, to be added; section 4.1) and the parity argument.
- The "per-device threads + lane queue" is refined by cards B, C and D. Threads alone do not parallelise the lane (single Mutex), the lane is statically bucketed, and the right changes are a dynamic claim (C) and per-worker buffers (D), not a new queue. Card B is the per-device threads item at stream granularity: each stream owns one card, so one driver thread per stream is one thread per device. Its thread-local batch state (section 3, card B) means each Session must stay on its own driver thread.
- Cards E and F (prefill) are additions relative to the card as summarised in the brief.

## 8. Unverified and discrepancies

- MoE-Lightning: no code linked. The mechanism comes from the arXiv HTML summary, not verbatim text.
- Fiddler and HybriMoE speedups: abstract summaries only.
- Fiddler objective: the comment says minimise max, and the code minimises the sum. The code is taken as the behaviour.
- HybriMoE cost constants: the code says "need to be tested". No calibration was found.
- FlexGen: the paper's zig-zag text was unreadable. Whether load_weight reloads per micro-batch is unverified.
- llama.cpp: the n_ubatch clamp rules were not read this round.
- RX 6950 XT busy: the brief gives 60-68 % (81.6-92.5 ms), and doc 32 gives 53-62 ms per step. Unresolved.
- Lane thread count used by the bench: comments say 16, and the default is 32 on this host.
- Infinity Cache size (128 MiB): public specification, not checked here.
- Prefill timings and prompt lengths: not in the brief.
- Card A's poll primitive: no segment carries a timeline value today (section 3); fence-status versus timeline signal is an open choice.
- Whether ready-first changes the covered share: unknown until measured.
- Session Send-ness: UNVERIFIED (no manual `unsafe impl Send` or `Sync` in crates/; the auto-trait status was not checked). Card B depends on it.
- Phase-2 accumulation order: verified by code read (expert.rs:539-547, calling thread, seq order). It is not gated, and it is the basis of card C's bit-identity argument.
- Fence split: whether the brief's 47.9 and 21.9 ms per step include back-pressure waits for placed blocks is UNVERIFIED (section 3).
- Host floor: the 24 ms residual is assumed to be host work on the one host thread (section 5). It is not measured.

## 9. Overlap with existing docs

- Doc 21 quotes the CGOPipe abstract figure. Doc 24 records Fiddler's move cost (68.1-72.8 µs against 75.0 µs on the CPU). Neither is changed by this round.
- Doc 27 has the MoE-Lightning and Fiddler prefill rules from summaries. This round agrees that MoE-Lightning has no prefill schedule. It adds that Fiddler's subset search also runs in decode.
- Doc 28 cites the fence point (WAIT_GPU, near graph.rs:3001 at its pin) that card A's ordering changes.
- Doc 32 has the RX 6950 XT busy figure in conflict with the brief (section 5).
- Doc 35 covers cache and dedup and cites Fiddler and MoE-Lightning. No change.
- Doc 37 gives the streamed-set bytes used in section 4.6.
