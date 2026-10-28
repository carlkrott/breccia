# 39 — Unified work queue, CPU L3 as a managed resource, persistent shader, coordinator cores and optional iGPU or lavapipe consumers

Scope: docs only, one round. Base f998d47; all line numbers are at that base. Nothing was built, run, benchmarked or changed on any system, and no process was touched. No gain is measured here. Every gain is UNVERIFIED until the GPU qualification card runs.

## 0. Verdict

- **Unified queue (Q1): the decode lane already packs its work, so a dynamic claim cannot move decode time.** At 32 pool threads, `choose_shards` returns 3 row shards per expert, which makes 30 tasks, one per worker (expert.rs:371-385). Perfect balance would cut the lane phase by 6.25% at most, and only a finer split than `choose_shards` allows can reach it (derived). A queue shared across the two streams merges nothing: the lane is one mutex-guarded pool (graph.rs:3134-3142) and the two streams' lanes never overlap. A dynamic claim pays only in prefill, where the task count is `distinct * k` (expert.rs:762, 804).
- **Miss-split (Q1, card 1.2) is not bit-identical, and it is worth a card only on the RX 6950 XT stream.** Each expert moved from the CPU lane to the GPU adds one copy: 0.061 ms on the RX 6950 XT link (28.7 GB/s, brief) against about 0.12-0.18 ms of lane time removed (derived). On the MI50 link (about 2.9 GB/s, brief) the same copy costs 0.60 ms, more than the lane time it saves. The gate is token parity plus a logit difference report, not bit identity.
- **CPU L3 (Q2): no CAT or MBA without owner approval, and the expected benefit is small.** Expert weights are read once per token with no reuse, so CAT can only protect the small hot state. Pseudo-lock looks unavailable on this CPU (fetched source, not checked live). CCD pinning has a low prior (doc 08 §6.3).
- **Persistent consumer (Q3): measure before building.** Commit a2c6dad puts the record path at 3.2 µs per dispatch (about 16 ms per token). The brief puts it at 0.28 µs (about 1.4 ms per token). Card 3.1 settles this with the existing timers and no code change. Any consumer must be bounded and MI50-first, and the brief's mega-kernel negative applies to it.
- **Coordinator cores (Q4): measure first.** Lane work runs only while the coordinator is blocked in `run_phase`, so the coordinator and the pool do not overlap by construction. Pinning cannot remove record or wait time. Under the chooser model, reserving one core leaves the decode plan at 3 shards. Reserving three crosses a cliff to 4 shards with a 0.5-expert makespan, against 0.333 (derived).
- **iGPU and lavapipe (Q5): one kill-test card for an iGPU consumer. Lavapipe is a correctness oracle, not a throughput consumer, and gets no card.**
- Eight cards are ranked in §9. The first is a measurement that needs no code change.

## 1. Scope, method and sources

Scope: the five questions in the round brief. For each: what exists, the mapping onto Breccia code, risks, and flag-OFF cards with a measurable gate.

Source labels used in the text: [code] read in the tree at f998d47; [doc] read in the tree's docs; [commit] git history at base; [brief] the round brief's measured figures, not re-measured here; [fetch] web fetch summary or extraction, not verbatim; [snippet] search-result snippet; [registry] grep of the local cargo registry; [live] read-only local check; [derived] arithmetic on the labelled inputs, with the formula given.

| Source | Used for | How read |
|---|---|---|
| `crates/breccia-engine/src/multi.rs` | two-stream interleave, `token_all`, lane coverage | [code] full file |
| `crates/breccia-kernels/src/cpu/expert.rs` | shard chooser, pool, `run_phase`, worker loop, `moe_topk` | [code] cited ranges |
| `crates/breccia-kernels/src/dispatch_batch.rs` | §8n marker, thread-local batch state, fence wait, record timers | [code] cited ranges and grep |
| `crates/breccia-engine/src/graph.rs` | pool lock, lane call, thread default, coverage metric, fence-only submit | [code] cited ranges and grep |
| `crates/breccia-vk/src/ctx.rs` | `Timeline::wait_host`, `WAIT_TIMEOUT_NS`, `wait_fence` | [code] cited ranges and grep |
| `crates/breccia-cli/src/main.rs` | `BRECCIA_GPU_TIME`, serial prefill per stream | [code] cited ranges |
| ash 0.38 `Device::get_fence_status` | non-blocking fence test signature | [registry] grep; named in prose only |
| `docs/BRIEF.md` | measured figures, settled negatives, safety rules | [brief] full read |
| `docs/research/38-cpu-gpu-pipelined-moe.md` | card format and flag names | [doc] |
| `docs/phase-4/02-untried-levers.md` | miss-split, parity gate, prefetch result | [doc] §5 and cited lines |
| `docs/research/30-cpu-lowbit-expert-kernels.md` | topology, taskset probe, per-CCD card | [doc] §6.2, cards 1 and 6 |
| `docs/phase-3/08-cpu-expert-lane.md` | CCD thread counts | [doc] §6.3, §7 |
| `docs/research/05-ple-io-and-tensorrt-llm.md` | host-visible gather cost | [doc] grep |
| `docs/research/01-vulkan-and-mlc.md` | per-expert region size | [doc] L121 |
| Commit a2c6dad | record-path claim, dispatch count | [commit] message and stats |
| Kernel resctrl documentation (docs.kernel.org/filesystems/resctrl.html) | mount, CAT, MBA, CDP | [fetch] summary |
| Upstream `fs/resctrl/pseudo_lock.c` | pseudo-lock on AMD | [fetch] extraction, not verbatim |
| Kernel cpu-isolation admin guide | isolcpus, nohz_full, cpuset partitions | [fetch] summary |
| Vulkan synchronization chapter | host write ordering | [fetch] truncated at 100,000 of 709,942 characters; host-write section not reached |
| Vulkan command-buffers chapter (offset 200000) | forward progress | [fetch] summary |
| amdgpu module parameters | `lockup_timeout` | [fetch] extraction; two versions conflict |
| sched_setaffinity(2) | permission rule | [fetch] summary |
| Mesa llvmpipe documentation | lavapipe architecture | [snippet] |
| Mesa issue 5472 | lavapipe conformance | [snippet] |
| GPU forward-progress paper (OOPSLA 2021) | spinning-kernel hazard | [snippet] |
| Local checks | `lockup_timeout` value; installed Vulkan ICD manifests | [live] read only |

Failed fetches (404): the kernel x86 resctrl page, the amdgpu gpu_reset page, the Vulkan fundamentals page and the Mesa lavapipe page. Their claims are not used.

## 2. Pinned facts

Host lane
- Decode: k = 10 routed experts per block; 48 blocks; 30 resident on the GPU and 18 streamed; one expert region is 1.746 MB [brief; doc 01 L121].
- Default pool threads are the processor count clamped to 1..32, which is 32 on this host (graph.rs:4155-4166). The comments that say 16 threads (multi.rs:25, graph.rs:440-442) are stale.
- Pool workers are spawned as named threads with no affinity (expert.rs:389-413). Doc 30 §6.2 says the same (L147-152).
- Shard chooser constants: `MIN_SHARD_ROWS` 16 (expert.rs:363), `MAX_SHARDS` 4 (365), `PACK_TOLERANCE` 1.15 (367), `choose_shards` (371-385). Phase 1 calls it with `k` tasks over `n_ff` rows (482); phase 2 with `n_embd` rows (516); prefill with `distinct * k` (762, 804).
- `run_phase` assigns task i to bucket `i % threads` (430), sends one batch per non-empty bucket, then blocks on `recv` once per busy worker (445). Results land in slots by sequence number (446-450). The worker loop blocks on `recv` (859-860). The phase-2 sum `out[r0+i] += w*v` runs on the calling thread in sequence order (546).
- Lane call: `self.pool.lock()` then `moe_topk` (graph.rs:3134-3142). Lane time counts as covered only when another stream was in flight.
- Interleave: `token_all` (multi.rs:178-218). The block order is fixed at 192-207. "Covered" means another stream is in flight (196).
- Prefill covered metric: `lane_covered_s += lane_s.min(0.0)` adds zero (graph.rs:2462).

GPU submit and wait
- Decode segments are submitted fence-only: `end_submit(gpu, None)` at graph.rs:1158 and 1212.
- Batch state is thread-local (dispatch_batch.rs:66-80), so a segment is visible only to the submitting thread. Other threads see a null fence slot (`ensure_slots`, 188). `wait` returns `Ok(0.0)` on a null fence (1011), calls `gpu.wait_fence`, and clears the slot (1030).
- §8n says the marker goes through `Timeline::wait_host` (dispatch_batch.rs:22-31). `wait_host` (ctx.rs:409) has no caller.
- Bounded wait: `WAIT_TIMEOUT_NS` is 10 s (ctx.rs:1310). Its comment says an unbounded wait on 2026-10-26 rebooted the host.
- Record and wait timers: `RECORD_NS` and `RECORDED` (dispatch_batch.rs:87-89), read and reset at 793-806; `WAIT_NS` (94), accumulated at 887.
- `BRECCIA_GPU_TIME` is read at main.rs:1715, and the GPU-time table is at main.rs:2836. Prefill runs serially per stream when `prefill_chunk > 0` (main.rs:2120-2128).

Brief measurements [brief, not re-measured]
- Dual decode: 13.06-14.81 tok/s aggregate (1024-token prompt); lane 42-63 ms per step; fence 69-88 ms per step; per-card fence MI50 47.9 ms and RX 6950 XT 21.9 ms per step; GPU busy MI50 45-60%, RX 6950 XT 60-68%.
- Prefill: serial 154.6 s (MI50 90.3, RX 6950 XT 64.4); concurrent 92.0-94.6 s (MI50 92.0-92.4, RX 6950 XT 68.3-69.4); parity true.
- Shared-card two-stream: 114 ms per token pair (multi.rs:29-33).
- Host links: MI50 PCIe gen3 x4, about 2.9 GB/s; RX 6950 XT 28.7 GB/s.

Record cost [commit a2c6dad]
- 5167 dispatches per token (36 per submit). Single stream 17.56 tok/s, 56.94 ms per token. The commit's components are record 22.4, wait-gpu 33.4, lane 13.4 and head 7.0 ms. They sum to 76.2 ms, so they overlap and are not disjoint.
- The commit says 3.2 µs per dispatch is "the whole record path", and that frozen-CB replay "deletes the whole path (~16 ms/token of host time)".

## 3. Q1: unified work queue with a proportional split

What exists
- The decode shard plan comes from `choose_shards`, not from a fixed split. Trace for k = 10 at the stated thread counts (derived; the rows are assumed to satisfy rows/4 >= 16, which also covers the 3-shard rows, and `n_ff` was not read):

| Threads | Shards | Tasks | Makespan (expert units) | Perfect balance |
|---|---|---|---|---|
| 32 | 3 | 30 | 1/3 = 0.333 | 10/32 = 0.3125 |
| 31 | 3 | 30 | 0.333 | 0.323 |
| 30 | 3 | 30 | 0.333 | 0.333 |
| 29 | 4 | 40 | 0.5 | 0.345 |
| 16 | 3 | 30 | 2/3 = 0.667 | 0.625 |

- At 32 threads each worker gets at most one task, so the makespan is the slowest single task under static assignment or a claim alike. A claim has zero decode effect there, even with task-time variance.
- Below 30 workers some worker takes two tasks, and a claim shortens the makespan only if task times differ. With equal rows, round-robin already gives the same makespan (checked by hand for 16 workers: 14 workers take two tasks under either assignment). So the claim matters for decode only below 30 threads and only under variance, such as the doc 30 16-thread probe.
- Ceiling for perfect balance at 32 threads: 1 - 0.3125/0.3333 = 6.25% of lane phase time (derived). A claim cannot reach it. Exactly 0.3125 needs s = 16 at k = 10 (160 tasks, 5 per worker), above MAX_SHARDS = 4 (expert.rs:365), so the bound is a shard-plan question, not an assignment one.
- Prefill is where a claim can matter. With T tasks on 32 workers, the static round-robin makespan is ceil(T/32) tasks against a perfect T/32, so the ceiling is 1 - (T/32)/ceil(T/32). It is zero when T is a multiple of 32 (derived). T was not measured.
- Cross-stream queue: both streams use one pool behind one mutex, and the loop runs their lanes one after another (multi.rs:192-207). A shared queue reorders serial work only, so it has no decode gain at 32 threads.

Mapping and cards
- Card 1.1 changes only the assignment in `run_phase`. Decode output should not change, because results are placed by sequence number and the phase-2 sum runs in sequence order (expert.rs:446-450, 546). The row independence of the shard routine was not read in full, and the card's gate checks it.
- Miss-split (card 1.2) is the doc 02 §5 idea: the s cheapest misses per block are copied into scratch VRAM slots and computed on the GPU, and the CPU lane keeps the rest. Doc 02 L200-204 states the gate as bit-identical logits against the production server, and cites an 8e-8 CPU-versus-GPU difference that flipped a near-tie token in Basalt. A split across devices therefore cannot be bit-identical. Doc 02 L241 says "Bit-identical", which contradicts L200-204 for this case.
- Derived trade, 18 streamed blocks per token. One stream's lane share is 21-32 ms per token (lane 42-63 ms per step, split across two streams), over 180 streamed experts, so about 0.12-0.18 ms of lane time per expert (derived).
  - MI50: 1.746 MB / 2.9 GB/s = 0.60 ms per copy. At s = 1 the added copies are 18 x 0.60 = 10.8 ms per token, against 2.1-3.2 ms of lane time removed on a per-expert count. Under the chooser model s = 1 raises the lane phase by 50% (k = 9 gives 0.5 against 0.333, §8 item 13). At s = 3 the copies are about 32 ms per token. The MI50 stream is not carded.
  - RX 6950 XT: 1.746 MB / 28.7 GB/s = 0.061 ms per copy. At s = 2 the added copies are 36 x 0.061 = 2.2 ms per token and about 63 MB of link traffic per token, against 5.3-8.0 ms of lane time removed (25% of the 21-32 ms lane share, the chooser-model cut from k = 10 to k = 8). At s = 3 the copies cost 3.3 ms and 94 MB and remove no more lane time, because k = 7 also gives 0.25 (derived).
- A prefill split across cards is closed. Each stream's KV state lives on its own card, and direct P2P is a settled negative (brief §7).

Card 1.1. Dynamic shard claim for the lane (flag OFF)
- Change: replace the static `buckets[i % threads]` assignment in `run_phase` (expert.rs:430) with a shared claim index, one task per claim. First add per-worker busy time and task counts per phase. Keep the claim only if the counters show skew above one task. The chooser fix (§8 item 13) rides the same flag: `choose_shards` returns the minimum-makespan `s`, ties to the smaller `s`. At 32 threads and k = 10 it still returns 3, so the fix matters only for other k and for prefill.
- Flag: BRECCIA_LANE_CLAIM. This is the same name as doc 38 card C. If that card ships, this card covers the prefill side only.
- Slots: none added. Results stay in the seq-indexed slots and the two phase barriers (expert.rs:446-450).
- Gate: bit identity (`to_bits`) of every output against the static path, on decode and prefill; lane ms per step; prefill wall time.
- Ceiling: decode zero at 32 threads (derived; a claim cannot change one task per worker). Prefill 1 - (T/32)/ceil(T/32) (derived).
- Risk: with equal task costs the static plan is already near ideal, so the card is worth building only if the counters show skew.

Gain: UNVERIFIED - requires the GPU qualification card

Card 1.2. Miss-split on the RX 6950 XT stream only (flag OFF)
- Change: on the RX 6950 XT stream's streamed blocks, s misses per block are copied into scratch VRAM slots and computed on the GPU, and the CPU lane takes the rest. s is fixed at 2 per block for the first run. s = 1 is excluded, because at k = 9 the chooser falls back to 0.5 (§8 item 13). The choice of misses follows a fixed rule (router weight order, doc 02 §5) and never depends on timing. Add counters for PCIe bytes per token and lane ms.
- Flag: BRECCIA_LANE_SPLIT.
- Slots: adds scratch VRAM slots of 1.746 MB each (doc 02 §5 proposes 16, about 27 MiB). Size them explicitly; the decode mailbox ring (`STREAM_SLOTS = 8`, graph.rs:271) is not sized for them.
- Gate: token parity 32/32 against the unsplit path, plus a logit difference report (max and mean absolute difference per step); lane ms per step falls; RX 6950 XT fence ms per step does not rise; PCIe bytes per token rise by the predicted about 63 MB at s = 2 (36 copies x 1.746 MB, derived); MI50 stream unchanged.
- Ceiling: lane time removed is 25% of the lane phase at s = 2, which is 5.3-8.0 ms per token (chooser model, derived); the copies it adds cost 2.2 ms on the link. s = 3 removes no more lane time, because k = 7 also gives 0.25. The benefit exists only if the RX 6950 XT stream waits on the lane, which is not measured.
- Risk: host f32 and GPU arithmetic differ numerically, so the parity gate is token-level, not bitwise.

Gain: UNVERIFIED - requires the GPU qualification card

## 4. Q2: CPU L3 as a managed resource

What exists
- No resctrl use, affinity call or libc binding appears in the engine or kernel crates (grep). `breccia-profile` declares libc 0.2 (Cargo.toml:15); `breccia-engine` and `breccia-kernels` do not.
- Pool threads are unpinned (expert.rs:389-413; doc 30 §6.2).
- Brief kernel facts: the CPU flags cat_l3, cdp_l3, mba, cqm_llc, cqm_occup_llc and rdt_a are present; CONFIG_X86_CPU_RESCTRL, CONFIG_RESCTRL_FS and CONFIG_RESCTRL_FS_PSEUDO_LOCK are set; resctrl is not mounted [brief].
- Topology: two CCDs with a 32 MiB L3 each. CCD0 is CPUs 0-7 and 16-23; CCD1 is 8-15 and 24-31 (doc 30 §6.2).

Mapping
- Decode streams 10 x 1.746 MB = 17.46 MB per block through the lane (derived; 52% of one 32 MiB L3). Over 18 streamed blocks that is about 314 MB per token, consistent with the brief's 301 MiB.
- No expert data is reused within or across tokens, so streamed weights only evict other lines. The state worth protecting (activations, output, scratch) is small. Its size was not measured.
- MBA shapes CPU-issued memory traffic. Whether it limits device DMA is unknown, so its effect on GPU phases must be measured.
- Pseudo-lock: a fetched source extraction says `rdtgroup_locksetup_enter` returns -EINVAL when the prefetch-disable bits are zero, and an earlier extraction says that helper returns 0 for non-Intel family 6. Pseudo-lock is probably unavailable on this CPU. It is not checked live and gets no card.
- isolcpus and nohz_full are boot-time kernel parameters. Owner approval is needed, with rollback (remove the parameter, reboot). Not carded.

Card 2.1. CAT partition for the lane workers (flag OFF, owner-approved mount)
- Change: after the owner approves the resctrl mount and its rollback text, put the lane workers in a CAT group with a small way mask and leave the coordinator in the default group. Read occupancy from monitoring groups to check the effect.
- Flag: BRECCIA_LANE_CAT.
- Slots: none added.
- Approval and rollback: mounting resctrl and writing group schemata are system changes. Rollback (proposed here; needs owner approval with the mount): write the full CBM and 100% MB to the default group, remove the groups, unmount. CDP needs its own mount option.
- Gate: bit identity (CAT changes timing, not numerics); L3 occupancy of the coordinator's state before and after; lane ms per step A/B; no rise in GPU-phase fence time.
- Prior: low. Streamed data has no reuse, so the saving is at most the refetch of the hot state.

Gain: UNVERIFIED - requires the GPU qualification card

Card 2.2. Per-CCD lane pool (flag OFF)
- Change: pin the pool's threads to one CCD with in-process `sched_setaffinity`. Run the doc 30 card 1 taskset probe first, at --threads 16, and continue only if it shows a placement effect. No system change is needed. The caller needs the same UID (sched_setaffinity(2)).
- Flag: BRECCIA_LANE_CCD.
- Slots: none added.
- Gate: the probe first; then bit identity; lane ms at 32 threads at or below the 36 ms gate in doc 30 card 6.
- Risk and prior: doc 08 §6.3 measured 8 threads on CCD0, CCD1 or spread at 12.8-13.0 GB/s, so packing rather than placement was the limit (old dispatch). Prior low. The engine and kernel crates need a libc binding or an `extern "C"` declaration.

Gain: UNVERIFIED - requires the GPU qualification card

## 5. Q3: persistent shader and a GPU-side queue consumer

What exists
- Decode submits fence-only and the host waits on each block's fence (graph.rs:1158, 1212; dispatch_batch.rs:1011-1030). The §8n marker is documented through `Timeline::wait_host`, which has no caller (§8).
- A non-blocking test exists in ash 0.38 as `Device::get_fence_status`. The breccia crates do not use it (grep). Doc 38 card A uses that test for ready-first order.
- The open question is the record cost (§2). 16.5 ms per token (a2c6dad; 5167 x 3.2 µs) against 1.45 ms per token (brief; 5167 x 0.28 µs). Per stream token (135-153 ms; derived from the brief's aggregate, assuming both streams advance one token per step): the first is 11-12% of one stream's token. The coordinator records both streams serially in `token_all` (multi.rs:178-218), so the two add: about 22-24% of the step for the a2c6dad figure, and about 2% for the brief figure (1.45 ms x 2).

Mapping (design)
- One workgroup per block's consumer. The host writes a work list for the block, and the consumer returns.
- The only mid-block host dependency is the router. The GPU computes router output, the host reads it and writes the misses, and the consumer picks them up. That is the only place a poll is needed. Cap the poll at 50 ms; then the consumer exits with a status word and the host re-dispatches. No cross-workgroup spin.
- Doc 05 measures a host-visible gather at 46.9 ns per row on MI50 and 64.6 ns per row on RX 6950 XT. That is a gather, not a poll; no poll cost was measured here.
- Host waits stay bounded at 10 s (ctx.rs:1310). The GPU poll cap sits well below the lowest `lockup_timeout` figure found (2 s; see §8).

Risks
- Forward progress: the Vulkan command-buffers chapter (fetch summary) leaves forward progress to the application and says nothing about spinning shaders. The brief's mega-kernel negative rests on the same hazard: a deadlock becomes a ring timeout and a BACO reset (brief §7). The consumer is kept to one workgroup and a bounded poll to stay out of that class.
- Host-write visibility: whether host writes to a mapped buffer are visible to a GPU poll without an extra barrier is not confirmed. The synchronization fetch was truncated at 100,000 of 709,942 characters, before the host-write section.
- The gain is bounded by the record share that card 3.1 measures.

Card 3.1. Gate 0: measure the record path (no code change)
- Change: none in code. Run the existing `BRECCIA_GPU_TIME` breakdown (main.rs:1715, table at main.rs:2836) with disjoint timers: record (`RECORD_NS`, dispatch_batch.rs:87-89) and wait (`WAIT_NS`, 94), per token, with the dispatch count. Compare with 5167 dispatches per token (a2c6dad).
- Flag: none. Uses the existing BRECCIA_GPU_TIME.
- Slots: none (measurement only).
- Gate: record share of a step. Proposed thresholds, not derived: 10% or more opens card 3.2 as worth a kernel; below 3% closes cards 3.2 and 4.1. The a2c6dad figure would be 22-24% of a step with both streams recording; the brief figure about 2%.
- Resolves: the record-cost conflict in §8 item 1.

Gain: UNVERIFIED - requires the GPU qualification card

Card 3.2. Persistent consumer for one block, MI50 first (flag OFF)
- Change: one workgroup consumes the host-written work list for one block and returns. The bounded poll covers only the router dependency in §5. The host checks readiness with `Device::get_fence_status`; the poll must not call `wait_fence` or clear the slot (dispatch_batch.rs:1011-1030).
- Flag: BRECCIA_PERSIST_CONSUMER.
- Slots: one host-written work-list buffer per block; no expert copies.
- Gate, run only after card 3.1 shows record at or above 10% of a step:
  - Every GPU run under `timeout 90`, one GPU process at a time (brief §9.1).
  - MI50 (headless) first; RX 6950 XT only after it passes (brief §9.2).
  - Validation layer on for the first runs; spirv-val at build (brief §9.3).
  - Zero GPU resets in the journal over the run. A reset stops the line (brief §9.6).
  - Token parity 32/32, and a record ms-per-token reduction measured with the same timers as card 3.1.

Gain: UNVERIFIED - requires the GPU qualification card

## 6. Q4: dedicated coordinator cores

What exists
- The coordinator is the calling thread of `token_all`. It records and waits on GPU work (dispatch_batch) and calls the lane, which runs on the pool.
- Lane work runs only while the coordinator is blocked in `run_phase`'s `recv` (expert.rs:445). Pool workers block in `recv` in the worker loop (859-860) whenever no batch is queued. The coordinator and the pool are therefore never busy together by construction. No other decode-time thread was found in the read ranges; §10 lists this as unchecked.
- Consequence: pinning the coordinator cannot remove record or wait time, which is coordinator time. It can only remove scheduler preemption on the coordinator. Card 4.1 measures that first.

Thread-count model (derived from the §3 table; not measured)
- At 31 threads the 3-shard plan is unchanged, so reserving one core costs nothing in this model. A linear model (lane x 32/31) would give +1.35 to +2.03 ms per step for a lane of 42-63 ms. That is an upper bound, not the expectation.
- At 29 threads (three cores reserved) the plan moves to 4 shards and the makespan rises from 1/3 to 1/2 expert, which is +50% on the lane phase under this model.
- The model ignores DRAM bandwidth. If the lane is DRAM-bound, fewer threads cost less and the cliff is smaller. Not measured.

Card 4.1. Coordinator core reservation, measure first (flag OFF)
- Change: step 1 measures the coordinator's involuntary context switches across `token_all`, with no pinning. Step 2 runs only if that count is material: the pool is sized to threads minus one, and the coordinator is pinned in-process.
- Flag: BRECCIA_COORD_CPU.
- Slots: none added.
- Gate: lane ms per step within +2.03 ms of baseline (the linear upper bound); greedy parity 32/32; step ms no worse; involuntary switches per token must fall.
- Guard: do not reserve more than one core without re-running the chooser; 29 threads is the cliff in §3.

Gain: UNVERIFIED - requires the GPU qualification card

## 7. Q5: iGPU consumer and lavapipe

iGPU
- The integrated GPU has 2 compute units (brief). The RX 6950 XT has 80 (spec knowledge, not fetched). The ratio is about 2.5%, so an iGPU consumer can add at most a few per cent (derived; unverified).
- It shares system DRAM with the host lane. Whether the lane is DRAM-bound is not measured, so the kill test is the gate.
- Devices are resolved by PCI identity, never by enumeration order (brief §2).

Card 5.1. iGPU consumer kill test (flag OFF)
- Change: a fixed slice of expert rows runs on the integrated GPU, selected by PCI identity, and results go back to the same rows.
- Flag: BRECCIA_IGPU_LANE.
- Slots: not designed here; any buffer needs a fixed size and a counter first.
- Kill test: lane ms per step must not rise more than 5% with the consumer on. Token parity 32/32. If the lane rises above 5%, the card is closed.

Gain: UNVERIFIED - requires the GPU qualification card

Lavapipe: no card
- Lavapipe is Mesa's CPU Vulkan driver (llvmpipe), multithreaded across the host cores (snippet). It would run on the lane's own cores, so it contends with the lane instead of adding throughput.
- Its only use is as a correctness oracle for shader work. That needs an ICD install (a system change, owner approval), and lavapipe is not conformant (Mesa issue 5472, snippet).
- The existing token parity gate against the production server already covers correctness for a GPU card. Deviation from the per-section card rule: lavapipe gets no card, and the reason is stated here.

## 8. Discrepancies

1. Record cost. Commit a2c6dad says 3.2 µs per dispatch (about 16 ms per token) and that frozen-CB replay "stays item 1". Brief §7 says the real record cost is 0.28 µs and replay is worth 0.8-1.1 ms. Card 3.1 settles this; this doc does not.
2. a2c6dad's components (record, wait, lane, head) sum to 76.2 ms against a 56.94 ms token, so they overlap. The record figure is not a disjoint timer.
3. Concurrent prefill (brief: 92.0-94.6 s) has no flag or code in the tree. BRECCIA_MULTI_PREFILL_CONCURRENT (brief) and BRECCIA_PREFILL_CONCURRENT (doc 38 L186) appear in no crate. The build behind the brief's numbers is unknown.
4. Stale thread counts. multi.rs:25 and graph.rs:440-442 say 16 threads; the default is 32 (graph.rs:4155-4166); doc 08 §6.3 recommends --threads 12.
5. The prefill covered metric is always 0 (graph.rs:2462).
6. §8n marker. dispatch_batch.rs:22-31 names `Timeline::wait_host`; that function (ctx.rs:409) has no caller; decode segments are fence-only (graph.rs:1158, 1212).
7. Miss-split bit identity. Doc 02 L241 says bit-identical; doc 02 L200-204 states the parity gate as bit-identical logits and cites an 8e-8 flip. This doc uses the token-parity gate.
8. Doc 02 §5 uses 33 GB/s for MI50 ("33 + 28.7 = 61.7 GB/s") against the brief's 2.9 GB/s. Doc 02's figures are not this host's. Not reconciled.
9. amdgpu `lockup_timeout`. A fetched 7.3.0 page gives 2000 ms as the default. An earlier 6.14 extraction gives 60000 ms (compute) and 10000 ms (others). Unresolved; the live value reads empty.
10. Vulkan synchronization fetch truncated before the host-write section (§5).
11. Doc 01 L121 gives 508 regions per block; the brief has 512 experts. Not reconciled. This doc uses 1.746 MB per expert.
12. Doc 30 card 1 and doc 30 card 6 gates are used as stated there. They were not re-measured.
13. `choose_shards` returns the first `s` whose makespan is within `PACK_TOLERANCE` of perfect, or the last `s` tried if none is (expert.rs:371-385; the loop sets `best = s` before its check). It is not the minimum-makespan `s`. Replicating the loop by hand at 32 threads gives 4, 4, 4, 4, 3 for k = 6 to 10. At k = 9 the minimum is 3 (0.333) and the chooser returns 4 (0.5). Prefill calls the same chooser with `distinct * k` tasks (expert.rs:762, 804), so it can reach prefill too. Card 1.2 at s = 2 (k = 8) is not affected, because k = 8 already returns the minimum. Card 1.1 carries the fix.

## 9. Ranked card list

Rank rule: expected gain per unit of risk. Card 3.1 is free and gates cards 3.2 and 4.1. Card 1.1 is bit-exact and cheap. Card 1.2 needs a logit report. The rest wait on a measurement or an owner approval.

| Rank | Card | Flag | Change | Gate | Gain |
|---|---|---|---|---|---|
| 1 | 3.1 Gate 0 measurement | none (BRECCIA_GPU_TIME) | disjoint record and wait timers per token | record at or above 10% of a step opens 3.2; below 3% closes 3.2 and 4.1 (proposed thresholds) | Gain: UNVERIFIED - requires the GPU qualification card |
| 2 | 1.1 Dynamic shard claim | BRECCIA_LANE_CLAIM | claim queue in `run_phase`; counters first | bit identity; lane ms per step; prefill wall time | Gain: UNVERIFIED - requires the GPU qualification card |
| 3 | 1.2 Miss-split, RX 6950 XT only | BRECCIA_LANE_SPLIT | fixed s per block by router output | token parity 32/32 plus logit report; PCIe bytes per token | Gain: UNVERIFIED - requires the GPU qualification card |
| 4 | 4.1 Coordinator core, measure first | BRECCIA_COORD_CPU | involuntary-switch counter, then one-core reservation | lane within +2.03 ms per step; parity 32/32 | Gain: UNVERIFIED - requires the GPU qualification card |
| 5 | 2.2 Per-CCD lane pool | BRECCIA_LANE_CCD | in-process affinity after the doc 30 probe | probe first; bit identity; lane at or below 36 ms at 32 threads | Gain: UNVERIFIED - requires the GPU qualification card |
| 6 | 2.1 CAT partition for lane workers | BRECCIA_LANE_CAT | after owner-approved mount | bit identity; L3 occupancy; lane A/B | Gain: UNVERIFIED - requires the GPU qualification card |
| 7 | 3.2 Persistent consumer, MI50 first | BRECCIA_PERSIST_CONSUMER | one-block bounded consumer | record ms per token down; parity 32/32; zero resets; validation layer clean | Gain: UNVERIFIED - requires the GPU qualification card |
| 8 | 5.1 iGPU consumer kill test | BRECCIA_IGPU_LANE | fixed slice selected by PCI identity | lane ms per step up no more than 5%; parity 32/32 | Gain: UNVERIFIED - requires the GPU qualification card |

Lavapipe is not ranked (§7).

## 10. Unverified

- Record cost per dispatch: 3.2 µs (a2c6dad) against 0.28 µs (brief). Card 3.1.
- Coordinator record and wait time in dual decode: derived only.
- Whether the lane is DRAM-bound. This decides whether the thread-count model in §6 and the iGPU kill test are realistic.
- Whether rows/4 >= 16 holds for the decode shapes. The expert row counts come from GGUF tensor shapes at run time (layout.rs:177-199), and the model file was not read, so the §3 table and card 1.2 assume it.
- Row independence of the shard routine (not read in full); card 1.1's bit-identity gate depends on it.
- Vulkan host-write ordering for a host-written work list (synchronization fetch truncated).
- `lockup_timeout` value on this host (2000 ms against 60000/10000 ms; live value unreadable).
- Resctrl state on this host beyond the brief (brief says not mounted; not checked live).
- Pseudo-lock on this CPU (fetched source only).
- Whether MBA affects device DMA.
- The iGPU's compute ratio (spec knowledge for the RX 6950 XT; the iGPU figure is from the brief).
- Whether any thread other than the coordinator and pool does work during decode (grep, not exhaustive).
- Whether the RX 6950 XT stream waits on lane time; this decides whether card 1.2 gains anything (§3).
- The build behind the brief's concurrent prefill numbers.
- The timing and behaviour of `Device::get_fence_status` on this driver (signature only, from the [registry] grep in §1).
- Lane time under the chooser's split: the 25% cut at s = 2 and the 50% regression at s = 1 are a model of the quoted loop. No lane time was measured.
