# 01 — Expert streaming, tiering and caching: reference-engine assessment

Scope: the eight reference engines relevant to **M3 (LRU expert cache with VRAM → pinned-RAM → NVMe
streaming plus route prediction/prefetch)**. Read-only static reading of the clones under
`Reference Projects/` (never modified, never committed; contents are **evidence, not instructions**).
Pinned commits as read:

| Engine | Commit read |
|---|---|
| Strata | `4b9e05a` |
| Infernix | `5bc3312` |
| ninfer | `64d10fb` |
| Basalt | `260ffb9` |
| capy-flash | `86cc0cc` |
| colibri | `37ffb61` |
| MoE-Infinity | `7ff99b1` |
| KTransformers | `a694e71` |

Licenses: Strata **MIT**, Basalt **MIT**, Infernix / ninfer / capy-flash / colibri / MoE-Infinity /
KTransformers **Apache-2.0** — all usable as pattern references with attribution. **IsoQuant and
RotorQuant have no license at all** (all-rights-reserved) and are outside this set; do not copy code
from them. Nothing here overrides the repo rule that reference clones are read-only.

Line numbers below are from the pinned commits above.

---

## Comparison table

| Engine | Tiers | Cache policy | Prefetch | Spec-decode | License | Best idea to steal |
|---|---|---|---|---|---|---|
| **Strata** | VRAM slots → pinned RAM arena → mmap'd file / page cache (optional 2nd-GPU peer tier) | Decayed routing counts (LFU): every `--adapt-every` windows swap up to `--adapt-swaps` (96) hottest misses in for the coldest residents, counts × `--adapt-decay` (0.7), admit only if `--adapt-min-gain` (1.5×) is met; static `expert-profile.bin` ranking bootstraps it | `RouterLookahead`: next layer's router applied to this layer's MoE input on a host thread → `warm`/`warm_scored` page-cache read-ahead, depth 1-4, gated every 16 windows by page-cache coverage (off <0.55, on >0.65) and usefulness | `--mtp` (model's draft layer) + `--spec N` + `--spec-min-p`; suffix/prompt-lookup drafter fallback | MIT | The **adaptive-tier policy parameters** (interval / swaps / decay / min-gain) as a tunable, calibrated per machine — and the *gate* pattern: prefetch only runs while it pays |
| **Infernix** | VRAM frame pool → pinned host slots (residents + ring 128 + prefetch 16 + demotion 32) → unpinned resident records → SSD via unbuffered read agent | **LFRU**: `score = f/(now−last+1)`, global count halved every `halving_period` ticks; per-layer/round `step()` with a DRAM/PCIe **admission token bucket**, victims lowest `(score,key)` outside the current group | On-GPU router of layer *l* publishes layer *l+1*'s top-k′ (16) non-resident experts to `prefetch_req`; transfer agent issues one contiguous DMA per expert and **stops when the next DMA could not land before layer l+1's router**; per-layer **useful-byte ratio <0.3 disables prefetch** for that layer | MTP + DFlash/DFlash2, `--draft-tokens auto` (learned-acceptance width) | Apache-2.0 | The **budget formula and self-disabling usefulness gate** for prefetch, plus the residency-entry encoding (state/generation/frame packed in one u32) for lock-free device-side lookup |
| **ninfer** | Host bank (CPU computes experts in place) → GPU expert cache (byte pool, fixed + reusable regions) | Static per-layer `default_expert_priority()` rank table (48×512, calibrated offline) drives fill order; cache `progress()`/`acquire` with `required`/`prefetch` lists, batched copies (`max_batch_bytes` 96 MiB, 4096 copies, 3 upload slots) | Per-layer `prefetch` list in the acquire request; `UINT64_MAX` absent sentinel in the offset table | Native MTP (1-5 drafts, exact verify) | Apache-2.0 | **Indirect addressing**: `plane_address = base + offsets[expert]` (offset table + divisor table stable for CUDA-graph capture) so slot contents change without re-capturing graphs |
| **Basalt** (Strata fork) | VRAM → pinned RAM (`resident-experts`, `resident-budget-gib` auto) → file mapping; **2nd GPU as peer expert tier**; `--layer-split` alternative | Same decayed-count adaptive tier as Strata but defaults to a round **every window** in serve (`adapt_every = 1`, `generate.cpp:1704`); peer card takes the next-hottest experts; experts dealt "hottest first, to the tier with the least work so far for its bandwidth" | **None** — no `RouterLookahead`, no file-tier io prefetch, no `STRATA_*` env knobs (verified by grep) | Drafter ships in the `.basalt` model file; `spec` default 8 with drafter, `relaxed-drafting` opt-in | MIT | **Startup cost pricing**: it prints `multi-gpu: ... ms per window: expert tier 7.9, layer split 13.5, tensor parallel 9.0 (not built); picked expert tier` — pick a tiering strategy from a measured cost model instead of guessing (`docs/configuration/gpus.md`) |
| **capy-flash** (ninfer fork) | GPU expert cache (per-partition pools, **elastic columns** shared with KV) → pinned host RAM (64.6 GiB) → CPU lane computes misses; `--miss-split` also copies a share of misses to GPU | The most rigorous policy in the set (`src/core/expert_cache_policy.h`): admission by **decayed frequency ≥ threshold** (half-life 256 tokens), insert cap, victim = **frequency × recency** `f·2^(−age/τ)` (τ=4 windows), prompt-window counts weighted by mean recency | **Rejected**: predictor `post_uni` (router of l+1 on l's post-attention state) measured **−2.7%** because PCIe copies steal DRAM bandwidth from the CPU lane; **cache warm-start after the prompt +4.1% adopted** | MTP **K=3** (acceptance 0.533, 2.59 tok/window), DFlash not used | Apache-2.0 | The **A/B discipline and the numbers**: admission alone −11% misses/token and +3.6–4.2% throughput; eviction headroom measured against Belady (+12.6% ideal) |
| **colibri** (pure C) | dense part resident in RAM → per-layer LRU of routed experts → **pinned hot-store** (VRAM/RAM) → disk via `pread`/`WILLNEED`, page cache as free L2 | Per-layer LRU slots (`glm.c:1893-2112`); hot-store via `tier_lfru_score()` = `(heat<<8)\|recency` with **25%+4 hysteresis** against ping-pong (`c/tier.h:13-46`), heat halved by `tier_decay`; `.coli_usage` histogram persists ranking across runs (AUTOPIN) | `PILOT=1`: dedicated I/O thread, lock-free 1P/1C ring, `posix_fadvise(WILLNEED)` for layer l+1's predicted experts (recall **71.6%** measured, vs 41.3% for "previous token" baseline); `PILOT_K` keeps only the confident head of the ranking; plus whole-batch async readahead while the block computes | Native MTP head (int8 only; int4 acceptance collapses), batch-union verify | Apache-2.0 | **Dedicated I/O thread + drop-on-full ring**: submitting fadvise inline blocked (~0.5 ms × 169k calls = +92 s/48 tokens), so the hint queue discards rather than blocks |
| **MoE-Infinity** (Archer) | GPU cache (`device_memory_ratio` 0.9) → host pinned pool → **SSD tensor store** (`O_DIRECT` async, 4 MB chunked reads, priority AIO) | `LRUPolicy` / `ARCPolicy` (`moe_infinity/memory/offloading_policy.py`), plus phase policy: **prefill admissions are transient-on-pressure, decode admissions are cached**, eviction weights 1.0 vs 4.0, starvation limit (`core/prefetch/expert_policy.h`) | Router-logit-driven speculative prefetch with an **overlap byte budget**: `B_window = bandwidth_ewma × max(0, safety_factor·compute_ewma − queue_wait − issue_overhead)`, whole-expert greedy pack sorted by `(-score, pos, id)`, `enforce` is fail-closed | MTP/`speculative_draft` (deprecated sync path); async server does continuous batching | Apache-2.0 | The **overlap-prefetch budget equation** (transfer only what fits in measured spare compute time) and the prefill/decode admission split |
| **KTransformers** | GPU (hot experts) → CPU (cold experts, NUMA-bound worker pool) — no disk tier | **Static placement mask** with four strategies (`uniform`, `frequency`, `front-loading`, `random`) + optional dynamic redistribution at prefill token threshold; `--kt-num-gpu-experts` / `--kt-gpu-experts-ratio` | none (offload scheduler only) | n/a (SGLang-integrated MTP elsewhere) | Apache-2.0 | **Measured value of a frequency-ranked placement**: at 10% GPU experts, 56.6 → 70.2 tok/s (random → dynamic), i.e. *which* experts you keep beats *how many* (`doc/en/kt-kernel/experts-sched-Tutorial.md:178-192`) |

---

## Feature checklist vs M3 needs

| Engine | LRU/LFU in VRAM | Host (pinned RAM) tier | Disk/NVMe tier | Route prediction | Threshold-based skipping | Continuous batching |
|---|---|---|---|---|---|---|
| Strata | ✅ decayed-count adaptive tier (not strict LRU; `ds4_store.hpp:14` has both kLru/kLfu) | ✅ `--resident-experts` / `--resident-budget-gib` | ✅ `FileExpertSource` + io-prefetch + zero-copy `STRATA_COLD_OVERLAP` | ✅ `RouterLookahead` (host thread, next-layer router) | ✅ `STRATA_ROUTE_TAIL_SKIP=7` (skip experts every token routes at rank ≥7) | ⚠️ fixed **batch slots** (`--batch N`, requests queue), not continuous |
| Infernix | ✅ LFRU + admission budget | ✅ pinned slots + unpinned resident records | ✅ SSD tier with read agent + ring | ⚠️ predictor exists but is used **only to protect from eviction**, prefetch class rejected | ❌ | ✅ `Scheduler` + `DecodeShare` prefill/decode time-share (`src/runtime/engine/scheduler.h:19-32`) |
| ninfer | ✅ static priority order + cache replace | ✅ host bank is the primary source (CPU lane) | ⚠️ via artifact reads, no dedicated tier | ❌ static calibrated table only | ❌ | ❌ `max_concurrency` admission only |
| Basalt | ✅ adaptive tier, every window in serve | ✅ resident arena (auto budget) | ⚠️ plain mmap (no io prefetch) | ❌ (dropped vs Strata) | ❌ (dropped vs Strata) | ⚠️ batch slots, + layer-split pipelined windows |
| capy-flash | ✅ `fr` victim = freq×recency + admission threshold | ✅ pinned host backing (64.6 GiB) | ⚠️ mapped artifact only | ✅ built (T19) then **measured and turned off** | ❌ | ❌ explicitly out of scope (`docs/maintainer/engine-architecture.md:46`) |
| colibri | ✅ per-layer LRU + LFRU hot-store with hysteresis | ✅ pinned hot-store (`mlock`) | ✅ `pread`/`WILLNEED` streaming, page cache as L2 | ✅ `PILOT` (71.6% recall), `couple_prefetch` cross-layer coupling | ❌ | ⚠️ single stream + `openai_server.py` |
| MoE-Infinity | ✅ LRU/ARC + phase policy | ✅ pinned memory pool | ✅ `O_DIRECT` AIO store | ✅ router-logit speculative prefetch + overlap budget | ❌ | ✅ async OpenAI server, paged KV |
| KTransformers | ❌ static mask (dynamic only at prefill threshold) | ✅ CPU experts (NUMA) | ❌ | ❌ | ❌ | ✅ via SGLang |

---

## Per-engine notes

### 1. Strata — the canonical reference (MIT)

**How it works.**
*Loading.* Experts are read from a packed `experts.bin` or the GGUF in place. A binary
`expert-profile.bin` (magic `STRP`, version 1) ranks all `n_layers × n_expert` experts —
`read_expert_profile()` at `src/core/expert_cache.cpp:74`, ranking merged with runtime heat by
`rank_learned_profile()` at `src/core/expert_cache.cpp:143-170` (resident-first, then heat, then prior
rank, stable). Learned profiles can be written back (`--expert-profile-save`, `docs/DETAILS.md:359`).

*Tiering.* Three low-RAM modes (`docs/DETAILS.md:223-259`): `--mmap-experts` (file mapping, page cache
is the host tier), `--resident-experts` (everything the GPU lacks is copied into page-locked RAM at
start), `--resident-budget-gib N` (only the N GiB hottest that fit, rest from file). The GPU tier is a
slot table sized by `--expert-cache N` blobs, `auto` = as many as fit after weights/session/reserve,
re-checked after allocation (`docs/DETAILS.md:366-375`). Slots are 512 MiB segments under
`--vram-elastic` so they can be shrunk (`src/core/expert_cache.cpp:297 open_segmented`, `:384 shrink`).

*Policy.* `src/program/generate.cpp:689-699`: every `adapt_every` (default 4) verify windows, swap up
to `adapt_swaps` (96) of the most-routed **missing** experts into VRAM in place of the least-routed
residents; after each round all counts are multiplied by `adapt_decay` (0.7); a swap happens only when
the incoming decayed count beats the victim's by `adapt_min_gain` (1.5×). `--adapt-async` runs the
rounds on a helper thread between windows (`include/strata/core/adapt_async_default.hpp`).
`tools/replay_cache_policy.py` replays a `--dump-routing` trace through this policy offline — a
**simulator for the policy is a deliverable in its own right**.

*Prefetch trigger.* `RouterLookahead` (`include/strata/core/expert_source.hpp:302-355`, wired at
`src/program/generate.cpp:6798-6824`): a host thread evaluates the *next* layer's router weights on
the current layer's MoE input, takes the top-k per token (k≈10), votes across tokens (`pvote_`), and
calls `warm`/`warm_scored(layer, experts, confidence, ahead)` so I/O threads `pread` whole blobs into
the page cache. `STRATA_IO_PREFETCH_DEPTH` (default 1) controls how many layers ahead; confidence lets
the file source rank work (`warm_scored`, `src/core/expert_source.cpp:1081`). The whole mechanism is
gated: `STRATA_IO_PF_GATE=1` re-judges every 16 decode windows and only runs while page cache covers
0.55-0.65 of the expert set **and** while it pays (`docs/DETAILS.md:261-275`).

*Threshold skipping.* `STRATA_ROUTE_TAIL_SKIP=7` (`docs/DETAILS.md:864`): a **missed** expert that
every token of a verify window routes at rank ≥7 is neither copied nor computed — disabled when ≥90%
of experts are cached (`STRATA_TAIL_SKIP_MAX_CACHED`), on batch slots, peer tiers, or non-CUDA
backends. Counters exposed as `tail_skipped`/`tail_experts`
(`include/strata/core/expert_source.hpp:610-613`).

*Scheduling.* Single request by default; `--batch N` keeps N conversations open and decodes them
together in shared windows (one row per slot) — fixed slots with a server-side queue, **not**
continuous batching (`docs/BATCHING.md:1-9`). CPU/GPU split of misses is a fitted model: `PcieModel`
(`include/strata/core/expert_source.hpp:398-481`) fits `--pcie-frac` from observed samples so a share
of misses is DMA'd over PCIe instead of computed on the host.

**Reusable techniques.**
- Adaptive-tier parameter set + decay + min-gain hysteresis (`generate.cpp:689-699`).
- Prefetch **gate** (coverage band + measured usefulness), `docs/DETAILS.md:269-273`.
- `rank_learned_profile` merge order: resident → heat → prior rank (persisted profile).
- Tail-skip with a cache-coverage escape hatch (`docs/DETAILS.md:864`).
- `cudaMemcpyBatchAsync` for groups of ≤128 independent expert uploads (`docs/BATCHED_DMA.md:1-14`).
- Tier observability: `--stats` `expert tiers` line, `/metrics` `hit_rate`, `pcie_share`,
  `ram_blobs`, `file_blobs` (`docs/DETAILS.md:288-301`).

**Warnings / anti-patterns.**
- Adaptation makes output **non-bit-exact** run-to-run; any gate comparison must fix
  `--adapt-every 100000` or accept statistical equivalence (`docs/DETAILS.md:316-318, 503`).
- `STRATA_STAGE_PIN=1` produced **corrupt decode** and is documented as "keep off until fixed"
  (`docs/DETAILS.md:888-890`) — pinned staging is not automatically safe.
- On a Radeon iGPU, io-prefetch plus adaptive tier under memory pressure **reset the GPU** in most
  runs (`docs/DETAILS.md:274-275`) — directly relevant to our RADV/ROCm boxes.
- Duplicate residency across two GPUs cost 15.5% tokens/s until a disjoint-ownership guard was added
  (`docs/DISJOINT_EXPERT_CACHE.md:9-44`) — with 2 GPUs we must not let both tiers cache the same blob.

### 2. Infernix — async prefetch ring and SSD tier (Apache-2.0)

**How it works.** `src/runtime/expert_offload/` is the whole subsystem:
`expert_cache.{h,cpp}` (host policy), `host_expert_tier.{h,cpp}` (pinned slots + SSD read agent),
`route_predictor.{h,cpp}`, `expert_stream`, `memory_plan`, `vram_monitor`, `placement`.

*Tiers.* Device frame pool with a residency table of one u32 per `(layer, expert)`:
bits 31-30 state (`Absent/Loading/Ready`), 29-24 generation, 23-0 frame
(`expert_cache.h:21-34`). Host side: `Options{slots, ring=128, prefetch=16, demotion=32, half_life=64,
slots_per_chunk=512, unpinned, pin_reserve}` (`host_expert_tier.h:53-70`) — pinned slots hold
residents, the read ring, in-flight prefetches and the demotion list; when the OS refuses to pin a
chunk the tier keeps `slots − bytes` and the rest become **unpinned resident records** the device
cannot see (host copies serve CPU demand, fetches copy them into ring slots).

*Policy.* **LFRU** (`expert_cache.h:39-51`): `score = f/(now − last + 1)` with a global use count that
survives eviction, and counts halved every `halving_period` ticks; `step()` takes one routed layer
call's distinct keys, admits misses **only within a DRAM/PCIe token-bucket admission budget** and only
if the newcomer outranks the victim; `credit()` lets a prompt's routes seed the ranking
(`expert_cache.h:69-95`).

*Prefetch trigger.* Design `docs/maintainer/qwen3_8-flash-next-design.md:1238-1266` (§8.7): between
router *l* and router *l+1* there is ≈84-134 µs of GPU work; the GPU router itself evaluates layer
*l+1*'s router on `x_l` (top-k′ = 16 union) and pushes non-resident experts to `prefetch_req[l]`
ordered by weight; the **transfer agent** issues one contiguous DMA per expert and stops when the next
DMA could not land before layer *l+1*'s router (calibrated completion model ⇒ ~1.5 experts/layer at
50 GB/s); predictions beyond the budget go to the CPU engine's warm list. Crucially: *"prefetch is a
latency tool… each layer's prediction recall and useful-byte ratio are published. Prefetch is
disabled automatically for a layer whose useful-byte ratio stays below 0.3."*

*Route predictor as eviction guard.* `RoutePredictor` (`route_predictor.h:6-14`) keeps per-layer
exponentially-decayed transition counts `counts[a][b]` (half-life 4096 tokens, 50 MiB for Flash-Next)
and its predicted keys are **evicted last** (`Guard` callback in `LfruPolicy::step`,
`expert_cache.h:46-49`). Note the deliberate split: the predictor *protects*, it does not prefetch —
"its prefetch half is the rejected prefetch class".

*Scheduling.* `Scheduler` with `DecodeShare` (`src/runtime/engine/scheduler.h:19-32`): each cycle runs
one Control/Decode batch before the prefill turn, and after a prefill unit of wall time *P* the
prefill yields until decode has had `share × P` — only whole units interleave, so chunk grids and
output bits are unchanged.

**Reusable techniques:** residency word encoding; admission token bucket; useful-ratio kill switch;
"staged-but-unused frames are promoted by a role change with no copy" (no-copy promotion,
`§8.7 step 4`); read-agent ring that reuses a slot within a round once its reader is done
(`host_expert_tier.h:4-8`).

**Warnings:** `INFERNIX_TMP_*` flags (route protect, prompt seed) are explicitly temporary A/Bs —
don't build on them; prefetch class was rejected here, so "Infernix has prefetch" is only half true.

### 3. ninfer — origin of Infernix (Apache-2.0)

**How it works.** `src/core/expert_cache.{h,cpp}` (801 lines): experts are `ExpertKey = {bank id,
logical expert}`; all banks share one **byte pool** with a fixed region and a reusable region;
`replace(region, placements)` rebuilds a region wholesale after releasing consumers and waiting for
the upload stream's completion event (`expert_cache.cpp:364-435`). Uploads are batched
(`max_batch_bytes = 96 MiB`, `max_batch_copies = 4096`, `upload_slots = 3`,
`max_in_flight_uses = 8`, `expert_cache.h:29-36`).

*Placement/loading* (`docs/maintainer/expert-weight-cache.md`): `plan_load → resolve_load →
materialize_model`; hosts keep the original quantized parent (no repack), CPU-only loads never create
a CUDA stream or pinned buffer; `CudaTransfer` regions are registered once for the whole run.

*GPU addressing* (the key trick): `resident: plane_address = base + expert*stride`;
`cached: plane_address = base + offsets[expert]`, with `UINT64_MAX` as the unpublished sentinel, and
the offset table / pool / divisor allocations are **address-stable so a CUDA graph can capture them**.

*Prefill ordering / warm:* `src/models/qwen4_exp/program/planning.cpp:32-45` loads
`layer.priority` from a supplied per-layer ranking, else `default_expert_priority()` — a static
48×512 table calibrated from 128 tasks across nine domains
(`src/models/qwen4_exp/program/expert_priority.cpp:8-11`). Warm start measured +4.1% in its descendant.

*Spec/prefix:* native MTP (1-5 drafts, exact verify, `docs/maintainer/qwen4_exp-execution.md:17`);
context reuse with prefix hits, device/host checkpoints (`:20`).

**Warnings:** the priority table is model- and workload-specific — reuse the *mechanism*, not the
numbers. No eviction policy beyond "replace region"; cache dynamics live in the program's planning.

### 4. Basalt — Strata fork: RAM + 2nd-GPU peer tier (MIT)

**How it works / diff vs Strata.** Same skeleton (`src/core/expert_cache.cpp`,
`expert_source.cpp`, `peer_experts.cpp`, `remote_experts.cpp`, `src/spec/*`), but Strata's `apps/`,
`metal/`, `sycl/`, `dsv41/`, `qwen35/` are gone and `vision/`, `program/{auto_plan,multi_gpu_plan,
probe_*}.cpp` are added; Basalt requires NVIDIA Blackwell sm_120 (`README.md:14-18`), so it is a
**design** reference for us, not a code one.

*Tiers.* `resident-experts` / `resident-budget-gib` (auto = free RAM above the floor, capped at what
the GPUs lack) / `mmap-experts` / `shared-expert-arena` (back the arena with a file on `/dev/shm` so
several processes share one copy) — `docs/configuration/index.md:142-149`.

*Peer tier.* Every GPU after the main one holds experts the main card lacks, computes its share of
each decode window, and follows the conversation like the main cache: *"experts are dealt out hottest
first, each to the tier with the least work so far for its memory bandwidth, until every tier is
full"* (`docs/configuration/gpus.md:22-27`). Auto-chosen settings include `peer-exclusive` and
`resident` (`docs/configuration/index.md:66-73`).

*Policy.* Identical decayed-count adaptive tier, but `generate.cpp:1704` sets `adapt_every = 1`
(a round every window) for a single-GPU serve, and `docs/configuration/index.md:153-158` documents
the round: main VRAM gets the most-used misses, the peer card the next most-used, then decay.

*Startup cost model.* `multi-gpu: 2 cards unlike; ms per window: expert tier 7.9, layer split 13.5,
tensor parallel 9.0 (not built); picked expert tier` — the engine prices expert-tier vs layer-split vs
TP from bandwidth, VRAM and model size before placing anything (`gpus.md:34-40`).

**Warnings:** dropped Strata's route prediction, file-tier io prefetch and route-tail skip — if we
want those, the upstream Strata sources are the place to read, not Basalt. `FileExpertSource`'s own
doc comment is a warning: *"IT IS NOT AN LRU, AN LFU OR A PREFETCHER: every read goes through the
page cache"* (`include/basalt/core/expert_source.hpp:417`).

### 5. capy-flash — ninfer fork: CPU offload, elastic KV, the best policy paper (Apache-2.0)

**How it works.** `src/core/expert_cache_policy.{h,cpp}` (691 lines) is a pure-host replacement policy
with `Routed{expert, count}` per window (`expert_cache_policy.h:15-18`) — it deliberately reproduces
offline simulators rule-for-rule (`Rule::kEngine`, `kF0KSinglePass`, `kF0MWindowStartFree`,
`:22-36`).

*Policy knobs* (`expert_cache_policy.h:38-79`): per-layer/per-partition slot pools (expert `e` lives
in partition `e % partitions`), `admission` with `admit_threshold` and `half_life_tokens`
(decay `f ← f·2^(−tokens/hl)` once per window, then `f += count`), `insert_cap`, victim = LRU or
`f · 2^(−age/evict_tau_windows)` (`fr`), `recency_weighted_prompts` (a T-token prompt chunk adds
`hl·(1−2^(−T/hl))/(T·ln2)` of its counts so a 16k prefill doesn't dominate thousands of decode
tokens), and `elastic_columns` (extra slots that KV may lend/borrow — `--elastic`).

*Measured results* (`docs/capy-flash/development-log.md`):
- **T14**: per-layer slot tables, partition by ID parity, LRU + admission + insert cap; fills via
  `cudaMemcpyBatchAsync` reach 28.86 GB/s/link, 1.1 µs host time/expert; *"a fill lands one window
  later"* — simulated hit rate 0.894 → 0.871 at 181 slots.
- **T18/T22**: misses split between CPU and GPU (16 scratch slots over PCIe); GPU computes residents,
  CPU computes misses.
- **T19/T25**: `post_uni` predictor predicted +11-14%, measured **−2.7%** (PCIe copies took DRAM
  bandwidth: +13% µs per CPU expert) → **prefetch off and refused at TP2**; post-prompt warm start
  **+4.1% adopted**.
- **T24**: MTP K=3, acceptance 0.533, exact verify 20/20.
- **T54**: admission threshold 2 with a 256-token half-life → misses/token 204.5 → 182.6, throughput
  +3.6-4.2% in every cell, NLL unchanged.
- **T61**: Belady would give +12.6% (1-GPU) / +16.7% (TP2); ARC/2Q/TinyLFU/LFU/LIRS gave −9…+2%;
  adopted `fr` (τ = 4 windows): **+5.2% / +5.8%**.
- **T69**: refill hottest-first across layers after a prompt, recency-weight prompt counts → time lost
  in the first 10 windows 247 → ~96 ms.
- **T98**: mixing CPU W4A8 and GPU W4A4 expert math caused quality drift — heterogeneous *compute*
  paths must share one quantization contract.

*Staging for late-landing prefetch:* `src/core/expert_staging.h` — device slots outside the slot
table, tagged `(epoch, layer, expert)` with stream-ordered writes; `staging_overlay()` merges the
table row with matching staged slots **after the router**, so a window's view of the table never
changes mid-window and a staged expert is only ever used by the layer/window it was fetched for.

**Warnings:** no continuous batching (explicitly out of scope,
`docs/maintainer/engine-architecture.md:46`); `"Timing and numerics are coupled in the expert offload
backend"` (`docs/capy-flash/diagnostics.md:36`) — residency changes change rounding, so A/Bs need
`EXPERT_FREEZE=all`-style snapshotting.

### 6. colibri — pure C, disk-streaming 21k experts (Apache-2.0)

**How it works.** `c/glm.c` (~4,200 lines incl. comments). Dense part resident in RAM (9.9 GB int4);
21,504 routed experts (~19 MB each, ~370 GB total on disk) are streamed on demand.

*Tiers / policy.* Per-layer `ESlot **ecache` with an LRU clock (`glm.c:153`,
promotion loop `glm.c:2108-2115`: fill free slots first, else evict the lowest `used` stamp), plus a
**pinned hot-store** managed by `c/tier.h`:
- `tier_pick_swap` (heat-only) and `tier_pick_lfru` (score `(heat<<8) | recency`, recency capped at
  255 so "merely recent cannot displace genuinely hotter") both require the candidate to beat the
  victim by **25% + 4** — hysteresis against ping-pong (`c/tier.h:13-46`).
- `tier_decay` halves every heat counter (`c/tier.h:53-55`).
- Cache capacity is auto-sized from `MemAvailable` with an honest peak projection (`glm.c:3545`,
  `:3886-3930`) and re-clamped if RAM falls.
- Usage histogram persisted as `layer eid count` lines (`stats_dump_q`, `glm.c:3641-3655`) so startup
  AUTOPIN warms the right experts (`glm.c:3725-3740`), with optional live `REPIN` re-pinning between
  turns using the same 25%+4 hysteresis (`glm.c:2988-3051`).

*Prefetch.* `expert_prefetch()` issues `posix_fadvise(WILLNEED)` per tensor + `.qs` scale
(`glm.c:1500-1508`); `PILOT=1` predicts layer *l+1*'s top-K from layer *l*'s post-attention state
(recall **71.6%** vs 41.3% for the naive baseline), sends only the top `PILOT_K`, and does a **real
`pread` load into the future layer's LRU slot** on a dedicated I/O thread with a lock-free 1P/1C ring
that **drops when full** — inline submission blocked (~0.5 ms × 169k fadvise = +92 s/48 tokens)
(`glm.c:2162-2215`). `couple_prefetch` (`.coli_pairs` coupling table) is an alternative cross-layer
source (`glm.c:835-862`, commit `37ffb61`).

*Scheduling / batch-union.* Per MoE call, distinct experts of the whole batch are unioned first
(`glm.c:1869`), resident ones are submitted to the GPU **before** misses are read from disk so the
preads overlap the compute (`glm.c:1899-1903`).

**Warnings / anti-patterns:**
- Cold cost is brutal: **~11 GB of disk reads per token** before the cache warms (README "Honest
  numbers") — the disk tier only pays with a warm LRU + prediction.
- Speculation on a cold cache routes extra experts (660 → 1100 loads/token) and can be a **net time
  loss** (README, issue #100) — measure MTP and the expert cache **together**.
- Not byte-identical across kernel families/speculation (README, #100) — same class of problem as
  Strata's adaptation and capy's residency/rounding coupling.

### 7. MoE-Infinity — activation-aware RAM+SSD cache, research reference (Apache-2.0)

**How it works.** Python front-end (`moe_infinity/`) over a C++/CUDA engine (`core/`).
`ArcherPrefetchHandle` (`core/prefetch/archer_prefetch_handle.h`) owns offload/acquire/release/
prefetch/replace; `core/prefetch/{expert_residency,task_scheduler,expert_policy}` hold the policy.

*Tiers.* GPU cache sized by `device_memory_ratio` (0.9) vs `kv_cache_memory_ratio`
(`docs/configuration.md:36-40`); host pinned pool (`core/memory/pinned_memory_pool.*`); SSD store read
through `O_DIRECT` (fallback to buffered) in async 4 MB chunked reads with a priority AIO threadpool
(`core/aio/reader.h:140,283`, `archer_prio_aio_handle.cpp`, `MOE_IO_THREADS`,
`MOE_IO_BLOCK_SIZE_KB` default 1 MiB).

*Policy.* Phase-aware admission (`core/prefetch/expert_policy.h:19-31`): prefill admissions are
`TRANSIENT_ON_PRESSURE`, decode admissions are `CACHE`, eviction weights 1.0 (prefill) vs 4.0 (decode),
with a starvation limit — i.e. **prompt-driven traffic must not evict decode-critical experts** (the
same problem capy solved with `recency_weighted_prompts`). Replacement policies available as LRU and
full ARC (`moe_infinity/memory/offloading_policy.py:34,75`). Priority score =
topological decay × recent-trace matrix × global frequency
(`moe_infinity/memory/expert_priority_score.py:39-124`) — layers *ahead* of the current layer score
higher, which is exactly the anticipation a prefetcher needs.

*Prefetch trigger.* `speculative_prefetch` from router logits, optionally issued **before the dispatch
barrier** so PCIe copies overlap compute; `overlap_prefetch_policy = off|observe|enforce` budgets it
(`docs/configuration.md:70-95`):

```
T_window_ns(l) = max(0, safety_factor·compute_ewma_ns[l] − queue_wait_ewma_ns − issue_overhead_ns)
B_window(l)    = floor(bandwidth_ewma_bytes_per_ns × T_window_ns(l))
B_admit(l)     = clamp(B_window(l) − current_inflight_bytes, 0, max_window_bytes)
```

Whole-expert greedy packing over candidates sorted by `(-score, position, expert_id)`, cold-start
limited to `overlap_prefetch_cold_start_experts`, `max_inflight_bytes` 512 MiB, and **fail-closed**
on missing telemetry. `observe` computes metrics without changing transfers — ship that first.

**Warnings:** much of the config surface is documented as *reserved/experimental/legacy*
(`docs/configuration.md:44-80`); the synchronous `MoE.generate()` path is deprecated. Treat as a
source of **policies and formulas**, not of production code paths.

### 8. KTransformers — hot-GPU / cold-CPU NUMA placement (Apache-2.0)

**How it works (placement + offload scheduler only).**
- *Placement.* A **static GPU expert mask** with strategies `uniform | frequency | front-loading |
  random`, sized by `--kt-num-gpu-experts` (per layer) or `--kt-gpu-experts-ratio`, seeded by an
  activation-statistics file, optionally **redistributed dynamically when prefill length ≥
  `--kt-gpu-prefill-token-threshold`** (`doc/en/kt-kernel/experts-sched-Tutorial.md:96-140`).
  Measured on Qwen3-Next-80B, 4×4090, ShareGPT (same doc `:178-192`): at 10% GPU experts,
  random 56.6 → frequency 58.6 → dynamic **70.2** tok/s; at 10%, 70.2 vs 53.0 baseline.
- *CPU lane.* `KExpertsCPU` / `KExpertsMarlin` / `KExpertsTorch` behind `KExpertsBase`
  (`archive/ktransformers/operators/experts.py:68-642`); the CPU backend is a worker pool fed by a
  task queue between GPU callbacks and CPU compute (`kt-kernel/cpu_backend/cpuinfer.h`,
  `task_queue.*`, `worker_pool.*`).
- *NUMA.* Threads bind once at start: `numa_node = thread_id * numa_num_configured_nodes() /
  thread_num_` then `numa_bind(mask)` (`archive/csrc/ktransformers_ext/cpu_backend/backend.cpp:89-97`);
  `kt-kernel/cpu_backend/worker_pool.cpp` builds per-node subpools with hwloc and per-NUMA thread
  counters (`make_numa_thread_counters`, `:35-46`).

**Warnings:** no disk tier, no cache policy (placement is a mask, not a cache); the dynamic-update
logic lives in the sglang-kt fork, not in this clone — the tutorial is the citation, not source.
Requires 256 GB+ RAM for the documented configurations.

---

## Synthesis for breccia M3

### Our constraints (what the reference numbers imply)

- Model: 512 experts × 48 layers = **24,576 routed experts**, 75 GB IQ3_XXS ⇒ **≈3.1 MB per blob**.
- Tier budget: **~30 GB usable host RAM ⇒ ≈9,600 blobs ≈ 39% of the set**; VRAM tier (after dense
  weights, KV, scratch on the main card) realistically **2,000-3,500 blobs ≈ 8-14%**.
- Consequence #1: Strata's io-prefetch gate only enables when the page cache covers **≥55%** of the
  expert set (`docs/DETAILS.md:271`). At 39% we are **below** its own "don't bother" line, so
  page-cache-only streaming must not be our tier-2 default — we need an explicit, budgeted host tier.
- Consequence #2: colibri's cold path (11 GB reads/token) and capy's T14 ("a fill lands one window
  later") both say the disk/host tiers only win if the **policy is right and the fill is a window
  early**. M3's success metric should be steady-state miss rate and bytes/read latency, not raw hit
  rate.
- Consequence #3 (heterogeneous AMD): capy T98 (mixed CPU/GPU expert math drifting quality), Strata's
  HIP-specific caveats (SDMA vs kernel copies `generate.cpp:267-277`, prefetch-induced GPU resets on
  iGPU), and the MI50 (gfx906, Wave64) vs RX 6950 XT (gfx1030, Wave32) split mean **per-device tier
  policy and copy paths, not one shared assumption**.

### Recommended design

**1. Data structures (from ninfer + capy-flash).**
- One slot table per layer per device: `expert → slot` offset indirection with an absent sentinel, so
  GPU addressing is `pool_base + offsets[expert]` and slot contents can change without touching
  anything captured (ninfer `expert_cache.h`, "cached" addressing).
- Host-side policy as a **pure, testable module** with an explicit window input
  `Routed{expert, count}` (capy `expert_cache_policy.h`) plus a **replay harness** — copy Strata's
  `tools/replay_cache_policy.py` idea: record a routing trace, replay candidate policies offline, and
  only then run the engine. This is the single highest-leverage item; capy and Strata both credit it.
- Packed residency word (state/generation/frame) if we ever need a device-visible table (Infernix
  `expert_cache.h:21-34`); start host-side-only and only pay for it in M4/M5.

**2. Tiers.**
- **Tier 1 — VRAM:** per-GPU slot pools. Partition experts across the two devices by *hotness to the
  least-loaded tier* (Basalt `gpus.md:22-27`) rather than blind `id % 2` (capy's parity split), and
  enforce **disjoint ownership** so GPU0 and GPU1 never both hold a blob (Strata
  `DISJOINT_EXPERT_CACHE.md` — measured +15.5% when fixed).
- **Tier 2 — pinned host, 30 GB budget:** an arena of *chunks* pinned incrementally with a reserved
  unpinned fallback (Infernix `pin_reserve` / `ResidentHostBuffer`), a RAM floor auto-derived from
  `MemAvailable` (colibri `glm.c:3886-3930`), and `mincore()`-style checks before assuming a pin
  exists. Budget = hottest ~9,600 blobs by the profile + learned heat.
- **Tier 3 — NVMe:** keep the mmap page-cache path *as a fallback* (Basalt's `FileExpertSource` doc
  comment tells us it has no policy of its own), plus an **explicit read path**: a small O_DIRECT/
  `pread` ring (Infernix `Options.ring = 128`, colibri's drop-on-full I/O thread) for blobs we can
  name a window ahead. Never let the page cache be the only eviction mechanism below 55% coverage.

**3. Cache policy (capy-flash is the authority).**
- Admission: decayed frequency ≥ threshold, half-life ~256 tokens, insert cap per layer per window —
  measured −11% misses/token, +3.6-4.2% throughput (T54).
- Victim: `f · 2^(−age/τ)`, τ ≈ 4 windows (`fr`), LRU fallback — measured +5.2% vs previous rule and
  only ~5 pts below the Belady bound of +12.6% (T61). Do **not** start from ARC/2Q/TinyLFU; capy
  measured them at −9…+2%.
- Hysteresis on every swap: either capy's admission threshold or Strata's `min-gain` 1.5×, plus
  colibri's 25%+4 rule for the pinned hot-store. Ping-pong is the failure mode at our 39%/14% coverage.
- Weight prompt-window counts by recency (capy `recency_weighted_prompts`, T69) so a 16k prefill
  doesn't evict decode's working set — and/or phase-split admission like MoE-Infinity (prefill
  transient, decode cached).
- Decay every window or every few windows (`adapt_every 1..4`, decay 0.7); persist the learned ranking
  to a profile file on exit (Strata `expert-profile-save`, colibri `.coli_usage`).

**4. Prefetch (budget first, prediction second).**
- Prediction source: the **next layer's router applied to the current layer's MoE input**, top-k union
  across tokens, depth 1 (Strata `RouterLookahead`; capy T19 `post_uni`; Infernix §8.7; colibri
  `PILOT` at 71.6% recall; MoE-Infinity's topology-preferring score). Implement it as a host thread
  off the recorded router weights; cap to the confident head (`PILOT_K`, `warm_scored` confidences).
- Budget: adopt MoE-Infinity's equation — `B_window = bandwidth_ewma × max(0, 0.8·compute_ewma −
  queue_wait − issue_overhead)`, whole-blob packing, capped inflight bytes, **fail-closed** on missing
  telemetry, shipped `observe` before `enforce`.
- Gate: Infernix's **useful-byte-ratio ≥ 0.3 per layer** and Strata's periodic usefulness/coverage
  gate. If a prediction source doesn't pay for two evaluation windows, turn it off (that is exactly
  what capy did at −2.7%).
- Promote without copying when the frame/slot is already staged (Infernix §8.7 step 4), and let a
  still-in-flight copy be usable in the *next* window only — capy's `expert_staging.h` epoch/layer tag
  is the clean pattern for "a fill lands one window later".

**5. Threshold-based skipping (defer to M4, design for now).**
- Keep `route_tail_skip_rank` as a config knob from day one: skip *copying and computing* a missed
  expert when every token of the window routes it at rank ≥ R (Strata default 7), with escape hatches
  (disable when ≥90% cached, on batch slots, on peer tiers) and counters in `--stats`
  (`tail_skipped`/`tail_experts`). Answers will differ slightly from a non-skipping run — gate
  comparisons must fix the setting.

**6. Scheduling / batching.**
- M3 needs only **per-window batch-union routing** (colibri `glm.c:1869`; Infernix §8.8 "each
  distinct expert fetched or computed once for all its columns") and per-request lane accounting.
  Continuous batching is *not* a solved, copyable piece: Strata/Basalt use fixed slots with a queue,
  capy declares it out of scope, only Infernix's `DecodeShare` and MoE-Infinity's async server do it.
  If concurrency matters later, take Infernix's rule: only whole prefill units interleave so chunk
  grids and output bits are unchanged.

**7. Measurement gates (non-negotiable for this repo).**
- Report `hit_rate` (VRAM), `ram_blobs`, `file_blobs/file_mb`, prefetch `useful ratio` and
  `tail_skipped` — Strata's metric names are a good schema (`docs/DETAILS.md:288-301`).
- Freeze residency for numerics A/Bs (capy `EXPERT_FREEZE=all`) or fix adaptation to a single
  deterministic schedule, since adaptation changes rounding (Strata, capy `diagnostics.md:36`).
- Cold vs warm must both be reported; pinning, prefetch and adaptation each have a documented
  "made it worse" case in this set.

### What to take, and from whom (short form)

| Piece | Take from |
|---|---|
| Slot tables, offset indirection, batched uploads | ninfer |
| Policy shape: admission threshold + freq×recency victim + prompt recency weighting + replay tool | capy-flash |
| Swap hysteresis, learned profile file, tier metrics, tail-skip, prefetch gate | Strata |
| Pinned-chunk arena, unpinned fallback, read ring, useful-ratio kill switch, residency word | Infernix |
| Per-layer LRU + hot-store hysteresis + MemAvailable auto-sizing + drop-on-full I/O thread | colibri |
| Prefetch byte budget equation, phase-aware admission, `observe→enforce` rollout | MoE-Infinity |
| Two-GPU placement cost model + disjoint ownership + hottest-to-least-loaded distribution | Basalt |
| Frequency-ranked placement value, NUMA-pinned CPU lane | KTransformers |
