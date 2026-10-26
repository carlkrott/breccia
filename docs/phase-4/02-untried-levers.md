# 02 — Levers we have not tried, ranked by expected effect on prefill and decode

Phase 4 / M3 scoping. Date: 2026-10-26. Read-only survey of `Reference Projects/` plus the model
artifacts in `/srv/strata-data/mtp/`. **No code, no benchmark run** — every number below is either
cited from an existing Breccia measurement/doc or is an explicitly labelled projection with its
formula. Owner of the resulting work: `crates/breccia-engine/**`, `crates/breccia-kernels/**`.

Baseline this document is scored against (all from `docs/phase-4/01-forward-pass.md`):

| | value | where |
|---|---|---|
| decode, 2 streams | **9.29 tok/s/stream, 20.28/19.98 aggregate**, 108 ms/step, lane 39 ms, fence 48 ms | §8p/§8q |
| decode, 4 streams | **24.74 aggregate**, 162 ms/step, lane 98 ms, fence 14 ms | §8p |
| decode, 6 streams | 25.03 aggregate, 240 ms (the plateau) | §8p |
| prefill, 4 agents × 300 | **3.14 tok/s/agent** (95.6 s for 1 200 tokens) | §8r/§8s |
| expert bytes/token | **799 MiB** = 48 × 10 × 1.665 MiB (40 925 MiB / 24 576 experts) | §2 |
| host lane rate | 33 GB/s at 32 threads (24.5 ms/token) | §8p, §8s |
| host single-core DRAM stream | **53.9 GB/s** — the lane is at 61 % of one core | `docs/phase-3/08` §4 |
| RX PCIe | **28.7 GB/s, 100 % overlap** | AGENTS.md (Gate A) |
| MI50 PCIe | ~2.9 GB/s (gen3×4) — no streaming lane | Gate A |

Two things follow from that table before any technique is discussed, and they set the rank order:

* **Prefill is not a kernel problem, it is a batching problem.** 1 200 prompt tokens read
  1 200 × 799 MiB = 937 GiB through the lane. The same 1 200 tokens, taken as one chunk, touch
  essentially every expert **once**: 48 × 512 × 1.665 MiB = **40.9 GiB**, i.e. 34 MiB/token — a
  **23×** byte reduction that needs no new numerics at all.
* **Decode is a bytes problem, not an ALU problem.** The lane at 32 threads moves 33 GB/s against a
  53.9 GB/s single-core stream, so there is ~1.6× of headroom in the lane itself, and the only way
  to remove bytes entirely is residency (technique 2/3) or a second lane (technique 2/4).

---

## 1. Ranked table

Rank = expected effect on the two headline numbers, divided by effort. Column "P" is prefill
(tok/s/agent), "D" is decode (aggregate). "est." = projection, formula given in the section.

| # | technique | source (file) | P | D | effort | risk |
|---|---|---|---|---|---|---|
| 1 | **Merged chunked prefill, chunk's experts streamed to the RX over PCIe and computed by the existing device MoE** | capy-flash `development-log.md` T34; Basalt `src/prefill/prefill.cpp` | **×6–14** (3.1 → 20–45) est. | 0 (enables 6) | L | med: VRAM ring, grouped MoE at m≈20, per-agent causal masks |
| 2 | **Per-expert frequency-ranked residency** instead of 9 whole blocks (`-ot`) | Basalt `core/expert_cache.hpp:3-40`; Strata `core/expert_cache.cpp:74` | ×1.1–1.15 | **+20–35 % at 4 streams, +12–25 % at 2** est. | M | med: needs the routing histogram first; capy T61 warns profile gains can be a prompt-repetition artifact |
| 3 | **Miss-split: copy a share of the misses to VRAM and compute them on the GPU** (second lane at T=1) | capy `T18` (+9.1 % auto / **+13.6 %** fixed 0.3); Strata `STRATA_PCIE_BALANCE` `expert_source.cpp:4877` | +10 % (same ring as 1) | **+13–21 %** at 4 streams | M | low-med: correct by construction (same kernels, same bits) |
| 4 | **Lane to the memory roof**: activations quantised to int8 + `vpdpbusd`/`vpmultishiftqb`, huge pages | Basalt `kernels/cpu/expert.hpp:1-25`; Strata `expert_source.cpp:5015` (`act_quant_q8_1`) | ×1.2–1.3 | **+25–30 %** (33 → 45–50 GB/s) | M | **high for parity**: changes numerics (see §4) |
| 5 | **Prefix/state snapshot reuse across agents** (radix cache) | capy `resource-scheduling-and-context-cache.md`; vLLM `v1/core/kv_cache_manager.py` | **×2–3 on agents 2..N** for a shared system prompt | 0 | M | low |
| 6 | **MTP K=3 + draft-vocabulary shortlist** | capy `T24`, `T43`, `T52`; Strata `spec/draft_policy.cpp`; Basalt `core/mtp.hpp` | 0 | **×1.2–1.4**, only after 2/3/4 | L | med-high: today it is a **loss** (§6, arithmetic) |
| 7 | **Fuse the per-block device MoE into one dispatch** (row-map `MUL_MAT_ID`), 2 ms → ~1 ms/block | llama.cpp `mul_mat_vec_id_*` (`vulkan-shaders-gen.cpp:811`); Basalt `prefill/moe_fused.cu` | +5–10 % | +5–10 % (unblocks 2) | M | low |
| 8 | **Route-tail skip** (drop the rank≥R missed expert) | Strata `STRATA_ROUTE_TAIL_SKIP`, `expert_source.cpp:4763-4790`, skip logic `:4884-4905` | +5 % | +4–5 % | S | **breaks parity** (changes output by design) |
| 9 | **Wave32 on gfx1030** via `VK_EXT_subgroup_size_control` | llama.cpp `RDNA_DEFAULT_SUBGROUP_SIZE = 32` (`ggml-vulkan-types.h:1388`) | 0–3 % | 0–5 % (GDN delta rule only) | S | low; reduction order changes |
| 10 | **int8 dot product (dp4a) for the chunk dense GEMM** | llama.cpp `mul_mmq.comp:7`, `vulkan-shaders-gen.cpp:628` | +5–10 % after (1) | 0 | L | med: no WMMA on this HW, dp4a only |
| 11 | **Interleave prefill chunks with other agents' decode** | vLLM `v1/core/sched/scheduler.py:1311`; SGLang | 0 | 0 (p95 only) | S–M | low |
| 12 | **KV quantisation** (q8/q4 tail) | ik_llama.cpp `-ctk-first/-last` (`common/common.cpp:2053`, `:3539`) | 0 | 0 at 300 tok; matters >2 k | M | low (Phase 6 already owns it) |
| 13 | **Tensor-parallel prefill/verify across both cards** | Basalt `configuration/gpus.md` (priced, **not built**); capy `T31/T33/T62` | negative | negative | L | — |
| 14 | **Persistent / whole-layer mega-kernel** | Basalt `docs/unified-kernel.md` (measured **slower**) | negative | negative | L | — |
| 15 | **Descriptor buffers / push descriptors / shader objects** | — | 0 | **0.3 %** (0.32 of 108 ms) | S | non-starter |

Not in the table because they are already in flight: dequant-once-dot-many (the correct half of the
prefill fix) and AVX-512 on the host lane. §4 says how they should be built; it does not re-propose them.

---

## 2. Merged chunked prefill + PCIe expert streaming (rank 1)

**Who does it.** capy-flash `docs/capy-flash/development-log.md` **T34** *streaming prefill*: "non-resident
experts are copied over PCIe into a ring of slots borrowed from the decode cache, and the next layer is
copied while this one computes. Chunks wider than 2,047 tokens run no experts on the CPU." Measured:
4 k-token prefill **413 → 1 774 tok/s**, 16 k **371 → 3 222 tok/s** (4.3× / 8.7×). Basalt's
`src/prefill/prefill.cpp` is the same design with a per-chunk plan (`experts_streamed`, `experts_dma`,
`cpu_share`) and a peer variant (`kPeerSlowRows = 2559`, `prefill.cpp:738`).

**Mechanics.** For a chunk of T tokens and one block: route all T tokens (T×10 pairs), bucket by
expert, read each distinct expert's blob **once** and run it against the m rows that chose it. Bytes per
chunk = `distinct(T) × 1.665 MiB × 48`; distinct saturates at 512 for T ≳ 256 (occupancy 512·(1−(1−10/512)^T)).

| T (one agent) | distinct/block | bytes/token | vs 799 MiB |
|---:|---:|---:|---:|
| 1 | 10 | 799 MiB | 1.00 |
| 32 | 232 | 302 MiB | 0.38 |
| 256 | 508 | 162 MiB | **0.20** |
| 300 × 4 agents in one chunk | 512 | **34 MiB** | **0.043** |

**Why our §8s attempt failed and this one does not.** §8s grouped the *reads* but kept `gemv_into` per
user, so the dequantisation was still paid per (expert, user) — and the lane is dequant-bound, so
grouping removed bytes the lane was never bound by (12–14 % slower). Two fixes are available and they
are *not* the same work:

* CPU side (in progress): dequantise once into registers, dot all m users — bit-identical, Strata proves
  it (`grouped_chunk_lane_matches_the_per_token_lane`, 1.5e-8).
* **GPU side (this item, untried here):** the chunk's blobs stream to the RX and are computed by
  `gemv_*_moe` + `MoeMap` at m ≈ 5 (one agent) to 23 (four agents) rows per expert. The dequantisation
  is amortised by the m rows *in the shader*, which is what the existing `_moe` shaders already do for
  10 experts × 1 row — they need only the m dimension.

**Numbers.** Byte floor for 4 × 300 merged: 40.9 GiB over 28.7 GB/s = **1.50 s**; today the same work
takes 95.6 s. The floor is 64× and it is a *floor*, so the honest target is the §8r-style projection
scaled by the second lane: 3.14 → **20 tok/s/agent conservative, 40–45 optimistic**, aggregate
prefill 12.5 → 80–170. The binding constraint after (1) is the dense chain, which is why §8r's
"grouping + T-batched dense chain → ~52 tok/s" is the right order of magnitude for the *single-agent*
case; merging four agents in the chunk amortises the expert bytes another 4× on top.

**Vulkan sketch.**
1. Host: `vkCmdCopyBuffer2` **with a region array** — one command per block, 508 `VkBufferCopy2`
   regions (src = the mmap'd GGUF shard offset, dst = ring slot). No gather kernel needed; the DMA
   engine does it. 48 commands per chunk. Pre-record 2 command buffers and ping-pong them on the
   transfer queue with the timeline semaphore the ring already uses (`crates/breccia-vk`).
2. Ring: 32 slots × 1.665 MiB = 53 MiB per card (measured arena is 0.9 GiB/block for 512 experts, so a
   slot is the blob only). Borrow from the `--gpu-experts` arena so this costs no new VRAM.
3. Device: build `expert_offsets[513]` + `row_expert[]` with a counting sort (two `atomicAdd` passes —
   Basalt `prefill/moe_fused.cu`, llama.cpp's `MUL_MAT_ID` row map). Dispatch one workgroup per
   (expert, row tile), indexing by binary search into `expert_offsets` — no host round trip, no
   per-expert dispatch. Values stay bit-identical to the per-token path because each row's dot is the
   same dot in the same order (this is what already gives 64/64 parity at 9 resident blocks).
4. Sync: copy(block b+1) runs on the transfer queue while compute(block b) runs on the compute queue,
   joined by timeline semaphores — the exact pattern §8p's in-batch reads already use.

**CUDA-only parts.** Basalt uses cuBLAS tensor-core GEMMs for the projections (`prefill/gemm.cu`) and
`mma.cuh` for MMQ. **Neither is available to us** (`VK_KHR_cooperative_matrix` missing): our equivalent
is fp16 `vec2`/`vec4` FMA (2× fp32 rate on gfx1030) or dp4a (§10). Do not port their kernel; port their
*schedule*.

**Measure.** `breccia bench agents --streams 4 --prefix 300 --prefill-chunk 256 --prefill-lane pcie`
(with the flag added). Number to move: prompt seconds 95.6 → < 30. Parity gate: the existing
`--same-prompt` 24/24 must stay bit-identical at chunk 1 vs chunk 256 (`docs/phase-4/01` §8s already
proved chunking is parity-neutral once the per-token inject slots are right).

---

## 3. Per-expert frequency-ranked residency (rank 2)

**Who does it.** Basalt `include/basalt/core/expert_cache.hpp:3-40` is a design document for exactly our
problem: *"the CPU expert pool is the engine's largest single cost… the only way past it is to stop
reading those bytes: put some of the 48 × 512 experts in VRAM and let the GPU compute them"*, and
*"only a small share of (layer, token) pairs have all ten experts resident — so this cannot be a
per-layer grouped kernel and the three-way split is not an optimisation"*. The profile is
`(layer, expert)` pairs ranked by routing frequency + a `heat` table `--expert-profile-save` writes
(`rank_learned_profile`). Strata has the same file format (`core/expert_cache.cpp:74` `read_expert_profile`,
`STRP` header) and the same idea. KTransformers' `--num-gpu-experts` is the coarser "first N of every
layer" variant (`ktransformers.py`); it is *positional*, not frequency-ranked, and is the weaker form.

**Mechanics.** We place **9 whole blocks** (`-ot 0,1,2,4,8,18,20,22,23`) = 4 608 slots = 18.75 % of the
24 576 experts, chosen by *block*, i.e. by luck. Per-expert placement spends the same VRAM on the
hottest `(layer, expert)` pairs instead: at 8.1 GiB/card that is **4 980 slots/card**, and with two
cards at 2 streams **~9 900 of 24 576 = 40 %**. If routing is even mildly skewed (top 40 % of pairs
carrying 60 % of slots), lane bytes go 799 → ~320 MiB/token.

**Numbers (estimate; the input is missing, see below).** 2 streams: lane 39 → 26 ms, but device MoE
grows from 9 to ~29 block-equivalents × 2 ms = +40 ms of GPU work that the fence (48 ms, GPU at
44/64 % util) can only partly absorb → net step 108 → **92–98 ms**, i.e. **+10–17 %** (aggregate 20.3 →
22.5–23.7). 4 streams: lane 98 → 65 ms, fence 14 → 23 ms → step 162 → **130–140 ms**, i.e. **+16–25 %**
(24.7 → 29–31). The 4-stream case is the better one because the lane is 60 % of that step.

**This is gated on one measurement we do not have**: the routing histogram over `(layer, expert)` for
real agent prompts. It is 20 lines (`--profile-routes` on `bench agents`) and it decides the whole item.
capy-flash **T61** is the caution: a per-layer slot profile *looked* like +13.4 % at TP2 but was
"+0.2 % (CI −9…+10) on non-repeated requests" — i.e. it was an artifact of repeated prompts. Our
`bench agents` mixes LCG prefixes (good) but `--same-prompt` must not be used for this A/B.

**Vulkan sketch.** `MoeMap` already builds a device-side row map from expert ids; the change is that the
map's value becomes "device slot" *or* "host blob" *per row* rather than per block, so the same dispatch
covers resident rows while `ExpertPool` computes the misses. Two buffers: `res_slot[l*512+e]` (int32,
-1 = miss) and a slot arena indexed by rank. Nothing in the kernel changes; the row map is already
gathered on device (`docs/phase-4/01` §8d).

**Risk.** Device MoE is **2.6× more expensive per block than the host lane** (2 ms vs 0.76 ms, §8d) —
so residency pays only while the GPU has slack. That is precisely why item 7 (fused single-dispatch
MoE) is the multiplier: at ~1 ms/block the crossover moves and more residency becomes free.

**Measure.** `breccia bench agents --streams 4 --prefix 300 --expert-profile <file>` with
`--gpu-resident-per-expert`; target: lane ms/step 98 → 75, aggregate 24.7 → 28+. Also run
`breccia bench token --streams 2` with parity 64/64.

---

## 4. The host lane to the memory roof (rank 4, and the form AVX-512 should take)

**Who does it.** Basalt `include/basalt/kernels/cpu/expert.hpp:1-25` — same model, same shapes
(`H=2560, FF=640, NE=512, QK=64, BLOB=1 382 400` B), and it states the recipe in one paragraph:
`vpdpbusd` (AVX512-VNNI) *"does four int8 MACs per instruction, which keeps the kernel memory-bound
instead of compute-bound"*; `vpmultishiftqb` (AVX512-VBMI) unpacks 2-bit codes 8 at a time; the
accumulator stays f32 and is reduced **once per row**. Strata does the same from the other side:
`expert_source.cpp:5015` quantises the whole activation row with `act_quant_q8_1(x_f + t*H, H, …)`
before the CPU experts run, so the products are int8×int8.

**Mechanics and why it is not the same as "AVX-512".** AVX-512 f32 FMA is 16 f32 MACs/instruction;
`vpdpbusd zmm` is **64 int8 MACs/instruction** — 4× the MAC rate *and* 4× fewer activation bytes
(1 B vs 4 B/element). For our inner loop the dequantisation is the cost, and VBMI's
`vpmultishiftqb` removes 8 code-unpacks per instruction, which is the other half.

**Numbers.** Our lane: 33 GB/s at 32 threads, against a **53.9 GB/s single-core stream** on this host —
so the lane is at 61 % of what one core can pull. Two independent engines on the same model class report
**54 GB/s** (capy T16, per-expert CPU path) and **63–65 GB/s "at the bandwidth roof"** (capy T35
microbenchmark). Taking a conservative 45–50 GB/s at 32 threads: lane 24.5 ms/token → 16–18 ms, i.e.
2 streams 39 → 26–29 ms/step (**+10–13 %** aggregate) and 4 streams 98 → 65–72 ms (**+15–22 %**).

**Two sub-experiments, one cheap.** (a) `madvise(MADV_HUGEPAGE)` + `mlock` on the *hot* expert subset:
capy's arena is one page-locked allocation; ours is page-cache mmap, and 1.665 MiB random blobs are
exactly the TLB-hostile pattern. This is a 10-line experiment with no numerics change. (b) The int8
form, which **changes the numerics** — see the risk line.

**Risk: it breaks our parity gate.** Our gate is bit-identical logits vs the production HIP server.
int8 activations put ~1e-3 relative error into every expert row; Basalt states plainly that "an expert
on the CPU tier and the same expert on a GPU differ at 8e-8, which flips a near-tie token between
unpinned runs", and capy measures CPU-W4A8 vs GPU-W4A4 at 0.13–0.17 relative L2 (§T16). So the int8
lane must be qualified as **"same top-1 on the parity set + KL/NLL within bound"**, not as bit-identical,
or restricted to the draft/verify rows. The f32 AVX-512 form keeps bit-exactness because the per-row
accumulation order is unchanged — that is the version to land first if the gate must hold.

**Numerics trap worth copying verbatim.** One Q2_0 weight block is 64 weights with **one** fp16 scale,
while one activation chunk is **32** elements with its own scale, so a weight block spans *two*
activation chunks with two different scales. "One scale per weight block" is the natural-looking
mistake (`expert.hpp:16`).

**Measure.** `breccia bench cpu-experts --threads 32` (33 GB/s today); then
`breccia bench agents --streams 4 --prefix 300`. Number: 33 → 45+ GB/s, lane 98 → 72 ms.

---

## 5. Miss-split over PCIe at T=1 (rank 3)

**Who does it.** capy **T18**: *"misses split between CPU and GPU: the GPU copies part of the misses into
16 scratch slots over PCIe and computes them. 1 GPU, MTP: the window drops from 58.7 to 52.6 ms with
`--miss-split auto` (+9.1 %) and to 51.5 ms with a fixed 0.3 share (+13.6 %)."* Off at TP2 (T42).
Strata has the same mechanism as `STRATA_PCIE_BALANCE` (`expert_source.cpp:4877`, "the count comes from
the measured CPU and GPU costs instead of the share") with a minimum-of-one rule (`STRATA_PCIE_MIN1`).

**Mechanics.** At T=1 there is no amortisation; the only gain is a *second lane*. Pick the s cheapest
misses of the block, copy their blobs into scratch VRAM slots inside the block's own batch, and let them
be computed by the same device MoE path; the CPU computes the rest. Aggregate bandwidth becomes
33 + 28.7 = 61.7 GB/s, and the two lanes end together at share ≈ 0.47 (measured best 0.3 on capy's box).

**Numbers.** At 4 streams the lane is 98 of 162 ms. Moving 35 % of the bytes (34 ms) to a PCIe lane that
overlaps: lane 64 ms, fence 14 → 20 ms, step 162 → 128 ms → **30.5 tok/s (+23 %)**. At 2 streams the
fence is the constraint (48 of 108) so the gain is smaller: 39 → 26 ms lane but +7 ms of GPU → step ~99 ms
(**+9 %**). capy measured +9.1/+13.6 % on a machine whose CPU lane was already 54 GB/s (so *their* lane
share was smaller) — our numbers should be better, not worse.

**Vulkan sketch.** We already have every part: a staging ring + timeline semaphores (`breccia-vk`),
`vkCmdCopyBuffer2` for the blob, and `PlacedExperts` for the destination. New: a small `int scratch_slot`
allocator (16 slots × 1.665 MiB = 27 MiB) and a per-block policy that picks the misses by *ascending
weight* (cheapest first, like Basalt's `std::sort(cand)` in `prefill.cpp:2099`) so the CPU keeps the
expensive ones. Bit-identical: same kernels, same order, results written to the same rows.

**Do NOT use prefetch for this.** capy **T25** built the L+1 predictor and measured **−2.7 %, rejected**:
"PCIe copies take DRAM bandwidth from the CPU (+13 % µs per CPU expert)". colibri's PILOT
(`colibri/README.md:43`: L+1 router on L's post-attention state recalls **71.6 %** of the true top-8) and
capy's `post_uni` (top-10 overlap **0.69**, precision on non-resident 0.38 vs break-even 0.24) are both
*accurate enough*; the problem is that speculative bytes steal the shared host read bandwidth. Use the
predictor to choose *which* misses go over PCIe, not to warm a cache.

**Measure.** `--pcie-miss-share 0.3` on `bench agents --streams 4 --prefix 300`; target aggregate
24.7 → 28.5+; parity must stay 24/24.

---

## 6. Speculative decoding: MTP and the suffix drafter (rank 6)

**Who does it.** Basalt `core/mtp.hpp` + `core/verify.cpp` + `src/spec/draft_policy.cpp`; Strata
`core/mtp.cpp` (1 400+ lines, CUDA-graph captured) + `src/spec/draft_policy.cpp`; capy
`development-log.md` T24/T41/T43/T52; llama.cpp `common/speculative.cpp` + `common/ngram-map.cpp`
(n-gram/suffix drafting in the stock binary).

**What the artifacts cost us.** `/srv/strata-data/mtp/mtp-q2_0.gguf` is **889 MB total** — dense
proj + `q2_0` experts (236 MB) + `draft_vocab.bin` in `rt/`. The MTP block is *one* transformer layer,
512 experts top-10, `FF=640`, plus a hyper-connection mixer and its own QSA/indexer. Basalt keeps
**all 512 MTP experts in VRAM (708 MB)** and runs them "through the grouped hit kernels"
(`core/mtp.hpp:13-17`) — so the drafter is a **device-resident** layer and adds **no host lane traffic**.
That is the whole ballgame: the Phase-5 trap in `docs/plan/BRECCIA-PLAN.md` ("naive MTP collapsed decode
10–12 → 1.6 tok/s by doubling expert reads") is avoided by construction if the drafter lives in VRAM.

**Measured acceptance on this exact model** (IQ3_XXS/IQ3_S, prose + natural prompts):

| source | K | exact accept | tokens/window | window cost (single steps) | net |
|---|---|---|---|---|---|
| Basalt `docs/precision/features.md:217` | 7–8 | **61.1–64.5 %** | — | — | — |
| capy T24 | 3 | **0.533** | **2.59** (chat 2.99, pg19 2.16) | — | — |
| capy T52 | 3 | — | — | **2.10** (Strata 1.86) | **1.23×** (Strata 1.39×) |

**The arithmetic for us, which is why this ranks 6th and not 1st.** A verify window of one anchor + K
drafts costs, on our engine, `1 + K·0.95` lane-units today (each draft row routes to its own 10 experts;
distinct experts for 4 rows ≈ 38 of 40, so there is **no** byte amortisation). With p = 0.6 the expected
yield is 2.31 tokens at K=3 → 2.31/3.85 = **0.60×: a 40 % loss.** That is exactly the collapse the plan
note records. It becomes a win only when the marginal row is cheap:

| marginal row cost | window cost (K=3) | yield | net |
|---|---|---|---|
| 0.95 (today) | 3.85 | 2.31 | **0.60×** |
| 0.40 (after dequant-once-dot-many) | 2.20 | 2.31 | **1.05×** |
| 0.25 (after 2+3+7: rows are device-resident) | 1.75 | 2.31 | **1.32×** |
| 0.25 + K=5 | 2.25 | 2.62 | **1.16×** |

So MTP is a **multiplier on items 2/3/7, not a standalone lever**. Two things are worth building now
because they are cheap and independent: (a) the **suffix/ngram drafter** (Basalt
`include/basalt/spec/suffix_drafter.hpp`: trigram hash, 4 ways, ~20 B/history token, *microseconds*, no
weights, no GPU) which for agent workloads that quote their own input is the highest-yield-per-cost
drafter there is, and (b) the **draft-vocabulary shortlist** (capy T43: score 108 861 of 248 320 tokens
— the 32 768 most frequent + 33 specials from `rt/draft_vocab.bin` — draft cost 5.94 → **3.58 ms**,
**+7.6 %** decode). Both are numerics-neutral (verification is exact, drafts only change *which* rows
are proposed).

**Strata's controller is the part worth copying, not its kernels.** `src/spec/draft_policy.cpp:14-20`
prices window widths from *measured* round times with a shape table `{1, 1.35, 1.7, 2.05, 2.45, 2.85,
3.25, 3.6}` for T = 0..8, and picks the width that maximises `yield/cost` — an EMA-calibrated
`lookup_rate × cost_ms` decision (`draft_policy.cpp:63-100`). Our engine needs the same 30 lines before
any MTP result means anything.

**CUDA-only.** Strata/Basalt/capy capture the drafter in **CUDA graphs** (`mtp.cpp:755` `kMpGraphs = 17`,
`capture_round`, `capture_step`) and capy uses PDL + `cuStreamWriteValue32` handshakes (T16). Vulkan has
neither; our equivalent is the pre-recorded `VkCommandBuffer` + fence we already use, which is why our
recording cost is only 0.32 ms. Expect the *draft* step to cost us more than it costs them (~3.6 ms of
theirs is graph-replayed).

**Measure.** `bench token --gen 64 --mtp mtp-q2_0.gguf --spec 3` plus a `--mtp-off` arm; numbers:
acceptance per drafted token, tokens/window, and ms/window vs 108 ms (2 streams). Gate: 64/64 parity
with MTP on (verification is exact, so it must hold).

---

## 7. Fuse the device MoE block into one dispatch (rank 7)

**Why it matters.** Device MoE costs **~2 ms/block** against a **0.7 ms** byte floor at 1280 GB/s
(§8d/§8p) — the overhead is dispatch and small-kernel latency, not bandwidth. That 2 ms is what caps
technique 2 and what made `--gpu-experts` lose single-stream. llama.cpp does the whole expert GEMM in
one dispatch keyed by a row map (`mul_mat_vec_id_*` shaders, `vulkan-shaders-gen.cpp:811`;
`mul_mmq.comp` `MUL_MAT_ID` with `RowMap`), Basalt in `prefill/moe_fused.cu`; we issue MoeMap →
per-expert GEMV → SwiGLU → down → combine.

**Sketch.** One `vkCmdDispatch` with `ceil(rows/tile)` workgroups; each workgroup binary-searches
`expert_offsets[]` (built by two `atomicAdd` passes) to find its expert and tile, then reads the blob
from the slot arena. Combine stays a second dispatch (it needs a cross-expert reduction + the shared
expert).

**Expectation.** 2 ms → 1.0–1.2 ms/block. At 9 resident blocks that is −7 ms of a 108 ms step (+6 %, and
more once the GPU is the constraint); its real value is that it moves the residency crossover so item 2
can spend more VRAM profitably. capy's analogous result (T46): persistent grids + residents-only +
double buffering made the *kernel* 10–13 % faster (82 % of the roof) but end-to-end was flat **because
the CPU was binding** — i.e. do this only together with items 2/3.

---

## 8. Smaller items, with the measurement that bounds them

**Route-tail skip** (rank 8). Strata `expert_source.cpp:4763-4790`, skip logic `:4884-4905`: a missed expert that *every* token
of the window routes at rank ≥ R (default 7 on CUDA serve) is neither copied nor computed; its
contribution is zero and the others keep their router weight (no renormalisation). Saves ~10–15 % of
missed bytes — at 4 streams ~10 ms/step (+6 % aggregate). **It changes the output**, so it cannot go
behind our bit-identical gate; it is a `--quality-lossy` flag at best, and it is the right choice for
draft-row computation only.

**Wave32 on gfx1030** (rank 9). llama.cpp ships `RDNA_DEFAULT_SUBGROUP_SIZE = 32`
(`ggml-vulkan-types.h:1388`) with only `soft_max`/`im2col` pinned to 64 on RDNA2, and uses
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` (`ggml-vulkan.cpp:736`). We run 64
everywhere. Our GEMVs are at 83–106 % of RX DRAM so they cannot gain; the **GDN delta rule (217 GB/s
against 596 GB/s for a plain 640-row GEMV, 28 % of a layer)** is the one kernel with latency to hide,
and wave32 doubles the number of resident waves on a SIMD32 machine. Expect 0–5 % on decode, ~0 on
prefill. Cheap to test with a `--subgroup 32|64` flag on `bench gemv`/`bench ops`; watch the reduction
order (a 32-lane tree is not a 64-lane tree — if parity moves, do a fixed 64-wide shared-memory
reduction inside a wave32 kernel).

**int8 dot product on the GPU** (rank 10). llama.cpp gates `GL_EXT_integer_dot_product`
(`mul_mmq.comp:7`) on legacy/K-quant types incl. `iq3_s`, but **only for the q8_1-activation path**
(A is `block_q8_1_x4_packed128`, produced by `quantize_q8_1.comp`), and only when
`GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` is compiled and `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` is unset
(`ggml-vulkan.cpp:4090`). With no `VK_KHR_cooperative_matrix` this is **plain dp4a: 4 int8 MACs per lane
per cycle, ~4× f32** — real, but 4–8× short of the tensor-core path Basalt uses. Our dense chain is
latency/bandwidth-bound at T=1 (81–161 GB/s for 640-row shapes), so it buys nothing in decode; it only
matters for the chunk's dense GEMM after item 1 (where the MAC count is 256× and reuse is maximal).
Effort L, defer.

**Interleave prefill with decode** (rank 11). vLLM's chunked prefill (`vllm/v1/core/sched/scheduler.py:1311`)
and SGLang exist for *latency fairness*, not throughput. For us: one agent's 300-token prefix blocks the
other three for 95.6 s today; a chunk of 32 prefill tokens per decode step bounds their p95. This sets
no throughput record and only becomes attractive after item 1 makes prefill cheap.

**KV quantisation** (rank 12). Our KV is 2 048 B/cell f16 (`docs/phase-3/06` §3.1), 12 QSA layers →
**24 KiB/token**: at a 300-token prompt that is 7.2 MiB/token of attention reads against 799 MiB of
experts — **0.9 %**. It is not a lever for either headline number; it becomes one past ~2 k tokens
(100 MB/token at 4 k, still 12 % of expert bytes) and it is the VRAM lever for streams/card. Phase 6
already owns it; ik_llama.cpp's `-ctk-first/-last TYPE,N` (`common/common.cpp:2053`, `:3539`) is the
cheapest way to express "K at higher bits than V, tail layers lower" when we get there.

---

## 9. Measured negatives — do not spend time here

* **Tensor-parallel prefill/verify across both cards.** Basalt prices it at start-up and does not build
  it: *"the dense part is a third of a window's bytes, so splitting it saves about 2 ms between two
  cards and the exchanges between the cards (host-staged, about 5.4 µs per 10 KiB) cost about as much."*
  capy T30: 102 exchanges per window = 0.54 ms; T33: "rank 0 waits for the CPU about 7 ms per window
  while rank 1 idles at the all-reduce" — i.e. TP does not help when the host is the constraint, which
  is our situation. We also have no P2P. **Use expert-split (Basalt's peer prefill, `kPeerSlowRows =
  2559`) if the MI50 is ever given prefill work, never TP.**
* **Persistent / whole-layer mega-kernels.** Basalt built it and measured it **slower**: 324.6 → 288.2
  tok/s, with the reasons (expert half bound by cross-CTA dependency latency, 58 µs against a 30 µs byte
  floor; QSA half serialising projections; 1.5 MB of SASS and 20 % of PC samples stalled on instruction
  fetch). Vulkan is worse off: there is no grid-wide barrier, and the "every CTA resident" trick
  Basalt relies on (`docs/unified-kernel.md`) has no portable Vulkan equivalent on RADV. Our dispatch
  batching already gives one submission per block.
* **Descriptor buffers / push descriptors / shader objects.** Recording is 0.32 ms/token; the pool-reset
  change already banked +9 %. 0.3 % of a step — non-starter at 15 streams, let alone now.
* **Zero-copy BAR reads of host experts** (`VK_EXT_external_memory_host`). capy T22 measured "zero-copy
  GPU reads of misses were slower"; use explicit PCIe copies into a VRAM ring (item 3) instead. (Worth
  re-testing only because our extension list has it and the two drivers differ — one afternoon, after
  item 3 exists.)
* **Expert prefetch** (PILOT / `post_uni` / Strata lookahead): measured **−2.7 %** (capy T25) because the
  copies steal host read bandwidth. Keep the predictor, drop the prefetch.
* **fp16 accumulate** — already measured here (no speed, accuracy loss). **Timeline mid-batch sync,
  K>2 pipelining, grouped CPU MoE alone, arena sharing** — all measured here. Not revisited.

---

## 10. Top-5 next steps, one experiment each

Order is dependency order, not just impact: 2 and 3 are what make 5 and 6 pay.

1. **Merged chunked prefill, experts streamed to the RX.** Build `--prefill-chunk 300 --prefill-lane
   pcie` (one chunk = one block-run of `vkCmdCopyBuffer2` region arrays + one row-map dispatch per
   block). **Moves**: `bench agents --streams 4 --prefix 300` prompt seconds **95.6 → < 30** (prefill
   3.14 → > 10 tok/s/agent); parity 24/24 unchanged.
2. **Miss-split over PCIe at T=1** (shares the ring from 1). Build `--pcie-miss-share 0.3` with
   cheapest-weight-first selection. **Moves**: `bench agents --streams 4` aggregate **24.7 → 28.5**.
3. **Routing histogram, then per-expert residency.** Build `bench agents --profile-routes` (dump the
   `(layer, expert)` counts) and `--expert-profile`/`--resident-per-expert`. **Moves**: 4-stream lane
   **98 → 75 ms/step**, aggregate **24.7 → 28+**; the histogram alone tells us whether the ceiling is
   +10 % or +35 %.
4. **Lane to the roof.** Two arms in one window: (a) `MADV_HUGEPAGE` + `mlock` on the hot expert subset
   (no numerics change); (b) the VNNI/VBMI int8 form behind `--lane int8`. **Moves**:
   `bench cpu-experts --threads 32` **33 → 45–50 GB/s**, and 4-stream lane 98 → 72 ms. Numerics: (a) is
   bit-exact, (b) must be qualified as same-top-1 rather than bit-identical.
5. **Prefix snapshot reuse across agents.** Snapshot the per-session state (GDN recurrence + conv +
   indexer + PLE rolling state + KV page table) after a shared system prompt and admit agents from it.
   The page table is already identity-only and documented as the hook (`docs/phase-3/06` §3.1).
   **Moves**: for 4 agents sharing 200 of 300 prefix tokens, prompt seconds 95.6 → ~35 (independent of,
   and multiplicative with, item 1).

Deferred on purpose: MTP (§6) lands *after* 2/3/4 make the verify row cheap — building it first
reproduces the 0.60× loss in the table. Wave32, route-tail skip and GPU int8 dot are afternoons each
and should be scheduled as A/B cells, not projects.

---

## 11. What is still missing (measure before believing any projection above)

| unknown | why it decides the ranking | how to get it |
|---|---|---|
| `(layer, expert)` routing histogram on agent prompts | item 3's whole value: uniform → +10 %, skewed → +35 % | `bench agents --profile-routes` (new, ~20 lines) |
| device MoE cost at m = 5 and m = 23 rows/expert | item 1's and 7's ceiling | `bench gemv --experts 1,2,4,8,10,16` already sweeps expert count; add an m loop |
| RX PCIe at 48 concurrent 1.665 MiB region copies | item 1/3's byte floor (28.7 GB/s was measured on a synthetic ring) | `bench transfer --regions 508 --bytes 1745 000` |
| lane rate with page-locked/huge-page memory | item 4's (a) arm | 10-line experiment on `bench cpu-experts` |

One meta-observation worth recording: three independent engines on this model family
(Basalt, capy-flash, Strata) each converged on the same four mechanisms — *per-expert frequency
residency*, a *CPU/GPU split of the misses*, *chunked prefill with experts streamed to the GPU*, and
*MTP with a draft-vocabulary shortlist* — and all three measured the same negatives we did on
pipelining, fused mega-kernels and TP. The fact that capy-flash has our exact constraint (16 GB cards,
experts in host RAM, AVX2/AVX-512 host, agent workload) and reaches 1 774–3 222 tok/s prefill is the
strongest evidence in this survey that the prefill gap is a *scheduling* gap, not a kernel gap.

## Addendum (measured on this box, not from the clones): ReBAR is live and unused

Both cards expose a **16 GB BAR** (entire VRAM host-mapped), and Vulkan advertises device-local memory types with host access:

| Type | Heap | Flags | Meaning |
|---|---|---|---|
| 3, 4 | VRAM | `0x0007` | DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT |
| 9 | VRAM | `0x00c7` | + HOST_CACHED + DEVICE_COHERENT/CACHED (AMD) |

Today every per-block handoff goes through host-coherent system memory with explicit copies/fences (mailbox + readback path). These types allow **the handoff buffers themselves to live in VRAM, directly CPU-mapped**: the expert lane reads its input and writes its result in place — no staging copies, no extra submissions. Payloads are 10–40 KiB, so even the MI50's 2.9 GB/s BAR is irrelevant (latency-bound), and `0x00c7` is explicitly built for cached host access.

Expected: trims the mailbox/readback slice of the host-lane critical path (currently 55–71% of the step) and removes a class of fence round trips. Effort S–M; risk low (a memory-type selection plus mapping lifetime); measure with 2- and 4-stream `bench token`/`bench agents` plus the unchanged parity gate. It composes with every ranked item above, including the merged-prefill stream (item 1), where the CPU never touches the activations at all.

> **Status after §8u:** the budget is **72.3 MiB/token measured** (11.1x fewer
> expert bytes than the host lane's 799 MiB/token; **2.7x** if routing were uniform,
> which it is not - a 128-token chunk touches ~130 of 512 experts per block).
> Per-token cost: PCIe **2.6 ms** (RX) / 25 ms (MI50, so hot experts must be
> resident), **GEMM 0.5 ms measured**, dispatch 0.22 ms - against 163-320 ms today.
> The first kernel is built and validated (`GemmMany`, IQ4_NL `down`): m = 8 is
> **3.4x (MI50) / 5.1x (RX)** per-user throughput over m = 1, bit-identical to the
> single-user GEMV for every m. Still missing: the `Q2_0` and `gate`/`up` shaders,
> hot-expert residence, and the m-user fan-in.
