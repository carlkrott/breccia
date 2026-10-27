# 19 — MLX and "LLM in a flash": expert offload and locality for Breccia

Round 19 of the paired rotation. Research only: no code was changed, and no GPU, binary or benchmark was run. Every speed number is either cited, marked `derived` (arithmetic on cited inputs), or marked `UNVERIFIED - requires the GPU qualification card`.

## 1. Summary

- **MLX's MoE sort is a prefill trick.** `switch_layers.py` sorts by expert only when there are at least 64 indices (S1 L204, L244). Breccia already groups prefill work by expert (`graph.rs` L2674–2676, `prefill_block_merged` L2566). Decode is one token and never reaches MLX's threshold.
- **MLX's prompt cache is trim-based, and that does not port.** `can_trim_prompt_cache` (S2 L115) needs every layer to be trimmable. Breccia's GDN state is not, so doc 16's snapshot-and-fork design stands.
- **"LLM in a flash" (arXiv 2312.11514) has three mechanisms.** Windowing (§3.1) and row-column bundling (§3.2) carry over. The low-rank predictor (§3.1) does not, because Breccia's routing is exact top-10 and already on device.
- **What is new to this corpus.** Doc 02 §3 scores residency by frequency only. Doc 05 E3 proposes an LRU expert tier with no replay. Nothing measures recency. Expert bundling at expert granularity is also a gap.
- **Most of the rest is carried, not new.** Prefix snapshots (doc 16, doc 02 item 5), q8 KV (doc 09 A4), the gate/up map dedupe (doc 18 R2) and sampler partial selection (doc 18 R3) are already proposed.
- **Top 3:** (1) replay one ordered routing trace against VRAM-residency policies before building any expert cache, and fault-count the page cache separately; (2) relayout expert blobs so each streamed expert is one contiguous read; (3) sampler partial selection on temperature>0 decode (carried from doc 18 R3).

## 2. Sources

Fetched 2026-10-27. Hashes are the first 16 hex digits of the sha256 of the local copies, so the line numbers can be checked against the same bytes.

| ID | Source | What it supplies | Anchors |
|---|---|---|---|
| S1 | mlx-lm `mlx_lm/models/switch_layers.py` (ml-explore/mlx-lm, main), sha256 `438e000aa28c69f1` | MoE sort/unsort, gather_mm/gather_qmm | `_gather_sort` L11, `_scatter_unsort` L19, `QuantizedSwitchLinear` L38, `mx.gather_qmm` L94, `SwitchLinear` L112, `mx.gather_mm` L140, `SwitchGLU` L182, `do_sort = indices.size >= 64` L204 and L244, one shared sort for up/gate/down L207–217 |
| S2 | mlx-lm `mlx_lm/models/cache.py`, sha256 `c96cb169fd5e0ec4` | prompt cache, trim, rotating and quantized KV, prefix trie | `make_prompt_cache` L15 (`RotatingKVCache(max_size, keep=4)` L37), `can_trim_prompt_cache` L115, `trim_prompt_cache` L122, `QuantizedKVCache` L251 (`group_size=64, bits=8` L254), `KVCache.to_quantized` L396 (default bits 4), `RotatingKVCache` L423, `ArraysCache` L596, `PromptTrie` L1508, `LRUPromptCache` L1599 |
| S3 | mlx-lm `mlx_lm/sample_utils.py`, sha256 `3aac92d804b6b91c` | top-k and top-p sampling | `make_sampler` L13, `apply_top_k` L138 with `mx.argpartition` L155, `apply_top_p` L213 (sorts) |
| S4 | arXiv 2312.11514, "LLM in a flash" (HTML v3) | windowing §3.1, row-column bundling §3.2, predictor §3.1, App. D negative result | Table 2 (OPT 6.7B on M1 Max): bundling raises throughput 1.25 → 2.25 GB/s and cuts I/O latency 164 → 87 ms. Abstract: 4–5× on CPU, 20–25× on GPU versus naive loading. The window size k is stated inconsistently (Figure 4 vs App. C.1). |
| S5 | arXiv 2505.16056, "Not All Models Suit Expert Offloading" (ICLR 2026) | locality metric, cache sizing | Abstract only. Defines SCH (segment cache best hit rate). Says caches of about 2× the active experts usually balance well. Says shared experts can lower local routing consistency. No hit-rate numbers in the abstract, so none are used here. |
| S6 | arXiv 2410.17954, ExpertFlow | routing-path predictor with runtime correction | Abstract only. Its "up to" memory and throughput claims are quoted as claims only. |
| S7 | MLX docs, lazy evaluation (ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) | lazy graph semantics | "Currently, MLX does not compile and rerun compute graphs." |
| S8 | MLX docs, `mlx.core.gather_mm` | `sorted_indices` | "May allow a faster implementation if the passed indices are sorted. Default: False." |
| S9 | DeepWiki on ml-explore/mlx (secondary) | `mx.compile` element-wise fusion; `gather_qmm_rhs` when M==1 | Secondary, no line numbers. Used for no verdict. The community ssd-moe repo had no DeepWiki index. |

Breccia facts (repo-relative, read in this round):

- `crates/breccia-engine/src/graph.rs`: `STREAM_SLOTS` L271; stream-slot comment L254–259; `router_ids` allocation L722; router run L2035; `fill_stream_slot` L2516–2555, whose comment at L2509 says it replaces "the host lane's per-token work"; `prefill_block_merged` L2566 with its numerics comment above it; streamed groups from `pairs.sort_unstable()` L2674–2676; `run_placed_moe` L3324 (separate gate and up maps); `sh` and `sh_gate_logit` L350–351; `ExpertPool` L500 and L996.
- `crates/breccia-engine/src/xfer.rs`: `Mailbox` L435–452, preferring DEVICE_LOCAL|HOST_VISIBLE with a HOST_VISIBLE fallback.
- `crates/breccia-engine/src/model.rs`: routed experts 39.97 GiB, host mmap, demand-paged (L12).
- `crates/breccia-kernels/src/cpu/expert.rs`: `slices` L139–153. Each role has its own shard offset plus `expert × per_expert`, so the three role blobs of one expert are three separate regions.
- `crates/breccia-engine/src/sampler.rs`: greedy argmax returns at L96–97; full sort L105–110; top-k only, no top-p.
- `crates/breccia-engine/src/arena.rs`: first-fit decreasing placement, L202–213.
- `crates/breccia-cache/src/lib.rs` is a one-line stub. `docs/BRIEF.md` L59 describes it as holding expert residency primitives. The doc and the crate disagree, and this needs fixing before any policy code lands.

Brief numbers, used as given: 48 blocks, 30 resident, 18 streamed, top-10 of 512 experts, expert blob 1.665 MiB, 799 MiB expert bytes per token, 301 MiB streamed per token, 18.59 tok/s single stream.

Repo docs: doc 02 §1 row 5, §3 (L128–172), §9 (L404–405), §11 (L441–445); doc 05 E3 (L471–473); doc 09 A4 (L204–240) and NVMe knee (L178–180); doc 16 L11–13 and L67–72; doc 17 L151–152 (replay gate); doc 18 R2 (L14, L73, L77) and R3 (L15, L103–105).

## 3. Mapping table

Verdict key: **adopt** (build or measure next); **carried** (proposed in an earlier round, not built, no new action here); **already have** (in code or measured); **gap** (not in code, not yet proposed); **does not apply** (reason given).

| # | MLX / paper / other idea | Breccia equivalent or gap | Verdict | Reason |
|---|---|---|---|---|
| 1 | MoE sort by expert, `do_sort = indices.size >= 64` (S1 L204, L244) | Prefill groups distinct experts (`graph.rs` L2674–2676) | already have | MLX sorts only at 64+ indices, which is prefill-only. Decode is one token. |
| 2 | gather_mm / gather_qmm with sorted indices (S1 L94, L140; S8) | `prefill_block_merged` (L2566): one multi-user GEMM per distinct expert | already have | Same idea. Breccia's per-user dot is bit-identical to the single-user kernel (numerics comment above L2566). |
| 3 | One shared sort for up, gate and down (S1 L207–217) | Separate gate and up maps in `run_placed_moe` (L3324) | carried (doc 18 R2) | Doc 18 L73 splits 240 → 150 into 8 → 7 (drop `map_up`) and 7 → 5 (gate+up+SwiGLU fold). The dedupe alone is 240 → 210 per token (derived). The fold is doc 10 D1. |
| 4 | Lazy evaluation; no graph compile by default (S7) | Breccia records a command graph; the recording cost is small (doc 18 L77 cites 66 ns per dispatch from doc 11 Part A) | already have | MLX gives no new mechanism here. |
| 5 | `mx.compile` element-wise fusion (S9) | Fusion ideas in doc 10 D1–D4 | does not apply | Secondary source, no line numbers. Decode time is GEMV-issue and dispatch bound (README round 10 and round 12 verdicts), so element-wise fusion is not the lever. |
| 6 | `RotatingKVCache`, `keep=4` (S2 L37, L423) | Full-cell QSA cache with top-512 block selection (`qsa_topk.comp`) | does not apply | A sliding window drops cells the indexer can select. That changes QSA output and breaks the greedy parity contract. |
| 7 | `QuantizedKVCache`, group 64, bits 8 (S2 L251–254) | q8_0 KV priced in doc 09 A4 (attention 0.32×; q4 9% error) | already have | Priced in doc 09. MLX is a second precedent, not a new number. |
| 8 | `can_trim_prompt_cache` and `trim_prompt_cache` (S2 L115–122) | GDN recurrent state is updated in place each token (doc 16 L13) | does not apply (as trim) | MLX's trim reuse needs every layer to be trimmable. Breccia must snapshot and fork instead, which doc 16 already specifies. |
| 9 | `PromptTrie` + `LRUPromptCache` (S2 L1508, L1599) | Grid-aligned GDN snapshots and fork-point rule (doc 16 L11–13, L67–72); doc 02 item 5 | carried (doc 16, doc 02 #5) | Same goal. MLX's trie is per token; Breccia's reuse unit is the grid-aligned GDN state. |
| 10 | Full-vocabulary sort for top-k (S3 L138–155 via `argpartition`) | Full sort on the stochastic path (`sampler.rs` L105–110); greedy returns before it (L96–97) | carried (doc 18 R3) | Only temperature>0 decode moves. Ranked 3 below. |
| 11 | top-p sorts (S3 L213) | Breccia has no top-p; doc 18 R3 notes it | gap, not adopted | Not needed for the greedy parity contract. Not proposed here. |
| 12 | Windowing: keep only neurons the last k tokens used (S4 §3.1) | No temporal expert cache. Residency is static block placement. Doc 02 §3 is a frequency proposal. Decode's non-resident experts run on the host lane (`graph.rs` L2509 comment, L500, L996). | gap, measure first | The paper's windowing is a recency policy. Breccia has no recency data at all. Ranked 1. |
| 13 | Explicit LRU expert tier (doc 05 E3, L471–473) | Same as row 12 | gap; proposed in doc 05, never replayed | Doc 17 L151–152 is the precedent: replay first, threshold set in advance. |
| 14 | Row-column bundling: up column i with down row i (S4 §3.2, Table 2) | `fill_stream_slot` writes gate, up and down from three shard regions (`graph.rs` L2516–2555; `cpu/expert.rs` L139–153) | gap; adopt as a measured candidate | Each expert is three scattered reads. Bundling makes it one. Ranked 2. |
| 15 | Co-activation bundling (S4 App. D) | none | does not apply | The paper's own negative result. |
| 16 | Low-rank sparsity predictor (S4 §3.1) | Exact top-10 routing, already on device (`router_ids` L722, L2035) | does not apply | Breccia does not need to predict its routing. A predictor-driven prefetch was measured −2.7% (doc 02 §9, L404–405). |
| 17 | Routing-path predictor with runtime correction (S6) | none | does not apply | Same reason as row 16. The correction step is what `ExpertPool` already does on the host lane. |
| 18 | Preallocation and swap-with-last deletion (S4 §3.3) | First-fit arena (`arena.rs` L202–213); fixed `STREAM_SLOTS = 8` (`graph.rs` L271) | already have | Fixed slots need no swap-with-last. |
| 19 | Expert cache at 2× active experts (S5) | Would be 20 experts per streamed block, 360 in total, about 599 MiB (derived) | gap; sets the replay capacities | Used in Top 3, item 1. |
| 20 | SCH best-hit metric (S5) | none | gap; use in Top 3, item 1 | Report per-segment best hit with the paper's own definition. Not re-derived here. |
| 21 | OS page cache over the expert mmap | Routed experts are demand-paged from the model volume (`model.rs` L12); no explicit host cache | already have, implicitly | The kernel already runs a recency LRU over these bytes. A host-RAM cache must be scored against it, not against zero. |
| 22 | Apple unified memory (MLX's target) | Discrete GPUs over PCIe, with ReBAR where it is live (`xfer.rs` L435–452) | does not apply | The unified-memory argument does not carry to a host-to-VRAM lane. |
| 23 | Community SSD-offload projects (ssd-moe, Flash-MoE "71% hit") | none | not used | Community or self-reported numbers. Listed in section 6 only. |

## 4. Top 3

### 1. Ordered routing trace, then an offline replay (new)

- **Part (a), VRAM residency replay.** Decides between three expert-cache policies at the current resident byte budget and at the S5 sizes: (i) static frequency per (layer, expert), which is doc 02 §3; (ii) LRU per (layer, expert), which is doc 05 E3; (iii) a window of the last k tokens, capped at the same budget, which is paper §3.1. The OS page cache is not an arm here. It lives in host RAM, and an expert held in VRAM is not in host RAM, so the two tiers cannot be compared at equal bytes.
  - Metric: host-lane expert computations per token. Baseline: 18 streamed blocks × 10 experts = 180 per token (derived). Each cache hit removes one, so 180 × (1 − h) remain (derived; h UNVERIFIED - requires the GPU qualification card).
  - Sizes: the current free-VRAM budget (not in BRIEF; to be read from the device), plus S5's 2× rule: 20 experts per streamed block, 360 in total, about 599 MiB (derived). For the 1× and 4× arms, 180 and 720 experts, about 300 MiB and 1.17 GiB (derived).
  - The replay runs on the CPU from the trace, with no GPU. Collecting the trace needs one GPU run (see the plan).
- **Part (b), host-RAM fault check.** Decides whether a host-RAM tier, and item 2, matter at all. The 18 streamed blocks hold 18 × 512 × 1.665 MiB = 15.0 GiB of expert bytes (derived). That is the most the page cache must hold for h near 1 after warmup. BRIEF L17 gives 58 GB of RAM and L29 a model budget of about 30 GB of host RAM. BRIEF L22 gives a 27 GB PLE table. `model.rs` L13 and L238 say it is a demand-paged host mmap, so it competes for the same cache. `model.rs` L13 gives 26.82 GiB and the L940 comment gives 28.8 GiB; that conflict is unresolved. Whether the 15.0 GiB stays resident is therefore a measurement, not a given. Measure major faults per token on the GPU card from the process's rusage delta. Near-zero faults after warmup means the lane is not NVMe-bound, so the 5.87 GB/s arithmetic below is a coincidence and item 2's decode gain is small. High faults mean the lane is NVMe-bound, and both part (a) and item 2 matter.
  - Arithmetic (derived, not evidence): 301 MiB × 18.59 tok/s ≈ 5.87 GB/s, matching doc 09's NVMe QD 8 knee (5.8–5.9 GB/s).
- **Files:**
  - `crates/breccia-engine/src/graph.rs` L722 and L2035: router ids are on device.
  - `graph.rs` L500 (the `ExpertPool` field) and L996 (pool construction). The host-lane call site was not read.
  - `crates/breccia-kernels/src/cpu/expert.rs` L139: `slices`, the one place that counts per-expert reads.
  - A new bench flag is needed. None exists: `grep` of `crates/` for `profile-routes` and `expert-profile` finds nothing, so doc 02 §11 remains a proposal.
  - `crates/breccia-cache/src/lib.rs`: the stub where a policy would live (see the BRIEF inconsistency in §5).
- **Readback cost:** the trace needs (block, 10 ids) per token. If the host lane already receives ids for the 18 streamed blocks, logging adds no readback there and only the 30 resident blocks need one: 30 × 10 int32 = 1,200 B per token (derived). If not, 48 blocks give 1,920 B per token (derived). Which case holds was not confirmed in one read. Sync cost UNVERIFIED - requires the GPU qualification card.
- **Measurement plan:**
  1. GPU card, one process, quiet window, `timeout 90`. Run a fixed agent prompt mix. Log (token, block, 10 expert ids) and the per-token major-fault count from the process's rusage delta (suggested method, not an existing Breccia tool). Output must not change: greedy 32/32.
  2. CPU only: replay the trace offline under arms (i) to (iii) at the budgets above, reporting host-lane computations per token, hit rate, and the SCH-style best hit using S5's definition.
  3. **Proposed gate (my proposal, not sourced):** build a policy only if it beats static frequency by at least 5 hit-rate points at the same budget. Fix the threshold before the run, as doc 17 L151–152 does.

### 2. Expert-blob relayout: one contiguous read per expert (paper §3.2 analog, new)

- **Moves:** region reads per token, and the size of each read. Derived: decode streams 18 blocks × 10 experts × 3 roles = 540 region reads per token; with one contiguous span per expert it is 180 (derived). The bytes do not change, so the parity contract is untouched. The TPS effect is UNVERIFIED - requires the GPU qualification card. S4's gains were measured on flash IOPS, not on a warm page cache, so its decode value depends on item 1 part (b) showing high major faults (the lane being NVMe-bound). With near-zero faults the expected effect is small.
- **Mechanism:** an offline step writes a sidecar where each (block, expert) is `[gate | up | down]`, 1.665 MiB contiguous. `slices()` returns three subslices of one span, and `fill_stream_slot` becomes one write.
- **Cost:** a second copy of the routed experts on the model volume (39.97 GiB, `model.rs` L12). The span only works if the three roles have equal byte sizes, which holds only if the quant type is the same for all three. Whether it holds is UNVERIFIED; if not, the layout needs per-role offsets.
- **Files:** `crates/breccia-kernels/src/cpu/expert.rs` L139–153 (one span in the slices plan); `crates/breccia-engine/src/graph.rs` L2516–2555 (one write); `crates/breccia-engine/src/model.rs` (loader reads the sidecar); a new offline repack step, whose location is still to choose and which does not exist in the repo today.
- **Measurement plan:**
  1. No GPU: NVMe read throughput at 568 KiB (1.665 MiB ÷ 3, derived, assuming equal role sizes) versus 1.665 MiB requests at QD 8, against the knee in doc 09 L178–180. This step runs without a GPU and does not need item 1 part (b); the decode gain in step 2 does.
  2. GPU card: host-lane time per token and region-read count, relayout versus current; output bit-identical.
  3. **Proposed gate (my proposal, not sourced):** adopt only if the bundled size reads faster per byte than the split size by a margin fixed before the run.

### 3. Sampler partial selection on temperature>0 decode (carried from doc 18 R3)

- **Carried:** doc 18 R3 already proposes it, and already states that it has no greedy gain. This round confirms the greedy path returns argmax before the sort (`sampler.rs` L96–97). MLX's `apply_top_k` (S3 L155) is a second precedent for `argpartition`.
- **Moves:** host sampler time per token, only for temperature > 0. Greedy decode, the parity contract and the 18.59 tok/s figure are untouched. The per-token cost is UNVERIFIED - requires the GPU qualification card. Derived: a full sort of V logits is about V·log₂V comparisons. At V = 248,320 (doc 18's figure, not re-read from the model config), that is about 4.45 M comparisons.
- **Files:** `crates/breccia-engine/src/sampler.rs` L105–110 (the sort) and its unit tests (around L139). Keep the tie-break comparator (score descending, index ascending), so draws stay identical for a given seed.
- **Measurement plan:** host time per sampled token at V = 248,320 and top_k 20, before and after. Draws identical over seeds 0 to 999, with planted ties and NaNs, as doc 18's plan specifies. Greedy 32/32 unchanged.

## 5. Corrections and overlaps

- **Doc 18 R2 is not in error.** Its L73 chain is 8 → 7 (drop `map_up`), then 7 → 5 (fold). The dedupe alone is 240 → 210 per token (derived: 30 placed blocks × 1). The 150 figure needs the fold, which is doc 10 D1.
- **Doc 18 R3 already says "no token-time gain".** This round confirms it at `sampler.rs` L96–97.
- **Top 3, items 1 and 2 are new to the corpus.** Doc 02 §3 measures frequency only. Its §11 names the missing routing histogram (L441–445). Doc 05 E3 proposes an LRU tier without a replay. The Top 3, item 1 replay is the decision instrument for both.
- **The OS page cache is an existing recency LRU** over the expert bytes (`model.rs` L12). A host-RAM cache is judged against it in item 1 part (b), not in the VRAM replay.
- **The `breccia-cache` stub contradicts BRIEF L59.** Fix before any policy code lands.

## 6. Sourced and unsourced claims

Sourced: everything in sections 2–5 that carries an S-number, a file and line, or a repo doc and line.

Unsourced or not used:

- ExpertFlow's 91.96% hit ratio and 61.15% figure came from a search snippet, not the abstract. Not used.
- Flash-MoE's 71% OS-page-cache hit rate is self-reported by a community project. Not used.
- DeepWiki's `gather_qmm_rhs` M==1 note has no line numbers. Not used for any verdict.
- No speedup numbers for MLX's sorted gather were found in S1 or S8. None are quoted.
- The 2× cache sizing, the SCH definition and the shared-expert consistency note come from the S5 abstract. The body was not read.
- The paper's window size is inconsistent across its own sections (k = 4 vs 5). Not resolved.
- All per-token timings (sampler time, lane time, read rates at the new sizes) are UNVERIFIED - requires the GPU qualification card.
- The 5.87 GB/s match is arithmetic, not a measurement.
- The 15.0 GiB working set is derived (18 × 512 × 1.665 MiB). The 58 GB RAM, about 30 GB model budget and 27 GB PLE table are from BRIEF L17, L29 and L22; the free VRAM after the arena is not in BRIEF and is not read here.
- Whether the host lane receives router ids for streamed blocks was not confirmed (graph.rs L500 and L996 are the pool field and constructor, not the call site).
- The 5-point gate and the sample counts are my proposals, labelled where they appear.
- V = 248,320 is doc 18's figure, not re-read from the model config this round.

## 7. Skipped

No GPU run, no trace collection, no read of the host-lane call site for streamed-block router ids, no model-config read of the vocabulary size, and no read of the paper's body beyond the HTML sections. Not reviewed: Fiddler, MoE-Infinity, HOBBIT, AdapMoE, and the community SSD-offload repos (not indexed).
