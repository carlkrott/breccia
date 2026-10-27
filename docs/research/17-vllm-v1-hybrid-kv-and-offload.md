# Round 17: vLLM v1 hybrid KV manager, block-hash prefix cache, CPU offload, unified scheduling

**Scope.** Read-only audit of vLLM v1 at the pinned SHA `274a5e19a18460ed41d47117398494b721e3df16` (2026-10-27). Sources were read through the GitHub API and streamed to stdout; none are stored locally. Line numbers are at the pinned SHA. Breccia baseline: `ad34751` (round 16). No code, build, or GPU work. Rounds 13 to 16 are cited, not re-audited.

**Caveat on the vLLM design doc.** `docs/design/hybrid_kv_cache_manager.md` predates the current code. Its "exactly two groups" text is stale. Its "Case 4" rule (grow the attention block until its page covers the state, then pad the state) is what the code runs for hybrid models (Q1). Use the design doc for intuition only.

**Labels.** `UNVERIFIED - requires the GPU qualification card` marks any number that needs a GPU run. `derived` marks arithmetic from Breccia's source constants and vLLM's rules. `UNVERIFIED - requires trace replay` marks hit rates that need recorded traces.

## Verdict

On its hybrid path vLLM does not pad KV pages to the state page. The platform hook grows the attention block to 1,600 tokens, so the KV page covers Breccia's 3.27 MB GDN page, and pads the state by 0.25 %. The ~100x padding applies only to the unify fallback at 16-cell pages, which the hybrid path does not reach (Q1). What remains is granularity. Under prefix caching, vLLM's default ties the state grid and the KV allocation grain to 1,600 tokens. Breccia should keep KV at 16 cells and use a 64-token state grid.

Ranked:

1. Sparse snapshot retention (`reachable_block_mask` with `retention_interval = 0`: fork junctions plus tail, periodic snapshots dropped). The 64-token grid then applies only to on-demand snapshots (R-1 fork points and the tail). Derived: a linear 100K prefix goes from 6 snapshots (673 MiB) to 1 or 2 (112 to 224 MiB); a positive interval keeps one state per segment and saves less. Partial-hit recompute cost: UNVERIFIED - requires trace replay.
2. 16-cell KV pages with a 64-token state grid, not vLLM's 1,600-token coupling. Derived: KV waste per stream falls from up to 39.3 MB to up to 0.37 MB, and a fork-point or tail restore recomputes at most 63 tokens past the grid point instead of up to 1,599. Hit-rate gain: UNVERIFIED - requires trace replay.
3. ARC ghost-list eviction for the host snapshot tier, gated on a replay win over LRU.

Unified token-budget scheduling folds into round 13 C4 and B10 and phase-4 section 2. Its throughput effect on Breccia is UNVERIFIED - requires the GPU qualification card.

## Breccia numbers used

| Quantity | Value | Source |
|---|---|---|
| GDN state per layer | 3,145,728 B (786,432 f32) | `crates/breccia-kernels/src/gdn/mod.rs` L79-88 |
| GDN conv state per layer | 122,880 B (120 KiB) | same, `conv` field |
| GDN state + conv per layer | 3,268,608 B = 2^13 x 399 = 4,096 x 798 | derived |
| One snapshot (36 GDN layers) | 117,669,888 B = 112.2 MiB | derived; matches rounds 15 and 16 |
| Padded state per layer (1,600-token attention page) | 3,276,800 B, +8,192 B (+0.25 %) | derived, Q1 |
| QSA page (16 cells, one layer, K+V f16) | 32,768 B | `crates/breccia-kernels/src/qsa/mod.rs` L100-102; `PAGE_CELLS` L89, `MODEL_N_KV_HEAD` L71, `MODEL_HEAD_DIM` L73 |
| QSA per token per layer | 2,048 B | 2 KV heads x 256 x (K,V) x 2 B |
| QSA per 16-token block, 12 layers | 393,216 B (384 KiB) | derived |

**Note on BRIEF figures.** The 340 MB per token in `docs/BRIEF.md` is traffic, about 3x the 113 MB state size (36 x 3,145,728 B). It is not a state size. The BRIEF's 275 µs one-pass floor, 2 x 36 x 3,145,728 B / 825 GB/s, is one read plus one write. Both figures are consistent with the source sizes.

## Q1. Hybrid attention and GDN block sizes

### What vLLM does (pinned SHA)

- **Config step, not the alignment.** `HybridAttentionMambaModelConfig.verify_and_update_config` (`vllm/model_executor/models/config.py` L636) delegates to `MambaModelConfig` (L801). Its docstring says alignment with mamba page sizes happens later, in the platform hook. `MambaModelConfig` sets `mamba_cache_mode` to `"align"` when prefix caching is on, and sets `mamba_block_size` to `max_model_len` when it is off (both within L801-840). The field default is `"none"` (`vllm/config/cache.py` L304).
- **Pre-alignment (live).** `Platform.update_block_size_for_backend` (`vllm/platforms/interface.py` L750) runs phase 2, `_align_hybrid_block_size` (from about L940), when `model_config.is_hybrid`. The executors call it after `load_model` (`vllm/v1/executor/uniproc_executor.py` L75; the multiprocess executor also). The rule:
  - attention page for one token: 2 x kv_heads x head_dim x dtype, for K and V;
  - mamba page from state shapes and dtypes (`MambaSpec`, `vllm/v1/kv_cache_interface.py` L1094-1135);
  - kernel alignment k = max(smallest backend kernel block, block_size), raised to lcm(k, checkpoint alignment) in align mode;
  - `attn_block = k x ceil(mamba_page / (k x attn_page_1_token))`, rounded up to a multiple of every backend's smallest kernel block (L1098-1104);
  - the mamba page is padded to `attn_block x attn_page_1_token`.
- **Kernel split.** The manager block is not the kernel block. `prepare_kernel_block_sizes` (`vllm/v1/worker/utils.py` L630) calls `select_common_block_size`, which picks a kernel block the backend supports that divides the manager block. The `prefix_match_unit` docstring (`vllm/config/cache.py` L206-208) gives a 1,024-token hybrid-model block as an example of a coarser unit.
- **Fallback.** `unify_kv_cache_spec_page_size` (`vllm/v1/core/kv_cache_planning.py` L346-401) runs only when pages still disagree after pre-alignment, for example a draft model with a larger page. Branches: (a) a mamba layer with a smaller page gets `page_size_padded` = target; (b) an attention layer whose page divides the target gets `block_size *= target / page`; (c) other non-MLA attention gets `page_size_padded` through a strided view. MLA that does not divide raises `NotImplementedError`.
- **Scheduler block.** `resolve_kv_cache_block_sizes` (`vllm/v1/core/kv_cache_utils.py` L816) sets the scheduler block to the LCM of the group block sizes. Hash grain: Q2.
- **Hit rule.** `HybridKVCacheCoordinator.find_longest_cache_hit` (`vllm/v1/core/kv_cache_coordinator.py` L806) iterates to a fixed point over attention groups. A mamba group hits only at one state boundary. `_cache_hit_alignment_tokens` (L709) gives the hit granularity.

### Applying the rule to Breccia (derived, no GPU)

Inputs: attention page 2,048 B per token per layer; mamba page 3,268,608 B per layer.

- **Prefix caching off.** k = 16, the QSA page cells. `attn_block = 16 x ceil(3,268,608 / 32,768) = 16 x 100 = 1,600` tokens. The padded state page is 1,600 x 2,048 = 3,276,800 B, +8,192 B. The mamba group's block is `max_model_len` here, so the 1,600-token block is only a KV allocation grain.
- **Prefix caching on (align mode).** k = lcm(16, 64) = 64. The 64 is `FLASHINFER_GDN_CHECKPOINT_ALIGNMENT` (`vllm/v1/attention/backends/gdn_attn.py` L34), set on the GDN layer at `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` L696 (condition not re-read). `attn_block = 64 x ceil(3,268,608 / 131,072) = 64 x 25 = 1,600`. Same result.
- **Cost of a 1,600-token block.** 12 QSA layers allocate 12 x 1,600 x 2,048 = 39,321,600 B per block. One state boundary for 36 padded layers is 117,964,800 B (112.5 MiB).
- **Padding is not the discriminator.** Pre-alignment costs +8,192 B (+0.25 %) at 1,600. Padding the state to a 16-cell page costs the same, because 1,600 = 100 × 16 and the state is 8,192 B short of 100 such pages. Both are small.

The ~100x figure applies only to fallback branch (c) at 16-cell pages. The hybrid path does not reach it.

### Two designs, one question

vLLM shows the state and KV can share one block grid at 1,600 tokens for 0.25 % padding. So padding no longer rules a shared grid out. The design question is granularity: where a restore may land (the state grid) versus where KV memory is wasted (the allocation grain). vLLM couples them at 1,600 under prefix caching. Q2 and Top 3, item 2 separate them. Breccia's QSA kernel reads 16-cell pages natively, so the kernel-level split vLLM uses is already in Breccia's layout.

## Q2. Block-hash prefix caching vs round 15 R1 and round 16 R-1

### vLLM

- **Hash chain.** `hash_block_tokens` (`vllm/v1/core/kv_cache_utils.py` L707) and `get_request_block_hasher` (L953) hash the parent hash, the block tokens and extra keys. Extra keys cover LoRA, multimodal hashes and `cache_salt`. Only full blocks are hashed. Algorithms: sha256 (default), sha256_cbor, xxhash (`docs/design/prefix_caching.md`).
- **Hash grain (corrected).** The hash block size is `prefix_match_unit` if set, else the GCD of the prefix-cacheable group block sizes (`resolve_kv_cache_block_sizes`, L831 and L890-895). With pre-alignment both groups are 1,600, so the default grain is 1,600, not 16. `prefix_match_unit = 16` passes the divisibility check, since 16 divides 1,600. Whether that yields 16-token hits depends on `partial_hash_hits_enabled` (`vllm/v1/core/kv_cache_utils.py` L762; call sites at L909 and L943 read, body not read). Hit alignment is resolved in `resolve_cache_hit_alignment_tokens` (L921; read only as far as its docstring).
- **State grid.** With `prefix_match_unit` set, `get_mamba_prefill_checkpoint_grid` (`vllm/model_executor/layers/mamba/abstract.py` L116-129) returns lcm(prefix_match_unit, checkpoint alignment), which is 64 for GDN. Unset, it returns `mamba_block_size`, which is 1,600.
- **Lookup.** `BlockPool.get_cached_block` (`vllm/v1/core/block_pool.py` L202) returns a block only if every group has it. `cache_full_blocks` is at L230.
- **Eviction.** `BlockPool.free_blocks` (L816) puts unhashed blocks at the head and hashed blocks at the tail, so hashed blocks are evicted last. That is LRU-like. The GPU pool has no ARC and no LFU.
- **Sparse retention, mamba only.** `MambaManager.reachable_block_mask` (`vllm/v1/core/single_type_kv_cache_manager.py` L1549). `retention_interval` None means dense (default). 0 keeps only reachable boundaries. A positive value keeps one state per segment. Reachable boundaries (the replay boundary and shared-prefix junctions) are always kept.

### Breccia

- **Two different knobs.** The snapshot cadence (R1: periodic at 16K, round 15) and the hit grid (where a restore may land) are separate. Retention decides which snapshots exist. The hit grid matters only where snapshots are placed on demand: round 16 R-1 fork points and the prompt tail. Periodic R1 snapshots are already 16K-aligned.
- **Dense versus sparse.** R1 keeps a periodic set at a 16K cadence. A 100K prefix gives 6 snapshots, 673 MiB at 112.2 MiB each. At vLLM's default 1,600-token grid, dense periodic snapshots would be about 62 per 100K, about 7.3 GB (derived). At a 64-token grid they would be 1,562 per 100K, about 184 GB (derived). So the 64-token grid applies to on-demand snapshots only (R-1 fork points and the tail), and periodic snapshots need a retention rule.
- **Sparse mode.** For a linear 100K prefix with no shared junction, sparse retention keeps 1 to 2 snapshots, 112 to 224 MiB: a 3x to 6x cut in T2 bytes (derived). A partial-prefix hit between kept boundaries recomputes from the nearest kept state. UNVERIFIED - requires trace replay.
- **Hit grid.** Chain KV at 16 cells. Place fork points and the prompt tail on G = 64 (lcm(C,16), C at least 64). That is vLLM's `prefix_match_unit = 16` option with its 64 checkpoint alignment. Do not copy the 1,600 default.
- **Extra keys.** vLLM folds LoRA and `cache_salt` into the hash. Breccia's equivalent must fold in the KV dtype (f16 versus the q8_0 layout), the model GGUF identity and the quant type. Without them, two configs can share a key and reuse the wrong state. This matches round 15 R3's identity-sealed hash.

**Delta vs R1 and R-1.** R-1 stands. R1's periodic set can shrink to junctions plus the tail once sparse retention is live and replay bounds the recompute. The new item is the 64-token hit grid for on-demand snapshots, in place of vLLM's default 1,600.

## Q3. CPU offload: policy, async DMA, and the Vulkan equivalent

### vLLM

- **Manager.** `CPUOffloadingManager` (`vllm/v1/kv_offload/cpu/manager.py` L49). The `cache_policy` default is `"lru"`, with `"arc"` as the alternative, built through `CachePolicyFactory`. An out-of-tree policy module can be loaded. `max_tracker_size` is 64,000. `store_threshold` defaults to 1. At 2 or more, the manager keeps a `counts` OrderedDict as a second-hit admission filter (`prepare_store`, L261).
- **ARC.** `ARCCachePolicy` (`vllm/v1/kv_offload/cpu/policies/arc.py` L16) has T1 (seen once) and T2 (seen again), plus ghost lists B1 and B2. `_adapt_to_ghost_hit` (L78) raises the T1 target by max(1, |B2|/|B1|) on a B1 hit, and lowers it by max(1, |B1|/|B2|) on a B2 hit. `evict` (L150) takes from T1 if |T1| >= target, else from T2. New chunks always enter T1. `on_store_miss` (L92) runs once per request before the insert.
- **LRU.** `vllm/v1/kv_offload/cpu/policies/lru.py`.
- **Keys.** `OffloadKey = bytes`, the block hash plus a 4-byte group index (`vllm/v1/kv_offload/base.py`).
- **Async DMA.** `vllm/v1/kv_offload/cpu/gpu_worker.py`. Each transfer has its own stream and start and end events (`Transfer`, L69). Submit (about L675-716) orders compute before transfer with `stream.wait_stream(current_stream())`, chains FIFO with `stream.wait_event(last_transfer.end_event)`, records `start_event`, issues the batched copy and records `end_event`. `get_finished` (L735-757) polls the head of the queue with `end_event.query()`, in order, non-blocking, then recycles the stream and events. `wait` (L759) is the blocking `event.synchronize()`, used only for forced completion or preemption.
- **Ordering hazard named in source.** Loads must wait for the zeroing of destination blocks. With async scheduling, nothing else orders that zeroing against the copy (comment on the submit path).
- **Connector split.** `KVConnectorBase_V1` (`vllm/distributed/kv_transfer/kv_connector/v1/base.py` L181). The scheduler side decides: `get_num_new_matched_tokens` L516, `build_connector_meta` L595, `request_finished` L631, `has_pending_block_frees` L674. The worker side executes with per-layer waits: `start_load_kv` L333, `wait_for_layer_load` L350, `save_kv_layer` L364, `wait_for_save` L386, `get_finished` L398.

### Breccia mapping

- **What is offloaded.** GDN snapshot records and spilled QSA pages in the T2 pinned tier (round 15 C3). Experts are outside this policy. Expert residency covers 30 of 48 blocks, and the host lane carries 301 MiB per token (BRIEF L79).
- **Admission.** Round 16 R-2's second-hit admission is vLLM's `store_threshold` at 2 or more. vLLM bounds its tracker. R-2 should bound its tracker too, with a bound set from the snapshot count, not copied from vLLM's 64,000.
- **Eviction.** LRU is the default in both. ARC's ghost lists recognise a record that was evicted just before its second use. That is worth building only if replay shows LRU loses hits (Top 3, item 3).
- **Vulkan equivalent.** One timeline semaphore per copy queue. Each copy signals an increasing value. Completion is polled in submission order with `vkGetSemaphoreCounterValue`, which is non-blocking and follows the same head-of-queue rule as `get_finished`. Compute-to-transfer ordering is a timeline wait on the compute value. A forced wait uses `vkWaitSemaphores` with a bounded timeout, never `u64::MAX`, which matches the Breccia rule.
- **Hazard to add to R-3.** A GDN slot must be cleared on the same timeline before a restore copy writes it. Otherwise the restore can race the clear. Breccia's GDN slot is reused across requests, so the hazard applies here too.

**Delta vs R-3 and R-2.** R-3 specifies per-block events. This adds the reset-before-restore edge and the head-of-queue rule. Against R-2 it adds a tracker bound and the ARC option.

## Q4. Unified token-budget scheduling vs the R2 chunk sweep

### vLLM

- **One budget per step.** `Scheduler.schedule()` (`vllm/v1/core/sched/scheduler.py` L688) uses one token budget. The running loop runs first (`while req_index < len(self.running) and token_budget > 0`, L784), then the waiting requests. The clamp `num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)` is near L831-836. The budget is decremented near L991 and L1544.
- **Equal-progress share.** When several requests prefill, the per-request floor is `input_budget // num_eligible_reqs` (L743-757). The long-prefill threshold is disabled when only one request is eligible.
- **Mamba chunk clip.** `_mamba_block_aligned_split` (L535) ends each prefill chunk on a mamba-cacheable boundary. Its comment reads "State is written at chunk ends, so chunk ends must be block aligned." The clip applies at L850 (running) and L1340 (waiting).
- **Citation fix.** Phase-4 section 8 cites `scheduler.py:1311`, an older SHA. At the pinned SHA the waiting-chunk clamp is at L1335-1340 and the running loop is at L784.

### Breccia

- **R2 (round 14), the chunk-size sweep.** The flag is `--prefill-chunk` at `crates/breccia-cli/src/main.rs` L235 (default 0, token by token). The prefill entry is `crates/breccia-engine/src/graph.rs` L2102; the chunk scratch cap is `prefill_cap` at L4101.
- **Round 16 L92 (footnote).** Keep C = 64. Sweep {32, 64, 128, 256} only on the no-snapshot path until the exactness test passes. Unified token-budget scheduling does not override this: C stays 64, so G = lcm(C, 16) = 64 holds, and the sweep stays on the no-snapshot path.
- **Round 16 L88.** Interleaving needs per-stream positions first. `multi.rs` steps all streams at one shared position today.
- **Round 13 C4 and B10.** C4 sets one prefill chunk per round with round-robin decode. B10 names EQUAL_PROGRESS as the fair chunk policy.
- **Phase-4 section 2.** Merged chunked prefill is the Breccia-side lever these rules govern.

**Assessment.** The unified budget has one piece not already in rounds 13 to 16: the chunk-end clip onto the state grid, which is round 16 G, lcm(C, 16). Its fairness rule is round 13 C4 and B10. Breccia already decodes two streams together (22.8 to 23.7 aggregate tok/s, BRIEF). The open question is how prefill chunks interleave with that decode, and round 16 L88 and phase-4 section 2 govern it. The throughput effect of a shared budget is UNVERIFIED - requires the GPU qualification card. Not built now.

## Top 3

### 1. Sparse snapshot retention. Keep junctions and the tail. Drop periodic R1 snapshots once R-1 is live.

- **vLLM.** `MambaManager.reachable_block_mask` (`vllm/v1/core/single_type_kv_cache_manager.py` L1549), with `retention_interval` 0 versus None.
- **Breccia adaptation.** In the T2 tier, keep a snapshot record if it is a fork point (R-1), the prompt tail, or within a fixed count K of the latest. Drop periodic 16K records beyond K. Add a `retention` knob (dense versus reachable-only) to the planned `crates/breccia-kv` planner.
- **Expected effect.** Derived at `retention_interval = 0`: a linear 100K prefix goes from 6 snapshots (673 MiB, round 15) to 1 or 2 (112 to 224 MiB), a 3x to 6x cut in T2 bytes. A positive interval saves less. Cost: a partial-prefix hit between kept boundaries recomputes from the nearest kept state. Hit-token rate and recompute tokens: UNVERIFIED - requires trace replay. Restore cost: UNVERIFIED - requires the GPU qualification card.
- **Measurement plan.** Replay recorded agent prompt traces through the planner under dense, reachable-only and dense-plus-K policies. Report steady-state T2 bytes, hit tokens and recompute tokens on partial hits. Choose the smallest byte footprint whose recompute stays under a fraction of the prefill budget. Set that fraction before the run. Cheap-offline.
- **Delta vs rounds 13-16.** R1's periodic set is reduced, not extended. R-1's junction set is kept. New: the drop rule and its measured recompute cost.

### 2. 16-cell KV pages with a 64-token state grid. Not vLLM's 1,600-token coupling.

- **vLLM.** `resolve_kv_cache_block_sizes` (`vllm/v1/core/kv_cache_utils.py` L816; hash grain L885-891); `get_mamba_prefill_checkpoint_grid` (`vllm/model_executor/layers/mamba/abstract.py` L116-129); `_align_hybrid_block_size` (`vllm/platforms/interface.py` from about L940).
- **Breccia adaptation.** The planned `crates/breccia-kv` keeps QSA KV on 16-cell pages (`PAGE_CELLS`, `crates/breccia-kernels/src/qsa/mod.rs` L89) with the chained hash at 16. Fork points and the prompt tail are placed on the 64-token grid, G = lcm(C,16) with C at least 64. Periodic R1 snapshots stay at 16K.
- **Expected effect.** Derived: KV waste per stream falls from up to 39.3 MB (1,599 tokens x 24,576 B over 12 layers) to up to 0.37 MB. A restore from a fork-point or tail snapshot recomputes at most 63 tokens past its grid point, against up to 1,599 at vLLM's grid. A restore from a periodic 16K snapshot still recomputes up to 16K. TPS: none directly. Prefill and T2 hit rate: UNVERIFIED - requires trace replay.
- **Measurement plan.** (a) Host-only allocator test: bytes wasted per stream at 1, 4 and 16 concurrent streams, at both grains, over a range of prompt lengths (cheap-offline). (b) Exactness: a restore at a 64-token fork point is bit-identical to a full prefill, using round 16's exactness test (cheap-offline). (c) Hit tokens on the replay traces at grid 1,600 versus 64 (UNVERIFIED - requires trace replay).
- **Delta vs rounds 13-16.** Round 16 G = 64 already fixes the state grid, so the grid is not new here. New: the KV allocation grain stays at 16 cells rather than vLLM's 1,600, which bounds per-stream KV waste (derived above) and keeps the KV grain separate from the snapshot cadence (16K, R1). The gain is memory, not hit rate.

### 3. ARC ghost-list eviction for the host snapshot tier, gated on a replay win over LRU.

- **vLLM.** `ARCCachePolicy` (`vllm/v1/kv_offload/cpu/policies/arc.py` L16), `_adapt_to_ghost_hit` (L78), `evict` (L150). Selected by `cache_policy` in `vllm/v1/kv_offload/cpu/manager.py`. The default is LRU (`policies/lru.py`).
- **Breccia adaptation.** A Rust `EvictionPolicy` trait with LRU first. ARC is a second implementation whose ghost lists are bounded to the tier's capacity. Admission stays round 16 R-2's second-hit rule, with the bounded tracker from Q3.
- **Expected effect.** UNVERIFIED - requires trace replay: hit rate per GB, ARC versus LRU, at the T2 capacity. If replay does not clear a margin set in advance, LRU stays and ARC is not built. TPS effect: none directly. The effect is prefill avoided through more hits.
- **Measurement plan.** Cheap-offline replay on the same traces as item 1. Compare ARC and LRU at 1x, 2x and 4x the T2 capacity. Set the decision threshold before the run.
- **Delta vs rounds 13-16.** Round 15 C3 names the T2 tier and round 16 R-2 names admission. Neither specifies an eviction policy. This does.

## Rejected or folded

- **Unified token-budget scheduling** (`scheduler.py` L688, L784, L743-757). Folded into round 13 C4 and B10 and phase-4 section 2. Throughput effect UNVERIFIED. Not built.
- **Mamba chunk-end clip** (`_mamba_block_aligned_split`, `scheduler.py` L535). Already round 16 G.
- **Unify fallback as the hybrid path** (`unify_kv_cache_spec_page_size`, branches (b) and (c)). Hybrid models do not reach it after pre-alignment. Breccia does not need it.
- **Offload-side zeroing hazard** (`gpu_worker.py` submit path). Folded into round 16 R-3 as the reset-before-restore edge.

## Topics in the brief not re-read at source this round

- **Cascade attention.** Located, not read: the merge kernel is `vllm/v1/attention/ops/merge_attn_states.py`, and its cascade use was not read. It does not transfer at Breccia's two-stream batch. Cascade pays off when many co-batched requests read one prefix, and the shared-prefix saving is already taken by the prefix cache (Q2).
- **FP8 KV.** Not read. To my knowledge gfx1030 (RX 6950 XT) and gfx906 (MI50) have no native FP8 arithmetic. UNVERIFIED against the Vulkan extension list on the target GPUs. FP8 KV stores 8 bits per element, the same bytes as q8_0, so it is a different decode path, not a capacity gain. Docs 09 and 10 cover q8_0 KV with dequant in attention and the exllamav2 KV-quant choices.
- **MoE expert parallel and offload.** Not read; the EP and EPLB paths are located under `vllm/distributed/eplb/`. EP shards experts across devices with all-to-all dispatch. Breccia's expert residency covers 30 of 48 blocks, with 301 MiB per token on the host lane (BRIEF L79). The hybrid MoE questions are in doc 14. EP has no counterpart here.
- **Speculative decode.** Drafters located (`vllm/v1/spec_decode/eagle.py`, `vllm/v1/spec_decode/ngram_proposer.py`, `vllm/v1/spec_decode/suffix_decoding.py`). Not re-costed. Breccia's MTP at 0.60x and the suffix drafter are in phase-4 section 6 (suffix drafter at L255). The verify width competes with the two-stream decode batch.

## Unverified or not checked

- The GDN backend enum in vLLM (the `mamba_type` for Qwen3-Next's GDN) was not checked. The FlashInfer 64 alignment applies only on that path; its condition at `qwen_gdn_linear_attn.py` L696 was not re-read.
- `partial_hash_hits_enabled` (`vllm/v1/core/kv_cache_utils.py` L762): body not read; its call sites (L909, L943) were. It decides whether a 16-token hash gives 16-token hits.
- `resolve_cache_hit_alignment_tokens` was read for its docstring only.
- `_align_hybrid_block_size` was read through its mamba padding step. Its tail was not read.
- Hit rates for R1, R-1 and ARC: UNVERIFIED - requires trace replay.
- `kv_cache_manager.py` (`get_computed_blocks` L266, `allocate_slots` L379) was located, not read in full.
- `tiering/manager.py` was not read.
- No GPU numbers appear in this doc. Every GPU figure is labelled UNVERIFIED.
