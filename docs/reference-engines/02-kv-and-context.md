# Reference engines 02 — KV cache, context tiers, prefix reuse, spec scheduling

READ-ONLY survey of the cloned references under `Reference Projects/` for Breccia **M5**
(tiered paged KV: VRAM → pinned RAM → NVMe, radix prefix reuse, compressed KV body with FP16
tail, sliding/sparse variants, read-only injected context blocks) and **M6-style scheduling**
(continuous batching, agent slots), with M4 (MTP/speculative) in view.

Nothing in `Reference Projects/` was modified or committed. Repos were read as clones; git state
is `llama.cpp@6050b46`, `beellama.cpp@ce1a999` (v0.4.8), `TurboQuant@570f5e7` (archived),
`vllm@237d2ff`, `sglang@8252e3a`, `ik_llama.cpp@e34d163`, `mistral.rs@9459891`, `exo@bc412e2`,
`IsoQuant@5f8bd65`, `RotorQuant@fcd7676`.

> ⚠️ **`IsoQuant/` and `RotorQuant/` carry NO LICENSE file** (verified: no `LICENSE`/`COPYING`
> in either tree). Everything taken from them below is **algorithm understanding only, no code
> copying** — no structs, no kernels, no transcription of their source. Same rule applies to
> their `paper/` PDFs. All other repos: llama.cpp / beellama.cpp / TurboQuant / ik_llama.cpp /
> mistral.rs = MIT; vllm / sglang / exo = Apache-2.0.

---

## 1. Comparison table

| Engine | KV paging | KV quant | Prefix reuse | Offload tiers | Spec decode | License | Best idea to steal |
|---|---|---|---|---|---|---|---|
| **llama.cpp** (mainline) | 1-token **cells** in a ring buffer per stream (`find_slot`, `llama_kv_cache`); no fixed page; cells shared by seq via `bitset` refcount (`llama_kv_cells`) | `--cache-type-k/v`: f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1 (`common/arg.cpp:304`); **Hadamard `attn_rot`** auto-enabled for quantized K/V (`src/llama-kv-cache.cpp:326-356`, PR #21038 refs at :348/:356); WHT kernels `ggml/src/ggml-cuda/fwht.cu` | `--cache-prompt` + longest common prefix; `--cache-reuse N` **KV-shift** of matching chunks (`tools/server/server-context.cpp:4024-4070`); per-cell seq bitsets = shareable prefixes | Slot save/restore to file: `--slot-save-path`, `/slots/{save,restore}` (`server-context.cpp:6633/6667`); no host-RAM tier | `--spec-draft-type-k/v` separate draft cache (`arg.cpp:4139`), ngram/EAGLE/DFlash | MIT | **Cell-level seq bitset + `state_write/read` split** (`llama-kv-cache.h:349-354`): metadata and data saved separately → cheap snapshot/restore of one sequence out of a shared cache |
| **beellama.cpp** | same cell model as llama.cpp, plus **group descriptor staging**: `KVAR_N_GROUP = 128` records, `llama_kvarn_live_stage_groups` / `llama_kvarn_reconcile_stage_slots` (`src/llama-kv-cache.cpp:2493-2635`) | **KVarN** kvarn2/3/4/5/6/8: per-head Hadamard after RoPE + normalise both axes of each 128-token tile + structured records (`docs/beellama-features.md:11-24`); ops `GGML_OP_KVARN_{WHT,STORE,VIEW,MATERIALIZE}` (`ggml/include/ggml.h:623-626`); CUDA `kvarn.cu` (3198 L), `kvarn-wht.cu`, `fattn-mma-kvarn-*.cuh` | KVarN durable prompt cache is **descriptor-group aligned** — checkpoints only at complete 128-token boundaries (`docs/beellama-features.md:57-60`) | no host tier, but **precision tail** is a second physical pool (exact F16 stage slots, `src/llama-kvarn.h:16-46` encodes `cell + stage_slot` in one int64 index) | `--kv-tail` applies to draft caches too; **adaptive draft-max** `common/speculative.cpp:1097,1432,2603-2621` (grow/shrink draft on acceptance) | MIT | **The precision tail**: recent + first-128 sink tokens stored exact F16 in a separate stage buffer while the body stays compressed, addressed by one index word (`llama-kvarn.h:16-32`) |
| **TurboQuant** (llama.cpp fork) | unmodified llama.cpp cells; TBQ types plug in as `ggml_type` of the existing `k/v` tensors | `tbq3/tbq4/tbqp3/tbqp4/amx3` shorthands resolve to `_0/_1/_2/_4` by head_dim (`common/arg.cpp:411-459`); block structs `block_tbq3_0` (QK_K=256), `block_tbq3_1` (TBQ_K128), `block_amx3_1` (`ggml/src/ggml-common.h:285-394`); encoders in `ggml-cuda/set-rows.cu:584-635`, dequant+IWHT in `ggml-cuda/convert.cu:702+` | same as llama.cpp (`--cache-prompt`) | none | TriAttention token pruning (`ggml-cuda/triattention.cu`) evicts ~95% of KV slots per layer | MIT | **K/V asymmetric pairing + internal-vs-external rotation rule**: every `TBQ*` encoder carries its own WHT, so the external `attn_rot` is force-disabled (`src/llama-kv-cache.cpp:320-330`) — one rotation owner only |
| **IsoQuant / RotorQuant** | n/a (prototypes) | **algorithm only**: block-diagonal rotations — 2D Givens / 4D quaternion SO(4) / Clifford rotor — instead of a dense or butterfly WHT; O(d) instead of O(d·log d) | n/a | n/a | n/a | **NO LICENSE → algorithm understanding only, no code copying** | The *claim to verify ourselves*: small block rotations match or beat global WHT on KV PPL at a fraction of the transform cost (README tables) — worth an A/B harness, not a copy |
| **vllm** | **fixed-size block tables**: `KVCacheBlock(block_id, ref_cnt, block_hash, free list)` (`vllm/v1/core/kv_cache_utils.py:158-218`), `FreeKVCacheBlockQueue`, worker-side `vllm/v1/worker/gpu/block_table.py` | **TurboQuant backend** `vllm/v1/attention/backends/turboquant_attn.py` — cache layout `(num_blocks, block_size, num_kv_heads, slot_size)` with per-slot `[key_packed \| value]` (docstring :5-17); presets `turboquant_k8v4`, `_4bit_nc`, `_k3v4_nc`, `_3bit_nc` (`quantization/turboquant/config.py:16-35`); registered as a *kv_cache_dtype* (`vllm/config/cache.py:54,111`) | radix-style coordinator: `KVCacheCoordinator.find_longest_cache_hit` over **block hashes**, `KVCacheManager.get_computed_blocks` / `allocate_slots` (`v1/core/kv_cache_manager.py:264,378`; coordinator classes at `kv_cache_coordinator.py:89,476,524,609`) | **`kv_offload/`**: `CPUOffloadingSpec` (pinned host) and `TieringOffloadingSpec` (CPU primary + N secondary tiers, all secondary traffic staged through CPU), chunks + async DMA (`docs/features/kv_offloading_usage.md`, `vllm/v1/kv_offload/{cpu,tiering}`) | continuous batching in `v1/core/sched/scheduler.py:677 schedule()`; **preemption = recompute only** (`_preempt_request` :1754 frees blocks, no swap in v1); `allocate_slots(num_lookahead_tokens=…)` reserves for proposers | Apache-2.0 | **The offload tiering contract**: one *primary* tier that is GPU-addressable, secondary tiers only via the primary, chunk = N blocks, async DMA, per-request `max_load_tokens` cap |
| **sglang** | paged pools + per-layer hybrid routing: `allocation → hybrid_cache/ → allocator/ → pool(L1) → pool_host(L2) → storage(L3)` (`mem_cache/README.md`); **QSA pool for Qwen4-Exp** `mem_cache/qsa_kv_pool.py:48` (compressed index-K, `compressed_slot = full_slot // ratio`, page must be a multiple of compress ratio :51-54) | `kv_cache_dtype`, `--qsa-indexer-dtype {bfloat16,fp8_e4m3}` (`qsa_kv_pool.py:37-46`) | **RadixAttention**: `RadixCache`/`TreeNode{lock_ref,last_access_time,hash_value,priority}` (`mem_cache/radix_cache.py:250,552-619`), pluggable eviction `evict_policy.py` (LRU / T-LRU / S-LRU) | L2 host mirror + write-back: `unified_radix_cache.py:453 init_hicache`, `:1976 write_backup_storage`, `HiCacheController` (`managers/cache_controller.py:295`), `pool_host/{mha,qsa,mamba,unified}.py` (pinned); L3 `storage/` backends | `speculative/adaptive_spec_params.py` (runtime draft-step ladder by measured acceptance), `frozen_kv_mtp_*` MTP workers, EAGLE v2; retraction backup `unified_radix_cache.py:1547 backup_kv_cache / :1587 restore_kv_cache` | Apache-2.0 | **Two things**: (1) the `mem_cache` layer stack with the rule *"layers do not import upwards"*, (2) **`mem_cache/rust_tree_core/`** — sglang already ships a Rust radix-tree core we can study in our own language |
| **ik_llama.cpp** | llama.cpp cell model; **per-layer-group cache types** `-ctk-first TYPE,N` / `-ctk-last TYPE,N` applied in `src/llama.cpp:1584-1588`; indexer cache has its own type `-ictk` (`common/common.cpp:2048`) | `kv_cache_type_from_str` incl. `q6_0` and **`q8_KV`** (`common/common.cpp:4545-4580`; `GGML_TYPE_Q8_KV = 151`, `Q8_KV_R8` in `ggml/include/ggml.h:454,498`); low-ppl `Q4_0` KV (README:152, PR 1547/1556) | same as llama.cpp | **`-nkvo/--no-kv-offload` keeps KV on CPU** (`common/common.cpp:2034`, `docs/parameters.md:218`) → hybrid GPU/CPU KV path; `-dple` defers the Qwen3.8-Flash-Next PLE table (`docs/parameters.md:119`) | `--cache-type-k-draft` (`common/common.cpp:2102`) | MIT | **Mixed KV precision by layer group** — give the layers that matter more bits and the rest fewer, at the *cache allocation* level, not the model level |
| **mistral.rs** | real PagedAttention: `paged_attention/{kv_cache_manager.rs:45,280 allocate_slots, block_pool.rs:555-626, block_hash.rs, cache_engine.rs, scheduler.rs}` + kernel crate `mistralrs-paged-attn` | `PagedCacheType {Auto, F8E4M3}` (`cache_engine.rs:47`); non-paged `kv_cache/{single,rotating,full,hybrid}_cache.rs` (hybrid = GDN/recurrent + attention, i.e. **our exact model shape**) | `prefix_cacher.rs` + `block_pool` `PrefixBlockRetention(Lease)`, block hashes incl. multimodal spans (`block_hash.rs`) | device-only; CPU only for the non-paged `cpu_kv_f16()` path (`kv_cache/mod.rs:71`) | `speculative/{builtin_mtp.rs, proposer.rs, verifier.rs, driver.rs, policy.rs, autotuner.rs}` — **`builtin_mtp.rs` explicitly targets Qwen4-Exp** | MIT | **Crate/module layout** (see §10) + `memory_plan.rs` `MemoryGoal{Utilization,KvBytes,KvTokens}` with reserve fractions |
| **exo** | — (brief only) | — | — | — | — | Apache-2.0 | For a future **Tier-4 ring**: `--instance-meta ring` node discovery + `--sharding pipeline\|tensor` (`README.md:27,558-561`); no KV-tier lessons |

Two findings that dominate everything below:

1. **Both sglang and mistral.rs already implement Qwen4-Exp.** `sglang/python/sglang/srt/models/qwen4_exp.py`
   (2669 lines, indexer + PLE + QSA) with `layers/attention/qsa/{config,qsa_indexer,sparse_attn,metadata}.py`
   and `mem_cache/{qsa_kv_pool,ple_state_pool}.py`; `mistralrs-core/src/vision_models/qwen4_exp/{qsa,ple,mtp,hyper}.rs`
   and `speculative/builtin_mtp.rs`. These are the closest existing references to *our* model and should be
   read before writing any QSA KV code.
2. **Our KV is unusually wide**: GQA-2 with head_dim 256 over 12 attention layers ⇒ 24 KiB/token at f16
   (§9), so KV is a first-class budget item — but the **indexer side-cache at f32 is another 12 KiB/token**
   (`docs/specs/qwen4exp-architecture.md` §12.2), i.e. half as expensive again. Any tier design that moves
   KV must move the indexer cache in lock-step (cell-for-cell, `qwen4exp.cpp:414-418`).

---

## 2. llama.cpp (mainline)

### 2.1 Cache structure — cells, not pages

`class llama_kv_cache : public llama_memory_i` (`src/llama-kv-cache.h:20`) allocates, **per layer**,
`k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)` (`src/llama-kv-cache.cpp:233`),
i.e. a ring of `kv_size` rows of `n_embd_k_gqa = n_head_kv × head_dim` elements, one such tensor per
layer, sliced into `n_stream` streams. Key entry points:

* `find_slot(const llama_ubatch&, bool cont)` — `src/llama-kv-cache.cpp:921`, declared `:216`. Reserves
  cells for the incoming ubatch; `slot_info` records stream/idx mapping consumed by `set_input_k_idxs`
  (`:1474` region) and `set_input_kq_mask` (`:1770+`).
* `llama_kv_cells` (`src/llama-kv-cells.h:30+`): per-cell `pos`, `ext` (2-D M-RoPE position, token),
  `shift` (for K-shift), and **`seq_set_t = std::bitset<LLAMA_MAX_SEQ>`** — a cell may belong to many
  sequences at once, which is what makes prefix sharing across slots free.
* Save/restore is split into **metadata** and **data**: `state_write_meta/state_write_data`,
  `state_read_meta/state_read_data` (`src/llama-kv-cache.h:349-354`), plus `has_cell_ext()` gate.
  A caller can persist one sequence's cells out of a shared cache without touching the rest.

Hybrid models (our shape) go through `llama_memory_hybrid` (`src/llama-memory-hybrid.h:14-45`), whose
constructor takes `type_k/type_v/v_trans/kv_size/n_pad/n_swa/swa_type` **and** `type_r/type_s/rs_size`
for the recurrent half, plus `offload` and `unified` flags and per-layer filter callbacks
(`filter_attn`/`filter_recr`). The Qwen4-Exp indexer side-cache lives in
`src/llama-memory-hybrid-idx.{h,cpp}`.

### 2.2 Quantized KV + the WHT rotation lineage

`kv_cache_types` (`common/arg.cpp:304-314`) = f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1;
parsed by `kv_cache_type_from_str` (`:316`) and consumed by `-ctk/--cache-type-k` (`:2449`) and
`-ctv/--cache-type-v` (`:2467`), with a separate draft pair `-ctkd/-ctvd` (`:4139,:4157`).

Rotation (the PR #21038 lineage) is decided in the `llama_kv_cache` constructor
(`src/llama-kv-cache.cpp:326-372`):

```cpp
bool attn_rot_k = !attn_rot_disable && n_embd_head_k_all > 0 &&
                  ggml_is_quantized(type_k) && hparams.n_embd_head_k() % 64 == 0;
if (attn_rot_k) { n_rot_k = 64; while (n_embd_head_k_all % (2*n_rot_k) == 0) n_rot_k *= 2; }
// n_rot_v stays 64 — "using smaller rotation matrices for V seems beneficial"
//   ref: PR #21038#issuecomment-4146397570   (K: ...4141323088)
```

then Hadamard matrices of every needed size are precomputed **on the host** (`:376-396`,
`ggml_gen_hadamard`) and fed into the graph as inputs `attn_inp_k_rot` / `attn_inp_v_rot`
(`build_input_k_rot` :1474 / `build_input_v_rot` :1486). The graph applies them right before the K/V
write: `q_cur/k_cur = llama_mul_mat_hadamard(…, inp->self_k_rot)` (`src/llama-graph.cpp:3237-3242`).
Gates: env `LLAMA_ATTN_ROT_DISABLE`, API `llama_model_set_attn_rot` (`src/llama-model.cpp:3844`).
Backend kernels: `ggml/src/ggml-cuda/fwht.cu` / `fwht.cuh`, `ggml/src/ggml-sycl/fwht.cpp`.

Note the rotation is applied to **K and V (post-RoPE)**, only when the cache type is quantized, and
only when head_dim is a multiple of 64 — the design is "rotate, then quantize, then rotate back at
read time" rather than a model-level transform.

### 2.3 Prefix reuse and slot persistence

* `--cache-prompt/--no-cache-prompt` (`common/arg.cpp:3622`) — per-request prompt caching.
* Server: `n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`
  (`tools/server/server-context.cpp:3993`), then `--cache-reuse N` (`common/arg.cpp:3631`) searches
  *below* the common prefix for matching chunks of ≥ N tokens and **moves them with a KV shift**:
  `slot.mem.seq_rm(...)` + `slot.mem.seq_add(id, head_c, head_c+n_match, kv_shift)`
  (`server-context.cpp:4041-4056`). The shift is executed as a graph (`build_graph_shift`,
  `src/llama-kv-cache.cpp:2022`) that re-applies RoPE with `k_shift` (`set_input_k_shift` :1543).
  *Borrowable idea*: reuse is a **metadata operation** (positions change, bytes do not move) as long
  as positions stay within RoPE-shiftable range.
* Slot save/restore: `--slot-save-path` (`common/arg.cpp:3666`), routes
  `handle_slots_save`/`handle_slots_restore` (`server-context.cpp:6633`, `:6667`) writing to
  `params.slot_save_path + filename` (`:6638`, `:6674`).
* Slots: `-np/--parallel N` = "number of server slots" (`common/arg.cpp:2556`), `-ns/--sequences`
  (`:2567`); `-fa/--flash-attn` (`:1766`); `-sm/--split-mode {none,layer,row,tensor}` (`:2829`).

---

## 3. beellama.cpp — KVarN + precision tail + adaptive draft

### 3.1 KVarN (algorithm shape)

Documented in `docs/beellama-features.md:7-34` and `docs/beellama-args.md:9-45`. Per the repo's own
description: *"applies a per-head Hadamard rotation after RoPE, normalizes both axes of each 128-token
tile, and stores structured 2/3/4/5/6/8-bit records with scale metadata."* Calibration-free (no
imatrix), K and V widths independent (`kvarn2…kvarn8`, one-sided selection promotes the other side,
`docs/beellama-args.md:20-27`), SWA layers can carry a *different* pair (`--cache-type-k-swa`).

Code map:

* API surface: `ggml/include/ggml.h:2776-2808` — `ggml_kvarn_wht`, `ggml_kvarn_store`,
  `ggml_kvarn_view`, `ggml_kvarn_materialize`; ops `GGML_OP_KVARN_*` (`:623-626`); flash-attn
  domain params `GGML_FLASH_ATTN_EXT_KVARN_DOMAIN_{AUTO,ROTATED,ORIGINAL,ROTATED_K_ORIGINAL_V}`
  (`:469-480`) and `…_WINDOW_CHUNK` (`:472`) — i.e. the attention kernel can read the compressed
  record directly, or a materialized rotated/original view.
* Geometry: `KVAR_N_GROUP = 128` (`src/llama-kvarn.h:10`), CUDA `KVAR_N_DIM = 128`
  (`ggml/src/ggml-cuda/kvarn.cu:12`), dynamic stage/tail depth in `op_params`
  `KVAR_N_OP_PARAM_STAGE_GROUPS` / `…_TAIL_GROUPS` (`kvarn.cu:32-33`).
* Kernels: `ggml/src/ggml-cuda/kvarn.cu` (3198 lines — store/materialize), `kvarn-wht.cu`,
  `fattn-kvarn-dispatch.cu` (1340 lines), `fattn-mma-kvarn-{load,impl,decode,window}-*.cuh`,
  `fattn-tail.cuh`, portable fallback `fattn-kvarn-portable.cuh`.
* Cache integration: `src/llama-kv-cache.cpp:2493-2635` calls
  `llama_kvarn_live_stage_groups` / `llama_kvarn_reconcile_stage_slots` /
  `llama_kvarn_group_owner_compatible` — staging is reconciled per *group*, not per token.

### 3.2 The precision tail (the piece to steal)

Args (`docs/beellama-args.md:72-95`):

| flag | default | meaning |
|---|---|---|
| `--kv-tail-tokens SPEC` | `0` → KVarN keeps its **intrinsic 128-token exact suffix**; `auto` = 1024 exact tokens | exact F16/BF16 window at the newest end; numeric values **round up to complete 128-token groups**; `N0,N1` per canonical group; `full=N,swa=N` role syntax |
| `--kv-tail-type f16\|bf16` | `bf16` standard / `f16` KVarN | dtype of the tail |

Mechanics:

* Params plumbed in `src/llama-cparams.h:68-75`: `kv_tail_tokens`, `kv_tail_tokens_swa`,
  `…_requested`, `kv_tail_native_exact`, `kv_tail_rollback_tokens`, `kv_tail_type`.
* The tail is a **separate physical stage**: `llama_kvarn_encode_store_cell(cell, stage_slot)` packs
  `{stage_slot+1 << 32 | cell}` into the single int64 row index used by the kernels
  (`src/llama-kvarn.h:16-30`), so the *same* index tensor addresses "compressed body record" or
  "exact F16 staging slot" — `kvarn_read_cell()` branches on `encoded < -1`
  (`ggml/src/ggml-cuda/kvarn.cu:37-51`).
* Graph routing: `llm_graph_kv_tail_identity::capture` (`src/llama-graph.h:350-364`) + per-layer
  `llama_kv_tail_route`; native-tail layers get `fattn-tail.cuh`, others fall back to a generic
  route (which is several times more expensive — hence `LLAMA_KV_TAIL_ALLOW_GENERIC`).
* Additional policy: **first 128 attention-sink tokens stay exact too**
  (`docs/beellama-features.md:16-21`), and the tail never splits sealed compressed history.
* Tail also enables rollback: `kv_tail_rollback_tokens` + the "live exact frontier" support the
  speculative micro-rollback path (`docs/beellama-features.md:57-61`).

### 3.3 Adaptive draft control

`common/speculative.cpp:1097,1432,2603-2621` — `adaptive_dm_supported()` per implementation
(draft-simple / EAGLE3 / DFlash) and `common_speculative_adaptive_dm_supported()`; DFlash2 falls back
to its fixed block limit + selector confidence when adaptive draft-max is unsupported. KVarN is
allowed on draft caches for the qualified list (`docs/beellama-args.md:27-45`), with a smaller
default prefill materialization window (`--spec-draft-kvarn-window-chunk 2048` vs `65536` target) —
*because a shallow draft's transient F16 workspace can otherwise outweigh its persistent saving.*
That is a good sizing rule for our MTP head (§12.4 of our spec: 32768-cell draft window = 64 MiB).

---

## 4. TurboQuant (llama.cpp fork)

Repo layout: the fork's *modified* upstream files are mirrored in `turboquant/modified_files/`
(diff source), while the applied tree lives at the repo root. Docs: `turboquant/README.md` /
`modified_files/README.md` (project is marked discontinued at `570f5e7`).

* **Types**: `GGML_TYPE_TBQ3_0 = 42` … `GGML_TYPE_AMX3_1 = 62`, `AMXV3_1 = 63`
  (`ggml/include/ggml.h:432-455`; the root copy shows the enum was later renumbered to 65 to free
  a slot — enum values are not stable, names are).
* **Block records** (`ggml/src/ggml-common.h:285-394`): `block_tbq3_0` = fp16 scale + `QK_K*3/8`
  bytes (256-element block), `block_tbq3_1` = scale + `TBQ_K128*3/8` (128-element block),
  `block_tbq3_2` = 64-element block, `block_tbqp3_*` = 2-bit Lloyd-Max + 1-bit QJL signs,
  `block_amx3_1` = 108 B hybrid (Part A WHT 50 B for FA + Part B polar 58 B for scoring).
* **CLI** (`common/arg.cpp:388-459`): shorthands `tbq3/tbq4/tbqp3/tbqp4/amx3` are resolved to the
  right `_N` variant **by head_dim later**, and `amx3` maps to a *different* enum for K vs V
  (`-ctk` → `AMX3_1`, `-ctv` → `AMXV3_1`). Display names collapse the suffixes.
* **Integration into the KV layout**: they are ordinary `ggml_type`s of the existing
  `k`/`v` tensors, so layout is unchanged (`[n_embd_k_gqa, kv_size, n_stream]`) and only the row
  write/read path changes:
  * quantize on store: `xhead_quantize_tbq3/4`, `xhead_quantize_tbqp3/4`
    (`ggml/src/ggml-cuda/set-rows.cu:584-635`);
  * dequant on load: `dequantize_tbq_4_spatial_f16_kernel`, `dequantize_tbqp_4_wht_f16_kernel`,
    plus `tbq_q_wht` / `tbq_output_iwht` kernels (`ggml/src/ggml-cuda/convert.cu:702-935`);
  * hard gating by head_dim (`src/llama-kv-cache.cpp:109-123`): e.g. `tbqv3_1` only supports
    head_dim = 128 and is V-only;
  * rotation ownership (`src/llama-kv-cache.cpp:320-330`): *"All TBQ/TBQP/AMX encoders carry their
    own internal WHT … external rotation OFF for every TBQ\* type."*
* **TriAttention** (`ggml/src/ggml-cuda/triattention.cu`): periodic (default every 128 tokens)
  per-layer top-B slot retention with `--tri-keep-first` attention sink — a *token-level* eviction
  axis on top of byte-level compression. Numbers in `turboquant/README.md` (2.37× packing, ~4.7×
  effective).

**Take-away for us**: for head_dim = 256 our `_0` (256-element) block is the natural granularity, and
`k8v4`-style asymmetry (better K than V) is what vLLM shipped too (§5).

---

## 5. IsoQuant / RotorQuant — algorithm understanding only, no code copying

**License: none. Do not copy any code, structs, or file contents. The following is a summary of the
published algorithm idea only, for the purpose of deciding whether to re-implement and benchmark it
independently.**

* **RotorQuant** (`RotorQuant/README.md`, `paper/rotorquant.pdf`): replace TurboQuant's dense/butterfly
  WHT with **block-diagonal rotations** — Clifford Cl(3,0) rotor sandwich (group size 3), later reduced
  to 2D Givens rotations ("PlanarQuant") and 4D quaternion rotations ("IsoQuant"). Claim: O(d) FMAs
  vs O(d·log d), ~128 parameters instead of 16 384, and *better* KV PPL because small orthogonal
  blocks preserve directional structure that global scrambling destroys.
* **IsoQuant** (`IsoQuant/README.md`, `paper/`): 4D blocks, `SO(4)` rotations parameterised by one or
  two unit quaternions (`v → q_L v q_R*` full, `v → q_L v` fast, plus a 2D planar special case); the
  repo ships only stage-1 quantize-dequantize CUDA kernels and explicitly says end-to-end KV quality
  is *not yet validated* (`IsoQuant/README.md` "Current Status").
* Relevance to us: our head_dim = 256 divides cleanly by 4 and by 2, so a block-diagonal rotation is
  trivially tileable; and because our attention layers are only 12 and wide (256), the transform cost
  per store is small. **Action for M5**: put global-WHT vs block-rotation behind one trait and A/B
  them on our own harness; do not import their source.

---

## 6. vllm

### 6.1 Block tables and prefix caching

* `KVCacheBlock` (`vllm/v1/core/kv_cache_utils.py:158-218`): `block_id`, **`ref_cnt`**, `pool`,
  `_block_hash` + `_block_hash_num_tokens`, doubly-linked free-list links, `is_null` (used to trim a
  sliding-window table's skipped prefix — `null_head_len`/`trim_null_head` :225-250). This is the
  whole state model: **id + refcount + hash + free queue**.
* `KVCacheManager.get_computed_blocks` (`v1/core/kv_cache_manager.py:264`) →
  `coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)`; note it caps the
  hit at `prompt_length - 1` because the last token must be recomputed for logits.
* `allocate_slots(...)` (`:378`) is the allocator for one step and its docstring is the best
  spec-of-record for the block layout:
  `| <comp> | <new_comp> | <ext_comp> | <new> | <lookahead> |` with
  `num_lookahead_tokens` (spec proposers), `num_external_computed_tokens` (connector-provided KV),
  `reserved_blocks` (admission gate so async loads can't starve in-flight sequences),
  `full_sequence_must_fit` (admission control).
* Coordinator hierarchy (`v1/core/kv_cache_coordinator.py`): abstract `KVCacheCoordinator:89`,
  `…NoPrefixCache:476`, `UnitaryKVCacheCoordinator:524`, `HybridKVCacheCoordinator:609` (multiple KV
  groups — SWA/full/Mamba — must agree on which groups participate in hashing).

### 6.2 TurboQuant attention backend

`vllm/v1/attention/backends/turboquant_attn.py` (registered `registry.py:162`,
`--kv-cache-dtype turboquant_k8v4` accepted via `vllm/config/cache.py:54,111`):

* Cache layout (docstring :5-17): `(num_blocks, block_size, num_kv_heads, slot_size)` where
  `slot_size = key_packed_size + value_packed_size`; per-head per-position slot =
  `[key_packed | value]`. For `turboquant_k3v4_nc` with head_dim 256 the doc quotes
  `[100 B key | 512 B value] = 612 B` (≈3.125 bpw K).
* Prefill runs plain SDPA on uncompressed K/V, quantizes, and stores; decode scores from the packed
  cache (`triton_turboquant_store.py`, `triton_turboquant_decode.py`, SoA variants
  `ops/turboquant_soa/*`, plus a FlyDSL path for AMD gfx950 `ops/flydsl_turboquant_decode.py`).
* Rotation: `_build_hadamard(d)` Sylvester matrix, *"shared across all layers: random sign … Pure
  Hadamard: orthonormal + symmetric (H = H^T)"* (`:105-116`, `:412`, `:458`) — matmul-based WHT as a
  single GEMM rather than a butterfly.
* Presets (`quantization/turboquant/config.py:16-35`): `turboquant_k8v4` (K 8-bit / V 4-bit, no norm
  correction), `turboquant_4bit_nc`, `_k3v4_nc`, `_3bit_nc`; the config docstring credits the pattern
  to DRIVE/EDEN and notes equivalence to scalar HIGGS.

### 6.3 Scheduler: continuous batching + preemption

* `schedule()` (`v1/core/sched/scheduler.py:677`) runs each step over `self.running` (:232) and a
  waiting queue with `deferred_waiting` (:231); admission honours watermark/`reserved_blocks`.
* Overload → **preempt the newest running request** (`:911-957`, victim chosen from the tail of
  `running`), then `_preempt_request` (`:1754`) frees its blocks, encoder cache and in-flight
  prefills and pushes it back to waiting. **v1 has no swap path** — grep for `swap` in the scheduler
  returns nothing; preemption = free + recompute.
* Optional connectors can take over the hand-off (`aux_output_connector.request_finished`,
  `drop_stale_output` for pending KV hand-offs).

### 6.4 Offload tiers (`vllm/v1/kv_offload/`)

`docs/features/kv_offloading_usage.md` is the spec:

* `OffloadingConnector` extends the prefix cache: completed GPU blocks → tier, hits promoted back.
* Two specs: `CPUOffloadingSpec` (single CPU tier, **pinned** host memory) and `TieringOffloadingSpec`
  (CPU primary + N secondary tiers; **only the CPU tier is GPU-addressable — every secondary transfer
  is staged through it**).
* Unit of work = **chunk** = `blocks_per_chunk` accelerator blocks (bigger = larger I/O).
* Transfers are async DMA alongside compute; per-request `max_load_tokens` / `max_offload_tokens`
  caps; only CUDA/ROCm/XPU supported.
* Code: `kv_offload/cpu/{manager,host_register,swap_blocks_triton,policies}`,
  `kv_offload/tiering/{factory,manager,fs,obj,p2p,kvcr,backpressure}`.

---

## 7. sglang

### 7.1 The layer stack (read `mem_cache/README.md` first)

```
scheduler / model_runner / attention backend
            |
      allocation.py            per-batch allocation policy (batch -> out_cache_loc)
            |
      hybrid_cache/            multi-pool router (layer_id -> pool)
            |
      allocator/               "give me N slots" (need_size -> indices)
            |
      pool/ (device,L1) --hicache--> pool_host/ (host,L2) --> storage/ (L3)
```

with the rule **"Layers do not import upwards"**. Two axes sit outside: the **radix cache** (what to
keep) and **construction** (`kv_cache_configurator.py`, `kv_cache_builder.py`, `allocation_sizing.py`,
`kv_cache_dtype.py`). This is the cleanest reference architecture for our own crate split.

### 7.2 RadixAttention

`mem_cache/radix_cache.py`:

* `RadixKey` (:52) — token ids + `extra_key`/`cache_salt` (namespacing) + optional `limit` (sliced
  view without copying) + bigram mode.
* `TreeNode` (:250) — `children`, `parent`, `key`, `value` (page-aligned KV rows), **`lock_ref`**,
  `last_access_time`, `creation_time`, `hit_count`, `hash_value` (per-page hashes), `priority`.
* `evict()` (:552) — heap of evictable leaves ordered by `eviction_strategy.get_priority(node)`;
  pops until enough tokens are freed; pushes newly-childless unlocked parents.
* `inc_lock_ref` / `dec_lock_ref` (:582, :597) walk root→node, moving tokens between
  `evictable_size_` and `protected_size_`.
* `evict_policy.py`: `LRUStrategy:16`, `TLRUStrategy:49` (tail-optimised LRU, arXiv:2510.15152),
  `SLRUStrategy:84`, plus priority-aware eviction — **the eviction policy is a strategy object, not
  a hardcoded branch.**
* `rust_tree_core/` — Rust implementation of the tree with `adapter.py`/`extension.py` shims: the
  closest thing to a radix tree we can read in our own language.

### 7.3 Host tier (HiCache) and retraction

* L2 host mirror: `pool_host/{base,common,mha,qsa,mamba,mla,unified,page_unified}.py` — pinned host
  tensors (`HostTensorAllocator`), `mem_cache/README.md` lines 58-76; `HiCacheController`
  (`managers/cache_controller.py:295`) orchestrates H2D/D2H.
* Write-back of prefixes: `unified_radix_cache.py:453 init_hicache`, `:1976 write_backup_storage`,
  `:3398 _sync_hicache_ready_counts` (rank-divergence guard), `:3637 check_hicache_events`.
* Preemption-with-memory ("retraction") keeps a backup so a request can resume:
  `backup_kv_cache` (:1547) / `restore_kv_cache` (:1587) / `discard_kv_cache_backup` (:1634) with
  `RetractionBackup` — **exactly the primitive M6 needs** when a slot is demoted rather than killed.
* L3: `mem_cache/storage/` backends (file, NIXL, HF3FS, Mooncake, …), `hicache_auto_size.py`
  estimates the host budget from model/config.

### 7.4 Qwen4-Exp specifics (our model)

* `mem_cache/qsa_kv_pool.py:48 QSATokenToKVPool(HybridLinearKVPool)` — carries the **pending
  index-key/RoPE ring** (bf16, never quantised: "the raw keys are averaged from here") + the paged
  compressed-K cache. Invariant at :51-54: *"Full-KV pages are a multiple of the compress ratio, so
  no group straddles pages; `compressed_slot = full_slot // ratio` needs no ownership bookkeeping;
  lifecycle rides the full-KV allocator and radix tree."* Per-token cost helper
  `qsa_bytes_per_token(kv_heads, head_dim, compress_ratio, num_layers)` (:59-73).
* Indexer dtype switch `--qsa-indexer-dtype {auto,bfloat16,fp8_e4m3}` (:37-46).
* `mem_cache/ple_state_pool.py` — PLE short-conv window + N-gram context **attached to the MambaPool
  slot**, with the `SlotIndexedState` protocol (`reset_slots`, `copy_slots`, `get_cpu_slots`,
  `load_cpu_slots`, `iter_transfer_state_entries`) — i.e. non-KV per-sequence state already has a
  *tier-migration interface*.
* `layers/attention/qsa/`: `config.py` (`QSAProfile`, `block_topk = budget // compress_ratio`,
  fast_topk_v2 supports block topk ∈ {512, 2048}), `qsa_indexer.py` (compress store, ring slots),
  `metadata.py:228 build_pending_ring_slots`, `sparse_attn.py`, `mqa.py`, `fused_kv.py`.
* `models/qwen4_exp.py` (2669 lines) — indexer runs **outside** the decode graph
  (`:1856 "the indexer never runs inside the decode graph"`), PLE speculative is topk=1 only
  (`:256-258`).

### 7.5 Speculative / MTP scheduling

* `speculative/adaptive_spec_params.py` — runtime ladder for `speculative_num_steps` from measured
  acceptance: `DEFAULT_ADAPTIVE_CONFIG` keys by target batch size (`1 → [1,3,5,7]`, `8 → [0,1,3]`,
  `32 → [0,1]`, `64 → [0]`) with up/down hysteresis and a ceiling coefficient. A tiny, data-driven
  controller — cheap to replicate for M4.
* `frozen_kv_mtp_{worker_v2,info,utils}.py` + `eagle_worker_v2.py` + `decoupled_spec_io.py`
  (draft and verify overlapped), `draft_kv_budget.py`.

---

## 8. ik_llama.cpp

* **KV types**: `kv_cache_type_from_str` (`common/common.cpp:4545-4580`) accepts
  f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1/q6_0/**q8_KV**. `GGML_TYPE_Q8_KV = 151` and
  `GGML_TYPE_Q8_KV_R8 = 398` (`ggml/include/ggml.h:454,498`), announced as *"Q8_KV — new type for
  8-bit KV-cache quantization"* (README:206, PR 208) and *"Low perplexity Q4_0 KV cache"*
  (README:152, PRs 1547/1556).
* **Per-layer-group precision** — the standout idea:
  `-ctk-first TYPE,N` / `-ctk-last TYPE,N` (and `-ctv-first/last`) parsed at
  `common/common.cpp:2053,2064` → `mparams.type_k_first/…_last` (`:4624-4627`) → applied per layer
  in `src/llama.cpp:1584-1588`:

  ```cpp
  if (model.arch != LLM_ARCH_OPENPANGU && type_k_first != type_k && n_k_first > 0 && i < n_k_first)
      this_type_k = type_k_first;
  if (... && type_k_last != type_k && n_k_last > 0 && i >= n_layer - n_k_last)
      this_type_k = type_k_last;
  ```
* **Indexer gets its own cache type**: `-ictk/--indexer-cache-type-k` (`common/common.cpp:2048`,
  `mparams.idx_type_k` :4623) — confirms our spec's observation that the QSA side-cache is a
  first-class, independently-quantizable object.
* **Hybrid GPU/CPU**: `-nkvo/--no-kv-offload` → `cparams.offload_kqv = false`
  (`common/common.cpp:2034,:4724`; `docs/parameters.md:218 "Keep KV on CPU"`), i.e. attention runs
  against host memory. Also `docs/parameters.md:59-60` "hybrid cpu/gpu" mode, `-dexp` defer-experts
  and `-dple` defer-**PLE for Qwen3.8-Flash-Next** (`docs/parameters.md:118-119`) — the same
  resident-memory trick we will need for a 75 GB MoE on 2×16 GB.
* No context-swap-to-file or pinned-tier implementation found; the offload story here is
  "compute with KV on CPU", not "KV in tiers".

---

## 9. mistral.rs — language/architecture reference

### 9.1 Workspace

`Cargo.toml` members: `mistralrs-core`, `mistralrs-server-core`, `mistralrs-cli`, `mistralrs`,
`mistralrs-pyo3`, `mistralrs-vision`, `mistralrs-quant`, **`mistralrs-paged-attn`**,
`mistralrs-flash-attn`, `mistralrs-metal-compile`, plus `mistralrs-{audio,mcp,macros,code-exec,sandbox}`.
Candle-based, MIT, `rust-version = 1.94`.

### 9.2 Scheduler + paged attention

* `mistralrs-core/src/scheduler/mod.rs:177 pub trait Scheduler` — `schedule() -> SchedulerOutput`,
  `add_seq`, `waiting_len/running_len`, `free_finished_sequence_groups`, `get_finished_sequence_ids`,
  `block_size()`, `kv_cache_manager()`, `set_prefix_caching_enabled(...)`, plus ~15 capability
  toggles (`set_supports_packed_prefill`, `set_prefill_has_per_sequence_state`, …) that have default
  no-op bodies. **Trait with defaulted capability flags** is a good pattern for our M6 scheduler.
* `PagedPrefixCacheValidator` (`scheduler/mod.rs:158`) — `validate_prefix_cache_hit(seq, block_hashes,
  cached_tokens, block_size)`; the scheduler owns prefix validation, the engine owns storage.
* `paged_attention/`: `kv_cache_manager.rs:45` + `:280 allocate_slots`, `block_pool.rs:555-626`
  (`BlockPool::new(num_gpu_blocks, enable_caching, hash_block_size)` with retained-prefix free-block
  metrics), `block_hash.rs` (block hashes incl. multimodal spans and `clamp_prefix_cache_hit_len`),
  `cache_engine.rs` (`PagedCacheType {Auto, F8E4M3}` :47, Metal/CUDA layout selection),
  `scheduler.rs` ("*Uses a KVCacheManager to schedule and automatically batch sequences*"),
  `windowed_pool.rs` (SWA), `block_table_rows.rs`.
* `kv_cache/{single,rotating,full,hybrid}_cache.rs` — `hybrid_cache.rs` is the non-paged
  attention+recurrent hybrid (the same architectural problem as our GDN/QSA split);
  `kv_cache/mod.rs:71 cpu_kv_f16()`.
* `sequence.rs:752 struct Sequence`, `:2269 struct SequenceGroup` — the scheduler's unit of work.
* `memory_plan.rs` — `MemoryGoal {Auto, Utilization(f32), KvBytes, KvTokens}` with
  `DISCRETE_RESERVE_FRACTION 0.05` / `UNIFIED_RESERVE_FRACTION 0.25` (clamped 1–16 GiB) and
  `DEFAULT_MAX_NUM_SEQS = 32`; `memory_budget.rs`, `resource_plan.rs` alongside.
* **Speculative**: `speculative/builtin_mtp.rs` — *"Built-in MTP head proposer shared by hybrid GDN
  targets (Qwen3.5 / Qwen3.8, Qwen4-Exp)"*, EAGLE-style refresh after every target step, writes the
  drafter's own paged KV; siblings `proposer.rs`, `verifier.rs`, `driver.rs`, `policy.rs`,
  `autotuner.rs`, `hybrid_state.rs`, `paged_rows.rs`, `gdn_transitions.rs`.
* **Prefix/persistence**: `prefix_cacher.rs` (token-prefix map with `RecurrentStateSnapshot`,
  `PrefixBlockRetention`/`Lease`, capacity + block-pressure eviction reasons, metrics per owner) —
  the closest existing design to **read-only injected context blocks**.
* **Adapters**: `mistralrs-core/src/{adapter,lora,xlora,amoe}/` (`adapter/{generation,registry,
  runtime,selection}.rs`, `AdapterLease`, `DynamicLoRARuntime`).
* **Qwen4-Exp**: `vision_models/qwen4_exp/{config,hyper,ple,qsa,mtp,speculative,text}.rs` plus
  `cuda/qwen4_exp.{rs,cu}` and `gguf/qwen4exp.rs`.

---

## 10. exo (brief)

Apache-2.0. Rust core + Python. Distributed ring for a possible **Tier-4** later: node discovery via
`--instance-meta {ring,jaccl}` and `--sharding {pipeline,tensor}` (`README.md:558-561`), tensor-parallel
sharding claims (`README.md:27`), `src/` (Rust) + `python/exo` + `bench/`. No KV-tier or prefix-cache
lessons relevant to M5 — **skip for now.**

---

## 11. Synthesis for Breccia M5

### 11.1 The arithmetic (our model, `docs/specs/qwen4exp-architecture.md` §12)

Geometry: **12 KV-bearing layers** (3,7,…,47 of 48; the 36 GDN layers have *no* KV),
`head_count_kv = 2`, `klen = vlen = 256` ⇒ **1024 KV elements per token per layer**
(2 heads × 256 × K+V).

**KV bytes / token / layer** (block overheads included):

| cache type | block | bytes per 1024 elements | notes |
|---|---|---|---|
| f16 | — | **2048 B** | matches spec §12.1 (`2048 B/cell/layer`) |
| q8_0 | 32 elems = 34 B (8.5 bpw) | **1088 B** | +6.25 % over ideal 1024 |
| q4_0 / q4_K | 32 = 18 B / 256 = 144 B (4.5 bpw) | **576 B** | |
| K8V4 hybrid (K q8 + V q4) | mixed | **832 B** | strata's existing `ST/src/core/layer.cpp:1081-1101` |
| 3-bit (kvarn3 / tbq3, 128-elem record ≈ 50 B) | 128 = 50 B (≈3.1 bpw) | **≈400 B** | incl. scale; rotation lives in-band |

**KV bytes per token, whole model (× 12 layers)**:

| type | B/token | KiB/token |
|---|---|---|
| f16 | 24 576 | **24.00 KiB** |
| K8V4 | 9 984 | **9.75 KiB** |
| q8_0 | 13 056 | **12.75 KiB** |
| q4_0/q4_K | 6 912 | **6.75 KiB** |
| 3-bit | ≈4 800 | **≈4.69 KiB** |

**Indexer side-cache** (12 layers × `2×128` f32 = 1024 B/cell/layer, spec §12.2): **12 KiB/token at f32**,
6 KiB/token at bf16, 3 KiB/token at fp8_e4m3 (sglang's `--qsa-indexer-dtype` choices).

**Non-KV per-sequence state (constant, not paged)**: GDN recurrent `S` 3 MiB + conv 120 KiB per layer ×
36 layers ≈ **113 MiB/sequence**, + PLE conv history 360 KiB, + MTP head window (spec §12.4: 32768 cells
→ 64 MiB at f16 if fully allocated). Budget these **per agent slot**, not per token.

Total resident cost per token (KV + index, bf16 index): f16 **30 KiB**, q8_0 **18.75 KiB**,
q4 **12.75 KiB**.

Context cost, KV only:

| ctx | f16 (24 KiB/tok) | q8_0 (12.75 KiB/tok) | q4 (6.75 KiB/tok) |
|---|---|---|---|
| 32 Ki | 768 MiB | 408 MiB | 216 MiB |
| 64 Ki | 1.50 GiB | 816 MiB | 432 MiB |
| 128 Ki | 3.00 GiB | 1.59 GiB | 864 MiB |
| 262 Ki (full) | 6.00 GiB | 3.19 GiB | 1.69 GiB |

Add the indexer: f32 costs another 384/768 MiB/1.5/3.0 GiB at those lengths — **quantize it**.

### 11.2 Hardware budget (2 × 16 GiB + ≈30 GiB host, 75 GB weights)

Weights dominate: with 75 GB of weights on 32 GB VRAM, experts must live in mmap'd page cache on
`/srv` (the ik_llama `-dexp/-dple` and KTransformers/MoE-Infinity pattern) — so **pinned host memory
is the scarce resource, not NVMe**. Pinned pages are unevictable and will fight the expert page cache.

Proposed tiers:

| tier | medium | reserve | what fits (our numbers) |
|---|---|---|---|
| **T1 hot** | both GPUs' VRAM | **≈6 GiB total** (3 GiB/GPU) + ≈1 GiB workspace | 4 agent slots × 32 Ki tokens at q8_0 (408 MiB) + bf16 index (192 MiB) + GDN 113 MiB ≈ **2.85 GiB**; 8 slots at 16 Ki ≈ same |
| **T2 warm** | **pinned host RAM** | **6 GiB hard cap** (set at startup, warn above) | at q4 + bf16 index (12.75 KiB/tok): ≈ **470 k tokens** of reusable prefix; at q4 KV-only ≈ 890 k tokens |
| **T3 cold** | NVMe (mmap/`pread`) | effectively unlimited (hundreds of GiB) | injected read-only context + dormant conversation prefixes, stored q4 body + f16 header; read in 64 KiB–1 MiB chunks |
| **T4 ring** | other nodes (exo) | later | — |

**Page geometry**: `page_size = 16 tokens`, enforced to be a multiple of `compress_ratio = 4`
(sglang `qsa_kv_pool.py:51-54` — otherwise compressed and full slots desynchronise). Per page across
12 layers: f16 384 KiB, q8_0 204 KiB, q4 108 KiB — a natural NVMe/DMA unit. Block table for a 128 Ki
sequence = 8192 × u32 = 32 KiB (negligible). Use a **separate hash granularity of 16 tokens** too
(mistral.rs `hash_block_size`), so prefix hits are page-aligned. Keep strata's `page_size = 4` as the
*sub-page* granularity for the existing kernels rather than changing them.

**Body/tail policy** (steal from beellama, adapt to our sparsity):

* Body: q8_0 for the first ≈8 Ki tokens of a sequence, q4_K below that (or q4 everywhere for T2/T3).
  One rotation owner only (TurboQuant's rule) — a Hadamard applied on the store path, in-band scale.
* **Tail: last 1024 tokens exact f16** in a separate stage buffer, rounded up to 256-token groups,
  addressed with a packed `tier | slot | cell` index word exactly as `llama_kvarn.h:16-32` does, so
  kernels never branch on "is this row compressed?" — they branch on the index tag.
* **Sink: first 128 tokens exact** (beellama's rule) — our indexer budget (2048) and the 2051-cell
  sparsity threshold mean the head of the context is disproportionately important.
* **Indexer cache is never compressed below bf16**: selection quality, not KV bytes, is what fails
  first on long contexts. (sglang's default is bf16 with fp8 as opt-in; our spec's §12.2 shows it is
  12 KiB/token at f32 — bf16 halves that to 6.)
* **Layer-group asymmetry** (ik_llama): keep the last 3 attention layers (45/47 and 43) at q8_0 and
  the rest at q4 via an explicit `type_for_layer[12]` table — implement it as a per-layer type array
  from day one, even if the initial policy is uniform.

**Tier movement rules** (from vLLM's offload contract + sglang's retraction backup):

1. Only **complete pages** move; a page is in exactly one tier at a time (single-owner invariant,
   like vLLM's `KVCacheBlocks` + coordinator).
2. T1←T2/T3 promotion is **async and chunked** (`blocks_per_chunk` equivalent), never on the decode
   critical path; a miss during decode ⇒ **retract the request** (`sglang unified_radix_cache.py:1547
   backup_kv_cache`) rather than blocking the step.
3. T2→T1 demotion of a *whole sequence* is a metadata operation: keep the page bytes in T2 and let the
   block table point at a **T2-resident page** (llama.cpp's `offload_kqv=false` proves attention can
   read host memory; for us, prefill-only reads from host are acceptable, decode should not).
4. Read-only injected blocks: a radix subtree with `ref_count` pinned to ∞ (never evicted, only
   dropped on explicit invalidate), with `PrefixBlockRetentionLease`-style accounting
   (`mistralrs-core/src/prefix_cacher.rs`) so we can report how much T1 is held by injected context.
5. The QSA index page, the full-KV page, and the GDN/PLE state **move together**; GDN/PLE are
   per-sequence and slot-indexed (`ple_state_pool.py` `SlotIndexedState::get_cpu_slots/load_cpu_slots`
   gives the exact interface to copy).

### 11.3 Crate layout sketch (mistral.rs structure × sglang's layer rule)

```
breccia/
  crates/
    breccia-kv/                 # owns every byte of per-sequence state. No imports upward.
      layout/                   # page geometry (16 tok), dtypes, tier tag in the index word
        mod.rs  page.rs  dtype.rs  index.rs
      quant/                    # f16/q8_0/q4_K codecs + one Hadamard owner + KVarN-style norm
        wht.rs  q8.rs  q4.rs  rotate.rs
      pool/                     # allocator per tier: "give me N slots" (need_size -> indices)
        device.rs               # T1 VRAM pages
        host.rs                 # T2 pinned, hard byte cap
        disk.rs                 # T3 NVMe, chunked async read
      radix/                    # prefix tree: RadixKey, TreeNode{refcount,last_access,priority}
        tree.rs  evict.rs       # eviction as a strategy trait (LRU/TLRU/…)
        injected.rs             # read-only pinned subtree (M5 feature)
      qsa/                      # indexer side-cache: compressed K + pending ring, cell-for-cell
      state/                    # GDN S + conv + PLE: slot-indexed, tier-migratable, never paged
    breccia-sched/              # continuous batching + agent slots
      scheduler.rs              # trait Scheduler with defaulted capability flags (mistral.rs:177)
      sequence.rs               # Sequence / SequenceGroup
      retract.rs                # backup/restore on preemption (sglang RetractionBackup)
      admission.rs              # reserved_blocks / full_sequence_must_fit gates (vLLM:378)
    breccia-spec/               # M4
      mtp.rs  proposer.rs  verifier.rs  policy.rs   # builtin_mtp.rs / adaptive_spec_params.py
    breccia-attn/               # kernels: paged attention, QSA sparse, tail-window attention
    breccia-model/              # qwen4exp, GDN, MoE
    breccia-engine/             # pipeline, memory_plan (MemoryGoal), resource_plan
```

Dependency rule to enforce in CI: `breccia-attn`/`breccia-sched`/`breccia-spec` → `breccia-kv`;
**nothing** may import back down into a lower layer (sglang `mem_cache/README.md`).

### 11.4 M5 milestone split suggested by these references

1. `kv/layout` + `kv/pool/device` + f16 pages + block table (vLLM `KVCacheBlock` minimal state:
   id, refcount, hash, free queue).
2. q8_0/q4 codecs + one rotation + the **f16 tail stage** (beellama's packed index) → measured
   bytes/token vs §11.1 table.
3. `kv/radix` prefix reuse with page-aligned hashes (sglang tree + mistral.rs block hashes).
4. T2 pinned tier with a hard cap + async chunk promotion (vLLM `kv_offload` contract).
5. T3 NVMe + read-only injected subtree + prefetch.
6. QSA/PLE/GDN state migration + retract-based demotion → feeds M6 scheduling.
