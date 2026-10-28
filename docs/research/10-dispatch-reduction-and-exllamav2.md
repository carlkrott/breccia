# 10 — Dispatch reduction: a ranked, costed path from ~4 700 to ~500, and the exllamav2 bit-allocation audit

Round 10. Paired with the implementation round that owns the single-stream lane.
Engine audited: **exllamav2** (`Reference Projects/exllamav2`, shallow, `7dc12af`) — deep-dive on the
KV-quant / bit-allocation path, which `03-io-and-exllamav2.md` only surveyed.

**This round contains no GPU measurements of its own, by design.** The RX was on the critical path of a
sibling agent's benchmark for the whole window (a `breccia bench token` process plus `gpu_busy_percent`
55 % on `0000:03:00.0` at the start of the round; still ≥1 `breccia` process at the end). Everything
below is therefore derived from **existing measurements** (`docs/research/04`, `06`, `06a`, `07`, `08`,
`09`), **our own code**, the **Vulkan spec / driver capability surface**, and the **cloned reference
engines**. Where a number is an inference it says so. Two decisive experiments are handed to the
operator with exact commands (§A7) rather than guessed at.

---

## A0. First: the denominator is wrong, and it is the gate

The round's target is stated as "~4 700 dispatch-equivalents/token → ~500". **The 4 700 is not
reachable by fusion, and a static walk of the decode path does not produce it.** Both statements are
checks, not opinions:

1. **Static census of the decode path** (this round, read from the code — `graph.rs`
   `submit_block`/`record_dense_prelude`, `qsa::forward`, `gdn::forward`, `HcMixer::read`/`write`,
   `run_placed_moe`, `dispatch_batch::record`). Every kernel entry point in the batch is exactly one
   `record()` call (`batch.rs:275`, `gemv.rs:346`, and `gemv.rs:350` for the reduce), so the count is
   enumerable:

   | block kind | stage | dispatches | what they are |
   |---|---|---:|---|
   | both | `hc_read` ×2 | 12 | norm, down (split+reduce), act, up, gated — per read |
   | both | `hc_write` ×2 | 2 | one gated-write each |
   | GDN (36 blocks) | mixer | 10 | qkv, z, beta, alpha, conv, qk-l2, beta-gate, delta-rule, norm-gate, ssm-out |
   | QSA (12 blocks) | mixer | 19 | 5 proj, 3 norms, 3 rope, kv-append, idx-pool, idx-tail, idx-score, topk, attend, apply, out |
   | both | router | 2 | router GEMV + top-k |
   | both | shared expert | 5 | gate, up, swiGLU, down, shexp-gate |
   | both | routed MoE | 1 (host lane) / 8 (placed) | placed: 3 map + gate + up + swiGLU + down + combine |

   → **≈ 32 dispatches per GDN block, ≈ 41 per QSA block**, i.e. **1.65 k/token** (+ head ≈ 8, + embed),
   with 30 placed blocks adding 7 each (+210) → **≈ 1.9 k/token**. The full structural maximum that
   *any* fusion can remove is therefore ≈ 1.05 k/token (§A3), which caps a fusion-only campaign at
   ~800/token — a **2.4×**, not a 10×, against the static count.

2. **The 4 700 cannot be produced by this code.** 4 700/48 = 98 dispatches per block; the enumerated
   path has 32–41. The engine counts `self.dispatches += rec` where `rec = batch_cmd::recorded()` is
   the *count of the batch just closed* (`graph.rs:1015`, `:1057`), and `--gen 6` on a prompt executed
   one token at a time divides that by the *generated* tokens only — a prompt of *P* tokens adds
   *P* × 1.65 k, which at 6 generated tokens lands squarely in the 3–5 k/token range. That is the
   likely mechanism; it is not proven.

3. **The 6.7 µs/barrier figure is not safe to build on, and the round's own experiments say so.**
   `docs/phase-4` §8am derives 26.2 ms / 3 900 barriers = 6.7 µs *of GPU-timeline serialisation per
   barrier* from a v1-`ALL_COMMANDS` vs sync2 `COMPUTE→COMPUTE` A/B. But (a) that A/B ran in a window
   the doc itself flags as contaminated ("the lane swung 13.4–26.2 ms between identical configs"), and
   (b) `06a` §2 measured the two barrier *forms* as **identical** — 5.624 / 5.624 / 5.624 µs on a
   `bpr=10` chain, 1.150 µs on a trivial one, 0.0 % apart. A 1.45× wall-clock difference cannot come
   from a primitive that measures 0 % apart, and (c) **barrier pruning removed 19.6 % of barriers for
   no gain at all**. All three cannot be true. Something in the v1/sync2 A/B moved outside the
   barrier.

**Consequence for the round: express every option as *dispatches removed per block*, never as a
fraction of 4 700.** Those numbers survive whichever denominator wins. And spend the first ten minutes
of the next GPU window resolving the denominator (§A7 E0) — it changes the per-leg price by 2.5–3×,
which is the difference between "fusion is worth 2×" and "fusion is worth 6×".

---

## A1. What a barrier actually costs — and why per-block buffers do not fix it

### A1.1 The measurement ladder we already have (all RX, dependent chain, 1 barrier/dispatch)

| kernel | working set | `bpr` (dependent blocks/row) | workgroups | µs per leg |
|---|---:|---:|---:|---:|
| trivial (1 lane, K=256) | ~1 KiB | 1 | 1 | **1.15** |
| 640×512 | 172 KiB | 2 | 320 | **1.62** |
| 640×640 | 200 KiB | 3 | 320 | **2.48** |
| 640×2560 (IQ4_XS) | 870 KiB | 10 | 320 | **5.56** |
| 160×2560 | 218 KiB | 10 | 80 | **5.56** |
| 8×2560 | 11 KiB | 10 | 4 | **5.63** |
| 1×2560 | 1.4 KiB | 10 | 1 | **7.40** |

Two facts fall straight out, and they are the whole answer:

* **Grid size is irrelevant.** 1 → 4 → 80 → 320 workgroups is flat (7.40 / 5.63 / 5.56 / 5.84). The
  wave-drain hypothesis is dead.
* **Total bytes are irrelevant.** 1.4 KiB and 870 KiB cost the same. So the "L2 invalidation of a
  large dirty footprint" hypothesis is *also* dead at these sizes.
* **`bpr` is the only axis**, and it fits `t ≈ 1.1 + 0.45·bpr` µs (bpr = 1/2/3/10 → 1.55/2.0/2.45/5.6
  predicted against 1.15/1.62/2.48/5.6 measured).

### A1.2 The reading

The barrier's own price is a **fixed ≈1.1 µs drain/launch floor**, identical for every legal barrier
form on RADV (0.0 % between `ALL_COMMANDS`, sync2 `write→read` and sync2 `read|write→read|write`). The
rest is **not the barrier**: it is the *next* dispatch re-fetching its dependent weight blocks cold,
because the acquire invalidated the caches, and each of the `bpr` dependent block loads is a DRAM
round trip (~450 ns) that the kernel's inner loop does not hide. `06a`'s no-barrier control
(20 000 identical dispatches at 0.021 µs each) confirms it: with nothing to invalidate, the same
kernel runs 265× faster per dispatch because the loads stay in L2 and the dispatches overlap.

So the honest cost model for one chain leg is

```
t_leg ≈ 1.1 µs                       (irreducible: drain + launch, form-independent)
      + bytes_re-read / 585 GB/s     (only if the leg's working set WAS resident and is now cold)
      + 0.45 µs × bpr_unhidden       (only if the kernel's block loop is not pipelined)
```

### A1.3 Would per-block buffers reduce it?

**No — not the cost, only the count, and the count experiment already failed.** Reasoning:

* In our graph each block's expert GEMV reads **fresh** weights (0.87 MB per expert, 43 GB aggregate).
  Those bytes are cold *regardless of the barrier*: the previous leg read a different expert. The
  `bytes_re-read` term is therefore a property of **streaming**, not of the barrier, and is identical
  in both barrier forms — which is exactly what `06a` measured.
* Per-block (per-branch) buffers do not change `bpr`, and they do not change the weight stream.
* What they *do* change is aliasing: if `attention`, the `shared expert` and the `routed experts` each
  write their own output instead of all writing `r`, the two barriers between them disappear. That is
  a **leg-count** win of **2 legs/block = 96 legs/token**, not a per-barrier win.
* Memory cost is trivial: `r` is `hc × n_embd` = 4 × 2 560 f32 = **40 KiB**. Three branch copies
  × 48 blocks = **5.9 MB**, against a 16 GB card. Even f16 + double-buffering is < 10 MB.

**Decision rule (use this instead of re-measuring):**

| condition | what dominates | lever |
|---|---|---|
| next leg's set is L2-resident and reused (e.g. the shared expert's 0.92 MB, IC-resident) | the 1.1 µs floor | fold the leg away |
| next leg streams fresh weights (our expert GEMVs, everything at `bpr=10`) | floor + `0.45·bpr` cold-load | **`UNROLL`/prefetch in the kernel** (−9 % measured, `06` §4) |
| next leg is *bandwidth*-bound over ≫L2 | floor only | leg count |
| a barrier separates two dispatches whose storage sets are disjoint | **nothing** — that is why pruning bought 0 | prune (already done) |

So: **do not build per-block buffers to make barriers cheaper. Build branch-private outputs only where
they delete a serial fan-out**, and put the effort into (i) fewer legs, (ii) a pipelined block loop.

---

## A2. Ranked, costed design — D1…D8

Dispatch savings are **per token** at the static census (~1.9 k) and, in brackets, at the engine counter
(~4.7 k) — both shown because A0 is unresolved. "Impl cost" is the shape of the work, not hours.
Every item is independently flaggable and independently revertible.

### D1 — Fused `gate+up+SwiGLU`, one dispatch (shared expert **and** every routed expert)
*Saving:* −2/block for the shared expert, −2/block for the routed path ⇒ **−192 [−192]**.
*Evidence:* `06` measured **15.6 → 4.7–5.0 µs (m=1)** and **50.7 → 9.3 µs (m=8)** — 3.1×/5.3× on the
dependent path, no LDS staging needed, no spill risk.
*Impl cost:* one new shader variant per quant type that reads two weight pointers and applies
`silu(g)·u` in the epilogue; the existing `blocks.glsl` decode is reused unchanged.
*Risk:* **low**. Same maths, same weights, one dispatch instead of three; bit-identical to within the
f32 accumulate order (must be gated, not assumed).
*Validation:* parity 32/32 greedy tokens vs the HIP server; digests at `Bf16`/`Mixed` buffers
unchanged; dispatch census shows −2/block.
*Why first:* largest single dispatch reduction per unit of work, already priced, and no dependency on
anything else in this list.

### D2 — Expert-indexed MoE: one `gate+up` dispatch over all `(expert, row)` pairs, one `down`, one combine
*Saving:* the placed path today is 3 map dispatches + 3 GEMVs + swiGLU + combine = **8 → 3** ⇒ **−5 on
every placed block** (30 blocks ⇒ **−150 [−150]**), *and* it is the shape that lets the host lane's
experts be computed by the same kernel later (one dispatch covering resident + host-resident rows).
*Evidence:* this is `ggml`'s `GGML_OP_MUL_MAT_ID` (`09` §B3, own tile schedule with `BK_STEP=1`), MLC's
`indptr`-driven group MoE (`01`), exllamav2's `moe_forward` taking **all experts' quant handles as an
array** in one C++ call (`moe_mlp.py:120-124`), and PowerInfer's residency-aware row skip (`02` §4).
Our kernels already carry an expert map (`v.moe_map_*`), so this is closer than it looks — the map
dispatches exist only to build the index that D2 folds into the GEMV's own row addressing.
*Impl cost:* extend `GemvBatch` to (a) take the row map as a buffer the shader indexes, rather than a
host-built map, and (b) carry per-row weight base from an expert-id array. Moderate; one shader.
*Risk:* **medium** — a wrong map is a silent numerical error, not a crash. Gate on digests.
*Validation:* layer digests for 3 QSA + 3 GDN blocks, then 32-token parity; census −5/placed block.

### D3 — Multi-projection bundling: one dispatch for all of a sublayer's independent input projections
*Saving:* GDN `qkv,z,beta,alpha` 4 → 1 (**−3**), QSA `q,k,v,idx_q,idx_k` 5 → 1 (**−4**) ⇒
**≈ −170 [−170]** over 48 blocks. Also folds `shexp_gate` into the router's epilogue (−1 more).
*Evidence:* all five read the *same* activation row and write disjoint outputs; only the weight matrix
differs. Concatenating them offline into one `[Σrows, K]` tensor makes it a single GEMV over a wider
`N`. This is the one item that is **pure layout** — no new shader maths, so it is the cheapest large
win and it composes with the existing fused `gate+up+SwiGLU` idea (D1 is the same trick applied to
two matrices that are *not* adjacent).
*Caveat:* the quant types of the five tensors need not match (`q3_K`, `q4_K`, `q5_K`…). The bundle must
therefore be **per-type-grouped** (a "segment table" of quant types with one dispatch per type group),
which costs at most 2 dispatches per sublayer instead of 1 if all types differ. Check the artifact
first — `docs/specs/` has the per-tensor map.
*Impl cost:* an offline repack (no requantisation, byte copy) + a `rows_offset` push constant into the
existing GEMV. **Low.**
*Risk:* **low** (bytes are copied verbatim); the only new failure mode is a wrong segment table.
*Validation:* bit-identical vs the un-bundled path on the same weights (this one *must* be exactly
bit-identical — same instruction stream, different address), then census.

### D4 — Elementwise bundles + `RoPE`/`KV-append` fusion
*Saving:* GDN `conv → qk-l2 → beta-gate` 3 → 1 (**−2**); `norm_gate + ssm_out` 2 → 1 (**−1**); QSA
`q/k/idx-q norm` 3 → 1 (**−2**); `rope_q + rope_k + rope_idx_q` 3 → 1 and that fused with `kv_append`
(**−2**, 12 blocks); `idx_pool/idx_tail/idx_score/topk` 4 → 2 (**−2**, 12 blocks) ⇒ **≈ −130 [−130]**.
*Evidence:* every one of these is elementwise, or elementwise-then-GEMV, over the same buffer; `RoPE`
needs only `pos`, already a push constant. `phase-4` §8am lists RMSNorm+QKV and RoPE+KV as items 2–3
of the fusion order and they have never been written.
*Impl cost:* mechanical shader merges; several small shaders rather than one big one.
*Risk:* **low**; the RoPE+append fusion must respect the KV-write ordering (it writes the cache the
next block reads — that is a real hazard, and `is_src_of`-style alias checking is the pattern to copy,
§A6).
*Validation:* digests + parity; census.

### D5 — Collapse `hc_read`/`hc_write`: 14 dispatches per block down to 6–8
*Saving:* **−6 to −8 per block ⇒ −290 to −380 [same]**. This is the *biggest structural item in the
whole list* and nobody has pointed at it: `HcMixer::read` (norm, down-split, reduce, act, up, gated) plus
`write` costs **7 dispatches, twice per block, = 14 of 32** — 44 % of a GDN block's dispatch count is
the hyper-connection projector.
*Geometry:* `hc = 4`, `n_embd = 2 560`, `hc_lr = 320`; `down` is `10 240 → 1 280`, `up` is `1 280 →
10 240`, `gated` is `10 240 → 2 560`.
*What fuses cheaply:* `act + up` (act over 1 280 sits in LDS, one dispatch saved); `gated + write`
(the write's `inject` is already produced by the previous read, one dispatch saved); `norm + down` by
**recomputing the norm inside each `down` workgroup** (each of 160 WGs reads all 10 240 of `xn` =
1.6 M loads = 6.5 MB from L2, ~2 µs, against a 1.1 µs barrier — roughly a wash, and it *removes* a leg).
*What does not:* `down` and `up` cannot be one dispatch (2560-wide rows at parallelism 1 — the same
argument that killed `swiGLU+down` in `06`).
*Realistic target:* **14 → 7–8**, i.e. the safe end of the range.
*Impl cost:* three shader changes and a workspace layout for the hc intermediate. **Medium.**
*Risk:* **medium** — `hc` is the residual plumbing, so an error is a whole-model numerical drift, and
`maybe_checksum` at `Mixed` is exactly the guard that catches it.
*Validation:* `--debug-layers` digests on every block for one token vs HEAD, then parity.

### D6 — `vkCmdDispatchIndirect` over a device-built work list
*Saving:* **0 by itself** — `04` priced indirect at **+0.77 µs (RX)** per dispatch over direct
(1.418 vs 0.650 in a dependent chain). Its value is *enabling*: a work list written by the router's own
epilogue lets a single dispatch cover a variable set of expert rows (the D2 shape) without the host
touching the map. Adopt it **only inside D2**, never as a standalone item.
*Risk:* **low**; already present on both cards (and `VK_EXT_device_generated_commands` is present with
COMPUTE pipeline binding, `04` — still worthless here, because every token it emits is an indirect
dispatch).

### D7 — Pre-recorded command buffers (frozen graph replay)
*Saving:* **≈1.4–1.8 µs per dispatch, and it lands in `vkQueueSubmit`, not on the GPU timeline**
(`08`: replaying a frozen CB is 39 µs cheaper per 12-dispatch round than re-recording it). At the
post-fusion count (~600 legs) that is **0.8–1.1 ms/token**; at ~1.9 k it is 2.7–3.4 ms. It does **not**
touch the 26–32 ms of `wait-gpu`, which is GPU-timeline serialisation.
*Verdict:* **last, and mostly for host-wall reasons.** It is also strictly *downstream* of D1–D5: every
fusion invalidates every frozen CB, so building replay first means recording twice.
*What it needs (the concrete blockers):*
* **Push constants are the killer.** Everything that varies per token (position, KV cell offset, expert
  map base, row counts) rides `cmd_push_constants`. A replayed CB re-executes the *recorded* push
  constants. Making the graph replayable means moving those into a **device-visible argument buffer**
  that a shader loads (the classic CUDA/DGC "kernel params in memory" pattern) — +1 load per dispatch,
  ~free, but it is a change to every shader's entry.
* **Descriptor sets.** Today `record()` allocates a *fresh* set per dispatch from a per-slot pool
  (`dispatch_batch.rs:797`) — mandatory, because `vkUpdateDescriptorSets` may not touch a set a pending
  CB references (the MI50 GPUVM fault that the module docs record). A frozen CB needs a set that is
  *stable for the life of the CB* ⇒ per-slot persistent sets, or bindless (§A4).
* **Barriers inside secondaries are fine.** `08` verified 12/12 dependent increments with a barrier
  inside a secondary **and** with one in the primary, sync-validation silent. So a whole block can be
  one secondary CB with its own barriers. **But** with *no* barrier the answer is wrong — a secondary
  does not implicitly order against its parent, so every dependency must still be recorded as a barrier.
* **Query resets are the real trap.** A pool shared across ring slots is exactly the bug that made
  `BRECCIA_GPU_TIME=1` abort on the first forward pass (`phase-4` §8am). Rule: **one query pool per ring
  slot, reset outside the CB** (`vkResetQueryPool` host-side, or `cmd_reset_query_pool` in the primary
  once per submit), stamp inside. Never `cmd_reset_query_pool` a pool a replayed CB will also write.
* **`VK_KHR_pipeline_binary` is unrelated to any of this** — it is pipeline-cache blob control
  (load-time), not per-dispatch cost. Present on all three devices. Low value; skip unless we go to
  per-shape-specialised pipelines.
* **Secondary-CB overhead is not a concern:** `08` measured a CB boundary at **1.7 µs of GPU timeline,
  identical for secondary and primary**. 48 secondaries/token = 0.08 ms.

### D8 — Persistent / mega-kernels: what the execution model actually allows
*Honest verdict: mostly no, and the parts that work are D1/D4 again.*
* **Vulkan has no cooperative launch and no grid-sync.** The spec gives no forward-progress guarantee
  between workgroups; a shader that spins on another workgroup's write is undefined behaviour. A
  device-wide software barrier only works if the whole grid is resident, which an app can only
  *estimate* (60 CUs × waves-per-CU from LDS/register occupancy).
* **Crash-safety makes this a stop-the-line risk, not a tuning risk.** A deadlocked grid is a ring
  timeout; on the display card (MI50) recovery is a BACO reset that **rebooted the box** on 2026-10-26.
  A mega-kernel is therefore disqualified unless the grid provably fits, and even then only on the
  headless RX.
* **What is legal and useful:** a single *workgroup* that owns a whole sub-step, with LDS-resident
  intermediates and no cross-WG sync. Then the "mega-kernel" is just a fusion, and it has already been
  priced: `gate+up+SwiGLU` = 1 dispatch (D1), `RMSNorm+QKV` = 1 (D3/D4), `RoPE+KV-append` = 1 (D4).
* **What is not legal/viable:** one dispatch per block (needs ~10 sequential phases with grid sync);
  `swiGLU+down` in one WG (all 2 560 rows need the 640 intermediates — parallelism 1, `06` §4);
  a persistent SSM chain (`delta_step` has `H_V × S` ≈ 16 k independent lanes and 10 dependent phases).
* **Parity validation if it were attempted:** bit-identical digests are the wrong gate (a different
  work decomposition changes f32 accumulation order). It would need a *tolerance* gate plus a
  deadlock-proof grid bound printed at record time plus a bounded fence wait (never `u64::MAX`). That is
  a large amount of new machinery for a fusion that D1/D4 already deliver legally.

### D9 (not a dispatch item, but the honest ceiling) — the lane, not the legs
`09` already priced the endgame: zero-copy host-resident experts remove the 13.4 ms lane and the staging
copy. **After D1–D5 the dispatch chain stops being the bottleneck** and the arithmetic is:
`exec 7 + lane 13.4 + head 7 + ple 1.2 + xfer 0.6 ≈ 29 ms` → ~34 tok/s. Beyond that, `09`'s item 1 is
the only thing that moves.

### A3. Cumulative budget

| after | GDN block | QSA block | per token (48 blocks) | vs static 1.9 k |
|---|---:|---:|---:|---:|
| today | 32 | 41 | **1 650** (+210 placed) | — |
| D1 | 30 | 39 | 1 550 | −6 % |
| + D2 | 25 | 34 | 1 310 | −21 % |
| + D3 | 21 | 29 | 1 110 | −33 % |
| + D4 | 17 | 22 | **900** | −45 % |
| + D5 | **10–11** | **15–16** | **≈ 600** | **−64 %** |
| + router/combine fold | 9 | 14 | **≈ 530** | −68 % |

So **~500 is reachable, but only with D5 (the hc collapse) in the set** — D1–D4 alone stop at ~900. If
the true denominator is 4 700, the same removals (1 120 legs) leave 3.6 k and the 10× target is
arithmetically impossible: another confirmation that A0 has to be resolved first.

---

## A4. Descriptor cost — we pay the most expensive possible path, and it is free to fix

**What we do now, per dispatch** (`dispatch_batch.rs:790-870`):
`vkAllocateDescriptorSets` → `vkUpdateDescriptorSets` with 4–5 writes → `cmd_bind_pipeline` →
`cmd_bind_descriptor_sets` → `cmd_push_constants` → [barrier] → `cmd_dispatch`. **7 commands, one of
which is a driver allocation.** `04` isolated that allocation as the whole difference between 68 ns and
212 ns of record time per dispatch.

**What the driver gives us** (verified from a `vulkaninfo` dump taken on this host, all three devices —
`<SCRATCH>/opencode/vinfo.txt`, not re-probed this round):

| capability | status on RX / MI50 / iGPU | what it buys |
|---|---|---|
| `VK_EXT_descriptor_indexing` | present | a **bindless SSBO array**: one `cmd_bind_descriptor_sets` per command buffer, **zero per dispatch** |
| `VK_KHR_push_descriptor` rev 2, `bufferlessPushDescriptors = true` | present | push a set with no allocation — ncnn's exact pattern |
| `VK_KHR_descriptor_update_template` | present | `vkCmdPushDescriptorSetWithTemplate` takes a **pre-built byte blob**: one memcpy'd call instead of N writes |
| `VK_EXT_descriptor_buffer` (`descriptorBuffer = true`) | present | descriptors live in **plain buffer memory**, `storageBufferDescriptorSize = 16 B`, `descriptorBufferOffsetAlignment = 4 B` |
| `VK_EXT_descriptor_heap` | present | Vulkan 1.4's mutable heap — newest, least mature path |
| `maxPerStageDescriptorStorageBuffers = 8 388 606` | all devices | the array size is a non-issue |
| `VK_KHR_maintenance6` | present | `vkCmdBindDescriptorSets2` / `vkCmdPushDescriptorSet2` |

**ncnn's pattern, concretely** (`src/command.cpp:1386-1440`): build a flat `std::vector<unsigned char>`
of `VkDescriptorBufferInfo`/`VkDescriptorImageInfo` by `memcpy`, then
`vkCmdPushDescriptorSetWithTemplateKHR(cmd, pipeline->descriptor_update_template(), layout, 0, blob)`.
No pool, no allocation, no set bind. About 10 lines of record code, and it removes the allocation that
`04` measured as the dominant record cost.

**Our better option** (cheapest *and* fastest on the GPU timeline): **bindless + a push-constant base
index.** One `[N]` SSBO array bound once per command buffer; each dispatch passes `base` in its push
constants and addresses `ssbo[base + i]`. Then per dispatch the descriptor work is **zero Vulkan
calls** — the host writes `base` into the push-constant block it was already writing.

**Sizing** (our numbers, not a guess): a dispatch binds 4–5 SSBOs. At 16 B each:
`1 900 legs × 5 × 16 B = 152 KB/token`; with an 8-slot ring and 4 k descriptors/slot = **2 MB** of
descriptor buffer. Nothing.

**Measured-expected saving:** removes ~130 ns of *host record* time per dispatch (`04`'s 212 → 68 ns
delta) — at 1.9 k legs that is **−0.25 ms/token of host lane**, i.e. *small*. The reason to do it is
not the milliseconds: it is that it removes the **last host-side blocker to D7 (frozen CBs)** and it
removes `cmd_bind_descriptor_sets` from the GPU packet stream. Rank it **D0 — infrastructure, not a
win**, and do it in the same round as D1 so the re-record traffic of iterating on fusions is cheap.

---

## A5. Whole-graph optimisation — `ggml_vk_graph_optimize` extracted

`llama.cpp-ggml-vulkan` @ `7aca0d2`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp:14865-15140`. The algorithm,
concretely:

1. **Greedy list schedule over a 20-node lookahead window** (`NUM_TO_CHECK = 20`). Take the first unused
   node into `current_set`; then scan the next 19 and pull in every node whose inputs are all
   *already scheduled* — i.e. hoist independent downstream work up beside its siblings.
2. **Dependency test `is_src_of(dst, src)`** — direct `src[]` match, **or** a base-storage alias in
   either direction (`dst->src[s]` views `src`'s base, or `src->src[s]` views `dst`'s base), **or** a
   shared base. That third clause is the part our `DIRTY`-buffer tracker does not have: it catches
   *views* (our `Buf` slices) aliasing each other, which is precisely the class of bug that
   `skip_next_barrier` failed on (`phase-4` §8z/§8aa).
3. **Fusion patterns are pinned, not discovered.** A table of op sequences
   (`rms_norm+mul+add+mul`, `rms_norm+mul+add`, `rms_norm+mul+rope+view+set_rows`,
   `rope+view+set_rows`, `snake`, `topk_qsa`, five `topk_moe` variants, `hc_post_gate`) is matched at
   the cursor and emitted verbatim, and *every interior index of a pattern is skipped* by the window
   scan so the reorder cannot separate it.
4. **Dependency is explicitly allowed inside a fusion.** A node that *does* depend on the previous one
   is still pulled in when the pair is a supported fusion: `RMS_NORM→MUL`, `UNARY→MUL`, `MUL_MAT→ADD`,
   `MUL_MAT_ID→ADD_ID`, `MUL_MAT_ID→MUL`, `ADD→ADD`, `SSM_CONV→ADD`, `SSM_CONV→UNARY`. That is the
   whole trick: **the pair becomes one dispatch, so there is no barrier to lose.**
5. **Views are grabbed in a second pass** and deliberately *not* interleaved, "to not break fusion".
6. **The fusion itself is `ctx->num_additional_fused_ops`** (`:6933-6964`): a mat-vec consumes the next
   **one or two** nodes (bias/scale) via push-constant flags `MAT_VEC_FUSION_FLAGS_BIAS0/BIAS1`, with
   the result written to the *later* node's tensor. The extra operands ride as two more descriptor
   bindings (`d_F0`, `d_F1`) — no branch in the shader, just a flag.

**What it would gain for us: essentially nothing as a reorder, and that is the honest answer.**
Our recorder already emits siblings adjacently (`gate` then `up`, `qkv`/`z`/`beta`/`alpha` in order),
and our `DIRTY` tracker already skips barriers between storage-disjoint dispatches. A 20-node window
reorder can only help a graph whose *emission order* is worse than its dependency order; ours is not.
The `is_src_of` hazard check is genuinely better than our dirty-set test and should be **copied into
the recorder's fusion planner**; the pattern-pinning idea is what makes fusion safe under any future
scheduler. But the dispatch reduction in this round comes from **D1–D5 (actual fusions)**, not from a
graph pass. Rank: **D10 — a correctness tool, not a performance item.**

---

## Part B — exllamav2 audit (`7dc12af`, shallow clone present at `Reference Projects/exllamav2`)

`03-io-and-exllamav2.md` already surveyed the engine. This is the deep-dive on the four things the
round asked for. All of its hot kernels are CUDA; the **policy** is the transferable part, and I mark
which is which.

### B1. EXL2 bit allocation — the algorithm, in enough detail to reimplement

The code: `exllamav2/conversion/measure.py` → `optimize.py` → `quantize.py`.

**Unit of allocation is a whole sublayer, not a tensor.** `num_modules = num_layers * 2` — one slot for
`self_attn` (q+k+v+o together) and one for the MLP (gate+up+down together, **× `num_experts`** for an
MoE: `optimize.py:81-84`). So the search space is *2 × n_layers* slots, not *n_tensors*.

**Step 1 — measure, don't model.** `measure.py` runs the real model with real calibration data
(`standard_cal_data`) and, for each slot, tries **every combination of candidate bit-widths** across the
tensors in that slot (`measure_attn` iterates `bits_q × bits_k × bits_v × bits_o`, `mlp` the analogous
product), quantising the module with RTN at each option and measuring the output error against the f16
reference (`test_error`). Each result is recorded as `{ "accuracy": …, "total_bits": … }` — a real
measured (cost, quality) point per option. Early-exit: if a slot's best accuracy is `< 0.1` the module is
declared too sensitive and the slot is constrained (`:195`, `:284`).

**Step 2 — simulated annealing on a global byte budget.** `ext_c.sim_anneal(slots, weight_budget, …)`
with `slots[i] = [(total_bits, 1-accuracy), …]`:
`anneal_temp_max = 2`, `cooling = 0.995`, `temp_min = 1e-4`, `anneal_iter = 1000`, and
**`anneal_samples = 80` restarts across `anneal_stages = 3`**. The objective is not mean error: it takes
a **norm** parameter, swept `1.5 → 3.5` in stage 1 (search for the right mean-vs-peak trade-off), then
bracketed `bestnorm ± 0.075` in stage 2, then fixed in stage 3. Successive halving keeps the best
`(s, s_idx, p_norm, bits_used, max_err)`.

**Step 3 — greedy budget top-up.** `while` any slot can step up one bit-width without exceeding
`weight_budget`, do it (`optimize.py:158-168`). This is what converts "an ok solution" into "the target
size, exactly".

**Transfer to our expert-placement problem — partially, and the mapping is exact if you rename
things:**

| exllamav2 | ours |
|---|---|
| slot = one sublayer (attn / mlp×experts) | slot = **one block's expert set** (48 slots, or 48 × the 36-tensor dense bundle) |
| option = a bit width per tensor | option = **a placement**: `k_gpu ∈ {0,2,4,6,10}` experts resident per block |
| cost = `total_bits` | cost = **VRAM bytes** (measured, exact — no estimation needed) |
| value = measured accuracy | value = **measured lane ms/token** (also exact, from `bench token`) |
| objective = tunable p-norm of error | objective = **whole-model ms/token**, which is additive ⇒ a plain knapsack |

So the *annealing machinery is unnecessary for us* — our values are deterministic and additive where
EXL2's are noisy measurements — but two ideas do transfer and are worth taking:

1. **Per-slot, measured (cost, value) tables instead of a heuristic.** We currently choose placement
   from `--split-blocks`/`--gpu-experts` lists tuned by hand. A one-off sweep that measures, for each
   block, the lane time with `k_gpu = 0/2/4/6/10` gives a 48 × 5 table; the allocation then becomes an
   exact integer knapsack (greedy by value-per-byte is optimal enough at 48 items) *and* it is
   re-solvable at load time from measured free VRAM — which is exactly `phase-4` §8am item 3b
   ("adaptive coverage"), now with a principled objective.
2. **The "anneal the norm" trick is a real answer to a real problem**: it tunes *mean vs worst case*.
   Our analogue is *mean ms/token vs worst-stream latency* in a multi-stream run, which is the metric
   §8ak shows is *not* the mean. Worth remembering when the multi-stream allocator is written; not
   worth porting the annealer for it.

**What does not transfer:** per-tensor mixed bit-rates (`bits_prop`). Our weights are a fixed artifact
behind a bit-identical gate (`AGENTS.md`: never copy/move weight files), so we cannot re-allocate bits
at all. `03` already reached this conclusion; this round confirms it from the source.

### B2. KV cache — exllamav2 is **not** the paged/quantised-in-kernel design, and its Q-cache is a warning

* **The base cache is a single contiguous preallocated tensor**, `[batch, 2, max_seq_len, heads, dim]`,
  with `roll_left()` for context shifting and `copy_states()` for forks. **Not paged.**
* **Paging exists only as a thin delegation to FlashAttention-2**:
  `has_flash_attn_with_paged` (`attn.py:59`) with `page_size = PAGED_PAGE_SIZE = 256` tokens
  (`generator/dynamic.py:33`), a `block_table` and `cache_seqlens`. The policy is vLLM's; exllamav2
  only supplies the table. **No Vulkan equivalent, and none needed** — our KV is per-block contiguous
  and our slot count is a capacity question (`09` §A4), not a fragmentation question.
* **The Q4/Q6/Q8 cache is the counter-example to copy from, not the design to copy.** `ExLlamaV2Cache_Q`
  stores grouped-RTN `uint8` K/V plus scales, and `get_kv_state` **dequantises the whole cache up to
  `max_cache_seqlen` into f16 temp tensors** before attention
  (`cache.py:472-514`; called at `attn.py:500` with `0, attn_params.max_cache_seqlen`). So every token,
  every layer, pays an **O(ctx) f16 materialisation** — and the authors say so:
  `attn.py:493-496`: *"Would need custom kernel to update paged cache if not calling
  `flash_attn_with_kvcache`… TODO: Potentially we could emulate paged cache when in Q4 mode, since that
  requires copying the active part."* This is exactly the shape `09` §A4 measured as wrong: at 32 K/64 K
  the dequant pass alone is O(ctx × 12 layers) of f16 writes. **The right form is dequant inside the
  attention kernel** (`09` §B4's `ggml_vk_fa_scalar_uses_mmq`), and exllamav2 is evidence for that by
  being the negative case.
* **One transferable mechanic: quant-block alignment to whole tokens.** `Q_CACHE_BLOCKSIZE_Q = 512`
  elements per quant block, and `q_block` tokens are grouped so that `kv_dim × q_block % 512 == 0`
  (`cache.py:449-451`), because "models with odd key/value dims need to quantize/dequantize in
  multi-token blocks". For our QSA geometry (`2 KV heads × 256 = 512` elements/token) `q_block = 1` —
  the constraint is satisfied trivially, but the *rule* is the one to write down before implementing
  q8 KV, and it also bounds how a page/block boundary may be chosen.

### B3. Dynamic batching + selective masking — the pattern is "frozen (B,Q) geometry + a mask"

`ExLlamaV2DynamicGenerator` (`generator/dynamic.py`) schedules arbitrary jobs against a cache with
`batch_size = 1` by treating the **graph shape as fixed**:

* `max_batch_size` jobs × `max_q_size = 8` tokens per job per step (`:249`, `:362-371`) — every step's
  tensors are allocated `(max_batch_size, max_q_size, …)` (`:419`, `:426`, `:431`), so **the graph never
  changes shape**: a job feeding a 300-token prompt chunk and a job decoding one token occupy the same
  allocations and the same kernels.
* Per-job variation is carried by **`cu_seqlens` + a mask**, not by geometry: `seq_len` per job,
  `past_lens` per sequence, and `build_single_attn_mask` constructs a dense additive `-65504` mask
  combining the causal triangle with an optional `input_mask` (`attn_params.py:109-122`).
* For **independent sequences packed into one attention** it uses `get_block_diag_mask`
  (`attn_params.py:190-198`): `labels = searchsorted(cu_seqlens[1:], positions)`, then
  `mask = where(label_i == label_j, 0, -inf)` — a block-diagonal mask that makes one attention call
  serve N unrelated sequences. `block_diag_layers` restricts it to the layers where it is safe.
* When `seq_len == 1` the mask is **not built at all** (`build_attn_mask` returns `None`) — the decode
  fast path pays nothing for the mechanism.

**Transferable, and it is the concrete form of `09` §B6's recommendation.** Our answer to "one graph
per step with all streams in the batch dimension" is exactly this: freeze the per-step geometry at
`(n_streams, q_max)`, carry per-stream length in a `cu_seqlens`-style small buffer, and mask in the
attention kernel. Two specifics worth taking:
* **The mask must be free in the decode path.** `seq_len == 1 && input_mask is None` ⇒ no mask, no
  branch. Our multi-stream path always has `q = 1` per stream at decode, so the mask should reduce to a
  `cu_seqlens` lookup and nothing else.
* **`i32`-free block-diagonal trick:** `searchsorted` + compare is one integer op per element; it fits
  in a push-constant-sized helper buffer. No new pass.

**What does not transfer:** the scheduling policy itself (jobs, pages, `active_jobs`) is a server concern
that `07`/`04` already rank above this for us.

### B4. Speculative decoding — draft model + n-gram, and why it is a later item for us

* **Draft model:** `max_q_size` is the verify width, `num_draft_tokens = 4` by default
  (`dynamic.py:252`), the draft cache **must be `batch_size 1` and the same `max_seq_len`**
  (`:348-351`), and the returned per-job stats are `accepted_draft_tokens` / `rejected_draft_tokens`
  (`:967-968`). The verification is batched as a `q_len = 1 + num_draft_tokens` pass, i.e. it reuses
  the same frozen-geometry machinery as B3 — **speculative decoding in exllamav2 is a *scheduling*
  feature, not a kernel feature.** That is the transferable insight.
* **N-gram / prompt-lookup draft** (`iterate_ngram_gen`, `:1009-1030`): a per-job **trie of n-grams**
  built from the sequence as it grows, `max_ngram` long, drafted at zero model cost. This is the
  cheapest possible speculator and it composes with B3 (draft tokens are just more `cu_seqlens` rows).
* **Our position:** `09` and `05` both leave spec decoding aside because a *sinusoidal* win here would
  be a win over a broken baseline — at 57.7 ms/token with 32.5 ms of launch serialisation, the accepted
  fraction is irrelevant. Revisit **after** D1–D5: then the ratio `accepted/q_len` becomes the metric,
  and the n-gram trie is the cheap first step (no second model, no second cache, no extra VRAM). We
  already have `crates/breccia-draft` — check whether it is a draft *model* or an n-gram trie before
  scheduling this.

### B5. MoE dispatch — one C++ call over an array of expert handles, gated on a tiny batch

`moe_mlp.py:222-253`: `forward()` takes the CUDA path only when
`batch_size * sequence_length <= 4` **and** `num_experts ∈ {4, 8, 16, 128}` **and** no LoRA; otherwise
it falls back to `forward_torch`, which loops `for expert_idx in range(num_experts)` and calls three
`ExLlamaV2Linear` forwards per expert (`:291-311`). The CUDA path passes
`[w.q_handle for w in self.w1/w2/w3]` — **all experts' quantised weights as arrays** — plus
`num_experts_per_token`, i.e. the expert id is a *kernel-side index*, not a host-side dispatch decision.
That is `09` §B3's `MUL_MAT_ID` from a third engine, and it is **direct evidence for D2**: the shape is a
single dispatch, and the `<= 4` gate tells you the crossover is about *rows per expert*, not about
expert count.

### B6. Ranked adaptations

| # | Adaptation | Value | Cost | Risk |
|---|---|---|---|---|
| 1 | **Per-block measured (bytes, lane-ms) table → exact knapsack placement**, re-solved from free VRAM at load. §B1. | High — this *is* §8am item 3b, with an objective and a method | S | low (measurement only) |
| 2 | **Frozen `(n_streams, q_max)` step geometry + `cu_seqlens` + block-diagonal mask.** §B3 | High for multi-stream; it is the structural version of our hand-rolled interleave | M | low (mask-free at `q=1`) |
| 3 | **Expert-indexed MoE dispatch** (converges with D2, `09` §B3, `01`) | High | M | medium |
| 4 | **Quant-block ↔ token alignment rule** before any KV quant work. §B2 | Guards the q8 KV implementation | XS | none |
| 5 | **N-gram trie speculator** after D1–D5. §B4 | Unknown, cheap to try | S | low |
| 6 | Draft-model speculation, paged KV, `bits_prop` mixed rates | Low / not applicable | — | — |
| 7 | **Do not** copy the Q-cache dequant-to-f16-per-token design (§B2) | — | — | — |

---

## A7. Top-3 experiments for the next implementation round

**E0 (do this in the first two minutes of a GPU window — it re-prices everything else).**
Resolve A0 with the engine's own census, no new code, one command, run on the **headless RX**:

```
timeout 300 env BRECCIA_TRACE_DISPATCH=1 ./target/release/breccia bench token \
  --devices 0000:03:00.0 --prompt-ids 760 --gen 2 --runs 1 2>&1 | grep "dispatch census"
```
The per-stage printout is a proper delta (`graph.rs:145-151`), so it is the ground truth. Then re-run
the v1/sync2 A/B **adjacent and interleaved 3×** (`BRECCIA_BARRIER=full` vs default) on an idle card —
it is the only measurement that can separate "6.7 µs per barrier" from "1.1 µs per barrier + cold
loads", and the round's decision rule flips on it. Expected: ~1.9 k/token and a ≤1.6× wall delta, not
1.45× from a contaminated window.

**E1 — D1 + D3 + D4 in one round (three independent files, each with its own flag).**
`gate+up+SwiGLU` for the shared and routed paths; the multi-projection bundle for GDN's four and QSA's
five input projections; `conv+qk-l2+beta-gate` and `rope+KV-append` bundles. Gate: **bit-identical**
digests on every block for one token (`--debug-layers`) — D3 in particular must be exactly bit-identical
because it changes only an address — then 32/32 greedy parity vs the HIP server, then the census. Expected
**32 → 17 dispatches per GDN block, 41 → 22 per QSA block** (`A3`); measure `wait-gpu` and confirm the
sign before adding D5.

**E2 — D5, the `hc` collapse, alone and last.** It is the biggest single item (−6 to −8 dispatches per
block) and the one with real numerical risk, so it gets its own round and its own flag. Start with the
two cheap fusions (`act+up`, `gated+write`) which are structurally identical to D1/D4, then decide on
norm-recompute-inside-`down` from the measured trade (`1.1 µs` of barrier vs `~2 µs` of L2 re-read).
Gate: per-block digests, then parity, then census. Expected **17 → 10–11** per GDN block, ≈ **600/token**.

Grouped-round notes (per `AGENTS.md`): E1's three items touch different shaders and record at three
different stages, so one alternating sweep attributes them; E2 must not share a window with E1 because
both change the same block's critical path.

**Not scheduled, on purpose:** frozen command-buffer replay (D7) — it is worth ~1 ms and every E1 fusion
invalidates it; indirect dispatch (D6) standalone — measured negative; mega-kernels (D8) — a hang is a
stop-the-line event and the legal subset is D1/D4. The descriptor work (`A4`) is **D0**: do it *inside*
E1 because it makes iterating on fusions cheaper, but do not count it as a win.

---

### Sources

* Our code: `crates/breccia-kernels/src/{dispatch_batch.rs,batch.rs,gemv.rs,proj.rs,gdn/mod.rs,qsa/mod.rs,ops/mod.rs}`,
  `crates/breccia-engine/src/{graph.rs,config.rs}`.
* Capability surface: `<SCRATCH>/opencode/vinfo.txt` + `rx_ext.txt` (vulkaninfo, all
  three devices, taken 2026-10-27 00:05 — **not re-probed this round**); driver string check on
  `/usr/lib/libvulkan_radeon.so` for the same extension set.
* Live numbers: `<SCRATCH>/opencode/p4-token/an32.json` (17.19 tok/s, 58.16 ms/token,
  `wait-gpu` 32.47, lane 13.36).
* Prior rounds: `04` (launch count, indirect), `06`/`06a` (barrier ladder, `bpr` law, fusion prices),
  `07` (attribution, timestamp rules), `08` (submit/CB economics, secondary-CB barriers, replay),
  `09` (zero-copy, KV quant, `ggml_vulkan` internals).
* Engines: `Reference Projects/llama.cpp-ggml-vulkan` @ `7aca0d2` (`ggml-vulkan.cpp:14865-15140`,
  `:6933-6964`); `Reference Projects/ncnn` (`src/command.cpp:1260-1440`);
  `Reference Projects/exllamav2` @ `7dc12af` (`conversion/{measure,optimize,quantize}.py`,
  `cache.py:409-560`, `attn.py:466-520`, `attn_params.py:29-200`, `generator/dynamic.py`,
  `moe_mlp.py:222-320`).
