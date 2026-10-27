# Round 18: FlashInfer (primary) and FasterTransformer MoE and fused kernels (secondary)

Research only. No code was changed, built, run on a GPU, or benchmarked. Public sources were cloned to scratch, read, and not executed.

Pinned sources: FlashInfer `bdf84a181276f5012789b224b74b36297cff7fdc` (2026-10-27); FasterTransformer `df4a7534860137e060e18d2ebf019906120ea204` (2023-10-19, frozen). Breccia paths are repo-relative at this branch's base `7a4cf83`. Items the docs already settle (Wave32, cooperative matrix, DGC, mega-kernel, fp16 accumulate, q8_1 activations, descriptor buffers, KV quant as a speed lever, tensor parallel) are not re-proposed.

## 1. Summary

The two engines differ on what matters for Breccia. FlashInfer's split-KV rule and cascade merge target long KV attention. Breccia's QSA attention is context-independent: each token attends to 4·k cells plus its own block, with k ≤ 512. The context-dependent cost is the indexer (`qsa_idx_score` plus `qsa_topk`). The `k`-pass argmax in `qsa_topk.comp` runs 10 workgroup barriers per pass and 512 passes per QSA layer per token once the context exceeds 2,048 tokens. That is a count; its wall-time cost is UNVERIFIED. On the MoE side, Breccia builds three expert maps per placed block (gate, up, down). The up map repeats the gate map when the two batches match in type, rows and columns, and FasterTransformer reuses one offset table for both expert GEMMs. The sampler fully sorts the vocabulary on the stochastic path. That path is correct and off the greedy parity path.

## 2. Top 3

1. **QSA indexer: sort-free exact top-512 by radix select, split across workgroups (section 3, R1).** Cuts barriers per QSA layer from 5,120 to about 75 on the critical path, and element reads at 100K context by about 68x (derived counts). Card: QSA long-context. Effect UNVERIFIED.
2. **MoE: one expert map for gate and up when types match, then one gate+up dispatch with the SwiGLU fused in (section 4, R2).** Takes dispatches per placed block from 8 to 5, or 240 to 150 per token (derived). Card: phase-4/02 §7 and merged chunked prefill. Effect UNVERIFIED. The gated fold has no precedent in either reference.
3. **Sampler: partial selection instead of a full sort on the stochastic path (section 5, R3).** Gives bit-identical draws for a given seed, with roughly 6–9x fewer comparisons at top_k 20 (derived from an assumed 2–3·V select cost, UNVERIFIED). Host-only and off the greedy path, so no token-time gain. Also flags a real gap: top_p is not implemented, and the model's metadata sampling values are not applied. That gap is a product decision, not an engineering card.

## 3. Question (a): attention scheduling

### What the code does

- Per QSA layer, `qsa_idx_score.comp` writes `score[t][b]` for every pooled block (`crates/breccia-kernels/shaders/qsa_idx_score.comp:5`). It reads f32 pooled keys (`:29`, `idx_dim` 128, 4 index heads), one 128-thread workgroup per (token, block). Then `qsa_topk.comp` picks the top k blocks. Attention runs over those cells plus the own block (`docs/phase-3/06-attention-gdn.md:118-134`).
- `qsa_topk.comp:69-74` assigns one workgroup per token with 256 threads. The loop at `:78` runs `want = min(n_blocks, topk)` passes. Each pass has a strided scan of `n_blocks` (`:81`), a barrier after the seed write (`:90`), an 8-level tree reduction with a barrier per level (`:100`), and a barrier after the write (`:110`). That is 10 barriers per pass, so 5,120 per layer per token whenever n_blocks ≥ 512 (any context above 2,048 tokens). Across 12 QSA blocks, that is 61,440 barriers per token. The barrier count is flat beyond 2K; the reads are not (derived counts, section 3 R1).
- Derived at 100K context (n_blocks = 25,000): the selection makes 512 × 25,000 = 12.8 M score reads per layer. The score buffer is 100 KB f32, so these are L2 reads. That is 153.6 M per token over 12 layers. The scoring pass reads 25,000 × 128 × 4 B = 12.8 MB of pooled keys per layer, or 153.6 MB per token (f32 as declared at `qsa_idx_score.comp:29`; whether a packed copy exists on the GPU path was not checked).
- `qsa_topk.comp:39-41` says radix-select is needed only at n_blocks = 65,536. The cost is k·n_blocks, which is already 12.8 M reads at 100K. The trigger should be the read count, not the block count alone.
- At T = 1 (single-stream decode), each QSA layer's selection runs in one workgroup.

### Reference points

- FlashInfer `include/flashinfer/attention/scheduler.cuh:145-212`: split KV work only when `batch·gdy < max_grid_size` (`:201`), that is, when the grid would be under-filled. This is the rule to apply to the indexer's chunk count at T = 1.
- FlashInfer `include/flashinfer/attention/cascade.cuh:44-70`: base-2 LSE merge of two attention states, with a -inf guard. Not needed here, since QSA has one softmax over at most 2,048 cells per token. See section 6.
- FlashInfer `include/flashinfer/sampling.cuh:1230-1354`: sort-free dual-pivot bracket for top-k. Each round is a full pass (`while (low<high)` at `:1354`). This is the same primitive R1 needs.
- FasterTransformer `src/fastertransformer/kernels/sampling_topk_kernels.cu:130-160` (stage 1: per-block partial top-k with cub BlockReduce, no sort) and `:205-300` (stage 2: k argmax passes over the partials). The two-stage partition is the part to copy.

### R1: sort-free exact select, split across workgroups

Mechanism:
1. Stage 1 runs one workgroup per chunk of pooled blocks. The chunk count C is chosen so that T·C fills the GPU's workgroup slots, following the rule at `scheduler.cuh:201`. Each chunk finds its own top-512 with a radix select on monotone uint keys derived from the float scores. That is four 8-bit digit passes, each with a 256-bin shared-memory histogram, then a compaction that emits the scores above the pivot and the lowest-index ties at the pivot. The tie emit must be index-ordered: a prefix scan over block indices in chunk order, not an atomic append, because an atomic append picks arbitrary ties and changes the IDS set. The bitonic sort in step 4 fixes order, not membership, so it cannot repair a wrong set. When T alone fills the workgroup slots (the split condition at `scheduler.cuh:201` fails), C = 1 and stage 2 is skipped, so multi-stream decode uses the chunk result directly.
2. Stage 2 runs one workgroup that radix-selects the global top-512 from C·512 candidates. At C = 25 and 100K context, that is 12,800 candidates. The result is exact, because every global top-512 element is in its own chunk's top-512.
3. Tie-break stays lower index wins, as in `qsa_topk.comp:83`. FlashInfer packs `(value, 65535 − idx)` into one 64-bit key (`csrc/fused_moe/moeTopKFuncs.cuh:35-75`). That packing holds only up to 65,536 blocks, the same limit the header names. Use it only below that limit.
4. Output order. The current shader writes in descending score order (`:106`). The radix output is an unordered set. To keep dispatch 17 (softmax over slots) unchanged, sort the 512 keys in shared memory with a bitonic network before writing. That is 45 compare-exchange stages for 512 keys, derived as log2(512)·(log2(512)+1)/2. Without that sort, the softmax reduction order changes and the output is not bit-identical.

Derived counts per QSA layer per token at 100K:
- Barriers on the critical path: stage 1 about 14 (4 digits × about 3, plus compaction), stage 2 about 14, bitonic 45. Total about 75, versus 5,120 now.
- Element reads: stage 1 about 5 × 25,000 = 125,000, stage 2 about 5 × 12,800 = 64,000. Total about 0.19 M, versus 12.8 M now.
- Dispatches: one extra per QSA layer, 12 per token. At the 66 ns per dispatch recording figure from `docs/research/11-record-floor-and-mlc-vulkan.md` Part A, that is about 0.8 µs of recording per token. The cost that matters is the GPU-side dependency between stage 1 and stage 2, which is UNMEASURED: the round-10 per-barrier figure is flagged unsafe in the docs, so no time is assigned to it here.

Expected effect: UNVERIFIED - requires the GPU qualification card. The derived counts bound the algorithmic work only. The kernel's share of the 53.78 ms token is not measured.

Card: QSA long-context (the radix-select open item in `docs/phase-3/06-attention-gdn.md`). It helps because the selection cost scales with context, and it matters at the 100K-context point used across this docs set.

Measurement plan (for the qualification card; not run here):
- Correctness first, on the CPU reference in `crates/breccia-kernels/src/qsa/cpu.rs`, then on the GPU. Use random scores with planted ties and dead blocks at -1e30, n_blocks in {1, 511, 512, 513, 6,250, 25,000, 65,536}, with and without the own-block append (`qsa_topk.comp:116-121`). Gate: the IDS array is byte-identical to the current shader's.
- Timing: GPU timestamps around the current `qsa_topk` and the new stages, at T = 1 and T = 8, at 25,000 blocks, on the RX and the MI50. Keep the change only if the median time falls.
- End to end: one 100K-context decode token with QSA on, A/B against the current path.

## 4. Question (b): fused MoE

### What the code does

- `run_placed_moe` (`crates/breccia-engine/src/graph.rs:3324-3372`) issues 8 dispatches per placed block: map_gate (`:3342-3343`), gate.run (`:3345`), map_up (`:3348-3349`), up.run (`:3351`), swiglu (`:3353`), map_down with per_expert = true (`:3357-3358`), down.run (`:3360`), and moe_combine (`:3362`). Callers are decode at `graph.rs:1926` and the prefill fallback at `:2340`.
- The gate and up maps are built from the same `v.router_ids`, the same k, and per_expert = false (`:3343`, `:3349`). The comment at `crates/breccia-kernels/src/moe.rs:40-45` says the separate map exists because the gate and up tensors "need not share a type", and nothing enforces that. `docs/specs/model-artifacts-inventory.md` §4.2 says this artifact's blocks do share types. Line 245 lists per-layer type choices, so per-block equality was not checked.
- FasterTransformer computes the expert offsets once, `src/fastertransformer/kernels/moe_kernels.cu:718-721`, and reuses `total_rows_before_expert_` for fc1 (`:725`) and fc2.
- `docs/phase-4/02-untried-levers.md` §7 (`:323-325`) describes the pipeline as "MoeMap → per-expert GEMV → SwiGLU → down → combine". The code has three map builds, not one. §7 should say so.
- FlashInfer `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`: the gated activation is a separate kernel (`doGatedActivation`, `:2371-2415`). A single-block radix rank for at most 256 tokens builds the expert rows without atomics (`fusedBuildExpertMapsSortFirstTokenKernel`, `:411-497`, limit `:555-557`). `finalizeMoeRoutingKernel` (`:1963-2040`) sums the k experts in fixed order with no atomics.
- FasterTransformer `src/fastertransformer/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h:834-886` folds the activation into the fc1 epilogue, and only for non-gated fc1.

### R2: one map, one gate+up dispatch

Mechanism:
1. Build one expert map per placed block when the gate and up batches match in `ty()`, `rows_per_expert()` and `cols()`. Those three fields set the map's contents: `block_words(ty)` and `blob_words()` come from the quant type, and `blocks_per_row()` from `cols` (`crates/breccia-kernels/src/moe.rs:21-75`, `crates/breccia-kernels/src/batch.rs:181-212`). Both call sites pass `base_words = 0` and `per_expert = false` (`graph.rs:3342-3349`), so nothing else differs. Check the three fields at load time rather than assuming them. Drop map_up (`graph.rs:3348-3349`) and share the gate map buffer.
2. Fold gate.run, up.run and swiglu into one dispatch. Each workgroup takes an (expert, row-tile) pair, computes the gate rows and the up rows from two weight bindings (`pe.gate.buf`, `pe.up.buf`), and writes `silu(g)·u` into `moe_act`. This uses two ordinary bindings, not descriptor buffers, so the settled item is not reopened.

Derived counts: per placed block, 8 → 7 (drop map_up) → 5 (gate.run, up.run and swiglu become one dispatch). BRIEF places 30 blocks per token (`docs/BRIEF.md`), so 240 → 150 dispatches per token (−37.5%).

The gated fold has no precedent in either reference. FlashInfer and FasterTransformer both keep the gated SwiGLU out of the GEMM epilogue. R2 is therefore a Breccia-specific bet and must be measured as one.

Dispatch cost: 240 dispatches × 66 ns is about 16 µs per token of recording, so recording is a small share. Any gain is execution time of the small kernels and the barriers between them. Time UNVERIFIED.

Relation to phase-4/02 §7 (`:318-339`, fuse device MoE into one dispatch, expectation 2 ms → 1.0-1.2 ms per block). R2 is a smaller step toward the same target. It is new in the map count and the gate+up fold, and it corrects §7's description of the pipeline.

Merged chunked prefill (phase-4/02 §2, `:102-113`; the counting sort uses two atomicAdds at `:108-109`): FlashInfer's single-workgroup rank for chunks up to 256 tokens (`:411-497`) replaces the atomics with a deterministic rank. This is a sub-item of R2 for the prefill card. Effect UNVERIFIED.

Expected effect: 240 → 150 dispatches per token (derived). Time UNVERIFIED - requires the GPU qualification card. The gate, up, and down budget is about 3,700 µs per token (BRIEF), so the share taken by the map and swiglu dispatches is the first number to measure.

Measurement plan:
- Timestamp each of the 8 dispatches per placed block at T = 1 on the RX and the MI50. The existing `MOE_MAP` probe covers the whole MoE at the call site (`graph.rs:1926` area), so per-dispatch stamps need one more probe each.
- Keep the 8-dispatch path whenever the types differ.
- Parity: the shared map is data-only and should be bit-identical. The fold changes epilogue order, so gate it with the round-12 accuracy gate (per-stage rel L2 ≤ 2e-3, top-1 agreement, logit delta ≤ 0.02, PPL +0.3%).

## 5. Question (c): sampling

### What the code does

- Greedy path: `Sampler::argmax` (`crates/breccia-engine/src/sampler.rs:76`), O(V), NaN skipped, lowest index wins. Default `Sampling` is temperature 0 and top_k 0 (`:25-32`). The CLI describes `0` as "greedy, the parity path" (`crates/breccia-cli/src/main.rs:286-288`).
- Stochastic path (`sampler.rs:92-123`): builds `order` of all non-NaN indices (`:101`), then does a full `sort_by` with partial_cmp and an index tie-break (`:105-110`). It truncates to top_k (`:111-113`), computes an f64 softmax (`:114-119`), and draws with xorshift64* (`:123`).
- There is no top_p. The model metadata `general.sampling` values top_k 20, top_p 0.95, temp 1.0 (`docs/specs/model-artifacts-inventory.md:116`) are not applied.
- Logits readback is vocab × 4 = 993,280 B per token (`graph.rs:684` comment, read at `:3275`).

Derived: the full sort is about n·log2 n = 248,320 × 17.9 ≈ 4.45 M comparisons per sampled token. The readback is 993,280 B per token and sits on the critical path of every token, so the cost is the per-token transfer time, not the average rate. At the 22–30 GB/s host-transfer rates in the round-09 verdict (measured in the opposite direction, GPU reads from host, so this is a derived rate, not a readback measurement), the copy takes about 33–45 µs, roughly 0.1% of the 53.78 ms token.

References (GPU-side; not needed on the host): FasterTransformer `src/fastertransformer/kernels/sampling_topp_kernels.cu:209-550` (segmented single-pass top-p, radix bit-mask refinement, `MAX_TOP_K` 3072 at `:206`). FlashInfer `include/flashinfer/sampling.cuh:869` (top-k from probs) and `:1004` (top-p from probs).

### R3: partial selection on the stochastic path

Mechanism: for top_k > 0, call `select_nth_unstable_by` with the existing comparator (score descending, index ascending), then sort only the k survivors. The comparator is total because NaNs are filtered and the index breaks ties. The set and the order therefore match the full sort, and the draw for a given seed is unchanged. Cost is O(V) for the select (introselect, about 2–3·V comparisons; constant UNVERIFIED) plus O(k log k) for the sort, instead of O(V log V). It uses the standard library, with no new dependency. For top_k = 0, keep the full sort, because the CDF walk needs the whole sorted order.

Expected effect: only the stochastic path changes. The greedy argmax is untouched. Derived: at V = 248,320 the select is about 0.50–0.75 M comparisons (the 2–3·V constant is assumed), plus about 90 to sort the 20 survivors, against 4.45 M for the full sort: roughly 6–9x fewer. It runs on the host, after readback, outside the 13 ms lane (BRIEF `:98`), so there is no decode token-time gain. The host time is UNVERIFIED.

Gap, not an engineering card: top_p is absent, and the metadata top_k and top_p are not applied. Whether to apply the model's recommended sampling is a product decision for the owner. If top_p is wanted, FasterTransformer's single-pass radix top-p (section 5 references) is the reference.

Measurement plan: a bit-identical draw test, old path against new, over seeds 0 to 999, top_k in {1, 20, 40}, V = 248,320, with planted ties and NaNs. Time 100 calls of each path on the host. CPU only.

## 6. Dispositions

| Item | Source (repo path at pinned SHA) | Disposition |
|---|---|---|
| Split-KV decode partition rule | FlashInfer `include/flashinfer/attention/scheduler.cuh:145-212` | Folded into R1 for the indexer only. QSA attention is not split. |
| Cascade LSE merge | FlashInfer `include/flashinfer/attention/cascade.cuh:44-70` | Not adopted. No multi-chunk attention exists. Revisit if attention is ever split. |
| Paged KV | FlashInfer `include/flashinfer/page.cuh:38-60` | Match. Breccia already pages f16 KV. No action. |
| Prefill flat work list | FlashInfer `include/flashinfer/attention/scheduler.cuh:550-690` | Reference for the merged chunked prefill card. No action now. |
| Packed top-k tie key | FlashInfer `csrc/fused_moe/moeTopKFuncs.cuh:35-75` | Used in R1 only below 65,536 blocks. |
| Sort-first expert rows, ≤256 tokens | FlashInfer `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh:411-497` | Sub-item of R2 for prefill. Effect UNVERIFIED. |
| Min-latency expert maps | FlashInfer `…cutlass_fused_moe_kernels.cuh:271-376` | Decode T = 1 case. Breccia's decode path is already this minimal. No action. |
| Routing finalize, fixed k order, no atomics | FlashInfer `…cutlass_fused_moe_kernels.cuh:1963-2040` | Matches `moe_combine` in shape (`graph.rs:3362`). Shader summation order not read. |
| Gated activation kept separate | FlashInfer `…cutlass_fused_moe_kernels.cuh:2371-2415`; FT `moe_gemm_kernels_template.h:834-886` | Counter-evidence to R2's fold. Measure before adopting. |
| Expert offsets computed once, reused for fc1 and fc2 | FT `src/fastertransformer/kernels/moe_kernels.cu:534-546, :718-740` | Precedent for R2's shared map. |
| Residual fused into finalize | FT `src/fastertransformer/kernels/moe_kernels.cu:891-935` | Candidate only. The block output goes to `hc.write` (`graph.rs:1932`, `:2343`). Whether that call adds the residual as a separate dispatch was not read. |
| Fused top-k gating softmax | FT `src/fastertransformer/kernels/moe_kernels.cu:173-200` | Matches the router design in `docs/phase-3/04-op-kernels.md:176-189`. Router shader not read this round. |
| Persistent grouped GEMM | FT `moe_gemm_kernels_template.h:156-158` | Already in phase-4/02 §7. Not re-proposed. |
| Two-stage top-k | FT `src/fastertransformer/kernels/sampling_topk_kernels.cu:130-300` | Adopted in R1. |
| Top-p single-pass radix | FT `src/fastertransformer/kernels/sampling_topp_kernels.cu:209-550` | Reference only, if top_p is wanted. |
| Sort-free top-k bracket | FlashInfer `include/flashinfer/sampling.cuh:1230-1354` | Primitive for R1 on the GPU. Not needed on the host (R3). |
| Full vocabulary sort | `crates/breccia-engine/src/sampler.rs:105` | R3. |
| Logits readback, 993,280 B | `crates/breccia-engine/src/graph.rs:684, :3275` | Derived bound: about 33–45 µs per token (about 0.1%) at round-09 host rates (opposite direction). The only lever is GPU-side argmax with a 4-byte readback, and its gain is capped near 0.1%. No card proposed. |
| Settled items | Docs list | None proposed. R2 uses ordinary bindings, not descriptor buffers. |

## 7. Unverified

- Wall time of `qsa_topk` and `qsa_idx_score` at any context. The round-10 per-barrier figure (6.7 µs) is marked unsafe in the docs, so this round uses counts only.
- Whether a packed f16 copy of pooled keys exists on the GPU decode path. The shader declares f32.
- Whether the batch keys `pe.gate.key` and `pe.up.key` carry `ty`, `rows_per_expert` and `cols`. `pipes.batch` (`crates/breccia-engine/src/model.rs:1069`) was not read. The load-time check in R2 step 1 covers this, but the check is not yet written.
- Whether gate and up share a type in every block. Inventory §4.2 claims it; line 245 lists per-layer choices.
- Whether the streamed-expert path (the 18 blocks that are not placed) uses the same 8-dispatch sequence. Not traced.
- Whether `hc.write` (`graph.rs:1932`) adds the residual as a separate dispatch.
- Breccia's router tie-break and the summation order in `moe_combine` at the shader level.
- The residual finalize fusion candidate in section 6.
- Not read: FlashInfer `topk.cuh`, `noAuxTcKernels.cu`, `RoutingKernelTopK.cuh`, `flashinfer/fused_moe/core.py`, `sampling.py`, and the `decode.cuh` partition condition (around `:700-745`). FT `moe_problem_visitor.h` and the FT sampling kernel bodies beyond the cited lines. The full body of `qsa_topk.comp` beyond the sections cited (its header and selection loop were read).
- Every timing claim. All effects in R1 to R3 are UNVERIFIED until the GPU qualification card runs them.

## 8. Round verdict inputs

- The QSA indexer's cost is a count of 512 passes times 10 barriers per layer, and its barrier count is flat in context above 2K (the reads grow linearly with n_blocks). Radix select with a canonical 512-key sort gives about 75 critical-path barriers.
- The MoE lane has three map builds where one suffices when types match. Dispatches per token go 240 → 150 at best, but the gated fold is unprecedented.
- The sampler's full sort is off the parity path and costs about 4.45 M comparisons per sampled token. Partial selection keeps draws identical. The top_p gap is a product question.
