# 20 — DeepSeek open-source stack (FlashMLA, DeepGEMM, 3FS, EPLB, DeepEP) for Breccia

Round 20 of the paired rotation. Research only: no code was changed, and no GPU, binary or benchmark was run. Every speed number is either cited, marked `derived` (arithmetic on cited inputs), or marked `UNVERIFIED - requires the GPU qualification card`.

## 1. Summary

- **Most of the DeepSeek stack is CUDA-scale or multi-node.** The five repositories target SM90/SM100 tensor cores, NVLink/RDMA clusters, or a distributed file system. None of their kernels can run on RADV, which has no `cooperative_matrix`. Their transferable ideas are mostly already in Breccia or already carried in `docs/research`.
- **The one real gap is the K side of 8-bit KV.** Breccia's V cache already has a q8_0 variant behind `BRECCIA_KV_Q8V`. K stays f16. BRIEF projects 6 agents at 32K with K+V quantised, against 4 today (BRIEF L76; a projection, not a measurement). FlashMLA's 528 B/token layout is a format precedent for quantising K, RoPE dims included.
- **Indexer low-precision scoring does not pay at 32K.** Derived: about 0.086 ms per token at 32K, which is about 0.16% of the 53.78 ms token and about 0.10% of the 84.2 ms per-stream token at 32K (BRIEF L75; the 53.78 ms context is UNVERIFIED) (§4, settled rows).
- **A gap I found in the QSA path.** `qsa/mod.rs:1062-1063` runs the indexer scoring and `topk` on every QSA layer, including when the whole cache is selected (at most 2051 cells). A bypass changes the block order and so the softmax summation order. It needs a parity gate, and it helps only at short context.
- **Top 3.** (1) K-side 8-bit KV, carried in direction. (2) Frequency-per-byte greedy expert packing, EPLB-style, carried in spirit. (3) A short-list bypass for the indexer and `topk` at 2051 cells or fewer, new. Overlaps are declared in §5.
- **Search result.** No dedicated audit of the DeepSeek stack exists in `docs/research`. DeepSeek is mentioned only incidentally.

## 2. Sources

Fetched 2026-10-27. Each README was read through the fetch tool and checked against the raw file. Line numbers below are raw-file lines. Breccia code anchors are given inline as `file:line` and were checked in this round.

| ID | Source | What it supplies | Anchors |
|---|---|---|---|
| S1 | FlashMLA README, `https://github.com/deepseek-ai/FlashMLA` | Sparse decode with an `indices` tensor (`flash_mla_with_kvcache`); sparse prefill (`flash_mla_sparse_fwd`, value dim 512, `attn_sink`, `topk_length`); fused norm-RoPE-attn-RoPE-cast decode kernel; KV layouts (V4.1 528 B/token = 512 B FP8 E4M3 plus 16 B E8M0 scales, one per 32 values; FP4 288 B); requirements SM100/SM103, CUDA 13.1+ | L13, L17 (fused kernel "around 10% - 15%", 2026.09.30), L26, L29, L51–63 (vendor "up to" figures, B200), L140–143 (KV layout), L152 (`indices_in_kvcache` = block × page size + offset) |
| S2 | DeepGEMM README, `https://github.com/deepseek-ai/DeepGEMM` | FP8/FP4 GEMM set: dense; contiguous and masked grouped MoE; MQA logits for the V3.2 indexer (`fp8_fp4_mqa_logits`, paged variant); Mega MoE; "Mega mHC" listed in the 2026.09.10 note with no description | L14 (2026.09.10 note), L19 (indexer scoring), L25 (1550 TFLOPS on H800, no shape or precision), L31 (SM90/SM100, CUDA 12.9+), L88 (MQA logits), L112–125 (Mega MoE) |
| S3 | 3FS README, `https://github.com/deepseek-ai/3FS` | Distributed FS: CRAQ chain replication, FoundationDB metadata, USRBIO shared-memory ring API (`src/lib/api/UsrbIo.md`, cited by path only, not read), FUSE, KVCache use case | L30 (180 storage nodes, about 6.6 TiB/s stress read), L40 (GraySort, cluster), L48 (KVCache peak 40 GiB/s, client count not stated) |
| S4 | EPLB README, `https://github.com/deepseek-ai/eplb` | `rebalance_experts`: heuristic packing of redundant experts; group-limited routing; hierarchical policy when server nodes divide expert groups; global policy otherwise | L5–7, L21, L27–29, L37 (2×12 example). No imbalance or throughput numbers |
| S5 | DeepEP README, `https://github.com/deepseek-ai/DeepEP` | EP all-to-all dispatch and combine over NVLink (intranode) and RDMA (internode); `lb_prefetch_weights` and `lb_reduce_grads` for dynamic redundant experts | L8 (Ascend claim, no numbers), L15, L49 (Hopper or newer), L332–343, L406 (`EP_DISABLE_GIN`), L452 (RoCE theoretical). No bandwidth or latency numbers |
| S6 | arXiv 2405.04434, DeepSeek-V2 (MLA) | Multi-head latent attention; abstract: KV cache cut by 93.3% (compared with DeepSeek 67B, not with MLA); maximum generation throughput 5.76× | abstract |
| S7 | arXiv 2412.19437, DeepSeek-V3 technical report | Auxiliary-loss-free load balancing (training); multi-token prediction objective | abstract only (per the fetched summary; not read in full) |
| S8 | arXiv 2512.02556, DeepSeek-V3.2 | Names DSA in the abstract. The lightning indexer and `k` were **not confirmed** from the abstract; HTML and PDF not read | abstract only; use as a reference for the DSA name, nothing more |

Local code is cited by path and line, and the Breccia doc rounds by number (doc 02 is `docs/phase-4/02-untried-levers.md`; doc 18 is `docs/research/18-flashinfer-and-ft-moe.md`; the others are `docs/research/NN-*.md`).

## 3. Mapping table

Verdicts: **already have**, **carried** (with the doc that carries it), **gap**, **does not apply**. A row marked "top 3 #n" is in §4.

| # | Idea (source) | Breccia equivalent or gap | Verdict | Reason |
|---|---|---|---|---|
| 1 | Sparse decode over an index list (S1, `flash_mla_with_kvcache` with `indices`, L152) | QSA: `qsa_topk` writes the ids and a count per token (`qsa_topk.comp`); `qsa_attn_apply` walks `CNT[t]` (`qsa_attn_apply.comp:97–107`) | already have | The attention walk is count-driven, so padded slots are not walked |
| 2 | Variable-length selection (`topk_length`, S1) | `CNT[t]` per token (the same mechanism as row 1); `max_ids = min(n_blocks, 512) + 1` (`qsa/mod.rs:665`) | already have | Same as row 1 |
| 3 | 8-bit KV with E8M0 scales per 32 values, RoPE dims included (S1 L140–143) | V side: q8_0 variant wired on HEAD behind `BRECCIA_KV_Q8V` (`build.rs:249–267`; `KV_Q8V_SHADERS` at `:267`; the define is in `qsa_kv_append.comp` and `qsa_attn_apply.comp`). K side: f16 (`placement.rs:306–318`, 2048 B K/V per cell) | carried (doc 02 #12, doc 09 A4; BRIEF L76, L169) | K side is missing. The FlashMLA layout is a new format precedent. It is a different model, so the byte ratio is not like-for-like. **Top 3 #1** |
| 4 | Fused norm-RoPE-attn-RoPE-cast decode kernel, "10–15%" (S1 L17) | Separate dispatches: `q_norm`, `rope_q`, `rope_k`, `kv_append`, `idx_pool` (RoPE inside), `idx_tail`, `idx_score`, `topk`, `attend`, `apply` (`qsa/mod.rs:1040–1063`) | gap, low value | The 10–15% is a CUDA decode figure. BRIEF L177 gives about 0.31 µs per removed dispatch (derived: 97 dispatches = 30 µs). Barrier rule 1.1 + 0.45 × bpr µs per boundary (BRIEF L115) |
| 5 | Sparse prefill attention, `flash_mla_sparse_fwd` (S1) | QSA prefill path | does not apply | 32K prefill is NVMe-streaming-bound (BRIEF L284, item 4b), not attention-compute-bound |
| 6 | Indexer scoring in FP8/FP4, MQA logits (S2 L88, L19) | `qsa_idx_score.comp` in float32, one 128-thread workgroup per (token, block) | gap, does not pay | Derived ≤0.1 ms per token at 32K (§4, settled rows) |
| 7 | Contiguous grouped FP8 MoE GEMM for prefill, `m_grouped_fp8_gemm_*_contiguous` (S2) | `prefill_block_merged` (`graph.rs:2566`) groups prefill work by expert | carried (doc 02 #7, row-map `MUL_MAT_ID`) | FP8 needs SM90/SM100 tensor cores (S2 L31). RADV has no `cooperative_matrix` |
| 8 | Masked grouped GEMM for decode under CUDA graphs (S2) | Command-buffer replay (BRIEF L180: 0.8–1.1 ms after fusion) | carried (BRIEF L180) | The replay lever is already priced |
| 9 | FP8 E4M3 / E8M0 block-scaled tensor-core GEMM (S1, S2) | int8 dp4a path rejected for accuracy (BRIEF L188: logits 0.51 off, greedy changes) | does not apply | No `cooperative_matrix`; the accuracy rule is already set |
| 10 | Fused EP dispatch, MoE and combine ("Mega MoE", S2 L112–125) | Per-block MoE dispatch exists; no EP all-to-all | carried for the single-dispatch part (doc 02 #7); EP part does not apply | Single host, no inter-device all-to-all |
| 11 | "Mega mHC" kernel (S2 L14, title only) | `hc-read` at 5,311 µs after the BRIEF L164 fusion (BRIEF L164: 7,142 to 5,311 µs, −96 dispatches) | already have in part; mechanism UNVERIFIED | The README does not say what the kernel fuses. If the 5,311 µs is a per-token figure, the stage is about 10% of 53.78 ms (derived: 5,311 / 53,780 ≈ 9.9%) |
| 12 | Lightning-indexer scoring kernels (S2 L19) | `qsa_idx_pool`, `qsa_idx_score`, `qsa_topk` | already have | Same as rows 1 and 6 |
| 13 | Multi-token prediction (S7) | MTP | carried (doc 02 #6) | Already carried in doc 02 #6; no new figure here |
| 14 | Batched ring reads, USRBIO-style (S3) | `memmap2` plus memcpy; no `io_uring`, `O_DIRECT` or `pread` in `crates/` (verified). PLE: 16 page faults per token (`graph.rs:3517–3526`) | carried (doc 03 Part A; doc 03 L273 parallel gather; doc 05) | A cold PLE row is latency-bound at about 65 µs (doc 03 L237). The batched-read concept is the doc 03 gather option |
| 15 | SSD KV tier, 3FS KVCache use case (S3 L48) | `breccia-cache` is a one-line stub (`crates/breccia-cache/src/lib.rs`). A T3 NVMe KV snapshot skeleton exists on another ref (commit `8566ea5`) | carried (doc 02 item 5, doc 16) | Would contend with the NVMe-bound 32K prefill (BRIEF item 4b) |
| 16 | CRAQ chain replication, FoundationDB metadata, multi-node (S3) | None | does not apply | Single host |
| 17 | EPLB greedy packing of experts (S4 L5–7) | `arena.rs:202–213`: first-fit-decreasing by bytes per block | carried in spirit (doc 02 #2, doc 19 #1). The frequency-per-byte variant is **top 3 #2** | The greedy shape is the transferable part |
| 18 | EPLB min-max load objective (S4) | Per-token latency is a sum over sequential blocks; the layer-split placement rule is in `docs/phase-0/gate-A-report.md:31` | does not apply | Balancing across devices does not shorten a serial sum |
| 19 | EPLB redundant replicas across devices (S4 L37) | One device per block; no per-token cross-device parallel work | does not apply | Replicating an expert across cards does not shorten the serial sum |
| 20 | EPLB group-limited routing and hierarchical policy (S4 L21) | Softmax over all 512 experts, top-10 (`docs/specs/qwen4exp-architecture.md:313–318`); "No expert groups (`n_expert_groups == 1`)" (`:324`); no group key read in `crates/` | does not apply | The model spec is the source. The crates grep is consistent with it |
| 21 | DeepEP dispatch and combine kernels (S5) | None | does not apply | No inter-device all-to-all |
| 22 | Pre-sized staging for dispatch (DeepEP buffers, S5) | `STREAM_SLOTS` ring of 8 slots (`graph.rs:271`; comment at `:2715`); `Mailbox` (`xfer.rs:435–452`, doc 19) | already have | Same concept |
| 23 | Dynamic redundant-expert prefetch over NVLink (`lb_prefetch_weights`, S5 L15) | Doc 02 §9: expert prefetch measured −2.7% (doc 02 L404) | does not apply | No NVLink; measured negative |
| 24 | Auxiliary-loss-free balancing (S7) | Training-time | does not apply | Inference only |
| 25 | MLA latent KV (S6) | Checkpoint architecture change | does not apply | Not a retrofit. The 93.3% is against DeepSeek 67B, not MLA |
| 26 | Lightning indexer and `k` (S8) | QSA with `qsa_idx_*` and `qsa_topk`, a Breccia design | already have on the Breccia side. DeepSeek side UNVERIFIED | S8's abstract does not confirm it |
| 27 | Sort-free top-k selection primitive (FlashInfer, doc 18) | `qsa_topk` k-pass: 512 passes per QSA layer above 2K cells (`qsa_topk.comp:39, 78–100`) | carried (doc 18 R1) | This is the long-context top-k cost |
| 28 | Short-list selection bypass (≤512 blocks) | Absent: `idx_score` and `topk` run unconditionally (`qsa/mod.rs:1062–1063`) | gap, **top 3 #3** | Changes block order, so parity not identity |

## 4. Top 3

### 1. K-side 8-bit KV, completing K+V q8_0 (carried in direction; FlashMLA layout as the format precedent)

- **Moves:** extend the V-side q8_0 path to K. Quantise after K's RoPE, since `rope_k` (`qsa/mod.rs:1054`) runs before `kv_append` (`:1058`). Decide whether the RoPE dims are quantised with the rest, as FlashMLA does (S1 L140–143, all 512 dims with per-32 E8M0 scales). Start from the V-side wiring rather than designing a new format.
- **Files:**
  - `crates/breccia-kernels/build.rs:249–267`: the wired variant on `HEAD`, with `KV_Q8V_SHADERS` at `:267`.
  - K-side reader: `qsa_attn_scores.comp` (binding 1, `KPool`, paged f16 pairs). K-side writer: `qsa_kv_append`. Only V is wired for q8_0; K has no variant today.
  - `crates/breccia-engine/src/placement.rs:306–318`: the per-cell budget (2048 B K/V f16 plus about 128 B pooled indexer key).
- **Value (derived from BRIEF L169):** the V-side change alone saves 168 MiB per agent at 32K (932 to 764 MiB, a measured figure in BRIEF). BRIEF's "6 with K+V" (L76) is a projection, not a measurement. This is a capacity lever. BRIEF L190 gives attention as latency-bound at T=1 (about 13 GB/s), so this lever does not speed up decode. Use it for long-context agents, which BRIEF L284 names as the gate.
- **Measurement plan:**
  1. CPU first. No CPU q8 reference exists on `HEAD`. A K-q8_0 record check exists in the history around commit `0788cd7` (not read in this round); port it. Measure the per-element K error and the q·k score error at head dim 128 on recorded K vectors.
  2. GPU qualification card (not run): greedy 32/32 at 2K and at 32K against f16 K, and conformance against the 1e-5 bar BRIEF used for V (L169).
- **Proposed gate (ours):** greedy 32/32 at both lengths, conformance below 1e-5, and at least 6 agents at 32K (the BRIEF L76 projection). Kill if the top-1 token changes at either length.

### 2. Frequency-per-byte greedy packing for expert residency (EPLB-style; carried in spirit)

- **Moves:** replace the first-fit-decreasing-by-bytes choice (`arena.rs:202–213`) with a greedy that ranks (block, expert) pairs by routing frequency per byte and fills the VRAM budget in that order. This is a change to placement granularity (residency inside a block), not a re-sort of the per-block list. The transferable part of EPLB is the greedy placement shape (S4 L5–7). Its min-max objective does not apply (row 18).
- **Files:** `crates/breccia-engine/src/arena.rs:91, 202–213`. Residency is decided per block: a block either has an arena (`graph.rs:2588`, `let streamed = pe.is_none();`) or streams all of its experts. A per-(block, expert) split needs a mixed-block path in the expert loop (`graph.rs:2723–2746`), not found in the read this round. The router ids are allocated at `graph.rs:722` and written by the router at `graph.rs:2035`. The replay harness does not exist, so it is proposed.
- **Value:** BRIEF L159 gives single-stream 11.82 to 16.94 tok/s when coverage goes from 9 to 30 blocks, and expert bytes from 799 to 301 MiB per token. Residency sets decode speed. The gain of frequency-per-byte packing over first-fit-by-bytes is UNVERIFIED.
- **Overlap:** doc 19 top-3 #1 (replay arm (i), static frequency, which is doc 02 §3) and doc 02 item 2. The new part here is the per-(block, expert) greedy as the arm to test against first-fit. A per-expert arm also needs the mixed-block dispatch, so its build cost overlaps doc 19 #1 as well.
- **Measurement plan:**
  1. One GPU run on the qualification card to collect a routing trace (the only GPU step).
  2. CPU replay of first-fit-by-bytes against frequency-per-byte greedy at equal bytes, reporting host-lane expert computations per token. Baseline: 18 streamed blocks (48 − 30 resident, derived) × 10 experts = 180 per token (derived).
- **Proposed gate (ours; the same threshold as doc 19 #1, so the two rounds agree):** the greedy must beat first-fit by at least 5 hit-rate points at equal bytes. Fix the threshold before the run. Drop the policy otherwise.

### 3. Short-list bypass: skip indexer scoring and top-k when the whole cache is selected (gap; new)

- **Moves:** when `n_blocks ≤ 512` (n_kv ≤ 2051), write the causal block list and the count, and skip `idx_score` and `topk`. `attend` already walks `CNT[t]`, so no padded slot is walked. That is Breccia's existing equivalent of FlashMLA's `topk_length` (row 2). Block order changes from score order to causal order. That changes the softmax summation order, so the result will not be bit-identical. The gate is parity. `idx_pool` keeps running (it sits before the `QSA_AUX` probe, `qsa/mod.rs:1059`), and so does `idx_tail` (`:1061`): it maintains the partial-block indexer state. Its consumers were not traced this round, so it stays on the bypass path.
- **Files:**
  - `crates/breccia-kernels/src/qsa/mod.rs:1062–1063`: the unconditional `idx_score` and `topk` calls. The `QSA_AUX` probe opens at `:1060` and also covers `idx_tail` (`:1061`), so it overstates the saving.
  - `crates/breccia-kernels/src/qsa/mod.rs:83–95`: `IDX_TOP_K` 2048, `IDX_TOPK_BLOCKS` 512, `SELECT_WIDTH` 2051 (asserted at `:95`).
  - `crates/breccia-kernels/src/qsa/mod.rs:665`: `max_ids`.
  - `crates/breccia-engine/src/graph.rs:3421–3450`: `run_qsa`.
  - `crates/breccia-kernels/shaders/qsa_topk.comp:1–25`: the 2051-cell rule in the header.
  - `crates/breccia-kernels/src/qsa/cpu.rs:464`: the `topk_blocks` CPU reference.
- **Scope limit:** the bypass applies only at 2051 cells or fewer. Above that, the k-pass runs `min(n_blocks, 512)` passes per layer (`qsa_topk.comp:39, 78–100`), and the bypass does not touch it. That cost is carried in doc 18 R1. If R1 lands, the two levers overlap at 2051 cells or fewer, so the gain must be re-measured after R1.
- **Measurement plan:**
  1. CPU first: on recorded Q and K at n_kv = 512, 1024 and 2048, compare the attend output for score order against causal order (maximum absolute difference).
  2. GPU qualification card (not run): the existing `BRECCIA_GPU_TIME` `QSA_AUX` probe at 1K and 2K gives an upper bound only. The gate needs a sub-probe that times `idx_score` plus `topk` alone (proposed, does not exist; split `QSA_AUX` at `qsa/mod.rs:1062`).
- **Proposed gate (ours):** the removed time (`idx_score` plus `topk`, from the proposed sub-probe, not the full `QSA_AUX`) must be at least 1% of the single-stream token at 2K or less, and greedy must stay 32/32. Kill otherwise. BRIEF does not state the context of its 53.78 ms figure, so the gate may not be reached on the benchmark as run.

### Settled rows (not in the top 3)

- **Indexer low-precision scoring (row 6).** The pooled indexer key is about 128 B per cell (`placement.rs:306–318` comment). At 32K that is 4 MiB per QSA layer (32,768 cells × 128 B), or 48 MiB per token over 12 layers (derived). At the 585 GB/s VRAM read rate in doc 09, that is about 0.086 ms per token (derived): 0.16% of 53.78 ms, or about 0.10% of the 84.2 ms per-stream token at 32K (BRIEF L75; both derived, and the 53.78 ms context is UNVERIFIED). The latency limit that BRIEF L190 gives for KV quant applies here too. Kill.
- **Fused norm-RoPE prologue (row 4).** Gap, low value. The 10–15% figure is for CUDA decode. On Breccia, BRIEF L177 prices a removed dispatch at about 0.31 µs (derived), so an assumed four removed dispatches per QSA layer (48 over 12 layers) would be about 15 µs of 53,780 µs (0.03%, derived).

## 5. Corrections and overlaps

- **Top-k threshold.** An early note put the top-k at 2048 blocks. It is 2048 cells, which is 512 blocks of 4 cells (`qsa/mod.rs:83–87`; `qsa_topk.comp` header). The all-selected threshold is therefore 2051 cells, not 8192.
- **Stale 800 MiB comment.** The comment at `graph.rs:3525` ("~800 MiB/token") is the no-arena figure (`model.rs:13`, 0.84 GB). With 30 resident blocks it is 301 MiB (BRIEF L159). Both figures are consistent with their configurations, but the comment describes the old lane.
- **93.3% is not MLA's figure.** The S6 abstract compares against DeepSeek 67B. Do not attribute it to MLA.
- **Fused-kernel figure.** The 10–15% (S1 L17) is a CUDA decode figure and is not transferable to Breccia's prologue (row 4).
- **DSA on the DeepSeek side.** The S8 abstract does not confirm the lightning indexer or `k`. The `qsa_idx_*` names come from Breccia's shaders and spec, not from S8.
- **KV-quant rejection reason.** BRIEF L190 rejects the KV-quant speedup because attention is latency-bound at about 13 GB/s at T=1 (32.5 to 32.3 ms, −1.0%). The same latency reason caps the indexer lever.
- **Layer-split rule.** The rule that a layer-pipeline split is allowed only after its handoff overhead is proven is stated at `docs/phase-0/gate-A-report.md:31`; an earlier attribution to BRIEF §7 was not confirmed.
- **Unstated hardware.** FlashMLA's 640/410 TFLOPS (S1 L26) do not name hardware. The 660 TFLOPS on H800 (S1 L29) does not name the kernel. DeepGEMM's 1550 TFLOPS (S2 L25) has no shape or precision. Use none of them as a Breccia figure.
- **Byte ratio.** Breccia's QSA KV is 2048 B f16 K/V plus about 128 B pooled indexer key per cell (`placement.rs:306–318`). FlashMLA's 528 B/token belongs to a different model, so the ratio is not like-for-like.
- **Overlaps.** Top 3 #1 overlaps doc 02 item 12 and doc 09 A4 (carried). Top 3 #2 overlaps doc 19 top-3 #1 and doc 02 item 2. Top 3 #3 overlaps doc 18 R1 at 2K and below.

## 6. Sourced and unsourced claims

**Sourced (URL plus raw line, or doc and BRIEF line):** the S1–S5 READMEs (raw lines as in §2); the S6 abstract; the Breccia file and line anchors in §3 and §4; BRIEF L76 (a projection, not a measurement), L159, L164, L169 (measured 168 MiB V-side saving per agent), L177, L180, L188, L190, L284; doc 02 L404 (−2.7%); doc 03 L237 (cold PLE, 65 µs) and L273; doc 05 L41; doc 09 (585 GB/s VRAM).

**Derived (arithmetic on the sourced figures):**
- 0.31 µs per removed dispatch (BRIEF L177: 30 µs ÷ 97).
- 4 MiB per QSA layer at 32K (128 B × 32,768 cells); 48 MiB per token over 12 layers; about 0.086 ms per token at 585 GB/s.
- 2051 cells as the all-selected threshold (`SELECT_WIDTH`, `qsa/mod.rs:87`; the `qsa_topk.comp` header says below 2051 cells every block is selected).
- The `hc-read` share, about 9.9% of 53.78 ms (5,311 / 53,780).
- 180 streamed expert computations per token (18 blocks × 10).

**UNVERIFIED (requires the GPU qualification card, or a source not read):**
- The indexer's share of a decode token; the time cost of `topk` and of the k-pass.
- The gain of frequency-per-byte packing over first-fit-by-bytes.
- K-side 8-bit accuracy, and whether the RoPE dims can be quantised.
- What DeepGEMM's Mega mHC fuses.
- The context length behind BRIEF's 53.78 ms single-stream figure.
- The DeepSeek side of DSA (S8 not confirmed; HTML and PDF not read).
- The S7 claims beyond the abstract.

## 7. Skipped

- **Not read:** the DeepGEMM Mega mHC source (README title only); the FlashMLA CUDA kernels; the DeepEP kernels; EPLB's `eplb.py`; 3FS's USRBIO API document (cited by path only); the S8 HTML and PDF; the S7 full text.
- **Not run:** any GPU work, any build, any benchmark, and any CPU replay. The CPU reference `ref/qsa.py` does not exist. The CPU reference in the tree is `crates/breccia-kernels/src/qsa/cpu.rs`.
- **Not measured:** the all-blocks bypass (top 3 #3), the K-side quantisation (top 3 #1), and the frequency-per-byte packing (top 3 #2).
- **Not quoted:** the launch-surface test in `breccia-flags`, which has machine-specific paths. Nothing from the scratch directory is cited.
- **Out of scope:** multi-node 3FS, DeepEP over NVLink or RDMA, and EPLB's cross-device replication.
