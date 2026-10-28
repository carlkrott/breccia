# 34 — Long-context KV retention and eviction on QSA blocks (100K–256K)

Round 34. Docs only: no code changed, no build, no GPU or model process run. Base commit 9720cc9. Every gain below is UNVERIFIED until the GPU qualification card runs. Arithmetic is labelled as estimate where it is one.

Citations are repo-relative: crate paths under `crates/`, docs under `docs/`. Docs 25 and 33 are cited for the lines named, not re-audited. The llama.cpp reference is commit 76f46ad29. Papers are cited by title and arXiv id.

## 0. Verdict

1. **Memory.** The full QSA cache is 20,736 B per cell at the q8-V layout: 1.931 GiB at 100K, 5.063 GiB at 256K. On one card a full 100K cache needs k = 2 displaced blocks (RX) or k = 3 (MI50) at q8-V. At 256K it needs k = 6 on both (§3). The per-card budget comes from split placement, so applying it to the one-model-per-card path is UNVERIFIED.
2. **Eviction is the only lever here that changes bytes by about an order of magnitude** (8.9× as built at 100K, 11.3× at 256K; 24× at 100K remapped). Sink 16 + window 4,096 = 4,112 cells. With the pooled-key buffer and page table remapped to retained cells it is 85.3 MB per stream, saving 1.852 GiB (95.9%) at 100K and 4.983 GiB (98.4%) at 256K. As built, both are sized by logical length (§0.4 (c)): 232.6 MB at 100K (saves 1.715 GiB, 88.8%) and 481.6 MB at 256K (saves 4.614 GiB, 91.1%). It drops every fact outside sink and window from the QSA cache. Its accuracy effect is UNVERIFIED and gates every card.
3. **Eviction alone does not extend length.** Sink + window (C1) keeps absolute positions, so the RoPE and position tables still need `max_pos` ≥ total length (mod.rs L619–623, L286, L326). Only context shift (C3) keeps positions bounded, so only C3 can run past `max_pos`. This trade-off is the reason for the card ordering in §5.
4. **Must-touch facts for any eviction.** (a) `kv_append` checks the write position against `capacity_cells` (mod.rs L813), and `run_qsa` checks `n_kv_after = pos + 1` against the same capacity (graph.rs L3423, L3428–3430). Both must test free physical pages, not positions. (b) The indexer addresses blocks by logical position: `n_blocks = n_kv_after / 4` (graph.rs L3442; mod.rs L664), the own block is `(pos0 + t) / 4` (qsa_topk.comp L115), the dead-block mask is `b * idx_block > pos` (qsa_idx_score.comp L58–60), and the pool tail test is `cell - pos0` (qsa_idx_pool.comp L65). Attention is already consistent with logical positions: it maps `cell` through the page table (qsa_attn_scores.comp L90–91, L96). The indexer is not: it scans every logical block up to `n_blocks`, evicted ones included, so it needs evicted blocks masked dead or a retained-block list. (c) The page table and the pooled-key buffer are sized from `capacity_cells` (pages at mod.rs L255; `pooled_blocks(cells)` at L257; allocations at L280–285) and indexed by logical page or block (`qsa_attn_scores.comp` L96; `qsa_idx_pool.comp` L65; `qsa_idx_score.comp` L29). A card whose capacity is below total length must size both by logical length (`max_pos`; pooled keys are 1,536 B per position over 12 layers) or add a logical-to-physical table for each. (d) The RoPE and position tables are sized to `max_pos` (mod.rs L619–623, L286, L326; reset at L411). C3 re-rotation must also cover the pooled keys, which are rotated at block start (`qsa_idx_pool.comp` L5; `rope_idx` at mod.rs L620).
5. **Placement ignores the V layout.** `state_bytes` charges QSA at 2,180 B per cell-layer whatever `BRECCIA_KV_Q8V` is (placement.rs L312–314). q8-V needs 1,728 B, so the charge is 26.2% high at q8-V (§1.9).
6. **Counted eviction work is open or closed-unmerged.** vLLM #43374 and #53871 (open), SGLang #33856 (open), llama.cpp #27583 (closed, not merged). No counted eviction PR is merged. llama.cpp's context-shift code is in-tree and is cited by file and line (§2.2).
7. **Five flag-OFF cards, ranked in §5.** None has a measured gain.

## 1. QSA as built

### 1.1 Layer split and state

- QSA (full attention with the lightning indexer) sits where `(i + 1) % qsa_interval == 0`, and the comment at config.rs L300 states `i % 4 == 3` (config.rs L300–303). That is 12 of 48 blocks. The other 36 are Gated DeltaNet (GDN), with fixed recurrent state of 3,268,608 B per layer, 112.2 MiB per stream, independent of context (doc 25 L104).

### 1.2 Constants (`crates/breccia-kernels/src/qsa/mod.rs`)

- `MODEL_N_HEAD` 24 (L69), `MODEL_N_KV_HEAD` 2 (L71), `MODEL_HEAD_DIM` 256 (L73), `MODEL_N_ROT` 64 (L75).
- `IDX_N_HEAD` 4 (L77), `IDX_DIM` 128 (L79), `IDX_BLOCK` 4 (L81), `IDX_TOP_K` 2048 (L83), `IDX_TOPK_BLOCKS` 512 (L85), `SELECT_WIDTH` 2051 (L87, asserted at L95), `PAGE_CELLS` 16 (L89).

### 1.3 Selection and attention

- Indexer score: `(1/sqrt(128)) × Σ_h relu(q_h · pooled_b)` (`qsa_idx_score.comp` L5). A block starting after the query position is dead at −1e30 (L18–20). The CPU reference uses −inf (`crates/breccia-kernels/src/qsa/cpu.rs` L93–95).
- One block list per token is shared by all 24 query heads. It is the top 512 blocks by score plus the incomplete tail block (mod.rs L12).
- Top-k is k argmax passes over the block row, O(k × n_blocks) (`qsa_topk.comp` L39–42). The shader's own note says radix-select is needed at 65,536 blocks, the count for a full 262,144-cell cache.
- A query attends to at most 2,051 cells whatever the cache length (`qsa_attn_scores.comp` L10–15). No dense 262,144-cell path was built (L15).
- Output gate: `A = A × sigmoid(gate)` per head, before `attn_output` (mod.rs L14).
- The query's own block is appended only when absent (cpu.rs L127–129; `qsa_topk.comp` L9–11).

### 1.4 Storage per cell

Per QSA cell-layer:

- K: f16, 2 KV heads × 256 × 2 B = 1,024 B.
- V: f16 1,024 B, or q8_0 576 B. The q8_0 format is 32 × i8 plus one f32 scale per 32-element block (mod.rs L113–115).
- Pooled indexer key: 128 B (`IDX_DIM` f32 per 4-cell block = 512 B per block).
- Phase-3 doc L146–147 gives 2,048 B per cell-layer for f16 K+V (16 KiB K + 16 KiB V per 16-cell page).

| Layout | Per cell-layer | Per cell (12 QSA layers) | Status |
|---|---|---|---|
| q8-V | 1,728 B | 20,736 B | built; selected by env (§1.5) |
| f16-V | 2,176 B | 26,112 B | built; the default |
| q8 K+V | 1,280 B | 15,360 B | doc 25 card A; not built, arithmetic only |

### 1.5 Layout selection

- `VLayout::from_env` (mod.rs L119–124): `BRECCIA_KV_Q8V` set to `1`, `true`, `on` or `yes` selects q8_0. Unset or any other value selects f16.
- Whether the deployed launch sets it is UNVERIFIED in this round. This doc says "q8-V layout" and "f16-V layout". Doc 33 L17 calls 20,736 B "shipped"; this doc does not.

### 1.6 Pool, pages and position table

- `QsaState::new` (mod.rs L236–240) takes `capacity_cells` and `max_pos`. Capacity rounds up to whole pages (`pages = capacity_cells.div_ceil(PAGE_CELLS)`, L255).
- The page table is the identity (L325). The position table is `max_pos × 4` bytes (L286) holding 0..max_pos (L326). `reset` restores the page table (L407–410) and the positions (L411).
- Rope tables are sized to `max_pos`: `Rope::new(…, max_pos)` (L619–620) and `RopeTable::model(max_pos)` (L623).
- The page table maps logical page = position / 16 to a physical page (attention reads it at `qsa_attn_scores.comp` L96). The pooled-key buffer is one flat `[blocks, idx_dim]` array indexed by logical block (`qsa_idx_score.comp` L29), with no page table. Both are sized from `capacity_cells` (mod.rs L255, L257, L280–285), so a card whose capacity is below total length must resize them (§0.4 (c)). The append shader indexes the same table by logical cell (`qsa_kv_append.comp` L100–101, L150–151), so the write path already uses the logical-to-physical lookup; its header (L10–11) calls the table identity today.

### 1.7 Writes and RoPE

- K is rotated at write time. `rope_k` (mod.rs L785–795) takes `pos0`. Retained keys keep the rotation they were written with. A shift would change that (C3, §5).
- `kv_append` (mod.rs L812–822). The guard at L813 is `if pos0 + ws.tokens > state.capacity_cells`, which uses a position as a cell index. The page-table comment (`qsa_kv_append.comp` L10–11) reads "identity today, a real table as soon as M5 evicts".
- RoPE rotates `MODEL_N_ROT` = 64 of the 256 head dims, in NEOX pairs (`ops_rope.comp` L5–8; `qsa_idx_pool.comp` L46).

### 1.8 Workspace

- `workspace` (mod.rs L660–666): `n_blocks = n_kv_after / 4` (L664), `max_ids = min(n_blocks, 512) + 1` (L665), `slots = 4 × max_ids` (L666). The selection list is capped at 2,052 slots whatever the length (comment at L658–659).
- The per-block count grows with `n_blocks`: 4 B × n_blocks per row is 97.7 KiB at 100K (25,000 blocks) and 256 KiB at 256K (65,536 blocks). The 4-byte entry is an assumption and the buffer was not traced (UNVERIFIED).

### 1.9 Placement charge and layout independence

- The QSA arm of `state_bytes` charges `capacity_cells × (2048 + 132)` (placement.rs L312–314), which is 2,180 B per cell per QSA block. The comment at L313 reads "f16 K/V per cell (2048 B) + pooled indexer key (~128 B/cell)".
- **Env-independence finding.** The QSA arm does not reference `BRECCIA_KV_Q8V` or `VLayout`. Placement charges f16-V bytes whatever the layout.
- Against q8-V (1,728 B per cell-layer) the charge is 2,180 / 1,728 − 1 = **26.2% high**.
- Doc 25 L221 records 2,180 against 2,176 (a 4 B, 0.18% gap) and calls the f16 assumption "conservative for V q8". Doc 33 L205 repeats the 2,180 vs 2,176 gap. The 4 B gap is minor. The layout mismatch is the one that matters for a q8-V ceiling. Neither doc states the 26.2% figure.

### 1.10 Multi-stream

- `multi.rs` L139–146 bails when a stream has `devices.len() != 1` ("a stream must own exactly one").
- `main.rs` L1686 assigns `devs[s % devs.len()]` to stream s, and L1690 gives it `devices: vec![dev.clone()]`. KV is not pooled across cards.

### 1.11 Timing references and scan size

- `multi.rs` L5–9 records a GPU busy time of 43 ms per token and a host-lane time of 38 ms per token (§8n of that source). The context length at which these were taken is not stated, so this doc makes no millisecond comparison.
- The indexer reads one pooled key per block in `[0, n_blocks)`, with `n_blocks = (pos + 1) / 4` (graph.rs L3442). That is 128 B per position per QSA layer, 1,536 B per position over 12 layers: 153.6 MB per token at 100K and 402.7 MB at 256K. Milliseconds are for the qualification card.
- The attended read is fixed at 2,051 cells × 12 layers × 1,600 B (K 1,024 + q8 V 576) = 39.4 MB per token, whatever the length, at q8-V.

## 2. Techniques

Each entry gives mechanism, bytes at 100K and 256K (formulas in §9), accuracy risk, and the published result as the abstract's own claim. Papers are cited by title and arXiv id. Upstream state is in §2.8.

### 2.1 StreamingLLM

- "Efficient Streaming Language Models with Attention Sinks", arXiv:2309.17453.
- Mechanism: keep the first few sink positions and a recent window; evict the middle.
- Abstract: stable on up to 4M tokens; up to 22.2× speedup over sliding-window recomputation.
- QSA mapping: sink = 1 page = 16 cells; window = 4,096 cells = 256 pages; retained = 4,112 cells.
- Bytes: 4,112 × 20,736 = 85.3 MB per stream with the pooled keys remapped to retained cells (§5.0), saving 1.852 GiB at 100K (95.9%) and 4.983 GiB at 256K (98.4%). As built (pooled keys sized by length): 232.6 MB and 481.6 MB, saving 1.715 GiB (88.8%) and 4.614 GiB (91.1%) (§0.2).
- Accuracy risk: facts outside sink and window are absent from the QSA cache. GDN keeps a compressed history; the effect is UNVERIFIED.
- Upstream: see §2.8 (vLLM #43374, SGLang #33856, llama.cpp #27583).

### 2.2 llama.cpp context shift

- No paper. Mechanism: when the window is full, drop `n_discard` tokens after `n_keep`, then move the remaining positions down by `n_discard` (`seq_rm` then `seq_add`). Whether the cached keys are re-rotated was not read (UNVERIFIED).
- Code at commit 76f46ad29:
  - `examples/passkey/passkey.cpp` L169–175 and L196–204: a `seq_rm` over the discarded range, then a `seq_add` with delta −`n_discard` (argument lists not quoted here).
  - `tools/completion/completion.cpp` L603–607: the context-shift call site (behaviour not re-read).
  - `common/arg.cpp` L1484–1489 (`--keep`) and L1541–1545 (`--context-shift`); `common/common.h` L455 (`n_keep` field) and L570 (`ctx_shift` field).
  - `include/llama.h` L725 (`seq_rm`) and L749 (`seq_add`).
- Bytes: holds the cache at window size. With sink 16 and window 4,096 the K/V footprint equals C1's 79.0 MB (4,112 cells; 85.3 MB with remapped pooled keys, §5.0). Positions stay below `max_pos` for any total length, so the pooled term is bounded too.
- Accuracy risk: discarded tokens are gone. Re-rotation exactness for QSA K is UNVERIFIED.
- Upstream: the shift code is in llama.cpp at the pinned commit. llama.cpp #14120 (`--keep` for llama-server, MERGED, title only) is the context-shift family and is not counted as eviction.

### 2.3 H2O

- "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", arXiv:2306.14048.
- Mechanism: keep the heavy hitters (tokens with the largest accumulated attention) plus a recent window; evict the rest.
- Abstract: 20% heavy hitters; up to 29× throughput; 3× over FlexGen; latency up to 1.9×.
- Bytes at a 20% budget, remapped basis (§5.0): 20,000 cells at 100K (saves 1.545 GiB, 80%); 52,416 cells at 256K (saves 4.050 GiB, 80%). As built the pooled term follows length, not retained cells: 153.6 MB at 100K and 402.7 MB at 256K, so the savings are 1.431 GiB and 3.750 GiB. Accumulator 48 B per cell, assumed one f32 per QSA layer: 4.80 MB at 100K, 12.58 MB at 256K.
- Accuracy risk: accumulated attention needs a score for every cell at every step. A QSA query scores at most 2,051 cells, so the accumulator is defined only on selected cells. How to define it for the rest is UNVERIFIED.
- Upstream: title-only PR-search negative (§2.8).

### 2.4 SnapKV

- "SnapKV: LLM Knows What You Are Looking for Before Generation", arXiv:2404.14469.
- Mechanism: prompt-only selection. An observation window at the end of the prompt picks prefix cells per head; decode then uses that fixed set.
- Abstract: 3.6× generation speed and 8.2× memory efficiency at 16K inputs; up to 380K tokens on one A100-80GB; negligible drop on needle-in-a-haystack.
- Illustrative budget of 2,080 cells (composition assumed: 16 sink + 2,048 selected + 16 observation): 43.1 MB per stream on the remapped basis (§5.0). Saves 1.891 GiB at 100K and 5.022 GiB at 256K on that basis. Indexer scan 3.2 MB per token.
- Accuracy risk: the set is fixed after prefill, so a decode-time needle outside it is lost. The observation window has to contain the question.
- Upstream: SGLang KVPress-related PRs #11829, #44130, #12933 (CLOSED, title). vLLM #47967 (KVCrush, OPEN, title) is a title hit only. Neither is counted.

### 2.5 PyramidKV

- "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling", arXiv:2406.02069.
- Mechanism: per-layer budgets that decline with depth, with selection by attention concentration.
- Abstract: matches full-cache accuracy at 12% of the KV cache.
- Bytes, uniform 12% reference only, remapped basis (§5.0): 12,000 cells at 100K (saves 1.699 GiB); 31,456 at 256K (saves 4.455 GiB). A per-layer schedule over the 12 QSA layers is not computed here. The paper's ratio does not transfer directly (UNVERIFIED).
- Accuracy risk: needs attention-mass selection like H2O (§2.3). The paper's layer count and model differ from QSA's.
- Upstream: vLLM #47967 is the only title hit (§2.8). Not counted.

### 2.6 Quest

- "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", arXiv:2406.10774.
- Mechanism: per-page min and max bounds on the pooled indexer key (`IDX_DIM` f32 per 4-cell block, the tensor the indexer scores, not attention K), and query-dependent page ranking. This adapts Quest's bound to the indexer; the paper bounds attention K. Per head, the bound on q_h · pooled is the sum over dimensions of the larger of q_hd × min_d and q_hd × max_d. ReLU and the sum over 4 heads are monotone, so the page bound stays an upper bound on the indexer score. No page is removed, so reads shrink and capacity does not.
- Abstract: 2.23× self-attention speedup and 7.03× latency reduction.
- Page size: Quest's page size is not in the abstract. This doc uses QSA's 16-cell page.
- Bound format (assumed): one min and one max per dimension per page per layer, over the 4 pooled block keys of the page: 2 × 128 × 4 B = 1 KiB per page per layer, 64 B per cell per layer, 768 B per cell over 12 layers. One pair serves all 4 indexer heads, because the pooled key is shared.
- Bytes: bound storage 73.2 MiB at 100K and 192 MiB at 256K. Per token: bound pass 76.8 MB at 100K (201.3 MB at 256K). Selected pass 128 pages × 16 cells × 1,536 B = 3.1 MB. Total 79.9 MB at 100K against 153.6 MB for the exact scan, and 204.5 MB at 256K against 402.7 MB.
- Accuracy risk: bounds are upper bounds, so the 128 pages chosen (512 blocks, the top-k count) can omit blocks that the exact top-512 would pick. Parity holds only while every block is selected, which is up to 2,048 cells (`qsa_topk.comp` L17–18).
- Upstream: SGLang #28882 (Quest in the FlashAttention decode path, CLOSED, title) is a title hit only, and it is not eviction. No llama.cpp or vLLM PR matched by title (title-only negative).

### 2.7 DuoAttention

- "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads", arXiv:2410.10819.
- Mechanism: a per-head split. Retrieval heads keep every cell. Streaming heads keep sink and recent window only.
- Abstract: memory reduction up to 2.55× (MHA) and 1.67× (GQA).
- Bytes: one streaming KV head is K 512 + V 288 = 800 B per cell per layer at q8-V. Assumed: one of the two KV heads streams in each QSA layer. The learned split for this model is UNVERIFIED. Saving = (N − 4,112) × 800 × 12: 0.857 GiB (44.4%) at 100K and 2.307 GiB (45.6%) at 256K.
- Accuracy risk: a head mislabelled as streaming drops its long-range facts. The head split needs an optimisation pass that this round did not run.
- Upstream: title-only PR-search negative (§2.8).

### 2.8 Upstream status (read or title, 2026-10-28)

| Item | State | Read or title | Counted as eviction | Note |
|---|---|---|---|---|
| vLLM #43374 | OPEN | read | yes | experimental session KV eviction with attention sinks; `AsyncLLM.evict_session_token_range`; env-gated |
| vLLM #53871 | OPEN | read | yes, primitive only | block-range eviction primitive; no policy |
| vLLM #59878 | OPEN | read | no | FlashInfer attention sinks on SM8x; not eviction |
| vLLM #43372 | OPEN | title | no | `AsyncLLM.truncate_session` |
| vLLM #47967 | OPEN | title | no | KVCrush |
| vLLM #5752 | CLOSED | title | no | sparse KV framework |
| SGLang #33856 | OPEN | read | yes | unified post-hoc sparsity framework with StreamingLLM visibility; first implementation for design PR #32657; freed-vs-masked storage UNVERIFIED |
| SGLang #28882 | CLOSED | title | no | Quest in the FlashAttention decode path |
| SGLang #11829, #44130, #12933 | CLOSED | title | no | KVPress-related |
| llama.cpp #27583 | CLOSED, not merged | read | yes | `--kv-evict-sink`, `--kv-evict-window`; physical cache capped at sink + recent + n_ubatch cells |
| llama.cpp #14120 | MERGED | title | no | `--keep` for llama-server; context-shift family |
| H2O, DuoAttention, Quest (llama.cpp, vLLM) | none matched | title-only PR-search negative | no | no matching PR by title |

Counted as eviction work: vLLM #43374, vLLM #53871, SGLang #33856, llama.cpp #27583. Nothing else in this table is counted.

## 3. Ceiling per card

**Model.** usable(card, k) = headroom + k × 930 MiB − 112.2 MiB (GDN). Cells = floor(usable ÷ bytes per cell), rounded down to a whole 16-cell page. Sources: doc 25 L121 (headroom about 304 MiB on RX, about 8 MiB on MI50), doc 25 L125 (930 MiB per displaced block; real blocks are 637–1,138 MiB by type, so ±20%), doc 25 L142 (N_max formula), and doc 33 L90 (budget = headroom + k × 930 MiB). Doc 25 L112 is the explicit `--split-blocks` path.

Limits, stated before the numbers:

- **The 930 MiB and the headroom come from split placement** (doc 25 L112–125). Applying them to the one-model-per-card path is UNVERIFIED.
- **The two cards do not pool.** A stream owns one device (multi.rs L139–146; main.rs L1686, L1690), so each cell count is bounded by one card's budget. The table is the single-card case: the card holds the stream's full KV (12 QSA layers) and full GDN state (112.2 MiB). That is the conservative case for bytes per cell. Split placement changes per-card bytes and is not tabulated here.
- **One stream** (S = 1). Doc 25 L142 and doc 33 L92–94 use S = 2 and S = 4 with their own per-card byte counts. These figures are not comparable with theirs.

Usable budget per card, and cells per stream (page-rounded):

| Card | k | Usable MiB | q8-V cells | f16-V cells | q8 K+V cells (not built) |
|---|---|---|---|---|---|
| RX | 0 | 191.8 | 9,696 | 7,696 | 13,088 |
| RX | 1 | 1,121.8 | 56,720 | 45,040 | 76,576 |
| RX | 2 | 2,051.8 | 103,744 | 82,384 | 140,064 |
| RX | 3 | 2,981.8 | 150,768 | 119,728 | 203,552 |
| MI50 | 0 | −104.2 (does not fit) | 0 | 0 | 0 |
| MI50 | 1 | 825.8 | 41,744 | 33,152 | 56,368 |
| MI50 | 2 | 1,755.8 | 88,784 | 70,496 | 119,856 |
| MI50 | 3 | 2,685.8 | 135,808 | 107,840 | 183,344 |

Minimum displaced k to hold a full cache of N cells on one stream (RX / MI50):

| N | q8-V | f16-V | q8 K+V |
|---|---|---|---|
| 32,768 | 1 / 1 | 1 / 1 | 1 / 1 |
| 100,000 | 2 / 3 | 3 / 3 | 2 / 2 |
| 262,144 | 6 / 6 | 7 / 8 | 4 / 5 |

Full-cache KV at q8-V: 0.633 GiB at 32K, 1.931 GiB at 100K, 5.063 GiB at 256K.

Largest full cache on one card at k = 3 (one stream): RX 150,768 cells at q8-V and 119,728 at f16-V; MI50 135,808 at q8-V and 107,840 at f16-V. With C1 or C3 the K/V cache holds 4,112 cells, so K/V bytes stop limiting length. For C1 the pooled term still grows with length as built (§0.4 (c)); C3 bounds it at `max_pos`. The position bound (`max_pos`) takes over (§5).

**Host RAM (labelled estimate, not a design).** No host KV tier exists: pools are device-allocated (mod.rs L244–264). If 30 GiB (32,212,254,720 B) of host RAM held QSA cells alone: 1,553,445 cells at q8-V, 1,233,618 at f16-V and 2,097,152 at q8 K+V. GDN and the arena are not subtracted.

## 4. Hybrid model and two streams

- GDN integrates the whole history into its recurrent state. Eviction in the QSA cache does not remove information that GDN carries. The effect on recall is UNVERIFIED.
- Two streams run on separate cards (§1.10). KV is not pooled, so eviction on one stream frees only that card's budget. Gates below use one stream unless they say otherwise.

## 5. Flag-OFF cards

Ranked: **C1, C3, C5, C2, C4.** C1 gives the largest saving with the simplest mechanism. C3 is the only card that bounds positions, at the cost of a new re-rotation path. C5 drops no information and saves reads, not capacity. C2 and C4 need an attention-mass definition that top-k does not provide.

### 5.0 Gate rules (apply to every card)

- **OFF runs.** Each OFF run is named by context, displaced blocks k, streams, card and layout:
  - O32: ctx 32,768; k = 1; 1 stream; RX or MI50; q8-V; full cache.
  - O100-RX: ctx 100,000; k = 2; 1 stream; RX; q8-V; full cache (fits: 103,744 cells, §3).
  - O100-MI50: ctx 100,000; k = 3; 1 stream; MI50; q8-V; full cache (fits: 135,808 cells).
- **Policy budget = eviction trigger.** An ON run evicts when positions exceed its policy budget, set in config, independent of cell capacity. A 100K ON run can therefore evict at the same k as O100 (§0.4).
- **Greedy parity (exact, not a threshold).** ON output is token-identical to OFF on every prompt where prompt + generated tokens ≤ policy budget. Generated length for every greedy comparison: PROPOSED 256 tokens. Each card's gate states the largest prompt that satisfies this.
- **Recall.** Passkey or needle at 32K and at 100K. In-window ON recall ≥ OFF recall − 5 points (PROPOSED). Beyond-window recall is reported, not gated.
- **Perplexity.** ON against OFF on one fixed held-out text at 32K: relative increase ≤ 1% (PROPOSED).
- **Bytes basis.** Card bytes use the remapped basis: 20,736 B per retained cell, with pooled keys following the retained cells (needs §0.4 (c)). As built, an ON total is retained cells × 19,200 B + 1,536 B × N, where N is total length: 153.6 MB at 100K and 402.7 MB at 256K. Each card's saving is smaller than its remapped figure by 1,536 B × (N − retained cells).
- **Receipt (each card).** A JSON file from the qualification run, named `34-<card>-receipt.json`, holding: ctx, k, card, layout, streams, policy budget, git commit, the parity result, the recall table and the perplexity pair. Not created in this round.

### C1 — Sink and window eviction, page-granular

- **Mechanism.** Keep the sink (1 page, 16 cells) and the newest 4,096 cells (256 pages): 4,112 cells. Evicted pages are unmapped in the page table (§1.6). Positions keep increasing, so the RoPE and position tables must cover total length: `max_pos` ≥ total length. Three changes come first (§0.4): both capacity guards (mod.rs L813, graph.rs L3428) test free physical pages; the indexer treats evicted blocks as dead or scans a retained-block list; and the page table and pooled-key buffer are sized by logical length or remapped (§0.4 (c)). The RoPE and position tables already cover `max_pos`.
- **Bytes.** Remapped basis (pooled keys follow retained cells): 4,112 × 20,736 = 85,266,432 B (85.3 MB), saving 1.852 GiB (95.9%) at 100K and 4.983 GiB (98.4%) at 256K. As built (pooled keys and page table sized by logical length, §0.4 (c)): 4,112 × 19,200 + 1,536 × N = 232.6 MB at 100K, saving 1.715 GiB (88.8%); 481.6 MB at 256K, saving 4.614 GiB (91.1%). The indexer scan over retained cells is 6.3 MB per token (4,112 × 1,536 B), but only with a retained-block list, which the code does not have. As built, the scan follows logical position: 153.6 MB per token at 100K.
- **Accuracy risk.** Facts outside sink and window are absent from the QSA cache. GDN carries a compressed history, so the net effect is UNVERIFIED.
- **Gates.**
  - Parity: prompt ≤ 3,856 tokens (4,112 − 256 generated), exact, against O32.
  - Recall at 32K: policy budget 4,112, against O32 (in-window and beyond).
  - Recall at 100K: ON at RX k = 2 (or MI50 k = 3), 1 stream, q8-V, policy budget 4,112, against O100 at the same k.
  - Perplexity at 32K (PROPOSED ≤ 1%).
- **Trade-off.** C1 cuts memory at a fixed `max_pos`. It does not allow a total length above `max_pos` (§0.3).
- Receipt: as §5.0.
- Gain: UNVERIFIED - requires the GPU qualification card

### C3 — Context shift (llama.cpp-style), bounded positions

- **Mechanism.** When the window is full, discard a page-aligned block after the sink, move the remaining logical positions down by that block, and re-rotate retained K and the pooled keys by the same delta (pooled keys are rotated at block start, `qsa_idx_pool.comp` L5). Positions stay below `max_pos` for any total length, so `max_pos` can be close to the window size. This is the only card that bounds positions. PROPOSED sizing: window 4,096 and discard block 2,048 cells (half the window), so `max_pos` = 4,096 + 2,048 = 6,144. The discard is block-aligned (4 cells) as well as page-aligned (16 cells); 2,048 cells is 512 blocks, so it qualifies.
- **Cost.** A re-rotation touches `MODEL_N_ROT` = 64 dims per KV head (`ops_rope.comp` L5–8). Per cell across 12 QSA layers that is 12 × 2 × 64 = 1,536 dims. Pooled keys add one 64-dim rotation per block per layer: 16 dims per cell per layer, 192 per cell over 12 layers, so 1,728 dims per cell in total. A shift re-rotates every retained cell, so amortised cost per generated token = retained cells × 1,536 ÷ tokens discarded per shift. Untimed; UNVERIFIED.
- **Bytes.** 4,112 cells of K/V (79.0 MB) plus pooled keys: 6.3 MB remapped to cells (85.3 MB per stream), or 9.4 MB if the pooled buffer is sized by `max_pos` (1,536 blocks × 512 B × 12 layers; 88.4 MB). Both need the cell index decoupled from logical position (the three changes in C1). With identity cell mapping, capacity must cover `max_pos`: 6,144 × 20,736 = 127.4 MB.
- **Accuracy risk.** Discarded tokens are unrecoverable. In `qsa/` (Rust and shaders), RoPE is applied only at write time (`rope_k`, `qsa/mod.rs` L785–795). A grep for re-rotation finds no call, and the pool shader avoids re-rotating earlier rows by design (`qsa_idx_pool.comp` L25). Re-rotation exactness for a moved cache is UNVERIFIED.
- **Gates.**
  - Parity: 2K-token prompts, exact, against O32 (no shift triggers).
  - Recall at 32K: window 4,096 as the policy budget, so shifts occur at 32K; in-window recall against O32.
  - Recall at 100K: ON at RX k = 2 (or MI50 k = 3), 1 stream, q8-V, window 4,096, `max_pos` 6,144 (PROPOSED), against O100.
  - Perplexity at 32K (PROPOSED ≤ 1%).
- Receipt: as §5.0, plus the shift count and the re-rotated dims.
- Gain: UNVERIFIED - requires the GPU qualification card

### C5 — Quest-style page bounds (reads only, no capacity change)

- **Mechanism.** For each 16-cell page and layer, store per-dimension min and max over the pooled indexer key (format in §2.6; the indexer's tensor, not attention K). The query ranks pages by bound and keeps 128 pages (2,048 cells, 512 blocks). The exact indexer then runs on those pages only.
- **Bytes.** No capacity change. Bound storage 768 B per cell: 73.2 MiB at 100K, 192 MiB at 256K. Per-token scan 79.9 MB at 100K (against 153.6 MB) and 204.5 MB at 256K (against 402.7 MB).
- **Accuracy risk.** Bounds are upper bounds, so the page pick can omit blocks the exact top-512 would select. Parity holds only while every block is selected.
- **Gates.**
  - Parity: prompt ≤ 1,792 tokens (1,792 + 256 generated ≤ 2,048), exact, against O32.
  - Recall at 32K and at 100K: bound-pass ON against O32 and O100 (same k, 1 stream, q8-V).
  - Perplexity at 32K (PROPOSED ≤ 1%).
- Gain: UNVERIFIED - requires the GPU qualification card

### C2 — H2O heavy-hitter eviction at 20%

- **Mechanism.** Accumulate attention mass per cell (48 B per cell, assumed one f32 per QSA layer). Keep the top 20% by mass plus the recent window; evict the rest.
- **Bytes.** Remapped basis (§5.0): 20,000 cells at 100K (395.5 MiB; saves 1.545 GiB; as built 1.431 GiB). 52,416 cells at 256K (saves 4.050 GiB; as built 3.750 GiB). Accumulator 4.80 MB at 100K and 12.58 MB at 256K.
- **Accuracy risk.** Accumulated mass exists only on the at most 2,051 cells each query scores, so the 20% policy needs a definition for unscored cells (UNVERIFIED). Heavy-hitter selection favours early and frequent tokens.
- **Gates.**
  - Parity: prompt ≤ 6,288 tokens (policy budget 6,544 = 20% of 32,768 rounded down to a page, less 256 generated), exact, against O32.
  - Recall at 32K: policy budget 6,544 cells, against O32.
  - Recall at 100K: ON at RX k = 2 with policy budget 20,000 cells, against O100.
  - Perplexity at 32K (PROPOSED ≤ 1%).
- Gain: UNVERIFIED - requires the GPU qualification card

### C4 — PyramidKV-style per-layer budgets

- **Mechanism.** A per-layer budget that declines with depth across the 12 QSA layers, with selection by attention mass as in C2.
- **Bytes.** Remapped basis (§5.0). Uniform 12% reference only: 12,000 cells at 100K (saves 1.699 GiB); 31,456 at 256K (saves 4.455 GiB). No per-layer schedule is computed.
- **Accuracy risk.** Same as C2, plus the paper's ratio does not transfer to this layer count (UNVERIFIED).
- **Gates.**
  - Parity: prompt ≤ 3,664 tokens (policy budget 3,920 = 12% of 32,768 rounded down to a page, less 256 generated), exact, against O32.
  - Recall at 32K and at 100K against O32 and O100 (at 100K, RX k = 2, budget 12,000 cells).
  - Perplexity at 32K (PROPOSED ≤ 1%).
- Gain: UNVERIFIED - requires the GPU qualification card

## 6. Unverified

- The single-card ceiling is not the split-placement budget (§3).
- Whether the deployed launch sets `BRECCIA_KV_Q8V` (§1.5).
- Re-rotation of retained K for C3: no call exists in `qsa/` by grep (§1.7, §5); the shift path is proposed, not built.
- Accumulated attention under top-k for C2 and C4 (§2.3, §5).
- The DuoAttention head split for this model (§2.7).
- The C5 bound format (1 KiB per page per layer over the pooled indexer key) is assumed (§2.6); its recall effect is not measured.
- Whether the pooled-key buffer and page table can be remapped to retained cells (§0.4 (c)). As built both are sized from `capacity_cells` and indexed by logical block or page; the remapped C1 figure depends on this.
- Freed-vs-masked storage semantics in SGLang #33856 (§2.8).
- GDN interaction with eviction (§4).
- Accuracy of every card: no run in this round.
- The llama.cpp kv-cache internals were not opened. Only the files cited in §2.2 were read.
- NSA and KIVI were not fetched.
- Paper results are the abstracts' own claims; full papers were not read.
- The 930 MiB block size varies by type (637–1,138 MiB; doc 25 L125).
- The 97.7 KiB and 256 KiB per-row figures assume 4-byte entries; the buffer was not traced (§1.8).
- The context length at which the `multi.rs` L5–9 timings were taken is not stated.
- Millisecond costs of the indexer and of any card are left to the qualification card.

## 7. Conflicts with cited docs

- **Placement (§1.9).** `state_bytes` charges f16-V bytes whatever the layout. Doc 25 L221 and doc 33 L205 record only the 4 B gap (2,180 against 2,176). The q8-V gap (26.2%) appears in neither.
- **"Shipped" (doc 33 L17).** Read here as the q8-V layout. The deployed setting is UNVERIFIED.
- **Per-card cell counts (§3).** Not comparable with doc 25 L129–132 or doc 33 L92–94, which use S = 2 and S = 4 and split placement. Not re-audited.
- **Extrapolation (doc 33 L92–96).** Doc 33 marks its k = 2 tail as extrapolated. The table here applies the same formula at k = 0–3, page-rounded; the formula rests on the UNVERIFIED applicability in §3.
- **RoPE rotation count.** The rotated dims are 64 per head (`ops_rope.comp`), not the full 256. The per-cell count in C3 uses 64.

## 8. Hygiene

This doc has no absolute paths, network addresses, host names or personal names. Papers are cited by title and arXiv id only. The llama.cpp reference is a commit hash only. Paths are repo-relative.

## 9. Arithmetic

All bytes are per stream on one card, one stream, unless stated.

- GDN: 36 × 3,268,608 = 117,669,888 B = 112.2 MiB.
- Per cell: q8-V 20,736 B; f16-V 26,112 B; q8 K+V 15,360 B.
- Full cache: N × per-cell. N = 32,768 gives 679,477,248 B (0.633 GiB). N = 100,000 gives 2,073,600,000 B (1.931 GiB). N = 262,144 gives 5,435,817,984 B (5.063 GiB).
- Usable: headroom + k × 930 MiB − 112.2 MiB. Cells = floor(usable ÷ per-cell), rounded down to 16. Headroom is 304 MiB (RX) and 8 MiB (MI50).
- C1 footprint (remapped basis): 4,112 × 20,736 = 85,266,432 B. Saving at 100K = (100,000 − 4,112) × 20,736 ÷ 2^30 = 1.852 GiB. Saving at 256K = (262,144 − 4,112) × 20,736 ÷ 2^30 = 4.983 GiB. As built (pooled keys sized by length): 4,112 × 19,200 + 1,536 × N = 232,550,400 B at 100K, saving (2,073,600,000 − 232,550,400) ÷ 2^30 = 1.715 GiB; 481,603,584 B at 256K, saving (5,435,817,984 − 481,603,584) ÷ 2^30 = 4.614 GiB.
- C3 rotation: K rotates 1,536 dims per cell over 12 layers, and pooled keys add 64 dims per block per layer, which is 16 per cell per layer or 192 per cell. Total 1,728 dims per cell.
- Indexer scan: 1,536 × N B per token: 153.6 MB at 100K, 402.7 MB at 256K.
- Attended read: 2,051 × 12 × 1,600 B = 39.4 MB per token.
- Quest: bound storage 768 × N B. Bound pass 768 × N B per token: 76.8 MB at 100K, 201.3 MB at 256K. Selected pass 128 × 16 × 1,536 B = 3.1 MB.
- DuoAttention saving: (N − 4,112) × 800 × 12 B; percentage against 20,736 × N.
- H2O: 20% budget; accumulator 48 × N B.
- Top-k compares per query: 512 × n_blocks. 12.8 M at 100K (25,000 blocks); 33.6 M at 256K (65,536 blocks).
- Placement over-charge: 2,180 ÷ 1,728 − 1 = 26.2%.
- Host, labelled estimate: 32,212,254,720 B ÷ per-cell bytes.
