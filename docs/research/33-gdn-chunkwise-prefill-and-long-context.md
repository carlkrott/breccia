# 33 — GDN chunkwise prefill and long context: what is worth building

Round 33, 2026-10-28. Scope: the Gated DeltaNet (GDN) recurrence and the context capacity that GDN state and QSA KV set, at base `8fd6654`. Four questions: (1) chunkwise WY/UT GDN prefill, (2) context of 100K tokens and more, (3) recurrent-state snapshots at chunk boundaries, (4) parity of a chunked path against the recurrent one. Read-only: no code change, no build, no GPU run.

Short names used below: **doc 01** = `docs/phase-4/01-forward-pass.md`, **doc 02** = `docs/phase-4/02-untried-levers.md` (no 11.1 heading; see §7 item 3), **doc 09** = `docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md`, **doc 15** = `docs/research/15-lmcache-mooncake-tiered-kv.md`, **doc 16** = `docs/research/16-sglang-radix-hicache-chunked-prefill.md`, **doc 21** = `docs/research/21-speculative-mtp-and-rpc.md`, **doc 25** = `docs/research/25-exl3-trellis-and-kv-quant-context.md`, **doc 27** = `docs/research/27-prefill-pipeline-and-et-vk.md`, **doc 3-06** = `docs/phase-3/06-attention-gdn.md`, **BRIEF** = `docs/BRIEF.md`. **RX** = RX 6950 XT (gfx1030, wave32, 576 GB/s per BRIEF:16). **MI50** = gfx906 (wave64). This round uses 825 GB/s for the MI50; BRIEF:17 gives about 1 TB/s nominal, and 825 is not re-measured here.

Citation convention: a bare path is in this repo at base `8fd6654`, and `L` is a line in that file. Upstream projects are named, not fetched (§8).

Denominator labels, used on every time figure: **[probe]** = GPU-time probe sum, 30,319 µs (BRIEF:88). **[wall]** = single-stream 18.59 tok/s, 53.78 ms per token (BRIEF:73). **[prefill]** = token-by-token prefill, 6.45 tok/s, 155 ms per token (doc 01 L2059; BRIEF:80 gives 6.0–6.45). The probe configuration (dual, 30 resident blocks) and the wall configuration are different runs, so probe-to-wall ratios are approximate.

## §0 Verdict

**Chunked prefill is a small lever. The GDN scan is 2.86 ms of a 155 ms prefill token (about 1.8%), and the whole GDN sublayer is at most 5.7%. Both ratios apply probe-config times to a prefill token, an assumption that Gate 0 must check (§9).** The WY/UT chunk form (card 4) therefore has a ceiling of about 1.8% of prefill, and it is the largest card. Prefill is slow for other reasons: doc 27 §0 puts the streamed-group cost at 429–530 µs per expert.

The part of the scan that is worth building is its memory traffic. Every token reads and writes the whole state twice and once (2R+1W, 339.7 MB per token), and prefill repeats that for each of its tokens, because prefill dispatches one token at a time (graph.rs:2336, :2611). Two flag-OFF cards target this. Card 1 keeps the state column in registers. The reverted 32-lane prior art (commit 808920a) attacked the same scan and gave wrong logits, so card 1 must be debugged from its gate, not copied. Card 2 dispatches T rows per call, which removes per-token submission; with card 1, the state is then read and written once per dispatch instead of twice per token.

Long context at 100K is set by KV bytes and displaced arena blocks, not by GDN state. GDN state is 112.2 MiB per stream whatever the length. Shipped KV is 20,736 B per cell, which is 1.93 GiB at 100K cells. Per doc 25's table, 100K per stream at two streams on RX needs about three displaced arena blocks, which is an extrapolation.

Gate 0 comes first (§Gate 0). Cards are ranked in §5. Every flag defaults OFF. Each gain is UNVERIFIED - requires the GPU qualification card, unless it is a bound derived from a measured stage time, which is labelled.

## Gate 0 (measurement only, no flag)

Per-stage GPU-time attribution of the prefill path. Confirm that the `BRECCIA_GPU_TIME` probe brackets the GDN stages on the per-token prefill path. The probe enters are at `crates/breccia-kernels/src/gdn/mod.rs` L468–492, and the prefill entry points are `prefill_block` (graph.rs:2258) and `prefill_block_merged` (graph.rs:2566).

Accept gate: (a) the probe stages (GDN_PROJ, GDN_Z, GDN_BETA, GDN_ALPHA, plus the scan and output stages) cover at least 90% of the GDN sublayer time on each card; (b) the scan share of a prefill token is measured, replacing the derived 1.8%; (c) the only change is probe labels. Receipt: one stage table per card. UNVERIFIED - requires the GPU qualification card.

## 1. Question 1 — chunkwise / WY-UT GDN prefill

### 1.1 What the code does per token (verified)

- The recurrence is `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`. The header formulas are at `crates/breccia-kernels/shaders/gdn_step.comp` L3–12. The header folds `dec` into `sk`; the code applies it in `d`. The two forms are equal.
- The kernel computes `dec = exp(GATE)` (L82). The `sk` loop is a serial per-thread sum over `i` (L85–88). `d = (v_j − sk·dec)·β` (L89). The update is `st = ST[idx]·dec + k·d` (L94), written back (L95), with the output accumulated serially (L96) and stored (L98). One invocation owns one (head, column) pair (header L3, L26–29). No barrier appears (L29), and the workgroup is 128 threads (L43).
- Traffic: each token reads the state for `sk` and again for the update, then writes it. Per layer that is 3,145,728 B × 3 = 9.44 MB per token. For 36 layers it is 339.7 MB per token. The one-pass floor (1R+1W) is 226.5 MB, which is 274.5 µs at 825 GB/s.
- Measured scan: 2,860 µs (BRIEF:95, [probe]) is 118.8 GB/s, 14.4% of 825. BRIEF:279 states 119 GB/s, which matches the 2,860 µs run. The BRIEF:279 passage also states 2,768 µs, which gives 122.7 GB/s (14.9%). The two figures are from different runs. The bounds below use 2,860 µs, and 2,768 µs would give 1.79%.
- Prefill is per token. `run_gdn` (graph.rs:3375) uses a session GDN workspace sized for one token (graph.rs:667–669, `g.workspace(gpu, 1)`). The prefill chunk named in BRIEF:80 ("chunk 64") is the number of tokens per prefill call, not a GDN chunk. This doc writes `C_gdn` for the GDN chunk.

### 1.2 Upstream chunked forms (not fetched)

From knowledge, not fetched in this round (§8). Flash-linear-attention ships a chunked gated delta rule in a WY-style representation. vLLM and SGLang reach GDN prefill through fla-style kernels for the Qwen-Next family. llama.cpp has a chunked build in `delta-net-base.cpp`; the repo cites two different line ranges for it (§7 item 11).

Derived form, written from the algebra and not fetched. Per head, with a chunk of `C_gdn` tokens, let `g_t` be the in-chunk log of the cumulative decay. Ratios `exp(g_t − g_s)` are used so that `1/P_t` never overflows: f32 overflows once the in-chunk log-decay exceeds ln(f32 max) = 88.72.

- `A[t][s] = β_t (k_tᵀ k_s) exp(g_t − g_s)` for s < t, zero elsewhere. A is strictly lower, C×C, and it is per v-head.
- `T = (I + A)⁻¹`, computed by forward substitution.
- `W = T · diag(β) · K̃` and `U = T · diag(β) · (V − K̃ · S̄₀)`, where `K̃` carries the decay factors.
- `S̄_C = S̄₀ · exp(g_C) + K̃ᵀ (U − W · S̄₀)`.
- `O = Q̃ · S̄₀ − tril(Q̃ Kᵀ)(W · S̄₀) + tril(Q̃ Kᵀ) · U`, with `Q̃` the decay-scaled queries.

The decay placement is schematic. The f64 test in card 4 must fix it against the recurrence.

Tiling: the state per head is 128×128 f32 = 64 KiB. Tile over d_v in 32-column tiles = 16 KiB. The three v-heads that share a k-head (h_v = 48, h_k = 16; `crates/breccia-kernels/src/gdn/mod.rs` L62, L64) can share `K·Kᵀ`.

### 1.3 Cost (derived)

Per head per 64 tokens, S = 128, C = 64.

- Recurrent: 3·S²·C = 3,145,728 MAC.
- W-form chunk: four products of size C·S² (4,194,304) plus five triangle products of size C(C+1)/2·S (1,331,200). Total 5,525,504 MAC, which is 1.76×.
- d-form variant: three state products plus four triangles = 4,210,688 MAC, which is 1.34×.
- No reuse of `K·S̄₀` or `W·S̄₀`: about 8.4 M MAC, which is 2.7×.

With reuse the chunk form costs 1.34–1.76× the FLOPs, and without reuse 2.7×, so it wins only on traffic. Per layer per 64 tokens, the recurrent path moves 604 MB of state (64 × 9.44 MB). The chunk form moves one read and one write of the state per head, which is 2 × 64 KiB × 48 heads = 6.3 MB. That is 96× less state traffic, before the K, V and Q streams. If W and U go through device memory (LDS paragraph below), the chunk path moves about 12.6 MB, which is 48× less.

LDS fit (derived against a recorded limit). The working sets are K chunk 64×128 f32 = 32 KiB, A/T 64×64 f32 = 16 KiB and state tile 128×32 f32 = 16 KiB. Together they are 64 KiB, which is the whole limit. The limit is recorded: `maxComputeSharedMemorySize` is 65,536 B on both cards (`docs/research/01-vulkan-and-mlc.md` L106), and the harness reports 64 KiB on the RX (`docs/research/06-vulkan-gemm-shapes-and-ncnn.md` L113). W (64×128 f32 = 32 KiB) does not fit alongside those three sets, so the chunk form must split. Kernel A forms A/T, then W and U. Kernel B does the state update and output. W and U go to device memory and kernel B reads them back. Kernel A still holds K and A/T, which leaves 16 KiB: enough for one 32-column tile each of W and U (8 KiB each), but not all 128 columns. So kernel A must also tile over columns or stream K. That partition is UNVERIFIED - requires the GPU qualification card. The split adds about 6.3 MB of W and U traffic per layer per 64 tokens (derived: 32 KiB each, written and read, for 48 heads). Accumulation: all chunk matmuls and the forward substitution for T accumulate in f32. Card B (f16 state) must not move accumulation to f16. No cooperative matrix is available on either card (doc 25 L47, L202), so the chunk matmuls are plain GLSL.

Prefill gain (derived): the ceiling is the scan share, 2.86 ms / 155 ms = 1.8% [prefill]. Even a zero-cost chunk form removes at most that. Realised gain: UNVERIFIED - requires the GPU qualification card.

### 1.4 Verdict on Q1

Chunked GDN is correct in exact arithmetic, but GDN is not where prefill time goes. The ceiling is 1.8% of a prefill token. Build cards 1 and 2 first. They test the traffic model on the GPU at lower cost. Card 4 comes only after card 1 shows the scan is traffic-bound.

## 2. Question 2 — context of 100K and more

### 2.1 KV and indexer bytes per cell (derived, checked against doc 15)

Per QSA layer per cell: K f16 = 2 KV heads × 256 × 2 B = 1,024 B. V f16 is the same. q8_0 is 36 B per 32 values, so K q8_0 and V q8_0 are each 576 B per layer. The indexer keeps one f32 128-dim key per 4-cell block, which is 128 B per cell per layer (`crates/breccia-kernels/src/qsa/mod.rs` L37–48). There are 12 QSA layers.

| Layout | Bytes per cell (12 QSA layers + indexer) | At 100K cells | Cells in a 30 GiB host budget (upper bound) |
|---|---:|---:|---:|
| f16 K + f16 V | 26,112 (25.5 KiB) | 2.43 GiB | 1.23 M |
| Shipped: K f16 + V q8_0 | 20,736 (20.25 KiB) | 1.93 GiB | 1.55 M |
| K q8_0 + V q8_0 (doc 25 card A) | 15,360 (15.0 KiB) | 1.43 GiB | 2.10 M |

Sums: f16 = 12 × 2,048 + 1,536 = 26,112. Shipped = 12 × 1,600 + 1,536 = 20,736. q8 + q8 = 12 × 1,152 + 1,536 = 15,360. Cross-check: doc 15 L61 gives K f16 12 KiB, V f16 12 KiB and V q8 6.75 KiB per token, which matches 12 × 1,024 B, 12 × 1,024 B and 12 × 576 B.

The host budget is 30 GiB (BRIEF:29, about 30 GB; doc 16 L68 uses the same figure). The cell counts are upper bounds, because the mmap'd experts (40,925 MiB), PLE (27,465 MiB) and token_embd (260 MiB) share the same host RAM (doc 01 L67). The mmap'd pages are file-backed and compete with KV for that RAM, so the 30 GiB is not free.

### 2.2 Tokens that fit VRAM after weights and arena

Headroom after the auto arena is about 304 MiB on RX and about 8 MiB on MI50 (doc 25 L121). N_max = (budget_card/S − GDN_card)/KV_bytes_card, with budget = headroom + k × 930 MiB (doc 25 L142). Per stream, from doc 25 §2.5:

- RX, shipped, S = 2: 7,501 (k = 0), 47,811 (k = 1), 88,121 (k = 2) (doc 25 L129).
- MI50, shipped, S = 2: 0, 51,244, 107,678 (doc 25 L132).
- RX, shipped, S = 4: 913, 21,068, 41,223 (doc 25 L130). The k = 3 to 5 values are extrapolated and are not in the table.

On RX each displaced block adds about 40,310 cells at S = 2, so 100K needs k = 3, extrapolated to about 128,000 cells. MI50 reaches 100K at k = 2. At S = 4, RX reaches 100K at about k = 5 on its own slope of about 20,155 cells per block. The table stops at k = 2, so both figures are extrapolations, UNVERIFIED - requires the GPU qualification card.

Displaced-arena cost: doc 25 L146 gives an exchange rate of about 0.8 ms per token per GiB of KV, about 1.5% of [wall]. The rate is UNVERIFIED - requires the GPU qualification card. At the shipped 1.93 GiB that is about 1.5 ms per token, about 2.9% of [wall].

### 2.3 Spill and host reads

The RX-only spill of K f16 + V q8 is 21.9 MiB per stream per token, which is 0.80 ms at 28.7 GB/s (doc 25 L164; doc 09 L58). That is about 1.5% of [wall] if it is not overlapped. The MI50 link is 2.9 GB/s in doc 25 L176 and doc 01 L1619, but doc 09 L154 calls 2.84 GB/s a harness artefact (§7 item 5). The indexer keys stay resident (doc 25 L177).

### 2.4 Indexer scan is O(context)

Each decode query reads every cell's indexer key: 1,536 B per cell per query. At 100K cells that is 153.6 MB per token, which is 0.27 ms at 576 GB/s (RX) or 0.19 ms at 825 GB/s (MI50) (derived lower bounds). The GPU loop is not verified. The CPU reference loops over every block with a causal mask (`crates/breccia-kernels/src/qsa/cpu.rs` L95–112). The selection is `IDX_TOP_K = 2048` cells, which is 512 blocks of 4 (`crates/breccia-kernels/src/qsa/mod.rs` L83; doc 25 L177).

### 2.5 GDN does not grow with context

The GDN state is 3,268,608 B per layer at any context length (§3). At 100K the decode cost is set by the indexer scan (above), by the expert streaming that displaced blocks force, and by the KV reads for the 2,048 selected cells.

### 2.6 Verdict on Q2

Long context is bounded by KV bytes and by displaced arena blocks. The relevant levers are doc 25's card A (K q8) and card B (GDN state f16), plus the indexer read pattern. No new GDN lever is needed for context.

## 3. Question 3 — recurrent-state snapshots at chunk boundaries

- GDN state per layer: 3,145,728 B for the recurrent state (`crates/breccia-kernels/src/gdn/mod.rs` L88 allocation, MODEL_S = 128 at L60), plus 122,880 B of conv history (3 × 10,240 × 4; `CONV_CHANNELS` at L68 is 10,240, asserted at L72). Total 3,268,608 B per layer (doc 15 L69). For 36 layers that is 117,669,888 B = 112.2 MiB per snapshot.
- Where it lives. The live state for one stream is already about 112 MiB on device (doc 25 L96 gives a 117 MiB GDN-plus-workspace residual). MI50 headroom is about 8 MiB (doc 25 L121), so a snapshot cannot stay on device there. RX headroom is about 304 MiB, enough for at most two. Snapshots therefore go to host memory, and the copy crosses the card link.
- Copy costs. Device-local read and write of 117.7 MB at 825 GB/s is 0.29 ms (MI50, derived). Doc 21 L56 gives one copy of 0.204 ms at 576 GB/s (RX). A host transfer at 28.7 GB/s (doc 09 L58, pinned, RX) is 4.1 ms per snapshot. At 2.9 GB/s (MI50 link, doc 25 L176) it is 40.6 ms per snapshot. Written every 64 tokens, the RX host copy is 0.064 ms per token, about 0.04% of [prefill]. The MI50 link figure is about 0.63 ms per token, about 0.4% of [prefill].
- Grid. `G = lcm(C, 16)` (doc 16 L87). The QSA pages are 16 cells (`crates/breccia-kernels/src/qsa/mod.rs` L89). With `C_gdn = 64`, G = 64. A snapshot interval must be a multiple of 16 (doc 15 L77).
- Count per 100K tokens. At I = 16K, 6 snapshots = 673 MiB (doc 15 L69). At G = 64, 1,562.5 snapshots = 184 GB, which is about 171 GiB (doc 16 L68). The 30 GB host tier cannot hold that.
- Write policy. GDN snapshots: write-through at capture, admitted only on a second fork hit (doc 16 L107). QSA pages: write-back on full pages. Fork-point snapshot at R−1 (doc 16 L133).
- Latency. A 64-token re-prefill at 155 ms per token costs 9.9 s [prefill]. A host restore of 112 MiB costs 4.1 ms (RX link) to 40.6 ms (MI50 link). So snapshots buy prefix-cache latency, not throughput. The repo has no snapshot or prefix-cache code (the only match is the GPU probe's `snapshot_probe`, graph.rs L1377), so card 5 is new code.

## 4. Question 4 — parity

- Chunked and recurrent GDN are equal in exact arithmetic but not bit-identical. The chunk form changes the summation order and the decay factorisation. The recurrent kernel's serial sums (`gdn_step.comp` L85–88 and L92–97) are the order a bit-identical card must keep.
- Tolerance bar: 1e-5 for GDN state and output (doc 3-06 L224–232).
- Gates, by flag state.
  - **Flag OFF (default), chunked path compiled or not:** greedy parity is not the runtime gate, because the path is not taken. The gate is byte identity with base at `8fd6654`: `cmp` of the logits and of a state dump, on the same prompts. The CPU f64 test for the chunked form still runs.
  - **Flag ON, bit-identical cards (1, 2):** top-1 13.170510 and 32/32 greedy tokens at prefill chunk 8, 16, 32 and 128 (BRIEF:77), then `scripts/head-handoff-race.sh` (3 runs), and no `vkWaitForFences(u64::MAX)` (BRIEF:230).
  - **Flag ON, non-identical cards (3, 4):** 32/32 greedy identical to the recurrent path at chunk 8, 16, 32 and 128. Top-1 must match. Logit digits need not match byte for byte. State max error ≤ 1e-5 against the f64 CPU reference on both cards. Then `scripts/head-handoff-race.sh` (3 runs).
  - **Promotion to default ON** needs both sets, plus a long-horizon check. Doc 25 L190–194 gives the pattern for card B.

## 5. Ranked cards (every flag default OFF)

Gate 0 is listed in its own section above and is not counted as a card.

**Card 1 — register-resident state column in `gdn_step` (`BRECCIA_GDN_COLREG`, OFF).**
- Files: `crates/breccia-kernels/shaders/gdn_step.comp`; flag plumbing in `crates/breccia-kernels/src/gdn/mod.rs` (`forward`, L457); the CPU reference in `crates/breccia-kernels/src/gdn/cpu.rs` (`delta_step`, L125).
- Mechanism: each thread keeps its 128-float column in registers across the whole dispatch. The `sk` and update loops then read no global state. The i-order is unchanged, so the arithmetic is identical and bit identity is expected. Traffic falls from 2R+1W per token to 1R+1W per dispatch at any T. At T = 1 (decode) that removes one of the three passes. Prefill calls the kernel with one token per dispatch (§1.1), so card 1 saves prefill traffic only together with card 2.
- Risks: about 128 VGPRs for state plus temporaries, so spill must be checked per ISA on gfx1030 and gfx906 (UNVERIFIED - requires the GPU qualification card). Occupancy is grid-limited, not VGPR-limited: the launch is 48 workgroups of 128 threads, which the reverted commit 808920a diagnosed as 1.6 wave64 per CU on the MI50, so VGPR count does not set occupancy at this grid size.
- Prior art: commit 808920a. The 32-lane fix measured 1,547–1,920 µs (from 2,768 µs) but gave wrong logits, and it was reverted. The cause is not located. The commit lists three suspects: `v_base+j`, the `gh` pairing, and the 32-lane xor reduce crossing a Wave64 boundary (UNVERIFIED - requires the GPU qualification card). A register column with the same i-order has no reduction, so it avoids the reduce suspect. Do not revive the 32-lane variant as-is.
- Gate: flag ON against base: bitwise equal O and state at T = 1 and T = 8, 16, 32, 128. The CPU f64 `delta_step` within 1e-12 on the test set. 32/32 greedy and top-1 13.170510. `scripts/head-handoff-race.sh` (3 runs).
- Receipt: ISA VGPR and spill report for gfx1030 and gfx906; bitwise `cmp` of O and state against base at T = 1, 8, 16, 32 and 128; scan µs before and after under `BRECCIA_GPU_TIME`; 32/32 greedy log and top-1 13.170510; `scripts/head-handoff-race.sh` output, three runs.
- Gain: an upper bound, not an estimate. If the scan were purely traffic-bound, dropping one of its three passes would save 2,860 µs × 1/3 = 0.95 ms per decode token [probe], about 1.8% of [wall] (approximate, the probe and wall configurations differ). The doc's own evidence argues against a purely traffic-bound scan: it runs at 14.4% of peak bandwidth (§1.1), commit 808920a diagnosed it as MLP-starved, and its 32-lane speedup came from more parallelism, not fewer bytes. The realised gain may be much smaller. Without card 2, prefill gains at most 0.6% of [prefill], also an upper bound. UNVERIFIED - requires the GPU qualification card.
- Interaction: doc 25 card B (f16 state) changes the column type and would halve its register cost. The two must be sequenced, not stacked silently.

**Card 2 — T-row GDN dispatch for prefill (`BRECCIA_GDN_TROW`, OFF).**
- Files: `crates/breccia-engine/src/graph.rs` (`run_gdn` L3375; `prefill_block` L2258; `prefill_block_merged` L2566; the per-token `record_dense_prelude` at L2336 and L2611; the one-token GDN workspace at L667–669); `crates/breccia-kernels/src/gdn/mod.rs` (`workspace(gpu, tokens)`, L248, already takes a token count).
- Mechanism: run the GDN projections, conv, gate and scan for T rows in one dispatch set, hoisted above the per-token MoE loop. The math is unchanged. The conv test `conv_chunking_matches_per_token_calls` (`cpu.rs` L349) is the pattern for a multi-token call.
- Blockers to check first: the HC stage (`hc-read` 5,300 µs with 96 dependent reads, BRIEF:93) and the PLE block-1 dependence. Whether the MoE loop can move out is UNVERIFIED - requires the GPU qualification card, and was not measured here.
- Gate: bitwise logits and state equal to base at T = 8, 16, 32 and 128. 32/32 greedy. `scripts/head-handoff-race.sh` (3 runs).
- Receipt: bitwise `cmp` of logits and state against base at T = 8, 16, 32 and 128; 32/32 greedy log; `scripts/head-handoff-race.sh` output, three runs; dispatch count per prefill block before and after.
- Gain: submission overhead only. No doc gives a number. UNVERIFIED - requires the GPU qualification card.

**Card 3 — batched projection GEMM for the GDN block (`BRECCIA_GDN_PROJ_GEMM`, OFF).**
- Files: `crates/breccia-kernels/src/gemm.rs` (`GemmMany`, L24); `crates/breccia-kernels/src/gdn/mod.rs`.
- Mechanism: the GDN projection and output matmuls (gdn-proj 3,240 µs, BRIEF:94 [probe]; gdn-out 2,790 µs, BRIEF:95 [probe]; quant type not checked here) batched over T rows as a GEMM. The accumulation order changes, so this is not bit-identical.
- Gate: tolerance 1e-5 on projection outputs and state against base. 32/32 greedy and top-1 match. `scripts/head-handoff-race.sh`.
- Receipt: max-error log for projection outputs and state against base (≤ 1e-5); top-1 and 32/32 greedy logs; `BRECCIA_GPU_TIME` stage µs for gdn-proj and gdn-out before and after; `scripts/head-handoff-race.sh` output, three runs.
- Gain ceiling: 6,030 µs = 3.9% of [prefill] if all projection time disappears. The realised share is UNVERIFIED - requires the GPU qualification card.

**Card 4 — WY/UT chunkwise GDN prefill (`BRECCIA_GDN_WYCHUNK`, OFF; `C_gdn` = 64).**
- Files: new shaders under `crates/breccia-kernels/shaders/` (none exist yet); an f64 chunked reference beside `delta_step` in `crates/breccia-kernels/src/gdn/cpu.rs`; `crates/breccia-kernels/tests/attn_conformance.rs`; dispatch in `crates/breccia-kernels/src/gdn/mod.rs`.
- Mechanism: §1.2 and §1.3. Tile over d_v at 32 columns. A per v-head. `K·Kᵀ` shared across the three v-heads of each k-head. Log-ratio decay.
- Gate: the CPU f64 chunked form matches recurrent within 1e-12 before any GPU work (cpu.rs test pattern, L290–407). GPU state error ≤ 1e-5 against f64 (doc 3-06 L224–232). 32/32 greedy and top-1 against recurrent at chunk 8, 16, 32 and 128. `scripts/head-handoff-race.sh` (3 runs). Both RX (wave32) and MI50 (wave64).
- Receipt: CPU f64 chunked-vs-recurrent test output (≤ 1e-12); GPU state error log against f64 (≤ 1e-5); 32/32 greedy log at chunk 8, 16, 32 and 128; top-1 log; `BRECCIA_GPU_TIME` stage µs for the chunk kernels and the scan; `scripts/head-handoff-race.sh` output, three runs.
- Gain: ceiling 2.86 ms = 1.8% of [prefill] for a zero-cost chunk form. The chunk form adds 1.34–1.76× FLOPs with reuse (2.7× without) in exchange for 96× less state traffic (48× with W and U in device memory, §1.3), so the win depends on whether the scan is traffic-bound, which card 1 tests. UNVERIFIED - requires the GPU qualification card.
- Effort: the largest card. Do it after card 1 has tested the traffic model.

**Card 5 — chunk-boundary state snapshots (`BRECCIA_GDN_SNAP`, OFF).**
- Files: no snapshot or prefix-cache code exists in `crates/` (grep this round). New code goes in `crates/breccia-engine/src/graph.rs` and the cache crate `crates/breccia-cache/src/`. `crates/breccia-cache/src/lib.rs` is a one-line stub at this base.
- Mechanism: capture state and conv history on the grid G = 64, and copy to host, because device headroom cannot hold them (8 MiB MI50, 304 MiB RX). Write-through admitted on a second fork hit (doc 16 L107). Fork point at R−1 (doc 16 L133).
- Gate: restore followed by continuation gives logits byte-equal to an uninterrupted run at the same flag state. Flag OFF is byte-identical to base. Host memory stays inside the 30 GiB budget. `scripts/head-handoff-race.sh`.
- Receipt: restore-vs-uninterrupted logit `cmp` at the same flag state; host RSS before and after capture and restore, inside the 30 GiB budget; flag-OFF `cmp` against base; `scripts/head-handoff-race.sh` output.
- Gain: latency only. A 64-token re-prefill costs 9.9 s [prefill]. A host restore costs 4.1 ms (RX link) to 40.6 ms (MI50 link). UNVERIFIED - requires the GPU qualification card.

**Card F — existing doc 25 cards (not re-proposed).**
- Card A: K q8 for the QSA layers, saving 168 MiB per 32K stream (doc 25 L9, L144).
- Card B: GDN state f16, saving 54 MiB per stream and cutting per-token state traffic from 339.7 MB to 169.9 MB (doc 25 L10; gate at L190–194). Card B halves the GDN state part of a snapshot, so a snapshot falls from about 112 MiB to about 58 MiB while conv history stays f32. An f16 register column (card 1) changes the arithmetic too, so sequence them.

## 6. Rejected

1. **Chunked form for decode (T = 1).** Nothing to amortise. The recurrent kernel is already the T = 1 form.
2. **C_gdn = 32.** Not eligible for snapshots until an exactness test passes. It doubles the snapshot count (doc 16 L87).
3. **A snapshot every token (G = 1).** 117.67 MB × 100,000 = 11.77 TB per 100K tokens.
4. **Full-grid write-through snapshots at G = 64.** 184 GB per 100K tokens, which is over the 30 GB host tier (doc 16 L68). Replaced by second-hit admission (card 5).
5. **Device-resident snapshots.** Headroom is 8 MiB on MI50 and 304 MiB on RX after the arena (doc 25 L121).
6. **The 32-lane shuffle variant, as shipped.** Wrong logits, reverted (808920a). It returns only as a debugged card 1.
7. **Cooperative-matrix or MMA chunk matmuls.** Absent on both cards (doc 25 L47, L202).
8. **Spilling indexer keys to host.** They are read on every query, so spilling adds traffic on every token (doc 25 L177).
9. **Batched projections as a default-ON change.** Not bit-identical. Kept as flag-OFF card 3.
10. **GDN as the main prefill lever.** The whole GDN sublayer is at most 5.7% of [prefill]. The prefill time sits in the streamed expert path (doc 27 §0).

## 7. Conflicts found (no code changed)

1. `docs/phase-4/01-forward-pass.md` L4427–4428 gives a per-cell figure of "1 KiB per cell per block" and about 9.2 KiB per cell. K and V f16 for one layer are 2 KV heads × 256 × 2 B each, so 2 KiB per layer per cell; the 1 KiB and 9.2 KiB figures disagree with that. The doc's own 818 MiB / 32,768 cells gives 25.6 KiB per cell, which matches 12 layers × 2 KiB plus the indexer (25.5 KiB).
2. The 32K figure of 728 MiB appears in BRIEF:283 and doc 01 L4556 and L4564 (labelled K q8 there; doc 25 L218 calls it K+V q8). Doc 25 L217 gives 596 MiB for K q8 + V q8 (480 MiB of KV and indexer plus 116 MiB of GDN and workspace).
3. Doc 25 L218 reports doc 02 §11.1 as an indexer of 12 KiB per token and a q8_0 row of 13,056 B, using GGUF's 34 B per 32 values; the repo uses 36 B per 32 and 128 B per cell per layer. `docs/phase-4/02-untried-levers.md` has a section 11 (L441) but no 11.1 heading, so the claim is cited as doc 25 reports it and not checked.
4. Doc 09 L212 gives 786 MiB for the 32K K-q8 row. 24,576 × 32,768 = 805,306,368 B = 768 MiB.
5. Doc 09 L12 gives the MI50 as "8.0 GT/s x16 (gen3 x16)". Doc 01 L1619 and L3740 say gen3 x4 at 2.9 GB/s. Doc 09 L154 calls 2.84 GB/s a harness artefact, and doc 25 L176 uses 2.9. Unresolved.
6. BRIEF:279 gives 2,768 µs and 119 GB/s in one passage, and 119 GB/s corresponds to 2,860 µs (BRIEF:95). Two runs. The bounds here use 2,860 µs.
7. `placement.rs` state_bytes is 2,180 B against sizes at 2,176 B (doc 25 L221). Not re-checked here.
8. The header of `crates/breccia-kernels/shaders/gdn_qk_l2.comp` (L6–7) describes a delta-rule scale of 1/S on q. Code uses `inv * inversesqrt(S)` on q rows and `inv` on k rows (L76–82); `cpu.rs` `qk_l2_norm` (L81) is the reference. Flagged, not edited.
9. `gdn_step.comp` header (L6) folds `dec` into `sk`, while the code applies it in `d` (L89). Equivalent. Noted only.
10. Doc 21 L56 uses 576 GB/s (RX) for the snapshot copy. This doc uses 825 GB/s (MI50). Card-specific, not a conflict.
11. Upstream chunk-64 cites differ. `docs/specs/qwen4exp-architecture.md` L450 cites `delta-net-base.cpp:61-64`. `docs/phase-3/06-attention-gdn.md` L522–524 cites `:16-284`. Upstream was not fetched, so this stays open.
12. Rule breach, not a doc conflict: `docs/phase-4/01-forward-pass.md` L2456 and L2465 use `vkWaitForFences(u64::MAX)` in `gemm_many`, against BRIEF:230.

## 8. External sources (not fetched)

- **Flash-linear-attention (fla):** chunked gated delta rule, WY-style. From knowledge; not fetched.
- **vLLM and SGLang:** GDN prefill through fla-style kernels for the Qwen-Next family. From knowledge; not fetched.
- **llama.cpp `delta-net-base.cpp`:** chunked build. Line cites conflict (§7 item 11). Not fetched.
- **Gated DeltaNet paper:** arXiv 2412.06464, as cited by doc 25 L244. Read only through doc 25 in this round.

Every upstream claim above is from knowledge and is unverified.

## 9. Arithmetic and denominators

Caveat: the [probe] times are from the BRIEF:88 probe run (config: dual, 30 resident blocks) and the [wall] time is the BRIEF:73 run. Neither measures prefill, so using them for a prefill token assumes the GDN kernels cost the same per token there. That is not measured; Gate 0 checks it.

- [probe] 30,319 µs (BRIEF:88); wait-gpu 30,691 µs. The GDN sublayer is 8.89 ms [probe] (proj 3,240 + out 2,790 + scan 2,860), which is 29.3% of the probe sum. The scan alone is 9.4% [probe].
- [prefill] 8.89 / 155 = 5.7%. Scan 2.86 / 155 = 1.8%. The 2,768 µs run gives 1.79%.
- Not computed: 8.89 / 53.78 mixes [probe] with [wall]. Avoided.
- [wall] scan saving 0.95 ms / 53.78 ms = 1.8%, labelled approximate.
- State and traffic: per layer 3,145,728 B state plus 122,880 B conv = 3,268,608 B. 36 layers = 117,669,888 B = 112.2 MiB. Per token 339.74 MB (2R+1W). Floor 226.49 MB (1R+1W) = 274.5 µs at 825 GB/s. Achieved 339.74 MB / 2.86 ms = 118.8 GB/s, 14.4% of 825.
- Cells in 30 GiB = 32,212,254,720 B / bytes per cell: 1,233,618 (f16), 1,553,445 (shipped), 2,097,152 (q8 + q8), rounded down.
- Indexer per query at 100K = 153.6 MB: 0.27 ms at 576 GB/s, 0.19 ms at 825 GB/s.
- Snapshot counts and bytes: §3.

## 10. Not measured or checked in this round

- VGPR counts and spill on gfx1030 and gfx906 (card 1). Occupancy is grid-limited, so it is not the open item.
- The LDS partition for card 4 (§1.3). The limit is recorded (65,536 B on both cards); no split kernel is built or compiled.
- The indexer GPU loop and its read pattern (§2.4).
- The HC and PLE dependence that gates card 2.
- The exact decay placement in the WY/UT form (card 4).
- Any upstream code: nothing was fetched (§8).
- The 32-lane wrong-logits cause (card 1).
- The N_max rows for k = 3 to 5, which are extrapolated.
