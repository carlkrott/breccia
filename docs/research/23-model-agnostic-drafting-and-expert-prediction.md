# 23 — Model-agnostic drafting, expert prediction and placement

Round 23 of the paired rotation. Research only: no code was changed, and no GPU, binary, benchmark or model load was run. The only computation is CPU-only Python arithmetic, which is not committed. Every speed number is cited, marked `derived` (arithmetic on cited inputs), or marked `UNVERIFIED - requires the GPU qualification card`.

## 1. Summary

- **A (drafting from local traffic):** break-even at k=3 is p = 0.42 under the doc 21 model, 0.43 with a batched head and 0.56 with a per-row head (derived). The decision needs only a CPU-only acceptance replay on recorded text.
- **B and C (prefetch, router prediction):** each streamed use that hits VRAM saves 72 µs per token (derived). +5% tok/s needs a hit rate of 0.198. An exposed copy needs 92–97% precision, so prefetch pays only if the copy is overlapped.
- **D (self-drafting):** a full-depth top-2 draft costs d = 0.72. Quantised drafts at the measured int8 ratio cost d = 0.79 and never break even. Even a 2× byte cut leaves d = 0.49 (derived).
- **E (replication, tiers):** the residual 5.3 GiB of VRAM bounds replication at about +9% tok/s (loose derived ceiling); the stranded remainder under whole-block fill is at most about +2.9% (derived). Replicas do not shorten the serial T=1 sum. Expert offload to the second card over gen3x4 costs 602 µs per expert, 8× the CPU lane.
- **Ranked:** (1) CPU acceptance replay, (2) per-token expert-id trace, (3) residual-VRAM measurement.

## 2. Breccia state (HEAD 1e393d9)

- Model: 48 blocks, 512 experts per block, top-10. Router `blk.N.ffn_gate_inp.weight` is BF16 `[2560, 512]`, 2,621,440 B per block (`docs/phase-2/02-format-loader.md:201`, `docs/specs/qwen4exp-architecture.md:344`).
- Expert bytes: 42,912,972,800 B over 48×512 gives 1,746,133 B (1.665 MiB) per expert (derived).
- Residency: 30 of 48 blocks in a 26.7 GiB arena. The other 18 stream through the host lane at 301 MiB/token (`docs/BRIEF.md:79`). Each card holds 16 GiB (`:189`). Check: 180 streamed uses × 1.665 MiB = 299.7 MiB (derived, within 0.5%).
- Decode: streamed experts are computed on the CPU from host memory (`moe_topk`, `crates/breccia-engine/src/graph.rs:3137`). No expert is copied to VRAM on the decode path. The only copy path is the prefill chunk path (`fill_stream_slot` `graph.rs:2516`, `STREAM_SLOTS = 8` at `:271`).
- Router: runs on the device (`graph.rs:2030-2035`), ids copied out at `:1958`, allocated at `:722`. Kernel: `crates/breccia-kernels/src/ops/mod.rs:823-848` (n_expert ≤ 512, k ≤ 16, one 512-thread workgroup).
- Route trace: `BRECCIA_TRACE_ROUTE` (`graph.rs:2432`) prints distinct counts per block, not per-token expert ids.
- Verify and rollback: absent at HEAD (`verify` and `rollback_to` do not exist in `graph.rs`). The design `0acdc39:docs/design/spec-verify-gdn-rollback.md` (GO for slice 1, default off) names the call sites. `swap_r_in` (`graph.rs:2909`), `submit_head` (`:3223`) and `reset` (`:1613`) exist.
- Stubs: `crates/breccia-draft/src/lib.rs:1` and `crates/breccia-cache/src/lib.rs:1` are one-line files.
- Measured anchors (BRIEF): single stream 18.59 tok/s, 53.78 ms (§4, `docs/BRIEF.md:73`). Lane 13.5 ms at 30 resident blocks (`:159`). Fenced submit 33.2 µs, lane and GPU serial (§5.1). BF16 GEMV 810 GB/s (§5.5). Host link 25–26.5 GB/s (§5.6). Zero-copy host experts priced, not wired (`:285`).

## 3. Derived numbers

Per-expert costs (one streamed expert = 1.746 MB):

| Quantity | Value | Derivation |
|---|---|---|
| Lane cost L | 75.0 µs | 13.5 ms / (18 blocks × 10); 23.3 GB/s effective |
| VRAM hit H | 3.03 µs (2.16 at 810 GB/s) | 1.746 MB / 576 GB/s |
| H2D copy c | 65.9–69.8 µs | 1.746 MB at 26.5–25 GB/s |
| Gain per hit L−H | 72.0 µs | |
| Block window | 1.12 ms | 53.78 ms / 48 |
| Dense lead per block | 0.691 ms | (53.78 − 13.5 − 7.1) / 48; serial; head 7.1 ms from the doc 21 ladder |
| Two fence waits per block | 66.4 µs | 2 × 33.2 µs |

Doc 21 model: r = (1−s) + s·D(k+1)/10 + k·(d + h_row), Y = Σ p^i, S = Y/r, D(n) = 512(1−(502/512)^n). D(2..5) = 19.80, 29.42, 38.84, 48.08. Here h_row = 7.1 / 53.78 = 0.132 (derived). Batched head 0.01 is an assumption (UNVERIFIED).

| Case | s | h_row | d | p* k=1 | k=2 | k=3 | k=4 | S, p=0.6, k=3 |
|---|---|---|---|---|---|---|---|---|
| doc 21 (no head) | 0.230 | 0 | 0 | 0.226 | 0.335 | 0.417 | 0.481 | 1.308 |
| batched head | 0.230 | 0.010 | 0 | 0.236 | 0.347 | 0.430 | 0.493 | 1.285 |
| per-row head | 0.230 | 0.132 | 0 | 0.358 | 0.480 | 0.563 | 0.623 | 1.057 |
| s = 13.5/53.78 | 0.251 | 0 | 0 | 0.246 | 0.359 | 0.442 | 0.506 | |

Union effect: verifying k=3 reads D(4)/10 = 3.88× one token's expert set under uniform routing (derived; skew shrinks it).

Prefetch hit rate (B, C, E): gain per token = 180 uses × h × 72 µs = 12.95·h ms.

| Hit rate h | Gain | tok/s (from 18.59) |
|---|---|---|
| 0.10 | +2.5% | 19.05 |
| 0.198 | +5.0% | 19.52 |
| 0.25 | +6.4% | 19.79 |
| 0.50 | +13.7% | 21.14 |
| 1.00 | +31.7% | 24.49 |

- Exposed copy: break-even precision c/(L−H) = 0.92–0.97 (derived). Hidden copies need concurrent arena buffers (doc 22 Top 1).
- Lead: 10 copies take 0.66–0.70 ms against a 0.69 ms lead, so about 10 candidates per block fill it. The window allows 17.
- Ping-pong at T=1: copy plus VRAM compute is 68.1–72.8 µs against L = 75.0, a saving of 2.9–9.3% of lane time (derived).
- Router GEMV: 62.91 M elements, 125.8 MB. Per block 2.62 MB = 3.24 µs at 810 GB/s; full pass 0.155 ms/token (0.29% of T1; 0.218 ms at 576 GB/s).
- Crossings (layer split): `mixed` 10,240 B = 3.53 µs at 2.9 GB/s; HC state 40,960 B = 14.1 µs.
- Other card over gen3x4: 1.746 MB / 2.9 GB/s = 602 µs per expert. MI50 expert GEMV for 10 experts per block: 21.2 µs at 825 GB/s. NVMe at 5.9 GB/s: 296 µs per miss (not a tier).
- Residual VRAM: 32 − 26.7 = 5.3 GiB = 3,259 expert copies = 6.4 blocks. This is a loose ceiling: dense weights, head, KV, scratch and the arena reserves (`arena.rs:85-86`) are not subtracted. One extra resident block = 10 × 72 µs = 0.72 ms = +1.4% tok/s; six blocks = +8.7%; 3,259 copies at uniform h = 3,259 / 9,216 = 0.354 = +9.3%. Stranded remainder under whole-block fill is below one block arena per device (911 MiB, derived), so at most 1,095 experts across both devices: h = 0.119, 1.54 ms, +2.9% (derived ceiling).
- Self-draft d: dense 33.18/53.78 = 0.617, draft head 3.04/53.78 = 0.057 (3.04 ms = 7.1 × 106,299 / 248,320, the draft-vocab head in doc 21 line 35; derived), experts 0.2 × 13.5/53.78 = 0.050; total 0.724 for top-2. Hypothetical byte cuts, not QSpec: int8 at the measured ratio 1/1.18 gives 0.792; cutting the bytes to 0.5 gives 0.491; to 0.25 gives 0.274. Hash lookup at layer 12 of 48: 0.617 × 12/48 = 0.154 (uniform per-block cost).

| Draft | d | p* k=1..4 |
|---|---|---|
| full-depth top-2 | 0.724 | 0.949, 0.964, 0.972, 0.977 (S at p=0.6, k=3: 0.567) |
| int8 at measured ratio 1/1.18 (not QSpec) | 0.792 | none (no break-even at any p) |
| byte cut 0.5 (not QSpec) | 0.491 | 0.716, 0.795, 0.839, 0.867 |
| byte cut 0.25 (not QSpec) | 0.274 | 0.499, 0.615, 0.687, 0.736 |
| hash lookup, layer 12 | 0.154 | 0.380, 0.503, 0.585, 0.643 |

## 4. Findings by question

**A. Model-agnostic drafting (prompt-lookup, suffix automaton, REST, agent n-gram tables).** A proposer reads only token history and drafts k tokens at d = 0. One multi-row pass verifies them. Acceptance must exceed the p* in §3 (0.42 / 0.43 / 0.56 at k=3). Verify also reads the union of experts and needs per-row logits and GDN snapshots (3,268,608 B per block; 112.2 MiB for 36 GDN blocks, derived from the 0acdc39 design). Acceptance on local agent output is UNVERIFIED: no measurement exists. The proposer is cheap; the verify path is the cost. Plug-in: `crates/breccia-draft/src/lib.rs:1` (stub).

**B. N-gram to expert prefetch.** An n-gram match predicts token ids, not expert ids. Expert ids are known only after the block's router runs. Prefetch therefore needs a token-conditioned table (token to expert set per block), which needs a per-token trace that `BRECCIA_TRACE_ROUTE` does not emit (`graph.rs:2432`). Each prefetch costs c (66–70 µs) on a copy queue and pays only if the expert is used. The §3 table gives the hit rate needed (+5% at h = 0.198). Two blockers: decode has no copy path (`graph.rs:3137`), and arena weights are EXCLUSIVE (`model.rs:683-700`, doc 22 Top 1).

**C. Cross-layer router prediction.** Pre-gated MoE (arXiv 2308.12066) trains the gate to name the next block's experts. These weights are not pre-gated, so an untrained predictor applies block b+1's router to block b's MoE input. Cost: 3.24 µs per block (derived). Hit rate is UNVERIFIED: it needs a CPU reference over real hidden states, which this round cannot run. Gate: predicted top-10 hit rate ≥ 0.198 on that reference, with the pass threshold fixed before the run. ProMoE and ExpertFlow are named in the brief, not fetched, and have no arXiv id here. arXiv 2308.12066 was not read either (same session limit); its pre-gating description is UNVERIFIED against the paper. Plug-in: `graph.rs:2030-2035`, with block router weights loaded at `model.rs:474`.

**D. Quantised self-speculation, reduced top-k, hash lookup.** A full-depth top-2 draft costs d = 0.72 and needs p ≥ 0.95: rejected. QSpec (arXiv 2410.11305) was not read: WebFetch hit the session limit, so its draft cost model is UNVERIFIED. The rows here model a draft as a byte cut only. If QSpec's draft keeps the same weights at lower activation precision, a bandwidth-bound GEMV gains little, and the int8 row is the honest analog. The only measured quantised path is int8 at 1.11× (RX) and 1.18× (MI50), which fails accuracy (`docs/BRIEF.md:188`). At 1/1.18 the draft costs d = 0.79. A hidden-state hash lookup costs at least the dense prefix to its layer (d = 0.154 at layer 12). Its hits are the same repetition quantity as A, so A's replay gates it first.

**E. Replication, placement and tiers.** EPLB-style redundant experts (DeepSeek EPLB, repo not fetched; doc 20 row 19) balance expert-parallel load. For one serial T=1 stream they only add VRAM hits, and VRAM is the limit: 5.3 GiB is left before reserves (loose), so the ceiling is +9.3%; the stranded remainder under the arena's block rule is at most about +2.9%. From the brief's tier order (RX VRAM, MI50 VRAM, host RAM, page cache), the MI50 is reachable from the RX only over gen3x4 (602 µs per expert), so it acts as a compute tier, not a cache tier. NVMe is a miss cost, not a tier. LFU/LRU swap with hysteresis: a promotion costs about 68 µs once and each reuse saves 72 µs, so payback is one reuse. Promote only when expected reuse in the window is ≥ 1. The BRIEF §7 layer-split rejection cites "activation shipping dominates", but derived shipping is 3.5–14.1 µs per crossing; the 33.2 µs fence is the larger term.

## 5. Top 3 (with gates)

1. **CPU acceptance replay for A (decides whether verify work starts).** What: replay prompt-lookup, suffix-automaton and n-gram proposers over recorded local agent outputs (text only, no model load), measuring acceptance for k = 1..4. Moves: the choice between 1.29× (batched head) and 1.06× (per-row head) at p = 0.6 (derived). Evidence: §3 table; `crates/breccia-draft/src/lib.rs:1`. Gate: acceptance at k=3 ≥ 0.43 to start verify work, else stop. Speed: UNVERIFIED - requires the GPU qualification card.
2. **Per-token expert-id trace (unlocks every B, C and E gate).** What: make `BRECCIA_TRACE_ROUTE` (`graph.rs:2432-2436`) emit per-token expert ids per block. No kernel change. Moves: an offline replay of static, LFU, LRU and predictor hit rates at the VRAM budget. Gate, fixed before the run: replay hit rate ≥ 0.198 and ≥ 5 points above the static-frequency arm (doc 19 gate). Speed: UNVERIFIED - requires the GPU qualification card.
3. **Stranded-VRAM measurement after reserves, then a per-expert fill (E).** What: at the 30-block configuration, measure each device's free bytes after dense weights, head, KV, scratch and the arena reserves (display 768 MiB, headroom 256 MiB; `arena.rs:85-86`). The arena fills whole blocks, so the remainder is below one block arena per device. Moves: a per-expert fill of both remainders is bounded at +2.9% tok/s (derived ceiling). Evidence: `docs/BRIEF.md:79`, `:189`, `crates/breccia-engine/src/arena.rs:8-39`, `placement.rs:156`. Gate: gain = (stranded bytes / 1.665 MiB) × replayed per-token hit rate × 180 × 72 µs; ship only if it reaches +1% on the item 2 replay (derived: 0.53 ms, 7.4 uses per token, about 379 experts or 630 MiB stranded). Free bytes: UNVERIFIED - requires the GPU qualification card. A fill is a load-time change, not a per-token copy.

**Rejected:** full-depth top-2 self-draft (d = 0.72, p* ≥ 0.95); quantised self-draft at the measured int8 ratio (d = 0.79, no break-even) and BRECCIA_GEMV_INT8 as a drafter (accuracy fails, `:188`); expert offload to the second card over gen3x4 (602 µs per expert); per-expert host import (doc 22); EPLB replicas as a serial-sum speedup (doc 20 row 19); stream ping-pong at T=1 as a main lever (2.9–9.3% of lane time; prefill unquantified); NVMe as a tier (weights never move to NVMe).

## 6. Doc-to-code conflicts

1. Doc 21's break-even omits the per-row head that `0acdc39` includes (k=3: p* 0.42 vs 0.56).
2. T1 basis: BRIEF §4 gives 53.78 ms; doc 21's ladder row 30 gives 59.0 ms. On 59.0, h_row would be 0.120. This doc uses 53.78.
3. Host-lane share: doc 21 uses s = 0.23 as a floor; BRIEF's lane 13.5 / 53.78 gives s = 0.251 (k=3: p* 0.417 vs 0.442).
4. BRIEF §7 rejects the layer split for "activation shipping dominates"; derived shipping is 3.5–14.1 µs per crossing, and BRIEF has no crossing measurement. The fence is the larger term.
5. Decode has no copy path for streamed experts (`graph.rs:3137`), so copy-based prefetch is new code.
6. `BRECCIA_TRACE_ROUTE` prints distinct counts (`graph.rs:2432`); doc 19 #1 needs an ordered per-token trace.
7. `breccia-cache`: BRIEF `:59` lists it as expert residency primitives, and `crates/breccia-cache/src/lib.rs` is a one-line stub. BRIEF `:60` names only draft and server as stubs.
8. The verify design (`0acdc39`) cites `graph.rs:2102` and `:2244-2256`. Neither is a verify path at HEAD, which has no `verify` or `rollback_to`.
9. Bandwidth basis: docs 21 and 22 use 576 GB/s (spec); BRIEF §5.5 uses 810 GB/s (measured). Both are shown in §3.
10. The 5.3 GiB residual (BRIEF 32 − 26.7) is before the arena reserves (`arena.rs:85-86`: 768 MiB display, 256 MiB headroom), and the arena fills whole blocks, so the stranded remainder is the real per-expert headroom (derived, §3).

## 7. UNVERIFIED list

- Acceptance of proposers on local agent output (A). Batched head cost ≈ 0.01 T1 (A). Per-row head 7.1 ms is the doc 21 ladder, not a per-row measurement (A).
- Predictor hit rate and ProMoE / ExpertFlow results (C). Hash-lookup shallow match rate (D).
- Skew-dependent hit rates. Doc 20b's synthetic zipf (180 to 33–104 host uses per token) is synthetic, not a gain (B, E).
- Free VRAM per device after dense weights, head, KV, scratch and the arena reserves (E; reserves are constants at `arena.rs:85-86`). QSpec's draft cost model (D; abstract not read). Prefill gain from ping-pong (E). Copy overlap with GPU work (B, E; needs doc 22 Top 1).
- Every speed or gain figure: UNVERIFIED - requires the GPU qualification card.

## 8. Sources

- arXiv 2411.04975 (SuffixDecoding), 2311.08252 (REST), 2410.11305 (QSpec), 2308.12066 (pre-gated MoE): ids from the round brief, not fetched in this round (WebFetch hit the session limit).
- ProMoE, ExpertFlow: named in the brief, no arXiv id, not fetched. DeepSeek EPLB: named in doc 20, repo not fetched.
- Repo: `docs/BRIEF.md` (§4 lines 73, 79, 159; §5.1, §5.5–5.7, lines 128, 285; §7 lines 186–193); `docs/phase-2/02-format-loader.md:201`; `docs/specs/qwen4exp-architecture.md:344`; docs 19, 20, 21, 22; `0acdc39:docs/design/spec-verify-gdn-rollback.md` (not in the tree); `9a38d27:docs/research/20b-frequency-per-byte-packing.md` (not on this branch).
- Code: `graph.rs` :271, :722, :1958, :2030-2035, :2432, :2516, :3137; `model.rs` :474, :683-700; `placement.rs:156`; `ops/mod.rs:823-848`; `arena.rs` header.

## 9. Not read and overlaps

- Not read: `crates/breccia-engine/src/xfer.rs`; `crates/breccia-kernels/src/cpu/expert.rs` (CPU lane); `arena.rs` past its header; `model.rs` past the router declaration; `head.rs`; most of `graph.rs`. `docs/design/` is absent in the worktree. No binary, benchmark or model load was run.
- Overlaps: doc 19 (routing trace gate, relayout, predictor rows), doc 20 (EPLB row 19, frequency packing row 17), doc 20b (block-granular packing gains nothing), doc 21 (break-even, union, GDN rollback), doc 22 (EXCLUSIVE arena and copy-queue sync contract), `0acdc39` design.
