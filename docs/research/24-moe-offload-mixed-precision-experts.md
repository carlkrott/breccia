# 24 — MoE offload with mixed-precision experts

Round 24 of the paired rotation. Research only: no code was changed, and no GPU program, benchmark or model load was run. The only computation is CPU-only arithmetic, not committed. Every number is cited to a repository file, marked `derived` (arithmetic on cited inputs, shown inline), or marked `UNVERIFIED - requires the GPU qualification card`. Literature figures come from fetched abstracts and summaries, not from the PDFs, and are UNVERIFIED.

## 1. Summary

- **Q1 (2-bit shadows of cold experts).** A 2-bit shadow moves 21–25% fewer bytes per miss than the exact IQ3 expert. S1 (IQ2_XXS gate and up, Q2_0 down) is 1,305,600 B, S2 (Q2_0 gate, up and down) is 1,382,400 B, and the exact expert is 1,746,133 B (derived). On a copy path that is a real saving: 49–55 µs against 66–70 µs per miss. Decode has no copy path today; a cold use is computed on the CPU lane, and the bytes do not convert one-for-one there. Using the single-thread per-type rates of `docs/phase-3/08-cpu-expert-lane.md` §3.1 as weights, S1 cuts lane time by 16.8% (derived) and S2 by 2.1% (UNVERIFIED: the Q2_0 gate/up rate was never measured). S1 is the better shadow on both bytes (25.2% against 20.8%) and lane time. Keeping the top-1 expert exact costs 2.5 points of bytes and 1.7 points of lane time. Accuracy of every 2-bit variant is UNVERIFIED.
- **Q2 (cache policy).** The planned per-token trace (doc 23 §5 item 2) should also carry the gate weight `w`, as `token block e:w` rows. Replay four arms on one trace: static frequency fitted on a held-out half, LFU with hysteresis (the RX slot plan named in the brief), decayed frequency (half-life 256 tokens) and LRU as the control. Under uniform routing every policy caps at coverage 0.354 (derived), so only routing skew moves the hit rate, and the replay measures that skew. This round does not pick an arm.
- **Q3 (prefetch distance).** No distance result exists to test against the round-23 gate (hit ≥ 0.198 and ≥ 5 points over static frequency), because Breccia has no router replay. Distance 1 fits one block window: ten copies fill 59–62% of the 1.120 ms window (derived). Distance 2 adds latency slack, not copy throughput. If the ProMoE stride cost (about 5 points per step, UNVERIFIED) applies, distance 2 passes only if distance 1 clears static by about 10 points.
- **Ranked by value per effort:** (1) per-token route trace with `w` and offline replay; (2) tail-skip parity plus a per-type lane timing run, both CPU-only; (3) down-only Q2_0 on the 18 IQ4_NL down layers (+1.9–2.6% tok/s, derived), which ranks above the S1 gate/up requant on value per effort. The S1 gate/up requant gives more lane time (+2.5% to +4.0%) but needs an IQ2_XXS gate/up source, and the tree has no source precision or imatrix for one.

## 2. Breccia state (worktree base af556bd)

Doc 23 cites HEAD 1e393d9. The points below were read in this worktree at af556bd.

- Model: 48 blocks, 512 experts per block, top-10. Gate and up are 640×2560 and down is 2560×640, each 1,638,400 elements (`docs/phase-3/08-cpu-expert-lane.md` §3.1 table).
- Block geometry, `crates/breccia-format/src/quant.rs` `block_geometry` (lines 142–177): IQ2_XXS 256 values / 66 B; IQ2_XS 256 / 74; IQ2_S 256 / 82; IQ3_XXS 256 / 98; IQ3_S 256 / 110; IQ4_NL 32 / 18; Q2_0 64 / 18 (fork-specific; `pub const Q2_0` at line 89, doc comment at line 88).
- Bytes per matrix (derived from the geometry): gate/up = (1,638,400 / 256) × B. IQ2_XXS 6,400 × 66 = 422,400; IQ2_XS 6,400 × 74 = 473,600; IQ2_S 6,400 × 82 = 524,800; IQ3_XXS 6,400 × 98 = 627,200; IQ3_S 6,400 × 110 = 704,000. Down: IQ4_NL 51,200 × 18 = 921,600; Q2_0 25,600 × 18 = 460,800.
- Layer types (`docs/specs/model-artifacts-inventory.md` §4.2, the expert index; legend lines 422–423): gate/up type 16 ×9, 17 ×10, 18 ×6, 21 ×13, 22 ×10 (sum 48); down type 42 ×30 (Q2_0), 20 ×18 (IQ4_NL). Rows 0, 1 and 47 are printed; line 413 elides the rest, so the per-layer pairing is unknown. Cross-check (derived): the index's `blob_bytes` is 2 × gate/up + down. Row 0 gives 1,868,800 = 2 × 473,600 + 921,600; row 1 gives 1,305,600 = 2 × 422,400 + 460,800. Row 1 is already in the S1 layout, so S1 saves nothing there, and the class arithmetic in §3 gives it zero.
- Residency and budget: 30 of 48 blocks resident in a 26.7 GiB arena; 18 blocks stream through the host lane at 301 MiB per token; lane 13.5 ms; token 53.78 ms; 18.59 tok/s (`docs/BRIEF.md` lines 73, 79, 159). Host RAM is 58 GB, and the brief's success shape uses about 30 GB of it (`docs/BRIEF.md` lines 14–32).
- Decode path: streamed experts are computed on the CPU in `moe_topk` (`crates/breccia-engine/src/graph.rs:3137`, inside `finish_block`). The streaming copy `fill_stream_slot` (`graph.rs:2516`, called at 2734 from `prefill_block_merged`, 2566) is the prefill path. It copies into `STREAM_SLOTS = 8` host-visible slots (struct doc 250–256; constant 271).
- Router readback: `graph.rs:416` and `:461` describe a readback layout with router ids and router weights. Whether decode fills the weights to the host per block is UNVERIFIED (not read).
- Route trace: `BRECCIA_TRACE_ROUTE` (`graph.rs:2432`, inside `prefill_block` at 2258) prints distinct counts per block on the prefill path. No per-token expert-id trace exists on decode.
- Multi-user CPU GEMV: `crates/breccia-kernels/src/cpu/expert.rs:602` uses the grouped path only for Down shards with more than one user; `crates/breccia-kernels/src/cpu/mod.rs:134–136` allows it only on AVX2 and only for Iq4Nl or Q2_0. Single-stream decode never reaches it.
- Reserves: display 768 MiB, headroom 256 MiB (`crates/breccia-engine/src/arena.rs:85–86`).
- Host memory: 58 GB DDR5 and 42.913 GB of routed expert tensors (`docs/phase-3/08-cpu-expert-lane.md` lines 263–264).

## 3. Derived numbers

**Bytes per use and per token.** Sum over 48 layers per expert index: 83,814,400 B. Per layer: 83,814,400 / 48 = 1,746,133.3 B (1.6652 MiB). Over 512 experts: 83,814,400 × 512 = 42,912,972,800 B, matching the 42.913 GB at phase-3/08 lines 263–264. Lane uses per token: 18 blocks × 10 = 180. Lane bytes per token: 180 × 1,746,133.3 = 314,304,000 B = 299.7 MiB (BRIEF: 301 MiB, 0.4% apart). Bytes-to-time: 301 MiB = 315,621,376 B over 13.5 ms = 23.4 GB/s (the BRIEF basis, used throughout; the derived 299.7 MiB gives 23.3 GB/s).

**Per-use costs.** L = 13.5 ms / 180 = 75.0 µs. H (VRAM hit) = 1,746,133 / 576 GB/s = 3.03 µs (2.16 µs at 810 GB/s). L − H = 71.97 µs. H2D copy c, exact: 1,746,133.3 / 26.5 GB/s = 65.9 µs and / 25 GB/s = 69.8 µs. S1: 1,305,600 B gives 49.3 µs and 52.2 µs. S2: 1,382,400 B gives 52.2 µs and 55.3 µs. Block window = 53.78 ms / 48 = 1.120 ms. Ten copies = 0.659–0.698 ms, which is 58.8–62.3% of the window (derived). NVMe miss = 1,746,133 / 5.9 GB/s = 296 µs (5.9 GB/s is the BRIEF figure at QD8, `docs/BRIEF.md` line 145, not re-measured; a single miss at lower queue depth is slower, so 296 µs is a floor). Move-or-compute (Fiddler rule): c + H = 65.9 + 2.16 = 68.1 µs to 69.8 + 3.03 = 72.8 µs, against L = 75.0 µs, a gain of 2.9–9.2%.

**Shadow sizes and class savings (per expert index, 48 layers).** The class arms are histogram-exact. Only the combined S2 min rule depends on the layer pairing.

| Arm | Bytes saved | Share of 83,814,400 |
|---|---|---|
| gate/up-only S1 (IQ2_XXS gate/up) | 2 × (26,700,800 − 48 × 422,400) = 12,851,200 | 15.3% |
| down-only Q2_0 (IQ4_NL down → Q2_0) | 30,412,800 − 48 × 460,800 = 8,294,400 | 9.9% |
| S1 (both) | 21,145,600 | 25.2% |
| gate/up-only S2 (Q2_0 gate/up), net | 53,401,600 − 48 × 921,600 = 9,164,800 | 10.9% |
| gate/up-only S2, min rule | 9,164,800 + 9 × 76,800 = 9,856,000 | 11.8% |
| S2 (both), net | 9,164,800 + 8,294,400 = 17,459,200 | 20.8% |
| S2 (both), min rule | 17,459,200 + 76,800 × n, n ≤ 9 | 20.8–21.7% |

Where Σ gate/up = 9×422,400 + 10×473,600 + 6×627,200 + 13×704,000 + 10×524,800 = 26,700,800, and Σ down = 18×921,600 + 30×460,800 = 30,412,800. The min rule adds back 76,800 B for each (type 16, type 42) layer, where IQ2_XXS gate/up plus Q2_0 down (1,305,600 B) is smaller than S2 (1,382,400 B). Check: the gate/up-only net plus the down-only saving equals the S2 net, 17,459,200.

Per-token substitution (uniform routing, 180 uses): S1 = 180 × 1,305,600 = 235,008,000 B (−25.2%). S2 = 180 × 1,382,400 = 248,832,000 B (−20.8%). Top-1 exact with the other 162 uses shadowed (HOBBIT-style): S1 = 18 × 1,746,133.3 + 162 × 1,305,600 = 242,937,600 B (−22.7%); S2 = 31,430,400 + 223,948,800 = 255,379,200 B (−18.8%).

**Per-type CPU time model.** Rates are the AVX2 single-thread GB/s of `docs/phase-3/08-cpu-expert-lane.md` §3.1 (lines 122–135). Matrix time = bytes / rate, with GB/s = B/ns. Per-type times are single-thread. They are applied only as a fraction s of the 13.5 ms lane; absolute single-thread times are never compared with L.

| Type | Tensor | Rate (GB/s) | Matrix bytes | Matrix time (µs) |
|---|---|---|---|---|
| IQ2_XXS | gate/up | 2.89 gate, 3.05 up; mean 2.97 | 422,400 | 142.2 |
| IQ2_XS | gate/up | 2.98, 2.88; mean 2.93 | 473,600 | 161.6 |
| IQ2_S | gate/up | 3.20, 3.13; mean 3.165 | 524,800 | 165.8 |
| IQ3_XXS | gate/up | 3.82, 3.82 | 627,200 | 164.2 |
| IQ3_S | gate/up | 3.55, 3.55 | 704,000 | 198.3 |
| IQ4_NL | down | 3.15 | 921,600 | 292.6 |
| Q2_0 | down | 2.51 | 460,800 | 183.6 |
| Q2_0 | gate/up | not measured; down rate used | 460,800 | 183.6 (UNVERIFIED) |

Histogram-weighted per expert: exact gate/up = 2 × (9×142.2 + 10×161.6 + 6×164.2 + 13×198.3 + 10×165.8) / 48 = 2 × 8,116.9 / 48 = 338.2 µs. Exact down = (18×292.6 + 30×183.6) / 48 = 224.5 µs. Exact total = 562.7 µs.
- S1 = 2×142.2 + 183.6 = 468.0 µs, saving 94.7 µs = 16.8%. Of which gate/up-only S1 saves 338.2 − 284.4 = 53.8 µs (9.6%) and down-only Q2_0 saves 224.5 − 183.6 = 40.9 µs (7.3%).
- S2 = 3 × 183.6 = 550.8 µs, saving 11.9 µs = 2.1% (UNVERIFIED, since the Q2_0 gate/up rate is missing). Gate/up-only Q2_0 costs 367.2 − 338.2 = 29.0 µs more per expert; 35 of the 48 layers get slower (only the 13 IQ3_S gate/up layers are faster).

**Regime (which model applies).** phase-3/08 lines 144–146 describe a single core as compute-bound: per-type decode rates are flat from a 1 MB working set to 121 MB (2.94 against 2.69 GB/s on the same type). The thread sweep (§6.2) is the only in-repo evidence on the regime, and it does not label DRAM limits. One thread gives 3.06–3.09 GB/s (lines 271, 284). Eight threads give 20.41 GB/s (line 272), or 2.55 GB/s per thread: a 6.6× scaling, which is 82.5% parallel efficiency (6.6 / 8 = 0.825, derived). The source states about 86% as per-worker 2.55 GB/s against the 2.9–3.1 GB/s kernel ceiling (line 293); the two measures differ, and both are cited. Twelve threads give 21.1–23.9 GB/s and 16 threads 22.5–30.5 GB/s (lines 288–289), a flat range that the source does not label; reading it as DRAM-limited is an inference (UNVERIFIED). Production runs at 23.4 GB/s (derived), inside that flat range. Phase-4/02 line 28 reads the 32-thread lane as bytes-bound (conflict 14), which would make bytes decide; that holds only if the production thread count is confirmed. The production thread count is UNVERIFIED. `docs/phase-4/02-untried-levers.md` line 213 benchmarks 32 threads at 33 GB/s, and at 33 GB/s the 301 MiB lane would take 315.6 MB / 33 GB/s = 9.6 ms, not 13.5 ms (derived). So both models are reported, and the timing run in §5 item 2 decides between them.

**Lane-time bounds.** tok/s = 1000 / (53.78 − 13.5 × s), where s is the lane-time saving fraction.

| Arm | Bytes model: s, tok/s (vs 18.59) | Per-type model: s, tok/s |
|---|---|---|
| S1 (both) | 25.2%, 19.85 (+6.8%) | 16.8%, 19.41 (+4.4%) |
| S2 (both) | 20.8%, 19.62 (+5.5%) | 2.1%, 18.69 (+0.5%) (UNVERIFIED) |
| gate/up-only S1 | 15.3%, 19.34 (+4.0%) | 9.6%, 19.05 (+2.5%) |
| down-only Q2_0 | 9.9%, 19.07 (+2.6%) | 7.3%, 18.94 (+1.9%) |
| Top-1 exact + S1 for the rest | 22.7%, 19.72 (+6.1%) | 15.1%, 19.33 (+4.0%) |
| Lane to zero (cap) | 100%, 24.83 (+33.5%) | — |

**Host RAM.** Exact streamed set: 9,216 × 1,746,133.3 = 16,092,364,800 B = 14.99 GiB. S1 shadow set: 9,216 × 1,305,600 = 12,032,409,600 B = 11.21 GiB. S2 shadow set: 9,216 × 1,382,400 = 12,740,198,400 B = 11.87 GiB. Dual, exact + S1: 28,124,774,400 B = 28.12 GB = 26.19 GiB. Dual, exact + S2: 28,832,563,200 B = 28.83 GB = 26.85 GiB. Against about 30 GB of host RAM (`docs/BRIEF.md` lines 14–32), headroom is 1.88 GB (S1) and 1.17 GB (S2). Substitution, with the exact set dropped, saves 14.99 − 11.21 = 3.78 GiB (S1) or 14.99 − 11.87 = 3.12 GiB (S2). With a partial shadow fraction f: ΔRAM = 11.21·f GiB (S1), and Δbytes ≈ 22.7%·f under uniform routing in the top-1 exact variant.

**VRAM pool, uniform routing.** Residual 5.3 GiB = 5,690,831,667 B (32 − 26.7 GiB, `docs/BRIEF.md`). Exact copies: 5,690,831,667 / 1,746,133.3 = 3,259, coverage 3,259 / 9,216 = 0.354. Gain 0.354 × 180 × 71.97 µs = 4.58 ms, which gives +9.3% (replicates doc 23). S1 copies: 5,690,831,667 / 1,305,600 = 4,358, coverage 0.473, 1,099 more copies (+33.7%). Gain 0.473 × 180 × 72.73 µs = 6.19 ms, which gives +13.0% (UNVERIFIED - requires the GPU qualification card; every served use is 2-bit). S1 pool fill: 4,358 × 1,305,600 = 5.69 GB, which takes 0.215–0.228 s at 25–26.5 GB/s.

**Prefetch and predictors.** Gain per token = 180 × h × 71.97 µs = 12.95·h ms. At h = 0.198 that is 2.57 ms, +5.0% tok/s. Exposed-copy break-even precision = c / (L − H) = 65.9/71.97 = 0.92 to 69.8/71.97 = 0.97. Learned CPU predictor (ProMoE, ~200 µs per layer, UNVERIFIED for Breccia): 18 × 0.2 = 3.60 ms per token, break-even h = 3.60 / 12.95 = 0.278 > 0.198. On the device, a ~2M-parameter f16 predictor (the ProMoE size, UNVERIFIED for Breccia) is 4 MB, which reads in 4 MB / 576 GB/s = 6.9 µs per block (derived). Chance for 10 of 512 is 10/512 = 0.0195, so the gate 0.198 is 10.1× chance. Router: 2,621,440 B per block (512 × 2560 × 2, BF16), 4.55 µs on the device at 576 GB/s, 112.0 µs on the host at 23.4 GB/s (2,621,440 / 23.4 GB/s = 112.0 µs). Host DRAM demand: CPU lane 23.4 GB/s + H2D 25 GB/s ≈ 48.4 GB/s combined (derived; contention UNVERIFIED).

**Cache-policy arithmetic.** A promotion costs c = 65.9–69.8 µs once; a reuse saves L − H = 71.97 µs. Payback is therefore 0.92–0.97 reuses, so one reuse is the break-even hysteresis. Under uniform routing a resident set of 3,259 of 9,216 copies gets 0.354 of uses under any policy. Normalised top-k weights average 1/10 = 0.1 for top-10 and 1/2 = 0.5 for top-2 (derived).

## 4. Findings by question

**Q1. 2-bit shadow of cold experts, against streaming the IQ3 expert.**

- *Where the cost sits.* Decode computes a cold expert on the CPU (`graph.rs:3137`) and copies nothing. A shadow helps decode only through the CPU lane, unless a copy path is added (new code; the prefetch items in doc 23 §4 B and §5 item 2 would need one). Both cost models are therefore reported.
- *Bytes moved per miss (copy path).* Exact 1,746,133 B (65.9–69.8 µs), S1 1,305,600 B (49.3–52.2 µs), S2 1,382,400 B (52.2–55.3 µs). Per token, S1 saves 25.2% and S2 saves 20.8% (§3).
- *Lane time (decode path).* The per-type model gives S1 a 16.8% lane saving (+4.4% tok/s) and S2 2.1% (+0.5%, UNVERIFIED). Two reasons. The Q2_0 down kernel is the slowest in the table (2.51 GB/s, against 3.15 for IQ4_NL down). And Q2_0 gate/up was never measured, while IQ2_XXS gate/up is the cheapest gate/up type in time (142.2 µs per matrix). S1 keeps its gate/up value; S2 does not.
- *Which model applies.* UNVERIFIED (§3, regime). A single core is compute-bound, so per-type time decides there; production's 23.4 GB/s sits in the flat 12–32-thread range (phase-4/02 line 28 reads that range as bytes-bound, conflict 14). The production thread count and one per-type timing run at that count decide which model holds. That run is CPU-only but is a benchmark, so it was not run here.
- *Accuracy risk.* No 2-bit variant of this model has a quality result in the tree. The Q2_0 release GGUF is outside the tree (inventory line 28), so inventory §3 is a count check, not a quality measurement. HOBBIT (fetched summary, UNVERIFIED) reports int2 for cold experts with small loss on its models; its Table 3 conflicts with its summary (§6, conflict 9). The gate is the round-12 accuracy gate that `docs/research/18-flashinfer-and-ft-moe.md` line 88 names for a change that is not bit-identical: per-stage rel L2 ≤ 2e-3, top-1 agreement, logit delta ≤ 0.02, and PPL +0.3%. A 2-bit shadow is not bit-identical, so the gate applies. Any substitution is lossy, so its threshold is an owner decision (§5, item 2).
- *Host RAM.* Substitution saves 3.78 GiB (S1) or 3.12 GiB (S2). The HOBBIT-style top-1-exact variant needs the full exact set plus the shadow: 26.2 GiB (28.1 GB) with S1 and 26.9 GiB (28.8 GB) with S2, against about 30 GB. Headroom is 1.9 GB or 1.2 GB, so this is likely binding (measured headroom at the 30-block configuration UNVERIFIED).
- *VRAM 2-bit pool.* The same arithmetic gives +13.0%, but every served use is 2-bit, so it is a substitution. Not recommended until the accuracy gate passes.
- *Verdict.* A 2-bit shadow saves bytes on both models: S1 −25.2%, S2 −20.8% per token. On the CPU lane, S1 saves 16.8% of lane time (+4.4% tok/s, per-type model) and S2 saves 2.1% (+0.5%, UNVERIFIED). S1 beats S2 on both measures, so S2 is not the better shadow. Keeping the top-1 expert exact (HOBBIT-style) costs 2.5 points of bytes (25.2 → 22.7) and 1.7 points of lane time (16.8 → 15.1). Gate: §5, items 2 and 3.

**Q2. Trace-based cache policy for the planned route trace and the LFU+hysteresis RX plan.**

- *Trace contents.* Per token and per block, 10 (expert, w) pairs, written as `token block e:w` rows. Doc 23 §5 item 2 asked for ids only. This round adds `w` for three reasons: tail-skip (w < τ) keys on it, HOBBIT-style thresholds key on it, and decayed frequency could be weighted by it. No flag name is chosen here.
- *Where the trace comes from.* Decode has no trace today. The ids are copied out at `graph.rs:1958` (doc 23), and the readback layout includes weights (`graph.rs:461`), but whether weights reach the host per block in decode is UNVERIFIED. If not, the readback is a new sync point and must be timed before use.
- *Arms, all replayed on the same trace with a held-out split.*
  1. Static frequency: top-M per block by count on the first half, scored on the second. Its in-sample score is an upper bound, not the baseline.
  2. LFU with hysteresis (the RX slot plan): promote on reuse, and evict only past a margin. It matches the measured payback of one reuse (§3).
  3. Decayed frequency, half-life 256 tokens (capy-flash, `docs/reference-engines/01-expert-streaming.md` line 36).
  4. LRU (colibri per-layer LRU at reference line 37 and MoE-Infinity LRUPolicy at line 38), as the control.
  A fifth arm, LFRU (Infernix score at reference line 33; colibri hot-store score with its "25%+4 hysteresis" at line 37), is optional.
- *Fit.* LFU with hysteresis is the best-fitting arm: its promotion rule matches the payback arithmetic, and it is the RX slot plan the brief names. Decayed frequency is the nearest alternative if skew drifts. LRU promotes on every miss at c = 66–70 µs each, so it wins only when reuse falls inside its window; the replay measures that. Under uniform routing all four arms tie at 0.354 (derived). The trace must show skew before any arm is preferred. The gate picks the arm; this round does not pick one.
- *Shadow subset (Q1).* RAM for shadows is worth most where use frequency is highest, so the trace ranks which experts get shadows.

**Q3. Prefetch distance, against the round-23 gate (hit ≥ 0.198 and ≥ 5 points over static frequency).**

- *Gate restated.* The gate is a predicted top-10 hit rate measured on Breccia's own replay, with static frequency as the comparator. Prefetch copies go into a 1.120 ms window per block, and ten copies fill 59–62% of it (derived). Distance 1 fits that window. Distance d > 1 adds latency slack (copies can start d blocks earlier), not copy throughput: the H2D bytes per block are unchanged.
- *Results in hand.*
  - ProMoE (arXiv 2410.22134, fetched summary, UNVERIFIED): a learned 2-layer predictor (~2M parameters, ~200 µs on CPU) reaches 84.7% on average, against 58.3% token-based and 66.9% skip-based. One extra stride step costs about 5 points. These are the paper's own setup numbers.
  - HOBBIT (arXiv 2411.01433, fetched summary, UNVERIFIED): next-layer top-1 about 96%. Not Breccia's numbers.
  - ExpertFlow (arXiv 2410.17954, fetched summary): predicts all layers from the input token sequence in one pass. That suits prefill and verify. At T=1 decode there is no future token to read, so it does not transfer. Its hit ratio appears as 91.96% in one place and 91.90% in another (§6, conflict 10).
  - In-repo engine results, not Breccia's (`docs/reference-engines/01-expert-streaming.md`): capy-flash's post-uni predictor (next layer's router on the current layer's MoE input) was predicted at +11–14% and measured at −2.7%. The notes attribute the loss to PCIe copies taking DRAM bandwidth (+13% µs per CPU expert), and prefetch was turned off (line 251). colibri's PILOT predicts layer l+1's top-K from layer l's post-attention state at 71.6% recall, against 41.3% for the naive baseline (line 294; as the notes state, not re-measured here).
- *Usability against the gate.* No Breccia replay of any distance exists, so no result passes the gate. Two usable points. (a) Distance 1 fits the one-block window (59–62% used, derived); distance 2 adds slack, not throughput. (b) If the ProMoE stride cost (about 5 points, UNVERIFIED) applies, distance 2 loses about 5 points against distance 1. Distance 2 clears the gate only if distance 1 clears static by about 10 points (5 for the gate plus about 5 for the stride cost).
- *Host bandwidth.* H2D (25 GB/s) and the CPU lane (23.4 GB/s) draw on the same host DRAM, about 48.4 GB/s combined (derived). capy-flash's −2.7% is the same contention, measured in another engine. The distance-1 fit assumes none, which a replay cannot measure. This is a separate gate line (UNVERIFIED).
- *Verdict.* No prefetch-distance result is usable as a pass. The distance decision waits for the item-1 replay at d = 1 and d = 2, plus a contention measurement once a copy path exists.

## 5. Top 3 with gates

1. **Per-token route trace with `w`, and offline replay (dependency: doc 23 §5 item 2, extended).** What: a decode-path trace of 10 (expert, w) pairs per block per token. No kernel change if weights already reach the host per block (UNVERIFIED); otherwise a readback that must be timed. Moves: replay of static, LFU+hysteresis, decayed frequency and LRU (Q2); d = 1 and d = 2 prefetch hit rates (Q3); the tail-skip distribution (w < τ) for item 2; the shadow subset for Q1. Gate, fixed before the run: replayed hit ≥ 0.198 and ≥ 5 points over static, with static fitted on the first half and scored on the second. Collecting the trace needs a model run on the GPU qualification card; none was run here. Speed: UNVERIFIED - requires the GPU qualification card.

2. **Tail-skip parity and a per-type lane timing run (CPU-only, no new artifact).** What: (a) time the existing CPU lane per type at the production thread count, to pick the bytes or per-type model (§3); (b) skip uses with w < τ, which removes both bytes and compute for the skipped use. This is AdapMoE-style, reclassified here as an arm of this run rather than rejected. Moves: lane saving = (skipped-use fraction) × 13.5 ms (derived). Gate: (a) the timing decides the ranking, and nothing ships on the bytes model alone; (b) the round-12 accuracy gate of `docs/research/18-flashinfer-and-ft-moe.md` line 88 on the same prompts, full path against skipped path: per-stage rel L2 ≤ 2e-3, top-1 agreement, logit delta ≤ 0.02, PPL +0.3%. Any τ that changes outputs is a lossy threshold. Its value is an owner decision, flagged to the parent and not chosen here. Speed: UNVERIFIED - requires the GPU qualification card.

3. **Down-only Q2_0 on the 18 IQ4_NL down layers (S2 down arm).** What: replace the 18 IQ4_NL down matrices with Q2_0. The §4.2 histogram already types 30 down tensors as Q2_0 (type 42), so the Q2_0 down kernel is in use. The Q2_0 release GGUF is not in the tree (inventory line 28), so the 18 layers need a requant from the IQ4_NL down tensors of the model file, which is also outside the tree; the step cannot be run or checked here, and its quality is UNVERIFIED. Moves: 9.9% of bytes, 1.34 ms per token, +2.6% tok/s on the bytes model; 7.3% of lane time, 0.98 ms, +1.9% on the per-type model. It ranks above the S1 gate/up requant on value per effort: the work is an 18-layer requant into a format the engine already runs, while the gate/up S1 (+2.5% to +4.0%) needs a new IQ2_XXS gate/up source with no precision or imatrix in the tree. Gate: parity on the down-only arm (as in item 2b), and the item-2a timing must confirm the per-type saving. No ship on a parity failure. Speed: UNVERIFIED - requires the GPU qualification card.

Parked behind the measured host headroom at the 30-block configuration: the dual exact-plus-shadow host RAM variant (needs 11.2 GiB with S1 or 11.9 GiB with S2 free beyond the exact set; UNVERIFIED).

**Rejected.**
- AdapMoE per-layer expert-count allocation (abstract: 25% fewer experts, 1.35×, UNVERIFIED). Needs per-layer sensitivity sweeps. Its per-token tail skip is kept as item 2.
- Learned CPU predictor. Host break-even h = 0.278 > 0.198 (§3). The device predictor (~7 µs per block) is cheap, so revisit only with a trained predictor and a measured hit rate.
- Fiddler in-place-vs-move beyond the cost rule. The move costs 68.1–72.8 µs against 75.0 µs on the CPU, a gain of 2.9–9.2%, and decode has no copy path to use it.
- HOBBIT full three-tier cache. Its third tier is the system NVMe: one expert read at the BRIEF's QD8 rate costs at least 296 µs (derived), 3.9× the 75.0 µs CPU-lane cost per use, so NVMe is not a tier for these experts. The 5.3 GiB VRAM residual is the other limit.
- HOBBIT thresholds T1 = 0.6 and T2 = 0.9 as published. Set on top-2 models (UNVERIFIED); the mean top-10 weight is 0.1 (derived).
- ExpertFlow token-sequence predictor for decode. At T=1 there is no future token.
- MoE-Infinity as new work. Already covered (README list and phase-4/02).
- Q2_0 gate/up shadow on the CPU lane. The time model costs 29.0 µs more per expert, with 35 of 48 layers slower (derived). IQ2_XXS gate/up is the right gate/up shadow.
- S2 as a standalone CPU-lane shadow. 2.1% of lane time (UNVERIFIED), and S1 beats it on bytes and lane time.
- VRAM 2-bit pool. A substitution (+13.0%, UNVERIFIED) until the accuracy gate passes.
- Full-depth top-2 self-draft (doc 23).
- Dual host RAM as the default. Headroom is 1.2–1.9 GB; parked as above.

## 6. Doc-to-code conflicts

1. Decode copy claim. The brief frames streamed experts as a CPU memcpy on decode, and doc 22 line 18 says the same (`graph.rs:254-255, :553`). The code: the memcpy into `STREAM_SLOTS` is the prefill merged path (struct doc `graph.rs:250–256`; `fill_stream_slot` `:2516`, call site `:2734` in `prefill_block_merged` `:2566`). `:553` is a prefill stats field (`pf_stream_bytes`), not a decode copy. Decode computes the cold expert on the CPU in `moe_topk` (`:3137`). A decode shadow changes compute, not a copy, unless a copy path is built.
2. Route trace. `BRECCIA_TRACE_ROUTE` prints distinct counts on the prefill path (`graph.rs:2432`). Doc 23 §5 item 2 expects per-token ids. None is in the tree.
3. Inventory verification. `docs/specs/model-artifacts-inventory.md` §3 records a Q2_0 allocation and a GGUF check, but the GGUF is outside the tree (line 28 points there). The claim cannot be re-checked here.
4. Inventory elision. §4.2 prints 3 of 48 rows (0, 1, 47); line 413 elides the rest. The S2 min rule is therefore bounded (n ≤ 9), not computed.
5. Q2_0 target. Inventory line 330 gives a 2.40 bpw whole-file target for the Q2_0 release (RCO allocation record), while `quant.rs:88–89` and inventory lines 282 and 298 give 2.25 bpw for a Q2_0 block (64 values in 18 bytes). Possibly a mixed-tensor target; not reconciled.
6. Q2_0 cost. The phase-3/08 §3.1 table has Q2_0 only for down, at 2.51 GB/s, the slowest kernel in the table and below IQ4_NL down (3.15). A byte cut to Q2_0 does not give a proportional lane saving. This contradicts the bytes model for down.
7. Multi-user gate scope. `cpu/expert.rs:602` and `cpu/mod.rs:134–136` apply the grouped GEMV only to Down shards of Iq4Nl and Q2_0 with more than one user, which is the prefill case. Doc 14 point 4 (lines 13–14) raises it as possibly relevant. For single-stream decode it is never reached. The phase-3/08 per-type rates are single-user and cannot show the batched effect.
8. HOBBIT thresholds across top-k. The published T1 = 0.6 and T2 = 0.9 (fetched summary, top-2 models) do not transfer to top-10 normalised weights (mean 0.1, derived).
9. HOBBIT Table 3 against its summary. The summary says accuracy loss is "≤1%", but Table 3 shows TruthfulQA 0.90 against 0.94 for F16, a 4.3% relative drop (UNVERIFIED; PDF not checked).
10. ExpertFlow internal figures. The fetched summary gives hit ratio 91.96% and 91.90%, and memory 93.72% and "up to 93%". `docs/research/19-mlx-and-llm-in-a-flash.md` line 128 quotes the 91.96% figure and says it came from a search snippet, not the abstract, and marks it not used. This round does not use it either.
11. Doc 23 "no kernel change". Doc 23 §5 item 2 (line 92) says the trace needs no kernel change. Per-block host weights are not confirmed (`graph.rs:416` and `:461` describe the layout only). A readback would be a sync point and must be timed.
12. Doc 23 base. Doc 23 cites HEAD 1e393d9; this worktree is af556bd. Doc 23's line numbers were not re-checked at af556bd.
13. Lane rate. `docs/phase-4/02-untried-levers.md` line 17 gives 33 GB/s at 32 threads (24.5 ms per token). Token basis: 33 × 24.5 ms = 808.5 MB, within 3.5% of the 838.1 MB all-48-block token at phase-3/08 line 272 (derived), so the 24.5 ms figure is most likely a full-token figure, not an 18-block lane figure. The BRIEF lane (13.5 ms over 301 MiB) implies 23.4 GB/s (derived), about 71% of the 33 GB/s bench rate. The production thread count is therefore UNVERIFIED and is the open variable.
14. Compute or bytes at 32 threads. `docs/phase-4/02-untried-levers.md` line 28 says "Decode is a bytes problem, not an ALU problem" and puts the 32-thread lane at 33 GB/s against a 53.9 GB/s single-core stream. `docs/phase-3/08-cpu-expert-lane.md` lines 144–146 call the lane compute-bound, a single core decoding 2.5–3.8 GB/s of quants. Both can hold at different thread counts; the regime (§3) decides which model applies. This round does not resolve it.

## 7. UNVERIFIED list

- The lane regime (bytes or per-type) at the production thread count. Candidates are 12–16 (phase-3/08 range, read as DRAM-limited by inference) and 32 (phase-4/02 line 213), and they give different models (§3, conflict 13).
- The Q2_0 gate/up rate (not measured; the down rate stands in for the S2 time figure).
- Accuracy of every 2-bit variant (Q1). Quality of the Q2_0 release and of an IQ4_NL → Q2_0 requant (Q1, §5 item 3). Feasibility and quality of an IQ2_XXS gate/up requant (no source precision or imatrix in the tree).
- Whether decode fills router weights on the host per block (Q2, §5 item 1).
- Routing skew, the w distribution and the tail-skip fraction (Q2, §5 item 2). Whether Breccia's w is normalised over top-k or over all 512 (router body not read).
- Replay hit rates for every arm and distance (Q2, Q3). The ProMoE stride cost (~5 points), the ProMoE CPU predictor time (~200 µs per layer), the ProMoE predictor size (~2M parameters) and HOBBIT's next-layer accuracy (fetched summaries; PDFs not checked).
- The capy-flash (−2.7%) and colibri (71.6% recall) figures are as stated in `docs/reference-engines/01-expert-streaming.md`; neither was re-measured.
- Host headroom at the 30-block configuration (dual RAM). Host DRAM contention between H2D and the lane (Q3).
- The NVMe rate of 5.9 GB/s at QD8 (BRIEF figure, not re-measured); a single miss at lower queue depth is slower, so 296 µs is a floor.
- Every tok/s figure in this document. UNVERIFIED - requires the GPU qualification card.
- The S1 VRAM pool figure (+13.0%): UNVERIFIED, GPU card, and accuracy.
- The PPL +0.3% term of the parity gate: stated in `docs/research/18-flashinfer-and-ft-moe.md` line 88; its measurement method was not re-read.

## 8. Sources

Repository (relative paths, read in this worktree at af556bd):
- `crates/breccia-format/src/quant.rs` lines 88–89 (Q2_0 doc comment and const), 142–177 (`block_geometry`; Q2_0 row at 177).
- `docs/specs/model-artifacts-inventory.md` §4.2 lines 410–425 (expert index; legend 422–423; line 413 elision; rows 0, 1, 47); line 28 (Q2_0 GGUF outside the tree); §3 lines 282, 298, 330 (Q2_0 2.25 bpw and the 2.40 bpw RCO target).
- `docs/BRIEF.md` lines 14–32 (58 GB RAM, 32 GB VRAM, ~30 GB host RAM in the success shape); lines 73, 79, 145 (NVMe about 5.9 GB/s at QD8), 159 (token, residency, lane); 125 (PCIe 25–26.5 GB/s) and 123 (BF16 GEMV 810 GB/s), as cited in §2 and §3.
- `docs/phase-3/08-cpu-expert-lane.md` §3.1 table (lines 122–135) and §3.1 text (lines 144–146: compute-bound single core; flat per-type rates); §6 lines 263–264 (58 GB DDR5; 42.913 GB); §6.1 table (lines 271–273; 272 gives 20.41 GB/s and 838.1 MB per token), §6.2 sweep (lines 284–289), scaling text (lines 291–293).
- `docs/research/14-ktransformers-hybrid-moe.md` lines 13–14 (prefill host lane; multi-user GEMV gate) and 84 (R1 measurement).
- `docs/research/23-model-agnostic-drafting-and-expert-prediction.md` (§2 residency; §3 gates; §4 B; §5 items 1–3, with item 2 at line 92; §6 line 66 for the 3,259 copies).
- `docs/research/18-flashinfer-and-ft-moe.md` line 88 (round-12 accuracy gate).
- `docs/research/19-mlx-and-llm-in-a-flash.md` line 128 (ExpertFlow 91.96% flagged as a search-snippet figure; not used).
- `docs/research/22-vulkan-driver-surface-tiering.md` lines 7 (exclusive arena weights; sync is the blocker) and 18 (streamed experts as a CPU memcpy; the claim in conflict 1).
- `docs/reference-engines/01-expert-streaming.md` lines 33 (LFRU, Infernix), 36 (capy-flash decayed frequency, half-life 256), 37 (colibri hot-store score with "25%+4 hysteresis"; colibri per-layer LRU), 38 (MoE-Infinity LRUPolicy/ARCPolicy), 52 (colibri PILOT 71.6%), 138 (LFRU policy detail, same score formula as line 33), 251 (capy-flash −2.7%), 294 (colibri PILOT recall 71.6% vs 41.3%).
- `docs/phase-4/02-untried-levers.md` lines 17 (host lane 33 GB/s at 32 threads, 24.5 ms per token) 28 (decode is a bytes problem at 32 threads), and 213 (`--threads 32`, 33 GB/s).
- `crates/breccia-engine/src/graph.rs` lines 250–256 (struct doc, prefill streamed slots), 271 (`STREAM_SLOTS`), 416 and 461 (readback layout), 553 (prefill stats field), 1958 (router id copy, via doc 23), 2258 (`prefill_block`), 2432 (`BRECCIA_TRACE_ROUTE`), 2516 (`fill_stream_slot`), 2566 (`prefill_block_merged`), 2734 (streaming call), 3137 (`moe_topk` in `finish_block`).
- `crates/breccia-kernels/src/cpu/expert.rs:602`; `crates/breccia-kernels/src/cpu/mod.rs:134–136`.
- `crates/breccia-engine/src/arena.rs:85–86` (reserve values).

External (abstracts or fetched summaries, all UNVERIFIED; PDFs not read):
- arXiv 2411.01433 (HOBBIT); 2401.14361 (MoE-Infinity, covered); 2410.22134 (ProMoE); 2408.10284 (AdapMoE); 2402.07033 (Fiddler); 2410.17954 (ExpertFlow).

## 9. Not read and overlaps

- Not read in this round: the `graph.rs` router and readback bodies (416 and 461 layout only; 1958 and 2030–2035 via doc 23); `crates/breccia-kernels/src/cpu/expert.rs` beyond line 602; `arena.rs` beyond the reserve lines; the inventory beyond §4.2 and the §3 lines cited above; `docs/research/19` beyond line 128; the PDFs of all six literature items.
- Not run: any GPU program, benchmark, model load or CPU benchmark. The per-type lane timing in §5 item 2 is a proposed run, not a result.
- Overlaps: doc 23 (break-even, prefetch gates, the trace item, the route-trace item and its §4 B reference); doc 14 (multi-user gate, prefill grouping); doc 18 (accuracy gate, line 88); doc 19 (the ExpertFlow 91.96% figure, line 128); doc 22 (exclusive arena weights, lines 7 and 18, which carry the copy-queue sync point and the memcpy claim); reference-engines 01 (policy names, predictor and PILOT figures); phase-3/08 (per-type rates, thread regimes); phase-4/02 (33 GB/s at 32 threads); MoE-Infinity is covered in phase-4/02 and the README list and is not re-audited.
