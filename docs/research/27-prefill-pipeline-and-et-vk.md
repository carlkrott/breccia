# 27 — Prefill pipeline and ExecuTorch Vulkan: what reaches Breccia

Round 27, 2026-10-28. Scope: prefill (PP) throughput levers for merged chunked prefill (doc 02 §1 lever 1, `docs/phase-4/02-untried-levers.md:41`), from MoE-Lightning (arXiv 2411.11217), Fiddler (arXiv 2402.07033) and the ExecuTorch Vulkan backend (`pytorch/executorch`, `backends/vulkan`). Round 26's list (Burn, CubeCL, MNN, TNN, rust-gpu, krnl, strix-rt) is not repeated. Read only: no code changed, no GPU run, no model load, no build.

Short names used below: **doc 01** = `docs/phase-4/01-forward-pass.md`, **doc 02** = `docs/phase-4/02-untried-levers.md`, **doc 09** = `docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md`, **doc 19** = `docs/research/19-mlx-and-llm-in-a-flash.md`, **doc 22** = `docs/research/22-vulkan-driver-surface-tiering.md`, **doc 24** = `docs/research/24-moe-offload-mixed-precision-experts.md`, **BRIEF** = `docs/BRIEF.md`.

Citation convention: a bare path is in this repo at base `8881fc9`, and `L` is a line in that file. Upstream sources are `pytorch/executorch` repo-relative paths, read on 2026-10-28; the two papers were read through WebFetch summaries, so their figures are UNVERIFIED unless stated. Every number is cited (file and line, or paper section) or **derived** (arithmetic in §1). The derivation script is in the round-27 scratch area and is not committed. **UNVERIFIED** marks anything not measured or not checked.

## §0 Verdict

**Gate 0, before any card is scheduled.** The streamed-group cost of 429–530 µs per expert (doc 01 `:2069`, §8x) is not decomposed anywhere in the docs. Its known stages sum to **220–237 µs** (derived: fill 87 + DMA 66 + GEMM 67–84, §1), which leaves **192–310 µs unattributed**. The merged path counts fill and wait (`pf_stream_fill_s`, `pf_stream_wait_s`, returned by `merged_stream_detail`, `crates/breccia-engine/src/graph.rs` L2960–2966, traced under `BRECCIA_TRACE_MERGED`, L2883). It has no DMA-time or GEMM-time counter in that path (UNVERIFIED that none exists elsewhere), and `pf_distinct` (L2881) gives distinct experts but not rows per expert. Gate 0 therefore needs three things from one run:

1. DMA and GEMM timers added to the trace, so that the stages can be summed.
2. A per-card split of the streamed-group time. The §8x run is labelled "dual" (doc 01 `:2055`), and the MI50 stream is copy-bound (§1).
3. A routing histogram (distinct experts per block, rows per expert) from the same run.

Accept gate 0 when the named stages (fill, DMA, GEMM, wait) cover at least 80% of streamed-group wall time on each card, and when storage reads during the timed pass are near zero (page-cache resident, measured by a counter the run records; UNVERIFIED which counter). If the fill is storage-bound, card 1 is void and disk-side prefetch becomes the lever (owner decision a).

**Ranked cards.**

1. **Card 1 — RX streamed-expert overlap.** Move the per-expert copy and CPU memcpy onto a transfer path with per-slot fences, replacing the full flush every 8 experts (`graph.rs` L2727, `expert_index.is_multiple_of(STREAM_SLOTS)`; `STREAM_SLOTS = 8` at L271). Derived ceiling: a streamed group falls from 429–530 µs to **108 µs** when the three host-DRAM passes per expert (5.24 MB) are charged to the 48.4 GB/s shared budget (doc 24 `:123`). Accept on the §8x configuration (2 agents, 300-token prefix, merged on), with gate 0 passed:
   - RX-stream streamed-group p50 ≤ 110 µs (about 1.3× the 84 µs resident top; derived bound 108 µs);
   - `pf_stream_wait_s` ≤ 5% of streamed-pass time;
   - RX-stream tok/s ≥ the same run's RX-stream token-by-token tok/s (the per-stream split of the bench output is UNVERIFIED);
   - per-agent tok/s ≥ 6.45 with the MI50 gate (card 3) applied. 6.45 is the same-bench token-by-token figure (doc 01 `:2059`), and the §8x merged run measured 6.07 (`:2060`), so an overlap that does not beat token-by-token fails;
   - bit-identical to the serialised merged build at the same chunk size (128 for the §8x case; same bytes, same GEMM order, no math change), 32/32 greedy, and the same top-1 logit as the serialised build on the same commit;
   - three consecutive parity runs with the copy-fence trace, since the copy-fence race is on record (doc 01 §8x item 3, `:2073–2084`).

   Sizing (derived, §1): at uniform T = 150, serialised streaming costs 54–67 ms per row against a lane of 38–48 ms (quiet readings). Overlapped it costs 13.6 ms per row. At the one measured skew point (about 130 distinct experts per block at T = 128, doc 02 `:470–471`), serialised streaming costs 17–21 ms and overlapped 4.3 ms. The skew case would put serialised streaming below the lane, which §8x does not show (§4 item 12). **UNVERIFIED:** the realised ratio, since the docs contain no measured fill/wait split and no routing histogram for the §8x run.

2. **Card 2 — zero-copy host-mapped expert read.** Let the GEMM read mapped expert blobs directly through a host-visible import (BRIEF `:128`, rule 7; BRIEF `:285`). This removes the fill memcpy (87 µs) and two of the three DRAM passes. Derived ceiling: max(PCIe read 1.746 MB ÷ 25 GB/s = 70 µs, GEMM 67–84 µs) = **70–84 µs per group**, below card 1's ceiling. Confidence is lower. Doc 09 `:27–30` records a negative BAR result, and its A1 "YES, at 22–26 GB/s" (`:85–88`) is a projection, not a measurement. The import granularity is one import per streamed block, 18 blocks (doc 22 `:10`; BRIEF `:79`). Doc 22 `:10` calls each block 0.89 GiB, but 512 × 1,746,133 B = 894 MB decimal = 0.833 GiB, so the 0.89 figure matches decimal GB and the unit label is the slip (§4 item 8; UNVERIFIED which unit was meant). Accept: GPU read rate from the mapped blobs ≥ 22 GB/s by trace; streamed group p50 ≤ 105 µs (about 1.25× resident); bit-identical to card 1 (same bytes); `pf_stream_fill_s` → 0. Re-test doc 09's negative before scheduling (owner decision d).

3. **Card 3 — MI50 merged-streaming gate.** The single-device guard does not block the production bench path. `bench token --streams 2` and `bench agents` build `Multi` with one device per stream (`crates/breccia-cli/src/main.rs` L1718 and L2035–2036; device assignment `devs[s % devs.len()]` with `devices: vec![dev.clone()]`, L2001–2006). `Multi::new` requires exactly one device per stream (`crates/breccia-engine/src/multi.rs` L140–145), and doc 01 gives one card per stream as the design (`:1056–1067`). The guards in `graph.rs` (chunk slots L686–691, chunk L778–783, `prefill_target` and `prefill_merged` L1044–1053) apply only to a `Model` with more than one device, i.e. a layer-split session (doc 01 §5.3, `:217`). The live issue is the MI50 stream. The prefill scratch gate (`graph.rs` L855–858) checks only `chunk > 0`, `cfg.prefill_merged` and `!model.placed.is_empty()`, so the MI50 stream streams non-resident experts at 602 µs of copy per expert whenever both flags are set. The defaults are `prefill_chunk = 0` (`main.rs` L234, L349) and `prefill_merged = false` (L239, L352), so merged streaming is opt-in. Accept: at chunk 128 (T = 150) the MI50 stream streams no non-resident expert merged under the serialised path (T* = 285–357 > 150); under overlap, only experts with at least 5–7 routed rows (s* = 4.9–6.1); per-agent tok/s on the dual config ≥ the same-run token-by-token figure (`:2059`); 32/32 greedy against the same build with the gate off (bit-identity not assumed, since the host lane and the merged GEMM may round differently; UNVERIFIED).

**Which card takes which blocks (derived; owner decision f).** RX takes streamed prefill for experts with at least 2 routed rows under overlap (s* = 0.88–1.10). MI50 keeps its resident blocks for prefill and leaves non-resident experts on the host lane, unless an expert has at least 5–7 routed rows under overlap or the chunk reaches T* = 285–357 under serialisation. Experts that stay on the lane are priced at 98.5–123 µs per use (§1).

## §1 Derivation: chunk size vs PP, per card

**Inputs.**
- Expert blob 1,746,133 B = 1.6652 MiB (doc 24 `:30`). 48 blocks × 512 experts, 10 active per token (doc 02 `:14–16`). 39 streamed blocks per doc 01 §8ag (BRIEF `:79` has 18; §4 item 5).
- Lane use L, host CPU lane, per block-expert use. Three RX `experts(host)` readings:
  - **38.40 ms per token** at the shipped 9-placed default, 2 streams, 64-token chunks (doc 01 `:2681`, §8ag sweep). Over 390 uses (39 streamed blocks × 10), **L = 98.5 µs**.
  - **48.1 ms per token** (doc 01 `:2669`, §8af window). Chunk and stream settings are not stated on that line (UNVERIFIED). **L = 123 µs**.
  - **83.04 ms per agent-token** (doc 01 `:2734`, §8ah stage table). Doc 01 `:2787` attributes it to a loaded box ("83 vs 38-48 in quieter windows"). Used only in the loaded reference rows (L = 213 µs). Per-stream versus combined is UNVERIFIED.
  - §8ag's closing window (doc 01 `:2712–2714`) is called contended. Whether the 38.40 sweep ran in a contended window is not stated, so 98.5 µs is a best reading, not a floor.
  - The sweep script `sweep.sh` (cited at doc 01 `:1211`) is not in the repo, so its per-token denominator is UNVERIFIED.
- Copy: RX 26.5 GB/s (BRIEF `:125`, rule 6), so one blob is 65.9 µs (doc 24 `:32`). MI50 2.9 GB/s (doc 02 baseline table; gen3 x4 link at BRIEF `:124`), so one blob is **602 µs** (derived: 1.746 MB ÷ 2.9 GB/s).
- Fill (host memcpy out of the mapped file): about 20 GB/s (doc 09 `:87`, "799 MiB at ~20 GB/s"), so one blob is **87 µs** (derived).
- GEMM, resident group 67–84 µs (doc 01 `:2069–2070`, §8v). The MI50 GEMM time is not measured. The RX range is used as a proxy (UNVERIFIED). The MI50 copy-bound group is 602 + 84 = **686 µs** (derived).
- Measured serialised streamed group 429–530 µs (doc 01 `:2069`, §8x). The card is not named (blended caveat below).
- Host DRAM 48.4 GB/s combined (doc 24 `:123`, derived there).

**Distinct experts per pass D(T).** Uniform routing: D(T) = 512·(1 − (1 − 10/512)^T) (doc 02 `:73`). Derived: D(1) = 10, D(32) = 239.6, D(64) = 367.1, D(128) = 471.0, D(150) = 485.4, D(256) = 508.7, D(1200) = 512. Measured skew: about 130 of 512 experts per block for a 128-token chunk (doc 02 `:470–473`, one point, UNVERIFIED beyond it). Doc 02's "2.7× if uniform" is not reproduced by this formula (§4 item 3).

**Rows per pass T.** The planner gives `prefill_cap(target) = target + target/2`, and `prefill_plan` splits a prompt of `len` rows evenly into `ceil(len/cap)` chunks (`graph.rs` L4101 and L4117). A target of 128 gives a cap of 192, so a 300-token prefix runs as 2 × 150 rows. Doc 01 documents the §8x merged row as `chunk 128 + merged (39 blocks streamed)` (`:2060`), so T = 150 is the documented configuration. The merged path needs `chunk > 0` and `prefill_merged` (`graph.rs` L855–858, L2316), so chunk 0 would switch it off. Rows belong to one session (`prefill(&mut self, toks)`, `graph.rs` L2102). The code has no cross-session merge (UNVERIFIED for the bench harness).

Sanity check at T = 1 (derived, with an UNVERIFIED group cost). Token-by-token 6.45 tok/s is 155 ms per token. At T = 1, the streamed path would replace the 38–48 ms lane with 390 × 429 µs = 167 ms, giving 274–284 ms per token, or 3.5–3.6 tok/s. A 1-row group has no 128-row GEMM, though, so its cost may sit nearer fill plus DMA (153 µs). At that cost the T = 1 prediction is 5.7–6.0 tok/s, close to 6.07. The arithmetic alone therefore does not exclude T = 1. The documented chunk-128 row (`:2060`) settles it.

**Blended-group caveat (UNVERIFIED).** §8x does not say which card produced the 429–530 µs group figure. If it is an equal-weight mean of an RX group and the copy-bound MI50 group (686 µs), the RX-only figure is 2 × (429–530) − 686 = **172–374 µs** (derived). That range is a second RX row in the break-even table.

**Serialised break-even.** Streaming wins when D(T) · g ≤ 10 · T · L, that is D(T)/T ≤ 10L/g. Solved numerically on the uniform formula. Each value was re-checked by hand against the formula. The MI50 rows reuse the RX lane readings as the MI50 host lane (UNVERIFIED; the MI50 stream's own lane is not measured, §4 item 12(c)).

| case | L (µs) | g (µs) | 10L/g | T* (uniform) |
|---|---:|---:|---:|---:|
| RX, quiet, 38.40 ms | 98.5 | 429 | 2.30 | 221 |
| RX, quiet, 38.40 ms | 98.5 | 530 | 1.86 | 275 |
| RX, quiet, 48.1 ms | 123 | 429 | 2.87 | 173 |
| RX, quiet, 48.1 ms | 123 | 530 | 2.32 | 218 |
| RX, blended group (caveat) | 98.5 | 172 | 5.73 | 65 |
| RX, blended group (caveat) | 98.5 | 374 | 2.63 | 190 |
| MI50, copy-bound (g = 686) | 98.5 | 686 | 1.44 | 357 |
| MI50, copy-bound (g = 686) | 123 | 686 | 1.79 | 285 |
| loaded box, RX (reference, excluded) | 213 | 429 / 530 | 4.97 / 4.02 | 84 / 114 |
| loaded box, MI50 (reference, excluded) | 213 | 686 | 3.10 | 158 |

The quiet RX readings put serialised T* at **173–275** tokens under uniform routing. The blended-group caveat would move RX to 65–190. MI50 is **285–357**. The loaded-box readings (84–114) are excluded from sizing. The measured skew point (about 130 distinct at T = 128) would put RX serialised streaming below the lane at that chunk size, which §8x does not show (§4 item 12). Break-even therefore depends on routing that the docs do not measure.

**Overlapped break-even.** With copy and GEMM overlapped, the streamed group costs g_ov, and streaming wins for an expert with s routed rows when s ≥ s* = g_ov/L.
- RX, g_ov = 108 µs (DRAM bound): s* = **1.10** (L = 98.5) or **0.88** (L = 123). Any expert with at least 2 routed rows streams.
- RX, g_ov = 87 µs (fill-bound, no DRAM contention): s* = 0.88 (L = 98.5) or 0.71 (L = 123).
- MI50, g_ov = 602 µs (copy-bound): s* = **6.1** (L = 98.5) or **4.9** (L = 123), that is at least 7 or at least 5 routed rows.

**Per-row cost (ms per token), T = 150 uniform and T = 128 measured skew.**
- Lane: 39 × 10 × L = **38.4** (L = 98.5) or **48.1** (L = 123). The loaded-box reference is 83.1. Independent of T.
- Streamed, serialised: 126.2 groups per row uniform → 54.1–66.9 ms; 39.6 groups per row skew → 17.0–21.0 ms.
- Streamed, overlapped at 108 µs: 13.6 ms (uniform); 4.3 ms (skew).
- Zero-copy ceiling at 70 µs: 8.8 ms (uniform).

**Calibration against §8x: not used.** An earlier draft calibrated lane time against §8x by subtracting stage-table non-expert time from the per-agent figure. The two use different conventions, so no effective lane or group cost is derived from them. §8x supports only the sign: its merged path is slower than token-by-token (6.07 against 6.45 tok/s). Uniform routing with serialised groups predicts a slower path (54–67 ms per row against a 38–48 ms lane), and the measured skew predicts a faster one (17–21 ms). The routing of the §8x run is not in the docs, so the sign is the test gate 0 must settle.

**Chunk-size model per card (summary).** RX, serialised: streaming beats the lane above T* = 173–275 under uniform routing; under the measured skew point it would beat the lane at about T = 128, which §8x does not show. RX, overlapped: streaming beats the lane for any expert with at least 2 routed rows. MI50, serialised: above T* = 285–357; overlapped: at 5–7 routed rows per expert. Beyond T = 256, D(T) is nearly saturated (D(256) = 508.7 of 512), so bytes per token fall only as 1/T.

## §2 Technique table

| Technique | Source | What transfers to Breccia | Verdict |
|---|---|---|---|
| Overlapped copy and compute on separate queues (CGOPipe three-stage schedule) | arXiv 2411.11217 §4.1, Alg. 1 (WebFetch summary, UNVERIFIED) | Principle for card 1: copy and compute on separate queues. The paper's schedule is decode-only, with attention on the CPU | Principle adopted; schedule not copied |
| Paged weights with a 2 × W_L GPU double buffer | arXiv 2411.11217 §4.1, App. A.1, Fig. 11 (UNVERIFIED) | Breccia has `STREAM_SLOTS = 8` slots of 1.6652 MiB, 13.3 MiB per block (`graph.rs` L271; derived). Doc 02 `:106` plans a 32-slot ring of 53 MiB per card; the code constant is 8, not 32 | Redundant with the slot ring. The 32-slot plan is not in code |
| HRM roofline, turning point P1 (eq. 9) | arXiv 2411.11217 §3 (summariser's transcription, UNVERIFIED) | P1 is the same threshold as Fiddler's s* (§3.3). One rule, presented once | Adopt the threshold, not the model |
| Policy latency T = max(comm, cpu, gpu) (eq. 12) | arXiv 2411.11217 §3 (UNVERIFIED) | Per-layer decode latency, not prefill | Not used |
| Move-or-compute per expert: compute on CPU if cpu_lat(s) ≤ gpu_lat + trans_lat (Alg. 1) | arXiv 2402.07033 §3.3 (WebFetch summary, UNVERIFIED) | s = routed rows per expert. The same threshold as s* in §1 | Adopt the threshold for prefill. Split routing needs the 32/32 contract, not bit-identity. Decode case already rejected (doc 24 `:139`) |
| Symmetric per-channel int8 weight-only linear, `_SUPPORTED_OPS = ["linear"]` | `pytorch/executorch` `backends/vulkan/quantizer/vulkan_quantizer.py` (read) | Uniform int8 per channel does not match IQ3_XXS codebook blocks | Rejected |
| Int4 weight packing, transposed and interleaved at load | `backends/vulkan/runtime/graph/ops/glsl/pack_weights/pack_int4_linear_weight_transposed_interleaved.glsl` (name only; body not read) | A load-time layout for plain int4 weights. Breccia's experts are IQ3_XXS | Rejected |
| Batched linear GEMM tile | `backends/vulkan/runtime/graph/ops/glsl/gemm/linear/linear_vec.glsl` (read; floating-point weight path) | Breccia's merged path already batches rows per expert (`graph.rs` L2566, `prefill_block_merged`). Shader-level M handling not compared (UNVERIFIED) | Not transferable as code; no card |
| Dynamic per-tensor int8 activation quantisation | `vulkan_quantizer.py`; `gemm/linear/qw/dq8ca_q4gsw` (name only) | BRIEF `:188` records 6.2e-3 per element for dp4a activation quantisation, which fails parity | Rejected |
| Cooperative-matrix linear | `backends/vulkan/runtime/graph/ops/impl/gemm/linear/LinearCoopmat.cpp` (name only) | `VK_KHR_cooperative_matrix` is absent on both cards (doc 22 `:13`, Q6 at `:58`) | Rejected, settled |
| MoE-Lightning full CGOPipe pipeline | arXiv 2411.11217 §4.1, §5 (UNVERIFIED) | Decode-only; attention offloaded to CPU; the paper gives no prefill schedule | Rejected for prefill |
| Fiddler expert-in-place for decode | doc 24 `:139` | Decode has no copy path | Already rejected |
| MLX sort by expert at ≥ 64 indices | doc 19 `:7` | Breccia groups prefill rows by expert (`graph.rs` L2566, L2674–2676) | Already have |
| Zero-copy host-mapped experts | BRIEF `:128`, `:285`; doc 09 `:27–30` (negative), `:85–88` (A1 projection) | Card 2 | Conditional, see §0 |
| Chunks above 256 rows | derived §1 (D(T) saturates at 512) | Little left per token above T = 256 | Rejected |

## §3 Rejected list

- ET-VK int8 per-channel and dynamic per-tensor activation schemes: they do not match IQ3_XXS codebooks, and activation quantisation fails parity (BRIEF `:188`).
- ET-VK int4 nibble packing: layout only, with no codebook. IQ3_XXS is codebook-coded (general GGUF format knowledge, not cited from the repo).
- Cooperative-matrix linear: absent on both cards (doc 22 `:13`, Q6 at `:58`).
- MoE-Lightning CGOPipe as a prefill schedule: decode-only, attention on CPU, no prefill schedule in the paper (UNVERIFIED, WebFetch).
- MoE-Lightning paged weights: redundant with the slot ring (`graph.rs` L271, `STREAM_SLOTS = 8`).
- MoE-Lightning "up to 10.3×" as a target: decode-only. Table 4, as summarised, gives about 4.7× and 8.6× against the stated 3.5× and 6.7×, not 10.3× (UNVERIFIED, WebFetch).
- Fiddler expert-in-place for decode: already rejected (doc 24 `:139`).
- Streaming MI50 experts below the break-even (card 3): 602 µs of copy per expert (§1).
- Lifting the single-device guard for the multi-stream path: not needed, since `Multi` already gives each stream one device (`main.rs` L1718, L2035–2036; `multi.rs` L140–145). Lifting it for layer-split sessions is a separate owner decision (§5b, default no).
- Chunks above 256 rows: D(256) = 508.7 of 512 by uniform routing, so the expert-count gain is saturated.
- Layer-pipeline split across cards for prefill: MI50 activation shipping dominates (BRIEF `:192`).

## §4 Conflicts with existing docs (file:line)

1. Doc 02 `:78`: the T = 32 row shows 232 distinct → 302 MiB per token. Derived from the same row's inputs: 232 × 1.6652 × 48 ÷ 32 = **579.5 MiB**, or 471 MiB with 39 streamed blocks. The printed 302 cannot be reproduced. Recorded as found, not corrected.
2. Doc 02 `:41`: the rank-1 "×6–14 (3.1 → 20–45) est." predates §8x. §8x measured merged at 6.07 against token-by-token 6.45 (doc 01 `:2059–2060`), so the projection is not met on the serialised path.
3. Doc 02 `:470–473`: "2.7× if routing were uniform" is not reproduced. The §1 formula gives 471 distinct experts at T = 128 against about 130 measured, a 3.6× ratio. Uniform bytes at T = 128 with 39 streamed blocks are 471 × 1.6652 × 39 ÷ 128 = 239 MiB per token (derived), 3.3× the 72.3 MiB per token measured after §8u (doc 02 `:473`). The basis is "41.9 GiB ÷ 600" at doc 01 `:1596–1600`, which gives 71.5 MiB by arithmetic; the printed 72.3 implies a divisor of about 593 (UNVERIFIED rounding). UNRESOLVED: which routing the 72.3 budget was measured on.
4. Lane readings: doc 01 `:2681` (38.40, §8ag sweep, 2 streams, 64-token chunks), `:2669` (48.1, §8af window, settings not stated), `:2734` (83.04, §8ah stage table; per-stream convention per `:2729–2745`, UNVERIFIED). `:2787` attributes the 83 to load. The sweep denominator is not in the repo (`sweep.sh`, `:1211`). Sizing uses the quiet readings only (§1).
5. Streamed-block count: BRIEF `:79` has 18 streamed (301 MiB per token, 30 resident); doc 01 §8ag has 39 streamed (9 resident); doc 02 `:15` uses a third baseline (3.14 tok/s per agent, 4 agents × 300, 799 MiB per token). Owner decision c.
6. Parity reference: BRIEF `:77` gives top-1 13.170510. Doc 01 `:1894–1902` gives 13.170521 (chunk 0), 13.170561 (chunks 8–128) and 13.170429 (128 + merged, 9 resident). Four values for one reference prompt, from different builds. The gate must be bit-identity against the current merged build, not an absolute value.
7. Single-device guard and the production bench path. `graph.rs` L686–691, L778–783 and L1044–1053 set chunk and `prefill_target` to 0 when `devices.len() != 1`. The bench harnesses build `Multi` with one device per stream (`main.rs` L1718, L2035–2036), and `Multi::new` requires exactly one device per stream (`multi.rs` L140–145). Each stream therefore passes the guard, so the merged streamed path does run on a card-per-stream bench. The §8x run is labelled "dual" (doc 01 `:2055`), which implies one stream per card. That is UNVERIFIED, since the bench flags and device mapping of that run are not recorded, and it means the 6.07 mean combines an RX stream with a copy-bound MI50 stream. The layer-split session is the case the guard blocks (doc 01 §5.3, `:217`). The serve function (`main.rs` L883) prints a device plan, but no direct `Model::load` call appears in L883–1686, so its load path is not traced (§6). Earlier drafts called the guard a production blocker. For the bench path it is not.
8. Zero-copy evidence and units. Doc 09 `:27–30` (BAR negative) against `:85–88` (A1 "YES, at 22–26 GB/s", a projection) and BRIEF `:251` ("work at 22–26 GB/s"). Not reconciled. Unit label: doc 22 `:10` ("each about 0.89 GiB") and doc 02 `:106` ("0.9 GiB/block") match 894 MB decimal for 512 experts, while 512 × 1.6652 MiB = 0.833 GiB. Which unit the docs meant is UNVERIFIED. The difference is a label, not a gap.
9. Lane use: doc 24 `:32` gives 75.0 µs per use (decode-era, 13.5 ms ÷ 180). The prefill derivations here give 98.5–123 µs from the quiet readings. The 213 µs loaded-box reading is not used for sizing (§1).
10. Baselines: doc 02 `:15` (3.14 tok/s per agent, 4 agents × 300), doc 01 §8x (6.45, 2 agents × 300) and BRIEF `:80` (6.0–6.45, chunk 64). These are different configurations, not one baseline.
11. Counter naming: the §8x "bytes" (0.15 GiB merged, 4.30 GiB token-by-token, doc 01 `:2029–2034`) are not named in the doc. Per-token columns divide by about 600 (2 agents × 300 prefix tokens; the convention at doc 01 `:1596–1600`; UNVERIFIED for this table). The quantity is not `pf_stream_bytes` (`graph.rs` L2960–2966, bytes copied out of the mapping). Its magnitude suggests it counts storage misses: 7.3 MiB per token against 649 MiB per token addressed (39 blocks × 10 uses × 1.6652 MiB, derived), about 1%. UNVERIFIED definition.
12. Measured skew against the §8x sign. §1 derives serialised streaming below the lane at the measured 130 distinct experts (17–21 ms per row against 38–48 ms). §8x measures the merged path slower than token-by-token (6.07 against 6.45, doc 01 `:2059–2060`). Doc 02 `:470–471` is the only routing measurement in the docs. Three candidates: (a) the routing of the §8x run is not that skew; (b) the group cost at chunk 128 is above 530 µs; (c) the 6.07 mean dilutes the RX gain with the copy-bound MI50 stream (item 7). At 686 µs per group the MI50 stream costs 27 ms per row at the measured skew (39.6 × 686) and 87 ms at uniform routing (126.2 × 686, derived); its lane is not measured (UNVERIFIED). Not reconciled. Gate 0 records the routing histogram, the group cost and the per-card split, which separate the three.

## §5 Owner decisions

a. Authorise gate 0: a trace run on the §8x configuration with DMA and GEMM timers added to the trace, a per-card split, and a routing histogram from the same run (§4 items 7 and 12). Card 1 is not scheduled until it reports.
b. Whether to lift the single-device guard for layer-split sessions (`graph.rs` L686–691, L778–783, L1044–1053). Not needed for card 3 (§4 item 7). Default: no.
c. The canonical PP baseline and streamed-block count: 18 (BRIEF `:79`, 30 resident) or 39 (doc 01 §8ag, 9 resident) (§4 item 5).
d. Re-test the doc 09 zero-copy negative (`:27–30`) before card 2 is scheduled.
e. Whether RADV on the RX exposes a separate transfer queue for a copy ring. **UNVERIFIED**, not checked here.
f. Whether the MI50 may ever take merged streaming, and at what chunk size (card 3 gate).
g. The §8x routing histogram at chunk 128, and the per-card split of the 429–530 µs group figure and of the 6.07 mean. The answer decides whether card 1 is sized on uniform or measured-skew routing (§4 item 12).

## §6 Not checked

- Gate 0 itself: no GPU run was permitted. The DMA and GEMM timers, the per-card split, the routing histogram, the page-cache state during the §8x run, and the bench command line (flags, T, session count) are not in the docs or the trace.
- Whether the §8x figures of 6.45 and 6.07 count prefill tokens or the whole run (300-token prefix plus 8 generated tokens). The table header (doc 01 `:2055`) does not say.
- The sweep script (`sweep.sh`, doc 01 `:1211`) is not in the repo. Its per-token denominator, and whether the §8ag sweep was per stream or combined, were not checked.
- T = 150 is derived from the planner rules (`graph.rs` L4101 and L4117) and the documented chunk 128 (doc 01 `:2060`), not from a run log.
- Serve path (`crates/breccia-cli/src/main.rs` L883): no direct `Model::load` call appears in L883–1686, so whether the server runs a layer-split session or per-card streams is UNVERIFIED. The `Model::load` call sites are at L1687 (token path, before `Multi::new` at L1718), L2003 (agent streams, L2001–2006) and L2458 (single session); which subcommand serves L2458 is UNVERIFIED.
- MI50 scratch gate (`graph.rs` L855–858): the code was read, not run. Whether the MI50 stream merges in the bench runs is UNVERIFIED.
- ET-VK: `linear_vec.glsl` was read (floating-point path only). `Linear.cpp`, `LinearCoopmat.cpp`, the `gemm/linear/qw/` kernels and the `impl/gemm/` tree were listed by name, not read. The M = 1 versus M > 1 dispatch rule is UNVERIFIED. The non-gemm `runtime/` directories, `patterns/` and the Vulkan overview were not read. The quantiser's int4 behaviour is taken from its name and the file header only.
- MoE-Lightning and Fiddler: WebFetch summaries. The figure timelines (MoE-Lightning Fig. 6), Fig. 3 numeric p_g and b_g values, the HRM equation forms (summariser's transcription) and Fiddler's prefill latency figures (1.07–1.65×, §4.2) were not checked against the PDFs.
- MI50 GEMM group time: not measured. The RX 84 µs is a proxy.
- Host DRAM: 48.4 GB/s is doc 24's combined figure. The 108 µs bound assumes that memcpy and DMA traffic both count against it. UNVERIFIED.
- Real routing skew: one measured point (about 130 distinct at T = 128), no histogram.
- Doc 09's A1 projection (22–26 GB/s) was not re-tested.
- Cooperative matrix on the two cards is settled by the existing probe (doc 22), not re-probed.
- Doc 01 line numbers come from reads and may drift by a few lines.
- No measurements were taken. Every figure is cited or derived; none is a new measurement.
