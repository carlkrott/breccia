# 32 — Heterogeneous GPU balance and determinism

Round 32. Research only: no code, no GPU runs. Every speed estimate carries the tag `UNVERIFIED - requires the GPU qualification card`. Per-card facts are "measured, round C3". Other measured figures cite their doc or BRIEF line.

Baseline conflict, not reconciled here: aggregate 2-stream decode is 13.2–14.6 tok/s in docs 04 and 28, 20.0–20.8 in doc 02 §8p and doc 30 §8y, and 22.8–23.7 in BRIEF:74. Every candidate below is compared to the 20.2 tok/s row that doc 30 measured on the same model, not to the BRIEF figure.

## 1. What the two-stream step is made of

**Placement.** The CLI gives stream `s` device `devs[s % devs.len()]` (`crates/breccia-cli/src/main.rs:1686`) and loads each stream's `Model` with one device (`main.rs:1683-1697`). `Multi::new` (`crates/breccia-engine/src/multi.rs:135`) takes one device per model. Two streams are therefore two independent models, one per card. No layer or expert crosses a card.

**Step order.** `token_all` (`multi.rs:178-218`) runs block `b` for each stream in turn. `finish_block` runs the host expert lane synchronously (`crates/breccia-engine/src/graph.rs:3134`). The lane is one CPU resource: a single `Arc<Mutex<ExpertPool>>` (`graph.rs:442`, built or shared at `:994-1000`), and the driver is single-threaded (doc 30 line 33). The lane pool's thread count is `default_threads()`, the logical CPU count clamped to 32 (`graph.rs:4155`, used at `:989-990`), which is 32 on the 16-core, two-SMT host in doc 31 line 33; `Multi::new` shares the first session's pool (`crates/breccia-engine/src/multi.rs:155-156`). The comments that say 16 threads (`graph.rs:439-441`, `multi.rs:25`) do not match this code. The 2-stream CLI path is `bench_token_multi` (`crates/breccia-cli/src/main.rs:1636`).

**Pair-step model.** Doc 30 fits a pair step `S = L + F + R` (lane + fence + residual), with aggregate `2000 / S` tok/s:

| Source | L (lane) | F (fence) | R | S | Aggregate |
|---|---|---|---|---|---|
| doc 30 §8y, 32 threads, run 1 (table line 27) | 41.2 | 39.5 | 18.4 | 99.1 | 20.2 |
| doc 30 §8y, 32 threads, run 2 (line 28) | 41.2 | 36.6 | 18.5 | 96.3 | 20.8 |
| doc 30 §8y, 16 threads (line 26) | 55.2 | 44.7 | 17.8 | 117.7 | 17.0 |
| doc 02 §8p, 2 streams (`docs/phase-4/02-untried-levers.md:12`) | 39 | 48 | ~21 | 108 | 20.0 |

The three rows agree within a few ms. The additive form is a fit, not a proof: doc 28 card 1 names the timer attribution that could make it look additive (`docs/research/28-iree-async-overlap.md`).

**Residency basis.** The 2-stream CLI default gives each stream 9 GPU-resident blocks, that is 39 host blocks (`crates/breccia-cli/src/main.rs:1659-1664`; doc 30 line 53). The BRIEF's "30 of 48 resident, 26.7 GiB" (BRIEF:79) is the single-stream dual-card configuration: BRIEF:73 labels it "single stream, both cards", and BRIEF:159 gives lane 47.5 ms at 9 resident and 13.5 ms at 30 resident. A 26.7 GiB arena cannot sit on one 16 GiB card, so the 30-block figure cannot be one 2-stream model's arena.

**Lane cost per block-token** (derived from the pair model, 39 + 39 host blocks = 78 block-tokens; `UNVERIFIED - requires the GPU qualification card`):
- 41.2 ms / 78 = 0.53 ms at 32 threads (the default here).
- 55.2 ms / 78 = 0.71 ms at 16 threads (doc 30 line 26).
- 39 ms / 78 = 0.50 ms (doc 02 §8p, same residency).
- 85.7 ms / 78 = 1.10 ms at 8 threads (doc 30 §8y).

**Expert bytes.** A routed expert is about 1.75 MB and each token routes 10 per block (doc 30 line 70), so one block-token moves 17.5 MB (cross-check: doc 31 table, 314 MB per stream-token at 18 blocks).

**Card facts (measured, round C3).** RX 6950 XT gfx1030, 16 GiB, 28.7 GB/s host link, full DMA/compute overlap, DRAM-bound GEMV. MI50 gfx906, 16 GiB HBM, 2.9 GB/s gen3 x4, about 0 overlap, ALU-bound GEMV. CPU host lane AVX2, about 30 GiB host budget. Two streams with `BRECCIA_GPU_TIME=1`: MI50 busy 33%, RX busy 55–64%. At `S` of 96–99 ms, RX GPU time is about 53–62 ms per token and MI50 about 32 ms. The link is saturated by one RX stream at 25–26.5 GB/s, and the MI50 is never used for streaming (BRIEF:125-127).

## 2. Q1 — how heterogeneous systems split work, and what maps to Breccia

### 2.1 The reference designs

Not re-read this round. Each is cited by project name and mechanism; section pins are open (see §6).

- **Helix, HexGen, ThunderServe.** Partition layers or requests across unequal GPUs, sized to measured per-device throughput and memory, and charge the link cost of activation exchange. Each needs cross-device exchange per layer or per request. Breccia avoids the exchange by never splitting a layer across cards (BRIEF:192 rejects the layer-pipeline split: activation shipping at 2.9 GB/s).
- **Mélange.** Cost-aware choice of GPU types for a fleet. Fleet-scale; does not apply to two cards in one process.
- **Cephalo.** Memory-balanced sharding of unequal GPUs for training, with per-layer collectives. Not applicable to decode.
- **llama.cpp `--split-mode` and `--tensor-split`.** Layer split and row split. Covered by README rows 12 and 18; not re-audited.
- **exllama device map.** Layer split. Same rejection as above.
- **KTransformers device map.** Expert on host, attention on GPU. Covered by row 20 and doc 14; one-layer-late deferral fails bit parity there.

What transfers is the proportional-placement idea: put work where the measured capacity is, and count the link cost. Breccia's only cross-card-free levers are expert placement and stream count. Those are the candidates below.

### 2.2 Ranked candidates

Aggregate figures are against the 20.2 tok/s baseline (`S` = 99.1 ms). Every speed estimate is `UNVERIFIED - requires the GPU qualification card`.

**R1. Held, not ranked: per-stream host lanes on a split pool (overlap the lane with the own-stream fence).** The one-thread overlap already exists: a stream's lane runs while the other stream's GPU work is in flight (`covered`, `crates/breccia-engine/src/multi.rs:196`). What remains is each stream's own fence wait, which comes before that stream's lane (`crates/breccia-engine/src/multi.rs:178-218`). On one thread no lane work is ready during that wait, so hiding it needs a second host thread. Under the additive model, `S = max(L, F) + R = 59.6` ms, or 33.6 tok/s. With the RX busy floor, `S >= max(L, RX busy) + R` gives 71–80 ms, or 25–28 tok/s from 20.2 at 32 threads, and 25–27 tok/s from 17.0 at 16 threads. Doc 28 rejects a second host lane thread unless the pool is split, and the split is not checked (doc 28 line 105). Doc 30 line 144 rejects two 32-thread pools. R1 stays out of the ranking until a 16+16 split is measured and doc 28 card 1 shows the fence is real idle time. `UNVERIFIED - requires the GPU qualification card`.

**R2. Unequal stream counts: a third stream on the RX.** Two RX streams need 2 × (53–62) = 106–124 ms of RX time per step, so 3 tokens per step give `3000 / 106–124` = 24–28 tok/s. The lane grows to 3 × 20.6 = 62 ms, which fits under the RX floor. Cost: a second RX arena. `Session::new` borrows `&Model` (`crates/breccia-engine/src/graph.rs:636`), so sharing one model between two sessions is not excluded by the signature, but the CLI builds one model per stream (`main.rs:1683-1697`) and whether `Multi` accepts two sessions over one model is untested (`multi.rs:135`). Without sharing, the RX resident set shrinks and the lane grows; net unknown. A measured aggregate plateau set by the GPU (24.74 tok/s at 4 streams, 25.03 at 6; doc 13 line 144, phase-4 §8p) is a reason to doubt the 3-stream figure. Gate: the shared-model check plus the RX auto-rule budget (`crates/breccia-engine/src/model.rs:844-866`). `UNVERIFIED - requires the GPU qualification card`.

**R3. Asymmetric residency (more resident blocks on the MI50 stream, which has about 67% slack).** Each block moved off the lane saves `41.2 / 78 = 0.53` ms per pair step. Four blocks: `S` 99.1 → 97.0, 20.6 tok/s, +2%. Eight blocks: 94.9, 21.1 tok/s, +4%. Removing the whole lane (`S` 57.9, 34.5 tok/s) is an unreachable ceiling. MI50 GPU time per moved block is unmeasured, and VRAM headroom is set by the auto rule. `UNVERIFIED - requires the GPU qualification card`.

**R4. Type-aware placement (doc 12).** Doc 12 puts ALU-heavy codebook types on RX (rate ≈ 127 000 / I_unit on RX vs 62 400 / I_unit on MI50, line 119, and codebook types are ALU-bound on MI50 by 4–5×) and bandwidth-bound types on MI50. Load balance points the other way: MI50 is the less busy card, so it should take work. The two rules conflict, and the gain is unmeasured. Needs per-block, per-card GPU timers. `UNVERIFIED - requires the GPU qualification card`.

**R5. RX stream-in of its host-lane experts (decode copy path).** Copy cost per block-token is 17.5 MB ÷ 25.5 GB/s = 0.69 ms (0.61 ms at the 28.7 GB/s peak). The lane costs 0.53 ms per block-token at 32 threads (the default) and 0.71 at 16 threads, so at 32 threads the copy is about 30% more expensive per block and at 16 threads the two are level. One point keeps it alive: the copy can overlap RX compute (doc 27 card 1), but RX busy time is the floor (R1). At 8 threads the lane is 1.10 ms per block-token, so the copy would win there. Prior art: the prefill streamed-expert copy path (doc 27 card 1, `crates/breccia-engine/src/graph.rs:2727`); doc 24 line 139 rejected Fiddler in-place decode because "decode has no copy path", which this card does not contradict because it reuses the prefill path. Gate: lane ms per host block-token at the production thread count ≥ 0.8 (copy 0.69 plus margin). `UNVERIFIED - requires the GPU qualification card`.

**R6. MI50 two-row batching (amortise dequant).** Doc 12 puts dequant at 92–94% of arithmetic, so a 2-row pass costs about 1.07 units against 2.0 for two single-row passes, up to about 1.9× tokens per MI50 pass. The MI50 is not on the critical path (busy 33%, RX 55–64%), so the aggregate gain is about 0 now. Needs the doc 21 verify path. `UNVERIFIED - requires the GPU qualification card`.

**R7. Work stealing between streams and cards.** The lane is already shared, so stealing lane work between streams has no target. Stealing experts to the other card runs into the MI50 link: 17.5 MB ÷ 2.9 GB/s = 6.0 ms per block-token, 5–11× the lane's 0.53–1.10 ms. A whole block (about 0.9 GB, 512 experts × 1.75 MB, doc 30 line 70) takes about 310 ms (`UNVERIFIED - requires the GPU qualification card`). Rejected.

**Ranking.** R2 (gate: shared-model check) > R3 > R4 > R5 (fails at the 32-thread default; ties at 16 threads) > R6 > R7 (rejected). R1 is held outside the ranking until its split is measured.

### 2.3 Batch size per card

Decode runs one token per stream per dispatch (M = 1). Per-card batch size therefore means multi-row verify on one stream (R6, doc 21), not a bigger concurrent batch.

## 3. Q2 — batch-invariance and determinism

### 3.1 The reference mechanism, and what it does not cover

Thinking Machines, "Defeating Nondeterminism in LLM Inference" (not re-read this round): in serving, a request's logits change with how many other requests share the batch, because the kernel's reduction order or tiling changes with batch shape. The fix is batch-invariant kernels with a fixed reduction strategy. The SGLang and vLLM deterministic modes apply the same idea with fixed attention splits and matmul tiling (by name only).

Breccia does no cross-stream batching inside a kernel. Decode is M = 1 per stream. Prefill chunk size is a per-session configuration (merged-prefill gate, `crates/breccia-engine/src/graph.rs:855`), not a function of concurrency. So the batch-shape mechanism does not appear. The concurrency paths that do exist go through shared host and thread-local state, listed below.

### 3.2 Mechanisms by which one stream's output can depend on the other

- **M1. Global batch count sets cut points.** `ensure_batch`'s SOFT_LIMIT test calls `batch_cmd::recorded()` (`crates/breccia-engine/src/graph.rs:1234`), which sums open batches across all devices (`crates/breccia-kernels/src/dispatch_batch.rs:344-346`). The MERGED_BATCH_CAP checks at `graph.rs:2362`, `:2640` and `:2813` use the same global sum. `recorded_on` (`dispatch_batch.rs:349`) exists but these sites do not use it. So one stream's command-buffer cut points depend on the other stream's recording. This changes sync placement; it changes numerics only if a cross-buffer dependency lacks a wait.
- **M2. Global DIRTY set.** `DIRTY` is thread-local, not per device (`dispatch_batch.rs:98`), and is cleared by `begin` (`:830`), by `end_submit` (`:976-979`), and by copies (`:1151`, `:1236`). A barrier is skipped when no bound buffer is DIRTY (`:1496-1531`). A `begin` on one device clears entries that belong to another device's open batch. Not reachable in the current call order (traced; latent).
- **M3. Descriptor-set cache keyed without device.** The key is `(pipeline handle, buffer handles)` (`dispatch_batch.rs:1365`, insert at `:1406`), and it is cleared only at shutdown (`:293`). Handles are unique per live object, so there is no collision while both models live. A handle reused after a model drop could alias a dead set (UNVERIFIED).
- **M4. Shared host expert pool.** One `Mutex<ExpertPool>` (`graph.rs:442`). The doc comment says the result does not depend on the thread count, only which worker runs a shard does (`crates/breccia-kernels/src/cpu/expert.rs:290-300`). The bit-identity test covers 2, 4 and 8 threads only (`crates/breccia-kernels/tests/cpu_experts.rs`); the default decode pool is 32 threads (`graph.rs:4155`), which the test does not cover. `last_distinct` (`expert.rs:305`, `:758`) is stats only. Error path only: an early `?` in `run_phase` (`expert.rs:425-449`) leaves Done messages on the shared channel.
- **M5. Device-dependent numerics (not concurrency).** `gemv_body.glsl` uses a fixed 32-lane logical tree (`:60`, `:258`), which is wave-size portable. `qsa_attn_scores.comp` takes its tree from `gl_SubgroupSize` (`:63`, `:65`, `:105`), so the tree changes between wave32 (RX) and wave64 (MI50). The single-stream result matched across devices for the tested prompt, which weakens this for that prompt only.
- **M6. Router near-tie amplification.** A rounding difference that flips a top-k near tie changes the route, and the stream then follows the new route. This is an amplifier, not a source.

**Ruled out for numerics:** `covered` and `in_flight` (stats only: `multi.rs:194-196` feeds `graph.rs:3141`); `skip_next_barrier` (no-op, `dispatch_batch.rs:319-331`); float atomics (no `atomicAdd` in any shader or Rust source); `co_tenant_check` (startup only, `crates/breccia-engine/src/model.rs:1360-1366`).

**On the observed divergence.** Streams 0 and 1 diverge at token 21 with top-2 margins 0.043 and 0.037. Those margins do not bound the logit difference between the two runs, so no bound is claimed. The first suspect is the residency confound in §3.3, not concurrency.

### 3.3 Candidate sites, ranked

| # | Site (file:line) | Mechanism | Test (CPU-only or 1-GPU) | What a result means |
|---|---|---|---|---|
| 1 | `crates/breccia-cli/src/main.rs:1659-1664` vs the single-stream default; plan lines `model.rs:844`, `:863` | Two-stream residency (9 GPU blocks) differs from single-stream. Different expert paths (GPU GEMV vs host AVX2) change numerics with no concurrency at all. | 1-GPU, load only: print the plan for each run and diff. Then rerun 2-stream with identical pinned `--gpu-experts` on both streams. | If the pinned runs agree, the divergence is the residency confound. |
| 2 | Same-device control: `--streams 2` on one device with pinned identical residency (small list) | Separates concurrency from device numerics. | 1-GPU. | Identical streams: concurrency is clean on one device. Divergence: M1–M4 are live. |
| 3 | `graph.rs:1234`, `:2362`, `:2640`, `:2813`; `dispatch_batch.rs:344` | M1: global cut points. | 1-GPU: diff `BRECCIA_TRACE_BATCH` cut points (`graph.rs:1907`, `dispatch_batch.rs:294`) between 1-stream and 2-stream. | Different cut points with identical tokens: no numeric effect shown. Different tokens: a missing-wait finding. |
| 4 | `dispatch_batch.rs:98`, `:830`, `:976-979`, `:1151`, `:1236`, `:1505` | M2: global DIRTY. | Assert in `begin` and `end_submit` that DIRTY holds only the current device's buffers. 1-GPU rules out same-device sharing only. The cross-device case needs a 2-GPU run (not a 1-GPU test). | Assert fires: M2 is reachable. Silent on a 2-GPU two-stream run: latent still. |
| 5 | `dispatch_batch.rs:1365`, `:1406`, `:293` | M3: set-cache handle reuse after drop. | 1-GPU: open, close and reopen a model; assert no cached key references a dropped buffer. | Assert fires: M3 is a live hazard, and the fix is a device-keyed cache. |
| 6 | `crates/breccia-kernels/src/cpu/expert.rs:290-300`; `cpu_experts.rs` | M4: thread count not covered at production 32. | CPU-only: bitwise match against `moe_topk_sequential` (`expert.rs:881`) at 32 threads with k = 10 routes. | Mismatch: M4 is a numeric source at production. Match: closed. |
| 7 | `crates/breccia-kernels/shaders/qsa_attn_scores.comp:63`, `:65`, `:105` | M5: wave-size tree. | 1-GPU per device: per-layer hidden-state hash, single stream, attention lengths that cross tile boundaries. | First mismatch layer identifies the reduction. |
| 8 | `crates/breccia-engine/src/graph.rs:2432` (`BRECCIA_TRACE_ROUTE`), per-token top-k | M6: near-tie route flip. | Diff the top-k sets per token between the two streams; at the first differing token, read the logit gap at the layer. | Gap below the per-device rounding bound: amplifier confirmed. |
| 9 | `expert.rs:425-449` | Error path: stale Done messages. | CPU-only: kill one worker, call again, compare to sequential. | Mismatch: error path leaks state. Low priority. |
| 10 | `graph.rs:1204-1218` | Deferred `pending_seg` overwrite (per session). | Code read only; a test needs the fault injector (`model.rs:1186`). | Low. |

Ranks 1 and 2 settle the observed divergence before any shared-state claim. The brief forbids a root-cause claim, so this order is a test plan, not a diagnosis.

## 4. Q3 — measurable cards

Every card is flag OFF by default and has a gate and a receipt. Cards 1 and 5 change no runtime path.

**Card 1: residency-matched control (script only).** Flag: none (no runtime change). Gate: the arena plan lines for the compared runs match, with identical residency per stream. Receipt: per-stream token sequences for (a) 1-stream per device, (b) 2-stream on one device, (c) 2-stream across devices, and the first divergent token for each pair. Test: 1-GPU, then 2-GPU. Resolves ranks 1 and 2.

**Card 2: per-layer hidden-state hash, `BRECCIA_LAYER_HASH=1` (default OFF).** Writes one hash per layer per token. Gate: single stream on RX and single stream on MI50 produce identical per-layer hashes for N prompts at attention lengths that cross tile boundaries (M5). Receipt: first (token, layer) mismatch, or "none". 1-GPU per device.

**Card 3: per-device cut counts, `BRECCIA_BATCH_LOCAL_COUNT=1` (default OFF).** The SOFT_LIMIT and MERGED checks (`graph.rs:1234`, `:2362`, `:2640`, `:2813`) read `recorded_on(dev)` instead of the global sum. Gate: 2-stream `BRECCIA_TRACE_BATCH` cut points equal the 1-stream cut points, and 64-token output is unchanged. Receipt: trace diff. If the tokens change, that is a missing-wait finding (M1). 1-GPU.

**Card 4: dispatch-state assert, `BRECCIA_DISPATCH_ASSERT=1` (default OFF).** Asserts that DIRTY holds only the current device's buffers (`dispatch_batch.rs:830`, `:979`) and that no set-cache key references a dropped buffer (`:1365`, `:293`). Gate: zero asserts on the 1-GPU run and on the 2-GPU two-stream run (only the 2-GPU run tests the cross-device case). Receipt: assert count per run.

**Card 5: lane bit-identity at production thread count (test only, no runtime flag).** Extend `crates/breccia-kernels/tests/cpu_experts.rs` to 32 threads (the default decode pool, `graph.rs:4155`) and 16 threads, with k = 10 routes. Gate: bitwise equal to `moe_topk_sequential`. Receipt: the test output. CPU-only.

**Card 6: RX stream-in of host-lane experts, `BRECCIA_RX_STREAM_BLOCKS=<n>` (default 0, OFF; conditional).** Reuses the prefill copy path (doc 27 card 1, `graph.rs:2727`). Gate before any build: measured lane ms per host block-token at the production thread count ≥ 0.8, and RX busy + copy + streamed compute ≤ `S`. Receipt: aggregate tok/s against the same-day baseline, a 64-token output bit-identical to the baseline, and RX busy. `UNVERIFIED - requires the GPU qualification card`.

## 5. Rejected

- **Layer split or row split across cards** (llama.cpp split modes, exllama device map). Per-layer sync; BRIEF:192 rejects it (activation shipping at 2.9 GB/s).
- **Device-crossing head handoff.** Removed after 2 of 16 processes gave a wrong sequence (BRIEF:187).
- **MI50 streaming of routed experts or whole blocks.** 6.0 ms per block-token (`UNVERIFIED - requires the GPU qualification card`) against a 0.53–1.10 ms lane; a whole block takes about 310 ms; BRIEF:127 rules out MI50 streaming. R7.
- **Work stealing between streams.** The lane is already shared, so there is no target except residency (R7).
- **MI50 two-row batching now.** The MI50 is off the critical path (R6). Revisit only after the doc 21 verify path lands.
- **Re-arming `skip_next_barrier`** (`dispatch_batch.rs:319-331`). Parked: it failed parity (20 of 32). The call sites are not dispatches (split-K).
- **Device-keyed set cache now** (`dispatch_batch.rs:1365`). No reachable failure is shown. Revisit only if card 4 fires.
- **Lifting `Multi`'s single-device guard** (`multi.rs:135`). None of the cards above needs it.
- **Helix, HexGen, ThunderServe, Mélange, Cephalo as ports.** Fleet-scale or per-layer. Their mechanisms are folded into R2–R4.
- **KTransformers one-layer-late deferral** (doc 14). Fails bit parity.
- **Fiddler in-place decode** (doc 24 line 139). "Decode has no copy path." Card 6 is conditional and reuses the prefill path rather than contradicting this.
- **Lane-cost tuning at 32 threads as a gain.** The lane is 0.53 ms per block-token here; the thread-count sweep is doc 30's card 1.

## 6. Gate 0 and open items

Measure before ranking changes:
1. Lane ms per host block-token at the production thread count, with the production residency read from the plan lines (`model.rs:844`, `:863`). Decides R5 and card 6.
2. Whether the pair step is additive (doc 28 card 1), and whether a 16+16 split lane runs at a usable rate. Both are needed before R1 can be reopened.
3. Whether two sessions can share one `Model` (`multi.rs:135`, `graph.rs:636`). Decides R2.
4. Per-block GPU time per card (`BRECCIA_GPU_TIME` per block). Needed for R3 and R4.
5. Block size: doc 27 line 31 flags that 0.89 GiB (doc 22 line 10) and 0.833 GiB (512 × 1.75 MB) differ by a unit slip; UNVERIFIED which was meant.

Open items:
- Thread default conflict: the code path gives 32 (`graph.rs:4155`); comments at `graph.rs:439-441` and `multi.rs:25` say 16, and the help text at `main.rs:277-279` describes a physical-cores-over-two default capped at 16. The 16-thread row is 17.0 tok/s and the 32-thread row is 20.2 (§1).
- Aggregate baselines conflict (13.2–14.6, 20.0–20.8, 22.8–23.7; §1).
- Residency basis: BRIEF:79 is single-stream dual-card; the 2-stream CLI default is 39 host blocks per stream (§1).
- Doc 31 uses both 18 and 39 host blocks in its tables (lines 5, 24 and 25).
- External papers (Helix, HexGen, ThunderServe, Mélange, Cephalo, Thinking Machines, SGLang and vLLM deterministic modes): mechanisms stated from prior knowledge; section pins not read this round.
- `docs/research/README.md` has a second "Round verdicts" heading (lines 54 and 80). Not fixed here.
