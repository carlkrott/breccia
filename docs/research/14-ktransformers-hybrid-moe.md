# 14 — KTransformers: hybrid CPU/GPU MoE, audited against Breccia prefill

Round 14. Closes README row 20. Section A is the Vulkan capability half of this round. Sections B–E are the KTransformers audit and the Breccia adaptation.
Engine audited: **KTransformers** (KT) at upstream commit `a694e71` (v0.7.2), read-only. Upstream paths are repo-relative to that tree (`kt-kernel/…`, `doc/en/…`, `archive/…`); Breccia paths are repo-relative to this repo. Nothing was built or run on a GPU, and no benchmark was run. Anything that needs the card is marked `UNVERIFIED - requires the GPU qualification card`.

**Label note.** "D1–D5" below means the phase-5 host-expert-import plan (`docs/phase-5/01-host-expert-import.md` at commit `cb4042d`; the file is not on this branch). It is not doc 10's dispatch fusions D1–D5 (see doc 13's label note).

**Headline.** Six findings decide the design.

1. **KT's default expert placement is positional, not frequency-ranked.** `--kt-expert-placement-strategy` defaults to `uniform` (`doc/en/kt-kernel/experts-sched-Tutorial.md:102,135`). The frequency-ranked helper `generate_gpu_experts_masks` (`kt-kernel/python/experts_base.py:162`) is exported (`kt-kernel/python/__init__.py:62,107`) and tested (`kt-kernel/test/test_generate_gpu_experts_masks.py`), with no other in-repo call site. Its sglang caller is UNVERIFIED. Basalt's expert-level frequency ranking (`docs/phase-4/02-untried-levers.md:42`) is the stronger form. Breccia's own residency is block-level (30 of 48 blocks, `docs/BRIEF.md:79`).
2. **KT's deferral is a one-layer-late approximation and cannot pass Breccia's bit-parity gate.** Deferred experts are held in a two-slot buffer keyed by layer index (`experts_base.py:227`, slot = `layer_idx % buffer_depth`) and folded in one layer late (`select_deferred_experts` at `:501`, `forward_on_pinned_buffers` at `:609`). The AMX merge adds into an fp32 accumulator from a bf16 output and writes bf16 back (`kt-kernel/operators/amx/moe_base.hpp:760–790`), so the output is re-rounded. KT's own defaults disagree: code 0 (`experts_base.py:455`), README table 2 (`kt-kernel/README.md:505`).
3. **Breccia's prefill already groups by expert per chunk.** The grouped lane runs each expert once across the chunk (`crates/breccia-engine/src/graph.rs:2098,2381–2423`), and the merged path runs one multi-user GEMM per distinct expert (`graph.rs:2713–2734`). The "expert-major grouping" idea from the first draft is redundant and is dropped. KT's layerwise prefill is a coarser form of the same idea: a whole-layer copy gated by `--kt-gpu-prefill-token-threshold`, whose two documented readings are in B2 (`kt-kernel/README.md:366`, `:543–551`). It copies 852 MiB per block against 611 MiB of distinct-expert bytes at T=64, so the copy is at least +39 % (uniform routing, assumption A1; Section C).
4. **The prefill host lane is not byte-bound at its measured rate.** Doc 07 §C1.3 (`docs/research/07-attribution-and-ollama.md:552–572`, an older measurement, UNVERIFIED for current code) puts `experts(host)` at 61.8–81.6 ms/token of an 87.7–107.1 ms/token full chunk, about 3.9–5.2 s per 64 tokens. The union bytes for that chunk are about 11.5 GB (Section C), so the prefill lane moves about 2–3 GB/s. The token-by-token decode lane reads at 33 GB/s with 32 threads (`docs/phase-4/02-untried-levers.md:17`). The likely cause is access pattern or compute, not bandwidth: the multi-user CPU GEMV is gated to Down shards of `Iq4Nl` or `Q2_0` (`crates/breccia-kernels/src/cpu/expert.rs:602`, `cpu/mod.rs:134–136`), and whether production IQ3_XXS experts reach it is UNVERIFIED. R1 separates these before any transfer change is scheduled.
5. **KT's AMX matrix path has no Vulkan analogue on these cards, and the benchmarked 64-token chunk sits below its switch point.** KT switches to the matrix kernel above `qlen > 4·E/k` (`kt-kernel/operators/amx/moe.hpp:244,257`), which is 204.8 at E=512, k=10 (so from 205 tokens). The benchmarked 64-token chunk (`docs/BRIEF.md:80`) is below it, and R2's 256 crosses it. The CLI default is `--prefill-chunk 0` (token by token, `crates/breccia-cli/src/main.rs:228–241`). Neither card exposes `VK_KHR_cooperative_matrix` (`docs/research/06-vulkan-gemm-shapes-and-ncnn.md:238,352`), so a WMMA-class prefill GEMM is off the table. The nearest Breccia analogue is `GemmMany` with `USERS_MAX = 8` (`crates/breccia-kernels/src/gemm.rs:12`).
6. **The CPU lane is optional by standing verdict, and Breccia's CPU path has no AVX-512 kernel.** `Path` has only `Avx2` and `Scalar` (`crates/breccia-kernels/src/cpu/mod.rs:141`). The target CPU (Ryzen 9 7950X class) reports AVX-512 f/bw/vnni/bf16/vbmi and no AMX. Any AVX-512 lane kernel is low priority; R3 must justify it first.

**Standing verdicts respected.** Quantised GEMV is instruction-issue bound (doc 12). Submit count is not the lever. Recording costs 66 ns per dispatch (doc 11). The GPU executes host-resident experts at 22–29 GB/s, so the CPU lane is optional. The RX is pinned at 500 MHz. The lead lever is legs, led by the hyper-connection projector (doc 10); nothing here outranks it. Breccia's T3 NVMe tier never shares gpu-manager's 4d kv-slots/ directory, and weights never go on NVMe.

## A. Vulkan capability angle

What the GPU side already measures (doc 09):
- GPU dequant GEMV reading host-pinned experts: f16 26.4, q8 25.5, q4 22.4 GB/s. Copy-engine ring DMA 28.7 GB/s; pinned host 28.7–29.7 GB/s; duplex 17.9 GB/s each way (`docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md:58–59,77–78`).
- Imported host pointers need 4096 B alignment (`doc 09:39`, `minImportedHostPointerAlignment`). Pinning is required for correctness, and `LimitMEMLOCK=infinity` is the prerequisite (`doc 09:99–101`).
- No cooperative matrix on either card (`doc 06:238,352`). Under doc 12's law (ALU-bound iff `I_unit > 8·bpw` on MI50, `53·bpw` on RX), the per-type classification for IQ3_XXS is UNVERIFIED here; doc 12 holds the instruction counts. The prefill GEMV path has no matrix unit to fall back on.

**Decisive check.** One check decides whether the CPU lane adds anything beside the GPU's host reads. Three runs over the same expert set (one block's host-resident IQ3_XXS experts, 10 per token):
1. GPU only: dequant GEMV reading pinned host memory. The 22.4–26.4 GB/s figures come from doc 09's harness (`r09-zerocopy`, external and not in this tree; doc 09 lines 16–17, 71–76, 447–456). `TransferBenchOptions` (`crates/breccia-vk/src/transfer_bench.rs:36`) is the DMA-ring copy bench, not a GEMV, so run 1 needs a GEMV-from-pinned-host driver that is not in the tree; it stays in R3's touch list.
2. CPU only: AVX2 expert GEMV over the same set, run through `breccia bench cpu-experts` (`crates/breccia-cli/src/main.rs:399`; options struct `CpuExpertBenchOptions`, `crates/breccia-kernels/src/cpu/bench.rs:47`).
3. Both at once. A driver that starts both legs together is not in the tree; building it is part of R3.

Record GB/s per leg, the RX core clock during each run, and host DRAM bandwidth. **Decision rule:** if run 3 is at or below max(run 1, run 2), the CPU lane adds nothing beside the GPU's host reads, because both draw on the same host memory, and it should leave the prefill path. If run 3 approaches run 1 plus run 2, the lane is additive and keeps its share of the 18 host blocks. The standing verdict predicts the first outcome. UNVERIFIED - requires the GPU qualification card.

## B. KTransformers anatomy (upstream a694e71)

- **B1 Placement.** `--kt-num-gpu-experts` sets the GPU expert count (`kt-kernel/README.md:323,334,364`). `--kt-expert-placement-strategy` takes `uniform` (default), `frequency`, `front-loading` or `random` (`experts-sched-Tutorial.md:135`). Uniform spreads GPU experts evenly across layers (`:102`). The helper `generate_gpu_experts_masks` is a global top-k over flattened layers × experts by activation frequency (`experts_base.py:162`, example at `:186`). Three more flags are in the tutorial's key-parameter table (`doc/en/kt-kernel/experts-sched-Tutorial.md:133–138`): `--kt-gpu-experts-ratio` (`:134`; a fraction of all experts, overriding `--kt-num-gpu-experts`), `--init-expert-location` (`:136`; the `.pt` activation-statistics file the `frequency` strategy needs, passed at `:114` and `:124`), and `--kt-enable-dynamic-expert-update` (`:137`; in use at `:125`), which redistributes GPU experts from routing statistics during layerwise prefill (`kt-kernel/README.md:543–551`) and needs the prefill threshold set (trigger conditions at `doc/en/kt-kernel/experts-sched-Tutorial.md:205–212`).
- **B2 Decode versus prefill.** The mode is `'d'` for `qlen <= 1` and `'p'` otherwise (`kt-kernel/operators/moe_kernel/moe.hpp:415–416`). `--kt-gpu-prefill-token-threshold` has two documented readings, and this repo does not settle which one the runtime applies. **Reading 1** (`kt-kernel/README.md:366`, example 2048; `:543–551`, scoped to FP8 and RAWINT4): at or below the threshold, hybrid CPU+GPU prefill; above it, layerwise GPU prefill, which copies a whole layer's experts to the GPU for the batch and needs one MoE layer of extra VRAM. Layerwise prefill also zeroes resident GPU experts while it runs (`doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md:90`, `doc/en/kt-kernel/Native-Precision-Tutorial.md:268`). **Reading 2** (`doc/en/kt-kernel/experts-sched-Tutorial.md:138`, example 512 at `:126`): the token threshold that triggers dynamic expert redistribution during prefill, with no method limit stated. The readings agree only if redistribution happens inside layerwise prefill, as `:543–551` says. Which scope the runtime uses is UNVERIFIED (the orchestration is in sglang, not in the tree).
- **B3 CPU kernels.** AMX matrix path above `qlen > 4·E/k` (`amx/moe.hpp:244,257`), vector path below. The matrix path consumes KT's tiled 64-byte weight layout (`doc/en/AMX.md:78`, §1). Section 3 (`:105`) is an AVX-512 kernel for low arithmetic intensity, and §4 (`:119`) covers fused tasks and dynamic scheduling. The 21 TFLOPS BF16 figure on Xeon4 CPUs is KT-reported (`doc/en/AMX.md:127`; UNVERIFIED). Work stealing is one shared atomic cursor (`kt-kernel/cpu_backend/worker_pool.cpp:157–206`), with hwloc strict binding (`worker_pool.cpp:107,334`). The variant ladder amx → avx512_bf16 → avx512_vbmi → avx512_vnni → avx512_base → avx2 is in `kt-kernel/python/_cpu_detect.py:24–29,48–104,263–268`, overridable by `KT_KERNEL_CPU_VARIANT`. On the target's flags the ladder would select avx512_bf16 (UNVERIFIED; not run).
- **B4 Deferral.** Per-layer default 0 (`experts_base.py:425,455`). The protected count is `num_experts_per_tok − max_deferred` (`:576–577`). The base class throws "Not Implemented" for incremental merge (`kt-kernel/operators/moe-tp.hpp:252–256`); the AMX class implements it (`amx/moe_base.hpp:760–790`). CLI flag `--kt-max-deferred-experts-per-token` appears at `kt-kernel/README.md:324,335,365,452,493,505`.
- **B5 NUMA.** Subpool configuration is in `experts_base.py:313–349`. The target has one NUMA node, so subpools do not apply there.
- **B6 Long context and prefix KV.** `doc/en/long_context_introduction.md` and `doc/en/long_context_tutorial.md` describe sparse/anchor KV attention, which is approximate. `doc/en/prefix_cache.md` describes kvc2, a GPU-CPU-Disk KV scheme (`:38`) with a CPU memory budget (`cpu_memory_size_GB`, example 500, `:16`) and a disk path (`:17`); its code is under `archive/` (legacy).

## C. Breccia adaptation sketch (prefill first)

**C1 How Breccia prefill runs today** (code, not measurement).
- Entry `pub fn prefill(&mut self, toks, pos0)` (`crates/breccia-engine/src/graph.rs:2102`). Its doc comment (`:2094–2101`) requires a single-device placement and a configured `prefill_chunk`, and describes the once-per-expert grouped lane (§8s). `prefill_cap(target)` returns target + target/2 (`graph.rs:4101–4107`); where it is applied on this path is UNVERIFIED. Chunk buffers are allocated once for `prefill_chunk` (`graph.rs:227`).
- Placed blocks (30 of 48, `docs/BRIEF.md:79`) run whole on the GPU (`graph.rs:2337–2340`) and skip the CPU lane (`graph.rs:2377–2378`).
- The other 18 blocks take one of two paths:
  - Default (`--prefill-merged` off): the CPU grouped lane `moe_topk_chunk` (`graph.rs:2381–2434`, call at `:2423`). The CPU reads expert bytes from host memory; none cross PCIe.
  - `--prefill-merged` on (default off; single device only, `graph.rs:1049–1052`): each distinct expert's three blobs go mmap → host-visible slot (`fill_stream_slot`, `graph.rs:2516`, called at `:2734`) → device. One multi-user GEMM runs per expert (`prefill_block_merged`, `graph.rs:2566`; streamed flag at `:2580–2588`), with a submit every `STREAM_SLOTS = 8` experts (`graph.rs:271`, `:2713–2714`).
- Hooks already in tree: `BRECCIA_TRACE_PREFILL` (`graph.rs:2184–2209`), `BRECCIA_TRACE_ROUTE` (`:2432`), `BRECCIA_TRACE_CHUNK` (`:2441`), `BRECCIA_TRACE_MERGED` (`:2883`, streamed/fill/wait ms at `:2892–2893`). `bench agents` prefills streams sequentially (`crates/breccia-cli/src/main.rs:2080–2088`), divides by `n` (`:2234`), and prints the merged union floor (`:2229–2240`).
- Flags: `--prefill-chunk` and `--prefill-merged` at `crates/breccia-cli/src/main.rs:228–241`; parity via `--same-prompt` (`main.rs:254`).

**C2 Byte floors, with assumptions stated.**
- A1. Routing is uniform over 512 experts, top-10 per token. UNVERIFIED. Real routing is skewed, and skew lowers the distinct-expert count for a given draw count, so uniform gives an upper bound on union bytes.
- A2. Expert size 1.665 MiB (`docs/phase-4/02-untried-levers.md:16`); 18 host blocks; host bytes per token = 10 × 18 × 1.665 = 300 MiB (BRIEF says 301; rounding).
- A3. Distinct experts per block at T tokens is `512 · (1 − (1 − 10/512)^T)`: 367 at T=64 (5.7 expert loads per token), 471 at T=128 (3.7), 509 at T=256 (2.0). Decode is 10 per token. Union bytes at T=64: 367 × 1.665 × 18 ≈ 11.5 GB per chunk.
- A4. Zero-reuse bytes (one read per token-expert pair) at T=64: 64 × 301 MiB = 18.8 GiB ≈ 20.2 GB per chunk.
- A5. Chunk time = 64 ÷ per-agent rate. The 6.0–6.45 tok/s headline (`docs/BRIEF.md:80`) gives 9.9–10.7 s. If the headline divides by `n` for sequential streams (doc 07 §C1.2), the per-stream rate is about double and a chunk takes about 5 s. Doc 07's own stage sums (87.7 and 107.1 ms/token) imply 9–11 tok/s per stream, which matches neither reading. Which applies is UNVERIFIED.
- A6. Bandwidth: the token-by-token host lane at 33 GB/s with 32 threads (`docs/phase-4/02-untried-levers.md:17`; prefill rate UNVERIFIED); GPU reads and copy engine at 22.4–28.7 GB/s (doc 09).

**Floors per 64-token chunk (A1–A6):**

| Path | Bytes | Rate | Time | Share of 9.9–10.7 s chunk | Share of 5.0–5.3 s chunk |
|---|---:|---:|---:|---:|---:|
| CPU lane, union | 11.5 GB | 33 GB/s | 0.35 s | 3–4 % | 7 % |
| CPU lane, zero reuse | 20.2 GB | 33 GB/s | 0.61 s | 6 % | 12 % |
| GPU streamed, union | 11.5 GB | 22.4–28.7 GB/s | 0.40–0.51 s | 4–5 % | 8–10 % |
| GPU streamed, zero reuse | 20.2 GB | 22.4–28.7 GB/s | 0.70–0.90 s | 7–9 % | 13–18 % |

Reading the table: transfer floors are 3–18 % of a chunk under every assumption set. Doc 07's measured lane time (3.9–5.2 s per chunk, headline 4) is 11–15× the union floor, so the prefill lane is not limited by bytes at its measured rate. The floors still bound what any transfer fix could save.

**C3 Adaptation sketch.**
- Keep what Breccia has: per-expert-per-chunk grouping, the byte-floor check, and the existing trace hooks.
- Take only the coarse layerwise idea, and only at chunks of about 256 tokens or more. A whole-layer copy is 852 MiB per block at any T, against 848 MiB of distinct-expert bytes at T=256 and 611 MiB at T=64, so the copy is at least +39 % at T=64 (uniform routing, A1). It also costs one MoE layer of VRAM (`kt-kernel/README.md:545–551`). The threshold flag's scope is UNVERIFIED under both readings (B2), so this sketch proposes no KT threshold value.
- Do not take deferral, the AMX matrix path or KT's tiled repack (Section E).
- Sequence: R1 decides between R2–R5. If the lane is limited by per-user GEMV, the lever is the lane's reuse (R3), and widening the multi-user branch beyond the Down shard is a candidate, not a finding. If the copy and the GEMM both matter, R4–R5 come next. Prefill stays ahead of any transfer work.

**C4 Scope.** Decode legs are not this doc's subject; the lead lever stays the hyper-connection projector. The 32K regime is separate: BRIEF item 4b is NVMe-streaming-bound (`docs/BRIEF.md:284`, ~13 tok/s, ~9 s per 128-token chunk).

## D. Ranked recommendations (at most six)

**R1 — Split the prefill lane's time before any transfer work (gate).**
- Effect: none directly. It decides R2–R5. Expected to show whether the lane's time is byte reads, per-user GEMV compute, or waiting on the streamed copy.
- Measure: `bench agents` at 64-token chunks in a quiet window, with `BRECCIA_TRACE_PREFILL`, `BRECCIA_TRACE_ROUTE` and `BRECCIA_TRACE_CHUNK` on, and with trace off to price the overhead. Count GEMV calls per expert per chunk in the CPU lane against unique experts. Read the production model's Down-projection quant type (the gate at `cpu/expert.rs:602`) to settle whether the multi-user branch runs at all. Record the RX clock. Decision rule: if calls track tokens × experts (per-user), fix the reuse (R3); if calls are per expert and the lane is still slow, it is kernel work and low priority under the optional-lane verdict; if the lane is under 10 % of chunk time, drop it from the list.
- Risk: trace overhead only; no parity effect.
- Touches: `crates/breccia-engine/src/graph.rs:2184–2209, 2432, 2441`; `crates/breccia-cli/src/main.rs:2080–2088, 2229–2240`.

**R2 — Chunk-size sweep on existing flags (one small output change).**
- Effect: prefill through union bytes (5.7 → 3.7 → 2.0 expert loads per token at T=64/128/256), and through the per-chunk fixed cost (doc 07 §C1.3: under the older 64+64+4 split, the 4-token tail ran at 660.9 ms/token against 87.7 for a full chunk). Context: chunk scratch grows with `prefill_chunk` (`graph.rs:227`); the size is UNVERIFIED. Cache: none.
- Measure: `--prefill-chunk` in {32, 64, 128, 256} × `--prefill-merged` on and off, same prompt, PP per stream from per-stream walls (not `/n`); the tail chunk timed separately; parity by `--same-prompt` (`main.rs:254`).
- Risk: VRAM at 256. The planner may already give 66+66 (`docs/BRIEF.md:162`), so the tail case may differ from doc 07. The phase-4 gate is 24/24 bit-identical at chunk 1 versus 256 (`docs/phase-4/02-untried-levers.md:123`); the merged path at 256 is UNVERIFIED.
- Touches: flags at `crates/breccia-cli/src/main.rs:228–241`; `crates/breccia-engine/src/graph.rs:227, 4101–4107`; a per-stream wall print beside the `/n` mean at `crates/breccia-cli/src/main.rs:2271–2272, 2349` (the tree prints only the mean there, so R2 adds the per-stream line).

**R3 — Three-run CPU-lane check, then decide the lane (Section A).**
- Effect: if the lane adds nothing beside the GPU's host reads, the 18 host blocks' prefill moves to the streamed path (more PCIe bytes, less host-memory contention). If the lane is additive, it keeps its share. Either way PP and decode for those blocks change. Cache and context: none.
- Measure: Section A's three runs. UNVERIFIED - requires the GPU qualification card.
- Risk: the GPU leg needs pinned host memory (`LimitMEMLOCK=infinity`, `doc 09:99–101`). The standing verdict already calls the lane optional, so this confirms or drops it rather than overturning it. Any lane-kernel work (an AVX-512 path, or widening the Down-only multi-user branch) waits for this result.
- Touches: `crates/breccia-kernels/src/cpu/expert.rs:602` (multi-user gate), `crates/breccia-engine/src/graph.rs:2381–2434` (lane), `crates/breccia-vk/src/transfer_bench.rs:36`, `crates/breccia-kernels/src/cpu/bench.rs:47`.

**R4 — Overlap the next block's streamed expert copies with the current block's GEMM in `prefill_block_merged`.** Gated on R1 showing copy and GEMM each at 10 % or more of a chunk.
- Effect: prefill, by hiding copy time under GEMM. Saves at most min(copy, GEMM) per block. No bytes saved. Cache and context: none.
- Measure: `BRECCIA_TRACE_MERGED` streamed/fill/wait ms (`graph.rs:2892–2893`); near-zero wait means hidden. Compare PP with overlap on and off.
- Risk: a second staging ring doubles host-visible staging (`STREAM_SLOTS = 8` per submit today). Sync hazards; parity must stay bit-identical. The record overlap comment at `graph.rs:3010` describes the single-stream pattern; whether prefill reuses it is UNVERIFIED.
- Touches: `crates/breccia-engine/src/graph.rs:2566–2746`; `crates/breccia-vk/src/transfer.rs:47` (`RingConfig`).

**R5 — Phase-5 host-expert import (D3, `docs/phase-5/01-host-expert-import.md` at `cb4042d`) to replace the mmap → staging copy in `fill_stream_slot`.** Gated on R1 (copy share 10 % or more) and on R4 (if overlap hides the copy, the import buys little).
- Effect: transfer. Removes the host-side copy, which doc 09:87 prices at about 40 ms for 799 MiB (the token-by-token case); the prefill figure is UNVERIFIED. Bytes are unchanged; PP moves only if the copy is exposed.
- Measure: `BRECCIA_TRACE_MERGED` fill ms before and after; GB/s for pinned import versus staged copy.
- Risk: `minImportedHostPointerAlignment` is 4096 B (`doc 09:39`), so expert blobs must be aligned in the mapped file or re-staged (layout UNVERIFIED). Requires `LimitMEMLOCK=infinity` (`doc 09:99–101`). Parity-gated. The phase-5 plan's D1 (`8182899`) ships `--host-expert-import` default-off with no consumer; that commit is not on this branch.
- Touches: `crates/breccia-vk/src/` (import path); `crates/breccia-engine/src/graph.rs:2516` (`fill_stream_slot`).

**R6 — 32K regime: diagnose before design; KT's disk tier only after the class is known.**
- Effect: context and cache. BRIEF item 4b is NVMe-streaming-bound at 32K (~13 tok/s, ~9 s per 128-token chunk), a different regime from the 64-token chunk above. KT's kvc2 (`doc/en/prefix_cache.md`; code under `archive/`) is a CPU-plus-disk KV tier and is relevant only if the 32K cost is KV bytes rather than expert bytes.
- Measure: at 32K, per-chunk bytes read from disk, split into KV, expert and weight bytes, using the R1 trace hooks; wall time per chunk.
- Risk: weights never on NVMe; the T3 NVMe tier never shares gpu-manager's 4d kv-slots/ directory; kvc2 is legacy.
- Touches: none yet (design). Breccia's tier location is UNVERIFIED.

## E. What not to copy

1. **Deferred experts** (`experts_base.py:501–680`; `amx/moe_base.hpp:760–790`). A one-layer-late approximation with a bf16 round-trip on the merge. It changes the computed function, so it cannot meet the bit-identical chunk gate (`docs/phase-4/02-untried-levers.md:123`). Reject.
2. **Positional (`uniform`) placement as the default** (`experts-sched-Tutorial.md:102`). Breccia's residency is block-level, and Basalt's expert-level frequency ranking is the stronger form (`docs/phase-4/02-untried-levers.md:42`). KT's frequency helper is the thing to compare against; no KT default path uses it.
3. **Whole-layer prefill copy** (`kt-kernel/README.md:545–551`). It moves at least +39 % bytes at T=64 against the distinct-expert set (uniform routing, A1), costs one MoE layer of extra VRAM, and matches the union only near T≈256. Its threshold scope is UNVERIFIED (B2).
4. **AMX matrix kernels and the tiled repack** (`amx/moe.hpp:244,257`; `doc/en/AMX.md:74–134`). The target has no AMX. The kernels consume KT's tiled layout, so adopting them means a second copy of expert weights in host memory or a repack at load. KT's §3 AVX-512 kernel for low arithmetic intensity (`doc/en/AMX.md:105`) is the closer fit: the target has AVX-512 (VNNI, BF16, VBMI), and Breccia's CPU path has no AVX-512 kernel. Any such port is low priority, and R3 must justify it first.
5. **Work stealing and hwloc strict binding** (`worker_pool.cpp:157–206, 107, 334`). Built for many threads per NUMA node. The lane is optional, and the target has one NUMA node, so the subpools (`experts_base.py:313–349`) are inert.
6. **Sparse/anchor KV attention** (`doc/en/long_context_introduction.md`). Approximate attention changes outputs; Breccia's exact-KV parity forbids it.
7. **The kvc2 CPU-plus-disk KV tier** (`doc/en/prefix_cache.md`; `archive/`). Legacy. Revisit only under R6 and the NVMe rules (weights never on NVMe; never the 4d kv-slots/ directory).
8. **The CPU variant ladder** (`kt-kernel/python/_cpu_detect.py`). Breccia selects its own `Path` (`crates/breccia-kernels/src/cpu/mod.rs:141`). Copying the ladder would route the target to avx512_bf16 kernels Breccia does not have.
