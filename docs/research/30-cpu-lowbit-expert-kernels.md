# 30 — CPU low-bit expert kernels for the host lane

Round 30, read-only. Base 1316e0f, 2026-10-28. No code changed, no benchmark run on the live pair, nothing committed. Speed figures from papers, READMEs and other projects are UNVERIFIED. "Derived" marks arithmetic on cited measurements, not a new run.

Parent measurement context (used as given, not re-measured here): two-GPU decode (RX 6950 XT and MI50, one stream per GPU, Qwen3.8 Flash-Next IQ3_XXS, 70.6 GiB mmapped, 30 GiB memory cap). Host lane 43–101 ms/step, GPU fence 44–53 ms/step, aggregate 11.7–19.0 tok/s.

## 1. Verdict

The host lane is CPU-bound, and its kernel is already an AVX2 fused decode-and-dot on f32 activations. The largest measured lever is not the kernel. §8y shows the same binary moving the lane from 85.7 to 41.2 ms/step between 8 and 32 threads. The parent's 101 ms end is a residency or I/O question before it is a kernel question. So:

1. Gate 0 first (§3, no code): resolve how many blocks are host-resident, read the page-cache I/O counter during the window, and run at `--threads 32`.
2. A zero-code per-CCD probe (card 1) before any affinity code. Doc 08 §6.3 ran an 8-thread placement test that came back negative, so card 6 is low-prior (§6.2).
3. Kernel cards, each gated in lane ms/step and converted to tok/s with the step model (§2).
4. Reject the T-MAC and BitNet LUT approaches. Their table does not fit L2, and they need ternary or bit-plane weights.
5. VNNI int8 with per-block q8 activations is not a kernel-level card. One limb fails the 1e-5 bar on 94–99% of synthetic rows. Two limbs pass on Gaussian synthetic rows (0 of 4,000 over 1e-5) and fail 0.4% of outlier-heavy rows. It is worth one end-to-end-gated card (card 4). Real-activation and real-blob results are still open.

## 2. Step model (derived)

pair step ≈ lane + fence + R, with R ≈ 18 ms. Aggregate tok/s = 2000 / pair step (ms), since each step yields one token per stream.

R comes from the §8y A/B/A table (`docs/phase-4/01-forward-pass.md`, lines 2109–2175), which reports aggregate, lane and fence per step:

| threads | aggregate tok/s | step = 2000/agg (ms) | lane (ms) | fence (ms) | R = step − lane − fence (ms) |
|---|---:|---:|---:|---:|---:|
| 8 | 14.71 | 136.0 | 85.7 | 32.3 | 18.0 |
| 16 | 16.99 | 117.7 | 55.2 | 44.7 | 17.8 |
| 32, run 1 | 20.19 | 99.1 | 41.2 | 39.5 | 18.4 |
| 32, run 2 | 20.77 | 96.3 | 41.2 | 36.6 | 18.5 |

Consequences:

- The parent's bounds reproduce to within 0.1 tok/s: 43 + 44 + 18 = 105 ms gives 19.0 tok/s, and 101 + 53 + 18 = 172 ms gives 11.6 tok/s, against the parent's stated 11.7.
- The lane is additive per pair step. `lane_ms_per_step` sums both streams' lane time (`crates/breccia-engine/src/multi.rs:106–108`). The driver is single-threaded: `multi.rs` has no thread spawn and no lock, and the pool mutex (`graph.rs:442`) never contends. Each lane ms saved is one pair-step ms saved.
- Conversion, with fence held at 44 ms and R at 18 ms:

| lane (ms/step) | pair step (ms) | aggregate (tok/s) |
|---:|---:|---:|
| 101 | 163 | 12.3 |
| 60 | 122 | 16.4 |
| 43 | 105 | 19.0 |
| 41 | 103 | 19.4 |
| 31 | 93 | 21.5 |
| 21 | 83 | 24.1 |

- Slope: −0.213 tok/s per ms at a 97 ms step, and −0.189 at 103 ms (derived, 2000/step²).
- Caveat: the four §8y rows fit R to within 0.7 ms. That is a consistency check on one table, not an independent validation. The §8y fence moves with thread count (32–45 ms), so a card that also changes the fence changes the slope. Every gate must report the fence.

## 3. Gate 0: residency and thread regime (existing tooling only)

Run before any card. Record, in one output:

1. **Thread count.** Pass `--threads 32` and record the value the lane used. Decode resolves its default in `default_threads()` (`crates/breccia-engine/src/graph.rs:4155`, logical CPUs capped at 32). `bench cpu-experts` does not: its default is physical/2 = 8 (`crates/breccia-kernels/src/cpu/bench.rs:114–115`, used at `bench.rs:239`). Every kernel bench gate in this document is `--threads 32`.
2. **Resolved residency.** `crates/breccia-engine/src/model.rs:863` prints `arena rule=list (--gpu-experts): N of M blocks resident` (the auto rule prints at `model.rs:844`). Host blocks per stream = 48 − N. The CLI two-stream default (`crates/breccia-cli/src/main.rs:1662–1664`) makes blocks [0,1,2,4,8,18,20,22,23] GPU-resident per stream, which is 39 host blocks per stream. BRIEF records 30 of 48 resident (18 host). The parent's live residency is not known here.
3. **Page-cache state.** Run with `--warm N` to fill the page cache before the window. Read the `cache` line across the window (`crates/breccia-cli/src/main.rs:2282`: GiB read from storage in the measured window, from the process I/O counters). Non-zero read_bytes per steady-state token means the lane is re-reading expert pages from storage. The phase-4 slow-window note (`docs/phase-4/01-forward-pass.md`, about line 1913) records the same effect.
4. **Expert bytes.** `graph.rs:3144` accumulates `expert_bytes`. The multi-stream bench writes it per stream in its JSON (`crates/breccia-cli/src/main.rs:1915`, `bench_token_multi`), and the single-stream bench prints MiB per token (`main.rs:2618`). Use one of those outputs; if neither was produced for the parent's run, derive bytes per token from the geometry in §4.

Pass rule: residency, read_bytes per step and lane ms/step at `--threads 32` appear in the same output, and the implied lane bandwidth below is consistent with a measured ceiling. If not, no kernel card is gated until the mismatch is explained.

Implied lane bandwidth (derived; uniform routing; 10 experts per block per token; about 1.75 MB per expert from doc 08 §2):

| host blocks per stream | MB per stream-token | GB per pair step | GB/s at 41 ms | GB/s at 43 ms | GB/s at 101 ms | full host working set |
|---|---:|---:|---:|---:|---:|---|
| 39 (CLI default) | 681 | 1.36 | 33.2 | 31.7 | 13.5 | 32.5 GiB, above the 30 GiB cap |
| 18 (BRIEF) | 314 | 0.63 | 15.3 | 14.6 | 6.2 | 15.0 GiB |

Doc 08 §6 measured 20.4 GB/s at 8 threads, and §6.2 (lines 281–299) gives 22.5–30.5 GB/s at 16 threads. The 39-block row implies 33 GB/s at 41 ms with 32 threads, above the 8-thread measurement. Either the residency is the 18-block case or the byte accounting differs. This is open question 1. At the 18-block case, the 101 ms end moves only 6.2 GB/s. That is far below any memory ceiling, so slow windows point at storage reads or CPU contention. Gate 0's read_bytes and thread count separate the two.

## 4. Breccia's host-lane design (today)

Geometry (doc 08 §2, line 55): gate and up matrices are [cols 2560, rows 640]; down is [cols 640, rows 2560]. A routed expert is about 1.75 MB (mixed types). Each block has 512 experts and each token routes 10 per block. There are 48 blocks, and doc 08 gives 838 MB per token at 48 host blocks (§6.1, line 267; table line 25).

Code path:

- **Decode call.** `graph.rs:3132–3144`. The lane takes the pool mutex, calls `moe_topk` (`crates/breccia-kernels/src/cpu/expert.rs:463`), and times it. It adds to `lane_s` (3140), adds to `lane_covered_s` only when `covered` (3141–3142), and adds to `expert_bytes` (3144).
- **Pool.** `ExpertPool` (`expert.rs:301`). `ExpertPool::new` (389) spawns `breccia-expert-N` threads (399) with no CPU affinity. The pool is `Arc<Mutex<…>>` (`graph.rs:442`), shared by both streams.
- **Per block, per token.** `moe_topk` copies the activation into an `Arc` (`expert.rs:479`) and runs two barrier-separated phases: phase 1, gate + up + SwiGLU (`run_phase` at 505); phase 2, down (`run_phase` at 539). That is two barriers per block.
- **Sharding.** `choose_shards` (371), with `MIN_SHARD_ROWS` = 16 (363) and `MAX_SHARDS` = 4 (365). Rows are split across shards and handed to persistent workers (`worker_loop`, 859). Output is per row, so the result does not depend on thread count (bit-identity test at `crates/breccia-kernels/tests/cpu_experts.rs:325`).
- **Per-row kernel.** `dot_row_with` (`crates/breccia-kernels/src/cpu/mod.rs:209`, takes `x: &[f32]`) → AVX2 dispatch (`crates/breccia-kernels/src/cpu/avx2.rs:189`) → fused decode and f32 dot per type, for example `iq3_xxs` (`avx2.rs:444`). The IQ3_XXS grid is a static 256-entry `[f32; 4]` table (`avx2.rs:135`, 4 KiB), read at 460–461. The files in `crates/breccia-kernels/src/cpu/` contain no int8, VNNI or AVX-512 kernel (grep this round).
- **Grouped kernels.** `MANY_MAX` = 8 (`avx2.rs:592`); `iq4_nl_many` (607) and `q2_0_many` (656). `many_supported` (`mod.rs:134`) returns true only for Iq4Nl and Q2_0. The grouped prefill path `moe_topk_chunk` (`expert.rs:724`) uses them for down.
- **Per-user work in prefill.** `one_user` (`expert.rs:651`) decodes gate and up again for every user that routes to an expert.
- **Prefill lane accounting.** `graph.rs:2410–2462`. Line 2462 `lane_covered_s += self.lane_s.min(0.0)` is always zero (conflict f).
- **Tests** (`crates/breccia-kernels/tests/cpu_experts.rs`): TOL = 1e-5 (line 32); `check_row` metric |got − want| / max(Σ|w·x|, 1e-30) (67–91); every type on synthetic rows (103); real expert blobs, which must cover seven types (148–244); expert FFN against an independent f64 pipeline (245); scalar vs AVX2 (301–324); pool bit identity at 2, 4 and 8 threads (325); grouped chunk vs per-token at 1e-6 with a bit rerun (371–460); forced scalar (461); a 1 GB/s throughput floor (490).

## 5. Approaches

Source links, pinned to the SHAs used in each section (line numbers from the pinned clones):

| Cite | Link |
|---|---|
| llama.cpp IQ3_XXS AVX2 vec_dot | https://github.com/ggml-org/llama.cpp/blob/8ca4060b7e30e76baaa5670ee1ec1b760b7f8552/ggml/src/ggml-cpu/arch/x86/quants.c#L3324 |
| llama.cpp q8_K quantiser | https://github.com/ggml-org/llama.cpp/blob/8ca4060b7e30e76baaa5670ee1ec1b760b7f8552/ggml/src/ggml-quants.c#L2769 |
| llama.cpp block_q8_K | https://github.com/ggml-org/llama.cpp/blob/8ca4060b7e30e76baaa5670ee1ec1b760b7f8552/ggml/src/ggml-common.h#L375 |
| llamafile mixmul type cases | https://github.com/Mozilla-Ocho/llamafile/blob/a3ccf098ed9c7c77687f5430b0143156d3bcc45a/llamafile/llamafile/tinyblas_cpu_mixmul.inc#L240 |
| llamafile VNNI guard | https://github.com/Mozilla-Ocho/llamafile/blob/a3ccf098ed9c7c77687f5430b0143156d3bcc45a/llamafile/llamafile/tinyblas_cpu.h#L1042 |
| llamafile Zen 4 mixmul wrapper | https://github.com/Mozilla-Ocho/llamafile/blob/a3ccf098ed9c7c77687f5430b0143156d3bcc45a/llamafile/llamafile/tinyblas_cpu_mixmul_amd_zen4.cpp#L1 |
| BitNet TL2 guard | https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/src/ggml-bitnet-lut.cpp#L16 |
| BitNet TL2 shuffle | https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/include/bitnet-lut-kernels.h#L166 |
| BitNet README claim | https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/README.md#L66 |
| T-MAC README claim | https://github.com/microsoft/T-MAC/blob/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee/README.md#L33 |
| ZenDNN grouped-matmul options | https://github.com/amd/ZenDNN/blob/55882a7d7237b4817d747102e1b3ce2239fe423f/benchdnn/doc/grp_matmul.md#L29 |

### 5.1 T-MAC

- Source: `github.com/microsoft/T-MAC` @ 7042f8f73330bd083bc1e4bc5ccb3f88a4904aee. README line 33 claims 20 tok/s on one core and 48 tok/s on four for a 3B BitNet model on a Surface Laptop 7; the line does not name the CPU. Line 50 claims 10–20% more with `-fa`. Both UNVERIFIED, ARM, not reproduced here.
- Code: the tuned deploy kernels are for aarch64 models (`deploy/tuned/`). `deploy/tuned/kernels.cc` branches on `__ARM_NEON` (line 5) and `__AVX2__` (line 7). Whether the x86 branch is performance-tuned was not checked.
- Mechanism: an activation-side table of partial sums over a weight group, indexed by weight bits, so lookups replace multiplies. The weights are ternary or bit-plane.
- Breccia mapping (derived): IQ3_XXS is a codebook (256-entry grid plus sign patterns). An activation-side table over it needs 256 grid × 16 sign patterns per 4-element group. A 2560-column row has 640 groups, so 640 × 256 × 16 = 2,621,440 entries: 10.0 MiB as f32, 5.0 MiB as i16, 2.5 MiB as i8. The CPU has 1 MiB of L2 per core (16 MiB over 16 instances, one per physical core). The f32 table is ten times that, and the i8 table is still 2.5 times it. The Breccia kernel's current hot data (x, 10 KiB; grid, 4 KiB) fits in L2 with room to spare.
- Verdict: rejected. Both T-MAC and BitNet need requantised weights (doc 24 territory). IQ3_XXS cannot be used as it is.

### 5.2 llamafile tinyBLAS (sgemm) and mixmul

- Source: `github.com/Mozilla-Ocho/llamafile` @ a3ccf098ed9c7c77687f5430b0143156d3bcc45a.
- mixmul (`llamafile/llamafile/tinyblas_cpu_mixmul.inc`): a type gate for Q4_0 (line 240; thought activations F32 or Q8_0), Q1_0 (253) and Q8_0 (269). `mixmat()` (288) loops over experts (298). `build_row_pointers` (313) gathers each expert's (token, slot) rows. `quantize_thought` (338) builds the activation side. An expert's weights are read once per batch of rows. This is a grouping pattern, not a kernel, and it has no IQ types.
- tinyBLAS int8 dot products: `tinyblas_cpu.h:1042` and `:1186` are guarded by `__AVXVNNI__`, or by `__AVX512VNNI__ && __AVX512VL__`.
- Zen 4 build variant: `llamafile/llamafile/tinyblas_cpu_mixmul_amd_zen4.cpp`, `..._amd_avx512f.cpp` and `..._amd_avxvnni.cpp` are four-line wrappers that include the same `tinyblas_cpu_mixmul.inc` under a different function name. That file has F32, BF16, F16, Q4_0, Q1_0 and Q8_0 cases and no IQ type, so llamafile's Zen 4 path is compile-flag selection, not an AVX-512 IQ kernel.
- Verdict: copy the grouping for prefill (card 5). The sgemm kernels do not apply to IQ types.

### 5.3 llama.cpp IQ3_XXS vec_dot

- Source: `github.com/ggml-org/llama.cpp` @ 8ca4060b7e30e76baaa5670ee1ec1b760b7f8552. The x86 `ggml_vec_dot_iq3_xxs_q8_K` is at `ggml/src/ggml-cpu/arch/x86/quants.c:3324–3446`. The generic version is at `ggml/src/ggml-cpu/quants.c:1050`.
- Activations are quantised to `block_q8_K` (`ggml/src/ggml-common.h:375`) by `quantize_row_q8_K_ref` (`ggml/src/ggml-quants.c:2769`): scale = −127 / max|x| (2791), nearest rounding (2793), per 256-element block.
- Inside 3324–3446 the AVX2 path uses `_mm256_sign_epi8`, `_mm256_maddubs_epi16` and `_mm256_madd_epi16` on int8 activations, plus `_mm256_fmadd_ps`, under an `__AVX2__` guard (line 3337). The range has no `_mm512` or `__AVX512` tokens. The Zen 4 AVX-512 datapath is not used, and any benefit from it is UNVERIFIED.
- Implication: llama.cpp's IQ3_XXS is already an int8-activation kernel. Its activation quantisation is the single-limb form of §6.3, so Breccia's 1e-5 per-row bar is stricter than the scheme llama.cpp uses.

### 5.4 AMD ZenDNN low-bit MoE

- Source: `github.com/amd/ZenDNN` @ 55882a7d7237b4817d747102e1b3ce2239fe423f, `benchdnn/doc/grp_matmul.md`.
- Grouped-matmul algorithms (lines 29–31): `ZENDNNL_GRP_MATMUL_ALGO` 2 is a flat CCD M-tile, 3 a flat CCD N-tile, 4 a W8A8 fused MoE. Lines 34–35 gate the in-house AVX-512 microkernels and their DQ-INT8 family.
- The DQ-INT8 path quantises activations symmetrically per token or per group, `s8:s8:{bf16,f32}` (the dynamic_quant, compute_dt and group_size parameters).
- Weights are int8, not 2–3 bit. Use: per-CCD tiling as a reference for card 6, and grouped-MoE scheduling. There is no low-bit kernel to port.

### 5.5 bitnet.cpp TL1 and TL2

- Source: `github.com/microsoft/BitNet` @ 0b341e582afbf9e1011f24744b554c96a3477eb5. `src/ggml-bitnet-lut.cpp:16` selects the LUT path under `GGML_BITNET_ARM_TL1 || GGML_BITNET_X86_TL2`. Line 20 is the ARM TL1 block and line 136 the x86 TL2 block.
- TL2 builds ternary-group index tables and uses `_mm256_shuffle_epi8` (`include/bitnet-lut-kernels.h:166` and `:243`). `utils/codegen_tl2.py` is the TL2 codegen script in that repo; the TL2 preset headers sit under `preset_kernels/`.
- README line 66 claims up to 6.17× on x86 (5.07× on ARM) against full-precision models, for the BitNet-b1.58-2B-4T model named at line 62. UNVERIFIED; the line names no CPU and no baseline configuration.
- Verdict: rejected for the same reason as T-MAC. The weights are ternary, so TL2 means requantising IQ3_XXS experts, which is not lossless.

## 6. Breccia angles

### 6.1 Batching the two streams' host work

- Decode: each stream adds one token per step. Two independent top-10 sets out of 512 share about 0.2 experts per block (10 × 10 / 512, uniform routing, derived). A union batch per block therefore saves about 1% of expert decodes. The real route histogram is not measured, since decode has no per-token expert-id trace (doc 24, lines 23 and 153). Rejected unless a trace shows correlated routing.
- Two pools, one per stream at 32 threads each, would oversubscribe 32 logical CPUs. Rejected.
- Prefill: the valid batching card is gate/up `_many` for IQ2 and IQ3 (card 5). `one_user` re-decodes gate and up for each user. A `_many` kernel decodes once for all users of an expert. That moves prefill lane time, not the decode metric.

### 6.2 Per-CCD pinning

- Topology (kernel cache lists, this host): CCD0 is logical CPUs 0–7 and 16–23. CCD1 is 8–15 and 24–31. Each CCD has 16 logical CPUs, so the per-CCD probe is 16 threads on one CCD, not 32.
- Pool threads are not pinned (`expert.rs:399`). Doc 08 already tested placement under the old dispatch (§6.3, lines 300–316): 8 threads on CPUs 0–7 (CCD0 physical cores), on 8–15, or spread over both CCDs all measured 12.8–13.0 GB/s, and the doc concluded that packing, not placement, is the limit. Doc 08 §7 item 6 (line 346) says placement does not matter at these counts and that the lane has no affinity control (conflict b).
- Probe (card 1): `taskset` to one CCD at `--threads 16`, against unpinned `--threads 16`, in the same window. If pinned is at least 10% faster, one CCD's share is the cap. This is not doc 08's test: it runs 16 threads on one CCD including SMT siblings, under the current dispatch. The prior from doc 08 is negative.

### 6.3 VNNI int8 with per-block q8 activations

Accuracy. The GPU's int8 dp4a path failed accuracy: 6.2e-3 per element, logits 0.51 off, greedy changes. BRIEF attributes that to the q8_1 activation, not the dot. The figures below are L1-relative per row, |got − want| / Σ|w·x|, the metric of `tests/cpu_experts.rs:67–91`. That is a different metric from the GPU's per-element figure, so the two must not be compared.

Method (synthetic, seed 1234, K = 2560, 4,000 rows, Gaussian weights). Activations are quantised per 256-element block, q8_K style: scale = −127 / max|x|, nearest rounding, dequantised by division. Two limbs: x ≈ s1·q1 + s2·q2, where q2 quantises the residual with its own block scale. The scripts are in the round's scratch directory and are not committed.

| activation | x distribution | p50 | p99 | max | rows > 1e-5 |
|---|---|---:|---:|---:|---:|
| 1 limb, block 256 | Gaussian | 1.47e-4 | 5.52e-4 | 7.97e-4 | 95.9% |
| 1 limb, block 32 | Gaussian | 1.13e-4 | 4.12e-4 | 6.74e-4 | 94.6% |
| 2 limbs, block 256 | Gaussian | 5.68e-7 | 2.18e-6 | 3.29e-6 | 0.0% |
| 1 limb, block 256 | 1% of entries ×10 | 5.64e-4 | 2.29e-3 | 3.09e-3 | 98.9% |
| 1 limb, block 32 | 1% of entries ×10 | 2.55e-4 | 1.00e-3 | 1.45e-3 | 98.0% |
| 2 limbs, block 256 | 1% of entries ×10 | 2.14e-6 | 9.18e-6 | 1.34e-5 | 0.4% |

Reading:

- One limb fails the 1e-5 bar by construction, on 94–99% of rows. It cannot be a kernel-level gate.
- Two limbs pass on Gaussian rows and fail 0.4% of outlier-heavy rows. That is not a pass for real activations, which are not Gaussian and were not measured.
- Two limbs cost two int8 passes per block, about twice the int8 MACs. Not timed.

Breccia shape. x is quantised once per `moe_topk` call (the copy site is `expert.rs:479`) and shared by all rows and shards. The AVX2 decode of grid and signs is unchanged. Only the dot changes.

Host CPU. `/proc/cpuinfo` shows `avx512vl`, `avx512_vnni` and `avx512_bf16` on the first logical CPU, and `avx512_vnni` in all 32 blocks; `avx_vnni` (VEX) is absent. Doc 08 §7 item 1 (line 319) names avx512f, bw, vl and vnni on Zen 4. The tinyBLAS guard in §5.2 is the AVX512-VNNI + VL form, which this CPU meets.

Risk. The lane's output is added to the GPU's output, so a CPU int8 error sits in the same accuracy class as the GPU's. The gate is end-to-end greedy parity plus the kernel test, not the kernel test alone.

### 6.4 L2 and L3 prefetch

- Per-core L2 is 1 MiB, one instance per physical core. L3 is per CCD (the cache lists match the CCD sets above; lscpu reports 64 MiB across 2 instances).
- Working sizes (derived from GGML block sizes: IQ3_XXS is 98 B per 256 weights, IQ4_NL 18 B per 32): an IQ3_XXS gate or up matrix is 640 × 10 × 98 B = 627 KB. A mixed-type routed expert is about 1.75 MB (doc 08 §2).
- Within a block, the 10 routed experts are known before phase 1. They are the router output passed to `moe_topk` (`expert.rs:463`). Prefetching expert e+1 while e computes needs no predictor. This is card 3.
- Across blocks, the next block's experts depend on the next router output. That needs a predictor. Doc 23 found no prefetch that pays without a measurement. Doc 08 §7 item 5 (line 341) names two cross-block ideas outside this lane: overlap block N's down phase with block N+1's gate and up, and prefetch the next token's top-10 pages. Both need a dependency or route analysis that this round did not do.
- Expert slices are mmap windows (`tests/cpu_experts.rs:531`). A page that is not in the page cache is an I/O stall, not a cache miss. CPU prefetch does not help it. Page-cache hints are doc 29's territory.

## 7. Ranked cards

Each card gives the change, the code site, the correctness test and the lane gate. Conversions use §2 (fence 44 ms, R 18 ms, base lane 41 ms at 32 threads). Every gate is `--threads 32` and reports Gate 0's residency and read_bytes.

**Card 0 (precondition, no code): Gate 0 (§3).** Gate: residency, read_bytes per step and lane ms/step at `--threads 32` in one output, with the implied bandwidth inside the measured ceiling. If steady-state read_bytes is non-zero, the first card is page-cache residency, not a kernel.

**Card 1 (zero code): per-CCD probe.** Change: none. Run `taskset` to CCD0 (`0-7,16-23`) at `--threads 16`, and unpinned at `--threads 16`, in one window, with Gate 0's output. Gate: pinned lane at least 10% lower is a pass, and card 6 becomes live. Otherwise card 6 is dropped. This is not a decode gate by itself.

**Card 2: AVX-512 f32 path for `iq3_xxs` and the IQ2 kernels (no int8).**
- Change: a `#[target_feature(enable = "avx512f")]` variant of the decode-and-dot, selected in `dispatch` behind runtime detection (`avx2_available`, `mod.rs:168`).
- Code: `avx2.rs:189` (dispatch), `avx2.rs:444` (`iq3_xxs`), `mod.rs:141` (Path enum).
- Correctness: f64 tests (`tests/cpu_experts.rs:103–134`, `148–244`) at TOL 1e-5, and scalar vs AVX-512 within 1e-5 (the pattern of 301–324). Not required to be bit-identical with AVX2, since the sum order changes. The pool bit identity (325) must still hold for the path.
- Gate: lane ≤ 36 ms at `--threads 32` (5 ms cut, about +1.0 tok/s from the 41 ms base). Gain estimate: doc 08 §7 item 1 (line 319) says 10–25%, UNVERIFIED, not measured.

**Card 3: in-block expert prefetch (no numeric change).**
- Change: in `moe_topk` (`expert.rs:463`), before phase 1 (`run_phase` at 505), prefetch the next expert's gate and up rows. For phase 2, prefetch the down rows during the tail of phase 1.
- Code: `expert.rs:463–548`; `_mm_prefetch` in `avx2.rs`, inside the existing unsafe boundary.
- Correctness: bit identity with the current path (325), and the real-blob test (148–244) unchanged.
- Gate: lane ≤ 38 ms at `--threads 32` (3 ms cut, about +0.6 tok/s). If Gate 0 shows non-zero read_bytes, the measured gain is I/O and this card is void.

**Card 4: two-limb VNNI int8 for IQ3_XXS and IQ2 gate/up (end-to-end gated).**
- Change: a new Path variant. x is quantised to two int8 limbs once in `moe_topk`. VNNI dot kernels use `#[target_feature(enable = "avx512vnni,avx512vl")]` behind runtime detection. Decode is unchanged. Down stays f32.
- Code: `mod.rs:141` (Path), `mod.rs:209` (`dot_row_with`), `avx2.rs:189` (dispatch), new kernels beside `avx2.rs:444`, `expert.rs:479` (quantise once).
- Correctness, in order: (a) the 1e-5 real-blob test (148–244) must pass on every row; (b) an outlier-activation test is added; (c) pool bit identity across thread counts (325); (d) end-to-end greedy parity: N fixed prompts, token-for-token against the f32 path, with the logits max abs difference reported. If (a) fails on any row, the gate is (d) with a stated tolerance, not the 1e-5 bar.
- Gate: lane ≤ 31 ms at `--threads 32` (10 ms cut, about +2.1 tok/s from the 41 ms base), and (d) passes. The ceiling is unmeasured. The first measurement is the int8-to-f32 throughput ratio on this CPU, as a kernel micro-bench at `--threads 32`.

**Card 5: prefill gate/up `_many` for IQ2 and IQ3 (decode unaffected).**
- Change: add `iq3_xxs_many` and `iq2_*_many` beside `iq4_nl_many` (`avx2.rs:607`), and extend `many_supported` (`mod.rs:134`). `one_user` (`expert.rs:651`) becomes a per-expert batch in `moe_topk_chunk` (`expert.rs:724`) for gate and up.
- Correctness: grouped chunk vs per-token at 1e-6 with a bit rerun (371–460); pool identity (325).
- Gate: prefill lane ms per chunk at `--threads 32`, down by the measured gate+up re-decode share. Not a decode gate.

**Card 6 (only if card 1 passes; prior negative per doc 08 §6.3): per-CCD pinned pool.**
- Change: `ExpertPool::new` (`expert.rs:389–412`) pins each worker thread (spawn at 399) to its CCD with `sched_setaffinity` (unsafe, needs a libc call). Doc 08 §7 item 6 says the lane has none. `choose_shards` (371) splits rows per CCD.
- Correctness: pool bit identity at 2, 4 and 8 threads (325), plus a 32-thread run this card adds.
- Gate: lane ≤ 36 ms at `--threads 32` (5 ms cut, about +1.0 tok/s).

**Card 7: two-row register blocking in the IQ decode-and-dot (doc 08 §7 item 2, lines 324–326).**
- Change: process two output rows per activation-vector load in the `iq3_xxs` kernel (`avx2.rs:444`) and its IQ2 siblings. Doc 08 §7 item 2 gives two wordings for the gain (conflict k). This card uses its "3 loads/unit → 2.5" figure, about 17% (derived, not measured here).
- Correctness: f64 tests (`tests/cpu_experts.rs:103–134`, `148–244`) at TOL 1e-5; pool bit identity (325). Per-row summation order can stay the same, so bit identity with the current path is the target and the test decides.
- Gate: lane ≤ 38 ms at `--threads 32` (3 ms cut, about +0.6 tok/s from the 41 ms base).

Ordering notes: cards 2, 4 and 7 all touch the dot site; land them in the order 2, 7, 4 as each passes. Cards 3 and 6 both touch pool code.

## 8. Rejected

1. **Single-limb int8 as a kernel card.** 94–99% of rows over 1e-5 on synthetic data (§6.3). Kept only as the end-to-end path of card 4.
2. **T-MAC LUT over IQ3_XXS.** A 10 MiB f32 table against 1 MiB L2 (§5.1), and it needs requantised weights.
3. **BitNet TL1/TL2.** Ternary weights (§5.5). Needs requantisation. README speedups UNVERIFIED.
4. **ZenDNN int8 grouped MoE.** Int8 weights, not low-bit (§5.4). Kept as a tiling reference only.
5. **llamafile mixmul as a kernel.** A grouping pattern with no IQ types (§5.2). Per-expert grouping already exists in `moe_topk_chunk` (`expert.rs:724`).
6. **llamafile tinyBLAS sgemm for IQ types.** Q4_0, Q8_0 and Q1_0 only (§5.2).
7. **Cross-stream union batch at decode.** About 0.2 shared experts per block at uniform routing (§6.1).
8. **A second pool for the other stream.** Two pools of 32 threads on 32 logical CPUs (§6.1).
9. **Activation copy and allocation removal** (`expert.rs:479`, and the per-call `vec![0f32; k * n_ff]` at `expert.rs:506`). About a 10 KB copy and a 25.6 KB allocation per call, each in the single-digit-microsecond range (estimate, unmeasured). Negligible against 41 ms. Reopen only if Gate 0 shows allocator time.
10. **Cross-block expert prefetch.** Needs a router predictor, and none has been measured (§6.4).
11. **More than 32 threads.** The logical CPU count is 32, and `graph.rs:4155` clamps to it.

## 9. Conflicts between sources and the repo

a. **Thread default.** `bench.rs:114–115` and `main.rs:277` say physical/2 (= 8 here; the `main.rs` help also says capped at 16). The decode default at `graph.rs:4155` is logical capped at 32, which is the §8y fix. `multi.rs:25` says 16 threads. `bench cpu-experts` uses the physical/2 default (`bench.rs:239`).
b. **Affinity.** Doc 08 §6.3 (lines 300–316) tested 8-thread placement on CPUs 0–7, on 8–15 and spread over both CCDs; all three measured 12.8–13.0 GB/s under the old dispatch, and the doc concluded packing is the limit. §7 item 6 (line 346) says placement does not matter at these counts and that the lane has no affinity control. Card 6 runs against that prior, so card 1 comes first.
c. **Unsafe boundary.** `mod.rs:37–39` says unsafe appears only in `avx2.rs`. `mod.rs:105` calls the AVX2 kernels from an `unsafe` block.
d. **Lane units.** "Per step" means per pair step, the sum over both streams (`multi.rs:106–108`). BRIEF's single-stream lane (about 13 ms at 18 blocks) is per stream. The parent's 43–101 ms is per pair step.
e. **Expert types.** The parent's framing is IQ3_XXS. Doc 08 §2's table has gate and up as IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S, and down as IQ4_NL or Q2_0. The test asserts all seven (`tests/cpu_experts.rs:148–244`). The type mix comes from doc 08 and was not checked against the GGUF header.
f. **Covered time.** `graph.rs:2462` `lane_covered_s += self.lane_s.min(0.0)` is always zero, though its comment says "set by the driver when relevant". The decode path adds `lane` only when `covered` (3141–3142). Prefill accounting does not match its comment.
g. **Residency.** The CLI default (`main.rs:1662–1664`) gives 39 host blocks per stream. BRIEF gives 18. Doc 24 (line 80) uses 18 streamed blocks. The parent's live residency is unknown. Gate 0 settles it.
h. **Fence.** §8y's fence varies with thread count (32.3, 44.7, 39.5, 36.6 ms). The parent reports 44–53 ms. The step model holds fence as an input.
i. **Doc 08 bandwidth.** Doc 08 §6 (20.4 GB/s at 8 threads) is inconsistent with the 39-block row in §3 (33 GB/s at 41 ms, 32 threads), unless the byte accounting differs.
j. **Thread recommendation.** Doc 08 §6.3 (lines 300–316) recommends `--threads 12` on this CPU class. Phase-4 §8y found 32 logical CPUs better (lane 41 against 55 ms at 16 threads), and the decode default now follows it (`graph.rs:4155`). Doc 08 predates that.
k. **Two-row blocking gain.** Doc 08 §7 item 2 (lines 325–326) says two-row blocking "halves activation-vector loads per unit" and also "3 loads/unit → 2.5". Halving would be 3 → 1.5, so this doc uses 2.5 (about 17%, derived). Unmeasured.

## 10. Open questions

1. Live residency: which blocks are GPU-resident in the parent's run, and is the CLI default in use? (Gate 0.)
2. Byte accounting: `expert_bytes` is printed by the bench outputs in §3 step 4. Was either produced for the parent's run? The 33 GB/s implied at 39 host blocks is above the doc 08 ceiling.
3. Page cache: is read_bytes non-zero in steady state? At 18 host blocks the 101 ms end implies 6.2 GB/s, far below memory bandwidth, so storage or CPU contention would explain it.
4. Decode route histogram: no per-token trace exists (doc 24, lines 23 and 153). Needed for the batching verdict and for card 3's benefit.
5. Two-limb on real activations and real expert blobs: not run.
6. Int8-to-f32 throughput ratio on this CPU: not measured. Card 4's ceiling depends on it.
7. Zen 4 AVX-512 throughput for the table lookups: UNVERIFIED. Doc 08 §7 item 1 estimates 10–25%.
8. Barrier cost: two barriers per block per stream, 96 per stream-step, are not timed.
9. Whether the GPU fence stays flat as the lane shrinks. The §8y fence moves with thread count.

## 11. What was not checked

- No runs on the live pair. No Breccia build or bench. No timings.
- No GGUF header read. The type mix comes from doc 08.
- External speed claims (T-MAC, BitNet, llama.cpp, ZenDNN) are UNVERIFIED. Code citations are from pinned clones only.
- Zen 4 datapath claims and the L3 per-instance size were not measured.
- Synthetic activation results are not real-activation results.
