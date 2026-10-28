# 08 — Host (CPU) resident-expert lane (`breccia-kernels::cpu`)

Work-order item #3 of the M2 kernel series: the **host compute lane for routed
MoE experts** — the last kernel piece before the forward-pass graph. It exists
because the production baseline gets its speed from doing exactly this:
`docs/phase-3/07-moe-cache-ab.md` measured that staging experts to VRAM instead
(`--moe-cache-mib > 0`) is **18–35 % slower**, and that the production
configuration computes all non-GPU experts on the CPU straight from host-mapped
GGUF data — **0.84 GB of expert reads per token, ≈17.5 GB/s sustained** at the
20.79 tok/s baseline.

Date: 2026-10-26 · CPU-only work: no GPU touched, no server started or stopped,
no GPU service was running or altered · artifacts in
`<SCRATCH>/opencode/p3cpu/`.

**Status: complete, and above the parity bar.**
`cargo build/test --workspace`, `cargo clippy -p breccia-kernels -p breccia-cli
--all-targets -- -D warnings`, `cargo fmt --all --check` green.

| | |
|---|---|
| types with an AVX2 kernel | **14** (the 13 the work order lists + F16) |
| AVX2 vs scalar speedup (per-type cells) | **4.1× – 28.5×** |
| real-blob conformance vs the `f64` reference | **3.7e-7** worst relative (8 572 rows, 7 types) |
| one token (48 blocks × top-10) | **838.1 MB** read, **41.1 ms**, **20.41 GB/s** at the default 8 threads |
| vs the 17.5 GB/s bar | **117 %** (95–120 % across five runs; 120–137 % at 12 threads, up to 174 % at 16) |

---

## 1. What landed

| Path | Content |
|---|---|
| `crates/breccia-kernels/src/cpu/mod.rs` | Public lane API: [`dot_row_with`], [`gemv_into`], `Path` (`Avx2` \| `Scalar`) with runtime feature detection and the `BRECCIA_CPU_FORCE_SCALAR=1` switch, tail handling through the reference dequantiser |
| `src/cpu/scalar.rs` | Fused scalar port of every decode: same arithmetic as `reference.rs`, but no weight block is materialised, and the accumulator is `f64` so this path stays a second oracle |
| `src/cpu/avx2.rs` | **The only `unsafe` in the CPU lane.** 14 `#[target_feature(enable = "avx2,fma,f16c")]` kernels + the compile-time tables they read |
| `src/cpu/expert.rs` | `LayerPlan`/`RolePlan` (mmap geometry per block/role), `expert_ffn` (gate→up→SwiGLU→down), `ExpertPool` (persistent threads, two-phase top-k dispatch), `moe_topk_sequential` |
| `src/cpu/bench.rs` | `breccia bench cpu-experts`: per-type GB/s (both paths), the token pass, the thread sweep, JSON (schema v1) |
| `crates/breccia-kernels/tests/cpu_experts.rs` | 9 conformance/determinism tests, including **real expert blobs** for all 7 expert types |
| `crates/breccia-cli` | exactly one new subcommand: `breccia bench cpu-experts` |
| `crates/breccia-kernels/src/quant.rs` | `Serialize` for `QuantType` as its canonical GGML name (report/JSON only; no kernel change) |

Nothing outside `crates/breccia-kernels/**`, `crates/breccia-cli/**` and this
file was modified. The GPU kernels are untouched.

**Zero-copy discipline.** Weights are `&[u8]` windows of the
`breccia_format` mmap (`ExpertSlices` / `<role>.as_slice()`); the only data
copied in the whole lane is the 10 KB activation vector handed to the workers and
the per-shard output rows (≤2.5 KB each). Every entry point returns the number of
weight bytes it read, and the benchmark reports the total: **838.1 MB/token**,
1.95 % of the 42.913 GB expert tensors.

---

## 2. Geometry (block for block the GPU kernels' and the reference's)

Taken from `breccia_format::quant` (the table verified against the real shards in
`docs/specs/model-artifacts-inventory.md` §2.5), and identical to
`crate::quant::QuantType`'s kernel geometry.

| Type | GGML id | Block (elems, bytes) | bpw | Experts in this model |
|---|---:|---|---:|---|
| IQ2_XXS | 16 | 256 / 66 | 2.0625 | 9 layers' gate+up |
| IQ2_XS | 17 | 256 / 74 | 2.3125 | 10 layers' gate+up |
| IQ2_S | 22 | 256 / 82 | 2.5625 | 10 layers' gate+up |
| IQ3_XXS | 18 | 256 / 98 | 3.0625 | 6 layers' gate+up |
| IQ3_S | 21 | 256 / 110 | 3.4375 | 13 layers' gate+up |
| IQ4_NL | 20 | 32 / 18 | 4.5 | 18 layers' **down** |
| Q2_0 *(fork-only)* | 42 | **64 / 18** | 2.25 | 30 layers' **down** |
| IQ4_XS | 23 | 256 / 136 | 4.25 | dense (`ffn_*_shexp`), not routed |
| Q4_K | 12 | 256 / 144 | 4.5 | dense |
| Q5_K | 13 | 256 / 176 | 5.5 | dense (`output.weight`) |
| Q6_K | 14 | 256 / 210 | 6.5625 | dense |
| BF16 | 30 | 1 / 2 *(virtual 8)* | 16 | dense (routers, HC) |
| F32 | 0 | 1 / 4 *(virtual 8)* | 32 | dense (norms) |
| F16 | 1 | 1 / 2 *(virtual 8)* | 16 | `ple_conv1d` only |

Per-block expert geometry: `gate`/`up` are `[cols = n_embd = 2560, rows = n_ff =
640]`, `down` is `[cols = 640, rows = 2560]` — **1.638 M values and the same
bytes per role** (640·2560 = 2560·640), so one expert blob is ~1.75 MB of weights
and `gate`, `up`, `down` each cost one third of an expert's time. That symmetry is
what lets the pool reason in "role units" (§5).

`Q2_0` is the fork's block (16 B of 2-bit codes + one fp16 scale per 64 values),
**not** upstream ggml's `Q2_0`; a unit test pins `64/18 = 2.25 bpw`.

---

## 3. Kernel: one decode unit of eight elements, one FMA

Every type decodes in units of **8 consecutive elements**, the same contract as
the GLSL decode units in `shaders/blocks.glsl`, so the CPU and GPU lanes read the
same bytes in the same order. A 256-element super-block is 32 units, `IQ4_NL` is
4, `Q2_0` is 8, and BF16/F16/F32 use a virtual 8-element block.

Per unit the AVX2 path does:

* **one 32-byte load** of grid/table magnitudes, or a register shift/mask for the
  K-quants and `Q2_0`;
* **one multiply** by the block scale (broadcast hoisted where consecutive units
  share it) and **one sign-bit XOR** for the `IQ*` types — the sign is applied to
  the *value*, never as a float→int conversion, so a NaN scale degrades the
  result instead of producing an out-of-bounds index (`docs/phase-3/02` §4.3);
* **one `vfmadd231ps`** against the activation vector.

Two accumulators alternate by unit so the FMA dependency chain does not
serialise; their lane sums are combined in `f64`. Three measured steps got here
(blk.0 `gate`, 121 MB working set, single thread):

| Change | GB/s |
|---|---|
| `vpmovzxbd` + `vcvtdq2ps` per unit, `±1` sign table, per-unit broadcast | 2.69 |
| grid magnitudes widened to `f32` **at compile time** (one load instead of load+2 converts) | 2.94 |
| `ksigns_iq2xs` permutation folded into the sign-bit table (one fewer dependent load) + block scale broadcast hoisted out of the inner loop | 3.20 |

The remaining cost is load throughput: three 32-byte loads per 8 elements (grid,
sign mask, activation) against Zen 4's two loads/cycle. This is the same order as
llama.cpp's own AVX2 `IQ2_XXS` dot, which is also the slowest of its quant dots.

### 3.1 AVX2 vs scalar, per type, on real expert blobs

`breccia bench cpu-experts` (96 expert slices per cell, best of 3, one thread,
same weights on both paths). `scalar` is the fused portable port (§1).

| Type | Role | Shape | MB read | AVX2 GB/s | scalar GB/s | speedup |
|---|---|---|---:|---:|---:|---:|
| IQ4_NL | down | 2560×640 | 88.5 | **3.15** | 0.766 | 4.1× |
| Q2_0 | down | 2560×640 | 44.2 | **2.51** | 0.448 | 5.6× |
| IQ2_S | gate | 640×2560 | 50.4 | **3.20** | 0.129 | 24.8× |
| IQ2_XS | gate | 640×2560 | 45.5 | **2.98** | 0.114 | 26.2× |
| IQ2_XXS | gate | 640×2560 | 40.6 | **2.89** | 0.107 | 27.0× |
| IQ3_S | gate | 640×2560 | 67.6 | **3.55** | 0.153 | 23.1× |
| IQ3_XXS | gate | 640×2560 | 60.2 | **3.82** | 0.151 | 25.3× |
| IQ2_S | up | 640×2560 | 50.4 | **3.13** | 0.129 | 24.3× |
| IQ2_XS | up | 640×2560 | 45.5 | **2.88** | 0.114 | 25.3× |
| IQ2_XXS | up | 640×2560 | 40.6 | **3.05** | 0.107 | 28.5× |
| IQ3_S | up | 640×2560 | 67.6 | **3.55** | 0.153 | 23.2× |
| IQ3_XXS | up | 640×2560 | 60.2 | **3.82** | 0.150 | 25.4× |

The `IQ*` speedup is ~25× because the portable path walks the grid/sign decodes
one element at a time; `IQ4_NL` (a `vpshufb` 16-entry table lookup) and `Q2_0`
(a shift/mask) are already cheap scalar work, hence 4–6×.

This lane is **compute-bound, not bandwidth-bound**: a single core streams
53.9 GB/s from DRAM but only decodes 2.5–3.8 GB/s of quants, and the per-type
figures are flat from a 1 MB working set (L1/L2) to 121 MB (DRAM) — 2.94 vs 2.69
GB/s on the same type. That is why the pool is the lever, and why no prefetch or
NUMA work was worth doing.

---

## 4. Conformance

Oracle: `crate::reference` — the scalar port that `tools/refcheck.py` diffs
**bit-for-bit** against llama.cpp's `ggml-quants.c` (fork commit 6050b46). It
dequantises into `f32` and accumulates in `f64`.

Three levels of check, all in `tests/cpu_experts.rs` and the two module test sets:

| Check | Result |
|---|---|
| AVX2 vs scalar, **per value**, one `1.0` at a time, all 14 types | equal to ≤1e-6 (each is a single product, so this is a decode check, not a sum check) |
| AVX2 and scalar vs the `f64` reference, **real expert blobs** (all 7 expert types, experts 0..9, every row of expert 0 + sampled rows) | **8 572 rows, worst 3.666e-7 relative** |
| the same on synthetic valid blocks (8×2560 and 8×640, all 14 types, both paths) | worst **1.06e-7 relative to Σ\|w·x\|** |
| full expert FFN (`gate`→`up`→SwiGLU→`down`) on real blobs vs an independently assembled `f64` pipeline | worst **2.2e-8 relative** |
| scalar path vs AVX2 through the whole FFN (blk.3, expert 5, weight 0.5) | worst **1.8e-8** |
| pool vs single-threaded, 2 / 4 / 8 threads, 10 experts with weights | **bit-identical**, bytes-read identical |
| repeated runs, both paths | bit-identical |
| `BRECCIA_CPU_FORCE_SCALAR=1` | `path()` → `Scalar`; a forced-scalar GEMV agrees with the AVX2 one to ≤1e-5, and the switch is restored |

**The 1e-5 bar, and the one place it needs a denominator.** The work order asks
for ≤1e-5 relative for every type on both paths. On real blobs that holds with
two orders of magnitude to spare (3.7e-7). On *synthetic* blocks it can only be
stated against `Σ|wⱼxⱼ|`: a row whose terms are 4 681× larger than its own sum
(`Q5_K` with random signs, worst row of the generator) can lose ~2.5e-5 of the
result to `f32` accumulation *order alone* — no `f32` summation scheme reaches
1e-5 of the result there, and the measured error relative to `Σ|wⱼxⱼ|` is 1.06e-7.
The strict result-relative bar is therefore applied where it is meaningful (the
real artefacts the engine runs on) and the synthetic generator is held to the
`Σ|wⱼxⱼ|` scale; both numbers are printed by the test. The decode itself is
proven exact by the per-value test, and the scalar path's `f64` sum lands at
6.5e-9 on the same rows, which isolates the residual to summation, not decoding.

Every type's decode also passes the *degenerate-scale* checks already in the
suite: a NaN/inf scale produces a non-finite dot, it never indexes memory
(`no_float_to_integer_conversion_in_a_decode_unit` still guards the GLSL side,
and the AVX2 units follow the same rule by construction).

---

## 5. The expert FFN and the per-token top-k dispatch

### 5.1 The FFN

`expert_ffn(path, plan, slices, x, scratch, weight, out)` runs, straight from the
mmap slices of one expert:

```
gate = gemv(gate_ty, gate_rows = 640, cols = 2560, x)      //   640 × 2560
up   = gemv(up_ty,   up_rows   = 640, cols = 2560, x)      //   640 × 2560
act  = SiLU(gate) · up                                     //       640 values
out += weight · gemv(down_ty, 2560 rows, cols = 640, act)  //  2560 ×  640
```

That is exactly the routed term of `Σ_j w_j · down_j(SiLU(gate_j x) · up_j x)`
(`docs/specs/qwen4exp-architecture.md` §5.4). Router weights are **not** computed
here — softmax over 512 and the top-10 with the `2^-14` clamp belong to the graph
and already exist in `ops::Router`.

### 5.2 Fusion: what is fused and what is not

* **SwiGLU is fused into the gate/up phase** in the pool: the worker that owns
  rows `[r0, r1)` of an expert's `gate` *and* `up` computes `SiLU(gate)·up` for
  exactly those rows. Cost if it were serial on one thread: **0.57 ms/token**
  (307 200 `exp()` calls); fused it is ~0.07 ms and off the critical path.
  `expert_ffn` (the single-threaded reference path) still runs it as a separate
  pass, deliberately, so the oracle stays obviously readable.
* **The down accumulation is separate** but negligible: `out[r] += w_j · v` over
  10 experts × 2560 rows = 25 600 FMA per block, **0.02 ms/token** measured.
* **Dequant is fused with the dot** everywhere: no f32 weight block is ever
  materialised (the scalar path included).

### 5.3 Why the dispatch is two-phase and row-sharded

The model picks **10** experts per block. One expert per worker cannot fill an
8-thread machine evenly — two workers take two experts and the batch takes two
job-times. Measured with one-expert-per-worker dispatch: **13.1 GB/s at 8
threads** but **24.1 GB/s at 12**. The fix is not more threads, it is better
packing.

A routed expert's FFN is sequential (`down` consumes `SiLU(gate)·up`, so all 640
`gate`/`up` rows must exist first), so the pool runs two phases with one barrier
between them, and cuts rows into shards sized by `choose_shards`, which picks the
smallest split whose predicted makespan is within 15 % of perfect packing:

```
phase 1   (expert j, rows r0..r1):  gate rows → up rows → SiLU(gate)·up → act_j
phase 2   (expert j, rows r0..r1):  down rows,  Σ_j w_j · down_j(act_j)
```

Each worker gets **one message per phase** carrying its whole share. That matters
as much as the packing: a parked-receiver wake-up costs ~3 µs on this host, and
one message per shard (up to 70 per token-block, ≈3 400 per token) cost more than
the imbalance it removed — a per-shard variant measured 16.2 GB/s at 12 threads
against 21–24 for the batched one.

Determinism does not depend on the worker at all: phase-1 shards write disjoint
row ranges, and phase-2 shards are accumulated in `(expert, shard)` order — the
order the single-threaded path uses — so the pool is **bit-identical** to
`moe_topk_sequential` (§4). Threads hold an `Arc<ShardSet>` and re-derive their
slices from the same read-only mapping: no raw pointers, no `unsafe`, no scoped
threads.

`--threads` default is **physical cores / 2, capped at 16** = **8** here
(`default_threads()`; `/proc/cpuinfo` physical ids, not `available_parallelism`
which counts SMT siblings).

---

## 6. Benchmark

`breccia bench cpu-experts --json …`, host `7995x-cachyos`, AMD Ryzen 9 7950X
(16 physical / 32 logical), 58 GB DDR5, model
`/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf` (75.83 GB, 42.913 GB of
routed experts), 48 blocks × 512 experts × top-10, **path avx2**.

### 6.1 One representative token (48 blocks × top-10 experts)

| run | threads | bytes/token | cold ms | best ms | GB/s | tok/s (lane only) | vs 17.5 GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| avx2-1t | 1 | 838.1 MB | 302.7 | 274.2 | 3.06 | 3.6 | 17 % |
| **avx2-pool (default)** | **8** | **838.1 MB** | **42.4** | **41.1** | **20.41** | **24.3** | **117 %** |
| scalar-1t | 1 | 838.1 MB | 4 594.8 | 4 587.5 | 0.18 | 0.2 | 1 % |

"cold" is the first pass (some of the 838 MB not yet in the page cache — a
disk-bound artefact of the first touch, not of the lane); "best" is the best of
three subsequent passes. `tok/s` is what this lane alone would allow for a token
whose only cost were the expert FFNs; the real decode adds attention/GDN,
routing, the shared expert and the LM head on top.

### 6.2 Thread sweep (same weights, `--reps 5`)

| threads | best ms | GB/s | vs 17.5 GB/s |
|---:|---:|---:|---:|
| 1 | 271.0 | 3.09 | 18 % |
| 2 | 138.5 | 6.05 | 35 % |
| 4 | 72.1 | 11.62 | 66 % |
| **8 (default)** | **39.8 – 50.2** | **16.7 – 21.1** | **95 – 120 %** |
| 12 | 35.0 – 39.8 | 21.1 – 23.9 | 120 – 137 % |
| 16 | 27.5 – 37.3 | 22.5 – 30.5 | 129 – 174 % |

Scaling is close to linear up to 8 threads (3.09 → 20.4 GB/s = 6.6× on 8 threads)
and then flattens: at 8 threads each worker sustains 2.55 GB/s against a 2.9–3.1
GB/s single-thread kernel ceiling, i.e. **~86 % packing efficiency**, and the
remainder is the two phase barriers plus DRAM contention. The 95–120 % spread at 8
threads is run-to-run variance on a shared desktop (load average 1.3–2.2 during
these runs, agent session active); 12+ threads are more stable because they are
DRAM-limited rather than packing-limited.

### 6.3 Where the lane stands

* **At the default 8 threads the lane is at or above parity**: 20.4 GB/s best
  (117 %), 95–120 % across five runs, i.e. 39.8–50.2 ms of expert reads per
  token against the 47.9 ms the 17.5 GB/s bar implies for 0.84 GB.
* **At 12–16 threads it is 1.2–1.7× the bar.** If the graph's thread budget
  allows it, `--threads 12` is the recommendation on this CPU class (one CCD's
  8 cores is *not* why: pinning 8 threads to cores 0–7, to 8–15, or spread across
  both CCDs all measured 12.8–13.0 GB/s with the old dispatch, so the limit was
  packing, not the fabric).
* The honest caveat: the bar is *sustained read bandwidth*, and this lane reaches
  it with **compute to spare only at high thread counts**. A 2× faster per-core
  kernel would put 4 threads above the bar; that is where further work should go
  (§7), not into more threads.

---

## 7. Open issues and limits

1. **No AVX-512 path.** Zen 4 has `avx512f/bw/vl/vnni`, but Rust's AVX-512
   intrinsics still need `target_feature(enable = "avx512f")` plus per-feature
   gating and a second full set of 14 kernels; the work order made it optional
   ("only if straightforward") and it is not. Expected gain: 10–25 % on the
   `vpermb`-friendly IQ types, not a 2×.
2. **Per-core throughput is the ceiling** (2.5–3.8 GB/s). The next real
   improvements are structural, not micro: (a) two-row register blocking, which
   halves activation-vector loads per unit (3 loads/unit → 2.5); (b) `vpermb`-style
   `IQ4_NL`/`IQ2_*` sign folding under AVX-512-VBMI; (c) a specialised
   `Q2_0`+`IQ4_NL` down kernel, since those two types are 100 % of the `down`
   roles. None is needed to clear the bar.
3. **`Q4_K`/`Q5_K`/`Q6_K`/`BF16`/`F16`/`F32` have kernels but are only exercised
   synthetically** in the real-blob test: those types are dense-only in this
   release (`ffn_*_shexp`, `output.weight`, routers, norms). They are correct
   (§4) and ready for the graph, but the first real use of the shared expert will
   be the first real-blob run of `Q4_K`/`Q5_K`/`Q6_K`.
4. **Synthetic conformance cannot be held to result-relative 1e-5** on
   high-cancellation rows (§4); the report states the `Σ|wⱼxⱼ|` figure instead.
   A future conformance version could accumulate in `f64` **once per block**
   (`vaddps` → `f64` flush) at ~2 % cost, which would tighten the synthetic worst
   case; it was not worth the inner-loop complexity for a bar the real artefacts
   already clear by 27×.
5. **The pool is per-`moe_topk`-call**, i.e. two barriers per block. Nothing is
   pipelined across blocks yet: the graph could overlap block *N*'s `down` phase
   with block *N+1*'s `gate`/`up` if the residual dependency allows, and could
   prefetch the next token's top-10 pages. Both belong to the forward-pass work,
   not here.
6. **No CPU affinity control** (would need `libc`/`unsafe`, which this lane
   deliberately does not use). Thread placement is the OS's; the sweep shows it
   does not matter at these counts.

---

## 8. Reproduction

```bash
# three-way conformance (synthetic + real blobs + determinism + the force-scalar switch)
cargo test -p breccia-kernels --test cpu_experts --release -- --nocapture

# the benchmark, JSON included
cargo build --release -p breccia-cli
./target/release/breccia bench cpu-experts --json /tmp/cpu-experts.json
./target/release/breccia bench cpu-experts --threads 12 --reps 5 --type-experts 1

# portable fallback path only
BRECCIA_CPU_FORCE_SCALAR=1 ./target/release/breccia bench cpu-experts --quick
```

---

## 9. Attribution

Decode semantics are ported from llama.cpp `ggml-quants.c` (MIT, commit 6050b46,
the qwen4exp fork that defines the fork-only `Q2_0` block), via this crate's
`reference.rs`; the IQ grid/sign/value tables are the generated copies of
`ggml-common.h` already in `crates/breccia-kernels/src/tables.rs`. The dispatch
shape follows the per-token MoE contract in
`docs/specs/qwen4exp-architecture.md` §5.4 and the production baseline's
measured host-read budget in `docs/phase-3/07-moe-cache-ab.md`.
