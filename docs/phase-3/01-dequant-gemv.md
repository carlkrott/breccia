# 01 — Dequant + GEMV decode kernels (`breccia-kernels`)

Work-order item #1 of the M2 kernel work order (`docs/phase-0/gate-A-report.md` §3):
**fused dequantisation + matrix-vector multiply for batch-1 (decode) inference**, Vulkan
backend (2026-10-26 decision: Vulkan primary). This is the op that dominates a decode
token — the routing experts' weights alone are ~0.9 GB of the ~4.3 GB read per token —
so it sets the arithmetic ceiling for the 20 → 85 tok/s lever.

Date: 2026-10-26 · GPUs idle at the start of every run (`pgrep llama-server` empty,
`gpu_busy_percent` 0/7 at start) · no service was started or stopped · artifacts in
`<SCRATCH>/opencode/p3/`.

**Status: complete.** `cargo build/test --workspace`, `cargo clippy -p breccia-kernels
--all-targets -- -D warnings`, `cargo fmt --check` green; 0 validation-layer
errors/warnings on both GPUs; every kernel type is bit-exact against the CPU reference
for dequant and ≤ 4.1e-5 relative for GEMV; real expert tensors run straight out of the
GGUF mapping at 271–398 GB/s on the RX (61–89 % of the 445 GB/s reference).

---

## 1. What landed

| Path | Content |
|---|---|
| `crates/breccia-kernels/build.rs` | `glslc -O --target-env=vulkan1.3 -I shaders` → `spirv-val` for 19 shaders (same pattern as `crates/breccia-vk/build.rs`); SPIR-V is `include_bytes!`-embedded. |
| `shaders/blocks.glsl` | The decode contract: one function per type, `dequant_<type>_unit(off, unit, out float v[8])`, plus the shared buffer/push-constant layout. Attribution headers. |
| `shaders/gemv_body.glsl` | The fused kernel body (row/unit mapping, split-K, tail mask, subgroup reduction). |
| `shaders/dequant_body.glsl` | Standalone block dequantiser (same decode functions, so conformance covers one code path). |
| `shaders/gemv_<type>.comp`, `dequant_<type>.comp` | 18 thin wrappers (block geometry + which decode unit to use). |
| `shaders/gemv_reduce.comp` | Split-K reduction (coalesced per-row strided sum). |
| `src/quant.rs` | `QuantType` (9 types) in *kernel* geometry, GGML id ↔ name ↔ `GgmlType` mapping. |
| `src/gemv.rs` | `Gemv { ty, rows, cols }`: pipeline + table buffer; `run(&w, x, out)`, `run_timed`, `run_split`, `dequant_to_f32`, device-local upload/download helpers. |
| `src/reference.rs` | Scalar CPU port of `ggml-quants.c`: `dequant_block`/`dequant_row`/`dot_row`/`gemv`. |
| `src/bench.rs` | `breccia bench gemv` runner + JSON schema (schema_version 1), clock warm-up, reference-bandwidth comparison. |
| `src/real.rs` | Real-model mode: `ShardSet` mmap → expert role tensor (or any named tensor) → kernel; conformance check on real quantised bytes. |
| `src/tables.rs` | Generated from `ggml-common.h` (`tools/gen_tables.py`): grid/sign/value tables for the CPU reference **and** the u32 word layout the kernels upload. |
| `src/device.rs` | Device open (reuses `breccia_vk::ctx` instance/device/queues — no duplicated Vulkan init). |
| `tests/conformance.rs` | 6 tests: dequant vs reference, GEMV vs reference, non-finite scales, zero scales, split-K equivalence, `grid.y` wrap. |
| `tools/gen_tables.py`, `tools/refcheck.py`, `tools/refcheck_drv.c` | Table generator + the bit-exactness check against the verbatim `ggml-quants.c`. |
| `crates/breccia-cli` | exactly one new subcommand: `breccia bench gemv`. |

### Kernel shape (memory-bandwidth-bound decode)

* **One 32-lane group per output row**, 8 groups (256 threads) per workgroup. Lane `l`
  owns unit `l` of every block it walks, i.e. **8 consecutive elements** `[8l, 8l+8)`, so a
  256-element super-block is covered by exactly one warp iteration and every weight byte of
  a row is read **exactly once** (no re-reads between the dequant and the dot product — the
  fused kernel never materialises f32 weights).
* 32-lane groups are hardware-independent: an xor-shuffle reduce with offsets ≤ 16 stays
  inside an aligned 32-lane boundary, so the same shader is correct on Wave32 (gfx1030) and
  Wave64 (gfx906) — the RX reports `subgroupSize` 64 and the MI50 64; both were measured.
* **fp32 accumulation**, two alternating accumulators to keep two FMA chains in flight.
  The 4th push constant is the split-K factor (fp16 accumulation is a follow-up, §7).
* Weight blocks are read through a **16-bit-word storage-buffer view** (`uint16_t W[]`),
  because GGML block sizes are even but not multiples of four (66/74/98/110 B), so 32-bit
  loads at block starts are not always aligned. `storageBuffer16BitAccess` is the only
  feature needed (enabled by `ctx`); no `shaderInt16`, no 8-bit storage.
* The IQ grid/sign/value tables live in a **7 KB read-only storage buffer** (binding 3),
  not in GLSL `const` arrays — see §4.2 for the 100× measurement that decided this.
* Dispatches wrap over `grid.y`/`grid.z` so `grid.x` never exceeds the 65535 workgroups
  every Vulkan device guarantees: a 1.31 M-row expert tensor needs 163,840 row groups and
  the dequantiser needs 65,536 for a 75 MiB matrix. Both are covered by tests
  (`large_dispatch_wraps_over_grid_y`).

## 2. Types

| Type | GGML id | Kernel block | Bytes | Model usage (§4.2 of the inventory) |
|---|---:|---|---:|---|
| IQ3_S | 21 | 256 | 110 | 82 shard-1 tensors; 26 expert roles; embedding |
| Q2_0 *(fork-only)* | 42 | **64** | 18 | 30 expert `down` roles + 8 dense; 2.25 bpw |
| IQ2_XXS | 16 | 256 | 66 | 18 |
| IQ2_XS | 17 | 256 | 74 | 20 |
| IQ3_XXS | 18 | 256 | 98 | 12 |
| IQ4_NL | 20 | 32 | 18 | 58 (incl. the 26.8 GiB PLE table) |
| Q5_K | 13 | 256 | 176 | 15 (`output.weight`) |
| BF16 | 30 | 8 *(virtual)* | 16 | 484 |
| F32 | 0 | 8 *(virtual)* | 32 | 292 |

BF16/F32 are stored per element; the kernels view 8 elements as one virtual block so every
type shares the same unit-of-8 decode contract (and the same 8-element-aligned `x` reads).
Geometry is taken from `breccia_format::quant` (a unit test asserts equality for the eight
real block types); `Q2_0 = 64/18` is the fork's block, **not** upstream ggml's.

Not implemented yet (deliberate, see §7): IQ4_XS (67 tensors), Q6_K (71), Q4_K (45), IQ2_S
(20), Q2_K/Q3_K/Q8_0 (not present in this release but in the Q2_0 release).

## 3. Conformance evidence

### 3.1 The CPU reference is faithful to `ggml-quants.c` — verified bit-exactly

`tools/refcheck.py` compiles the **reference clone's own** `ggml/src/ggml-quants.c` and
calls `dequantize_row_iq2_xxs / _iq2_xs / _iq3_xxs / _iq3_s / _iq4_nl / _q5_K / _q2_0`
directly (`tools/refcheck_drv.c`), generates random block bytes (including the degenerate
cases: zero, subnormal, ±65504, ±inf, NaN scales) and diffs the result against this
crate's `reference::dequant_block` **bit for bit**:

```
$ python3 tools/refcheck.py --blocks 2000
OK IQ2_XXS: 2000 blocks / 512000 values bit-identical to ggml-quants.c
OK IQ2_XS:  2000 blocks / 512000 values bit-identical to ggml-quants.c
OK IQ3_XXS: 2000 blocks / 512000 values bit-identical to ggml-quants.c
OK IQ3_S:   2000 blocks / 512000 values bit-identical to ggml-quants.c
OK IQ4_NL:  2000 blocks /  64000 values bit-identical to ggml-quants.c
OK Q5_K:    2000 blocks / 512000 values bit-identical to ggml-quants.c
OK Q2_0:    2000 blocks / 128000 values bit-identical to ggml-quants.c
all 7 types bit-identical
```

(The 400-block run used while developing caught one real porting bug: IQ3_XXS/IQ3_S index
the second `qs` 32-bit word for `l ≥ 2` — `+2*(l/2)` — which the GPU shader had right and
the Rust reference did not.)

### 3.2 GPU vs CPU reference (`cargo test -p breccia-kernels --test conformance`)

Shapes: `1xN`, `3xN`, `640x2560`, `2560x640`, plus two tail shapes per type
(`cols % block_elems != 0`: one partial block, and several whole blocks + a partial one).
`|y|` up to ~1.9e2. Tolerance: dequant 1e-6, GEMV 1e-3.

| type | dequant max rel | GEMV max rel |
|---|---:|---:|
| IQ3_S | **0** | 1.3e-5 |
| Q2_0 | **0** | 2.3e-7 |
| IQ2_XXS | **0** | 4.7e-6 |
| IQ2_XS | **0** | 3.7e-6 |
| IQ3_XXS | **0** | 9.1e-6 |
| IQ4_NL | **0** | 5.1e-6 |
| Q5_K | **0** | 4.0e-5 |
| BF16 | **0** | 1.9e-5 |
| F32 | **0** | 2.8e-5 |

Dequant is **bit-identical** to the reference for every type and shape (the fp16 scale is
exact and the integer codes are exact; the GPU's `unpackHalf2x16` matches the reference
widening bit for bit — note that includes inf/NaN payload equivalence for finite inputs).
GEMV differences are fp32 accumulation order + fma contraction against an `f64` CPU dot,
2–4 orders of magnitude inside the tolerance.

Also covered by tests: inf/NaN block scales decode to non-finite values **and** the finite
block behind them still decodes exactly (no device loss); `d = 0` blocks decode to exactly
`0.0`; split-K factors 1/2/3/5/8 all agree with the CPU reference.

### 3.3 Real quantised bytes

Real-mode runs check the first 16 rows of the actual tensor against the CPU reference:
max relative error **2.4e-8 … 1.6e-7** across IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL,
Q2_0 and Q5_K (§5).

## 4. Throughput

Commands (both GPUs idle, `MESA_VK_DEVICE_SELECT` cleared inside `breccia-vk`):

```bash
cargo build --release -p breccia-cli
./target/release/breccia bench gemv --device 0000:03:00.0 --json synth-rx.json        # RX, full sweep
./target/release/breccia bench gemv --device 0000:0e:00.0 --rows 10240 --cols 2560    # MI50, medium shapes
./target/release/breccia bench gemv --device 0000:03:00.0 --real --layer 8 --role gate --json real-rx-8-gate.json
```

Each cell: weights uploaded once to device-local memory, a clock-ramp warm-up plus one
discarded window, then 3 timed windows (`vkCmdWriteTimestamp` bracketing, full barrier
between dispatches); the best window is reported. `%` is against the 445 GB/s RX /
825 GB/s MI50 reference figures in the work order, and §4.1 also gives the measured
read-only DRAM ceiling this compares against.

### 4.1 RX 6950 XT (RADV NAVI21, gfx1030) — GB/s (and % of 445 GB/s)

| type | 640x2560 | 2560x640 | 2560x2560 | 2560x6144 | 6400x2560 | 10240x2560 | 248320x2560 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ3_S | 127 (29 %) | 209 (47 %) | 293 (66 %) | 311 (70 %) | 334 (75 %) | 354 (79 %) | **369 (83 %)** |
| Q2_0 | 91 (21 %) | 156 (35 %) | 274 (62 %) | 300 (68 %) | 332 (75 %) | 363 (82 %) | **384 (86 %)** |
| IQ2_XXS | 78 (18 %) | 127 (29 %) | 207 (47 %) | 222 (50 %) | 251 (56 %) | 273 (61 %) | **281 (63 %)** |
| IQ2_XS | 87 (20 %) | 143 (32 %) | 227 (51 %) | 243 (55 %) | 266 (60 %) | 289 (65 %) | **303 (68 %)** |
| IQ3_XXS | 114 (26 %) | 191 (43 %) | 280 (63 %) | 298 (67 %) | 328 (74 %) | 357 (80 %) | **368 (83 %)** |
| IQ4_NL | 161 (36 %) | 260 (58 %) | 373 (84 %) | 398 (90 %) | 448 (101 %) | 480 (108 %) | **470 (106 %)** |
| Q5_K | 141 (32 %) | 185 (42 %) | 220 (49 %) | 229 (52 %) | 250 (56 %) | 259 (58 %) | **271 (61 %)** |
| BF16 | 596 (134 %) | 972 (218 %) | 1171 (263 %) | 809 (182 %) | 1401 (315 %) | 1652 (371 %) | **562 (126 %)** |
| F32 | 1162 (261 %) | 1445 (325 %) | 1208 (272 %) | 706 (159 %) | 1617 (363 %) | 1754 (394 %) | **561 (126 %)** |

Each cell takes the **best of 3 windows after a discarded warm-up window**; spread
between the reported windows is 0.1 % median / 3.8 % max. (Without the discarded window
the first window of every cell reads 20–25 % slow — the DPM ramp after the per-cell upload
idle, `docs/PROJECT-SETUP.md` — which is why it exists; the fixed 0.2 s global warm-up at
the start of the run only covers the first cells.)

**Two ceilings, and which one a cell hits.**

* **Read-only DRAM ceiling: ~561 GB/s = 97 % of the card's 576 GB/s spec.** The BF16/F32
  kernels at 1.2/2.4 GiB working sets (far beyond the 128 MiB Infinity Cache) move exactly
  the weight bytes and no decode arithmetic, so they measure the pure-read limit:
  561–562 GB/s. The 445 GB/s figure in the work order is not that: it is the Phase-0
  **read+write copy total** (231 GB/s read + 231 GB/s write = 462 GB/s,
  `docs/phase-0/02` §2). A GEMV only reads weights, so it can and does exceed it — the
  honest denominator for "fraction of achievable read bandwidth" is 561, which is why both
  percentages are given.
* **Small matrices are Infinity-Cache resident, not DRAM-bound.** A 640x2560 IQ3_S matrix
  is 0.67 MiB and a 6400x2560 one is 6.7 MiB — both fit the 128 MiB IC, which is why F32
  shows 259–391 % of "DRAM bandwidth". Those cells are ALU/latency-bound, and their GB/s
  number is **not** a decode prediction; the 248320x2560 column is the DRAM-bound one.

Against the 561 GB/s measured read ceiling, the DRAM-bound column (248320x2560) is:

| IQ4_NL | Q2_0 | IQ3_S | IQ3_XXS | IQ2_XS | IQ2_XXS | Q5_K |
|---:|---:|---:|---:|---:|---:|---:|
| 84 % | 68 % | 66 % | 66 % | 54 % | 50 % | 48 % |

and against the 576 GB/s vendor spec, 46–82 %.

So: **every type clears the ≥ 60 % target when measured against the 445 GB/s figure the
work order specifies** (61–106 %), and four of seven clear 66–84 % of the *pure-read*
ceiling. IQ2_XXS (50 %) and Q5_K (48 %) fall short of the read ceiling for a reason that is
visible in the numbers, not in the bandwidth: they are **element-rate bound**, not byte-rate
bound. IQ2_XXS at 2.06 bpw decodes 15.5× more elements per byte than F32 (281 GB/s at
2.06 bpw ≈ **1.09 T elements/s**), and Q5_K has the most per-element work of any type
(`get_scale_min_k4` unpack + 5th-bit test + two muls + subtract): **0.39 T elements/s**
versus **0.86 T/s** for IQ3_S, i.e. the same element-rate ceiling the MI50 hits everywhere
(§4.3). The fix is fewer instructions per element (§7), not more bandwidth.

### 4.2 Two measurements that changed the design

1. **GLSL `const` tables cost 100×.** The first version read the IQ grids from
   `const uint IQ3S_GRID[512]` arrays. On RADV a dynamically indexed const array is
   lowered to *private* memory and spills, and IQ3_S measured **0.9 GB/s** (0.794 ms for a
   0.67 MiB matrix) versus 438 GB/s for the table-free F32 kernel in the same run. Moving
   the tables to a 7 KB read-only storage buffer (which stays in L1/L2) fixed it:
   116 GB/s at that shape in that run (127 GB/s after the later split-K default),
   369 GB/s at the DRAM-bound shape. The offsets in
   `shaders/blocks.glsl` are checked against `tables::TBL_*` by a unit test.
2. **Split-K is worth it only where the kernel is ALU-bound.** A 640-row expert matrix is
   only 80 workgroups for 60 CUs (1.3/CU), so the obvious fix is split-K (each row's block
   range cut into `splits` chunks + a reduce pass). Measured at 640x2560, same run, best of
   3 windows:

   | type | 1 split | 2 | 3 | 4 | verdict |
   |---|---:|---:|---:|---:|---|
   | Q5_K | 113 | 129 | **142** | 144 | +26 % |
   | IQ4_NL | 141 | 153 | ~157 | 160 | +13 % |
   | IQ3_XXS | 103 | 107 | ~111 | 114 | +10 % |
   | IQ3_S | 116 | 117 | 127 | 124 | +9 % |
   | IQ2_XXS | 72 | 77 | **78** | 79 | +9 % |
   | IQ2_XS | 81 | 85 | 87 | 87 | +7 % |
   | Q2_0 | 89 | 88 | 91 | 89 | 0 % |
   | BF16 | 558 | 623 | 596 | 603 | +7 % (best at 2) |
   | F32 | 1161 | 987 | 819 | 782 | **−29 %** |

   So the reduce dispatch (2–3 µs: two full barriers + one small dispatch) is cheaper than
   the parallelism it buys for the decode-bound types, and more expensive than the bandwidth
   F32 already achieves. `Gemv::recommended_splits_for` therefore splits **quantised types
   only, only when the shape has < 4 workgroups per CU**, capped at 4 and by the blocks per
   row; the CLI can force it with `--splits N`. That heuristic is what produced the left
   column of §4.1 (640x2560 uses 3 splits); every other column is single-pass.

### 4.3 Radeon VII / MI50 (RADV VEGA20, gfx906) — the decode path is ALU-bound here

10240x2560 (10.7 MiB F32-equivalent, DRAM-bound), same binary, same shaders:

| type | GB/s | % of 825 GB/s | elements/s |
|---|---:|---:|---:|
| F32 | **774** | **94 %** | 0.19 T/s |
| BF16 | **660** | **80 %** | 0.33 T/s |
| Q2_0 | 146 | 18 % | 0.52 T/s |
| IQ3_S | 128 | 16 % | 0.30 T/s |
| IQ4_NL | 118 | 14 % | 0.21 T/s |
| IQ3_XXS | 102 | 12 % | 0.27 T/s |
| IQ2_XS | 90 | 11 % | 0.31 T/s |
| Q5_K | 90 | 11 % | 0.13 T/s |
| IQ2_XXS | 77 | 9 % | 0.30 T/s |

The table-free kernels reach 80–94 % of the 825 GB/s reference (774 GB/s F32 = 76 % of the
card's 1024 GB/s HBM2 spec — read-only does not beat the Phase-0 read+write total on this
memory subsystem, unlike the RX), so the memory path, the queue plumbing and the kernel
structure are fine; every *quantised* type collapses to
0.13–0.52 T elements/s, i.e. the same ~0.3 T/s ceiling regardless of bytes per element
(Q2_0's 0.52 T/s is the only type with almost no unpack work). That is the signature of an
instruction-rate limit, not a bandwidth limit — gfx906's VALU runs the byte-extraction /
integer-conversion mix slowly (the RX does the same code at 1.1–1.2 T/s). This changes the
placement conclusion in a useful way: **the MI50 should not host IQ\* decode weights**;
its 825 GB/s is only reachable with F32/BF16/fp16 kernels. An int8 dot-product decode
(gfx906 has `v_dot4_i32_i8`, the standard trick for IQ formats — grid bytes × int8-quantised
activations) is the way to move that number, and is a §7 item.

## 5. Real-blob numbers (mmap, no copy of the 75 GB model)

`breccia bench gemv --real` opens the two IQ3_XXS shards with
`breccia_format::ShardSet::open_glob`, takes a `TensorView` (a `&[u8]` slice of the
read-only mapping), verifies the kernel's geometry against the tensor's byte length, uploads
**only that tensor**, and runs the real kernel. Nothing is copied out of the file: the
page cache serves the upload, so the cost is the page-in of exactly the benchmarked tensor
(0.22–0.45 GiB, 0.3–0.9 s including H2D).

| tensor | type | shape | MiB | RX GB/s (% of 445) | conformance (max rel) |
|---|---|---|---:|---:|---:|
| `blk.0.ffn_down_exps` (512 experts) | IQ4_NL | 1310720x640 | 450 | **398 (89 %)** | 5.9e-8 |
| `blk.0.ffn_gate_exps` | IQ2_XS | 327680x2560 | 231 | 305 (69 %) | 1.6e-7 |
| `blk.8.ffn_gate_exps` | IQ3_S | 327680x2560 | 344 | 369 (83 %) | 1.6e-7 |
| `blk.35.ffn_gate_exps` | IQ3_XXS | 327680x2560 | 306 | 369 (83 %) | 1.5e-7 |
| `blk.1.ffn_down_exps` | Q2_0 | 1310720x640 | 225 | 286 (64 %) | 7.3e-8 |
| `blk.1.ffn_gate_exps` | IQ2_XXS | 327680x2560 | 206 | 281 (63 %) | 1.1e-7 |
| `token_embd.weight` | IQ3_S | 248320x2560 | 260 | 368 (83 %) | 2.4e-8 |
| `output.weight` | Q5_K | 248320x2560 | 417 | 271 (61 %) | 1.1e-7 |
| `blk.0.ffn_gate_inp.weight` (router) | BF16 | 512x2560 | 2 | 536 (121 %) | 1.6e-7 |

MI50, same command, two real tensors (upload, including paging the mmap in: 0.48 s / 0.46 s):

| tensor | type | MiB | MI50 GB/s (% of 825) | conformance |
|---|---|---:|---:|---:|
| `blk.0.ffn_gate_exps` | IQ2_XS | 231 | 100 (12 %) | 1.6e-7 |
| `blk.1.ffn_down_exps` | Q2_0 | 225 | 143 (17 %) | 5.5e-8 |

The real-blob numbers track the synthetic DRAM-bound cells (IQ3_S 369 vs 369, IQ2_XS 305 vs
303, Q5_K 271 vs 271, IQ4_NL 398 vs 470 at the closest square synthetic shape — the down
tensor has only 640 columns per row, which costs ~15 % of streaming efficiency), i.e. **the
benchmark is representative of the actual artefact** — the same bytes, the same scales, the same D=inf-adjacent edge
cases, and the CPU reference agrees to 1e-7.

### What this means for the token budget (honest arithmetic)

Per decode token the routed experts are 10 experts × 48 layers × ~1.87 MB (blk.0 mix)
≈ **0.90 GB**, plus ~3.4 GB of dense/HC/embedding weights and the shared expert. At the
measured 271–402 GB/s that is **10.7–15.9 ms of pure GEMV per token ≈ 63–93 tok/s** if the
GEMV kernels were the only work in the token — the 85 tok/s target is *reachable but has no
headroom* for attention/GDN/PLE with this kernel set, which is exactly why §7 item 1
(batching the layer's 10 experts into one dispatch, measured ~2.6× cheaper than 10 separate
640-row launches: 21 µs vs 55 µs for IQ3_S) and item 2 (fewer instructions per element) are
next.

## 6. Verification

```bash
cargo build --workspace                     # Finished
cargo test --workspace                      # all suites green (lib 7, conformance 6, + existing)
cargo clippy -p breccia-kernels --all-targets -- -D warnings   # clean
cargo clippy -p breccia-cli     --all-targets -- -D warnings   # clean
cargo fmt --check                           # clean

# validation layer, 0 errors / 0 warnings:
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./target/release/breccia bench gemv \
    --device 0000:03:00.0 --rows 640 --cols 2560      # RX synthetic (exercises split-K + reduce)
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./target/release/breccia bench gemv \
    --device 0000:0e:00.0 --quick                     # MI50 synthetic
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./target/release/breccia bench gemv \
    --device 0000:03:00.0 --real --layer 0 --role down --quick   # RX real mmap path

# bit-exactness against llama.cpp's own quant code:
python3 tools/refcheck.py --blocks 2000       # 7 types bit-identical
```

Logs/JSON: `<SCRATCH>/opencode/p3/` (`synth-rx.json`, `synth-mi50-a.json`,
`real-rx-*.json`, `real-mi50-*.json`, `conformance.log`, `validation-*.log`).

## 7. What's next (ordered by measured value)

1. **Batch the layer's routed experts into one dispatch.** The 640-row per-expert shape is
   ALU/latency-bound (78–161 GB/s); the same weights as a 10-expert matrix (6400 rows)
   measured 251–448 GB/s, i.e. ~2.6× the work per unit time. Needs a per-row base-offset indirection (u32 per row) so the 10
   selected experts' scattered slices can be addressed by one grid — the pack's
   expert-major layout makes the *single*-expert case contiguous, not the routed set.
2. **Fewer instructions per element.** Q5_K/IQ2_XXS (0.4 T elements/s) and the whole MI50
   number set are element-rate bound. Two concrete paths: (a) quantise `x` to int8 on the
   fly and use packed integer dot products (`v_dot4_i32_i8` on gfx906,
   `v_dot4_i32_i8`/DP4A equivalents on RDNA2 via `VK_KHR_shader_integer_dot_product`), so 4
   elements are consumed per instruction; (b) an fp16-accumulate variant (`pc.flags` bit 0
   is reserved for this) which halves the register/FMA pressure.
3. **fp16/bf16 activation path** for the dense BF16 tensors (`ffn_gate_inp`, HC), which
   currently go through the F32-with-u16-gather path.
4. **Tiling for prefill** (batch > 1): the current kernel is 1-D (one warp per row, one
   activation vector). Prefill wants a 2-D tile with `x` staged in LDS and N tokens per
   dispatch; nothing here is reusable for that except the decode units.
5. **Fused SwiGLU** (`gate`/`up` in one dispatch, `silu(g)*u` before the down projection) —
   saves a 640-wide intermediate round trip and one dispatch per expert.
6. **Remaining types**: IQ4_XS (67 shard-1 tensors), Q6_K (71), Q4_K (45), IQ2_S (20) —
   203 tensors, mostly dense attention/SSM plus the shared expert — then Q2_K/Q3_K/Q8_0/
   Q4_0/Q5_0 for the Q2_0 release. The decode units are ~40 lines each and the wrapper is
   generated.

## 8. Attributions

* **Decode semantics** — ported from llama.cpp `ggml/src/ggml-quants.c`
  (`dequantize_row_iq2_xxs`, `_iq2_xs`, `_iq3_xxs`, `_iq3_s`, `_iq4_nl`, `_q5_K` +
  `get_scale_min_k4`, `_q2_0`), MIT, commit **6050b46** (the qwen4exp fork — the one that
  defines the fork-only `Q2_0 = 64/18` block this model uses for 30 expert roles). Both the
  GLSL (`shaders/blocks.glsl`) and the Rust reference (`src/reference.rs`) carry the
  attribution header and a per-type note of which C function they follow.
* **GPU structuring** — adapted from **ggml-vulkan** (MIT, same commit):
  `vulkan-shaders/mul_mat_vec_iq3_s.comp` / `mul_mat_vec_iq2_xxs.comp` (one warp covers a
  super-block, per-lane 8-element unit, two-grid nibble unpack), `types.glsl` (16-bit-word
  block views for the misaligned 66/74/98/110-byte blocks), `dequant_funcs.glsl` +
  `dequant_iq3_s.comp` (standalone dequantiser shape). Each shader carries
  `Adapted from ggml-vulkan (MIT), <file>, commit 6050b46`. Differences from the originals:
  raw u16-word views instead of typed structs, fp32-only accumulation, an in-loop tail mask
  for `cols % QK != 0`, a table storage buffer instead of const arrays, and the split-K path.
* **Tables** — `iq2xxs_grid`, `iq2xs_grid`, `iq3xxs_grid`, `iq3s_grid`, `ksigns_iq2xs`,
  `kvalues_iq4nl` are *data* from llama.cpp's `ggml-common.h` (MIT), copied verbatim by
  `tools/gen_tables.py` into `src/tables.rs` (and from there into the GPU table buffer).
* `shaders/gemv_reduce.comp` is original.
* No code from IsoQuant or RotorQuant (unlicensed) was consulted or copied.
