# 02 — Batched expert dispatch + the remaining quant types (`breccia-kernels`)

Work-order items #1 and #2 of the M2 kernel work order's follow-up list
(`docs/phase-3/01-dequant-gemv.md` §7): **one dispatch over the layer's routed
experts** (the 2.6× lever measured there) and the **four quant types that had no
kernel** (203 tensors, 6.45 GiB — 9.1 % of the model).

Date: 2026-10-26 · GPUs idle at the start of every run (`pgrep llama-server`
empty, `gpu_busy_percent` 0) · no service was started or stopped · artifacts in
`/var/tmp/users/korphaus/tmp/opencode/p3/`.

> **This work order caused a GPU fault and a machine-wide freeze.** A conformance
> run on the display card (RX 6950 XT) took the box down at ~10:09; it recovered
> only after a reboot. §4 is the analysis, the hardening that came out of it, and
> the mandatory test procedure that is now in `AGENTS.md`. **Every number in this
> report was re-measured after that hardening**, on the non-display GPU first.

**Status: items #1 and #2 complete.** Item #3 (fp16 accumulate behind a flag) was
**not** attempted — `breccia-vk`'s pipeline layout fixes the push-constant block at
16 bytes and it is fully used, so a `flags` bit needs a change outside this work
order's write scope; §7.1 has the design and the expected (negative) accuracy
result.

* `cargo build/test --workspace`, `cargo clippy -p breccia-kernels` /
  `-p breccia-cli --all-targets -- -D warnings`, `cargo fmt --check`: green.
* 0 validation-layer messages on every run, **one quant type per invocation**,
  each `timeout 90`, Radeon VII (MI50) first and the RX only afterwards (§5).
* Batched dispatch on the real top-10 shape: **2.96–3.55× the per-expert loop** on
  the 640-row projections (0.058–0.061 ms → 0.017–0.021 ms), **1.61–1.95×** on the
  2560-row `down` projection; 3.20–4.15× at 16 experts.
* New types: **dequant bit-identical** to `ggml-quants.c` (0 relative error at
  every shape) and GEMV within **6.3e-6 … 5.9e-5** relative; real-model
  throughput 141–413 GB/s (32–93 % of the 445 GB/s reference).

---

## 1. What landed

| Path | Content |
|---|---|
| `src/batch.rs` | `GemvBatch` + `XLayout{Shared,PerExpert}` + `map_to_bytes`/`block_words`/`alloc_map`: one pipeline for `{type, rows_per_expert, cols}` with a per-row indirection table, `map(experts, layout)` / `map_for_blobs(&[u32], layout)`, `run`, `run_timed`, geometry helpers (`blob_bytes`, `blob_words`, `padded_cols`, `map_bytes`, `x_bytes`). |
| `shaders/gemv_body.glsl` | `#ifdef BRECCIA_EXPERT_MAP`: binding 4 = row map, `ROW_BASE(r)`/`X_BASE(r)` macros; the single-expert path expands to exactly the old expressions. |
| `shaders/gemv_<type>_moe.comp` | 13 wrappers (one per type) that set `BRECCIA_EXPERT_MAP`. |
| `src/gemv.rs` | `TAIL_PAD` (64 B of slack in every device allocation, §4.2), `run_looped_timed` / `run_split_looped_timed` (a *measured* per-expert-loop baseline, so the before/after comes from the same process). |
| `src/bench.rs` | batched mode (`--experts`): `BatchCell` in the report (loop ms, batched ms, speedup, GB/s both ways, single-expert latency, conformance), `default_expert_shape()`, `default_expert_counts()`. |
| `src/real.rs` | batched mode on real GGUF blobs: experts 0..N of a role tensor straight out of the mmap, both `x` layouts, conformance against the CPU reference on real bytes; `scale_offsets()` is now the single place that says where a type's scale fields live. |
| `src/quant.rs` | +`Iq2S`, `Iq4Xs`, `Q4K`, `Q6K` (13 types); `NOT_IMPLEMENTED` is now only types this release does not contain. |
| `shaders/blocks.glsl` | +4 decode units (`dequant_iq2_s_unit`, `dequant_iq4_xs_unit`, `dequant_q4_k_unit`, `dequant_q6_k_unit`), `block_byte`/`block_i8` helpers, the IQ2_S grid accessors, `scale_byte_k` shared by Q4_K/Q5_K, **and a mask on every table accessor** (§4.2). |
| `shaders/dequant_<type>.comp`, `gemv_<type>.comp` | 4 + 4 new wrappers. |
| `src/reference.rs` | +4 CPU ports (`dequantize_row_iq2_s`, `_iq4_xs`, `_q4_K`, `_q6_K`). |
| `src/tables.rs`, `tools/gen_tables.py` | +`IQ2S_GRID` (1024 × u64 = 8 KiB) and the GPU words `TBL_IQ2S_LO/HI` **appended** after the existing tables, so no earlier offset moved; big tables are now emitted `static`. |
| `tools/refcheck.py`, `tools/refcheck_drv.c` | the four types added to the bit-exactness check against llama.cpp's own `ggml-quants.c`. |
| `tests/conformance.rs` | +`batched_experts_match_the_per_expert_loop`, +`denormal_and_zero_scales_decode_exactly`; non-finite test now covers +inf/−inf/NaN and checks the neighbour block; `BRECCIA_QTYPE` / `BRECCIA_KERNEL_DEVICE` selectors (§4.3). |
| `src/lib.rs` | +`table_accessors_are_bounded`, +`no_float_to_integer_conversion_in_a_decode_unit` (§4.2). |
| `crates/breccia-cli` | one flag on `bench gemv`: `--experts N[,N...]` plus its summary table. Nothing else in the CLI changed. |

---

## 2. Batched multi-expert dispatch

### 2.1 Why the per-expert shape is latency-bound (the math)

One routed expert is `gate`/`up` = **640 rows × 2560 cols** and `down` =
**2560 rows × 640 cols**. With one 32-lane group per row and 8 rows per workgroup
(`ROWS_PER_WG`):

| projection | rows | workgroups | per CU (60 CUs) | measured GB/s alone |
|---|---:|---:|---:|---:|
| gate / up | 640 | 80 | **1.3** | 72–141 |
| down | 2560 | 320 | 5.3 | 150–251 |

The gate/up shape cannot fill the machine: 80 workgroups × 256 threads = 20 k
threads against 60 CUs × 2048 = 123 k thread slots, i.e. **16 % occupancy** and
~6 µs per dispatch of pure latency. Ten of them cost 10 × the latency of one
because they are serialised dispatches, and each only does 640 × 2560 MACs
(1.6 M) — 0.4 % of the card's per-ms MAC budget.

Batching fixes the *grid*, not the bytes: 10 experts = 6400 rows = 800
workgroups = 13 per CU. Weight traffic is identical, so the win is exactly the
idle latency removed — and it is capped by the per-type element rate, which is
why Q6_K (most decode arithmetic per byte) gains the least (§2.5).

### 2.2 Buffer layout and the row map

`W` is **one device buffer with the N expert blobs back to back** — the M3
slot-ring order (`breccia_vk::transfer::TransferRing` holds one expert blob per
slot; a layer's top-10 is a contiguous run of slots, and a scattered top-10 is N
blobs the caller concatenates or copies into one buffer). Rows no longer start at
`row * bpr` blocks, so the kernel reads a per-row table (binding 4, `u32`, 8 bytes
per row):

```
map[0     .. rows)   row weight base      — u16-word index into W
map[rows  .. 2rows)  row activation base  — vec4 index into X
```

* **weight base** = `blob_word_offset[e] + r * bpr * BLOCK_WORDS`, where
  `BLOCK_WORDS = block_bytes/2` (the u16 storage-buffer view every type already
  uses, because GGML blocks are not 4-byte aligned).
* **activation base** = `0` for every row under [`XLayout::Shared`] (gate/up: all
  routed experts read the same `mixed` vector — this model's MoE input), or
  `e * padded_cols/4` under [`XLayout::PerExpert`] (down: each expert has its own
  SwiGLU intermediate). `cols = 2560` is a multiple of 8, so the `/4` is exact.
* **output** = `out[rows_per_expert * e + r]`, i.e. the concatenation of what N
  single-expert `Gemv::run` calls would write. Nothing about consuming the
  projection changes.

Both layouts measured; they differ by < 2 % (§2.5), so the choice is a
layout/correctness decision, not a performance one.

### 2.3 API + call sequence for the forward-pass integrator

```rust
// once per (role, type): compile the pipeline
let batch = GemvBatch::new(gpu, ty, rows_per_expert, cols)?;   // 640x2560 or 2560x640

// once per expert-set change — NOT per token. The top-10 *set* changes per
// token, the blob offsets do not, so with a 10-slot ring the map is constant:
let blob_words: Vec<u32> = slots.iter().map(|s| s.word_offset).collect();
let map_host = batch.map_for_blobs(&blob_words, XLayout::Shared);
let map = upload(gpu, &map_to_bytes(&map_host))?;             // 8 B per row

// per token: one dispatch per projection instead of ten
batch.run(&w_slots, &x_mixed,  &gate_out, &map,      10)?;   // 10 x 640 rows
batch.run(&w_slots, &x_mixed,  &up_out,   &map,      10)?;   // (or fuse SwiGLU)
batch.run(&w_down,  &x_swiglu, &out,      &map_down, 10)?;   // PerExpert map
```

Sizes for the real shape: `blob_bytes()` = 704,000 B (IQ3_S gate 640×2560),
`total_bytes(10)` = 7.04 MB, `map_bytes(10)` = 51,200 B, `x_bytes(10, PerExpert)`
= 102,400 B. `map()` is available when the blobs really are end to end (GGUF
role-tensor order, one ring round-trip).

**Deliberately unsupported:** split-K inside a batch. The batched grid already
has 13 workgroups per CU and the reduce pass would need the indirection a second
time; §2.5 shows the batched quantised shapes sit at 57–101 % of the reference
bandwidth, which is where split-K stops helping (`01-dequant-gemv.md` §4.2).

### 2.4 Correctness

`tests/conformance.rs::batched_experts_match_the_per_expert_loop` places 5
experts **in a permuted slot order** (expert `e` in slot `perm[e]`), gives each
its own activation vector, and checks all 13 types × both `x` layouts:

| check | result |
|---|---|
| batched vs `reference::gemv` per expert (CPU, f64 dot) | 5.9e-8 … 8.7e-5 (tolerance 1e-3) |
| batched vs **the single-expert GPU kernel** on the same blob | **0e0 — bit-identical, 26/26 cells** |
| real GGUF blobs (experts 0..9), CPU reference | 2.1e-8 … 1.7e-7 (§2.6) |

Bit-identical output against the existing kernel is the strongest statement
available: the indirection changes *where* a row reads, not *what* it computes.

### 2.5 Measured — RX 6950 XT (RADV NAVI21, gfx1030)

Synthetic, expert shape 640×2560, per-expert `x` (the shared layout is within
1–2 % of every cell and is omitted). **Re-measured after the §4 hardening**, one
type per invocation, `timeout 90`, three 0.25 s windows, best reported:

| type | N | loop ms | batched ms | speedup | loop GB/s | batched GB/s | % of 445 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ3_S | 1 | 0.006 | 0.006 | 0.99× | 115 | 114 | 26 % |
| IQ3_S | 2 | 0.012 | 0.007 | 1.73× | 115 | 199 | 45 % |
| IQ3_S | 4 | 0.024 | 0.009 | 2.62× | 115 | 302 | 68 % |
| IQ3_S | 8 | 0.049 | 0.016 | 2.98× | 115 | 344 | 77 % |
| **IQ3_S** | **10** | **0.061** | **0.020** | **3.01×** | 115 | **347** | 78 % |
| IQ3_S | 16 | 0.098 | 0.030 | 3.25× | 115 | 373 | 84 % |
| IQ2_XS | 1 | 0.006 | 0.006 | 1.00× | 81 | 81 | 18 % |
| IQ2_XS | 2 | 0.012 | 0.007 | 1.74× | 81 | 141 | 32 % |
| IQ2_XS | 4 | 0.023 | 0.008 | 2.83× | 81 | 230 | 52 % |
| IQ2_XS | 8 | 0.047 | 0.014 | 3.35× | 81 | 272 | 61 % |
| **IQ2_XS** | **10** | **0.058** | **0.017** | **3.45×** | 81 | **280** | 63 % |
| IQ2_XS | 16 | 0.093 | 0.025 | 3.77× | 81 | 306 | 69 % |
| IQ2_XXS | 1 | 0.006 | 0.006 | 0.99× | 72 | 71 | 16 % |
| IQ2_XXS | 2 | 0.012 | 0.007 | 1.75× | 72 | 127 | 28 % |
| IQ2_XXS | 4 | 0.023 | 0.008 | 2.87× | 72 | 207 | 47 % |
| IQ2_XXS | 8 | 0.047 | 0.014 | 3.45× | 72 | 250 | 56 % |
| **IQ2_XXS** | **10** | **0.059** | **0.016** | **3.55×** | 72 | **256** | 58 % |
| IQ2_XXS | 16 | 0.093 | 0.024 | 3.90× | 72 | 282 | 63 % |
| IQ3_XXS | 1 | 0.006 | 0.006 | 0.99× | 102 | 102 | 23 % |
| IQ3_XXS | 2 | 0.012 | 0.007 | 1.82× | 102 | 187 | 42 % |
| IQ3_XXS | 4 | 0.025 | 0.009 | 2.80× | 102 | 286 | 64 % |
| IQ3_XXS | 8 | 0.049 | 0.015 | 3.27× | 102 | 335 | 75 % |
| **IQ3_XXS** | **10** | **0.061** | **0.018** | **3.38×** | 102 | **346** | 78 % |
| IQ3_XXS | 16 | 0.098 | 0.027 | 3.68× | 102 | 377 | 85 % |
| IQ4_NL | 1 | 0.007 | 0.007 | 0.99× | 141 | 140 | 31 % |
| IQ4_NL | 2 | 0.013 | 0.007 | 1.75× | 141 | 247 | 56 % |
| IQ4_NL | 4 | 0.026 | 0.010 | 2.60× | 141 | 367 | 82 % |
| IQ4_NL | 8 | 0.052 | 0.017 | 3.08× | 141 | 432 | 97 % |
| **IQ4_NL** | **10** | **0.065** | **0.021** | **3.18×** | 141 | **448** | 101 % |
| IQ4_NL | 16 | 0.105 | 0.031 | 3.42× | 141 | 482 | 108 % |
| Q2_0 | 1 | 0.005 | 0.005 | 0.99× | 89 | 88 | 20 % |
| Q2_0 | 2 | 0.010 | 0.006 | 1.87× | 89 | 166 | 37 % |
| Q2_0 | 4 | 0.021 | 0.007 | 3.02× | 89 | 268 | 60 % |
| Q2_0 | 8 | 0.041 | 0.011 | 3.66× | 89 | 325 | 73 % |
| **Q2_0** | **10** | **0.052** | **0.014** | **3.78×** | 89 | **336** | 76 % |
| Q2_0 | 16 | 0.083 | 0.020 | 4.15× | 89 | 369 | 83 % |
| IQ2_S | 1 | 0.006 | 0.006 | 0.99× | 88 | 87 | 20 % |
| IQ2_S | 2 | 0.012 | 0.007 | 1.73× | 88 | 152 | 34 % |
| IQ2_S | 4 | 0.024 | 0.008 | 2.84× | 87 | 248 | 56 % |
| IQ2_S | 8 | 0.048 | 0.015 | 3.26× | 87 | 285 | 64 % |
| **IQ2_S** | **10** | **0.060** | **0.018** | **3.35×** | 88 | **293** | 66 % |
| IQ2_S | 16 | 0.096 | 0.026 | 3.65× | 88 | 319 | 72 % |

N = 1 is the sanity cell: batched and single-expert are the same shape there and
the ratio is 0.99–1.00×, i.e. the indirection costs nothing measurable (and the
table-index masks added in §4.2 cost nothing either — every cell reproduces the
pre-hardening run within 1 %).

The other types are not routed experts, but the mechanism is type-independent
(same shape, N = 10):

| type | loop → batched ms | speedup | batched GB/s | % of 445 |
|---|---|---:|---:|---:|
| IQ4_XS | 0.064 → 0.023 | 2.81× | 384 | 86 % |
| Q4_K | 0.076 → 0.030 | 2.50× | 303 | 68 % |
| Q6_K | 0.100 → 0.056 | 1.79× | 241 | 54 % |
| BF16 | 0.060 → 0.031 | 1.91× | 1042 | 234 % |
| F32 | 0.057 → 0.041 | 1.37× | 1583 | 356 % |

**Q6_K is the informative outlier.** It is element-rate bound (0.31 T
elements/s, `01` §4.1): 10 experts of Q6_K already saturate the decode ALUs at
135 GB/s per expert, so batching only removes dispatch latency (1.79×) instead of
the 10× parallelism headroom Q2_0 exploits (3.78×). Beyond ~8 experts every type
converges on its own element-rate ceiling, not the memory ceiling — again the
"fewer instructions per element" item, not a batching item.

### 2.6 Real GGUF blobs (mmap, no copy) — experts 0..N-1 per role

Experts are contiguous inside each role tensor (`[ne0, ne1, 512]`), so experts
0..N are one zero-copy slice of the mapping; upload cost is the page-in of the
benchmarked blobs only (0.01–0.04 s for 10 experts):

| tensor | type | shape/expert | N | loop ms | batched ms | speedup | loop GB/s | batched GB/s | % of 445 | conformance |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blk.0.ffn_gate_exps` | IQ2_XS | 640×2560 | 1 | 0.006 | 0.006 | 1.00× | 81 | 81 | 18 % | 7.1e-8 |
| `blk.0.ffn_gate_exps` | IQ2_XS | 640×2560 | **10** | 0.058 | **0.017** | **3.41×** | 81 | 277 | 62 % | 9.5e-8 |
| `blk.0.ffn_gate_exps` | IQ2_XS | 640×2560 | 16 | 0.094 | 0.025 | 3.74× | 81 | 303 | 68 % | 9.5e-8 |
| `blk.0.ffn_down_exps` | IQ4_NL | 2560×640 | 1 | 0.004 | 0.004 | 0.97× | 251 | 243 | 55 % | 2.1e-8 |
| `blk.0.ffn_down_exps` | IQ4_NL | 2560×640 | **10** | 0.037 | **0.023** | **1.61×** | 248 | 398 | 90 % | 8.0e-8 |
| `blk.0.ffn_down_exps` | IQ4_NL | 2560×640 | 16 | 0.059 | 0.036 | 1.65× | 248 | 410 | 92 % | 8.0e-8 |
| `blk.8.ffn_gate_exps` | IQ3_S | 640×2560 | 10 | 0.061 | 0.021 | 2.96× | 115 | 340 | 76 % | 6.0e-8 |
| `blk.35.ffn_gate_exps` | IQ3_XXS | 640×2560 | 10 | 0.061 | 0.018 | 3.34× | 102 | 342 | 77 % | 1.3e-7 |
| `blk.1.ffn_gate_exps` | IQ2_XXS | 640×2560 | 10 | 0.059 | 0.017 | 3.51× | 72 | 253 | 57 % | 1.4e-7 |
| `blk.12.ffn_gate_exps` | IQ2_S | 640×2560 | 10 | 0.060 | 0.018 | 3.31× | 88 | 290 | 65 % | 1.1e-7 |
| `blk.1.ffn_down_exps` | Q2_0 | 2560×640 | 10 | 0.031 | 0.016 | 1.95× | 150 | 294 | 66 % | 3.5e-8 |

Real-blob numbers match the synthetic ones within 1–3 %, so the synthetic sweep is
representative of the artefact. **Note `down` is 2560 rows × 640 cols**, not 640
rows: it has 5.3 workgroups per CU already, which is why it gains only 1.6–2.0× —
never the bottleneck the gate/up projections are.

### 2.7 Radeon VII / MI50 (RADV VEGA20, gfx906) — same kernel, Wave64

| type | N | loop ms | batched ms | speedup | batched GB/s |
|---|---:|---:|---:|---:|---:|
| IQ3_S | 1 | 0.016 | 0.016 | 0.98× | 44 |
| IQ3_S | 10 | 0.157 | 0.058 | 2.71× | 121 |
| IQ3_S | 16 | 0.251 | 0.088 | 2.85× | 128 |
| Q2_0 | 1 | 0.014 | 0.014 | 1.00× | 32 |
| Q2_0 | 10 | 0.143 | 0.035 | 4.06× | 131 |
| Q2_0 | 16 | 0.228 | 0.051 | 4.48× | 145 |

Same conclusion as `01` §4.3 (this card's decode path is instruction-rate bound,
4–18 % of 825 GB/s), but the *latency* removal works identically. The batch does
not fix the MI50; it stops wasting it on 80-workgroup dispatches.

### 2.8 What this does to the token budget

Routed experts per decode token = 10 experts × 48 layers ≈ 0.90 GB. Projecting the
measured per-layer costs onto the model's real per-block type map
(`docs/phase-2/02-format-loader.md` §2.1: 8 IQ2_XXS/Q2_0, 7 IQ3_S/Q2_0,
6 IQ3_S/IQ4_NL, 6 IQ2_XS/Q2_0, 6 IQ2_S/Q2_0, 4 IQ2_XS/IQ4_NL, 4 IQ2_S/IQ4_NL,
3 IQ3_XXS/Q2_0, 3 IQ3_XXS/IQ4_NL, 1 IQ2_XXS/IQ4_NL — gate and up share a type):

| | per token, routed experts | effective GB/s |
|---|---:|---:|
| per-expert loop (what a forward pass would do today) | **7.34 ms** | 122 |
| one batched dispatch per projection | **2.62 ms** | 342 |

i.e. **−4.7 ms per token**, 2.8× aggregate. This is what makes the arithmetic in
`01-dequant-gemv.md` §5 honest: that report's 10.7–15.9 ms of GEMV per token
(63–93 tok/s) *assumed* the routed experts move at 271–402 GB/s, which the
per-expert loop could not do (72–115 GB/s). Without batching the same token costs
15.4–20.6 ms (49–65 tok/s); with it the assumed range is real. **The batch does not
create bandwidth, it removes 4.7 ms of dispatch latency per token.**

---

## 3. The remaining quant types: IQ4_XS, Q6_K, Q4_K, IQ2_S

### 3.1 Block layouts implemented (from `ggml-common.h`)

| type | id | block | bytes | fields |
|---|---:|---|---:|---|
| IQ4_XS | 23 | 256 | 136 | `d`, `scales_h:u16` (byte 2), `scales_l[4]`, `qs[128]`; per-32 scale `dl = d*(scale6−32)`, IQ4_NL value table, low nibbles = the group's first 16 elements |
| Q6_K | 14 | 256 | 210 | `ql[128]`, `qh[64]`, `scales[16]` **int8**, `d` **at the end** (byte 208); 6-bit value = `(nib | hi2<<4) − 32` |
| Q4_K | 12 | 256 | 144 | `d`, `dmin`, `scales[12]` via `get_scale_min_k4`, `qs[128]`; `d*sc*q − dmin*m` |
| IQ2_S | 22 | 256 | 82 | `d`, `qs[64]` (32 grid indices + 32 sign bytes), `qh[8]`, `scales[8]`; 10-bit grid index = `qs[l] | (qh[ib32]<<(8−2l) & 0x300)`, 1024-entry grid |

Geometry cross-checked three ways: `ggml-common.h`'s `static_assert`s, the
inventory spec (`docs/specs/model-artifacts-inventory.md` §3), and
`quant::geometry_matches_the_format_crate_for_real_block_types` (against
`breccia_format::GgmlType`). All four decode in the same unit-of-8 contract as the
existing types (`DEQUANT_UNIT(off, unit, out float v[8])`), so they reuse
`gemv_body.glsl` / `dequant_body.glsl` unchanged and their `_moe` batch variants
came for free.

### 3.2 Conformance

**(a) The CPU reference is faithful to `ggml-quants.c` — bit-exactly.** The oracle
is the reference clone's *own* `ggml-quants.c`, not a re-typed copy:

```
$ python3 tools/refcheck.py --blocks 2000
OK IQ2_XXS / IQ2_XS / IQ3_XXS / IQ2_S / IQ3_S: 2000 blocks / 512000 values bit-identical
OK IQ4_NL: 2000 blocks / 64000 values bit-identical
OK IQ4_XS / Q4_K / Q5_K / Q6_K: 2000 blocks / 512000 values bit-identical
OK Q2_0: 2000 blocks / 128000 values bit-identical
all 11 types bit-identical
```

Random block bytes include the degenerate scale cases (zero, subnormal, ±65504,
±inf, NaN), with the scale fields placed per type (`Q6_K` at byte 208,
`Q4_K`/`Q5_K` at 0 and 2).

**(b) GPU vs CPU** (`cargo test -p breccia-kernels --test conformance`, shapes
`1xN`, `3xN`, `640x2560`, `2560x640` + two tail shapes per type):

| type | dequant max rel | GEMV max rel (tolerance 1e-3) |
|---|---:|---:|
| IQ2_S | **0** | 6.3e-6 |
| IQ4_XS | **0** | 5.9e-5 |
| Q4_K | **0** | 3.0e-5 |
| Q6_K | **0** | 4.9e-5 |

Dequant is bit-identical at every shape (integer codes and fp16 scales are exact).
Also covered per type: +inf/−inf/NaN scales decode to non-finite values **and** the
finite block behind them still decodes exactly (§4.3), subnormal (±2⁻²⁴) and zero
scales decode exactly to the reference, split-K 1/2/3/5/8 agrees with the
reference, and the batched dispatch is bit-identical (§2.4).

Two porting bugs were caught here, both in the first draft of the new units:
IQ4_XS indexed `scales_l` with the *unit* index instead of the *scale* index
(`ib/2`, byte selector `i&1`, not `ib&1`), and used a 4-word stride per 32-element
group instead of 8. The CPU reference was already bit-identical, which is what
localised both to the shader.

### 3.3 Throughput — synthetic sweep, RX 6950 XT, GB/s (% of the 445 GB/s figure)

| type | 640x2560 | 2560x640 | 6400x2560 | 10240x2560 | 2560x6144 | 2560x2560 | 248320x2560 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ4_XS | 152 (34 %) | 236 (53 %) | 388 (87 %) | 413 (93 %) | 352 (79 %) | 329 (74 %) | **427 (96 %)** |
| Q6_K | 164 (37 %) | 185 (42 %) | 238 (53 %) | 243 (55 %) | 223 (50 %) | 217 (49 %) | **250 (56 %)** |
| Q4_K | 140 (31 %) | 196 (44 %) | 299 (67 %) | 312 (70 %) | 271 (61 %) | 256 (58 %) | **327 (73 %)** |
| IQ2_S | 97 (22 %) | 158 (36 %) | 276 (62 %) | 301 (68 %) | 260 (58 %) | 241 (54 %) | **290 (65 %)** |

Against the **measured read-only ceiling (561 GB/s, `01` §4.1)** the DRAM-bound
column is IQ4_XS 76 %, Q4_K 58 %, IQ2_S 52 %, Q6_K 45 %. Split-K factor: 3 for the
640-row shape, 1 elsewhere (the automatic heuristic, `01` §4.2).

IQ4_XS is now the fastest quantised type in the crate (96 % of the reference
figure): 16 table lookups per 32 elements and one scale per 32-element group is
cheaper per byte than IQ4_NL. Q6_K is the slowest (0.31 T elements/s: four nibble
extracts, a 2-bit merge and a signed int8 scale per 8 elements) — the same
element-rate wall Q5_K sits on.

### 3.4 Real-model tensors — which tensors use the new types, and how they run

203 tensors, 6.92 GB, **9.1 % of the model** (the task's 203 is exact):

| type | tensors | bytes | where |
|---|---:|---:|---|
| Q6_K | 71 | 864.5 MB | `attn_qkv` (14), `ssm_out` (19), `attn_output` (10), `attn_v` (9), `attn_k` (6), `attn_gate` (5), `attn_q` (4), `ffn_up_shexp` (3), `ffn_gate_shexp` (1) |
| IQ4_XS | 67 | 391.9 MB | `ffn_gate_shexp` (21), `attn_qkv` (11), `attn_gate` (11), `ffn_up_shexp` (10), `ssm_out` (6), `attn_q` (4), `attn_k` (2), `attn_v` (2) |
| Q4_K | 45 | 296.9 MB | `ffn_gate_shexp` (12), `attn_qkv` (8), `ffn_up_shexp` (7), `attn_gate` (7), `ssm_out` (4), `attn_q` (3), `attn_k` (2), `attn_v` (1), `attn_output` (1) |
| IQ2_S | 20 | 5.37 GB | `blk.{12,15,17,21,…}.ffn_{gate,up}_exps` — 10 blocks' routed experts, 512 experts each |

Real bytes through the kernel (`--real --tensor`, mmap, uploaded once):

| tensor | type | shape | MiB | RX GB/s (% of 445) | conformance |
|---|---|---|---:|---:|---:|
| `blk.0.attn_qkv.weight` | IQ4_XS | 10240×2560 | 13 | **413 (93 %)** | 1.4e-7 |
| `blk.0.ssm_out.weight` | IQ4_XS | 2560×6144 | 8 | 351 (79 %) | 2.6e-7 |
| `blk.0.ffn_gate_shexp.weight` | IQ4_XS | 640×2560 | 0.8 | 154 (35 %) | 8.6e-8 |
| `blk.4.attn_qkv.weight` | Q6_K | 10240×2560 | 20 | 244 (55 %) | 3.3e-7 |
| `blk.1.attn_gate.weight` | Q6_K | 6144×2560 | 12 | 231 (52 %) | 2.3e-7 |
| `blk.3.attn_output.weight` | Q6_K | 2560×6144 | 12 | 223 (50 %) | 2.3e-7 |
| `blk.2.ssm_out.weight` | Q4_K | 2560×6144 | 8 | 272 (61 %) | 4.1e-7 |
| `blk.2.ffn_gate_shexp.weight` | Q4_K | 640×2560 | 0.9 | 141 (32 %) | 1.7e-7 |
| `blk.12.ffn_gate_exps.weight` | IQ2_S | 327680×2560 | 256 | **292 (66 %)** | 7.5e-8 |

Real numbers track the synthetic cells (IQ4_XS 413 vs 412, Q6_K 244 vs 244,
`ssm_out` 351 vs 352, IQ2_S 292 vs 290), and the 256 MiB IQ2_S expert tensor is 8×
the Infinity Cache, so that one is a pure DRAM stream. The 640-row shared-expert
tensors are 31–55 % *slower* per byte — the same latency-bound shape §2.1
describes, and exactly the tensors (IQ4_XS ×21, Q4_K ×12, Q6_K ×1 shared-expert
projections) a forward pass should batch or fuse rather than launch alone.

---

## 4. The GPU fault: what happened, what changed

### 4.1 What the log says

Previous boot, RX 6950 XT (display card), process `conformance-6a0`, thread
`non_finite_scal`:

```
amdgpu 0000:03:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:6 pasid:894)
 Process conformance-6a0 pid 217321 thread non_finite_scal pid 217324
 in page starting at address 0x0000800100140000 from client 0x1b (UTCL2)
 GCVM_L2_PROTECTION_FAULT_STATUS:0x00600C31  PERMISSION_FAULTS:0x3
 [drm:gfx_v10_0_bad_op_irq] *ERROR* Illegal opcode in command stream
```

Reading it: a shader read (PERMISSION_FAULTS bits = read+write) at a virtual
address inside a GPU mapping, followed by the driver's recovery path reporting an
illegal CP packet. That is the fingerprint of **an access outside the bounds of a
bound storage buffer**, not of a numerics bug: a wrong value cannot fault, a wrong
*address* can. `non_finite_scales_do_not_corrupt_the_device` is the one test that
feeds ±inf/NaN *into the block decode*, so the suspicion that a non-finite scale
became an integer index is well founded — and it is the first thing I checked.

### 4.2 Root cause: two latent out-of-bounds hazards, both now closed

I could not reproduce the individual fault after the hardening (§4.4), so this is
an honest account of what the audit found rather than a "fixed line N". What it
found is that the crate had two properties that make an out-of-bounds read
*possible*, and one of them was newly reachable because of this work order:

**(a) Every storage buffer ended exactly at the last byte the kernel uses.** `W`
(weights), `X`, `O`, the 17,984-byte table buffer and the new row map were all
allocated to the exact size. The new IQ2_S table is the extreme case: with
`TBL_IQ2S_HI = 3472` and 1024 entries, the highest legal index is **word 4495 of
4496** — zero slack. A `u16`/`u32`/`vec4` load whose last access is wider than the
field being consumed (perfectly legal for a compiler to emit, and normal for a
vectorised tail read) then leaves the mapping. `gemv::TAIL_PAD = 64` now gives
every allocation slack; the kernels still only touch the documented range.

**(b) Table indices were bounded only by the caller's arithmetic.** Two of the new
units build a grid index from packet bits — `IQ2_S` masks 2 extra bits into a
10-bit index, `IQ4_XS` feeds a nibble to the value table — and the accessors
trusted them. The audit (below) shows the arithmetic *is* in range, but "the caller
is right" is not a property a GPU kernel can rely on to stay alive. Every accessor
now masks to its own table length:

```glsl
uint grid_iq2s_lo(uint i) { return T[TBL_IQ2S_LO + (i & 1023u)]; }  // 1024
uint grid_iq3s(uint i)    { return T[TBL_IQ3S_GRID + (i & 511u)]; } // 512
```

and `lib.rs::table_accessors_are_bounded` asserts, from the shader source, that
each mask matches its table size and that `base + entries <=
GPU_TABLE_WORDS_LEN`.

**Is the "NaN → huge index" mechanism real in this crate?** No — and that is now
mechanically checked. The audit of every float-derived value in the decode units:

| value | used for | can reach an address? |
|---|---|---|
| `d`, `dmin`, `db`, `dl`, `dm` (fp16 → f32, incl. inf/NaN) | multiplicand / addend in `v[j]` | **no** — never cast, never used as index |
| `sc` (from `int(ls) - 32`, integer packet bits) | multiplicand | no |
| grid/sign codes (`qs`, `qh`, `scales`, `aux` bytes) | **index** into the table | yes → masked at the accessor |
| `row`, `blk`, `off`, `bpr`, `u`, `j` | offsets | derived from push constants + `gl_*`, bounds-checked against `pc.rows`/`pc.cols` |

`lib.rs::no_float_to_integer_conversion_in_a_decode_unit` fails the build if any
decode unit ever applies `int(`/`uint(` to one of the scale identifiers — the
static half of the same invariant.

### 4.3 Tests added for exactly this

| test | asserts |
|---|---|
| `non_finite_scales_do_not_corrupt_the_device` | +inf, −inf and NaN in **every** scale field of a type (`d` and `dmin` for Q4_K/Q5_K, the trailing `d` for Q6_K): the poisoned blocks decode non-finite **and the finite block behind them still decodes exactly** — the observable proof that the faulting access path is intact |
| `denormal_and_zero_scales_decode_exactly` | ±2⁻²⁴ and 0.0 scales decode bit-exactly to the reference (no FTZ shortcut), `d = 0` gives exactly `0.0` |
| `table_accessors_are_bounded` | every table accessor's mask == its table size, and the last entry fits the uploaded buffer |
| `no_float_to_integer_conversion_in_a_decode_unit` | no scale value is cast to an integer anywhere in `blocks.glsl` |
| `batched_experts_match_the_per_expert_loop` | the row map cannot make a row read another expert's blob (permuted slots) |

### 4.4 Verifying safely (the procedure now in `AGENTS.md`)

The original run broke two rules that are cheap to keep: it used the **display
card** for a brand-new shader path, and it ran **all 13 types in one process**, so
the fault was unattributable. Both are now enforced by the harness:

* `BRECCIA_QTYPE=IQ4_XS` (or a comma list) narrows **every** test in
  `tests/conformance.rs` to the named types.
* `BRECCIA_KERNEL_DEVICE=0000:0e:00.0` selects the GPU **by PCI address**; the
  Radeon VII (MI50) is not driving a display, so a fault there does not take the
  desktop with it.
* every GPU invocation is wrapped in `timeout 90`, one GPU process at a time, with
  `VK_LAYER_KHRONOS_validation` for first runs of new shader code.

Re-verification order actually used: **13 separate MI50 runs (one type each, all
8 tests, validation on) → one full RX run (all types) → the benchmarks.** Result:
`8 passed` per type, `0` validation messages, no fault, machine healthy. Logs:
`p3/safe/mi50-<TYPE>.log` (13 files), `p3/safe/rx-all.log`.

---

## 5. Verification (commands and results)

```bash
cargo build --workspace                                            # Finished
cargo test --workspace                                             # all suites green
cargo clippy -p breccia-kernels --all-targets -- -D warnings        # clean
cargo clippy -p breccia-cli     --all-targets -- -D warnings        # clean
cargo fmt --check                                                  # clean

# correctness, one type per invocation, MI50 first (safe card), then the RX:
for T in IQ3_S Q2_0 IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ4_NL Q4_K Q5_K Q6_K BF16 F32 IQ4_XS; do
  timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation BRECCIA_QTYPE=$T \
      BRECCIA_KERNEL_DEVICE=0000:0e:00.0 \
      cargo test -p breccia-kernels --test conformance -- --test-threads=1 --nocapture
done                                                                # 8 passed each, 0 VU
timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation BRECCIA_KERNEL_DEVICE=0000:03:00.0 \
    cargo test -p breccia-kernels --test conformance -- --test-threads=1   # 8 passed, 0 VU

# benchmarks (one invocation per type, each bounded):
timeout 90 ./target/release/breccia bench gemv --device 0000:03:00.0 \
    --type IQ3_S --experts 1,2,4,8,10,16                              # batched sweep
timeout 90 ./target/release/breccia bench gemv --device 0000:03:00.0 \
    --real --layer 0 --role gate --experts 1,2,4,8,10,16               # real expert blobs
timeout 90 ./target/release/breccia bench gemv --device 0000:03:00.0 \
    --real --tensor blk.0.attn_qkv.weight --quick                      # new type, real bytes
timeout 90 ./target/release/breccia bench gemv --device 0000:0e:00.0 \
    --type Q2_0 --experts 10 --quick                                   # MI50 batched

# bit-exactness of the CPU reference against llama.cpp's own quants:
python3 tools/refcheck.py --blocks 2000      # 11 types bit-identical
```

Artifacts: `p3/safe/v2-*.json`, `p3/safe/mi50-*.log`, `p3/safe/rx-all.log`,
`p3/v2-newtypes.log`, `p3/conformance.log`, `p3/refcheck-2000.log`,
`p3/validation-*.log`.

## 6. Test inventory added

| test | covers |
|---|---|
| `batched_experts_match_the_per_expert_loop` | 13 types × 2 `x` layouts × 5 experts in permuted slots, tail shapes: batched == single-expert kernel bit-for-bit, and == CPU reference ≤ 1e-3 |
| `non_finite_scales_do_not_corrupt_the_device` | +inf/−inf/NaN in every scale field of every type; neighbour block still exact |
| `denormal_and_zero_scales_decode_exactly` | ±2⁻²⁴ and 0.0 scales exact; `d = 0` → exactly 0.0 |
| `table_accessors_are_bounded` | table-index masks vs table sizes vs buffer length (from the shader source) |
| `no_float_to_integer_conversion_in_a_decode_unit` | no scale value can become an index |
| `reference::dequant_block` via `refcheck.py` | the four new decode ports vs `ggml-quants.c`, bit-for-bit, 2000 blocks/type |
| `quant::geometry_matches_the_format_crate` | 13 types against `breccia_format::GgmlType` |
| `BRECCIA_QTYPE` / `BRECCIA_KERNEL_DEVICE` selectors | one type per invocation, by PCI identity |

## 7. Open issues

1. **fp16 accumulate (item #3) not implemented.** `breccia_vk::ctx::Gpu::kernel`
   hard-codes a 16-byte push-constant range and its four words are fully used
   (`rows/cols/bpr/splits`), so a `pc.flags` bit needs either a 20-byte range in
   `breccia-vk` (outside this work order's write scope) or a second pipeline per
   type (`gemv_<type>_f16acc`, +13 shader variants + an `Accum` argument on
   `Gemv`/`GemvBatch`). The expected result is also negative on accuracy: a decode
   row sums 2560 products and fp16 accumulation over that many terms lands around
   1e-2 relative, two orders outside the 1e-3 bar used everywhere here. The
   experiment worth running is f32-vs-f16 accumulate on the element-bound types
   (Q6_K, Q5_K, IQ2_XXS) reporting both throughput *and* error; ~1 day with the
   pipeline duplication above.
2. **The fault itself was not reproduced after hardening.** 13 per-type MI50 runs
   and a full RX run of the exact crashing test are clean under the validation
   layer, and the two hazards that can produce that signature are gone (§4.2). If
   it recurs, the per-type selector plus `timeout 90` makes it attributable in one
   run; `journalctl -b -1 | grep -E "amdgpu.*(page fault|Illegal opcode)"` plus the
   `BRECCIA_QTYPE` value is then the whole report.
3. **The element-rate wall is now the only wall.** On the 10-expert shape the
   quantised types sit at 57–101 % of the 445 GB/s figure, so the routed experts
   are close to bandwidth-bound; what remains is Q6_K/Q5_K/IQ2_XXS at 0.31–1.1 T
   elements/s. The int8 dot-product path (`v_dot4_i32_i8`, `01` §7 item 2) is the
   next lever and now applies to 244 GB/s of Q6_K weight traffic per token too.
4. **`F16` (ggml id 1) still has no kernel** — one tensor,
   `blk.1.ple_conv1d.weight` [4, 10240] = 80 KB. It is the only type in the
   released model without one; adding it is 10 lines (a virtual 8-element block
   like BF16). Every other type present (13 of 14) now has a kernel.
5. **`--real --tensor X --expert E` ignores `--expert`** (predates this work): the
   tensor path passes `None` for the expert selector, so the whole tensor is
   benchmarked. The IQ2_S cell in §3.4 is therefore the 512-expert tensor, which is
   the more useful number anyway.
6. **The loop baseline re-dispatches blob 0** rather than walking all N blobs
   (`bench.rs::run_batch_cell`): at 640×2560, 10 experts is 7 MB of weights, i.e.
   inside the 128 MiB Infinity Cache either way, so both variants read the same
   DRAM bytes and the cell measures dispatch latency — the effect under test. A
   caller wanting the N-separate-buffers variant (one ring slot per expert) needs a
   base-offset push constant on the single-expert kernel, which the fixed 16-byte
   range does not have (same constraint as item 1).
7. **Fused SwiGLU** (`01` §7 item 5) is the natural companion: gate and up batched
   in one dispatch with `silu(g)*u` written out would save one 640×10-wide round
   trip and one dispatch per layer per token.
8. **Prefill is still un-tiled** (batch > 1 tokens): the indirection is per row, so
   a multi-token version needs the `x` half of the map to become a stride
   (token-in-block layout) or a per-(token,expert) map. Not attempted.

## 8. Attributions

* **New decode semantics** — ported from llama.cpp `ggml/src/ggml-quants.c`
  (`dequantize_row_iq2_s`, `_iq4_xs`, `_q4_K`, `_q6_K`), MIT, commit **6050b46**.
  Both the GLSL (`shaders/blocks.glsl`) and the Rust reference
  (`src/reference.rs`) carry the per-type note of which C function they follow.
* **Batched dispatch** — the row-indirection idea follows ggml-vulkan's
  `MUL_MAT_ID` / `mul_mat_vec_*` id path (MIT, same commit); the layout (one
  contiguous blob buffer + per-row base, and a per-row activation base for
  per-expert `x`) is specific to this engine's M3 slot ring
  (`crates/breccia-vk/src/transfer.rs`). No ggml-vulkan code was copied for it.
* **IQ2_S grid** — `iq2s_grid` is *data* from llama.cpp's `ggml-common.h` (MIT),
  copied verbatim by `tools/gen_tables.py`; `lib.rs::table_offsets_*` and
  `table_accessors_are_bounded` cover its two new `TBL_*` regions.
* No code from IsoQuant or RotorQuant (unlicensed) was consulted or copied.
