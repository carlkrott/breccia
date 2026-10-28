# 03 — Three kernel gaps closed: F16, fp16 accumulate, `--expert`

Follow-ups from `docs/phase-3/02-expert-batching-quants.md` §7 items 1, 4 and 5:

1. **F16 (ggml id 1) had no kernel** — one tensor, `blk.1.ple_conv1d.weight`
   `[4, 10240]` = 80 KiB, the last type in the released model without one.
2. **fp16 accumulate (item #3 of the M2 work order)** was deferred because
   `breccia-vk`'s pipeline layout fixed the push-constant block at 16 bytes and all
   four words were used.
3. **`--real --tensor <name> --expert <e>` ignored `--expert`** — the tensor path
   passed `None` for the expert selector, so the whole 512-expert tensor was
   benchmarked.

Date: 2026-10-26 · GPUs idle at the start of every run (`pgrep llama-server` empty,
`gpu_busy_percent` 0 / 3 on the RX and 0 on the MI50) · no service was started or
stopped · artifacts in `<SCRATCH>/opencode/p3-gaps/`.

**Status: all three closed.** `cargo build/test --workspace`,
`cargo clippy -p breccia-kernels -p breccia-vk -p breccia-cli --all-targets --
-D warnings`, `cargo fmt --check`: green. **14 types**, each conformance run
**one type per process, `timeout 90`, validation layer on, MI50 first**: 10/10
tests pass per type, **0 validation messages** in all 18 logs, no fault.
`tools/refcheck.py` still reports the 11 quant types bit-identical to
`ggml-quants.c`.

**Headline numbers.** F16 dequant is **bit-identical** to the CPU reference at
every shape and its real 80 KiB tensor decodes at **3.7e-10** relative error;
fp16 accumulation is **0.0–1.0 % faster** and **up to 6.8e-1 worst-row wrong**, so
**f32 stays the default and fp16 is not recommended**; `--expert` now slices
473,600 B (= expert 3 of `blk.0.ffn_gate_exps`) out of the mapped tensor and runs
the correct 640x2560 shape.

---

## 1. What landed

| Path | Change |
|---|---|
| `crates/breccia-vk/src/ctx.rs` | push-constant range **16 → 32 B** (offset 0; existing 16-byte shader blocks keep working). Comment records why. |
| `src/quant.rs` | `QuantType::F16` — ggml id **1**, virtual 8-element block, 16 B/block, 16 bpw; `ALL` is now **14**. F16 joins the F16/BF16/F32 virtual-block exemption in the geometry test. |
| `shaders/blocks.glsl` | `dequant_f16_unit` (`f16_at` = exact `unpackHalf2x16`), next to the BF16/F32 virtual-block units. |
| `shaders/dequant_f16.comp`, `gemv_f16.comp`, `gemv_f16_moe.comp` | three wrappers (QK 8, 8 block words), same shape as the BF16 ones. |
| `shaders/gemv_*.comp` (28 files) | a conditional `#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require` for the fp16 variant (only enabled when `BRECCIA_F16_ACCUM` is defined). |
| `shaders/gemv_body.glsl` | `#ifdef BRECCIA_F16_ACCUM`: two accumulators packed as one `f16vec2` (one packed f16 add per block), converted to f32 before the cross-lane reduce. The f32 path is untouched. |
| `build.rs` | every fused GEMV is compiled **twice** (`gemv_<stem>` and `gemv_<stem>_f16acc` via `-DBRECCIA_F16_ACCUM`), 14 types × 2 bodies = 56 GEMV SPIR-V modules, each `spirv-val`-checked. |
| `src/gemv.rs`, `src/batch.rs` | `Accum { F32 (default), F16 }` + `Gemv::with_accum` / `GemvBatch::with_accum`; `new()` still means fp32. F16 is rejected up front when the device lacks `shaderFloat16`. |
| `src/reference.rs` | `F16` decode arm (`f16_to_f32`, exact). |
| `src/real.rs` | `expert_slice` (expert-major `[cols, rows, experts]` slicing + absolute file range), `usable_shape` (row folding), `verify_expert_against_pack` (slice must equal `ExpertBlock::expert_ranges`), F16 synthetic data, `--expert` honoured on the tensor path, `ScaleOffsets` for F16. |
| `src/bench.rs` | `accum` option, `accum`/`conformance_max_rel` on synthetic cells, `accum`/`file_range`/`folded_rows` on real cells. |
| `tests/conformance.rs` | +`fp16_accumulate_has_a_documented_tolerance_and_f32_is_unchanged`, +`batched_fp16_accumulate_matches_the_single_expert_path` (10 tests, was 8). |
| `crates/breccia-cli` | one flag: `breccia bench gemv --accum f32|f16` (default f32) + the file-range/accum lines in the summary. `profile`/`serve`/`transfer` untouched. |

### 1.1 Why the push-constant range had to change, and what it is now

`rows/cols/bpr/splits` filled the 16-byte block exactly, so a fifth word needed a
second pipeline layout per shader. The range is now **32 B at offset 0**:
`cmd_push_constants` may push a *prefix* of the declared range, so every existing
shader that declares (and receives) 16 bytes behaves identically — proven by the
unchanged 16-byte pushes in `bench.rs`/`transfer_bench.rs` and a
`breccia bench transfer` run under the validation layer (§5). Every device here
reports `maxPushConstantsSize` ≥ 128 (the Vulkan-guaranteed minimum is also 128),
so the headroom is free.

Only the *layout* widened; the fp16 selection itself is a **separate pipeline**
(`_f16acc`), not a runtime `pc.flags` branch, for two measured reasons: the f32
pipeline stays bit-identical to what phase-3 `01`/`02` measured, and the inner
loop keeps no uniform branch. The reserved range is what the *next* word needs —
the per-row weight base that `02` §7 item 6 wants for an N-buffer loop baseline.

---

## 2. Gap 1 — F16

### 2.1 Conformance (GPU vs CPU reference)

`cargo test -p breccia-kernels --test conformance`, one type per invocation,
`VK_LAYER_KHRONOS_validation=on`, shapes `1xN`, `3xN`, `640x2560`, `2560x640` +
two tail shapes (`7x27`, `2x9` for the 8-element block):

| check | RX 6950 XT | MI50 |
|---|---|---|
| dequant max rel (all 6 shapes) | **0e0 — bit-identical** | **0e0 — bit-identical** |
| GEMV max rel (tolerance 1e-3) | 1.53e-5 | 2.37e-5 |
| split-K 1/2/3/5/8 | agrees with the reference | agrees |
| non-finite / subnormal / zero scale tests | skipped (no scale field), as for BF16/F32 | skipped |
| batched vs single-expert kernel | **0e0 bit-identical** | **0e0 bit-identical** |

f16→f32 is exact and F16 has no scale, so dequant being bit-identical is the
expected result; it is stated because it is the strongest available statement.

### 2.2 The real tensor: `blk.1.ple_conv1d.weight`

Located with `cargo run -p breccia-format --release --example inspect`:
`blk.1.ple_conv1d.weight [4, 10240] F16 81920 B`, the only F16 tensor in the
model (`docs/specs/model-artifacts-inventory.md` §4.3).

A conv1d kernel with **4-element rows** is not addressable against the kernel's
8-element virtual block, so `real::usable_shape` folds two adjacent rows into one
(the tensor is contiguous, no byte moves): `[4, 10240]` → **5120x8**, and
`5120 × 1 block × 16 B = 81,920 B` — the whole tensor, read exactly once. The
report says `rows folded to a whole kernel block` so the shape is not mistaken for
a projection.

```bash
timeout 90 ./target/release/breccia bench gemv --device 0000:03:00.0 \
    --real --tensor blk.1.ple_conv1d.weight --json p3-gaps/real-rx-f16.json
timeout 90 ./target/release/breccia bench gemv --device 0000:0e:00.0 \
    --real --tensor blk.1.ple_conv1d.weight --json p3-gaps/real-mi50-f16.json
```

| device | shape | ms/pass | GB/s | conformance (16 rows) |
|---|---|---:|---:|---:|
| RX 6950 XT | 5120x8 (folded from [4,10240]) | 0.002 | **38.5** (9 % of 445) | **3.72e-10** |
| MI50 | 5120x8 | 0.007 | **12.3** (1 % of 825) | **3.77e-10** |

**The 38.5 GB/s is not a bandwidth number and should not be read as one.** The
tensor is 80 KiB: 5120 rows of exactly one block, i.e. 640 workgroups of one
warp-iteration each — pure latency, the same regime as the 640-row expert shape
in `02` §2.1, with an 8.6× smaller working set (80 KiB vs 0.67 MiB) and therefore
even less work to hide the latency behind. The kernel itself streams at memory rate
when the matrix is big enough (`--type F16 --rows 10240 --cols 2560`, 50 MiB,
synthetic F16): **RX 1685.8 GB/s** (379 % of 445 — Infinity-Cache resident, like
BF16/F32) and **MI50 712.6 GB/s** (86 % of 825), conformance 9.2e-7 / 4.2e-7.

`QuantType::ALL` is 14 and every exhaustive match/test that assumed 13 was
updated (`ALL`, `name`, `ggml_id`, `shader_stem`, geometry, `gemv_spv`,
`dequant_spv`, `gemv_moe_spv`, `synthetic_inputs`, `scale_offsets`, the
F16/BF16/F32 skips in the three scale tests, and the geometry test exemption).
`bench::default_types` is the model-weighted sweep and is deliberately unchanged
(the model's only F16 tensor is the 80 KiB `ple_conv1d`; F16 is reachable with
`--type F16`).

---

## 3. Gap 2 — fp16 accumulate behind a flag

### 3.1 What the path does

`Accum::F16` compiles the `_f16acc` variant: the per-block 8-element partial
product is still formed in f32 (the dequant stays exact), then added to a
**`f16vec2` holding the two alternating accumulators** — one packed f16 add per
block, `V_PK_ADD_F16` on gfx9/gfx10 instead of two scalar adds. After the loop the
pair is summed and converted to f32, and the 32-lane cross-lane reduction runs in
f32 (the same 5 shuffle-adds as the f32 path; 16-bit subgroup shuffles are not
universally supported and 5 adds are not where the error is). The runtime is
rejected when the device lacks `shaderFloat16`.

Accuracy is expected to be poor by construction: a 2560-column decode row is 10
super-blocks, so each lane rounds its running sum **10 times** to an 11-bit
mantissa, and the error is relative to the *magnitude of the terms*, not to the
result — rows whose two halves cancel amplify it without bound.

### 3.2 Accuracy (RX, `fp16_accumulate_has_a_documented_tolerance_and_f32_is_unchanged`)

Relative error per row = `|gpu − cpu_f64| / max(1, |cpu|)`; median / p95 / max over
every row of the matrix. f32 is the same test on the same bytes.

| type | shape | f32 max (bar 1e-3) | f16 median | f16 p95 | f16 max |
|---|---|---|---:|---:|---:|
| IQ3_S | 6400x2560 | **1.42e-5** | 4.59e-4 | 6.47e-3 | 9.67e-2 |
| IQ3_S | 2560x640 | **8.64e-6** | 3.07e-4 | 3.34e-3 | 3.40e-2 |
| Q6_K | 6400x2560 | **1.51e-4** | 4.55e-4 | 6.15e-3 | **6.77e-1** |
| Q6_K | 2560x640 | **2.79e-5** | 3.32e-4 | 3.92e-3 | 3.09e-1 |
| F16 | 6400x2560 | **2.45e-5** | 5.58e-4 | 6.93e-3 | 1.60e-1 |
| BF16 | 6400x2560 | **2.58e-5** | 5.54e-4 | 6.35e-3 | 1.07e-1 |

The documented bars in the test are **median ≤ 2e-2 and p95 ≤ 2e-2** (the
"~1e-2 for a 2560-term sum" expectation holds for the typical row) plus a
**worst-row ≤ 1.0** sanity bound — a row worse than 100 % is not a precision loss,
it is a different answer. The tail is the story: 1 row in 6400 of Q6_K is **68 %
wrong**, and a single 248,320-row F16 run measured a worst row of **2.5e-1**. The
MI50 reproduces the same numbers (IQ3_S 6400x2560: f32 2.1e-5, f16 median 4.6e-4,
p95 6.5e-3, max 9.7e-2) — this is IEEE f16, not a driver quirk.

The default conformance tolerances were **not** touched: f32 stays at 1e-3 in
`gemv_matches_cpu_dot_for_every_type`, and all 14 types pass it.

### 3.3 Throughput (RX, best of 3 windows, 3 independent runs per cell)

```bash
for A in f32 f16; do for S in "6400 2560" "248320 2560"; do set -- $S; \
  timeout 90 ./target/release/breccia bench gemv --device 0000:03:00.0 \
    --type IQ3_S --rows $1 --cols $2 --accum $A; done; done   # and Q6_K
```

| type | shape | f32 GB/s (3 runs) | f16 GB/s (3 runs) | delta |
|---|---|---|---:|---:|
| IQ3_S | 6400x2560 (top-10 experts) | 336.8 / 335.4 / 335.2 | 336.1 / 335.5 / 335.6 | **−0.2 %** |
| IQ3_S | 248320x2560 (DRAM-bound) | 371.6 / 371.2 / 371.2 | 371.7 / 372.5 / 372.2 | **+0.2 %** |
| Q6_K | 6400x2560 | 241.6 / 240.9 / 241.4 | 243.9 / 243.9 / 243.7 | **+1.0 %** |
| Q6_K | 248320x2560 | 252.0 / 252.5 / 252.3 | 254.3 / 254.7 / 254.7 | **+0.9 %** |
| IQ3_S (MI50) | 6400x2560 | 121.5 / 121.4 | 121.9 / 121.8 | +0.3 % |
| Q6_K (MI50) | 6400x2560 | 97.6 / 97.6 | 98.1 / 98.0 | +0.5 % |

### 3.4 Recommendation — **do not use fp16 accumulate**

* **Speed:** 0.0–1.0 %. The only cell that gains is Q6_K, the most
  element-rate-bound type, and 1 % is the same order as the run-to-run spread
  (0.1 % median, 3.8 % max per `01` §4.1). Nothing here survives contact with the
  token budget: 1 % of the ~2.6 ms/token routed-expert cost is 26 µs.
* **Accuracy:** the typical row is 1e-3–7e-3, but the p95 is already at the 2e-2
  bar and the worst row of a 6400-row matrix is 10–68 % wrong. That is not a
  numerically safe default for a decode path whose logits feed routing and
  sampling.
* **Verdict:** `f32` stays the default in the API and the CLI
  (`--accum` defaults to `f32`); the `_f16acc` pipelines stay in the crate as the
  documented measurement path (they cost 28 shader variants and ~0 runtime) so a
  future experiment — e.g. f16 accumulation *within* a 256-element super-block,
  resetting to f32 per super-block — has a baseline to beat.

---

## 4. Gap 3 — `--real --tensor <name> --expert <e>`

### 4.1 What was wrong

`real::run()` called `run_tensor(dev, opts, tensor, None)` for the `--tensor`
path, so `--expert` was silently dropped and the whole tensor was benchmarked.

### 4.2 What it does now

* `--expert e` on a 3-dim `[cols, rows, experts]` tensor slices the expert's
  contiguous `bytes/experts` range (`real::expert_slice`) — the same expert-major
  layout `breccia_format::ExpertBlock` uses for the pack blob, where one GGUF role
  tensor is one `[gate|up|down]` role slice of `blob_offsets()`.
  `verify_expert_against_pack` cross-checks the computed range against
  `ExpertBlock::expert_ranges`, and errors if they disagree (they do not).
* The shape comes from the tensor's own dims: gate/up `[2560, 640, 512]` →
  **640x2560**, down `[640, 2560, 512]` → **2560x640**, per expert.
* The summary prints the absolute file byte range used.
* `--expert` on a 2-dim tensor is a clear error, as is an out-of-range index (both
  verified, §5).

```bash
# whole tensor (the old behaviour, still what you get without --expert)
timeout 90 ./target/release/breccia bench gemv --device 0000:03:00.0 \
    --real --tensor blk.0.ffn_gate_exps.weight --quick
# one expert out of it
timeout 90 ./target/release/breccia bench gemv --device 0000:03:00.0 \
    --real --tensor blk.0.ffn_gate_exps.weight --expert 3
```

| invocation | shape | file range | ms/pass | GB/s | conformance |
|---|---|---|---:|---:|---:|
| RX `blk.0.ffn_gate_exps` (no `--expert`) | 327680x2560 | whole 231 MiB tensor | 0.787 | 308.0 (69 %) | 1.63e-7 |
| RX `blk.0.ffn_gate_exps --expert 3` | **640x2560** | **0x4944ed80..0x494c2780** (473,600 B) | 0.006 | 84.4 (19 %) | 1.17e-7 |
| MI50 `blk.0.ffn_gate_exps --expert 3` | 640x2560 | 0x4944ed80..0x494c2780 | 0.017 | 28.4 (3 %) | 1.09e-7 |

473,600 B is exactly `473600 × 512 = 242,483,200 B` ÷ 512 for this IQ2_XS tensor,
i.e. the slice is expert 3's own rows; the 84.4 GB/s is the expected
latency-bound per-expert shape (`01` §4.1: 81 GB/s for IQ2_XS at 640x2560), which
is what the whole-tensor number had been hiding.

### 4.3 Regression tests (no GPU)

* `real::tests::expert_slice_walks_a_synthetic_multi_expert_tensor` — a synthetic
  `[16, 4, 3]` tensor: the three expert slices tile the buffer exactly once, each
  slice starts at `e * per`, is that expert's own bytes, and the file range is
  `file_start + e*per`; out-of-range and 2-dim inputs are rejected.
* `real::tests::rows_shorter_than_a_kernel_block_fold_to_whole_blocks` — the
  `[4, 10240]` F16 fold to 5120x8 still accounts for all 81,920 B; aligned shapes
  are untouched.

---

## 5. Safety evidence (AGENTS.md GPU crash safety)

Every GPU run below was wrapped in `timeout 90`, ran **one process at a time**
with both cards idle, used `VK_LAYER_KHRONOS_validation`, and the **MI50
(`0000:0e:00.0`, non-display) went first**; the RX (`0000:03:00.0`) was only
touched after the MI50 passed. No service was started or stopped
(`qwen-dual`, `strata-rx`, `d1-mi50`: inactive before and after).

```bash
cargo build --workspace && cargo test --workspace        # green, GPU tests skipped (no device env)
timeout 600 cargo clippy -p breccia-kernels -p breccia-vk -p breccia-cli \
    --all-targets -- -D warnings && cargo fmt --check     # clean

# correctness, one type per invocation, MI50 first:
for T in F16 IQ3_S Q6_K Q2_0 IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ4_NL IQ4_XS Q4_K Q5_K BF16 F32; do
  timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
      BRECCIA_KERNEL_DEVICE=0000:0e:00.0 BRECCIA_QTYPE=$T \
      cargo test -p breccia-kernels --test conformance -- --test-threads=1
done                                                     # 10 passed each, 0 VU

# then the key paths on the display card, same one-type-per-process rule:
for T in F16 IQ3_S Q6_K BF16; do
  timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
      BRECCIA_KERNEL_DEVICE=0000:03:00.0 BRECCIA_QTYPE=$T \
      cargo test -p breccia-kernels --test conformance -- --test-threads=1
done                                                     # 10 passed each, 0 VU

# the push-constant range change also affects the Phase-2 shaders in breccia-vk:
timeout 120 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
    ./target/release/breccia bench transfer --device 0000:0e:00.0 --quick   # normal, 0 VU

# CPU reference unchanged:
cd crates/breccia-kernels && python3 tools/refcheck.py --blocks 200   # 11 types bit-identical
```

Logs: `p3-gaps/mi50-<TYPE>.log` (14), `p3-gaps/rx-<TYPE>.log` (4),
`p3-gaps/accum-rx.txt`, `p3-gaps/real-*.json`. Validation-message count
(`grep -ciE "VUID|Validation Error|Validation Warning"`): **0** in all 18
conformance logs.

Kernel log after the whole window: `journalctl -k -b | grep -cE
"amdgpu.*(page fault|Illegal opcode)"` = **0**. (The kernel log does carry 395
`amdgpu_dm_plane_helper_prepare_fb: Failed to pin framebuffer with error -22`
lines, but they start at 10:54:44 — before the first benchmark of this work order
— recur every ~5 min from the desktop compositor, and are a display-path error,
not a GPU fault. The systemd unit whose description mentions "amdgpu page faults"
is a watcher, not a fault record.)

---

## 6. Remaining gaps

1. **fp16 accumulate is documented as a negative result** (§3.4). Any future
   attempt should reset to f32 per super-block (or accumulate 32-lane partials in
   f32 and only the group reduction in f16), and must beat 1 % to be worth a
   shader variant — the current 2560-term f16 sum also fails the 1e-3 default bar
   in the tail, which is why `--accum` defaults to `f32`.
2. **F16 has no large tensor to exercise in this release.** The only F16 tensor is
   the 80 KiB `ple_conv1d` conv kernel; its real throughput number (38.5 GB/s RX)
   is a latency measurement, and the kernel's memory-rate behaviour is shown with
   the 50 MiB synthetic cell instead (1686 GB/s RX, 713 GB/s MI50). A future
   artefact with F16 matrices will exercise the streaming path for real.
3. **The real F16 run is `5120x8`, not `[4, 10240]`** (rows folded, §2.2). That is
   the only way to address 4-element rows with an 8-element decode unit; the
   bytes and the decode are the tensor's own.
4. **The 32-byte push-constant range currently carries 16 bytes.** The extra word
   is reserved for the per-row weight base that `02` §7 item 6 needs (an
   N-separate-buffer loop baseline); adding it is now a shader + host change with
   no `breccia-vk` work.
5. **Types still without kernels**: Q2_K, Q3_K, Q8_0, Q4_0, Q5_0 (`quant::NOT_IMPLEMENTED`,
   none in this release). Every type present in the model (14 of 14) has a kernel
   as of this report.
6. **Prefill remains un-tiled** and **SwiGLU is not fused** (`01` §7 items 4–5);
   `--experts` batching is still decode-only, unchanged here.

---

## 7. Attributions

* F16 decode semantics: `dequantize_row_f16` in llama.cpp `ggml/src/ggml-quants.c`
  (MIT, commit 6050b46) — `GGML_FP16_TO_FP32`, implemented here as the exact
  `unpackHalf2x16` widening in `blocks.glsl` and `f16_to_f32` in `reference.rs`.
  No new data tables.
* Everything else in this report reuses the existing ported decode units and the
  kernel structuring already attributed in `01-dequant-gemv.md` §8 and
  `02-expert-batching-quants.md` §8. `f16vec2` accumulation, the `_f16acc` build
  variant, `usable_shape` and `expert_slice` are original to this crate.
