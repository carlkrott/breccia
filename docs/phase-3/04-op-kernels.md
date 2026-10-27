# 04 — Elementwise / utility op kernels (`breccia-kernels::ops`)

Work-order item #3 of the M2 kernel block: everything the forward-pass graph
needs **around** the GEMVs — RMSNorm, IMRoPE, SwiGLU, softmax, the router top-k,
the quantised embedding gather, the hyper-connection mixer and the residual
add/scale — each with a CPU reference, a GPU↔CPU conformance test, and a
`breccia bench ops` cell at T = 1 (decode) and T = 256 (prefill chunk).

Date: 2026-10-26 · both GPUs idle at the start of every run
(`gpu_busy_percent` 0 / 7, `pgrep llama-server` empty) · no service was started
or stopped (`qwen-dual`, `strata-rx`, `d1-mi50`: inactive before and after) ·
artifacts in `<SCRATCH>/opencode/p3-ops/`.

**Status: complete.** `cargo build/test --workspace`, `cargo clippy -p
breccia-kernels -p breccia-cli --all-targets -- -D warnings`, `cargo fmt
--check`: green. Every op is inside **2.3e-7 relative** of its f64 CPU
reference (bar 1e-5), the router's **id sets match exactly**, and the embedding
gather is **bit-identical** (0 error) on both paths. Every GPU run was one op per
process, `timeout 90`, validation layer on, MI50 first: **16 conformance runs
(8 ops × 2 GPUs, `10 passed` each) and 17 bench runs, 0 validation
messages, no fault**.

Headline numbers, RX 6950 XT (T = 1 / T = 256):

| op | T=1 | T=256 |
|---|---|---|
| RMSNorm 2560 (grouped, HC) | **6.3 µs** | 20.4 µs · 1029 GB/s |
| IMRoPE Q (24 heads × 256) | **2.0 µs** | 10.7 µs · 1330 GB/s |
| SwiGLU 10×640 (routed set) | **1.8 µs** | 12.2 µs · 1615 GB/s |
| softmax 512 | **2.8 µs** | 3.5 µs · 303 GB/s |
| router (softmax 512 → top-10 → clamp) | **9.2 µs** | 10.5 µs |
| emb gather `token_embd` IQ3_S | **4.6 µs** | 8.0 µs · 365 GB/s |
| emb gather PLE IQ4_NL (16 rows/token) | **2.1 µs** | 6.2 µs · 482 GB/s |
| **HC_READ chain** (norm+down+act+up+gated+inject) | **60.5 µs** | **1.87 ms** |
| residual add / scale | 1.6 µs | 5.9 µs · 1326 GB/s |

**The number that changes the forward-pass plan: HC_READ costs 60 µs per call on
the RX, and 22 µs of it is the 4-row `inject` GEMV.** At 96 HC_READs per token
that is 5.8 ms/token of the 48.1 ms parity budget (`docs/phase-3/05` §1), 2.1 ms
of which is one latency-bound 4×10240 projection. §5 has the breakdown and §7
the fix.

---

## 1. What landed

| Path | Content |
|---|---|
| `src/ops/mod.rs` | The ten op pipelines (`RmsNorm`, `Rope` + `RopeTable`, `SwiGlu`, `Softmax`, `Router`, `EmbGather`, `Act`, `HcGatedMean`, `HcWrite`, `Add`) and the `HcMixer` chain + `HcWorkspace`. One `Pipe` wrapper per pipeline (compile, bind, dispatch, timestamped repeat), one `Pc` push-constant builder, `grid_for()` wrapping. |
| `src/ops/cpu.rs` | The f64 CPU references: `rmsnorm`, `rope_table`, `rope_apply`, `swiglu`, `softmax`, `router`, `hc_read_token`, `hc_write`, `add`, `scale`, `sigmoid_vec`, plus 9 self-tests (tie-break, shift invariance, NEOX-vs-adjacent pairing, mean-vs-sum…). |
| `src/ops/bench.rs` | `breccia bench ops` runner + JSON schema (`schema_version` 1), per-cell latency/GB/s/GFLOP/s, per-stage HC breakdown. |
| `shaders/ops_rmsnorm.comp` | row-wise RMSNorm, optional grouped gamma, configurable eps |
| `shaders/ops_rope.comp` | partial **NEOX**-paired rotary from a host cos/sin table + per-row int32 positions |
| `shaders/ops_swiglu.comp` | `silu(gate) * up` |
| `shaders/ops_softmax.comp` | stable row-wise softmax, any width |
| `shaders/ops_router.comp` | softmax(all) → top-k → `max(sum, 2^-14)` renormalise, one 512-thread workgroup |
| `shaders/ops_gather.glsl` + `ops_gather_<type>.comp` ×14 | quantised row gather + dequant (index-list indirection, one subgroup per row) |
| `shaders/ops_hc_gated_mean.comp` | `mean_c(xn[c] * sigmoid(gate[c]))` |
| `shaders/ops_hc_write.comp` | `R + block * 2*sigmoid(inject/hc)` per stream |
| `shaders/ops_act.comp` | SiLU / SiLU(x/hc) / sigmoid / scale / silu·scale |
| `shaders/ops_add.comp` | `a + beta*b` |
| `shaders/ops_math.glsl` | `breccia_silu` / `breccia_sigmoid`, the single definition both the SwiGLU and the HC gates use |
| `crates/breccia-kernels/build.rs` | +24 SPIR-V modules (10 ops + 14 gather wrappers), each `spirv-val`-checked |
| `src/batch.rs` | `GemvBatch::dispatch` → `pub(crate)` so the HC chain can put a projection inside its own command buffer |
| `src/lib.rs` | `pub mod ops` + re-exports |
| `tests/ops_conformance.rs` | 10 tests, `BRECCIA_OP` one-op-per-invocation selector |
| `crates/breccia-cli` | exactly one new subcommand: `breccia bench ops --op …` |

Nothing outside `crates/breccia-kernels/**`, `crates/breccia-cli/**` and this
report was touched; `profile`, `serve`, `bench gemv`, `bench transfer` are
unchanged (the workspace test suite proves it).

### 1.1 Geometry and conventions

* **Everything is f32.** The ops are activation-side only; the weights they
  consume (HC projections) go through `GemvBatch`, which decodes BF16/F32 and
  every quant type already.
* **Flat buffers, GGML `ne` order.** A `[d, h, t]` tensor is "rows of `cols`"
  with `row = h + n_head*t`; nothing is transposed or copied on the way in.
* **Push constants** fit the 32-byte range `breccia-vk` reserved in
  `docs/phase-3/03` §1.1 — four words per op, no `breccia-vk` change.
* **Dispatches wrap over `grid.y`** so `grid.x ≤ 65535` (the only value every
  Vulkan device guarantees), the same guard the decode kernels use.
* **`TAIL_PAD`** (64 B, `docs/phase-3/02` §4.2) applies to every allocation via
  `gemv::alloc`, so a vectorised tail read lands in mapped memory.

---

## 2. Per-op semantics (spec §ref) and what the kernel does

Full citations are in each shader's header; this is the checklist.

### 2.1 RMSNorm — `ops_rmsnorm.comp`

`y[r,i] = x[r,i] * rsqrt(mean_i(x[r]²) + eps) * gamma[(r % wgroups)*cols + i]`
(`docs/specs/qwen4exp-architecture.md` §13 ops 3/4/5, §14 "eps is 1e-6
everywhere"). One workgroup (256 threads) per row; the sum of squares is a
shared-memory tree.

`wgroups` is the only shape knob, and it is what makes **one** kernel serve all
five norm forms:

| use | `cols` | `rows` | `wgroups` |
|---|---|---:|---:|
| HC stack / final mixer gamma (`hc_*_norm` as `[2560,4]`) | 2560 | `hc*T` | **4** |
| per-head Q/K norm, indexer q/k norm | 256 | `24*T` / `4*T` | 1 |
| GDN post-recurrence `ssm_norm` | 128 | `48*T` | 1 |
| PLE `*_norm_*` (§9.2) | 2560 / 10240 | `T` / `hc*T` | 1 / 4 |
| **GDN q/k L2 norm** (`x/sqrt(Σx²+eps)`, `models.h:14-18`) | 128 | `16*T` | **0** + `eps/n` |

`wgroups = 0` means "no gamma", which is exactly the L2-normalisation case —
that is why the op exposes `with_eps` (pass `eps/cols` for the L2 form). The
gamma is used **as stored**: §14 says the stored value is the multiplier, and
whether the converter folds `(1+w)` into `hc_*_norm` (spec open question Q1) is
a question about which values get uploaded, not about this kernel.

### 2.2 IMRoPE — `ops_rope.comp` (+ `RopeTable`)

Host table, kernel rotation:

```
inv[j]  = theta^(-2j/n_rot)                    j in [0, n_rot/2)   f64, host
ang     = pos * inv[j]                                              f64, host
cos,sin = table[pos][j]                                             f32, uploaded
out[j]          = x[j]     * cos - x[j+n_rot/2] * sin
out[j + n_rot/2]= x[j]     * sin + x[j+n_rot/2] * cos
dims [n_rot, head_dim) copy through
```

**The pairing is NEOX — `(j, j + n_rot/2)`, not the adjacent `(2j, 2j+1)`.**
The work order calls this op "interleaved rotary"; the artifact's ruler is the
model's own code and both references agree on NEOX:

* llama.cpp `ggml_compute_forward_rope_flt` maps `NEOX`/`MROPE`/`IMROPE` to
  `rotate_pairs(n_dims, n_offset = n_dims/2, …)` (`ggml/src/ggml-cpu/ops.cpp`),
  and this artifact's `rope_type` is `LLAMA_ROPE_TYPE_IMROPE`
  (`src/llama-model.cpp`, `src/models/qwen4exp.cpp:1062,1068`).
* strata states it explicitly: *"Pairing is NEOX: (i, i + n_rot/2), NOT the
  adjacent pair (2i, 2i+1)"* (`ST/include/strata/kernels/rope.hpp:1-16`) with
  `rope_neox_pair` as the one shared definition, and its `build_rope_table`
  (`src/kernels/cuda/rope.cu:49-59`) is the loop transcribed into
  `ops::cpu::rope_table` **bit-for-bit** (asserted by a unit test).

"Interleaved" in the spec's sense refers to the *M-RoPE section* interleaving
(`sections [11,11,10,0]`, `sector % 3`), which is the identity for text because
all three position streams start from the same base and advance by the same
`theta_scale` (`ops.cpp:6224-6276`) — so `pos_t = pos_h = pos_w` and the op is
plain 1-D partial RoPE over the first 64 of 256 dims. Equivalently: one int32
position per row is the whole text-mode contract, which is what the kernel
takes. **In-place is safe** (the two halves are disjoint) and is tested.

Both Q and K are rotated on every QSA layer (§4.3 steps 4, §10.3); the GDN
layers rotate nothing.

### 2.3 SwiGLU — `ops_swiglu.comp`

`out = silu(gate) * up`, elementwise, one dispatch over the layer's whole routed
set (10 × 640) or the shared expert (640) — `docs/phase-3/01` §7 item 5 and
`02` §7 item 7 both asked for this as the companion of the batched GEMV, and it
removes one round trip of the 640-wide intermediate plus one dispatch per layer.
`silu` is ggml's `GGML_SILU_FP32 = x/(1+exp(-x))`, stabilised so `x = -inf`
gives 0 instead of NaN. The *shared expert scalar gate* is a **sigmoid**, not a
SiLU (§14) — it is `ops_act`'s `Sigmoid`/`Scale` mode applied to the 1-row GEMV
output, deliberately not fused into this kernel.

### 2.4 Softmax — `ops_softmax.comp`

Stable row-wise softmax, any `cols`. Two callers: the router (512) and, once the
attention kernel lands, the attention probabilities over the selection width
`min(n_kv, 2051)` (§10.4) — so 512…2048 with a ragged tail is the required
range, and 2049 is in the sweep. One workgroup per row, three passes over the
row (max, sum, write) with `exp` recomputed for the write: no shared-memory row
staging means no width limit, and the kernel is memory-bound at 2048 (950 GB/s
RX).

### 2.5 Router top-k — `ops_router.comp`

```
p    = softmax(logits) over ALL 512          (never over the selected subset)
ids  = stable descending argsort(-p)[:10]    ties -> lowest id
w    = p[ids] / max(sum(p[ids]), 2^-14)
```

Transcribed from `docs/specs/qwen4exp-architecture.md` §5.4/§8
(`llama-graph.cpp:2171,2237,2251,2265-2273`) and strata's independent kernel
(`ST/src/kernels/cuda/router_top10.cu:3-17,108`). The gather-then-clamp *form*
is reproduced rather than the equivalent subset-softmax, for the reason strata's
header gives: the clamp is part of the renormalisation. Selection is 10 passes
of a block-wide argmax that keeps the lowest index on a tie — deterministic,
no atomics, one 512-thread workgroup per token, **one dispatch**.

The one documented deviation: strata accumulates the exponentials, the row sum
and the renormalisation in **f64**; `breccia-vk` does not enable
`shaderFloat64`, so this kernel is f32 throughout. Consequences, measured: the
**ids are identical** on every row tested (including a 6-way exact tie) and the
weights agree to **4.8e-9 … 1.6e-7** relative. Exact ties still tie exactly,
because `exp` is deterministic on identical inputs.

### 2.6 Embedding gather + dequant — `ops_gather_<type>.comp`

Two tables, one kernel:

| table | shape | type | bytes | path |
|---|---|---|---:|---|
| `token_embd.weight` | `[2560, 248320]` | IQ3_S | 260 MiB | staged or resident |
| `per_layer_token_embd.weight` | `[160, 320001536]` | IQ4_NL | 26.8 GiB | staged, on demand |

Row selection is an **index list** (binding 4, `u32` per output row), so:

* **staged** (what the model does): `EmbGather::pack_rows(table, row_bytes, ids)`
  copies *only the requested rows* out of a `breccia_format` mapping
  (`TensorView::as_slice()`) into one block of `rows × row_bytes`, which is
  uploaded once; the index list is the identity. For the PLE table that is
  90 B/row × 16 rows/token — the table is never resident, and nothing but the
  gathered rows is ever copied.
* **resident**: bind the whole 260 MiB `token_embd` once and pass the token ids
  as the index list. Same kernel, no staging copy.

The PLE head-slowest flatten of §9.1 (`qwen4exp.cpp:1361-1364`) falls out for
free: gathering `16*T` rows token-major (`row = t*16 + head`) produces
`[16*160, T] = [2560, T]` as written.

Decode is `blocks.glsl`'s `DEQUANT_UNIT`, i.e. the *same* code the fused GEMV
and the standalone dequantiser use, so this path inherits their bit-exactness
(`tools/refcheck.py` holds them to `ggml-quants.c`).

**One design change came out of measurement.** The first version gave one
*thread* a whole block; on the MI50 at 256 rows that measured **25 GB/s** (each
thread striding through its own 110 B block — no coalescing) and 29.7 µs for the
single-token cell. Switching to the decode kernels' lane mapping — one
**subgroup** per row, lane `l` owning unit `l` of the block it is on (512 B
contiguous per subgroup), with `32/UNITS` blocks per iteration for the narrow
block types — took the same MI50 cells to **162 GB/s** (6.4×) and 12.7 µs, and
the RX cells to 365 GB/s / 4.6 µs (§5). The conformance numbers did not move
(still bit-identical), which is the point: the change was layout, not
arithmetic.

### 2.7 Hyper-connection mixer — `HcMixer` (+ `ops_hc_*.comp`)

`docs/specs/qwen4exp-architecture.md` §4.1, transcribed line by line
(`LC/src/models/qwen4exp.cpp:313-397`, `ST/include/strata/kernels/gr.hpp:77-124`):

```
HC_READ                                         R [hc, n_embd] per token
  xn     = rmsnorm(R, eps) * w_norm              RMS over ONE stream; gamma [n_embd, hc]
  lo     = silu( (xn @ w_down.T) / hc )          w_down [hc_dim, hc_lr] = [10240, 320]
  gate   = lo @ w_up.T                           w_up   [hc_lr, hc_dim] = [320, 10240]
  gated  = xn * sigmoid(gate)                    elementwise, per stream
  mixed  = mean_over_streams(gated)              -> [n_embd]
  inject = xn @ w_inject.T                       w_inject [hc_dim, hc] = [10240, 4]
HC_WRITE
  w[c]     = 2 * sigmoid(inject[c] / hc)         centres the gate on 1
  R_out[c] = R[c] + block_out * w[c]             the SAME block_out to every stream
```

Six dispatches per HC_READ, two per HC_WRITE:

| stage | kernel |
|---|---|
| `xn` | `RmsNorm` with `cols = n_embd`, `wgroups = hc` (grouped gamma) |
| `lo_raw` | `GemvBatch(rows = 320, cols = 10240)` |
| `lo` | `Act::SiluDiv` (`s = hc`) |
| `gate` | `GemvBatch(rows = 10240, cols = 320)` |
| `mixed` | `HcGatedMean` |
| `inject` | `GemvBatch(rows = 4, cols = 10240)` |
| combine | `HcWrite` |

**The three projections are one `GemvBatch` dispatch each with the batch of
tokens as the batch of "experts"** (every "expert" names weight base 0 and its
own activation base, which is exactly `XLayout::PerExpert` with a degenerate
blob map). So `T = 1` is a decode GEMV and `T > 1` is the same dispatch shape
with `T` row groups — *a batched GEMV, not a tiled GEMM*: the weights are
re-read once per token. That is measured (§5) and flagged (§7).

`HcMixer::new(gpu, ty, 2560, 4, 320, with_inject)` is the artifact's trunk
mixer; `with_inject = false` is the **final** mixer (spec: "inject = null for the
FINAL mixer"), whose `mixed` output *is* the output norm — there is no separate
`output_norm` in this model (§4.0).

### 2.8 Residual add / scale / activation — `ops_add.comp`, `ops_act.comp`

`out = a + beta*b` (one dispatch; `beta = 1` is the plain residual add, and
`out` may alias either input) and the five activation modes
`Silu | SiluDiv | Sigmoid | Scale | SiluScale`. `SiluDiv` exists because §4.1
writes `silu(x/hc)`; `Scale` covers the shared-expert sigmoid gate, the
attention output scaling, `1/sqrt(S)` for the GDN q, and `1/kpool` for the
indexer pooling.

---

## 3. Conformance evidence

Method: `tests/ops_conformance.rs`, GPU vs the f64 reference in `ops::cpu`, one
op per process, `timeout 90`, `VK_LAYER_KHRONOS_validation`, MI50 first.

```bash
cargo build --release -p breccia-kernels --tests
BIN=./target/release/deps/ops_conformance-<hash>
for CARD in 0000:0e:00.0 0000:03:00.0; do          # MI50 first
  for OP in rmsnorm rope swiglu softmax router emb_gather hc add; do
    timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
        BRECCIA_KERNEL_DEVICE=$CARD BRECCIA_OP=$OP \
        $BIN --test-threads=1 --nocapture
  done
done
```

Result: **`10 passed` per invocation on both GPUs** (the selected op's tests
run and the rest self-skip through `BRECCIA_OP`), `cargo test --test
ops_conformance` without `BRECCIA_KERNEL_DEVICE` skips everything (opt-in, as
required), and the validation-message count is **0** in all 16 logs
(`grep -ciE 'VUID|Validation Error|Validation Warning'`). Both GPUs produce the
same numbers to the last digit shown:

| op | case | bar | RX max rel | MI50 max rel |
|---|---|---|---:|---:|
| RMSNorm | hidden 2560, ±gamma, T=1..4 | 1e-5 | 1.7e-7 | 1.7e-7 |
| RMSNorm | per-head 256 (24 & 120 rows) | 1e-5 | 2.2e-7 | 2.2e-7 |
| RMSNorm | ssm 128 | 1e-5 | 2.0e-7 | 2.0e-7 |
| RMSNorm | HC grouped 2560×4, T=1..3 | 1e-5 | 2.0e-7 | 2.0e-7 |
| RMSNorm | odd tail 257 | 1e-5 | 1.5e-7 | 1.5e-7 |
| IMRoPE | Q 24 heads, K 2 heads, T=1..3 | 1e-5 | 1.3e-7 | 1.3e-7 |
| IMRoPE | indexer q (128) / pooled key / full-width | 1e-5 | 1.2e-7 | 1.2e-7 |
| IMRoPE | **in-place vs out-of-place** | 1e-6 | **0** | **0** |
| SwiGLU | routed 10×640, shared 640, T=4, tail 641 | 1e-5 | 2.3e-7 | 2.3e-7 |
| softmax | 512 (T=1, 8) / 512,1024,2048 / tails 2049, 1000 | 1e-5 | 1.8e-7 | 1.8e-7 |
| **router** | random T=1, T=16, narrow logits | **ids exact** | **exact** | **exact** |
| **router** | weights | 1e-5 | **1.6e-7** | **1.6e-7** |
| **router** | 6-way exact tie | ids exact | **[5,12,37,88,300,411]** | same |
| **gather** | `token_embd` IQ3_S (1100 B/row) | 1e-6 | **0** | **0** |
| **gather** | PLE IQ4_NL (90 B/row) | 1e-6 | **0** | **0** |
| **gather** | IQ4_XS tail (136 B/row) | 1e-6 | **0** | **0** |
| gather | resident vs staged path | 1e-6 | 0 | 0 |
| **HC** | `mixed`, inject, final (T=1,2) | 1e-5 | 2.1e-7 | 2.1e-7 |
| **HC** | `inject` | 1e-5 | 1.1e-7 | 0 |
| **HC** | HC_WRITE combine | 1e-5 | 0 | 0 |
| add | 2560, T=5, tail 257, β = 1 / −0.75 | 1e-6 | 0 … 2.1e-7 | same |
| act | 5 modes × {320, 10240} | 1e-5 | 0 … 1.8e-7 | same |

Notes on the two bars that are not 1e-5:

* **Router ids are exact**, which is the contract (§5.4 decides routing, so an
  id flip is a different model). The weights cannot be exact because the
  production kernel sums in f64; 1.6e-7 relative over 10 selected probabilities
  is 3 orders inside the 1e-3 that a logit needs.
* **The gather is bit-identical** because integer codes and fp16 scales are
  exact and there is no accumulation. It is the strongest statement available
  and it covers the same `DEQUANT_UNIT` the fused GEMV uses.

Also covered: `degenerate_shapes_do_not_fault` (n=1 for every elementwise op, a
1-row gather of both embedding types, a 1-token router with all-equal logits —
every strided loop runs once and every bound guard is exercised, under
validation), and `cpu_reference_self_checks` so the non-GPU path is exercised by
plain `cargo test`.

---

## 4. Buffer layouts and dispatch notes for the forward-pass integrator

Per `run` call; sizes are per token unless marked. `T` = tokens in the call.

| op | buffers (binding order) | dispatch |
|---|---|---|
| `RmsNorm` | `x[rows*cols]`, `gamma[wgroups*cols]` (or the internal dummy), `out[rows*cols]` — `out` may alias `x` | `rows` workgroups × 256 threads |
| `Rope` | `x[rows*head_dim]`, `table[max_pos*n_rot/2*2]`, `pos[rows]` i32, `out[…]` — `out` may alias `x` | `rows*head_dim/2` invocations, 64/wg |
| `SwiGlu` | `gate[n]`, `up[n]`, `out[n]` | `n` invocations, 256/wg |
| `Softmax` | `x[rows*cols]`, `out[…]` | `rows` wg × 256 |
| `Router` | `logits[rows*512]`, `ids[rows*10]` u32, `weights[rows*10]` f32 | `rows` wg × 512 |
| `EmbGather` | `packed[batch*row_bytes]` (or the whole table), `T` tables, `out[batch*padded_cols]`, `ix[batch]` u32 | one subgroup per row, 8 rows/wg |
| `Act` | `x[n]`, `out[n]` | `n` invocations, 256/wg |
| `HcGatedMean` | `xn[n_embd*hc*T]`, `gate[…]`, `out[n_embd*T]` | `n_embd*T` invocations |
| `HcWrite` | `r[n_embd*hc*T]`, `block[n_embd*T]`, `inject[hc*T]`, `out[…]` — `r` may be `out` | `ceil(n_embd/256) × T` wg |
| `Add` | `a[n]`, `b[n]`, `out[n]` | `n` invocations |

`HcMixer` extras the graph needs:

* `HcMixer::workspace(gpu, T)` allocates `xn`, `lo_raw`, `lo`, `gate`, `mixed`,
  `inject` and the three row maps — **175 KiB at T=1, 43.8 MiB at T=256** — and
  `release()` frees them. The maps are constant for a `(weights, T)` pair, so
  they are built once, not per token. At T=256 the maps are half of that
  (21.6 MiB) because the batched GEMV's indirection is 8 B *per row* and the
  `up` projection has `10240 × T` rows; a per-token base instead of a per-row
  base would make it 24 B in total (§7.2 item 3).
* `read(&ws, r, &HcWeights{norm, down, up, inject})` fills `ws.mixed` (and
  `ws.inject`); `write(&ws, r, block, out)` runs HC_WRITE from `ws.inject`.
  `HcWeights.inject = None` is the final mixer.
* `stage_timings()` / `STAGES` give the per-stage split used in §5.
* Everything the graph must keep across layers is either a `Buf` the caller owns
  or inside `HcWorkspace` — no hidden allocation per token except the embedding
  staging copy (`pack_rows`), which is `batch × row_bytes` (90 B × 16 = 1.4 KiB
  per token for PLE).

Push constants are ≤ 16 B per op, so the reserved 32-byte range has headroom
for the per-row weight base that `docs/phase-3/03` §6 item 4 wants.

---

## 5. Measured — latency and effective GB/s

Commands (each its own process, `timeout`, validation on, MI50 first; the
`--json` files are the artifacts):

```bash
for OP in rmsnorm rope swiglu softmax router emb_gather hc add; do
  timeout 150 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
    ./target/release/breccia bench ops --op $OP --tokens 1,256 \
    --device 0000:0e:00.0 --json p3-ops/bench-mi50-$OP.json
  timeout 150 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
    ./target/release/breccia bench ops --op $OP --tokens 1,256 \
    --device 0000:03:00.0 --json p3-ops/bench-rx-$OP.json
done
```

`µs` is one dispatch, **best of 3 windows after two discarded warm-ups**
(clock ramp: the first window of a cell reads 20-25 % slow,
`docs/PROJECT-SETUP.md`), GPU timestamps, full barrier between dispatches.
`GB/s` = (bytes read + bytes written, **including the intermediate activation
buffers**) / time.

### 5.1 RX 6950 XT (gfx1030) — the primary backend

| op | case | T=1 µs | T=1 GB/s | T=256 µs | T=256 GB/s |
|---|---|---:|---:|---:|---:|
| rmsnorm | hidden 2560 | 6.19 | 5.0 | 7.28 | 722 |
| rmsnorm | per-head 256 (24×T rows) | 2.23 | 22.5 | 12.87 | 978 |
| rmsnorm | ssm 128 (48×T rows) | 2.21 | 22.5 | 19.68 | 639 |
| rmsnorm | HC grouped 2560×4 | 6.28 | 19.6 | 20.42 | **1029** |
| rope | Q (24 heads × 256) | 2.02 | 27.4 | 10.66 | **1330** |
| rope | K (2 heads × 256) | 1.97 | 2.3 | 2.94 | 402 |
| swiglu | routed 10×640 | 1.84 | 41.7 | 12.17 | **1615** |
| swiglu | shared 640 | 1.77 | 4.3 | 2.76 | 711 |
| softmax | router 512 | 2.81 | 1.5 | 3.46 | 303 |
| softmax | attention 512 | 2.89 | 34.0 | 23.86 | 1055 |
| softmax | attention 2048 | 5.21 | 75.4 | 105.96 | 950 |
| softmax | attention 2049 (tail) | 5.51 | 71.4 | 111.22 | 906 |
| **router** | softmax512 → top-10 | **9.18** | – | 10.48 | 52 |
| emb_gather | `token_embd` IQ3_S | 4.62 | 2.5 | 7.96 | 365 |
| emb_gather | PLE IQ4_NL, 16 rows/token | 2.06 | 5.7 | 6.24 | 482 |
| **hc** | **HC_READ chain** | **60.45** | 221 | **1869.7** | 1830 |
| hc | elementwise bundle (norm+gated+combine) | 11.13 | 23.9 | 50.96 | 1338 |
| add | residual add | 1.55 | 19.8 | 5.93 | **1326** |
| add | residual scale | 1.55 | 13.2 | 4.85 | 1080 |

HC_READ per stage (RX), which is where the money is:

| stage | T=1 µs | T=256 µs |
|---|---:|---:|
| norm (grouped, 4×2560) | 6.27 | 20.66 |
| **down** (320 × 10240 BF16) | **22.75** | 820.82 |
| act (320) | 1.57 | 1.87 |
| up (10240 × 320 BF16) | 7.45 | 986.70 |
| gated mean (2560) | 2.34 | 14.22 |
| **inject** (4 × 10240 BF16) | **21.85** | 26.34 |

Two things to read out of that table:

1. **At T=1 the chain is latency, not bandwidth.** 60.45 µs moves 13.1 MB of
   BF16 weights → 221 GB/s, half the 596 GB/s a plain 640-row BF16 GEMV reaches
   at these sizes (`docs/phase-3/01` §4.1), i.e. the grid is under-filled and
   the dispatch floor is a real share. The **4-row `inject` projection is
   21.85 µs for an 82 KiB read** — one 8-row workgroup on a 60-CU card, i.e.
   almost pure dispatch + tail latency, 36 % of the whole HC_READ. `down`
   (40 workgroups) costs 22.75 µs for the same 6.29 MiB that `up`
   (1280 workgroups) does in 7.45 µs: **the 320-row shape is 3× worse per byte
   than the 10240-row one**, the same under-filled-grid effect
   `docs/phase-3/01` §4.1 measured for expert matrices.
2. **At T=256 the batched GEMV is the whole cost** and it re-reads the weights
   per token: 3.26 GB touched in 1.87 ms. The 1830 GB/s exceeds the card's
   561 GB/s measured read ceiling, i.e. **the 13 MB of HC weights sit in the
   128 MiB Infinity Cache across the 256 re-reads**; on the MI50 (4 MB L2) the
   same cell measures 548 GB/s, which is genuine DRAM traffic. Either way a
   tiled GEMM with weight reuse is the prefill fix (§7).

The T=1 column of *every* op is dispatch-bound: the measured floor for a
single dispatch + barrier on this card is **1.55 µs** (`add`) and 2.0 µs
(`rope`), so a T=1 cell is `max(work, ~2 µs)` — these are per-call latencies for
the token budget, not bandwidth numbers.

### 5.2 Radeon VII / MI50 (gfx906) — the non-display card

Same binary, same shaders, Wave64:

| op | case | T=1 µs | T=1 GB/s | T=256 µs | T=256 GB/s |
|---|---|---:|---:|---:|---:|
| rmsnorm | hidden 2560 | 13.69 | 2.2 | 17.40 | 302 |
| rmsnorm | per-head 256 | 5.87 | 8.5 | 34.15 | 369 |
| rmsnorm | ssm 128 | 5.89 | 8.4 | 52.62 | 239 |
| rmsnorm | HC grouped | 13.41 | 9.2 | 45.92 | 458 |
| rope | Q (24 heads) | 5.43 | 10.2 | 24.70 | 574 |
| rope | K (2 heads) | 5.25 | 0.9 | 6.85 | 173 |
| swiglu | routed 10×640 | 4.89 | 15.7 | 29.44 | 668 |
| swiglu | shared 640 | 4.89 | 1.6 | 6.98 | 282 |
| softmax | router 512 | 7.26 | 0.6 | 8.66 | 121 |
| softmax | attention 2048 | 12.48 | 31.5 | 272.23 | 370 |
| router | softmax512 → top-10 | 20.85 | – | 24.78 | 22 |
| emb_gather | `token_embd` IQ3_S | 12.66 | 0.9 | 17.92 | 162 |
| emb_gather | PLE IQ4_NL | 5.44 | 2.2 | 11.29 | 266 |
| **hc** | **HC_READ chain** | **154.49** | 86 | **6241.4** | **548** |
| hc | elementwise bundle | 25.43 | 10.5 | 115.18 | 592 |
| add | residual add | 4.55 | 6.8 | 14.19 | 554 |

The MI50 is ~2.4× slower per dispatch (its floor is 4.5 µs) and ~2.5× slower at
T=256, consistent with `docs/phase-3/01` §4.3: this card is instruction-rate
bound on quantised decode and its DRAM read ceiling (825 GB/s) is reached only
by table-free kernels. The op kernels are table-free f32 work, so the T=256
column is a fair 48-66 % of the reference bandwidth, and the HC chain's 548 GB/s
is 66 % of 825 GB/s — the best this card does anywhere in this report.

### 5.3 What this adds up to per token (RX, T=1)

Units per layer (`docs/specs/qwen4exp-architecture.md` §4.0): 2 × HC_READ, 2 ×
HC_WRITE, 1 router, 1 SwiGLU (routed + shared), plus per-type extras.

| item | per layer | × 48 layers |
|---|---:|---:|
| HC_READ chain (2 × 60.45 µs) | 121 µs | **5.80 ms** |
| — of which the `inject` GEMV (2 × 21.85) | 44 µs | **2.10 ms** |
| HC_WRITE combine (2 × 2.34 µs) | 4.7 µs | 0.23 ms |
| router | 9.18 µs | 0.44 ms |
| SwiGLU (routed + shared) | 3.6 µs | 0.17 ms |
| residual add (2 × 1.55) | 3.1 µs | 0.15 ms |
| IMRoPE (Q+K, QSA layers only) | ~4.0 µs | 0.05 ms (12 layers) |
| embedding gather (whole model, per token) | – | 0.005 ms |
| **total, op kernels** | | **≈ 6.8 ms/token** |

That is **14 % of the 48.1 ms parity baseline** (`docs/phase-3/05` §1), and 96 %
of it is the hyper-connection mixer. On the MI50 the same arithmetic is
~14.8 ms/token, i.e. the mixer would be a third of the budget — another reason
the HC projections belong on the RX.

---

## 6. Verification commands and results

```bash
cargo build --workspace                                             # Finished
cargo test --workspace                                              # all suites green
cargo clippy -p breccia-kernels -p breccia-cli --all-targets -- -D warnings   # clean
cargo fmt --check                                                   # clean

# conformance: one op per process, MI50 first, then the display card
cargo build --release -p breccia-kernels --tests
BIN=./target/release/deps/ops_conformance-<hash>
for CARD in 0000:0e:00.0 0000:03:00.0; do
  for OP in rmsnorm rope swiglu softmax router emb_gather hc add; do
    timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
      BRECCIA_KERNEL_DEVICE=$CARD BRECCIA_OP=$OP \
      $BIN --test-threads=1 --nocapture
  done
done                    # 10 passed per op per GPU, 0 validation messages

# benchmarks: 8 ops x 2 GPUs x 2 batch sizes, one op per process
for OP in rmsnorm rope swiglu softmax router emb_gather hc add; do
  timeout 150 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
    ./target/release/breccia bench ops --op $OP --tokens 1,256 \
    --device 0000:03:00.0 --json p3-ops/bench-rx-$OP.json
done                    # 0 validation messages, 38 cells

# the whole set in one process (smoke test only — the per-op runs are the evidence)
timeout 150 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ./target/release/breccia bench ops --quick --device 0000:03:00.0
```

Artifacts (`<SCRATCH>/opencode/p3-ops/`):
`final-{mi50,rx}-<op>.log` (16 conformance logs), `bench-{mi50,rx}-<op>.json`
(16 JSONs), `benchlog-{mi50,rx}-<op>.log` (16 bench logs + `benchlog-rx-all.log`).

Safety: every GPU invocation above ran under `timeout`, one op per process, on
one GPU at a time, with the MI50 (`0000:0e:00.0`, no display) first and the RX
(`0000:03:00.0`) only after the MI50 passed — `AGENTS.md` GPU crash safety.
`journalctl -k -b | grep -cE "amdgpu.*(page fault|Illegal opcode)"` = **0** after
the whole window; no service was started or stopped.

---

## 7. Open questions and what remains

### 7.1 Open questions (pin these before trusting end-to-end numbers)

1. **IMRoPE pairing** (§2.2). Implemented as **NEOX** `(j, j+n_rot/2)` on the
   evidence of both references, not as the adjacent-pair "interleaved" reading
   the work order's wording allows. This is the highest-consequence choice in
   this block: the wrong pairing produces correctly shaped, scrambled content.
   `cpu::tests::rope_is_the_neox_pairing_not_the_adjacent_one` pins it, and
   `RopeTable` matches strata's `build_rope_table` bit-for-bit. Verifying it
   against a real activation dump is still the definitive test.
2. **HC gamma folding (spec Q1).** The kernel multiplies by the stored gamma
   (§14: "treat the stored value as the multiplier"). Whether `hc_*_norm` and
   `output_hc_norm` really arrive as `(1+w)` is unresolved upstream — a wrong
   assumption is a plausible-looking per-layer scale error.
3. **HC `lo` division placement.** `silu(x/hc)` (implemented) vs `silu(x)/hc`.
   §4.1 writes the former; the difference is a factor of 4 inside the
   non-linearity, not a constant scale.
4. **`mixed` is a mean, not a sum** (`/hc`). `cpu::tests::hc_read_mean_is_a_mean_not_a_sum`
   pins it; a sum is a factor-of-4 error that would look like a badly calibrated
   mixer.
5. **`inject` mixes the whole `[hc_dim]` stack.** `hc_*_inject.weight` is
   `[10240, 4]` in `ne` order, so `inject[c]` is a dot product over all four
   streams. That is the only reading consistent with the shape, but it is worth
   confirming that the converter did not mean four per-stream reductions.
6. **Router accumulation precision** (§2.5): f32 here vs f64 in strata. Ids are
   exact on everything tested; weights differ by ≤ 1.6e-7 relative. If a real
   logit row ever times out on an id flip, the fix is an fp64 path (needs
   `shaderFloat64` enabled in `breccia-vk`) or Kahan summation of the 512 exps.
7. **RoPE table size.** The table is `[max_pos][32]` of `(cos,sin)` = **64 MiB
   at the 262144-token context**. Either size it to the live window and rebuild
   on demand (it is cheap: 262144 × 32 in f64 is ~50 ms once) or move the angle
   computation into the shader (which loses the exact f64 angles that make it
   match the reference — `ST` builds it on the host for exactly that reason).
   Positions beyond `max_pos` are clamped rather than read out of bounds.
8. **Vision / M-RoPE sections.** Text only: one position per row. The
   multimodal path needs strata's per-pair sector table
   (`ST/include/strata/kernels/mrope.hpp`: `tab[cell*3 + pair % 3]`), a
   `[cells][3]` int32 device array plus a `pair % 3` selector in the shader —
   ~10 lines, no new op, but not implemented because the mmproj is not on the
   trunk path.
9. **The PLE row index list is host-built.** `EmbGather` gathers what it is
   told; the n-gram hash (`ngram.hpp:9-27`, XOR/mul/mod over the last 3 token
   ids, 16 rows per token) is a host-side op in §9.1 and is not in this block.
   It is 16 `u64`-mul + `%` per token — negligible, but it is a per-token host
   step the graph must re-run (a captured graph that bakes it in replays the
   capture-time token, `ST/src/core/layer.cpp:1340-1348`).
10. **HC projections at T > 1 are a batched GEMV, not a GEMM** (§2.7): the
    weights are re-read once per token, so the T=256 chain is 1.87 ms (RX) /
    6.24 ms (MI50) per HC_READ. 96 HC_READs per 256-token batch = 179 ms (RX).
    This is correct and measured, not a placeholder — but a tiled GEMM with
    weight reuse is worth ~8× on the projection part and is the top prefill
    lever after the inject fix below.

### 7.2 Measured levers, in order of value

1. **Kill the 4-row `inject` GEMV (2.10 ms/token RX).** 21.85 µs to read 82 KiB
   because it is one workgroup. The natural fix is to fold the four dot products
   into `ops_hc_gated_mean.comp` (it already walks `xn` for every stream, so it
   can accumulate `inject[c] += xn[c][d] * w[c][d]` and write the four scalars
   with a workgroup reduction) — one dispatch instead of two, and the read
   becomes free because the bytes are already in registers. Expected saving:
   ~2 ms/token, 4 % of the parity budget, for one shader change.
2. **Fuse the six HC_READ dispatches into one kernel.** Six dispatches at a
   ~2 µs floor is 12 µs of the 60; after the inject fix the remaining
   bottleneck is the 320-row `down` GEMV (22.75 µs; a tiled or split version of
   `down` alone would be worth ~2× by `docs/phase-3/01` §4.2's split-K result).
3. **Tiled GEMM for the HC projections at T > 1** (§7.1 item 10), and with it a
   per-token row base instead of the current per-row map (21.6 MiB of the 43.8
   MiB workspace at T=256 is indirection the GEMM would not need).
4. **The router at 9.18 µs/token** (§5.1) is 0.44 ms/token and is two ops in
   one workload: the 512-wide softmax is memory-trivial and the 10 selection
   passes are 10 barriers in a single workgroup. It is fine for now; if 48
   layers of it ever matters, the selection can be one pass of the
   rank-by-counting form (which strata measured *slower* at n=512 — the current
   shape is already the cheap one).

### 7.3 What remains before the graph (and for attention/GDN)

Ops-inventory coverage (`docs/specs/qwen4exp-architecture.md` §13):

| # | primitive | status |
|---|---|---|
| 1 | quantised embedding gather | ✅ this block |
| 2 | quantised GEMV / batched expert GEMV | ✅ `docs/phase-3/01`, `02` |
| 3 | grouped RMSNorm over `[n_embd, hc]` | ✅ `wgroups = hc` |
| 4 | per-head RMSNorm | ✅ |
| 5 | plain RMSNorm | ✅ |
| 6 | L2 norm (GDN q/k) | ✅ `wgroups = 0`, `eps/n` |
| 7 | HC low-rank mixer | ✅ `HcMixer::read` |
| 8 | HC write (gated combine) | ✅ `HcMixer::write` |
| 9 | SiLU / sigmoid / exp | ✅ (`Act`, `ops_math`); **`softplus` missing** — needed by the GDN gate `softplus(alpha + ssm_dt) * ssm_a` (§4.2), ~6 lines in `ops_act` |
| 10 | depthwise causal conv1d (GDN, d=4) | ❌ next block |
| 11 | **GDN step** (decay, rank-1 delta rule, output) | ❌ next block |
| 12 | GDN chunked prefill | ❌ next block |
| 13 | IMRoPE | ✅ this block |
| 14 | attention with additive mask (dense + selected) | ❌ next block — `Softmax` (512…2049) is its row-wise half |
| 15 | top-k over selection width (numpy tie rules) | ❌ next block (`Router`'s argmax loop is the shape, but the width is 2049 not 512 and the tie rule is ascending-cell) |
| 16 | indexer score (per-head ReLU dot + tail bias) | ❌ |
| 17 | indexer key pool (mean of 4 → RMS → rope at block start) | ❌ (`RmsNorm` + `Rope` are its halves) |
| 18 | KV append / gather | ❌ |
| 19 | MoE router | ✅ this block |
| 20 | SwiGLU expert FFN | ✅ activation here; the 10-expert GEMVs are `GemvBatch` |
| 21 | shared expert + sigmoid scalar gate | ✅ `SwiGlu` + `Act::Sigmoid`/`Scale` around a 1-row `Gemv` |
| 22 | PLE host hash | ❌ host-side (§7.1 item 9) |
| 23 | PLE gather | ✅ this block |
| 24 | PLE gate (signed sqrt + sigmoid) + dilated conv (dilation 3, hist 9) | ❌ next block |
| 25 | lm_head GEMV + sampler | GEMV ✅ (`output.weight` Q5_K, `01` §5); **argmax/sampling ❌** |
| 26 | snapshot / rollback | ❌ M4 |

So this block closes items 3-9, 13, 19-21, 23, and the row-wise half of 14.
Still before a first token: **GDN step + conv (36 layers), QSA attention +
indexer (12 layers), PLE device block (layer 1), lm_head sampling** — the same
list `docs/phase-3/05` §6 carries, now with items 3-9/13/19-23 removed.

---

## 8. Attributions

* **RoPE semantics** — NEOX pairing, partial rotation and the host f64 table
  from strata `include/strata/kernels/rope.hpp` + `src/kernels/cuda/rope.cu:49-59`
  (MIT) and llama.cpp `ggml/src/ggml-cpu/ops.cpp` `rotate_pairs` /
  `ggml_mrope_cache_init` (MIT, commit 6050b46 — the qwen4exp fork).
  `ops::cpu::rope_table` is a transcription of strata's `build_rope_table`, and
  a unit test holds the two equal bit-for-bit.
* **Router semantics** — the contract and the tie rule from
  `docs/specs/qwen4exp-architecture.md` §5.4/§8 (llama-graph.cpp) and strata
  `src/kernels/cuda/router_top10.cu` (MIT): softmax over all experts, stable
  descending top-k with ascending-index ties, `max(sum, 2^-14)` renormalisation.
  No code was copied; the f32 accumulation is this crate's own deviation,
  documented in §2.5.
* **Decode units** — the gather reuses `shaders/blocks.glsl`, attributed in
  `docs/phase-3/01` §8 (llama.cpp `ggml-quants.c` + ggml-vulkan, MIT, commit
  6050b46). The gather's lane mapping follows ggml-vulkan's
  `mul_mat_vec_*` "one warp covers a super-block, per-lane 8-element unit" shape,
  already credited there.
* **SiLU / sigmoid** — `GGML_SILU_FP32` / `GGML_SIGMOID_FP32` from
  `ggml/src/ggml.c` (MIT), with the tail stabilisation noted in
  `ops_math.glsl`.
* **RMSNorm / softmax** are textbook; the row-wise workgroup-tree structure is
  the one `docs/phase-3/01` §1 established for the decode kernels, not taken
  from a reference engine.
* No code from IsoQuant or RotorQuant (unlicensed) was consulted or copied.
