# 06 — GDN/SSM step (36 layers) + QSA attention with the lightning indexer (12 layers)

Work-order items #4 and #5 of the M2 kernel block — the last two kernel blocks
before the forward-pass graph — plus the measured win item #1 from
`docs/phase-3/04` §7.2 (the HC `inject` fold).

Date: 2026-10-26 · both GPUs idle at the start of every run
(`gpu_busy_percent` 0 / 2, `pgrep llama-server` empty) · no service was started or
stopped (`qwen-dual`, `strata-rx`, `d1-mi50`: inactive before and after) ·
artifacts in `/var/tmp/users/korphaus/tmp/opencode/p3-attn/`.

**Status: complete.** `cargo build/test --workspace`, `cargo clippy
--workspace --all-targets -- -D warnings`, `cargo fmt --check`: green. GDN is
within **1.7e-7** of its f64 reference on both paths (`T = 1` and `T = 8`), its
recurrent state is **bit-stable**, QSA's selection id lists match **exactly**
(including at 2062 cells, i.e. past the 2051-cell budget) and its sublayer output
is within **8.6e-7**. Every GPU run was one sublayer per process, `timeout 90`,
validation on, **MI50 first**: 22 conformance invocations + 8 bench runs,
**0 validation messages, no fault**.

Headline (RX 6950 XT, per layer / per token):

| block | per layer, T=1 | ×layers ⇒ per token | per token, T=256 chunk | share of 48.1 ms |
|---|---:|---:|---:|---:|
| GDN/SSM (36) | 157.3 µs | **5.66 ms** | 88.3 µs/tok ⇒ 3.18 ms | 6.6 – 11.8 % |
| QSA @ ≤2051 cells (12) | 274.4 µs | **3.29 ms** | 52.3 µs/tok ⇒ 0.63 ms | 1.3 – 6.8 % |
| **HC inject fold** (96 reads) | −16.0 µs per HC_READ | **−1.54 ms** | −0.005 ms | −3.2 % |

So the two new blocks cost **≈ 6–9 ms of the 48.1 ms parity budget**, and the
`inject` fold gives 1.5 ms of it back. The rest of the budget is the MoE, which
these kernels do not touch.

---

## 1. What landed

| Path | Content |
|---|---|
| `src/gdn/mod.rs` | `Gdn` (5 pipelines), `GdnState` (recurrent + conv state), `GdnWorkspace`, `GdnWeights`, `GdnProjections`, `forward`, `stage_timings` |
| `src/gdn/cpu.rs` | f64 reference: conv, L2 norm, beta/gate, delta rule, norm+gate, the whole layer; 5 property tests |
| `src/gdn/bench.rs` | `breccia bench ops --op gdn` |
| `src/qsa/mod.rs` | `Qsa` (8 pipelines + softmax + 2 rope + 2 rmsnorm), `QsaState` (paged KV + indexer cache), `QsaWorkspace`, `QsaWeights`, `QsaProjections`, `forward`, `stage_timings` |
| `src/qsa/cpu.rs` | f64 reference: f16 model, indexer score/top-k, pooling, attention, whole layer; 5 unit tests |
| `src/qsa/bench.rs` | `breccia bench ops --op qsa` |
| `src/proj.rs` | `Proj` — one projection, decode `Gemv` or batched `GemvBatch` |
| `shaders/gdn_*.comp` ×5 | conv+SiLU+history, q/k L2 norm, beta/gate, delta rule, norm+gate |
| `shaders/qsa_*.comp` ×8 | q split+norm, kv append, indexer tail, indexer pool, indexer score, top-k, attention scores, attention apply+gate |
| `shaders/f16_storage.glsl` | the KV cache's f16 encoding, ours rather than the driver's (see §3) |
| `shaders/ops_hc_gated_mean.comp` | **+ the folded `inject` projection** (5 bindings) |
| `shaders/ops_hc_write.comp` | sums the `inject` partials |
| `shaders/ops_rope.comp` | + `pos_base`/`pos_div`, so a decode step never re-uploads positions |
| `src/ops/mod.rs` | `Rope::run_at`, `HcGatedMean::with_inject`, `HcWrite` `chunks`, `HcMixer` fold (+ `inject_values`), `pipe/gpu()` accessor |
| `tests/attn_conformance.rs` | 7 GPU tests, one sublayer per invocation via `BRECCIA_OP=gdn|qsa` |
| `crates/breccia-cli` | `breccia bench ops --op gdn|qsa` (help text only; no new subcommand) |

Nothing outside `crates/breccia-kernels/**`, `crates/breccia-cli/**` and this
report was touched. `profile`, `serve`, `bench gemv`, `bench transfer` and the
`bench ops` cells of `docs/phase-3/04` are unchanged (byte-identical numbers for
every op except `hc`, §6).

---

## 2. GDN / SSM step — the 19 steps, and the two layouts it fixes

Dispatch order in `Gdn::forward` (one call = one layer, `T` tokens):

```
1  attn_qkv @ x      -> qkv [T,10240]        Proj (decode Gemv / batched GemvBatch)
2  attn_gate @ x     -> z   [T,6144]
3  ssm_beta @ x      -> beta_raw [T,48]
4  ssm_alpha @ x     -> alpha   [T,48]
5  gdn_conv.comp     conv(kernel 4) over [state(3) | qkv] then SiLU on all 10240, in place
6  gdn_qk_l2.comp    q,k -> x/sqrt(sum x^2 + eps), q also x 1/S (v untouched)
7  gdn_gate.comp     beta = sigmoid(beta_raw); gate = softplus(alpha + ssm_dt) * ssm_a
8  gdn_step.comp     decay -> sk -> rank-1 update -> read out; state in place
9  gdn_norm_gate.comp rmsnorm_128(step, ssm_norm) * sigmoid(z)
10 ssm_out @ tail    -> [T,2560]             Proj
```

Three details that are easy to get wrong and are pinned in code and tests:

* **The `+ eps` is on the raw sum of squares, not the mean.** `build_gdn_l2_norm`
  is `rms_norm(x, eps/n) * 1/sqrt(n)`, which is `x / sqrt(Σx² + eps)`; a
  mean-form implementation would be 11.3× off at `S = 128` (`gdn.hpp:70-79`).
* **q carries two scales, k one.** The L2 norm contributes `1/sqrt(S)` and the
  delta rule contributes another `1/sqrt(S)` on q (spec §5.2), so q is scaled by
  `1/S` and k by `1/sqrt(S)` — one multiply each, folded into the norm kernel.
* **Order inside the step**: decay **before** the rank-1 update, and the output
  read from the *updated* state (`delta-net-base.cpp:349-366`;
  `cpu::tests::decay_is_applied_before_the_rank1_update`).

### 2.1 Recurrent state layout (spec open question Q3 — decided)

```
state  :: (S, h_v, S) = (i, h, j), j fastest   ->  flat index (i*h_v + h)*S + j
conv   :: (d_conv-1, channels), oldest first   ->  flat index  k*channels + c
```

`(S=128, h_v=48, S=128)` f32 = **3 MiB per layer**, plus 120 KiB of conv history —
context-length independent, and the only thing a GDN layer carries across a token
(spec §12.3). This is **strata's** layout, not llama.cpp's `(S, S, H_v)`: for a
fixed `i`, a warp walking consecutive `j` reads contiguous memory, and `j` is the
axis both the output and the update stream along. The two layouts hold the same
three numbers with different strides, so a mismatch is silent — `gdn::cpu`
indexes the same formula and `tests/attn_conformance.rs` compares the state
buffer itself, not just the output.

**`T > 1` is the decode kernel applied `T` times**, inside one invocation and one
dispatch: every line of the recurrence touches only the `(j, h)` column, so one
invocation owns a column for the whole chunk and the kernel has **no barriers at
all**. That is the equality `ST/include/strata/core/verify.hpp:3-11` asserts
between strata's chunked and single-token GDN paths, and the conformance suite
runs the same 32-token sequence both ways (§4). Cost is linear in `T`; the
chunked/UT-transform formulation that would make prefill sublinear is §8 item 2.

---

## 3. QSA — paged f16 KV, the indexer, and the two sparse-path decisions

Dispatch order in `Qsa::forward`:

```
1-5  attn_q @ x -> q_full [T,12288] | attn_k, attn_v -> [T,512] | indexer q -> [T,512], indexer k -> [T,128]
6    qsa_q_norm.comp   split the q half of every head's 512-block (stride 512), RMSNorm -> contiguous q [T,24,256]
7    rmsnorm K (rows 2T, gamma attn_k_norm)
8    rmsnorm indexer q (rows 4T, gamma indexer.q_norm)
9-11 IMRoPE q (rows 24T), k (rows 2T), indexer q (rows 4T) at pos = pos0 + t
12   qsa_kv_append.comp  f16 K/V into the paged cache at cell pos0+t
13   qsa_idx_pool.comp   pooled[b] = rope(rmsnorm(mean(raw[4b..4b+3])), pos = 4b) for each block this call completed
14   qsa_idx_tail.comp   raw tail (3 slots) of the block still being filled
15   qsa_idx_score.comp  score[t][b] = 1/sqrt(128) * Σ_{h<4} relu(q_h . pooled_b)
16   qsa_topk.comp       top 512 visible blocks by score + the query's own block
17   qsa_attn_scores.comp + softmax   over 4 * max_ids slots, -1e30 mask
18   qsa_attn_apply.comp  Σ p · V, then × sigmoid(gate) from the q projection's second half
19   attn_output @ A -> [T,2560]
```

### 3.1 Attention KV — paged f16, **16 cells per page**

```
k_pool, v_pool :: u32 words, two f16 per word
  element index = ((page * n_kv_head + kv_head) * PAGE_CELLS + cell_in_page) * head_dim + d
page_table     :: u32 per logical page (identity today; no reader assumes it)
```

One page of one layer is 16 KiB of K + 16 KiB of V = **2048 B/cell**, exactly the
budget `docs/phase-3/05` §3 assumes (2 heads × 256 × 2 types × 2 B = **24 KiB per
token across the 12 layers**). `PAGE_CELLS = 16` (not strata's 4) because it is
the granule a paged allocator can move, it keeps a page a round 16 KiB per type,
and it is the smallest granule at which a deferred fetch is worth starting on the
MI50's ~4 MB L2. The indexer's 4-cell block is a **score** granule, not a storage
one, so the two are independent — M5 can change the page size or quantise the
body behind the page table without touching the indexer.

### 3.2 f16 storage is ours, not the driver's

Measured on the MI50 (gfx906, RADV): `packHalf2x16(0x3F5FDBFC)` stores `0x3AFE`
where round-to-nearest-even gives `0x3AFF`. **Every** mismatching element of a
16 896-element sweep was exactly one ulp **low** — the driver's packer truncates
there. That is a systematic bias on every cached `|K|` and `|V|` (up to 2^-11
relative) and it is a model-precision decision, so `shaders/f16_storage.glsl`
implements both directions itself (RNE pack, exact unpack) and
`qsa::cpu::to_f16` is the same rule in Rust. `qsa_kv_cache_layout_and_f16_precision`
holds them equal **element-for-element** (16896/16896) on both devices.

### 3.3 Indexer key cache — pooled only

```
tail        :: (idx_block-1, 128) f32     the block still being filled (3 rows)
pooled      :: (max_blocks,  128) f32     one pooled key per completed 4-cell block
```

≈ 128 B/cell amortised (strata's shape, `qsa.hpp:197-237`) instead of llama.cpp's
`2 × 128` f32 per cell = 1024 B/cell, which would be **3.0 GiB** at the 262144
context. The raw tail is 3 rows because the cell that completes a block arrives in
registers on the pooling step. Pooling happens in `qsa_idx_pool.comp` with the
rotation at the **block's first cell** (`4b`), and a block's four cells may
straddle a call boundary — the kernel sources each cell from the raw tail or from
this call's rows accordingly (5-token chunks exercise exactly that path in
conformance).

One documented simplification: strata keeps a **spare** pooled row for the
incomplete block so that block can be scored. Here the query's own block is always
appended to the selection list instead — the same guarantee (a query can always
attend to itself, which is what the spec's `+ idx_block - 1` cells are for) with no
row whose position moves every time a block completes.

### 3.4 Selection: per-token, deduplicated, and the 2051-cell budget

`width = min(n_kv, idx_top_k + idx_block - 1) = min(n_kv, 2051)`. The list is

1. the `min(n_blocks, 512)` highest-scoring blocks **visible to this query**
   (a block whose first cell is in the future scores `-1e30`),
2. plus the query's own block `pos / idx_block`, **only if the selection missed
   it** (a duplicated block would double-weight its four cells).

Ties resolve to the **lower block index** (descending score, ascending index on
equality) — the same rule `ops_router.comp` uses for the MoE top-10, and the one
strata's numpy-style argsort gives. Below 2051 cells every block is selected, so
the layer degenerates to dense causal attention, which is the identity behaviour
spec §10.4 requires and what lets a short-context output be compared against a
trivial dense implementation.

The sparse path is bounded: one query touches at most 513 blocks = 2052 slots, so
no dense 262144-cell path exists (spec Q5). `qsa_attn_apply.comp` reads V only for
the selected cells, through the page table, with the causal test per cell.

---

## 4. Conformance evidence

Method: `tests/attn_conformance.rs`, GPU vs the f64 references in `gdn::cpu` /
`qsa::cpu`, one sublayer per process, `timeout 90`,
`VK_LAYER_KHRONOS_validation`, MI50 first.

```bash
cargo build --release -p breccia-kernels --tests
BIN=./target/release/deps/attn_conformance-<hash>
for CARD in 0000:0e:00.0 0000:03:00.0; do          # MI50 first
  for OP in gdn qsa; do
    timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
        BRECCIA_KERNEL_DEVICE=$CARD BRECCIA_OP=$OP \
        $BIN --test-threads=1 --nocapture
  done
done                    # "6 passed" per invocation per GPU
```

| test | bar | MI50 | RX |
|---|---|---:|---:|
| GDN 32 tokens, decode chunks (`T=1`) — output | 1e-5 | **1.68e-7** | **1.64e-7** |
| ... recurrent state | 1e-5 | **4.00e-7** | **1.16e-6** |
| GDN 32 tokens, prefill chunks (`T=8`) — output | 1e-5 | **1.68e-7** | 1.68e-7 |
| ... recurrent state | 1e-5 | 4.00e-7 | 1.16e-6 |
| GDN state bit-stability | **bit-identical** | ✅ | ✅ |
| QSA indexer pooling (chunks straddling blocks) | 1e-5 | **2.98e-7** | **2.38e-7** |
| QSA paged KV layout + f16 encoding | **bit-exact** | **16896/16896** | 16896/16896 |
| QSA selection at 2062 cells (515 blocks, 513 selected) | **id list exact** | ✅ | ✅ |
| QSA attention probabilities at 2062 cells | 1e-6 | **1.20e-7** | **1.23e-7** |
| QSA attention output at 2062 cells | 1e-5 | **2.63e-6** | **2.34e-6** |
| QSA 2 layers × 64 tokens, `T=1` — output / ids | 1e-5 / exact | **8.52e-7** / ✅ | 8.30e-7 / ✅ |
| QSA 2 layers × 64 tokens, `T=8` — output / ids | 1e-5 / exact | **8.52e-7** / ✅ | 8.30e-7 / ✅ |

Notes on the two bars that are not "relative to a value":

* **The GDN state bit-stability test** is two things: two consecutive downloads
  agree bit for bit (transfer determinism), *and* a **no-op call** — `conv`,
  `qk_l2_norm` and `delta_step` with `tokens = 0`, which still dispatch — leaves
  every state and conv-history byte unchanged. It caught a real defect while
  being written: the first conv kernel rewrote the history unconditionally, which
  is fine, but the first `qsa_idx_tail` did *not* guard its writes and raced when
  a chunk spanned more than one block (§5).
* **The QSA f16 test** is the strongest statement available for the cache: the
  physical address of every element and its exact binary16 encoding are checked
  against the documented layout, so a page-table mistake, a head-stride mistake or
  a rounding-mode change fails loudly instead of producing plausible numbers.
* The reference **rounds K/V to f16 exactly as the cache does**, so the 2.6e-6 /
  8.5e-7 numbers measure arithmetic, not the storage decision. With an f32 cache
  the same tests sit at 2.4e-6 / 8.5e-7 — i.e. the cache's precision is *not* the
  dominant term at these magnitudes, which is why f16 is the default (spec Q8:
  quantised bodies are an M5 choice, measured before adoption).

### 4.1 Two defects this suite caught (worth recording)

1. **`qsa_idx_tail` raced.** The first version wrote every cell of the call into
   the 3-slot raw tail. Two cells of the same chunk can share a slot (cells
   `4b..4b+3` with `slot = cell % 4`), so two invocations wrote the same address
   with no ordering. Symptom: block 0 pooled correctly, every later block off by
   ~1.4 relative, *and different on every run*. The fix writes only the cells of
   the **last, incomplete** block — the only keys a later call can still need —
   which makes each slot unique. (The whole suite was non-deterministic until
   this was fixed, because the paged cache was being corrupted by the same chunk.)
2. **The batched prefill append wrote past the cache.** `kv_append` appends
   `ws.tokens` cells; a caller with a 14-token tail chunk and a 256-token
   workspace wrote 242 uninitialised cells into pages beyond the page table. There
   is now a host-side range check (`pos0 + tokens <= capacity_cells`) that returns
   an error instead of reading an unmapped page index — a GPU page fault, not a
   wrong number, is the failure mode there.

---

## 5. HC `inject` fold — the measured win (`docs/phase-3/04` §7.2 item 1)

`inject[c] = Σ_{c',d} xn[c'][d] · w_inject[c][c',d]` is a 4 × 10240 projection: 82 KiB
of BF16 read by a single 8-row workgroup, measured at **21.85 µs** on the RX — pure
dispatch latency, 36 % of the HC_READ chain. It is now folded into
`ops_hc_gated_mean.comp`, which already walks every `xn` value for `mixed`: each
workgroup writes its 256-wide **partial** sums to `part[t][c][k]`, and
`ops_hc_write.comp` — which reads the four gates once per token anyway — sums the
`n_embd/256 = 10` partials. One dispatch fewer, no atomics, and a summation order
fixed by the chunk index rather than by race.

| | before (`04` §5) | after | Δ |
|---|---:|---:|---:|
| RX `stage-inject` (4×10240) | 21.85 µs | — (folded) | |
| RX `stage-gated` | 2.34 µs | 7.38 µs | +5.04 |
| **RX HC_READ chain, T=1** | **60.45 µs** | **44.43 µs** | **−16.02 µs (−26.5 %)** |
| RX HC_READ chain, T=256 | 1869.7 µs | 1864.7 µs | −5.0 µs |
| MI50 HC_READ chain, T=1 | 154.49 µs | 118.96 µs | −35.5 µs (−23 %) |
| MI50 HC_READ chain, T=256 | 6241.4 µs | 6237.1 µs | −4.3 µs |

**96 HC_READs per token** (2 per layer × 48) ⇒ **−1.54 ms/token (RX)**,
**−3.4 ms/token (MI50)** — 3.2 % of the 48.1 ms parity budget on the RX, and half
of the 2.1 ms the `inject` GEMV alone was predicted to cost, because the fold
adds 5 µs of work. At `T = 256` the saving is negligible: the batched GEMV was
already amortised across the chunk.

Conformance is **unchanged**: `hc` cells re-measured identically on both GPUs
(`mixed` 1.98e-7, `inject` 1.14e-7, `write` 0 — the `04` table's 2.1e-7 / 1.1e-7 /
0), and all 8 other ops' cells reproduce `04`'s numbers (`final-{mi50,rx}-*.log`).
Two API consequences for the integrator: `HcWorkspace.inject` now holds the
**partials** (`[T, hc, chunks]`; `HcMixer::inject_values` sums them on the host for
tests and debug dumps), and `HcMixer::STAGES` is 5 entries — `gated` now carries
the `inject` cost, so the per-stage table no longer has an `inject` row.

One thing this pins from the open questions: the dot is over the **whole 10240
stack** (`w_inject` is `[10240, 4]` in ne order, so row `c` is 10240 wide). The
first version of the fold implemented four per-stream reductions instead and the
conformance test rejected it at 7.6e-1 relative — the shape only admits the
mixed reading (`docs/phase-3/04` §7.1 item 5).

---

## 6. Measured — latency and extrapolation

Commands (one op per process, `timeout`, validation on, MI50 first; the `--json`
files are the artifacts):

```bash
for OP in gdn qsa hc; do
  for CARD in 0000:0e:00.0 0000:03:00.0; do
    timeout 250 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
      ./target/release/breccia bench ops --op $OP --tokens 1,64,256 \
      --device $CARD --json p3-attn/bench-<tag>-$OP.json
  done
done
```

`µs` is one dispatch chain, best of 3 windows after a discarded warm-up, GPU
timestamps, full barrier between dispatches. The **layer total is the sum of its
stage cells** — the same dispatches with the same barriers, so the sum is what one
sublayer costs (`Gdn::stage_timings` / `Qsa::stage_timings`).

Weights are synthetic **BF16** at the model's own shapes: that is the storage
class of `hc_*`/`ssm_alpha`/`ssm_beta`, and it gives the GEMV kernels the same
bytes-per-token to move. `attn_qkv`/`attn_q` are quantised in the artifact
(IQ4_XS/IQ3_S), so §6.3 converts the measured rates to the real types rather than
quoting a BF16 number as if it were the model's.

### 6.1 GDN/SSM per layer (RX 6950 XT / MI50)

| stage | T=1 RX | T=1 MI50 | T=256 RX | T=256 MI50 | note |
|---|---:|---:|---:|---:|---|
| qkv GEMV (10240×2560) | 31.65 | 77.63 | 1 464.0 | 5 355.6 | 50 MB BF16 |
| z GEMV (6144×2560) | 22.97 | 53.52 | 866.9 | 3 179.6 | 30 MB |
| beta / alpha GEMV (48×2560) | 6.30 / 6.45 | 13.60 / 13.62 | 7.7 / 7.8 | 28.1 / 28.1 | latency floor |
| conv+SiLU (10240 ch × 4 taps) | 1.95 | 5.22 | 18.18 | 37.44 | 121 KB history |
| q/k L2 norm | 2.03 | 5.47 | 4.26 | 10.33 | 2×16×128 rows |
| beta/gate | 1.60 | 4.46 | 1.68 | 4.59 | 48 values |
| **delta rule** | **43.41** | **92.91** | **~4 900** | **18 339.7** | 9 MB state traffic/token ⇒ 217 GB/s |
| norm+gate | 2.17 | 5.86 | 5.48 | 14.35 | 6144 values |
| ssm_out GEMV (2560×6144) | 38.77 | 85.42 | 1 182.5 | 3 443.9 | 30 MB |
| **layer total** | **157.3 µs** | **357.7 µs** | **22 597 µs** | **66 326 µs** | |

`T = 64` on RX: 5 726.7 µs (89.5 µs/token); `T = 256`: 22 597.3 µs (88.3 µs/token).
On MI50: 16 674.8 µs (260.5 µs/token) and 66 325.6 µs (259.1 µs/token).

Two readings:

* **The delta rule is the only stage that is not a GEMV** and it is 28 % of a
  layer at `T=1` (43.4 µs for 9 MiB of state traffic = 217 GB/s, i.e. it is
  latency/stride-bound, not DRAM-bound — 2.6× off the card's measured read
  ceiling). Its per-token cost falls to ~19 µs at `T = 256` because one invocation
  keeps its column for the whole chunk.
* Every other stage is at or below the dispatch floor (~2 µs) except the three big
  GEMVs, which run at 1 650 GB/s (RX) — above the 561 GB/s DRAM ceiling, i.e. the
  110 MiB of synthetic weights are Infinity-Cache-resident across the repeats.
  That is why §6.3 must not be a straight ×36.

### 6.2 QSA per layer (RX / MI50)

| | T=1, empty cache (16 slots) | T=1, **2052 slots** (2051-cell budget) | T=64 (68 slots) | T=256 (324 slots) |
|---|---:|---:|---:|---:|
| RX layer total | 131.8 µs | **274.4 µs** | 3 369.7 µs | 13 381.7 µs |
| — of which attention (scores+softmax+apply) | 8.5 | **148.4** | 36.3 | 234.4 |
| — of which q GEMV (12288×2560) | 37.1 | 37.1 | – | – |
| — of which out GEMV (2560×6144) | 39.4 | 39.4 | – | – |
| MI50 layer total | 310.5 µs | **859.3 µs** | 10 794.3 µs | 42 920.3 µs |
| — MI50 attention (scores+softmax+apply) | 21.5 | **561.3** | 79.4 | 544.5 |
| RX per-token at chunk | – | – | 52.7 µs | 52.3 µs |

The stage split at the full budget (RX): q GEMV 37.1, out GEMV 39.4, k/v/indexer
GEMVs 26.3, norms 6.8, rope 6.2, kv-append 2.5, indexer (tail+pool+score+top-k)
7.7, **attention 148.4**. Inside the attention stage, `scores+softmax` is 145.6 µs
and `apply` is 2.8 µs — a **50× imbalance with a specific cause**: the score
kernel is one subgroup per (token, head), so a decode step fields only **24
subgroups** on a 60-CU card, each walking 2052 slots serially, while `apply`
fields 6 144 threads and reads a 2 MB V pool that stays in L2. Fixing that is the
top post-integration lever (§8 item 1) and is worth up to ~10× on the decode
attention stage.

### 6.3 What this adds up to per token (RX, `T=1`)

| block | measured layer | ×layers | with the artifact's real weight types |
|---|---:|---:|---:|
| GDN (36) | 157.3 µs (BF16 synthetic) | 5.66 ms | **≈ 3.9 – 5.7 ms** |
| QSA (12), ≤2051 cells | 274.4 µs | 3.29 ms | **≈ 1.9 – 3.3 ms** |
| QSA (12), empty cache | 131.8 µs | 1.58 ms | ≈ 1.6 ms |
| HC inject fold | −16.0 µs per read × 96 | **−1.54 ms** | −1.54 ms |
| **both new blocks, total** | | **8.95 ms** | **≈ 5.8 – 9.0 ms** |

The "real weight types" column replaces the synthetic BF16 GEMV bytes with the
artifact's per-tensor types (§6.4: `attn_qkv` 4.25 bpw ⇒ 13.9 MB instead of 50 MB,
`attn_q` IQ4_XS ⇒ 19 MB instead of 63 MB) at the GEMV rates `docs/phase-3/01`
measured for those types (271–470 GB/s): the projections drop by roughly
`(110.5 → 31) MiB / layer` on the RX's DRAM path, and everything else — the delta
rule's 9 MiB of state, the conv/L2/gate kernels, the indexer, the attention — is
type-independent and already measured. So both blocks together cost **6–9 ms of
the 48.1 ms parity budget (12–19 %)**, the `inject` fold returns 1.5 ms, and the
MoE (which these kernels do not touch) keeps the remaining ~39–42 ms.

---

## 7. Integrator notes — call sequence and buffers

### 7.1 GDN sublayer (per layer, per call)

```rust
let st = GdnState::new(gpu)?;                       // once per sequence, per layer
st.reset(gpu)?;                                     // zeroed at sequence start
let ws = gdn.workspace(gpu, t)?;                    // per call shape
gdn.forward(&st, &ws, &weights, &projections, &mixed, &block_out)?;
```

* `mixed` is `[T, 2560]` (HC_READ's output), `block_out` is `[T, 2560]` (what
  HC_WRITE consumes). Everything else is inside `ws`.
* `GdnProjections` needs five `(Proj, weight)` pairs: `attn_qkv` 10240×2560,
  `attn_gate` 6144×2560, `ssm_beta`/`ssm_alpha` 48×2560, `ssm_out` 2560×6144.
  Use `Proj::Single(&Gemv)` at `T = 1` and `Proj::Batch { pipe, map }`
  (`proj::batch_map_buf`) at `T > 1` — one dispatch either way.
* `GdnWeights` is the four tensors the kernels read directly:
  `ssm_conv1d.weight` (f32, ne `[4, 10240]`), `ssm_dt.bias`, `ssm_a`, `ssm_norm`.
* State budget: **3.0 MiB + 120 KiB per layer per sequence** ⇒ ~112 MiB for the 36
  layers, context-independent (§12.3). `GdnState::bytes()` reports it.
* Scratch at `T = 1`: 2×40 KiB + 4×192 B ≈ 81 KiB; at `T = 256`: 20.5 MiB.

### 7.2 QSA sublayer

```rust
let st = QsaState::new(gpu, capacity_cells, max_pos)?;   // once per sequence, per layer
let ws = qsa.workspace(gpu, t, n_kv_after)?;             // n_kv_after = pos0 + t
qsa.forward(&st, &ws, &weights, &projections, &mixed, &block_out, pos0)?;
```

* `n_kv_after` is the **cache length**, not the block count: the workspace derives
  `n_blocks = n_kv_after/4`, `max_ids = min(n_blocks, 512) + 1` and the score
  width `4 * max_ids` from it, so a 64-token call allocates a 68-slot matrix
  instead of the 2052-slot worst case.
* `QsaState` owns the six buffers the layer carries: `k_pool`, `v_pool`,
  `page_table` (identity today), `idx_tail`, `idx_pooled`, and an **identity
  position table** for the whole window (`pos[i] = i`), so no call ever re-uploads
  positions — `Rope::run_at(..., pos_base, pos_div)` addresses token `i`'s
  position as `positions[pos_base + row/pos_div]`.
* `QsaProjections` is six `(Proj, weight)` pairs: `attn_q` 12288×2560,
  `attn_k`/`attn_v` 512×2560, `indexer.q_proj` 512×2560, `indexer.k_proj`
  128×2560, `attn_output` 2560×6144. `QsaWeights` is the four norm gammas.
* Host-side guards that turn a memory-safety bug into an error: `kv_append`
  refuses a write past `capacity_cells` (§4.1), `idx_pool`/`idx_score` refuse a
  write past the pooled rows. Keep them — a GPU page fault on the display card is
  not a debugging experience worth having.
* Scratch at the full budget (`T=1`, 2052 slots): ~1.4 MB, of which the score
  matrix is 197 KB; at `T=256` with 324 slots it is 42 MB. The score matrix is the
  one buffer whose size grows with `4 × max_ids × 24 × T`, and it is what a
  flash-style tiled variant would remove (§8 item 1).

### 7.3 Where the HC fold changes the graph's calls

`HcMixer::read` is now **five** dispatches, not six; `HcWrite` sums the `inject`
partials inside the combine it already did. Nothing else moves. The graph must
pass `hc_*_inject.weight` to `read` (as before, via `HcWeights.inject`) and read
`inject` back with `HcMixer::inject_values` only for debug dumps.

---

## 8. Open questions and what remains

### 8.1 Open questions

1. **Indexer score scale** (spec Q4). Implemented with llama.cpp's constant
   `1/sqrt(idx_dim)` per head (`qwen4exp.cpp:802-805`) rather than strata's raw-dot
   sum (`qsa.cu:276-277`). Ranking is invariant under a constant positive factor
   and both pin the current block, so the selection is the same; only a use of the
   score *as a probability* would differ. Not resolved empirically because no
   activation dump of the production indexer was available here.
2. **The query's own block when it is complete.** With `pos % 4 == 3` the query's
   block is complete and enters the score selection like any other, so a query can
   in principle fail to select the block containing itself if 512 better blocks
   exist — the artifact's own behaviour (llama.cpp scores every complete pool cell;
   strata's spare row only covers the *incomplete* block). Recorded rather than
   "fixed", because changing it changes output.
3. **f16 vs f32 K/V.** f16 is the spec's baseline (§12.1) and is within 2.6e-6 of
   an f32 cache on the same tests, so it costs nothing measurable at these
   magnitudes. Quantised bodies (int8/q4/K8V4) are M5; the page geometry and the
   page table already support them.
4. **`eos`/rollback.** Nothing here implements the state snapshot/rollback that
   speculation needs (spec §12.5, M4): `GdnState` is a plain buffer pair, so a
   rollback is `memcpy` of 3.1 MiB per layer per slot — the layout is deliberately
   snapshot-friendly (one contiguous recurrent buffer, one contiguous history).
5. **The conv kernel is per channel, `T`-sequential.** Correct and bit-equal to
   per-token calls, but a 10240-invocation grid with a serial inner loop is not the
   cheapest possible shape for a long chunk. Measured: 18.2 µs at `T=256`
   (1.8 µs/token), so it is not a problem yet.

### 8.2 Measured levers, in order of value

1. **Parallelise the attention score kernel over slots (up to ~10× on decode
   attention).** At the full 2051-cell budget the score stage is 145.6 µs of a
   274 µs layer because the grid is one subgroup per (token, head) = 24 subgroups
   at `T=1`. Splitting slots across subgroups (re-reading a 1 KB query per slot
   group, which is register-resident in the current form) or a flash-style tiled
   kernel that never materialises the 197 KB score matrix both attack it. Worth
   ~1.5 ms/token at 12 layers on the RX at full context.
2. **Chunked / UT-transform GDN prefill.** The delta rule is linear in `T`
   (43 µs at `T=1`, ~19 µs/token at `T=256`) because the state is re-read per
   token. The chunked formulation (chunk 64, `delta-net-base.cpp:16-284`) is
   sublinear in `T` and would matter for long prompts, not for decode.
3. **Delta-rule traffic at `T=1`.** 9 MiB of state per token per layer at 217 GB/s
   (2.6× off the card's ceiling). Keeping the 128-float column in local memory
   across the two passes, or a fused `sk`/update pass with a shared-memory
   reduction, is the obvious next step; it is 36 × 43 µs = 1.56 ms/token today.
4. **Tiled GEMM for the four large prefill projections** — the same lever
   `docs/phase-3/04` §7.2 item 3 asks for on the HC side; the GDN/QSA projections
   inherit the batched-GEMV shape (weights re-read per token) and at `T=256` they
   are 63 % of a GDN layer.

### 8.3 What remains before the first token

Ops-inventory coverage after this block (`docs/specs/qwen4exp-architecture.md` §13):

| # | primitive | status |
|---|---|---|
| 10 | depthwise causal conv1d (d=4, state carry) | ✅ `gdn_conv.comp` |
| 11 | GDN step (decay, rank-1 delta rule, output) | ✅ `gdn_step.comp` |
| 12 | GDN chunked prefill | ✅ per-token-equivalent in one dispatch; the sublinear chunked form is §8.2 item 2 |
| 14 | attention with additive mask (dense + selected) | ✅ `qsa_attn_scores/apply` |
| 15 | top-k over the selection width | ✅ `qsa_topk.comp` (ties → lower block) |
| 16 | indexer score (ReLU dot + tail bias) | ✅ `qsa_idx_score.comp` |
| 17 | indexer key pool | ✅ `qsa_idx_pool.comp` |
| 18 | KV append / gather (paged f16) | ✅ append ✅ sparse read through the page table |
| 9 | `softplus` | ✅ `gdn_gate.comp` |

Still before a first token: **PLE device block** (layer 1) and **lm_head sampling**
(`docs/phase-3/05` §6) — and the graph itself.

---

## 9. Verification commands and results

```bash
cargo build --workspace                                            # Finished
cargo test --workspace                                             # 23 suites, all ok
cargo clippy --workspace --all-targets -- -D warnings               # clean
cargo fmt --check                                                   # clean

# conformance: one sublayer per process, MI50 first
cargo build --release -p breccia-kernels --tests
BIN=./target/release/deps/attn_conformance-<hash>
for CARD in 0000:0e:00.0 0000:03:00.0; do
  for OP in gdn qsa; do
    timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
      BRECCIA_KERNEL_DEVICE=$CARD BRECCIA_OP=$OP $BIN --test-threads=1 --nocapture
  done
done                    # "6 passed" per invocation per GPU; 0 validation messages
                        # (and the 8-op suite re-run: final-{mi50,rx}-<op>.log, 10 passed each)

# benchmarks: 3 ops x 2 GPUs, one op per process
for OP in gdn qsa hc; do
  for CARD in 0000:0e:00.0 0000:03:00.0; do
    timeout 250 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
      ./target/release/breccia bench ops --op $OP --tokens 1,64,256 \
      --device $CARD --json p3-attn/bench-<tag>-$OP.json
  done
done                    # 0 validation messages, 221 cells (14 + 33 + 80 per GPU)
```

Artifacts (`/var/tmp/users/korphaus/tmp/opencode/p3-attn/`):
`conf-{mi50,rx}-{gdn,qsa}.log` (4 conformance logs), `final-{mi50,rx}-<op>.log`
(16 re-run op logs + 2 `hc` with `--nocapture`), `benchlog-{mi50,rx}-<op>.log`
(6 bench logs), `bench-{mi50,rx}-<op>.json` (6 JSONs).

Safety: every GPU invocation ran under `timeout`, one op per process, one GPU at a
time, MI50 (`0000:0e:00.0`, no display) first and the RX (`0000:03:00.0`) only
after the MI50 passed — `AGENTS.md` GPU crash safety.
`journalctl -k -b | grep -cE "amdgpu.*(page fault|Illegal opcode)"` = **0** after
the whole window; no service was started or stopped.

---

## 10. Attributions

* **GDN semantics** — the 19-step list and the delta rule from
  `docs/specs/qwen4exp-architecture.md` §4.2/§5.2 (transcribed there from
  llama.cpp `src/models/delta-net-base.cpp:286-368` and strata
  `include/strata/kernels/gdn.hpp`), both MIT. The state layout is strata's
  `(S, h_v, S)`; the `+eps`-on-the-raw-sum and the two-scale q were read from
  `build_gdn_l2_norm` (`src/models/models.h:14-18`) and `gdn.hpp:70-79`. No code
  was copied; the shaders are this crate's own.
* **QSA / lightning indexer semantics** — pooling, scoring, top-k and the gated
  output from `LC/src/models/qwen4exp.cpp:738-1013` and
  `ST/include/strata/kernels/qsa.hpp` / `src/kernels/cuda/qsa.cu` (MIT). The
  per-token own-block rule and the pooled-only indexer cache are this crate's
  choices, documented in §3.3/§3.4.
* **f16 storage** — the encoding in `shaders/f16_storage.glsl` is a plain IEEE
  binary16 round-to-nearest-even conversion, written because the driver's
  `packHalf2x16` truncates on gfx906 (§3.2). `qsa::cpu::to_f16` is the same rule.
* **HC mixer algebra** — unchanged from `docs/phase-3/04` §8 (llama.cpp
  `qwen4exp.cpp:313-397`, strata `gr.hpp:77-124`); this block only moves the
  `inject` projection inside the gated-mean dispatch.
* No code from IsoQuant or RotorQuant (unlicensed) was consulted or copied.
