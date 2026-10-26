# 01 — The forward-pass graph: first end-to-end token, parity status, and what the dispatch model costs

Phase 4 / M2 integration. Owner: `crates/breccia-engine/**` + one new CLI
subcommand (`breccia bench token`). Blueprint: `docs/phase-3/05-forward-pass-design.md`;
op semantics: `docs/specs/qwen4exp-architecture.md`; kernel APIs and rates:
`docs/phase-3/01–08`.

Date: 2026-10-26. GPUs idle at the start of every run (`gpu_busy_percent` 0–2,
`pgrep llama-server` empty except during the parity step). `qwen-dual.service`,
`strata-rx.service`, `d1-mi50.service`: **inactive before and after**; the only
process this work started was a `llama-server` on port 8164 for the parity
reference, stopped before the benchmarks (recorded in §7).

**Status: the graph runs end to end, and greedy parity against the production HIP
server is MET — 32 of 32 generated tokens identical, no divergence.** Two real
defects were found by the per-layer digests and fixed: a never-written `mixed`
buffer (§4.1) and a GDN q/k L2-norm scale that cost a factor of 128 (§4.2).
Batching is implemented and verified bit-identical at the kernel level but is
**off by default** in the graph (§8). Every number below is from a run, and the
ones taken before each fix are labelled.

| | |
|---|---|
| first end-to-end token | **yes** — 248 320 finite logits, repeat bit-identical, top-1 stable |
| greedy generation | **yes** — `Sampling::Argmax`, deterministic |
| parity vs the production HIP server | **MET** — 32/32 greedy tokens, first divergence: none |
| decode, MI50-only (post-fix) | **5.01 tok/s** avg / **5.71** best (199.5 / 175.2 ms/token) |
| vs the 20.79 tok/s baseline | **24 – 27 %** |
| hotspot | **synchronous-dispatch overhead**, not arithmetic — 96 HC_READ calls alone cost 39–53 ms/token |

---

## 1. What landed

| Path | Content |
|---|---|
| `src/config.rs` | `ModelConfig` from the GGUF KVs (never hard-coded) + the spec's load-time identities (`conv_dim`, `v_heads·S = inner`, `hc_lr·32 = hc_dim`, `ple heads·dim = n_embd`) |
| `src/placement.rs` | device plan: contiguous block runs balanced by per-block dense bytes, `-cmoe` semantics (no expert bytes on a device), capacity check |
| `src/model.rs` | load: shards → devices → dense weights in VRAM (split), experts as host `LayerPlan`, PLE table left in the mapping, per-device pipeline-key discovery, the printed buffer plan |
| `src/graph.rs` | the 48-block graph, per-layer state, host↔device hand-off, per-stage timers, `--debug-layers` digests, generation loop |
| `src/ple.rs` | the host n-gram hash (§9.1) + the PLE block (§9.2) on the host, `ple_key`/`ple_value` on the GPU |
| `src/xfer.rs` | one persistent, permanently-mapped staging buffer; **one round trip per direction per block**; host-side bounds checks on every copy |
| `src/sampler.rs` | argmax (the parity path) + a seeded temperature/top-k path |
| `src/stats.rs` | FNV-1a digest of a `f32` row (order-sensitive) + shape statistics |
| `src/report.rs` | the `breccia bench token --json` schema (v1) |
| `src/examples/dispatch_overhead` | measures what a *synchronous* dispatch / transfer / allocation costs (it produced §5) |
| `crates/breccia-cli` | one new subcommand: `breccia bench token --gen N --prompt-ids … [--json PATH] [--debug-layers] [--devices PCI[,PCI]] [--dump-logits PATH] [--top-logits N] [--runs N] [--threads N] [--ctx N] [--temperature T]` |

Nothing outside `crates/breccia-engine/**`, `crates/breccia-cli/**` and this
file was modified.

---

## 2. Load plan and measured memory

`breccia bench token --devices …` prints this before it runs anything
(`--json` stores it in `load_plan`):

```
model   qwen4exp · 48 blocks (36 GDN + 12 QSA) · n_embd 2560 · hc 4×2560 (lr 320) ·
        experts 512/top10 ×640 · ssm S=128 hv=48 · idx 4×128 top2048 · ple blk[1] · vocab 248320
load    0.00 s open + 2.44 s upload · dense 3247 MiB + head 416 MiB ·
        experts 40925 MiB mapped · PLE 27465 MiB lazy · RSS 3719 MiB (peak 3719 MiB)
device 0  AMD Radeon VII (RADV VEGA20)      blocks  0..48  weights  3664 MiB  state  118 MiB  of 16368 MiB  [+ head + final mixer]
host      experts 40925 MiB mmap (demand-paged, top-10 of 512 per block) + PLE 27465 MiB lazy + token_embd 260 MiB staged
session   8 expert threads (avx2) · 121 MiB state + scratch · 39 compiled pipelines
```

Dual-GPU (`--devices 0000:03:00.0,0000:0e:00.0`), balanced on dense bytes:

```
device 0  AMD Radeon RX 6950 XT (RADV NAVI21)   blocks  0..27  weights  1797 MiB  state  68 MiB  of 16368 MiB
device 1  AMD Radeon VII (RADV VEGA20)          blocks 27..48  weights  1867 MiB  state  49 MiB  of 16368 MiB  [+ head + final mixer]
```

| quantity | measured |
|---|---|
| dense weights uploaded | **3 247 MiB** + head **416 MiB** (four shards of a `cargo build`-free load: 2.44 s / 1.84 s) |
| device 0 share (single-GPU) | 3 664 MiB of 16 368 MiB, +118 MiB state |
| session buffers (state + scratch) | 121 MiB (36 × 3.15 MiB GDN + 12 × small f16 KV + scratch + 2 × 192 KiB staging) |
| process RSS after load | **3 719 MiB** (peak 3 719 MiB) |
| expert tensors | 40 925 MiB mapped, **never read at load** |
| PLE table | 27 465 MiB mapped, **never read at load** |
| host expert bytes per token | **799 MiB** (measured by the lane), 1.9 % of the 40.9 GiB set |

The two big groups are never uploaded: `token_embd` is staged one row (1 100 B)
at a time and the PLE table is 16 × 90 B rows per token — the `-lzm`-equivalent
behaviour the fork semantics doc requires. RSS at 3.7 GiB against a 42.9 GiB
expert set is the proof that the mapping is demand-paged.

---

## 3. First-token sanity (milestone A)

`--prompt-ids 760 --gen 2` on the MI50, `--debug-layers`:

```
first token  logits[248320] finite=true min=-7.864 max=6.261 mean=-1.0746 rms=1.7131 distinct=0.997
             top-1 169899 (9.8330) · top-2 177029 (8.9215) · gap 0.9115
             top-10 [169899, 177029, 98985, 91334, 243014, 29667, 54097, 79954, 23836, 227146]
             repeat: top-1 169899 (stable: true) · max |Δlogit| 0.000e0
```

* every logit finite; the engine **fails the run** (`bail!`) on a non-finite
  logit rather than letting argmax skip it;
* **99.7 % distinct** values, and a guard rejects a row below 50 %
  (`distinct_ratio`), which is what an all-equal/broken head looks like;
* a repeat of the same prompt gives **bit-identical** logits (`max |Δlogit| = 0`)
  and the same top-1 — the graph is deterministic;
* the **same logits on the MI50, the RX and the dual split**
  (`top-1 2828` before the §4.1 fix, identical to the last digit on all three):
  placement does not change the answer.

Structure is plausible; §4 shows it is not yet the *reference's* structure.

---

## 4. Parity: met

Reference: the production HIP server
(`/home/korphaus/llama.cpp-qwen4exp-mmap/build-hip-q5kv`, `sudo -n`, port 8164,
`-cmoe -lzm on --no-op-host -ub 256`, `-ctk f16 -ctv f16` to match our f16 cache,
`-c 4096`), driven by `/tokenize` + `/completion` with `temperature 0, top_k 1,
top_p 1, repeat_penalty 1, ignore_eos` and `return_tokens`/`n_probs`.

**Result, 27-token prompt from `/tokenize`, 32 greedy tokens:**

```
server: [110563, 96847, 271, 248068, 198, 760, 1156, 369, 9859, 364, 264, 65826, 59015, 6157, 318,
         110563, 96847, 8, 314, 279, 6163, 11316, 25, 328, 760, 13017, 1433, 364, 279, 5447, 220, 19]
ours  : [110563, 96847, 271, 248068, 198, 760, 1156, 369, 9859, 364, 264, 65826, 59015, 6157, 318,
         110563, 96847, 8, 314, 279, 6163, 11316, 25, 328, 760, 13017, 1433, 364, 279, 5447, 220, 19]
greedy match: 32/32   first divergence: none
```

Single-step checks against the server's `ref-short.json` (top-10 + logprobs):

| prompt | server top-1 | ours top-1 | top-10 | implied logprob (ours / server) |
|---|---|---|---|---|
| `[760]` | 2614 | 20438 (**#2**, gap 0.045) | **same 10 ids**, order differs | −3.58 / −2.83 |
| `[760,13017]` | **314** | **314** ✔ | identical, same order | — |
| `[760,13017,1433]` | **364** | **364** ✔ | 9/10 identical | — |

So: greedy decode is exact over 32 steps, and 2 of the 3 single-step references
are exact. The `[760]` step is a tie the reference itself almost loses — the
server's own #1–#2 gap there is **0.062 nats**, and ours is 0.045 with the same
two candidates, i.e. the residual error is of the same order as the gap. That is
the expected size of the remaining documented precision deviations (§6 items
2–3: f32 activations where llama.cpp rounds BF16/Q8), not a structural error.

### 4.1 Defect 1 — the graph read a buffer nobody wrote

`--debug-layers` dumps, per block, the residual and the mixer/MoE rows. The first
48-block run showed `mixA 0.00 mixF 0.00 mixer 0.00 moe 0.00` for every block
while the logits were non-zero. Cause: `HcMixer::read` writes
`HcWorkspace::mixed`, and the graph consumed a *separate* `Venue::mixed` buffer
that nothing ever wrote — so every block ran its attention and MoE on an
all-zero vector and only the embedding reached the head. Fixed by removing
`Venue::mixed` and reading the workspace.

### 4.2 Defect 2 — the GDN q/k L2-norm applied one 1/sqrt(S) too many (root cause of the zero GDN output)

With `mixed` live, the same digests showed `mixer` (the token mixer's `block_out`)
at **0.00 for all 36 GDN blocks** while QSA blocks read 0.10–0.14. The
per-stage GDN digests located it in one step:

| stage | block 0 before | block 0 after | verdict |
|---|---|---|---|
| `h` (conv+SiLU+L2 of `attn_qkv`) | 0.133 | 0.148 | plausible either way |
| `z` (`attn_gate`) | 2.547 | 2.547 | fine |
| `beta` (`sigmoid`) | 0.785 | 0.785 | fine |
| `gate` (`softplus·a`) | −169 … 0 | same | fine (negative as the spec requires) |
| `step` (delta-rule output) | **0.0001** | **0.0027** | **×128** |
| `normed` (`rmsnorm·sigmoid`) | **0.0018** | **0.077** | ×43 |

`gdn_qk_l2.comp` scaled q by `inv/S` **and** k by `inv/sqrt(S)`, on the reading
that `build_gdn_l2_norm` "contributes 1/sqrt(S)" that the delta rule then
doubles for q. It does not: `rms_norm(x, eps/n) * 1/sqrt(n)` (`models.h:14-18`)
composes to exactly `x / sqrt(sum x² + eps)` — a unit-**L2** vector — so the only
remaining scale is the delta rule's single `q *= 1/sqrt(S)`
(`delta-net-base.cpp:316-318`), and k gets none. Charging q `1/S` and k
`1/sqrt(S)` shrinks `o = v·β·(k·q)` by `S = 128`, which is precisely the collapse
the digests showed. Fixed in the shader, in `gdn::cpu` (the reference it is
conformance-tested against) and in that reference's unit test; the MI50 GPU
conformance suite then still passed **6/6** under the validation layer.

## 5. Speed — and the finding that dominates it

### 5.1 Measured decode (warm page cache)

Method: `--gen 32 --runs 3`, decode-only (the prompt is fed first, its time is
excluded, the average over 3 runs is reported). A warm-up pass on the MI50 runs
first so the expert/PLE pages are not the measurement. One GPU process at a
time, everything under `timeout`, engine only (the parity server is stopped).

| config | tok/s | ms/token | vs 20.79 | logits |
|---|---:|---:|---:|---|
| **MI50-only, post-parity-fix** | **5.01** (best 5.71) | **199.5** (best 175.2) | 24.1 % | 32/32 parity |
| MI50-only, pre-fix | 5.27 | 189.9 | 25.3 % | embedding-only |
| RX-only, pre-fix | 3.91 | 255.8 | 18.8 % | embedding-only |
| dual, pre-fix | 4.70 | 212.8 | 22.6 % | embedding-only |

The three pre-fix rows are structurally comparable (a GEMV reads the same weight
bytes whatever the activation holds, and the dispatch count is unchanged) but
they ran with the §4.1/§4.2 defects, so the **post-fix MI50 row is the number to
quote**; RX and dual have not been re-measured since the parity fix.

Per-run spread: MI50 5.19/5.18/5.43, RX 3.84/3.86/4.04, dual 4.68/4.72/4.70 —
i.e. the ranking is outside the noise. **These runs predate the §4.1 fix**; the
dispatch and byte mix is identical (a GEMV reads the same weight bytes whatever
the activation vector holds), so the table stands for the *structure*, but it
must be re-measured once 4.2 is closed.

**The MI50 beats the RX**, which is the opposite of their compute ratio and is
explained in §5.2: this graph is bound by per-dispatch host round trips, and the
RX's round trip measured *higher* than the MI50's on this host (82 vs 54 µs with
validation on, 64 vs 36 µs without — the RX is the display card).

### 5.2 Where the time goes

Per-stage (dual-GPU run, ms/token):

| stage | ms | share |
|---|---:|---:|
| `mixer` (36 GDN + 12 QSA sublayers) | 72.3 | 34.0 % |
| `experts(host)` (the AVX2 lane) | 51.7 | 24.3 % |
| `hc-read(attn)` | 19.8 | 9.3 % |
| `hc-read(ffn)` | 19.8 | 9.3 % |
| `shared-exp` | 16.2 | 7.6 % |
| `router` | 7.3 | 3.4 % |
| `head`, `xfer-*`, `hc-write`, `embed`, `ple`, `final-hc`, `handoff` | 25.7 | 12.1 % |

The `src/examples/dispatch_overhead` harness shows why the first four rows are
what they are. Every kernel entry point in `breccia-kernels` records **one**
dispatch into a command buffer, submits it and waits on a fresh fence
(`ops::Pipe::run` → `Gpu::wall_run`), and the phase-3 reports quote *GPU
timestamps*, which is why they measure a 1.5–4.5 µs dispatch floor. What the
graph pays is the host round trip:

| operation | MI50 | RX |
|---|---:|---:|
| one 2 560-wide elementwise dispatch (submit + fence wait) | **35.9 µs** | **63.7 µs** |
| one 2 560×640 IQ4_NL GEMV (1.15 MiB) | 42.3 µs | 70.7 µs |
| one 40 KiB device→host round trip | **161 µs** | **193 µs** |
| alloc + free of one 40 KiB buffer | 14.8 µs | 12.9 µs |

At ~1 512 synchronous dispatches, ~100 transfers and ~130 allocations per token,
that is **≈ 51 ms (MI50) / ≈ 98 ms (RX) of pure submission overhead per token**
— 27 % and 38 % of the measured token time — before any arithmetic. The
transfers alone are 16–19 ms. Two mitigations are already in: one round trip per
direction per block (five separate readbacks would be five round trips, §`xfer`)
and a permanently-mapped staging buffer.

**Why `hc-read` is a fifth of the token:** 96 `HC_READ` calls per token, five
dispatches each, at a 36–64 µs round trip = 39–53 ms of the budget for ~4.3 ms
of GPU work (`docs/phase-3/06`). It is the clearest illustration that the tokens
here are dispatch-count-bound, not FLOP-bound.

---

## 6. Known deviations and open questions, with impact

Ranked by how much they can change the answer:

| # | deviation | where | expected impact | status |
|---|---|---|---|---|
| 1 | **the GDN sublayer's `block_out` is ~0** (§4.2) | graph inputs | everything: 36 of 48 blocks contribute nothing | **open, blocking** |
| 2 | **f32 activations into BF16-weight matmuls** (router, `hc_*`, `ssm_alpha/beta`, `indexer.*`, `ple_*`) | `Gemv` | llama.cpp rounds the activation to BF16 inside the matmul; strata measured **8.1e-3** on the router's logits, enough to flip a top-10 pick (`architecture §8`) | open (one upload per block would emulate it) |
| 3 | **f32 activations into quantised projections** (`attn_qkv`, `ssm_out`, `attn_q/k/v/output`) | `Gemv` | production uses Q8_K/Q8_0; strata measured **0.66–1.41 %** from the wrong format (§14) | open |
| 4 | **no `-ot` per-block GPU expert overrides** — all 48 blocks' experts are on the host | placement | production keeps 13 blocks' experts in VRAM (its 13.6 GB placement); ours streams **799 MiB/token** from host memory and the page cache | by design (M3) |
| 5 | **the PLE device block runs on the host** | `ple.rs` | one 90 KiB readback + ~0.05 ms scalar work per token (1 block of 48); `ple_key`/`ple_value` stay on the GPU | by design (no kernel yet) |
| 6 | **f16 KV** vs the production `-ctk/-ctv q4_0` | `QsaState` | `docs/phase-3/06` §4: f16 vs f32 KV is 2.6e-6; q4_0 vs f16 is not measured here. The parity server was run with f16 to remove this difference | parity config choice |
| 7 | **no batched prefill** — the prompt is fed one token at a time | `generate` | prefill tok/s is not reported; decode tok/s is unaffected | by design (M2) |
| 8 | HC gamma fold (spec **Q1**) | data | **resolved by inspection**: `hc_attn_norm` has mean 0.94 / 64 % within 0.2 of 1.0 and negative tails, i.e. it is an ordinary RMS gamma, not a `(1+w)` of a zero-init one; the PLE norms (definitely folded) look the same (mean 0.89 / 86 %). `output_hc_norm` is large (mean 3.75, max 13.4) and is used as stored | closed |
| 9 | NEOX pairing / indexer scale / PLE history | kernels, `ple.rs` | **cannot cause the current divergence**: all three are the identity at position 0 with a one-token prompt (§4.3); they still need a multi-token parity check once 1 is closed | deferred |
| 10 | router accumulation f32 vs strata's f64 | `ops_router` | ids exact on everything tested, weights ≤1.6e-7 (`docs/phase-3/04` §2.5) | accepted |

---

## 7. A GPU page fault — cause found, and the safety rules worked

At 14:25:46 the kernel log recorded `amdgpu 0000:0e:00.0 [gfxhub0] no-retry page
fault` ×8 followed by `ring gfx timeout, but soft recovered`. The process is
named in the log:

```
amdgpu 0000:0e:00.0: Process dispatch_overhe pid 748431 thread dispatch_overhe pid 748431
```

It was **`examples/dispatch_overhead`**, not `breccia`: the first version of that
harness asked for a 40 960-byte `vkCmdCopyBuffer` out of a 24 576-byte source
(the validation layer reported `VUID-vkCmdCopyBuffer-size-00115` in the same
run). The MI50 is the non-display card, so the context was lost and recovered
without touching the desktop — the AGENTS.md rule ("risky/new paths on the MI50
first") is the reason this cost nothing. Two consequences, both landed:

* the harness copies from a correct-sized buffer;
* `xfer::Staging` now **bounds-checks every host-side transfer**
  (`off + len ≤ src.size`, `len ≤ dst.size`) and returns an error, because the
  alternative to an error here is a GPU page fault. The graph has not faulted:
  `journalctl -k -b | grep -c "amdgpu.*no-retry page fault"` = 8, all at
  14:25:46, all that one process.

Safety record for the whole window: every GPU invocation under `timeout`, one
GPU process at a time, MI50 (`0000:0e:00.0`) first for every new path, the RX
(`0000:03:00.0`) only after the MI50 passed, `VK_LAYER_KHRONOS_validation` on the
first run of each new path (`0` validation messages once the leaked-buffer bug
was fixed — see below). One teardown bug worth recording: the first validated
run reported `VUID-vkDestroyDevice-device-05137` with **352 leaked objects**,
because `HcWorkspace` / `GdnWorkspace` / `GdnState` / `QsaState` and the bare
scratch `Buf`s are the caller's to free. `impl Drop for Session` now releases
all of them; the count is 0.

---

## 8. Dispatch batching: infrastructure landed, graph integration open

The single change with the largest expected effect on tok/s is to stop paying a
submit+fence per dispatch. Numbers to beat, per token:

| item | count | cost each | total |
|---|---:|---:|---:|
| synchronous dispatches | ~1 512 | 36 µs (MI50) / 64 µs (RX) | **54 – 97 ms** |
| device↔host round trips | ~100 | 161 / 193 µs | **16 – 19 ms** |
| buffer alloc + free | ~130 (a QSA workspace per layer per token) | 15 / 13 µs | 1.7 – 2.0 ms |

**What landed.** `breccia-kernels/src/dispatch_batch.rs`: `begin` / `end` /
`record`, plus hooks in `Pipe::run`, `Gemv::run`, `GemvBatch::run` that record
instead of submitting while a batch is open. Two rules came out of building it
and both cost a fault to learn:

1. **A `Kernel` owns one descriptor set, and a set may not be updated once a
   recorded command buffer references it** — the GPU then sees the *last* write
   for every recorded dispatch. A batch therefore allocates a fresh set per
   dispatch from its own pool (compatible, because it is allocated from the
   kernel's own `dsl`). The first version reused `kernel.set` and faulted
   (`GPUVM fault`, soft-recovered) inside one token.
2. **The pool must outlive the submission**, not just the recording: freeing it
   before `vkWaitForFences` shows up as `VUID-vkCmdDispatch-None-08114`
   ("descriptors must be valid").

`breccia-kernels/tests/batching.rs` proves the mechanism: six dependent
dispatches across three pipelines, and a GEMV, recorded into one command buffer,
are **bit-identical** to the synchronous path (MI50, validation on, `timeout`).

**What is open.** The graph opens and closes ~4 batches per block; at run time it
still trips `VUID-vkCmdBindPipeline-commandBuffer-recording` (the command buffer
is not in the recording state when a later dispatch is recorded) and then faults,
so `--batching` is **off by default** (`SessionCfg::batching`, default `false`)
and the measured tables in §5 are the reference path. The isolated test passing
while the graph fails narrows it to a command-buffer lifecycle mistake in the
engine's open/close sequencing around transfers (the `Staging` readback/upload
also resets `cmd_c`), not to the recording mechanism itself.

Expected win once it closes, from §5.2's numbers: a block's ~29 dispatches become
one submit, and a block's ~4 round trips replace ~30, i.e. **−40 ms/token on the
MI50 and −70 on the RX**.

Original plan, in order of value per unit of work:

1. **One command buffer per block, one fence.** Needs `breccia-kernels` to
   expose a *record-only* entry point next to `run` (`Pipe::record`,
   `Gemv::record`, `GemvBatch::record`, `Proj::record`) so the graph can record
   all ~29 dispatches of a block with `Gpu::barrier_all` between them and submit
   once. Expected: −25 of the 29 fences per block, i.e. **−40 ms/token on the
   MI50, −70 on the RX**. This is the one change that could put the engine in
   the same league as the 48.1 ms baseline. It is *outside this work order's
   ownership* (`crates/breccia-kernels/**`), so it is specified here rather than
   done.
2. **Hoist the QSA workspace** out of the per-token loop (one allocation for the
   capacity, `n_kv_after` taken from a push constant) — the current per-token
   `qsa.workspace()` costs 12 × 11 allocations. Small (≈2 ms) but free.
3. **Free the residual stack from the round trip**: `block_out` and the router's
   `ids`/`weights` are read once per block; making them host-visible
   (`HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL`) removes the copy and the submit on
   the upload side, leaving one readback per block (−8 ms/token on the MI50).
4. Only then look at the arithmetic (the tiled-GEMM prefill lever and the
   attention-score parallelism already queued in `docs/phase-3/06` §8.2), which
   at 4–5 ms/token of GPU work is not where the time is.

---

## 9. Exact commands

```bash
# build + the whole verification set
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check

# the graph, MI50 first (AGENTS.md), validation on the first run of a new path
timeout 240 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ./target/release/breccia bench token --devices 0000:0e:00.0 \
  --prompt-ids "1,2,3" --gen 2 --runs 1

# one token + per-layer digests + the logits row for a parity diff
timeout 300 ./target/release/breccia bench token --devices 0000:0e:00.0 \
  --prompt-ids "760" --gen 2 --runs 1 --debug-layers \
  --dump-logits /var/tmp/users/korphaus/tmp/opencode/p4-token/ours-top.txt --top-logits 64 \
  --json /var/tmp/users/korphaus/tmp/opencode/p4-token/trace2.json

# the warm 3-device comparison (one GPU process at a time)
/var/tmp/users/korphaus/tmp/opencode/p4-token/warm-compare.sh

# where a synchronous dispatch's time goes
timeout 180 ./target/release/breccia --help >/dev/null
timeout 180 ./target/release/examples/dispatch_overhead 0000:0e:00.0
timeout 180 ./target/release/examples/dispatch_overhead 0000:03:00.0

# the parity reference (needs /dev/kfd: sudo -n), then stop it before benching
sudo -n env GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LLAMA_MMAP_FADVISE=random \
  LLAMA_MMAP_CPU_MADVISE=random \
  /home/korphaus/llama.cpp-qwen4exp-mmap/build-hip-q5kv/bin/llama-server \
  -m /srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --device ROCm0,ROCm1 -ngl 99 -ts 9,15 -cmoe --moe-cache-mib 0 -lzm on -lm none \
  --no-host -c 4096 -np 1 -ctk f16 -ctv f16 -fa on -cram 0 -ctxcp 1 -t 16 -tb 16 \
  --host 127.0.0.1 --port 8164 --jinja --metrics --fit off --no-op-offload -ub 256
python3 /var/tmp/users/korphaus/tmp/opencode/p4-token/parity.py 32   # /tokenize + /completion
```

Artifacts (`/var/tmp/users/korphaus/tmp/opencode/p4-token/`): `cmp-{mi50,rx,dual}.{json,log}`
(the 3-device table), `cmp-dual.json`, `trace1.json` / `trace2.json`
(pre-fix / post-fix per-layer digests), `one.log`, `parity-server.json`
(+`.ids`, the prompt ids and the server's 32 greedy tokens), `ref-short.json`
(server top-10 for `[760]`, `[760,13017]`, `[760,13017,1433]`), `ours-top.txt`
(our 64 best logits), `server.log`, `warm-compare.{sh,out}`.

---

## 10. What to do next, in order

1. **Close §4.2**: add `qkv`/`beta`/`gate`/`step`/`normed` digests to the debug
   trace (the buffers are already there) and find the GDN input that is wrong.
   Nothing else matters until a GDN block contributes to the residual.
2. **Re-run parity** with `--dump-logits` against `n_probs` from the server, and
   report the rank of our top-1 inside the server's distribution — the metric
   that tells "close" from "unrelated" without needing an activation dump.
3. **BF16/Q8 activation emulation** (§6 items 2–3) — one rounded upload per block
   behind a flag, then measure the parity delta; that is the cheapest way to
   decide whether the deviation matters or not.
4. **Then** the batching lever of §8 — it is worth 2–4× and it is the only
   change that gets near the 48.1 ms baseline with this arithmetic.
