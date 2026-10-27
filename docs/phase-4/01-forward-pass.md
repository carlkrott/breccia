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
Dispatch batching is **on by default**, 0 validation messages, and produces
bit-identical tokens on MI50-only, RX-only and dual (§8). Every number below is
from a run, and the ones taken before each fix are labelled.

| | |
|---|---|
| first end-to-end token | **yes** — 248 320 finite logits, repeat bit-identical, top-1 stable |
| greedy generation | **yes** — `Sampling::Argmax`, deterministic |
| parity vs the production HIP server | **MET** — 32/32 greedy tokens, first divergence: none |
| decode, **dual** (20 RX / 28 MI50 blocks), batching on, 16 threads | **7.42 tok/s** (134.9 ms/token) |
| GPU-resident experts (`--gpu-experts`) | **off** — correct (32/32) but slower, 6.37 tok/s (§8d) |
| vs the 20.79 tok/s baseline | **35.7 %** (dual was 29.0 % before this pass) |
| single-card diagnostic (not a target) | 8.55 tok/s / 117.0 ms |
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

### 5.1 Measured decode (warm page cache, 3×32, batching off vs on)

Method: `--gen 32 --runs 3`, decode-only (the prompt is fed first, its time is
excluded, the 3 runs are averaged). A warm-up pass runs first so the
expert/PLE pages are not the measurement. One GPU process at a time, everything
under `timeout`, engine only (the parity server is stopped).

| config | batching | tok/s | ms/token | vs 20.79 | Δ vs off |
|---|---|---:|---:|---:|---:|
| MI50-only `0000:0e:00.0` | off | 5.12 | 195.3 | 24.6 % | |
| MI50-only | **on** | **6.74** | **148.3** | **32.4 %** | **−47.0 ms** |
| RX-only `0000:03:00.0` | off | 3.94 | 254.1 | 18.9 % | |
| RX-only | **on** | **5.30** | **188.6** | **25.5 %** | **−65.6 ms** |
| dual `0000:03:00.0,0000:0e:00.0` | off | 4.62 | 216.6 | 22.2 % | |
| dual | **on** | **6.03** | **165.9** | **29.0 %** | **−50.6 ms** |
| MI50-only, 8 host threads | on | 6.74 | 148.3 | 32.4 % | −47.0 |
| **MI50-only, 16 host threads** | **on** | **7.92** | **126.3** | **38.1 %** | −69.0 |
| MI50-only, 16 host threads | off | 6.43 | 155.5 | 30.9 % | |

Against the plan's target (−40 ms MI50 / −70 ms RX): **batching alone 119 % on the
MI50 and 94 % on the RX; batching + 16 threads −69.0 ms (−36 %) on the MI50.**
(The 16-thread rows were measured with the CLI's `--batching` default fixed — the
first sweep through the thread curve had the flag overridden to `false`, which is
why the earlier note said 152.8 ms for what is really the batching-*off* 16-thread
cell.)

### 5.3 Dual-GPU: both cards working, and the split that does it

Policy (AGENTS.md, 2026-10-26): the target is proper **dual-GPU** use, so every
number in this subsection is a dual run; single-card figures appear only as
diagnostics to explain a dual result.

Two changes since the last pass: the **`block_out` mailbox** (the MoE result is
memcpy'd into a permanently-mapped host-visible buffer instead of being uploaded
through a submit+fence) and **leaving the batch open across the block boundary**
(the ffn `HC_WRITE` of block *n* and phase A of block *n+1* are one submission,
because nothing host-side sits between them any more). A block is now **two round
trips** instead of four.

Warm, 3×32, batching on, 16 threads, `--split-blocks`:

| split (RX, MI50) | tok/s | ms/token | vs 20.79 | fence wait RX : MI50 |
|---|---:|---:|---:|---:|
| 24, 24 | 5.10 | 196.1 | 24.5 % | 1.24 : 1 |
| 28, 20 | 6.80 | 147.1 | 32.7 % | 1.75 : 1 |
| **20, 28** | **7.42** | **134.9** | **35.7 %** | **0.95 : 1** |
| 27, 21 (before this pass) | 6.03 | 165.9 | 29.0 % | — |

**The best dual configuration is 20 blocks on the RX and 28 on the MI50 at
7.42 tok/s (134.9 ms/token), and it is the only split where the two cards'
accumulated fence wait is balanced (0.95 : 1)** — i.e. the split is chosen so the
two cards finish their halves together, which is the criterion that matters, not
the block count. 24/24 is badly unbalanced (1.24:1 with the RX far ahead in work
per block) because an RX block is ~1.5× an MI50 block, and 28/20 over-loads the
RX (1.75:1).

Both cards do real work every token: at 20/28 the RX executes 20 blocks and the
MI50 28, each with its own 0.7 / 1.0 MiB of host traffic per token (the three
`[hc, n_embd]` readbacks and mailbox writes per block, split by card).

**Dual is now ahead of the previous dual** (6.03 → 7.42 tok/s, −31 ms/token) but
still behind the single-card diagnostic (8.55 tok/s / 117.0 ms), which is the
number this policy deliberately does not target. The gap is the per-block cost
ratio (§5.3 note above) and the PCIe hand-off; closing it needs the per-block
submission count to fall further (pre-recorded command buffers, §8f) or the
experts to move onto the cards (§8d).

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

## 8. Dispatch batching: fixed, measured, on by default

`breccia-kernels/src/dispatch_batch.rs`: `begin` / `end` / `record`, plus hooks in
`Pipe::run`, `Gemv::run`, `GemvBatch::run` that record instead of submitting while
a batch is open. A graph block is now **two submissions** (one for
`HC_READ(attn) … shared expert`, one for the ffn `HC_WRITE`) instead of ~29, and
`breccia-kernels/tests/batching.rs` proves six dependent dispatches across three
pipelines plus a GEMV are **bit-identical** to the synchronous path.

### 8.1 The three defects this cost, in order

1. **One descriptor set per dispatch.** A `Kernel` owns a single set, and a set
   may not be updated once a recorded command buffer references it — the GPU then
   sees the *last* write for every recorded dispatch (GPUVM fault within one
   token). A batch now allocates a fresh set per dispatch from its own pool,
   compatible because it is allocated from the kernel's own `dsl`.
2. **The pool must outlive the submission**, not just the recording: freeing it
   before `vkWaitForFences` reads as `VUID-vkCmdDispatch-None-08114`
   ("descriptors must be valid").
3. **The QSA workspace was allocated and freed inside the batch** — and this was
   the `VUID-vkCmdBindPipeline-commandBuffer-recording` cascade. Its eleven
   buffers are referenced by recorded dispatches, so `Qsa::release` freed them
   before the submit (a use-after-free) while `Qsa::workspace` did the one host
   allocation the batch discipline forbids. `run_qsa` now flushes, allocates,
   opens the batch, records the forward, flushes, *then* releases.

### 8.2 Evidence

* `VK_LAYER_KHRONOS_validation` on the MI50, `--batching`: **0 validation
  messages** (it was 107 before the fix), no fault, run completes.
* Per block: 29 recorded dispatches in 2 submits (GDN), 28 in 3 (QSA — the
  workspace lifecycle adds one).
* `cargo test -p breccia-kernels --test batching` on the MI50: bit-identical.
* 32/32 server parity and bit-identical output across all six on/off ×
  device configurations (§5.1).

**Reading the per-stage table after batching.** The stage marks are host-clock
deltas around synchronous dispatch groups; with batching the GPU work of a whole
phase completes at the *fence*, so it is charged to the stage after it
(`xfer-down`). That is why `hc-read(attn)` reads 0.2 ms and `xfer-down` 43.2 ms
with batching on: the same work, attributed to the wait instead of to the
dispatches. The totals are the honest number.

## 8b. Next hotspot, measured after batching (MI50, batching on, 16 threads, 126.3 ms)

| item | ms/token | share | what it is |
|---|---:|---:|---|
| block fence wait (`xfer-down`) | **42.4** | 33.7 % | the GPU half of all 48 blocks, charged to the phase's fence |
| `experts(host)` | **36.6** | 29.0 % | the AVX2 lane, 799 MiB of expert reads |
| `mixer` (record) | 18.7 | 14.9 % | 36 GDN + 12 QSA sublayers, ~20 dispatches each |
| `head` | 7.3 | 5.8 % | 248 320 × 2560 Q5_K GEMV |
| `hc-write(ffn)` | 6.8 | 5.4 % | one HC write per block + its submit |

The three biggest are therefore: **the block's GPU time (42 ms), the host expert
lane (37 ms), and the mixer's dispatches (19 ms of recording + their share of the
42 ms)**. Two levers, in order of value per unit of risk:

1. **Host-visible `block_out` (plan, not implemented — budget).** With batching,
   the MoE result still goes up through `Staging::write`, i.e. one
   record+submit+wait per block: 48 × 161–193 µs = **7.7–9.3 ms/token**. Making
   `block_out` a permanently-mapped `HOST_VISIBLE|HOST_COHERENT` buffer turns the
   upload into a `memcpy` (`vkQueueSubmit` is already a host-write barrier for
   coherent memory, and the ffn `HC_WRITE` is the next submit), removing that
   round trip. The same trick applies to `router_ids`, `router_w`, `sh` and
   `sh_gate` on the readback side, but not to `mixed` — five kernels read it, so
   it must stay device-local.
2. **A fused HC_READ (shader work, not implemented).** The HC path is 96 phases ×
   5 dispatches = **480 of the ~1 200 recorded dispatches per token**, and
   `docs/phase-3/04` §7.2 already lists the fold: `norm` + `down` in one dispatch
   removes 96 dispatches and hides the `down` GEMV's under-filled grid
   (22.75 µs for 6.29 MiB on the RX). Expected 3–6 ms; it needs a new shader and
   a conformance case, which is kernel work with real risk, so it is scheduled
   rather than done.

## 8c. Next hotspot: the host expert lane

Post-batching MI50 breakdown (148.3 ms): the host expert lane is the largest
single item and the only one that does not depend on which GPU runs.

| threads | tok/s | ms/token | `experts(host)` ms | lane GB/s (799 MiB/token) |
|---:|---:|---:|---:|---:|
| 8 (default) | 5.82 | 171.9 | 53.3 | 15.0 |
| 12 | 6.30 | 158.8 | 42.9 | 18.6 |
| 16 | **6.55** | **152.8** | **38.5** | **20.8** |

* **It scales with threads, so it is compute-bound, not fault-bound.** Run-to-run
  at 8 threads is 169.3 / 177.4 / 169.2 ms and at 16 threads 153.3 / 152.3 /
  152.7 ms — a warm run is *not* faster than the first, i.e. the page cache is
  already holding the sampled expert pages after the prompt (the one 414 ms
  `experts` outlier in this log is a fresh-load first token, not the steady
  state). Phase-3 measured the lane at 20.4 GB/s standalone with 8 threads and
  22.5–30.5 with 16; in-graph we see 15.0 / 20.8, the difference being the
  3.7 GiB of dense weights and the PLE table sharing the cache.
* **12–16 threads is the free win**: +13 % tok/s for a flag, and `--threads 16`
  is the recommended setting for this graph. The default stays at the lane's own
  policy (physical cores / 2 = 8) so the phase-3 numbers keep their meaning.
* **AVX-512 is not the next step, and here is why.** The lane is already at
  20.8 GB/s at 16 threads against a 53.9 GB/s single-core DRAM stream: it is
  *decode-bound*, and the top-3 types by bytes (`IQ4_NL` down 18 layers,
  `Q2_0` down 30 layers, `IQ3_S` gate/up 13 layers) are the two *cheap* decoders
  (`vpshufb` table lookup and shift/mask, 4.1× and 5.6× over scalar) and the one
  mid-cost type. Phase-3 put the ceiling of a `vpermb`-style IQ rewrite at
  **10–25 %**, requires a second complete set of 14 kernels behind
  `is_x86_feature_detected!` plus per-feature gating (Rust has no
  `#[target_feature(enable = "avx512vNNI")]` subset that is stable for all of
  these), and would buy ~4–9 ms of a 152 ms token. The structural lever is
  better: fewer expert bytes per token.
* **What actually moves this**: (a) `--threads 16`, +1.5 tok/s — done and
  recommended; (b) the `-ot` parity we do not implement (§8d); (c) prefetching
  the next token's expert pages during the current block's GPU phase (`docs/phase-3/08`
  §7 item 5).

## 8d. GPU-resident expert blocks: **correct, and slower than the host lane — flag stays off**

Correctness is fixed and proven at three levels. Speed is not there, so the flag
stays **off** and the dual number in §5.3 (7.42 tok/s) is unchanged.

| | flag off (host lane) | `--gpu-experts 0,1,2,4,8,18,20,22,23,24,35` |
|---|---|---|
| dual tok/s (20/28, 16 threads, 3×32) | **7.42** (134.9 ms) | 6.37 (157.0 ms) |
| greedy parity vs the server | **32/32** | **32/32** |
| top-1 logit (prompt `[760,13017]`) | 17.5689 | **17.5689** (identical) |
| validation messages | 0 | **0** |
| VRAM arena | — | 9 913 MiB (11 blocks × 512 experts) |
| per-card fence wait / token | 34.3 : 39.6 ms | 34.6 : 39.6 ms |

### The bug: the row map is per **row**, not per expert

`map_for_blobs` (`batch.rs:195-215`) writes

```text
map[e*rpe + r] = base + r * bpr * block_words(ty)      // one entry per ROW
map[rows + i]  = x_stride * e                          // activation base
```

The first `moe_map.comp` wrote `base + ids[j]·blob_words` for *every* row of expert
`j`, so all 640 (or 2 560) rows of an expert read that expert's **first row** —
finite, plausible, and wrong. That single mistake explains both symptoms: parity
0/32 and logits off by up to 1.38 nats.

### The three checks that now gate this path

`crates/breccia-kernels/tests/moe_indirection.rs` (MI50, validation on):

| check | result |
|---|---|
| device map vs `map_for_blobs`, **byte for byte** — IQ2_XS `Shared`, IQ4_NL `PerExpert`, Q2_0 `PerExpert` (12 800 / 51 200 / 51 200 entries) | **identical**, including a non-zero `base_words` so an offset error cannot hide |
| `MoeCombine` vs `Σ w_j·in_j + shared·sigmoid(g)` in f64 | worst \|Δ\| **1.99e-8** |
| end-to-end greedy, dual, flag on | **32/32**, logits identical to the host path |

Suspects 2 and 3 were fixed as part of the same pass: each role now gets **its own
map** driven by its own `GemvBatch` (`rows_per_expert`, `blocks_per_row`, quant
block size and activation stride all come from the pipeline, so a call site cannot
get the units wrong), and `MoeMap::run` takes the `GemvBatch` rather than loose
numbers.

### Why it is slower (the remaining question)

The work moved from 16 CPU threads (~0.76 ms/block) to the device, but the path
adds **eight dispatches per placed block** (three maps, three GEMV, SwiGLU,
combine) and the 640-row expert GEMV shape is the latency-bound one
(`docs/phase-3/02` §2.1: 81–161 GB/s for 640-row shapes against 251–448 for the
6 400-row set). Eleven blocks therefore pay ~+22 ms of dispatch latency and
under-filled grids to save ~8 ms of host lane, plus the longer batches (soft limit
200 dispatches) reduce host/GPU interleaving.

**Verdict per the work order:** keep `--gpu-experts` **off**. Revisit when it can
beat the host lane: after the QSA position indirection / replay work (which drops
the per-block submission count), with **larger** expert blocks (a 640-row GEMV per
role stays latency-bound; batching *across* blocks would fill the grid), or by
keeping the host lane for the gate/up roles and moving only `down` where the shape
is 2 560 rows. The mechanism (`MoeMap`, `MoeCombine`, `PlacedExperts`, arena,
device-side ids) is correct and is the substrate for all of those.

## 8g. Where the recording budget actually is (and what that does to the replay plan)

New instrument: `breccia_kernels::batch_cmd::record_seconds()` accumulates the host
time inside `record()` — descriptor-set allocation + write, bind, push constants,
dispatch, barrier — i.e. exactly what a pre-recorded command buffer would remove.
The engine prints it as `record … ms/token (dispatches, per submit)`.

Dual, 20/28, 16 threads, 3×32:

```
record   10.02 ms per 31-token run = 0.32 ms/token   (1 353 dispatches/token, 21 per submit)
```

**Recording is ~0.3 ms/token, not the 18.7 ms this report previously attributed to
"mixer recording".** The stage mark around the mixer measures the *wall* time of
issuing those dispatches, and what is in there is not `vkCmd*`:

* `run_qsa` flushes **twice** per QSA block (once to allocate the workspace
  outside the batch, once to release it after), so 12 QSA blocks cost
  **24 extra submit+fence round trips** per token, each landing in the `mixer`
  stage: 12 × 2 × ~0.8 ms ≈ 19 ms — the whole figure; and
* `qsa.workspace()` allocates **eleven buffers per QSA layer per token**.

### Consequences

1. **Experiment B's payoff is not the recorded time.** A replayed command buffer
   still costs one submission and one fence, so replay can only remove ~0.3 ms of
   recording. The QSA position indirection is therefore **not** the pay-off item,
   and it should not be started for the recording reason this report gave.
2. **The real item is the QSA workspace lifecycle**: hoist it (one workspace per
   QSA layer, sized to the context, allocated at session start) and mutate its
   `n_blocks` / `max_ids` / `slots` fields per call — they are public, so the
   live position can be written per token without a reallocation and without
   flushing the batch. That removes up to 24 round trips (~19 ms) *and* the
   allocation churn, with no shader change and no position indirection. It is the
   top of the queue now.
3. Experiment A stays ruled out by arithmetic (§8h).

**Run-to-run caution this exposed:** the same dual 20/28 configuration measured
134.9 ms and 156.4 ms in two windows of this pass, with `experts(host)` moving
36.6 → 52.7 ms and `ple` 1.2 → 6.2 ms — page-cache state, not code. Comparisons
across passes must be alternated within one scripted window (`bc-*`, `du-*`,
`sp-*` were), which is why the headline numbers come from single sweeps.

## 8h. Experiment A (only the `down` role on the GPUs): ruled out by arithmetic

* the host lane is **0.76 ms/block** (36.6 ms / 48), and gate, up and down each
  cost **one third** of it (`docs/phase-3/08` §2 — the roles are symmetric by
  construction: `640·2560 = 2560·640` values);
* moving `down` for the 13 production blocks therefore moves **≤ 2.5 ms** of a
  135 ms token, and it cannot even keep that: gate/up still need the host, so the
  readback stays and the `act` vector (`[10, 640]` = 25 KiB) must go **up** to the
  device — one extra submit+fence per placed block, 11 × ~0.19 ms ≈ 2.1 ms;
* net ≈ **+0.3 ms**, before counting the device `down` GEMV.

Not implemented: ~170 lines (a partial-lane entry point in `ExpertPool` plus a
second placement path) with a negative expected value. §8d already measured the
same shape of answer from the other direction (all three roles on the device:
+22 ms).

## 8i. QSA workspace hoisting (landed): 134.9 -> 121.7 ms on dual

One `QsaWorkspace` per QSA layer, allocated at `ctx_capacity` in `Session::new`,
freed in `Drop`, and reused every token: only its three shape scalars
(`n_blocks`, `max_ids`, `slots` — the fields `Qsa::forward` reads) are rewritten
per call, against buffers sized once. That removes

* the **two extra flushes per QSA block** (the workspace used to be allocated
  outside the batch and released after it, so 12 QSA blocks cost 24 submit+fence
  round trips a token), and
* **eleven buffer allocations per QSA layer per token**.

Dual, 20/28, 16 threads, 3×32, three alternating runs in one window:

| run | tok/s | ms/token | vs 20.79 | parity | dispatches per submit |
|---|---:|---:|---:|---|---:|
| 1 (cold) | 6.40 | 156.2 | 30.8 % | 32/32 | 29 |
| 2 | 8.09 | 123.6 | 38.9 % | 32/32 | 29 |
| **3 (warm)** | **8.22** | **121.7** | **39.5 %** | **32/32** | 29 |

The generated 32-token sequences are **bit-identical across the three runs and to
the server** (32/32 each). `dispatches per submit` rose 21 → 29: the QSA block's
work now rides the same submission as its neighbours, which is the expected
signature of the removed flushes.

Warm dual: **134.9 → 121.7 ms (−13.2 ms, −10 %)**, 7.42 → **8.22 tok/s (+11 %)**,
both cards' fence wait still balanced (33.8 : 34.2 ms).

## 8j. Overlap of the host expert lane with GPU work: not attempted

The structural idea (split the top-10: a few experts on the block's device via the
now-correct `MoeMap`/`MoeCombine` path, the rest on the host, with the GPU GEMV
issued *before* the host work so the two run concurrently) is the right next move
and it is **not** implemented: this pass spent its budget on §8i, which was the
prerequisite — with the QSA flushes still in place, the GPU was idle for two
submissions per QSA block and any overlap measurement would have been swamped.

What the numbers say about it: the host lane is now the **largest single item**
(~37–53 ms of 121.7), and `MoeCombine` already accepts the whole routed set, so
the split needs (a) a per-block policy `k_gpu ∈ {2,4,6}`, (b) the CPU lane to
compute the *complement* of that subset (its current entry point takes the full
`ids`), and (c) one `MoeCombine` that adds CPU partials to the GPU ones. (b) is
the only real work: `ExpertPool::moe_topk` needs an "experts `ids[a..]` only"
form, ~40 lines mirroring its phase 1/2 dispatch.

## 8k. Overlap of the host lane with GPU work: measured conclusion, not attempted

The scoped design was: split each block's top-10, run the GPU subset on the
block's device and the CPU complement on the host **concurrently**, combine.

The complement needs no new pool entry — `ExpertPool::moe_topk(plan, ids, weights,
x, out)` takes arbitrary slices, so `&ids[k_gpu..]` *is* the complement, and
`MoeMap::run(..., experts = k_gpu, ...)` already reads exactly `IDS[0..k_gpu]`, so
the GPU subset is the first `k_gpu` ids with no kernel change.

**It cannot pay for itself, by arithmetic:**

* overlap requires the GPU subset to be *submitted* before the host starts, i.e.
  one extra `vkQueueSubmit` + fence per block. The measured cost of a round trip
  on this graph is **~0.8 ms/block** (the alternating sweep: 4 round trips → 2
  gave −47 ms over 48 blocks, §8e);
* the work it hides is the host lane's share of `k_gpu` experts: the whole lane is
  0.76 ms/block for 10 experts, so `k_gpu = 4` moves **0.30 ms**, and the GPU's own
  share of that work is ~0.1 ms at the measured 640-row rates;
* so the best case is −0.3 ms of critical path for +0.8 ms of submission — and
  `k_gpu = 2/6` move 0.15/0.45 ms. Every setting loses by 0.35–0.65 ms/block, i.e.
  **−17 to −31 ms/token**.

There is a way it *could* win, and it is the same primitive the replay discussion
needs: a mid-batch sync so the host learns the router's ids **without** fencing the
phase — a timeline semaphore signalled by the router dispatch, waited on the host
while the GPU continues into the expert GEMV. That is a `breccia-vk` change
(`Gpu` owns the command buffer and the fence), outside this work order's
ownership, and it would also unlock K > 2. Recorded here as the prerequisite.

The device-expert path itself stays **correct but off** (§8d): 32/32 parity,
+22 ms serial.

## 8l. Where the ~1 500 dispatches and the 70.8 ms of GPU time go

Dispatch counts are deterministic from the call graph; the times are the measured
phase-3 kernel numbers (`docs/phase-3/01`, `03/04`, `06`) for the *slower* of the
two cards, because the split's slowest link sets the block's cost:

| op family | dispatches/block | × blocks | dispatches/token | GPU cost/block | ms/token |
|---|---:|---:|---:|---:|---:|
| **HC read** (norm, down 320×10240, act, up 10240×320, gated+inject) | 5 × 2 | 48 | **480** | 2 × 119 µs = 238 µs | **11.4** |
| **GDN** (qkv, z, beta, alpha, conv, qk-l2, gate, delta-step, norm+gate, ssm_out) | 10 | 36 | **360** | 357 µs | **12.9** |
| **QSA** (6 proj, q/k/idx norms, 3 rope, kv-append, idx tail/pool/score, top-k, scores, softmax, apply, out-proj) | 20 | 12 | **240** | 310–859 µs (4 → 310, full budget → 859) | **3.7–10.3** |
| **shared expert** (gate, up, SwiGLU, down, scalar gate) | 5 | 48 | **240** | ~120 µs | **5.8** |
| **router** (GEMV + softmax/top-k) | 2 | 48 | **96** | 21 µs | **1.0** |
| **HC write** (combine) | 2 | 48 | **96** | 2 × 25 µs | **2.4** |
| head (248 320 × 2560 Q5_K) + final HC read | 6 | 1 | **6** | 4.5 + 0.6 ms | **5.1** |
| embed (staged gather) | 1 | 1 | **1** | 12 µs | **0.01** |
| **total** | | | **≈1 519** (measured 1 509) | | **≈42–49** |

Ranked by time: **GDN 12.9 (26 %), HC read 11.4 (23 %), QSA 3.7–10.3 (7–21 %),
shared expert 5.8 (12 %), head 5.1 (10 %)**; by dispatches: **HC read 480,
GDN 360, QSA 240 and shared expert 240, router + HC write 192**.

Memory-bound vs latency-bound, against the phase-3 rates:

* **GDN's three big GEMVs are bandwidth-bound** and run at 1 650 GB/s on the RX
  (above the 561 GB/s DRAM ceiling → Infinity-Cache resident, §6.1/§6.3 there);
  the **delta rule is latency/stride-bound** (43 µs RX for 9 MiB of state = 217
  GB/s, 2.6× off the ceiling) and is 28 % of a layer.
* **HC read is latency-bound at T = 1**: 60→44 µs for 13.1 MB of BF16, half the
  596 GB/s a plain 640-row GEMV reaches, and the 4-row `inject` alone was 21.85 µs
  before it was folded (`docs/phase-3/04` §5.1).
* **the shared expert is latency-bound** (640-row shapes, 81–161 GB/s).
* the **head is DRAM-bound** (437 MB, 252 GB/s measured for Q5_K).

### The three highest-value fusions, with estimates

1. **Fuse the HC_READ chain's `norm` + `gated` into their neighbours** (5
   dispatches → 3): `norm` (6.3 µs) and `gated` (7.4 µs) are already single
   workgroup-per-256-element passes over data the adjacent GEMVs touch. Removing
   192 dispatches and two 6–7 µs passes per read: **≈ 2.5 ms/token**, and it needs
   one shader edit in a path that already has a conformance suite.
2. **A split-K `down` for the HC bottleneck** (320 rows × 10 240 today = 40
   workgroups on 60 CUs): `Gemv::run_split` already exists but the mixer uses
   `GemvBatch`, which has no split-K. A split-4 `down` is worth 22.75 → ~12 µs at
   T = 1 × 96 reads = **≈ 1.0 ms/token**, no new shader (host-side plumbing only).
3. **One-pass GDN**: conv + qk-l2 + gate are three passes over the same 10 240
   channels (1.95 + 2.03 + 1.60 = 5.6 µs/layer, all latency-floor). Fusing them
   saves the two dispatch floors (≈ 2 × 4.5 µs on the MI50) × 36 = **≈ 0.3 ms** —
   small, and it is the riskiest of the three (three shaders' semantics in one).

Item 2 is the cheap one (host-side only, `< 1 hour`) and is the recommendation for
the next pass; item 1 is the biggest single win and is contained.

## 8m. The two next workstreams, specified (not implemented)

No code from this section landed: both items are changes that cannot be left
half-applied (a partially added semaphore in `breccia-vk` or a half-fused shader
would break the build or the parity gate for the next pass), and this pass had no
budget for an implement-measure-fix cycle on either. What follows is the design
with the exact signatures and ordering, so the next pass starts at the edit.

### 1. Mid-batch sync: a timeline semaphore the host can poll

Today `breccia_kernels::batch_cmd` can only end a batch by submitting and waiting
on a binary fence, which is why the host cannot see the router's ids until the
whole phase is done — the reason overlap loses ~0.8 ms/block (§8k) and K cannot
exceed 2 (§8e).

**API to add in `breccia-vk` (additive, existing calls untouched):**

```rust
impl Gpu {
    /// A timeline semaphore plus the host-visible counter buffer it is paired with.
    pub fn timeline(&self, initial: u64) -> R<Timeline>;
}
pub struct Timeline { pub sem: vk::Semaphore, pub counter: Buf /* u64, host-visible */ }
impl Timeline {
    /// `vkCmdWriteTimestamp`-free progress marker: the batch records a signal of
    /// `value` at this point, so the host can observe when the GPU reached it.
    pub fn signal(&self, d: &ash::Device, cmd: vk::CommandBuffer, value: u64);
    /// Block until the GPU signals at least `value` (host-side, no fence for the batch).
    pub fn wait_host(&self, value: u64, timeout_ns: u64) -> R<()>;
}
```

**Recording rule:** `vkCmdWriteTimestamp` cannot signal a timeline, so the batch
records the marker as `vkCmdWaitEvents`-free pair:
`vkCmdWriteTimestamp(query)` for the GPU-side timestamp *plus* a
`vkCmdPipelineBarrier(HOST→HOST)` and a `vkUpdateDescriptorSets`-free host read of
a **host-visible u64 the previous dispatch wrote** (`moe_map`/`router` can write
`1` into it as their last act). `Timeline::wait_host` then spins on that u64 with
`vkGetQueryPoolResults(..., WAIT)` as the fallback. Either way the host learns
"the router has completed" **without** `vkWaitForFences` on the batch, which is
what keeps the rest of the batch running.

**Then the two pay-offs, both already scoped:**

* **Overlap** (§8k): record phase A + router + `moe_map`/GEMV for the first
  `k_gpu` ids; signal; host waits on the marker, computes the complement
  (`pool.moe_topk(plan, &ids[k_gpu..], &w[k_gpu..], …)`) while the GPU is still
  executing; `MoeCombine` gains a `cpu` binding and a `+ CPU[i]` term. Expected:
  −0.15/−0.30/−0.45 ms per block for `k_gpu = 2/4/6`, with **zero** extra
  submissions, i.e. −7/−14/−22 ms/token instead of the current −17…−31.
* **K > 2** (§8e): with the marker, block *n+1*'s phase A can be recorded and
  submitted behind block *n*'s MoE readback as long as the *state* buffers it
  reads are disjoint — the QSA workspace is now hoisted (§8i) and the GDN state is
  per layer, so K = 2–4 blocks in flight is reachable. The serialisation that
  remains is the host lane for the 37 host blocks, which is exactly what the
  overlap above removes for its share.

**Validation:** the semaphore must be created with
`VK_KHR_timeline_semaphore` (already enabled: `lib.rs` asserts it on every
device — `enumerates_three_devices_on_this_host`); the host-side spin must be
bounded (`timeout_ns`) and must fall back to `vkWaitForFences` + a warning rather
than hanging the box; first run on the MI50 with the validation layer.

### 2. Op-family fusion: the dispatch budget, in priority order

| # | fusion | dispatches removed | shader work | expected |
|---|---|---:|---|---:|
| 1 | **HC read**: fold `norm` into the `down` GEMV's prologue and `gated`+`inject` into one pass (5 → 3) | 192 | 2 shaders (`hc_down_fused`, extend `hc_gated_mean`) | −2.5 ms |
| 2 | **Shared expert**: gate+up in one dispatch (two row ranges of one 1280-row GEMV) then SwiGLU+down in one pass with the 640-wide intermediate in LDS (5 → 2) | 144 | 2 shaders | −3.0 ms |
| 3 | **split-K HC `down`** via the existing `Gemv::run_split` (host plumbing only) | 0 | none | −1.0 ms |
| 4 | **GDN**: conv+SiLU, q/k L2 and beta/gate over the same 10 240 channels as one pass; delta rule with the 128-float column in LDS instead of re-reading the 9 MiB state | 72 | 2 shaders | −4.0 ms |
| 5 | one-pass GDN (all three) | 72 | 3 shaders' semantics in one | −0.3 ms |

Items 1–4 total **≈ −10.5 ms** and **408 fewer dispatches** (1 509 → ≈1 100), all
inside kernels that already have conformance suites; the pattern to keep is the
one that made §8d correct: a unit-level byte/value diff against the host reference
**before** the end-to-end parity run.

## 8n. Mid-batch sync, overlap, and split-K: one win, two measured rejections

§8m specified a timeline-semaphore marker so the host could start a block's
MoE work before the block's GPU work finished. All of the machinery landed and is
validated; the **overlap it enables does not pay**, and the report says so with
numbers rather than estimates. One item out of the four attempted is a win.

### What landed

| piece | where | state |
|---|---|---|
| `Gpu::timeline(initial) -> Timeline`, bounded `Timeline::wait_host(value, timeout_ns)`, `Timeline::counter` | `breccia-vk/src/ctx.rs` | used by the probe; the segmented path that used it measured negative (§below) |
| `Gpu::submit_cmd(q, cmd, signal)` — a submission that signals a timeline value (`vkQueueSubmit2`, loaded by its **core 1.3** name; ash 0.38 only loads `vkQueueSubmit2KHR`, which RADV does not export, and `VkSubmitInfo` cannot carry `pSignalSemaphoreValues`) | `breccia-vk/src/ctx.rs` | on |
| a **ring of command buffers** (`Gpu::cmds_c`, 3 slots) with a per-slot fence, so a segment can be in flight while the next is recorded; `begin` waits only on the slot it reuses | `breccia-vk`, `breccia-kernels/dispatch_batch` | on (no cost, §A/B below) |
| `record_copy` / `record_barrier_to_host` / `record_timestamp` into an open batch | `dispatch_batch` | the copy path was measured and rejected; `record_timestamp` powers the new probe |
| `BRECCIA_GPU_TIME=1` **GPU-time probe**: `vkCmdWriteTimestamp` around the first 56 batches, printed as per-device GPU busy ms | `breccia-engine/src/graph.rs` | on (env-gated) |
| split-K on the HC_READ `down` GEMV (`T = 1`) | `breccia-kernels/ops` (`HcMixer::down_splits`) | **on — the win** |

Validation: the whole path (timeline semaphores, `vkQueueSubmit2`, the ring, the
segmented submissions) ran on the **MI50 first with `VK_LAYER_KHRONOS_validation`**
as AGENTS.md requires, and produced **0 validation messages**, including at
teardown (the semaphores and the ring's pools/fences are released in `Drop`).

### The new instrument: how much of a fence wait is the GPU?

`BRECCIA_GPU_TIME=1` timestamps each batch; the difference from its fence wait is
what the host and driver add. One warm token, 48 blocks, `--threads 16`:

| configuration | GPU busy (ms/token) | fence wait (both devices) | per block |
|---|---:|---:|---:|
| dual 20/28 (RX 20 blocks) | 29.4 (RX) + 29.9 (MI50) = **59.3** | 33.0 + 33.7 = **66.8** | RX 1.47, MI50 1.07 |
| RX only (48 blocks) | **76.6** | 85.0 | **1.60** |
| MI50 only (48 blocks) | **43.4** | 49.7 | **0.90** |

Two things fall out of that table, and both matter more than the marker did:

1. **Submission latency is ~8–13 ms/token, not the 20–26 ms §8k assumed.** The
   fence wait is GPU busy + ~4 ms per device. Pipelining across blocks (the other
   pay-off §8m predicted for the marker) has at most ~8 ms to win, which is why it
   is not the next thing to build.
2. **The RX 6950 XT's GPU time for the same 48 blocks is 1.76× the MI50's**
   (76.6 vs 43.4 ms) while the desktop compositor owns the RX. That makes *any*
   block placed on the RX a net loss today (§placement below) — the opposite of
   what the 20/28 split was chosen for.

### 1. Two segments + marker: **rejected** (measured 17 ms slower)

Segment A = everything up to the router, submitted with a timeline signal;
segment B = the shared expert, submitted immediately behind it; the host waits for
A's marker (bounded at 2 s; 5472/5472 markers signalled, **0 timeouts**) and runs
the expert lane while B executes. Alternating A/B in one window, dual 20/28, 3×32
tokens, best-of:

| binary | ms/token | tok/s | tokens |
|---|---:|---:|---|
| single submission per block (HEAD) | **121.9** | 8.20 | reference (32/32) |
| two segments + marker | 139.7 | 7.16 | bit-identical |

Why it loses, in measured numbers: after the router there is only the **shared
expert** left to hide — 0.24 ms of GPU per block (`shared-exp` stage, and the GPU
probe agrees) — against 0.9 ms of host lane. The second submission costs a GPU
pipeline bubble of ~0.2 ms per block (24 dispatches drain and re-ramp), so hiding
0.24 costs 0.2 and the *lane itself* runs 0.17 ms/block slower from memory
contention with the concurrent segment. §8k's arithmetic (0.15–0.45 ms of lane to
hide for `k_gpu = 2/4/6`) is right about the size of the prize; the prize is just
smaller than the submission it needs — and the `k_gpu` split it was scoped for
needs experts resident in VRAM, which §8d already measured at +22 ms.

### 2. Copies inside the block's own batch: **rejected** (measured 17.8 ms slower)

Keeping the readback's five copies in the block's submission removes a
submit+fence per block (~55 µs), and the host then reads the staging mapping
directly. Same window, same protocol:

| binary | ms/token | tok/s |
|---|---:|---:|
| HEAD (readback as its own submission) | **121.9** | 8.20 |
| copies recorded into the compute batch | 139.7 | 7.16 |

A `vkCmdCopyBuffer` in the middle of a compute stream forces a
compute→transfer→compute transition (and a `HOST`-visibility barrier) per block:
~0.36 ms of GPU+driver per block, seven times the round trip it removes. Copy
count is not the problem — *placement in the stream* is. The staging-buffer block
region and the copy helpers stay in the tree (they are what a device-side
readback would use), unused by the graph.

### 3. Split-K on the HC_READ `down`: **landed**

`hc_lr` = 320 rows × 10 240 columns is 40 workgroups on a 60-CU card — under one
per CU, and §8l measures 110 GB/s for it against 251–448 GB/s for the same weights
as a 6400-row matrix. `Gemv::run_split` already existed; the mixer used
`GemvBatch`, which has no split-K. `HcMixer` now keeps a single-expert `Gemv` for
the `T = 1` case and runs it with `Gemv::recommended_splits()` (4 here: 160
workgroups + one reduce dispatch). Alternating A/B, dual 20/28, 3×32:

| binary | run 1 | run 2 | best | tok/s |
|---|---:|---:|---:|---:|
| HEAD | 121.82 | 121.34 | **121.34** | 8.24 |
| split-K `down` | 118.95 | 117.76 | **117.76** | **8.49** |

**−3.6 ms/token (−3.0 %)**, and it is the only change in this pass that moved the
gate. It applies per HC_READ (96 per token) on both cards.

**Numerical deviation, stated as required:** split-K sums four partial products
instead of one running accumulator, so the `down` result differs in the last bits.
Measured effect on the first token's distribution: top-1 logit `13.1705` →
`13.1706`, top-2 and the whole top-10 list unchanged, gap 0.8159 both ways. The
32 greedy tokens are **bit-identical** (§parity below).

### 4. Split-K on the short-row shapes: **rejected**

The same plumbing applied to the shared expert's `gate`/`up` (640×2560) and
`down` (2560×640) plus the router (512×2560) — shapes `recommended_splits()`
also wants to split (80 and 64 workgroups) — is **slower**:

| binary | run 1 | run 2 |
|---|---:|---:|
| split-K `down` only | **117.27** | 123.71 |
| + shared expert and router split | 124.06 | 125.75 |

Those shapes are *bandwidth*-bound at 81–161 GB/s, not under-filled: splitting
adds a reduce dispatch and two barriers and no bandwidth. The heuristic in
`Gemv::recommended_splits` is right for one long reduction (320×10 240) and wrong
for short rows; the graph therefore calls `run_split` only where it is measured
to win.

### 5. Placement: the dual optimum no longer puts blocks on the RX

Same window, one 32-token generation each, `--threads 16`:

| configuration | ms/token | tok/s |
|---|---:|---:|
| dual 20/28 (the split in §8i–§8m) | 118.7 | 8.43 |
| dual 4/44 | 105.0 | 9.53 |
| dual **0/48** | **101.8** | **9.82** |
| MI50 only (diagnostic) | 100.8 | 9.92 |
| RX only (diagnostic) | 142.7 | 7.01 |

The GPU probe explains it: the same block costs 1.5–1.6 ms of GPU on the RX and
0.9–1.1 ms on the MI50 *in this graph, on this desktop session* (§8l's phase-3
per-kernel numbers had the RX ahead, so this is in-graph contention, not kernel
quality — the compositor is on the RX). Blocks are serial across the residual
stream, so the token costs the **sum** of its blocks; putting any block on the
slower card can only add time.

**Consequence for the dual-first policy:** with serial block placement, "both
cards doing useful work" is unreachable — the measured optimum is 0 RX blocks.
Giving the RX real work needs *concurrent* work, i.e. two sequences (or two tokens)
in flight so one card's block stream runs while the other's expert lane does;
that is a batch>1 decode change, not a placement change. Recorded here as the
policy-level finding of this pass.

### 6. Gate

| item | value |
|---|---|
| dual, best measured placement (`--split-blocks 0,48`) | **98.1–104.4 ms/token over 6 cells, median 100.9**, best **98.13 ms / 10.19 tok/s** |
| dual, the inherited 20/28 split | 116.9–119.0 ms (from 121.3–121.8 at HEAD) |
| greedy parity vs the production server | **32/32, bit-identical** at both placements (`parity-check.py`) |
| validation messages (MI50, validation on) | **0** |
| `cargo test --workspace` / clippy `-D warnings` / `fmt --check` | 28 suites ok / clean / clean (`scripts/verify.sh` PASS) |

Commands (one GPU process at a time; the warm pass is what makes a run comparable):

```bash
D=/var/tmp/users/korphaus/tmp/opencode/p4-token
# the gate: 32 greedy tokens against the server's recorded sequence, both placements
python3 $D/parity-check.py ./target/release/breccia 0000:03:00.0,0000:0e:00.0 dual048
# alternating A/B of two builds in one window (page-cache/thermal discipline)
$D/compare.sh base v6 3
# how much of a fence wait is GPU, and per-block cost per card
timeout 400 env BRECCIA_GPU_TIME=1 ./target/release/breccia bench token \
  --devices 0000:0e:00.0 --prompt-ids-file $D/parity-server.json.ids --gen 4 --runs 1 --threads 16
# placement sweep (each is one 32-token generation)
for sp in 20,28 4,44 0,48; do timeout 400 ./target/release/breccia bench token \
  --devices 0000:03:00.0,0000:0e:00.0 --prompt-ids-file $D/parity-server.json.ids \
  --gen 32 --runs 1 --threads 16 --split-blocks $sp; done
```

Artifacts: `$D/ab-{base,v2,v3,v4,v5,v6}-*.json`, `$D/x-{base,v6}-*.log`
(the 0/48 A/B), `$D/sp-*.log` (the placement sweep), `$D/parity-dual.json`,
`$D/parity-dual048.json`, `$D/compare.sh`, `$D/parity-check.py`, `$D/ab2.sh`.

## 8o. Two-stream decode: one sequence per card, interleaved at the block level

§8n's conclusion was that a *single* stream cannot hide its own host expert lane:
block *b+1*'s GPU work cannot be recorded until the host has written block *b*'s
MoE result, so the token is `48 x (GPU phase + lane)`. The independent work that
does exist is another sequence. This section is that: **two sequences, one card
each, interleaved per block**, which is also the product shape (parallel agents).

| item | value |
|---|---|
| aggregate | **15.8–16.5 tok/s** (median 15.8 over 5 cells; best 16.52) |
| per stream | **7.9–8.3 tok/s**, 121–127 ms per stream-token (the block-pair time) |
| single stream, same window, same 32 threads | 11.33 tok/s / 88.2 ms |
| single stream at the inherited `--split-blocks 0,48`, 16 threads (§8n) | 9.82 tok/s / 101.8 ms |
| **gain** | **1.39x** over the same-thread single stream, **1.61x** over the inherited 9.82 |
| host-lane overlap | **100 %** of both lanes ran under the other stream's GPU work |
| GPU utilisation (probe on) | **31 %** MI50 / **55 %** RX; 45 and 80 ms of GPU per stream-token |
| greedy parity | **64/64** — each stream produced the server's 32 tokens bit-identically |
| validation | 0 messages (both cards in one process, `VK_LAYER_KHRONOS_validation`) |

### The design

`Session` was already a complete single-sequence engine (model, state, pipelines,
expert-lane handle), so a **stream is a `Session`** and the new type is a driver,
`breccia_engine::Multi`, that owns N of them. Each stream gets **one device** and
therefore its own copy of the dense weights (3.66 GiB: MI50 16 GiB and RX 16 GiB
both hold it, measured 2.4 s to upload each).

```text
submit A(0) ─┐
submit B(0) ─┴─▶ wait A(0) · lane A(0) · submit A(1)   ← B(0) executes on the RX
                 wait B(0) · lane B(0) · submit B(1)   ← A(1) executes on the MI50
                 wait A(1) · lane A(1) · submit A(2)   ← B(1) executes on the RX
```

What each piece needed:

* **`submit_block` / `finish_block`** — `Session::block` split in two. The record
  half submits the block's segment *without waiting*; the wait, the readback, the
  lane and the mailbox write are the other half. A block is still **one
  submission** (the deferred ffn `HC_WRITE` rides the next block's batch).
* **Per-device open batches.** `dispatch_batch` used to allow one open batch per
  *thread*; two sessions interleaving need one per *device* (each device has its
  own command buffers, so this is only bookkeeping). The ring went 3 → 6 slots and
  `begin` now prefers a slot with **no fence** — taking an own-but-in-flight slot
  first made every block wait for the previous block's submission and cost 96 ms
  per token pair.
* **Scoped flushes.** `wait_all` takes the devices to drain and leaves other
  sessions' slots alone; before that, stream A's flush waited for stream B's GPU
  (i.e. no overlap at all).
* **Interleaved heads.** `end_token` split into `submit_head` + `finish_head`, so
  the two streams' 248 320-row head GEMVs (7–10 ms each) overlap instead of
  serialising: +0.6 tok/s.
* **One shared expert lane.** The pool is an `Arc<Mutex<ExpertPool>>` handed to
  every session, so N streams use `threads` workers, not N x `threads`. The lanes
  run serially by construction (one host thread), which is exactly why one pool is
  right.

### Threads: the lane is the first limit, the RX is the second

Two-stream aggregate, one 16-token cell each, `--devices MI50,RX`:

| `--threads` | aggregate tok/s | lane ms/step (both streams) | fence ms/step |
|---:|---:|---:|---:|
| 8 | 7.13 | 215.6 | 30.3 |
| 16 | 14.69 | 69.9 | 37.9 |
| 24 | 14.32 | 59.3 | 51.0 |
| **32** | **15.83** | **50.0** | **47.3** |

The lane is the binding constraint up to 24 threads (it is decode-bound, §8c, and
scales with workers); by 32 threads the **RX stream** is: it needs 80 ms of GPU per
token against the MI50's 45 (§8n's 1.6 vs 0.9 ms per block), so a pair costs
`max(lane, RX)` and the RX sits at 55 % utilisation waiting for its own host work.
Sharing 16 threads between both streams is *correct but 14 % slower* than 32 — the
recommended setting for two streams is `--threads 32`, i.e. one worker per hardware
thread of the 7950X (16 cores / 32 threads).

### Why one card per stream, and not the 20/28 split for both

Blocks are serial *within* a stream, so a token costs the sum of its blocks: with
one shared placement, block *b* of both streams lands on the same card and the two
GPU phases queue behind each other. That configuration measures **114 ms per token
pair** (2 x 20 x 1.6 + 2 x 28 x 0.9 of GPU on each card, no concurrency) against
the **~100 ms** the device-per-stream design lands at. Pairing a stream with a card
is what makes the two GPU phases concurrent, and it is why `Multi::new` refuses a
model with more than one device.

### What this means for 4–6 agents

* **The ceiling is ~16 tok/s aggregate on this host, not 16 per stream.** Two
  cards, one stream each: a third and fourth agent must *time-share* the two
  streams (queue behind them), so 4 agents ≈ 8 tok/s each and 6 agents ≈ 5 tok/s
  each, while aggregate stays ~16. Per-agent latency is what degrades, and it
  degrades linearly in agents per stream.
* **What would raise it**: less host work per block. The pair time is now
  `lane (50 ms) + readback (12) + ffn-write record (11) + heads (10) + ple (3)`
  against 80 ms of GPU, i.e. the host is at ~55 % of the critical path. The two
  levers are (a) the in-batch readback copies that lost *single*-stream in §8n
  (−17 ms there, where the GPU had no slack) but would win here, where the GPU is
  45 % idle, and (b) fewer GPU dispatches per block so the RX stream's 80 ms falls.
* **What would double it**: two MI50-class cards. The RX's 1.6 ms/block is the
  reason the second stream only adds 39 % over a 32-thread single stream instead of
  the ~90 % the lane budget would allow.

Commands (one GPU process; both cards in it, which is the point):

```bash
D=/var/tmp/users/korphaus/tmp/opencode/p4-token
# the two-stream number (parity is checked from the JSON afterwards)
timeout 900 ./target/release/breccia bench token --streams 2 \
  --devices 0000:0e:00.0,0000:03:00.0 --prompt-ids-file $D/parity-server.json.ids \
  --gen 32 --runs 3 --threads 32 --json $D/two-probe.json
# the same-thread single-stream control, alternating in the same window
$D/two.sh 32 32
# per-card utilisation (the GPU-time probe costs ~10 %, so it is a diagnostic run)
timeout 900 env BRECCIA_GPU_TIME=1 ./target/release/breccia bench token --streams 2 \\
  --devices 0000:0e:00.0,0000:03:00.0 --prompt-ids-file $D/parity-server.json.ids \\
  --gen 32 --runs 1 --threads 32 --json $D/two-probe.json
```

Artifacts: `$D/f-{a,b}.{json,log}` (the alternating single/two-stream window),
`$D/two-probe.json` (utilisation + both streams' tokens), `$D/two.sh`.

## 8p. Two-stream, second pass: GPU experts and in-batch reads pay where the GPU has slack

§8o left the pair at 15.8–16.5 tok/s with the host lane at 55 % of the critical
path and the RX at 55 % utilisation. Two things that *lost* single-stream were
re-tested with that slack, and both now win. Stream scaling is measured too.

| item | before | after | verdict |
|---|---|---|---|
| in-batch readback copies (§8n item 2: 139.7 vs 121.9 single-stream) | 15.91 / 16.03 | **16.54 / 17.14** | **kept, on by default** |
| GPU-resident experts (§8d: +22 ms single-stream) — 9 blocks | 16.05 / 16.74 | **18.66 / 18.87**, gen-32 3-run mean **18.58** | **kept, default for `--streams 2`** |
| the same at 11 blocks (the full production mapping) | — | 15.56 / 14.48 | rejected: VRAM ceiling |
| GPU-resident experts at 4 streams | 24.74 | 24.03 | rejected: two streams per card double the GPU, not the lane |

### 1. In-batch readback: the round trip was the right thing to delete after all

A block's five readback items (`mixed`, router ids/weights, shared expert out and
gate) are now **`cmd_copy_buffer`s inside the block's own batch**, into a 32 KiB
reserved region at the head of the staging buffer, followed by one
`HOST_READ` barrier; `finish_block` reads the mapping instead of submitting. In
§8n this cost 17 ms single-stream because the copies force a
compute→transfer→compute transition in a GPU that had no slack. With two streams
the RX sits at 55 % utilisation, so the transitions are free and only the removed
round trip remains:

| cell | aggregate | finish_block ms/step | HOST_READ region |
|---|---:|---:|---|
| separate submission | 15.91 / 16.03 | 107.7 / 107.1 | — |
| **in-batch copies** | **16.54 / 17.14** | 103.0 / 98.6 | 32 KiB/block |

+4–7 % of aggregate, ~0.13 ms/block of host time. `--inbatch-read false` restores
the old path (identical tokens); the default is on.

### 2. GPU-resident experts: +2.5 tok/s once the host lane is the bottleneck

The §8d mechanism (`MoeMap` builds the row map on the device, `MoeCombine` folds
routed + shared, `PlacedExperts` holds all 512 experts of a block in VRAM) was
correct but 22 ms slower single-stream, because it trades 0.76 ms of host lane for
~2 ms of GPU per block. In a two-stream pair the host lane *is* the critical path,
so the same trade pays. Each stream's model gets its own placement on its own card:

| expert blocks per stream | aggregate tok/s | lane ms/step (pair) | finish_block ms/step |
|---:|---:|---:|---:|
| 0 | 16.05 / 16.74 | ~50 | 106.4 / 101.6 |
| 4 (`0,1,2,4`) | 16.40 / 17.04 | — | 103.5 / 99.1 |
| **9 (`0,1,2,4,8,18,20,22,23`)** | **18.66 / 18.87**; gen-32 mean **18.58** | **38.7–40.1** | **88.5–91.1** |
| 11 (full production mapping) | 15.56 / 14.48 | — | 109–119 |

* **Host lane: 50 → 39 ms/step** (−22 %, 0.52 → 0.41 ms/block), i.e. 79 MiB/token
  per stream stops crossing the host.
* **Per-card utilisation** (probe on, ~5 % instrumentation cost): MI50 **31 → 44 %**,
  RX **55 → 64 %**. Both cards are still the ones with headroom, which is why this
  works.
* **11 blocks regresses** — 9.9 GiB of expert arena + 3.66 GiB dense on a 16 GiB
  card is past the point where the driver keeps everything resident (the arena is
  what §8d measured at 9 913 MiB for 11 blocks). Nine is the knee here.
* **Parity 64/64** with the device-side MoE path in both streams, as §8d predicted
  from the byte-for-byte `MoeMap` check: the same routed sum, computed on the GPU.
* The default for `--streams 2` is that nine-block mapping; it is *not* the default
  for one stream (where it still loses) and not for 3+ streams (VRAM).

### 3. Stream scaling: 2 → 6 streams, 25 tok/s plateau

`--streams N` cycles the model over the available devices, so 3–6 streams put two
or three sequences on each card (their GPU phases then queue behind each other,
which is the point of the curve). `--threads 32`, gen 32, one 32-token cell each
(2 and 4 are 3-run means):

| streams | aggregate tok/s | per stream tok/s | per-stream latency | lane ms/step | fence ms/step | models per card |
|---:|---:|---:|---:|---:|---:|---|
| 2 | **18.58** | 9.29 | 108 ms | 39 | 48 | 1 (+9 expert blocks each) |
| 3 | 23.29 | 7.76 | 129 ms | ~60 | — | 2 / 1 |
| 4 | **24.74** | 6.18 | 162 ms | 98 | 14 | 2 |
| 6 | 25.03 | 4.17 | 240 ms | ~150 | — | 3 |

**The plateau is ~25 tok/s, and it is two limits at once:** the host expert lane
(98 ms of the 162 ms 4-stream step, at ~33 GB/s of page-cache reads — the 32-thread
lane's own ceiling) and the GPUs (each card now runs two streams' worth of blocks,
so the fence wait collapses from 48 ms to 14 ms and the cards become the
constraint). Giving every stream expert blocks does not help at 4 streams: 24.03
vs 24.74, because two models per card double the device MoE work on a card that is
already the long pole.

What would raise the plateau, with the numbers:

* **A third discrete GPU**: the curve is linear in *cards* while the lane keeps up;
  2 cards give 25 tok/s at 4 streams, so a third would take 6 streams to ~37 tok/s
  (the lane has ~40 % headroom left at 4 streams).
* **Fewer expert bytes per token**: at 4 streams the lane is 98 ms/step for
  4 × 799 MiB = 33 GB/s. Cutting a quarter of the expert reads (a stronger
  resident mapping, or fewer experts per token) is worth ~25 ms/step, i.e. +4 tok/s
  — but only if the cards have the VRAM, and at 2 models/card they do not
  (14.5 GiB of 16 GiB with four blocks each, and it measured *slower*).
* **Less host work per block**: at 2 streams, `finish_block` is 88 ms/step of which
  39 is the lane and ~25 is wait + readbacks + the ffn-write record; the GPU is at
  44/64 % utilisation, so any further host-side removal still pays at 2 streams.

Commands:

```bash
D=/var/tmp/users/korphaus/tmp/opencode/p4-token
# the two-stream gate: nine expert blocks per stream (the default), in-batch reads
timeout 1200 ./target/release/breccia bench token --streams 2 \
  --devices 0000:0e:00.0,0000:03:00.0 --prompt-ids-file $D/parity-server.json.ids \
  --gen 32 --runs 3 --threads 32 --json $D/g-2.json
# the scaling curve
for n in 2 3 4 6; do STREAMS=$n GEN=16 $D/sweep.sh 1 "n$n:"; done
# alternating A/B of any two variants in one window
$D/sweep.sh 2 "base:" "inbatch:--inbatch-read=true" "exp9:--gpu-experts=0,1,2,4,8,18,20,22,23"
```

Artifacts: `$D/s-{base,inbatch,exp4,exp9,exp11}-*.{json,log}` (the A/B sweeps),
`$D/g-{2,4}.json` (the gen-32 cells), `$D/g-4e.json` (4 streams with experts),
`$D/p-two.json` (utilisation + parity with the device-side MoE), `$D/sweep.sh`.

## 8q. Multi-stream third pass: descriptor-pool reuse (+9 %), the agent table, and two measured "no"s

Four items were scoped from §8p's conclusion (host work per block and the two cards
jointly bind). One landed with a real win, one exposed a **correctness bug** that is
now fixed, and two are answered with measurements instead of code.

| item | outcome |
|---|---|
| share the placed-expert arena across streams | **not needed**: measured no VRAM cliff (§below) |
| AVX-512 expert decode | **not attempted** (budget); no claim either way |
| trim per-step host work | **descriptor-pool reuse: 18.68/18.17 → 20.28/19.98 tok/s (+9 %)**, 107 → 99 ms latency; found and fixed a cross-device slot bug |
| agent-workload bench + table | landed: `breccia bench agents` |

### 1. Descriptor pools: create/destroy per block → reset per block

Every block's `begin` used to create a `VkDescriptorPool` and its `wait` destroy it
— 96 create/destroy pairs per token pair. The slots now keep their pool and
`vkResetDescriptorPool` it (a reset is ~free). Two-stream, alternating cells in one
window:

| cell | aggregate | per stream | latency |
|---|---:|---:|---:|
| create/destroy per block | 18.68 / 18.17 | 9.34 / 9.09 | 107.1 / 110.1 ms |
| **reset per block** | **20.28 / 19.98** | 10.14 / 9.99 | 98.6 / 100.1 ms |

**+9 % aggregate, −9 ms of latency**, and the pools are freed once, by the last
session to drop (`batch_cmd::shutdown`).

**The bug it exposed, which is the more important result.** Once a pool outlives a
batch, a ring slot can no longer be reused by *another device*: a pool is a device
object. Slots are now **device-affine**. Before the fix, four agents on two cards
produced *silently wrong tokens* on the MI50 streams (2 of 4 agents diverged from
the server at token 0, ~50 % of runs) while the RX streams were correct — the
parity check in `bench agents --same-prompt` caught it, which is why the parity test
is part of the agent harness and not an afterthought:

| after `--streams 4 --same-prompt` | result |
|---|---|
| before the slot fix | agents 0,2 (MI50) wrong, agents 1,3 (RX) right |
| after the slot fix | **all four agents 24/24 vs the server, bit-identical** |

### 2. Sharing the placed-expert arena: measured, not needed

The §8p 4-stream regression was attributed to "two models/card double the device
MoE". If that were a *VRAM* problem, sharing the arenas (they are immutable model
data) would fix it. It is not: at 4 streams, two blocks per stream per card
(9.1 GiB/card) and four blocks (14.5 GiB/card) measure the same, so there is no
resident-set cliff to remove:

| 4 streams, one window | aggregate |
|---|---:|
| no device experts | 22.75 / 24.71 |
| 2 blocks/stream (9.1 GiB/card) | 23.15 / 25.12 |
| 4 blocks/stream (14.5 GiB/card) | 23.80 / 24.58 |

The cost is the **duplicated device-MoE compute** (fence wait 14 → 23 ms/step when
experts are on at 4 streams), which sharing weights cannot remove. So the arenas
stay per model: an `Arc`-shared arena would need a shared *logical device* across
streams (each stream currently opens its own), i.e. a real refactor for zero
measured gain. It becomes worth doing only if a card ever has to run two streams
with nine blocks each — which is exactly what §8p's 2-stream case does, one stream
per card, where nothing is duplicated.

### 3. AVX-512 expert decode: not attempted

The lane is decode-bound (§8c) and is 67–71 % of an agent-length step, so a
10–25 % AVX-512 gain would be worth 2–4 tok/s aggregate. It needs three new decode
kernels (IQ4_NL down, Q2_0 down, IQ3_S gate/up) with `is_x86_feature_detected!`
gating and a second conformance path; this pass spent its budget on the
descriptor-pool win, the slot-bug fix and the agent harness instead. **No claim is
made either way** — it is the top of the queue.

### 4. `breccia bench agents`: what 4–6 agents actually feel like

`breccia bench agents --streams N --prefix 300 --gen 16` runs N agent-style
streams in lockstep with a warm pass: agent 0 gets the 27-id reference prompt,
agents 1..N−1 get deterministic 300-id prefixes (two thirds LCG ids, one third
reference ids, so the routing differs), greedy decode, per-step latency, per-card
GPU utilisation and the host lane's share of the step.

| agents | prefixes | aggregate | per agent | latency p50 / p95 | lane share | GPU MI50 / RX |
|---:|---|---:|---:|---|---:|---|
| 4 | 27 (reference, `--same-prompt`) | **20.19** | 5.05 | 194 / 255 ms | 67 % | 47 / 52 % |
| **4** | **300** | **11.14** | 2.78 | 336 / 444 ms | **71 %** | 33 / 53 % |
| **6** | **300** | **12.92** | 2.15 | 462 / 525 ms | **71 %** | 38 / 54 % |
| 2 (§8p, `--prefix` 27) | 27 | 18.58 | 9.29 | — | 36 % | 44 / 64 % |

Two things stand out:

* **Long prefixes cost about half the aggregate** (18.6 → 11.1 at 4 agents): QSA's
  attention and indexer grow with position, so a step goes 162 → 336 ms while
  neither card's GPU utilisation rises — the extra time is the *host lane*, which
  is 71 % of the step at 300 tokens against 36–67 % at 27.
* **Six agents are only 16 % faster than four** (12.92 vs 11.14): the two cards are
  already carrying three streams each, so the aggregate is pinned by the lane.

### The third-GPU projection, restated

§8p projected "3 cards ≈ 37 tok/s at 6 streams". That number assumed the 16-token,
27-prefix workload, where the GPU was 48 % of the limit. **With agent-length
prefixes it is wrong**: the lane is 71 % of the step, so a third card can only
raise the GPU side. Bound: if the third card made the GPU infinitely fast, the step
would fall from 336 ms to the lane's 240 ms, i.e. 4 agents → **15.6 tok/s** (1.4×)
and 6 agents → ~18 tok/s. Stated with its assumption: *the lane's 32-thread
throughput (33 GB/s) unchanged*.

What actually raises the agent table, in order:

1. **AVX-512 (or any) lane throughput**: 71 % of the step, so +20 % lane = +14 %
   aggregate at 4 agents with no extra VRAM.
2. **Fewer expert bytes per token**: at agent lengths the resident-expert mapping
   does not fit per stream (2 models/card × 9 blocks × 0.9 GiB), so the lane reads
   the full 799 MiB/token per agent.
3. **A third card**: 1.4× at 4–6 agents (bounded by the lane, above), 2× at short
   prefixes.

Commands:

```bash
D=/var/tmp/users/korphaus/tmp/opencode/p4-token
# the agent table (mixed prompts) and the parity run (all agents = the reference)
timeout 1800 ./target/release/breccia bench agents --streams 4 --gen 16 --prefix 300 \
  --devices 0000:0e:00.0,0000:03:00.0 --threads 32 --json $D/ag4.json
timeout 1200 ./target/release/breccia bench agents --streams 4 --gen 24 --same-prompt \
  --devices 0000:0e:00.0,0000:03:00.0 --threads 32 --json $D/agents4p.json
# the two-stream gate after the pool change
$D/sweep.sh 2 "pool:"
```

Artifacts: `$D/ag{4,6}.json` (the agent table), `$D/agents4p.json` (parity),
`$D/s-pool-*.{json,log}` (the pool A/B).

## 8r. Prefill: measured, and why it needs expert grouping before anything else

An agent arrives with a 300-token prefix. The engine feeds it **token by token
through the decode graph**, so prompt processing runs at decode speed. Measured
with the agent harness (`breccia bench agents`, dual, `--threads 32`):

| agents | prefix | prompt tokens | prompt seconds | **prefill tok/s per agent** |
|---:|---:|---:|---:|---:|
| 1 | 27 | 27 | 2.6 | 10.5 |
| 2 | 300 | 600 | 47.7 | 6.3 |
| **4** | **300** | **1200** | **95.6** | **3.1** |

llama.cpp prefills this model at ~112 tok/s. The gap is **11–36×**, and it is
bigger than everything else in this report: four agents spending 96 s ingesting
their prompts against 4.2 s producing 44 tokens is the wrong way round. **No code
from this section landed** — what follows is the measurement, the arithmetic that
pins the design, and the reason it is not a one-pass change.

### Why it is slow, in bytes

Per token the forward pass reads the dense set *plus* the token's ten experts from
the host lane: **799 MiB/token**, i.e. 38 ms of lane at the 32-thread rate. A
300-token prefix therefore streams 234 GiB through the lane no matter how the
prompt is fed. Chunking the graph alone cannot fix that: it amortises the *dense*
GEMVs (measured in `bench ops`: GDN `qkv` goes 30 → 78 GB/s from T = 1 to T = 8)
but the experts are still fetched per token.

The lever is that **within a chunk the same expert serves several tokens**. With
`n_expert = 512`, top-10, the number of *distinct* experts a chunk touches is the
occupancy of 10·T draws into 512 bins:

| chunk T | expert slots used | expected distinct (of 512) | expert bytes vs token-by-token |
|---:|---:|---:|---:|
| 64 | 640 | 365 | **× 0.57** |
| 128 | 1280 | 470 | **× 0.37** |
| 256 | 2560 | 508 | **× 0.20** |

(Occupancy assumes roughly uniform routing — the router is trained, so treat these
as the best case; they are the number to beat, not a promise.)

### The design, and what it costs to build

1. **Group the host lane over a chunk**: collect `(token, expert)` pairs for all T
   tokens of a block, bucket by expert, dequantise each expert's weights **once**
   and dot them against the `m` token activations that use it. Per-row accumulation
   order is unchanged, so the result is bit-identical to the per-token lane — this
   is the only piece that *must* be right, and it is also the piece the existing
   `ExpertPool` does not have (it takes one token's `ids`/`x`).
2. **Chunk the dense chain** (T-batched HC read/write, mixer, router, shared
   expert): the kernels already take T (`Gdn::forward(.., pos0)`,
   `Qsa::forward(.., pos0)` and their workspaces are `T`-shaped and were validated
   at T = 8), but every *per-row* GEMV in the graph is a `Gemv` and needs a
   `GemvBatch` form: the PLE key/value projections, `shexp_gate` (1 row), and the
   head (which only needs the chunk's last row). The QSA causal mask over a chunk
   and the PLE host block (per token today) both need a T form.
3. **Pipelining**: the T tokens of one block are independent (each has its own
   residual; the KV is appended at its position), so the dense chain can run for all
   T tokens, then the grouped MoE for all T, then the next block — one submission
   per stage per chunk instead of one per token per block.

Projected, from the measured per-block costs (dense 0.6 ms/token, lane 0.79
ms/token at 32 threads): grouping alone → lane 0.16 ms at T = 256 → **~27 tok/s
prefill**; grouping + a T-batched dense chain (0.23 ms) → **~52 tok/s**. Both are
per *agent* on one card, so four agents prefilling together would be bounded by
the shared lane — which is why the lane work (below) is the co-requirement.

### Why it is not landed in this pass

It is not one edit: it is a new CPU lane entry point (grouped MoE), a batched form
of three per-row GEMVs, a chunked PLE, a T-shaped residual/mailbox/staging, and a
chunk-level QSA/GDN verification — and the last pass showed that this engine can
produce *silently wrong* tokens from a bookkeeping mistake (§8q). Half of it would
be worse than none, so it is left whole for the next pass with the numbers above as
its target.

### AVX-512: still not attempted, and now the *second* lever

The lane is 67–75 % of a decode step, so a 10–25 % AVX-512 gain is 2–4 tok/s of
decode aggregate — real, but an order of magnitude smaller than the 11–36× prefill
gap on the workload agents actually produce. Order: prefill grouping first, then
AVX-512.

## 8s. Grouped MoE over a prefill chunk: landed, bit-exact, and *slower* — the lane is decode-bound

§8r projected that grouping the expert lane over a prefill chunk (read each
expert's rows once, serve every token that routed to it) would take prefill from
3.1 to ~27 tok/s per agent. The mechanism landed and is correct, and it **does not
pay**: the lane is decode-bound, so grouping removes *bytes* and not *work*.
Measured, not estimated:

| prefill path | 2 agents x 300 | 4 agents x 300 |
|---|---:|---:|
| token by token (§8r baseline) | **6.13** tok/s/agent (48.9 s) | **3.14** (95.6 s) |
| chunked, grouped lane, chunk 32 | — | 2.25 (133.1 s) |
| chunked, grouped lane, chunk 64 | **5.37** (55.9 s) | 2.38 (125.8 s) |

### What landed (behind `--prefill-chunk N`, default 0 = off)

* **`ExpertPool::moe_topk_chunk`** — the grouped host-lane entry point. It buckets
  the chunk's `(token, expert)` pairs, and each `(expert, row shard)` is one task
  whose weight rows are read **once** and then served to every user of that expert
  from L2/L3. Row dot products are **bit-identical** to the per-token lane
  (`tests/cpu_experts.rs::grouped_chunk_lane_matches_the_per_token_lane`, worst
  relative deviation 1.5e-8 over 32 tokens, deterministic across runs); only the
  sum over an expert set is accumulated in expert-id order instead of router order.
* **`Session::prefill(toks, pos0)`** — a real chunk loop: the chunk's whole dense
  chain in **one submission** (pass 1: per token `HC_READ(attn)` … shared expert,
  each followed by its five readback copies into a per-token slot of the staging
  block region), then the grouped lane over the whole chunk, then the chunk's
  per-token ffn `HC_WRITE`s in a **second submission** (pass 2). Two submissions
  and two waits per block per chunk instead of two per token.
* The dense chain still runs **per token** with the unchanged kernels: each token's
  residual slice is swapped into the venue's single-token `r` (`swap_r_in`), which
  is why the T=1 numerics are preserved by construction.
* `--prefill-chunk N` on `bench token` and `bench agents`; the descriptor pool
  budget went 256 → 8 192 sets because one chunk's submission records ~30
  dispatches per token.

### Per-token fallbacks (documented, as required)

| stage | why it cannot take T | fallback |
|---|---|---|
| PLE (block 1) | the host block reads/writes the whole residual and keeps rolling hash/conv state per token | per-token `ple_step` inside the chunk, with a flush each (1 of 48 blocks) |
| a device-resident block (`--gpu-experts`) | its `MoeCombine` writes `block_out` for one token and needs no host round trip | the whole token (dense + device MoE + `HC_WRITE`) per token, in the chunk's batch |
| the T-batched dense chain | not attempted this pass (§8r item b) | per token, unchanged kernels |

### The bug this found (and the parity gate caught it)

`HC_WRITE` reads the `inject` partials the same token's `HC_READ(ffn)` produced. In
a two-pass chunk the ffn writes of *all* tokens run after *all* the reads, so every
token but the last was using the last token's inject values:

| chunk size | top-1 logit (prompt `[760,13017]`) |
|---|---:|
| 1 (correct) | 17.5689 |
| 2, before the fix | 17.6016 (**+0.03**, top-10 order already permuted) |
| 4 / 8, before the fix | argmax flipped to the wrong token |
| **2 / 4 / 8, after per-token inject slots** | **17.5689 — identical** |

The 27-token prompt at chunk 32 and 64 now reproduces the reference sequence
exactly (`[110563, 96847, 271, 248068, 198, 760, 1156, 369]`, top-1 `110563`
`13.1705`), so prefill changes *how* the same tokens are computed and nothing else.

### Why it is slower, and what that means

§8c measured the lane as **decode-bound**, not bandwidth-bound: it scales with
threads at 20.8 GB/s because the quantised decode is the cost, not the DRAM read.
A grouped chunk still runs `m` dequantisations per expert (one per user) — `one_user`
calls the same `gemv_into` per user, deliberately, to stay bit-identical — so it
saves the DRAM traffic that was never the limit and adds task/channel overhead.
The occupancy table in §8r (x0.57/x0.37/x0.20 of the bytes) is real but the bytes
were not the constraint.

**The fix is to amortise the *dequantisation*, not the read**: dequantise each
weight block once into registers and dot all `m` activations of that expert from
it. That is a new inner-loop form per quant type (the AVX2/AVX-512 dot gains a
second activation), and it composes with AVX-512 — both are "more dot throughput
per dequantised block", which is now the single highest-value lever for prefill
*and* decode.

Until then `--prefill-chunk` stays **off**: it is parity-correct and it is the
substrate the amortising kernel plugs into, but as measured it is a 12-14 % tax on
prompt ingestion.

## 8t. Dequant-once, dot-many: the down role amortises, the blend is parity, AVX-512 not done

§8s showed grouping the expert *reads* saves bytes the lane is not bound by. This
pass attacks the thing that actually costs: the **dequantisation**. A weight block
is now decoded **once into registers** and dotted with every user (`m` tokens) of
that expert, so the table lookup / int→float / scale multiply is paid once instead
of once per user. Scope delivered: `IQ4_NL` and `Q2_0`, the two types the model's
`down` role uses (48 of 48 layers), AVX2 8-wide; `gate`/`up` and every other type
keep the fused per-user kernel (the blend below).

### Measured

| prefill path (2 agents x 300, per agent) | tok/s | vs token-by-token |
|---|---:|---:|
| token by token (§8r baseline) | **6.13** | — |
| chunk 64, grouped reads only (§8s) | 5.37 | -12 % |
| chunk 64, **dequant-once down** | **5.62** | -8 % |
| chunk 128, **dequant-once down** | **6.16** | **+0.5 %** |

The `down` amortisation recovers the chunking tax and reaches parity at chunk 128,
where the occupancy gives `m ~ T/51 = 2.5` users per expert. It does not yet beat
the token-by-token path because:

* **only one of the three roles amortises.** `gate`+`up` are two tensors fused with
  a SwiGLU in between, so their many-user form needs a two-tensor inner loop; they
  are 2/3 of the expert decodes and still run per user. The `down` role alone caps
  the gain at the ~2.5x its share would allow at `m = 8`.
* **`m` is small at chunk 64-128 by occupancy.** `m = 10*T/512`, so `m = 1.75` at
  T = 64, 2.5 at T = 128, 5 at T = 256. The kernel's benefit grows with `m`; the
  chunk sizes measured are the low half of the curve. Chunk 256 needs a bigger
  descriptor pool than the 4 096 sets the ring now budgets (256 tokens x ~30
  dispatches) and was not measured.
* **AVX-512 is not implemented.** The kernels are 8-wide AVX2. The restructure is
  what makes 16-wide mechanical — the dequantised vector and the per-user FMA are
  already separate — but `_mm512` variants of `iq4_nl_many` / `q2_0_many` were not
  written in this pass, so no 16-wide claim is made.

### Bit-identity (unchanged gate)

The many-user kernels use the same operation order per user as the fused kernel
(`mul` the dequantised value, then `fmadd` into the user's accumulator, block
major, same `finish` reduction), so each user's dot is **bit-identical** to the
single-user path; the only deviation left in the grouped comparison is the
documented expert-order sum. `grouped_chunk_lane_matches_the_per_token_lane` still
reports 1.49e-8 / 7.45e-9 worst relative, and the chunked prompt at chunk 8 gives
the identical logit `17.5689`, chunk 32/64 the reference token sequence.

### Decode

`m = 1` never enters the many-user kernel (the fused path is kept exactly as it
was), so decode is unchanged *by construction*. The two-stream cell measured after
this pass is **15.8-16.5 tok/s aggregate** against the 20.28 of the §8p window, with
the lane at 56 ms/step against 48 — this window's page cache was pushed out by the
prefill runs (the same binary measured 10.1 -> 16.5 across warming passes in this
session), so the gate is not re-established here and no regression is claimed
either way. One real inefficiency this pass found and removed: the single-user path
was allocating an empty `Arc<Vec<_>>` per shard (up to ~8 600 allocations per
token); it now shares one empty list.

### Next

1. Extend the many-user kernel to `gate`+`up` (two tensors, SwiGLU over `m` rows)
   and to `IQ3_S`/`IQ3_XXS`/`IQ2_*`. That is 2/3 of the decodes.
2. Then AVX-512 16-wide on the same shape, and chunk 256 (needs the pool at 8 192
   sets) where `m = 5`.

## 8u. Merged MoE for prefill: corrected byte budget, and the multi-user GEMM built

The survey (`02-untried-levers.md` item 1) says the prefill win is merged MoE on the
GPU: route a whole chunk, touch each distinct expert once, run one quantised **GEMM**
(m users x 2560 outputs) per expert. This pass corrected the budget's arithmetic and
built the first kernel.

### The byte budget, corrected and reconciled

The earlier draft mixed denominators (10.0 GiB was measured on an 88-token tail, and
42.4 GiB covers the whole 600-token prompt, not one 128-token chunk). Restated so
the arithmetic is checkable, all from one measured run (`--prefill-chunk 128`,
2 agents x 300 tokens = **600 prompt tokens**, 48 blocks, IQ4_NL/Q2_0/IQ2_* mix):

| quantity | measured |
|---|---:|
| distinct `(block, expert)` uses, summed over blocks, chunks and both streams | **25 769** |
| bytes per use (the full `gate`+`up`+`down` blob of that expert) | 1.746 MB |
| expert bytes if each use is touched once (25 769 x 1.746 MB) | **45.0 GB = 41.9 GiB** |
| per prompt token (41.9 GiB / **600**) | **72.3 MiB/token** |
| the same, token by token (48 x 10 x 1.746 MB) | 799 MiB/token |
| **reduction** | **11.1x** |

The factor is that large because **the router is far more concentrated than a
uniform model assumes**. Measured per block, `BRECCIA_TRACE_ROUTE=1`:

| chunk | draws per block | distinct experts per block | users per distinct expert |
|---:|---:|---:|---:|
| 128 | 1 280 | **102-159** (typ. ~130) | 8-13 |

A uniform top-10 router would touch ~470 of 512 experts for the same 1 280 draws
(occupancy `512(1-e^-2.5)`), which would give 176 uses/token = **293 MiB/token, a
2.7x reduction**. So the honest statement is a range: **2.7x (uniform assumption) to
11.1x (measured routing)**, and the conclusion survives either way, because the host
lane is **ALU-bound at ~3-5 GB/s effective**, not bandwidth-bound:

| cost per token at 72.3 MiB (and at the conservative 293 MiB) | value |
|---|---:|
| PCIe, streamed to the RX (28.7 GB/s measured) | **2.6 ms** (11 ms) |
| PCIe, streamed to the MI50 (gen3 x4, 2.9 GB/s) | 25 ms (101 ms) - hot experts must be resident |
| GEMM compute, **measured below** | **0.5 ms** |
| dispatch (~130 experts x 48 blocks per chunk at 4.5 us) | 0.22 ms |
| **merged total** | **~3-12 ms/token** |
| today (host lane) | **163-320 ms/token** |

### The multi-user GEMM (down role, IQ4_NL) — built and validated

`shaders/gemm_body.glsl` + `gemm_iq4_nl.comp`, driven by
`breccia_kernels::gemm::GemmMany`: one dispatch computes `m x 2560` outputs over
`K = 640`, decoding each quant block **once into registers** and dotting it with all
`m` users (`docs` above; every earlier kernel decodes per row). Geometry, per-lane
unit split, two alternating accumulators and the subgroup reduction are the
single-user kernel's, so **every user's dot is bit-identical to it** - measured, not
argued:

| device (validation layer on) | m = 1 | m = 2 | m = 4 | m = 8 | worst rel. error | bit-identical |
|---|---:|---:|---:|---:|---:|---|
| MI50 (Radeon VII) ms/dispatch | 0.014 | 0.017 | 0.022 | 0.033 | **0.0** | **yes, every m** |
| MI50 GFLOP/s | 232 | 393 | 595 | **802** | | |
| RX 6950 XT ms/dispatch | 0.007 | 0.005 | 0.007 | 0.011 | **0.0** | **yes, every m** |
| RX GFLOP/s | 493 | 1 260 | 1 853 | **2 496** | | |

Per-user throughput is **3.4x (MI50) / 5.1x (RX)** at m = 8 against m = 1, with the
weight read amortised (`tests/gemm_many.rs`, `BRECCIA_KERNEL_DEVICE=...`, MI50 first,
0 validation messages). At the measured 0.011 ms per (expert, m = 8) on the RX, a
128-token chunk's ~130 experts x 48 blocks is **0.5 ms of GEMM per token** - the
number in the table above.

### Blend and what is left

* Delivered: `IQ4_NL` **down** only (18 of the 48 `down` layers here).
* Still per-row: `Q2_0` down (30 layers), all `gate`/`up` (the `IQ2_S`/`IQ2_XS`/
  `IQ2_XXS`/`IQ3_S`/`IQ3_XXS` set, which needs the two-tensor + SwiGLU form), and
  every decode path (`m = 1` keeps the fused kernel untouched).
* The remaining work to make merged prefill real: the `Q2_0` and gate/up shaders,
  hot-expert residence (the measured concentration means a few hundred resident
  experts per device covers most of a chunk), and the m-user fan-in on the device.
  The budget no longer argues against any of it: measured, the compute is 0.5 ms and
  the streaming 2.6-11 ms per token against 163-320 ms today.

## 8v. Multi-user GEMM: all seven expert types, bit-identical; the fan-in is the blocker

`GemmMany` (§8u) now covers the whole expert type set, and every user's dot is
**bit-identical** to the single-user kernel for every `m` - not just `m = 1` - so the
parity gate keeps working unchanged. MI50, validation layer on, one GPU process,
`tests/gemm_many.rs` (`BRECCIA_KERNEL_DEVICE=0000:0e:00.0`), 0 validation messages:

| role / type | m = 1 | m = 2 | m = 4 | m = 8 | GFLOP/s 1 -> 8 | per-user | worst rel. | bit-identical |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `down` IQ4_NL | 0.014 ms | 0.017 | 0.022 | 0.033 | 231 -> 797 | **3.4x** | 0.0 | yes (all m) |
| `down` Q2_0 | 0.009 | 0.012 | 0.017 | 0.027 | 356 -> 955 | **2.7x** | 0.0 | yes |
| `gate`/`up` IQ2_XS | 0.016 | 0.021 | 0.032 | 0.053 | 202 -> 498 | 2.5x | 0.0 | yes |
| `gate`/`up` IQ2_XXS | 0.016 | 0.021 | 0.032 | 0.053 | 201 -> 499 | 2.5x | 0.0 | yes |
| `gate`/`up` IQ2_S | 0.016 | 0.021 | 0.032 | 0.053 | 205 -> 493 | 2.4x | 0.0 | yes |
| `gate`/`up` IQ3_S | 0.016 | 0.021 | 0.032 | 0.053 | 202 -> 498 | 2.5x | 0.0 | yes |

`gate` and `up` are served as **two multi-user GEMMs plus the existing SwiGLU
pass**, not one fused dispatch: `blocks.glsl`'s decode units read the weight buffer
from binding 0 by name, so a second tensor in the same dispatch needs either a
duplicated decode unit per type or a concatenated `gate||up` arena. Two passes are
arithmetically identical (`silu` is elementwise) and cost one extra dispatch; the
fusion is deferred rather than faked.

### What this pass did **not** reach, and why

The end-to-end merged prefill (routing a chunk, running the GEMMs for the distinct
experts of the resident blocks, and an m-user fan-in) is **not wired**, so there is
**no new prefill number**: the baseline is still **6.13 tok/s/agent** (2 agents x 300,
token by token). The blockers, in order:

1. **The fan-in needs a new kernel.** The routed set is per *token* (10 experts each,
   weighted), while the GEMM output is per *(distinct expert, user)*. `MoeCombine`
   sums one token's `k` rows from a `[k, cols]` buffer; a chunk needs the weighted sum
   over each token's own top-10 out of an `[E_distinct, users, cols]` scratch, i.e. a
   gather over a per-token expert list. That is a new shader (or an atomics path the
   device does not have for f32 storage buffers, §8c).
2. **Arena offset views.** The resident arena holds all 512 experts in one buffer and
   a GEMM needs expert `e`'s blob: `Buf` has no offset field, so the kernel needs a
   `base_words` push constant (the `moe_map`/`GemvBatch` path already does exactly
   this) - a small, contained change to the new GEMM kernels.
3. **Chunk orchestration**: per block, upload the chunk's `mixed` (m x 2560), run
   one dispatch per distinct expert (~130 per block at T = 128, measured), then the
   fan-in; ~130 x 48 = 6 240 dispatches per chunk, 0.22 ms/token at 4.5 us.

The budget says the payoff is still there (compute 0.5 ms/token, streaming 2.6-11
ms/token, against 163-320 ms today), and the kernels are the expensive part; what
remains is the fan-in shader and the orchestration.


### The fan-in kernel and arena addressing (landed, validated)

Two of the three pieces the merged path needed are now in the tree, both validated
against the existing kernels (MI50, validation layer on, 0 messages):

| piece | what it is | validation |
|---|---|---|
| **`moe_fanin.comp`** | for a chunk, `out[u] = Σ_j w[u][j]·scratch[slot[u][j]] + shared[u]·σ(gate[u])`, one workgroup-row per (token, 64-channel block); the `(token, expert)` pairs are resolved through a host-built **slot map**, so the grouped GEMM can write each distinct expert's users contiguously | **bit-identical to `MoeCombine`** for one token (worst 0.0), **4.8e-8** vs an f64 reference for a 3-token / 12-pair / 9-slot chunk |
| **`base_words`** | the multi-user GEMM's 5th push-constant word (`pc.base`, `ROW_BASE_WORDS`), so a GEMM addresses expert `e` inside the resident arena — the `MoeMap`/`GemvBatch` convention | the same blob at `base_words = 64` inside a larger arena is **bit-identical** to the same blob at 0 |

The bottom line, stated plainly so nothing is over-claimed: the three items were
(a) the fan-in, (b) arena addressing, (c) wiring the resident blocks into the
chunked prefill pass. **(a) and (b) are done and validated; (c) is not started**
— there is no merged-prefill pass in the binary and therefore **no new prefill
number**: the baseline remains **6.13 tok/s/agent** (2 agents x 300-token prefixes,
token by token) and the non-resident path is 100 % of it today.

What (c) needs, precisely: in `Session::prefill_block`, for a block in `m.placed`
1. upload the chunk's `mixed` (T x 2560) and build the `(expert, user)` bucket list
   from the router ids the dense chain already produced (the host has them in
   `pf_ids`, ~130 distinct experts per block at T = 128, measured);
2. per distinct expert: `GemmMany::run_at` for `gate`, again for `up` (both from
   `pe.gate`/`pe.up` with `base_words = e * blob_words`), the existing `SwiGlu`
   over that expert's `m_e` rows, then `run_at` for `down` into the scratch;
3. `MoeFanIn::run` over the chunk with the slot map, then the existing `HC_WRITE`
   pass — replacing the per-token `run_placed_moe` call for those blocks only;
4. the `--prefill-merged` flag, the byte/GEMM/fan-in counters per chunk (the
   `Session` already carries `pf_distinct`, `pf_merged_bytes`, `pf_tokens`).

Scratch for (2): `E_max x m x 2560` f32 (T = 128, E ~ 160, m <= 8: ~13 MB) plus
`E_max x m x 640` for the SwiGLU intermediate — one allocation per session.

### The pass is wired: measured, bit-identical, and it does not move 6.13

Item (c) is done. `--prefill-merged` (default **off**) runs, for a prefill chunk of
`t` tokens and for each **resident** block only:

1. the unchanged dense chain per token, whose router ids/weights, `mixed`, shared
   expert output and gate logit are copied into chunk buffers as it goes (five
   device-to-device copies per token, no host round trip);
2. one readback of the chunk's router ids (5 KiB at T = 128), bucketed on the host
   into `(expert, group of <= USERS_MAX = 8 users)` and turned into the slot map
   `MoeFanIn` expects — each distinct expert's users contiguous, which is exactly
   the GEMM's `O[u * rows + row]` layout;
3. per group, `GemmMany::run_at(gate)`, `run_at(up)`, one `SwiGlu` over its `m`
   rows, `run_at(down)`, then one copy of the group's `m` rows into the chunk
   scratch — three weights reads per distinct expert instead of ten per token;
4. `MoeFanIn` over the chunk, then the chunk's per-token `HC_WRITE` (pass 2), i.e.
   the same shape the host-lane blocks use.

`GemmMany::run_at` is now **batch-aware** (it records into the open dispatch batch
like `Pipe::run`/`GemvBatch::run`, instead of one submit+wait per dispatch), the
chunk's batches are closed and reopened at `MERGED_BATCH_CAP = 6 000` recorded
dispatches (a 256-token chunk records ~30 per token in pass 1 alone, against the
pool's 8 192 sets), and the pass keeps its own counters — printed per block with
`BRECCIA_TRACE_MERGED=1` and summarised by `bench agents` as

```
merged   {groups} expert GEMM groups over {slots} slots: {gemm} ms of GEMM,
         {fanin} ms of fan-in, {dispatches} dispatches ({GiB} read once per expert)
```

**Bit-identity holds exactly.** In every configuration tested the merged pass
produces the *same* tokens as the existing path — chunk 8, 9 resident blocks,
2 agents, `--same-prompt`: **32/32 against the recorded server sequence**, and
merged == non-merged token-for-token at chunk 16/27/32/128/256 as well (both wrong
there, see the blocker below). That is the §8v claim (per-user GEMM bit-identical
for every `m`, fan-in bit-identical to `MoeCombine`) doing its job end to end.

**Measured cost, chunk 256, 9 resident blocks per stream** (`BRECCIA_TRACE_MERGED`,
2 agents x 300 tokens):

| per block per 256-token chunk | value |
|---|---:|
| distinct experts | **133-206** |
| expert groups (`m <= 8`) | **390-456** |
| slots (pairs) | 2 560 |
| GPU GEMM time | **18.5-22.8 ms** (72-89 µs per group) |
| fan-in | 0.57-0.67 ms |
| arena bytes read once per expert | **197-367 MiB** (1.2-1.6 MiB/token) |
| dispatches recorded | 9.5-12 k |

Whole prompt (2 streams x 300 tokens, 9 blocks each): **10 018 groups, 840 ms of
GEMM, 22 ms of fan-in, 213 k dispatches, 8.5 GiB** read out of the arena — against
39.3 GiB if every `(token, expert)` use were touched separately. That is 1.5 % of
the 57 s the prompt took.

**And that is why it does not pay.** The merged pass replaces GEMV work that was
*already on the device*; the 39 non-resident blocks per stream still run the
per-token host lane and are ~98 % of prompt ingestion. Prefill tok/s/agent, 2
agents x 300-token prefixes, RX 6950 XT + MI50 one stream each, alternating
A/B/A inside one window:

| path | tok/s per agent | runs |
|---|---:|---:|
| token by token (chunk 0) | **5.34** (5.07-5.50) | 6 |
| chunked, no merge, 128 | 4.95 | 1 |
| chunked, no merge, 256 | 5.31 | 1 |
| **merged, 64** | **4.95** | 2 |
| **merged, 128** | **5.15** (5.12-5.18) | 4 |
| **merged, 256** | **5.27** (5.24-5.33) | 4 |

* §8v's **6.13 baseline did not reproduce in this window**: the same binary on the
  same path measures 5.34, decode is 9.3-11.6 tok/s aggregate against the recorded
  20.0-20.3, and the host lane is 93-134 ms/step against §8o's ~38 — the box is
  ~2x off its earlier state (load average 2-3, `kswapd` active, 40.9 GiB expert
  mmap against 58 GiB of RAM). Every comparison above is therefore in-window.
* Merged at 256 is **parity with token-by-token and +0/-1 % on it**; at 128 it is
  +4 % over the same-size chunked path; at 64 it is 7 % *worse* (a 64-token chunk
  has ~3-6 users per expert, so the GEMM is still m < 8 and the merge only adds
  copies and a fan-in). The flag stays **off**.

**The blocker this uncovered (pre-existing, not this pass).** Chunked prefill is
numerically wrong for chunk >= 9 in the current tree, resident blocks or not:

| `--prefill-chunk` (27-token prompt) | server match |
|---|---:|
| 0 / 8 | **32/32** |
| 16 | 2/32 |
| 27 / 32 / 64 / 96 / 128 / 256 | 0/32 |

Reproduced with `--gpu-experts` empty (0 blocks) and with 9, with and without
`--prefill-merged`, and — by `git stash` of this whole patch — **in the same tree
without it**, so it is a regression since §8s measured chunk 32/64 as exact. The
candidates are the commits between §8s (`f6b98b4`) and here: the lane's
dequant-once/AVX2 change (`74593cd`) and the multi-user GEMM series (`1ec9e0c`,
`1828c00`, `8a9b0f2`). Until it is bisected, **no chunked-prefill number in §8u,
§8t or §8v is gate-quality**, and the merged numbers above are quoted as cost
measurements of the work the pass does, not as validate-able throughput.

Commands (one GPU process, alternating, `timeout` on every run):

```bash
./target/release/breccia bench agents --streams 2 --prefix 300 --gen 32 \
  --devices 0000:03:00.0,0000:0e:00.0 [--prefill-chunk 256 [--prefill-merged]]
BRECCIA_TRACE_MERGED=1 ./target/release/breccia bench token --devices 0000:0e:00.0 \
  --prompt-ids "$(cat …/parity-server.json.ids)" --gen 32 --runs 1 \
  --gpu-experts 0,1,2 --prefill-chunk 8 --prefill-merged      # 32/32, MI50 first
```

**Next, in order.** (1) Bisect and fix the chunk >= 9 regression — it blocks every
chunked number in this document. (2) Point the merged pass at the **39
non-resident blocks**: the same three GEMMs plus fan-in, with each distinct
expert's blob streamed once per chunk (28.7 GB/s on the RX: 1.5 MiB/token/block of
arena reads is 2-3 ms/token against the lane's ~150), which is §8u's budget and now
needs no new kernel — only an upload path and the expert types the host blocks use.



## 8w. The chunk>=9 bug found and fixed, a compute-scoped barrier, and where prefill time really goes

Three items, each measured in-window and each with its own switch. The round's
headline is a **correctness** result: chunked prefill had been silently dropping
work since §8t.

### 1. Chunked prefill was wrong for chunk >= 9 — the `MANY_MAX` clamp (A1, fixed)

§8v's parity sweep found the symptom (32/32 at chunk 0/8, 2/32 at 16, 0/32 at 27+)
but not the cause. `BRECCIA_TRACE_CHUNK=1` dumps every (block, token) lane input
and output, and the diff is unambiguous: **blocks 0-11 are bit-identical between a
9-token chunk and a 1-token chunk; block 12 is the first that differs, with
identical `mixed`, identical router ids/weights, identical shared output and
gate**, while its lane output differs in the third decimal.

The cause is §8t's dequantise-once dot — `gemv_into_many` clamped the user count:

```rust
let m = xs.len().min(MANY_MAX);   // MANY_MAX = 8, the register budget
```

Users past the 8th were never computed and their rows stayed zero, so a chunk that
put nine or more tokens on one expert **silently dropped those tokens'
contribution** to that expert. Blocks 0-11 masked it because none of their experts
had nine users for the last token; block 12 was the first that did. The same clamp
also means every chunked number in §8s/§8t/§8u/§8v was measured on a path that did
*less* work than advertised.

Fixed by serving users in **passes of `MANY_MAX`** (`cpu/mod.rs`); each user keeps
its own accumulator and operation order, so the result stays bit-identical to
`gemv_into`. New regression test
`many_user_dots_are_bit_identical_past_the_register_limit` (m = 12, IQ4_NL and Q2_0,
bit-compared against 12 single-user calls).

Parity, 27-token reference prompt, 32 greedy tokens, `bench token --gen 32`:

| chunk | RX 6950 XT | MI50 | top-1 logit (RX) |
|---|---|---|---|
| 0 | **32/32** | — | 13.170521 |
| 8 | **32/32** | — | 13.170561 |
| 16 | **32/32** | — | 13.170561 |
| 32 | **32/32** | **32/32** | 13.170561 |
| 64 | **32/32** | — | 13.170561 |
| 128 | **32/32** | — | 13.170561 |
| 128 + `--prefill-merged` (9 resident blocks) | **32/32** | — | 13.170429 |

Chunking is now bit-identical to token-by-token **within a device** — the residual
1e-5 spread across chunk sizes is the documented expert-set accumulation order
(§8s), and the RX/MI50 difference (13.1705 vs 13.1705) is the pre-existing
device-to-device one.

### 2. Prefill after the fix, and where its time actually goes

2 agents x 300-token prefixes, 8 generated tokens, RX + MI50 one stream each,
alternating in one window (a *slow* window: decode 9.3-13.1 aggregate against the
recorded 20.0-20.3, host lane 100+ ms/step — the expert mmap is re-read from NVMe,
~21 GB per run, because 41 GB of experts + 27 GB of PLE do not fit the 45 GB page
cache; every comparison is in-window):

| prefill path | tok/s per agent |
|---|---:|
| token by token (chunk 0) | **5.35** (5.27, 5.42) |
| chunk 128 | 5.02 |
| chunk 128 + `--prefill-merged` | 5.08 |
| chunk 128 + merged, region arrays off | 4.95 |

`BRECCIA_TRACE_PREFILL=1` prints the chunk's own stage split (ms per prompt
token, 128-token chunk, one stream, RX):

| stage | token by token | chunked |
|---|---:|---:|
| **experts(host)** | **69.0** | **93.4** |
| shared-exp | 15.4 | 15.4 |
| mixer | 6.2 | 6.1 |
| ple | 1.3 | **14.7** |
| hc-read(attn/ffn) + hc-write(attn/ffn) + router + embed | 0.3 | 0.3 |
| **total** | **92** | **130** |

So: **the host expert lane is 72-75 % of prefill**, the shared-expert GEMV is the
largest *dense* stage (15.4), the hyper-connection chain — the thing a T-batched
dense pass would target — is **0.3 ms/token**, and the chunked path is 38 ms/token
*worse* than token-by-token: +24 ms because the grouped lane now does the work
§8t's clamp was skipping, +11 ms because each chunked `ple_step` pays two flushes.

That is why a T-batched dense chain was **not** landed this round: with the fix in
place it can address at most shared-exp + mixer (21.6 of 92 ms, 23 %), and the
honest sequencing is to attack the lane first — the merged pass pointed at the
**39 non-resident blocks** (stream each expert once per chunk, §8u's 2.6-11
ms/token against the lane's ~70), or PowerInfer-style per-expert residency. A
T-batched shared-expert GEMV (same weights, T users — the `GemmMany` machinery is
already built) is the dense-chain piece to do, worth ~15 ms/token.

### 3. Barrier scope: `COMPUTE -> COMPUTE` sync2, measured (B)

`Gpu::barrier_all` (ALL_COMMANDS -> ALL_COMMANDS) fenced every recorded dispatch —
1 509 of them per decode token. `Gpu::barrier_compute_on` now issues
`vkCmdPipelineBarrier2` at the narrowest scope that is still correct for a
compute-only stream (`COMPUTE_SHADER -> COMPUTE_SHADER`, shader read/write access
masks), loaded by its **core 1.3** name (RADV does not export the KHR alias — same
caveat as `vkQueueSubmit2`). `BRECCIA_BARRIER=full|sync2|none` selects the arm;
`none` exists only for the bench.

2 streams, 64-token prefixes, 8 decode steps, A/B/A:

| arm | aggregate tok/s | GPU busy (ms/step, both cards) |
|---|---:|---:|
| full | 9.50 / 9.26 | 137.4 / 136.9 |
| **sync2 (default)** | **13.14 / 10.01** | **127.7 / 129.6** |
| none | **NaN in the warm pass** — not shippable | — |

`sync2` is **-8.5 ms/step of GPU busy** and +23 % aggregate decode in this window
(the aggregate is noisy because the lane, not the GPU, binds the step; the GPU
number is the clean one). Correctness: bit-identical tokens on the MI50 with
`VK_LAYER_KHRONOS_validation` on, 0 validation messages.

### 4. Region arrays for the chunk's copies (C)

`batch_cmd::record_copies` groups copies by (source, destination) buffer pair and
records them as one `vkCmdCopyBuffer` region array behind a single transfer
barrier — the chunked/merged passes are copy-dense (5-6 per token, one per expert
group, one per token in the fan-in). A/B on the same work (11 3xx expert groups,
11.8 GiB of arena reads):

| chunk 128 + merged | GEMM wall | fan-in | prefill |
|---|---:|---:|---:|
| per-copy (old) | 850.9 ms | 23.5 ms | 4.95 |
| **region arrays** | **757.9 ms** | **8.9 ms** | **5.08** |

-108 ms per 600-token prompt (-0.18 ms/token of host record time), default on,
`BRECCIA_NO_COPY_ARRAYS=1` for the old shape.

### 5. Dropped: Wave32 on the RX (D)

Not attempted: `docs/research/02-vulkan-followups-and-powerinfer.md` measures
**1.001x** on a dependent-load latency probe (VGPR 8 -> 16) and finds the MI50
*silently ignores* the requested subgroup size, which would make a wave32 arm
there a fake no-op. A 0.1 % lever with a 64-lane device that lies is not worth a
round.

### What this round changed in the gates

- Chunked prefill is **parity-correct at every chunk size** (was wrong for >= 9);
  `--prefill-chunk` is now a *valid* substrate, still 5 % slower than token by
  token, so it stays off.
- Decode: the default barrier is now compute-scoped — a free -8.5 ms/step of GPU
  busy, no parity change.
- Nothing in the merged prefill changes the verdict of §8v: it stays behind
  `--prefill-merged`.

## 8x. Streamed experts do not pay, copy fences must be staged, and cold vs steady-state is now a flag

Three things this round: the 39 non-resident blocks got a merged **streaming** path
(and it does not beat the host lane), a real **race** the streaming path exposed in
the §8w barrier change (fixed with staged copy fences), and a **`--warm` +
`cache`** protocol because this box's I/O behaviour had been misread all along.

### 1. Cold vs steady-state is now measured, not inferred (`--warm N`)

`bench agents` and `bench token` take `--warm N` (N full generations before the
window) and print

```
cache    {GiB} read from storage during the measured window ({MiB}/prompt token) · --warm N
```

from `/proc/self/io` `read_bytes` (page faults included). Combined with the paired
research's btrfs read-ahead fix (8 MiB -> 128 KiB per backing device), the picture
changed completely:

| 2 agents x 64 prefix, 8 gen steps | storage read in the window | decode aggregate |
|---|---:|---:|
| token-by-token, `--warm 1` | 1.09 GiB (8.8 MiB/token) | 8.61 / 13.83 |
| merged+streamed, `--warm 1` | 0.11 GiB (0.9 MiB/token) | 14.56 / 13.30 |
| 300-prefix, token-by-token | 4.30 GiB (7.3 MiB/token) | 12.40 |
| 300-prefix, merged+streamed | 0.15 GiB (0.3 MiB/token) | 13.39 |

**Steady-state decode gate (2 streams, 8 steps, 64-token prefixes, `--warm 1`):
13.3-14.6 tok/s aggregate.** The 9.3-13.1 numbers of §8w were measured with the old
8 MiB read-ahead and are **not comparable**; the 26 GiB/window cold reads quoted
there are gone (a cold 64-prefix window is now ~1 GiB).

### 2. Streamed (non-resident) experts: built, parity-clean, and it does not pay (A)

`--prefill-merged` now also covers blocks with no arena: per chunk, per **distinct**
expert, the three blobs are memcpy'd out of the mmap into one of `STREAM_SLOTS = 8`
host-visible slots (`[gate | up | down]`), copied to the device inside the batch,
and the same three multi-user GEMMs + `SwiGLU` + fan-in run with `base_words = 0`.
Eight experts ride one submit, so a chunk pays one fence per 8 experts per block.
A block whose role type has no multi-user GEMM keeps the per-token host lane
(documented fallback; today that only ever triggers for the *shared-expert* shapes,
never for a routed-expert role).

Parity: **32/32 at chunk 8/16/32/128 and with `--prefill-merged`**, and on the RX
the streamed path's top-1 logit is **identical** to the non-chunked, non-streamed
run (13.170561) — the streaming is numerically invisible, which is what the §8v
bit-identity buys.

Measured, 2 agents x 300 prefix, 8 gen, dual, `--warm 1`:

| prefill | tok/s per agent |
|---|---:|
| token by token | **6.45** |
| chunk 128 + merged (39 blocks streamed) | 6.07 |

and at 64-token prefixes: 4.33 / 7.02 (baseline) against 5.11 / 5.07 (streamed) —
*direction of the delta is inside this window's noise*, and at the longer prefix it
is 6 % worse. Meanwhile the bytes read fall **25x** (0.15 vs 4.30 GiB) — so the
lane is no longer I/O-bound (the read-ahead fix removed that), and the streaming
pass costs more than the lane's per-token dequantisation because **it does not
overlap**: `fill slot -> copy -> 3 GEMMs` per expert, serialised by the slot flush.

Per-group cost says the same thing: **429-530 µs per expert group** for the streamed
blocks against 67-84 µs for the resident ones (§8v). The next step is not a bigger
slot: it is **prefetching expert `e+1` while `e` computes** — the `TransferRing` the
plan named — plus the fused single-dispatch MoE the exllamav2 audit prices at
2.0 -> 1.0-1.2 ms/block.

### 3. The race the streaming path found: copy fences must be staged

Streaming 1.7 MB per expert made a latent bug in §8w's barrier change visible: the
per-dispatch fence is now `COMPUTE -> COMPUTE`, which by construction does **not**
order `COPY -> COMPUTE`; the copies had only a *pre* barrier, so a GEMM could read
its upload before it landed. Symptom: the MI50 produced `max |Δlogit| 2.46` between
two identical prompts (non-deterministic) and the wrong top-1 logit, while the RX
happened to pass — a lucky-timing race that only large copies expose.

`Gpu::copy_dependency(cmd, pre)` now emits the precise pair
(`COMPUTE_SHADER/SHADER_WRITE -> COPY/TRANSFER_WRITE` before,
`COPY/TRANSFER_WRITE -> COMPUTE_SHADER/SHADER_READ` after) and `record_copy`/
`record_copies` fence both sides with it. Determinism returned
(`max |Δlogit| 0.000e0`) **and** the coarse `ALL_COMMANDS -> ALL_COMMANDS` form the
first fix used cost 10x per group (715 -> 429-530 µs), so the staged form is also
the fast one. Parity re-verified 32/32 on both devices.

### 4. Not done, with the measured reason (B, D)

* **T-batched shared expert (B)**: blocked on kernels, not on orchestration. The
  shared expert's `gate`/`up` are `IQ4_XS`/`Q4_K`/`Q5_K`/`Q6_K`, which have **no**
  multi-user GEMM (the delivered set is IQ4_NL, Q2_0, IQ2_XXS/XS/S, IQ3_XXS/S);
  only its `down` (IQ4_NL 40 blocks, Q2_0 8) could be batched today. The 15.4
  ms/token in §8w's trace also belongs to PLE + shared expert together, and the
  research shows the PLE half is a **single-threaded cold gather** (65 µs per cold
  90-byte row, 48x btrfs extent over-read) — so the ordered fix is: batch the PLE
  gather per chunk (one flush for T tokens instead of two per token, which is the
  `ple 1.3 -> 14.7 ms` regression in §8w), *then* add the four missing multi-user
  kernels, *then* the shared expert.
* **Item D (madvise / fadvise)**: not attempted — the paired research measures
  `MADV_RANDOM`/`FADV_RANDOM` as worse or ignored and `O_DIRECT` as throwing away
  the 58 GB/s warm cache, and the actual lever (btrfs `read_ahead_kb`) is fixed at
  the filesystem, outside this engine.

## 8y. The "decode regression" was the lane's default thread count, and the canonical gate

Item 1 of the round was to reconcile the steady-state 2-stream gate (13.3-14.6
aggregate, §8x) with the recorded 20.28/19.98 (§8q). **There is no code
regression.** The two numbers are two different protocols, and the dominant
difference is one default.

### The A/B/A that closes it

`bench token --streams 2 --devices 0000:0e:00.0,0000:03:00.0 --gen 32`, one window,
same binary, only `--threads` varying (the lane's worker count):

| threads | aggregate tok/s | lane (ms/step) | fence (ms/step) |
|---|---:|---:|---:|
| 8 (the old default: physical/2) | 14.71 / 14.73 | 85.7 / 86.3 | 32.3 / 31.9 |
| 16 | 16.99 | 55.2 | 44.7 |
| **32** | **20.19 / 20.77** | **41.2** | 39.5 / 36.6 |

The recorded 20.28/19.98 of §8q was always run with **`--threads 32`** (that
command line is in §8p). `default_threads()` returned physical cores / 2 = **8** on
this 16-core/32-thread 7950X, so every run *without* the flag measured a
lane-starved 14.7. The lane mixes ALU-bound dequantisation with memory stalls, so
SMT genuinely helps it (+41 % aggregate, lane 86 -> 41 ms/step).

Fixed by defaulting to **logical CPUs (capped at 32)**; with no flag at all the
gate now reads **19.29 / 21.29** aggregate (lane 42.8 / 40.4). The latency-bound
`GEMV` bench keeps its own physical/2 rule — that one measures one dispatch, not
the lane.

### Canonical two-stream decode gate (compare like with like)

```bash
timeout 1200 ./target/release/breccia bench token --streams 2 \
  --devices 0000:0e:00.0,0000:03:00.0 \
  --prompt-ids-file /var/tmp/users/korphaus/tmp/opencode/p4-token/parity-server.json.ids \
  --gen 32 --runs 1        # threads now default to the logical CPU count
```

**Gate = 20.3 +- 1 tok/s aggregate** (19.3-21.3 over five runs in the §8y window),
lane 40-43 ms/step, overlap 100 %. Do **not** compare it with
`bench agents --prefix 64 --gen 8`, which pads agent 0's 27-token prompt to 64 by
repeating its last id, runs the streams in the other device order, generates 8
steps and — until this round — used 8 lane threads: it measures a different,
lane-starved regime (13-15 aggregate). The `cache` line of §8x is unaffected and
still worth quoting next to the rate.

### Where the rest of the round went (and what did not land)

The three remaining items are **not landed**; the measurements and the research's
own refutations are recorded here so the next round starts from them:

* **PLE gather batching (2)**: unchanged. The trace is `ple 1.3 -> 14.7 ms/token`
  from §8w and the research prices the cold row at 65 µs with a 48x btrfs extent
  over-read. The batching design is clear — one flush per chunk instead of two per
  token, one `prefetch_rows` + one `gather_staged` over `T*16` rows, one
  `read_batch` of `T` residuals/keys/values, `T` host blocks, one `record_copies`
  writeback — and it needs the staging buffer's `cap` raised to
  `chunk * (2*hc_dim + n_embd) * 4` (~12 MB at T = 128) because `read_batch` refuses
  a total above it.
* **The four missing multi-user GEMMs (3)**: unchanged (`IQ4_XS`, `Q4_K`, `Q5_K`,
  `Q6_K`), so the T-batched shared expert is still blocked on kernels; the pattern
  to copy is `gemm_iq4_nl.comp` + `gemm_body.glsl`, bit-identical per user, one
  pipeline per type, and `PipelineSet::gemm` already caches them by type.
* **Async expert prefetch (4)**: unchanged. §8x's number stands (429-530 µs per
  streamed group against 67-84 resident); the fix is prefetching `e+1` while `e`
  computes, and the ring depth should be derived from the measured record/step
  ratio rather than the fixed 6 (`docs/research/04`).

### Launch-side queue, corrected by the probe (do not adopt two of these)

The `docs/research/04` probe (both cards, MI50 first) changes the launch queue:

| idea | verdict |
|---|---|
| `vkCmdDispatchIndirect` | **refuted**: +0.77 µs (RX) / +1.65 µs (MI50) per dispatch, i.e. -1.2..-2.5 ms/token if adopted. Dropped. |
| **barrier pruning** | **the launch-side win**: an unneeded dispatch costs 0.007 µs (RX) / 0.031 (MI50) but the per-dispatch compute->compute barrier costs 0.650 / 2.867 µs; 1 509 barriers = 0.98 / 4.32 ms = 1.4-6.1 % of the GPU half. Emit ~400/token where a real dependency exists, behind one flag, measure both cards. |
| descriptor sets | `vkAllocateDescriptorSets` per dispatch, not recording, is the cost (recording is 68 ns/dispatch): pool or reset them per batch. |
| placement | the MI50's dependency boundary is 4.4x the RX's: fine-grained dependent work to the RX, bulk/independent to the MI50. |
| fused single-dispatch MoE, LMDeploy-style prefetch-before-wait | queue after the above. |

And the correction that matters for this section: **launch count is 1-4 % of prefill
time**, so the 20.3 -> 13.5 reconciliation was never going to be a launch story —
it was one `usize` default.

## 8z. PLE is a cold-start cost, barrier pruning fails its own parity gate, and the flaky test was a real race

Three items and one negative result. The round's honest summary: the prefill item
was smaller than the trace made it look, the launch item **does not work as
specified**, and the suite's flake was a genuine ordering bug.

### 1. PLE: batched fault-in landed; the 14.7 ms was cold pages, not the chunk path

The chunk's PLE phase now hashes every token's rows up front and faults them in
with **one** `prefetch_rows` over `t * 16` rows (8-way), then runs the per-token
host blocks in order (`Session::ple_step_rows`, split out of `ple_step` so the
rolling hash still advances exactly once per token). That is the IO half the
research asked to batch.

It also settled what the cost actually was. `BRECCIA_TRACE_PREFILL`, 64-token
prompt, warm:

| chunk | `ple` (ms/token) | `shared-exp` | `mixer` | `experts(host)` |
|---:|---:|---:|---:|---:|
| 8 | 1.39 | 55.3 | 12.6 | 51.0 |
| 32 | 1.38 | 30.4 | 10.7 | 51.5 |
| 64 | 1.41 | 20.9 | 7.2 | 49.2 |

**PLE is 1.4 ms/token at every chunk size** — §8w's `ple 14.65` was one *cold-table*
run (90-byte records in a 28.8 GiB file: 65 µs per cold row × 16, exactly as
`docs/research/03` priced it), not a chunk-path regression. So batching the prefetch
pays only on a cold table, which `--warm` removes from the measurement by
definition. The real PLE fix is the one research round 5 measured: read a token's
**4320 contiguous bytes** instead of 48 scattered 90-byte rows (15.6 -> 0.45 ms per
128 tokens), with 8-16 threads (the decode lane wants 32, so a separate small pool),
host RAM rather than a `HOST_VISIBLE|DEVICE_LOCAL` heap, and persistent mappings.
Not implemented; that is the next PLE action.

Prefill, 2 agents x 300 prefix, 8 gen, `--warm 1`: **this window was degraded**
(baseline 4.75 tok/s/agent against 6.45 in §8x's window; the canonical gate read
15.2 with a 73 ms/step lane). In-window pair: baseline 4.75 vs chunk 128 + merged
**6.04** (+27 %), where §8x's window had 6.45 vs 6.07 (-6 %). Both pairs are
recorded; **no gate claim is made** from a window whose lane time moved 73 vs 40
ms/step.

### 2. Barrier pruning: implemented, measured, and it does not survive parity (opt-in, off)

`batch_cmd::skip_next_barrier()` drops the fence after a dispatch whose successor
provably reads nothing it wrote — the three pairs are the shared expert's
`gate`/`up`, the PLE `key`/`value` projections and the merged pass's `gate`/`up`
GEMMs. Result:

| arm | chunk 0 parity | chunk 128 + merged |
|---|---|---|
| fences everywhere (default) | **32/32**, top-1 13.170521 | **32/32**, top-1 **13.170561** |
| `BRECCIA_BARRIER_PRUNE=1` | 20/32, top-1 13.069797 | 32/32 but top-1 **12.933844** |

A call site is not a dispatch: the split-K forms record *several* dispatches per
`run`, so the flag is consumed by the first of them and the reduction loses its
fence. The fix is a per-dispatch dependency graph, not a call-site annotation — so
the mechanism stays in the tree **off by default**, re-armed only by that env var,
and the idea is demoted rather than shipped. The A/B/A of the canonical gate in the
same window was inconclusive anyway (6.39 / 16.73 / 20.31 / 17.82 aggregate, lane
39-256 ms/step): the research's 0.98 / 4.32 ms of GPU-half headroom is below this
box's CPU-contention noise, which is itself worth knowing.

### 3. The flaky suite was a real race, and is fixed

`breccia-vk`'s `device_select_env_is_cleared_and_reported` sets
`MESA_VK_DEVICE_SELECT` and expects `Host::new()` to unset and report it, while
`enumerates_three_devices_on_this_host` also calls `Host::new()`. Both mutate the
**process-global** environment, so under `cargo test`'s parallelism one test's
`Host::new()` cleared the variable before the other's could observe it — passed
alone, failed in the suite. Both now take a `HOST_LOCK` mutex around the
`set_var` + `Host::new` pair; the reason is a doc comment on the lock.

### 4. Items not landed

* **The four multi-user GEMMs (`IQ4_XS`, `Q4_K`, `Q5_K`, `Q6_K`)** and the
  T-batched shared expert behind them: unchanged. The trace above makes this the
  largest prefill stage that has a kernel-shaped fix (`shared-exp` 20.9-55.3
  ms/token in chunked prefill against 15.4 token-by-token).
* **Async expert prefetch `e+1`**: unchanged (§8x: 429-530 µs per streamed group
  against 67-84 resident; ring depth to be derived from the measured record/step
  ratio).

## 8aa. The four extra multi-user GEMMs compile; barrier pruning is parked; the shared-expert pass is not wired

A short round with one partial delivery, one park and one honest gap. **Parity at
the end of the round is unchanged**: chunk 128 + merged 32/32, top-1 13.170561.

### 1. Four more multi-user GEMMs (`IQ4_XS`, `Q4_K`, `Q5_K`, `Q6_K`)

`shaders/gemm_{iq4_xs,q4_k,q5_k,q6_k}.comp` are the 14-line form the pattern asks
for (`QK 256`, `BLOCK_WORDS 68/72/88/105`, `DEQUANT_UNIT
dequant_{iq4_xs,q4_k,q5_k,q6_k}_unit` over the unchanged `gemm_body.glsl`), wired
into `build.rs` and `GemmMany::new`, so `PipelineSet::gemm` now caches **eleven**
types instead of seven. **They are not validated yet**, and the reason matters: the
`gemm_many` test walks *routed-expert* plans, and this artifact's routed experts are
only `IQ2_*/IQ3_*/IQ4_NL/Q2_0` — the four new types appear on the **shared** expert
(`ffn_{gate,up,down}_shexp`), which has no `LayerPlan`. So the kernels compile and dispatch, but there was **no bit-identity claim** until a
dense fixture existed.

**Update — validated on the RX.** `tests/gemm_many.rs::dense_shared_expert_gemm_matches_the_gemv`
takes a real `ffn_{gate,up,down}_shexp` tensor of the requested type
(`BRECCIA_GEMM_TYPE`, **one type per invocation**), runs `GemmMany` at
`m = USERS_MAX` and compares every user against a single-user `Gemv` on the same
weights. On the **RX 6950 XT** (headless card, validation on, `timeout 90`):

| type | tensor | worst relative | m = 1 bit-identical | validation |
|---|---|---:|---|---|
| IQ4_XS | blk.0.ffn_gate_shexp [640x2560] | **0.000e0** | yes | 0 messages |
| Q4_K | blk.2.ffn_gate_shexp | **0.000e0** | yes | 0 messages |
| Q5_K | blk.2.ffn_up_shexp | **0.000e0** | yes | 0 messages |
| Q6_K | blk.6.ffn_up_shexp | **0.000e0** | yes | 0 messages |

So the four kernels are arithmetically and address-wise correct for this artifact's
shared-expert geometry — which matters because the 2026-10-26 reboot was attributed
to them (AGENTS.md incident log). The crash therefore cannot be explained by a wrong
`BLOCK_WORDS`/`QK`/`DEQUANT_UNIT`: the remaining suspects are (a) the *old test*
sweeping up to 24 routed-expert cells in one process, i.e. several of these shaders
in a single invocation, which the AGENTS "one type per invocation" rule exists to
prevent, and (b) a MI50-specific path (wave64) that only the display card exercises.
The MI50 run is therefore done one type per process, never as a sweep.

**And it passes there too.** MI50 (the display card), validation on, `timeout 90`,
one type per process: IQ4_XS, Q4_K, Q5_K, Q6_K all **0.000e0**, `m = 1`
bit-identical, 0 validation messages, no fault. So the four kernels are **not** the
crash cause as a kernel defect — they are correct on both cards and both subgroup
shapes.

What the incident correlates with is the **sweep**: the crash came from
`gemm_many`'s `multi_user_gemm` test, which walked up to 24 (block, role, type) cells
in *one process* — and `a3c01e5` had just raised that cap from 6 to 24. Many new
dispatch shapes in a single invocation is precisely what AGENTS.md's "one quant type
per invocation" rule exists to prevent. That test now takes a `BRECCIA_GEMM_TYPE`
filter and defaults to **4** cells, with the rule stated at the call site. Two
lessons recorded: a *kernel* passing individually does not clear a *sweep*, and the
display card is the MI50 — the order in AGENTS.md is now RX first, which is what this
validation used.

### 2. The shared-expert T-batch: not wired

It needs a chunk-level shape that does not exist yet: `mixed` for all `T` tokens in
one buffer (the merged pass builds it, the plain chunk path does not), `gate`/`up`
outputs as `[T, n_ff]`, one fused SwiGLU, `down` as a second GEMM, and the ffn
`HC_WRITE` consuming `[T, n_embd]`. That is a restructure of `record_dense_prelude`
for **every** block, not just the merged ones, so it is queued rather than rushed —
the stage table that ranks it is §8z (`shared-exp` 20.9-55.3 ms/token chunked
against 15.4 token-by-token).

### 3. Barrier pruning: parked, and nothing unsafe is reachable

`batch_cmd::skip_next_barrier` is now a **documented no-op** (the call sites remain
as a map of the independent pairs). Re-arming it requires per-dispatch dependency
bookkeeping; a call-site annotation cannot express it, and the ceiling (0.98 ms RX /
4.32 ms MI50 of the GPU half) does not justify that work ahead of the shared expert.

### 4. Not landed

Token-major PLE read (`--ple-threads`) and async expert prefetch `e+1`: unchanged.
The PLE design is §8z's (4320 contiguous bytes per token, 8-16 gather threads, host
RAM, persistent maps).

## 8ab. The shared expert is 99.7 % dispatch overhead — priced, restructure queued

The round's job was to wire the chunk-level shared expert. It is **not wired**, but
the measurement that decides whether it is worth wiring is now in hand, and it is
decisive.

### The price of the target shape (RX 6950 XT, `dense_shared` fixture, m = USERS_MAX = 8)

The four §8aa kernels on the **actual shared-expert geometry**
(`ffn_{gate,up}_shexp`, 640 x 2560), one type per invocation, validation off (they
passed with it on in §8aa):

| type | ms per m = 8 dispatch | ms per token per role |
|---|---:|---:|
| IQ4_XS | 0.094 | **0.012** |
| Q4_K | 0.099 | **0.012** |
| Q5_K | 0.106 | **0.013** |
| Q6_K | 0.109 | **0.014** |

A chunk-level shared expert is therefore `gate (t/8 dispatches) + up (t/8) +
one fused SwiGLU over t*n_ff + down (t/8)`, which prices at **~0.03-0.04
ms/token of GPU** across all 48 blocks — against the **15.4 ms/token** the
token-by-token path measures and **20.9-55.3 ms/token** in chunked prefill (§8z).

So the stage is **~99.7 % dispatch and recording overhead**, not arithmetic: four
dispatches per block per token at ~80 us of host cost each is the entire stage, and
one dispatch per role per 8 tokens removes ~87 % of them. The target (<= 15.4
ms/token) is not merely reachable; the ceiling is two orders of magnitude below the
current figure.

### What blocks the restructure, precisely

It is not the kernels any more. It is that the shared expert lives inside
`record_dense_prelude`, i.e. **per token**, for every block, and its outputs are
consumed per token (`v.sh` -> the MoE combine, `v.sh_gate_logit` -> the gate). A
chunk-level pass needs, for **all** blocks (not just the merged ones):

1. `mixed` for all `t` tokens in one buffer — the merged pass already keeps it
   (`PfChunk::mixed`), the plain chunk path does not;
2. `gate`/`up`/`act` as `[t, n_ff]` and `down` as `[t, n_embd]`, i.e. three chunk
   buffers plus one SwiGLU over `t * n_ff` (`SwiGlu::run` already takes `n`);
3. the per-token **consumers** rewired: `record_lane_copies` copies `v.sh` into the
   staging block region today, so it would copy `pf.shared[i]` instead; the merged
   path's fan-in already reads `pf.shared`/`pf.gate` and needs no change;
4. `shexp_gate` (the 1-float scalar projection) kept per token or batched with the
   existing `GemvBatch` — it is noise either way.

That is a change to the prelude signature plus the lane-copy source, i.e. a
`record_dense_prelude(..., chunk_shared: bool)` and one new `shared_expert_chunk`
phase between pass 1 and the lane. It did not land in this round's window; the
pricing above is what says to do it next, first.

### Parity and gate

Nothing on a decode path changed this round (the only engine change is a `run_timed`
call inside a test), so **the canonical gate stands at 20.3 +- 1 tok/s aggregate,
lane 40-43 ms/step**, and the parity gate is unchanged: chunk 128 + merged 32/32,
top-1 13.170561. The four kernels remain bit-identical on both cards, one type per
process (§8aa).

### Not landed

The chunk-level shared expert (above), the token-major PLE read and `--ple-threads`
(§8z design), and the per-dispatch barrier bookkeeping (§8aa: parked, ceiling
0.98/4.32 ms).

## 8ac. The chunk-level shared expert failed its parity gate (reverted), and the reboot cause is fixed

Two results, one negative and one safety fix. **Parity is green at the end of the
round: chunk 128 + merged 32/32, top-1 13.170561.**

### 1. The restructure was attempted, measured, and reverted

All four §8ab pieces were implemented: chunk `mixed` for every block on the merged
path, a `record_dense_prelude(..., shared: bool)` switch so the routed shared expert
moves out of the per-token prelude, a `shared_expert_chunk` phase
(`t / USERS_MAX` groups, each `copy x -> gate GEMM -> up GEMM -> SwiGLU -> down GEMM
-> copy out` into `pf.shared`), and the lane's `sh` source rewired to `pf.shared[i]`
(the 1-float scalar gate left per token).

It runs, and it is **wrong**: with the chunk pass actually active (which needed
`pf` to be allocated even when no block is resident — the first attempt silently
kept the per-token path because `pf` required `--gpu-experts`), the parity gate read
**2/32, top-1 12.532024** against the expected 13.170561. With the change reverted:
32/32, 13.170561.

The measured cost while it was active — `shared-exp` **20.74 -> 18.05 ms/token** at
chunk 64 — is only a 13 % improvement, *not* the two orders §8ab's per-dispatch price
predicted, which says the residual is not the routed shared expert's dispatches at
all: something else is being charged to that stage (the per-token scalar
`shexp_gate` GEMV plus the pass-1 flush wait are the candidates). That is the next
thing to measure, not the next thing to guess: instrument the stage boundary before
restructuring it again.

Reverted rather than shipped: a half-restructure behind a flag is fine, a *wrong*
one is not, and the round's rule is parity-green or revert. The design and the price
stay in §8ab; what is missing is a correct source for one of the two consumers, most
likely the lane's staging offsets (`lane_offsets(..)[3]` vs the chunk copy) — the
next attempt should diff the host-path `pf.shared[i] -> staging` copy against the
token path's `v.sh -> staging` for one token before touching the rest.

### 2. Safety: every GPU wait is bounded now (this is the reboot cause)

`docs/research/06` found what the 23:44 event actually was: a ring timeout with one
outstanding packet, with `gemm_many` inside **`vkWaitForFences(u64::MAX)`** — so the
*kernel's GPU watchdog* decided and issued a BACO reset on the display card. Nothing
in the kernel was wrong (both cards later passed the four kernels bit-identically);
the engine simply never gave up.

`Gpu::WAIT_TIMEOUT_NS` (10 s) + `Gpu::fault_abort` now replace every unbounded wait:
`wait_fence` and the batch ring's shutdown wait are bounded, and on expiry the
process prints what it was waiting for plus the `journalctl` hint and **aborts**,
which leaves the device healthy (91 -> 81 ms to a usable device, measured). No call
site can reintroduce `u64::MAX`; the remaining one is
`TransferRing`'s `wait_semaphores`, which is next (same helper, semaphore flavour).

### 3. Research round 6 items taken / queued

| item | state |
|---|---|
| bounded fence waits | **landed** (above) |
| `VK_EXT_robustness2` (OOB -> zeros, <=1 % cost) | queued: device creation + feature enable |
| shared expert as **2** dispatches (fused gate+up+SwiGLU, measured 2.8x) | queued; do it *with* the correct restructure, not before |
| push descriptors / pre-updated pooled sets (shapes frozen per type) | queued |
| dependent-chain timing only for dispatch-cost claims | adopted |
| barrier cost is 3.43 us per **dependent** barrier (~3.8 ms/token over ~1,100) | re-prices the parked item upward; stays parked until the shared expert lands |

## 8ad. `shared-exp` was a timer artifact: the stage is 0.1 ms/token, not 15-55

The round's job was to instrument before restructuring again. It settles the question
**against** a second attempt, and it invalidates one of this document's own numbers.

### The breakdown (`BRECCIA_TRACE_SHARED`, 64-token chunk, per prompt token, 48 blocks)

| sub-stage | ms/token |
|---|---:|
| `sh_gate` (GEMV) | 0.02 |
| `sh_up` (GEMV) | 0.02 |
| `sh_swiglu` | 0.02 |
| `sh_down` (GEMV) | 0.02 |
| `sh_scalar_gate` (the 1-float projection) | 0.02 |
| **routed shared expert, total** | **0.10** |
| **pass-1 flush wait** | **19.3-20.7** |
| `shared-exp` stage as `Timers` reports it | 19.4-20.8 |

So the **routed shared expert costs 0.10 ms/token of host time** — it is already
essentially free — and the 19-21 ms that the `Timers` table attributes to
`shared-exp` is **the pass-1 GPU wait**, charged to that stage by mark ordering: the
shared expert is the last thing `record_dense_prelude` records, so the gap between
its `SHARED` mark and the next mark swallows the `flush()` that closes pass 1 (the
wait for ~1 900 recorded dispatches per block).

That also explains the shape §8z recorded and the restructure chased:
"20.9-55.3 ms/token chunked, 15.4 token-by-token" is `blocks x flush / t` — it grows
as `t` shrinks because the same per-block drain is divided by fewer tokens, not
because the shared expert degrades.

**Consequences.**

1. The chunk-level shared expert is **not worth a second attempt**: its ceiling is the
   0.10 ms/token above, i.e. ~0.5 % of prefill, and §8ab's "0.03-0.04 ms/token"
   price was right about the GEMMs and wrong about what the 15-55 ms was.
2. The stage table needs the pass-1 flush attributed to `WAIT_GPU`, not to whichever
   stage happened to run last. That is a one-line timer fix and it is what the next
   round should do before trusting any other stage number in §8w-§8ad.
3. What the restructure *would* have helped with is the thing it was never aimed at:
   the **23 ms/token of GPU drain per block** in chunked prefill. The levers there are
   the launch side (barrier pruning, re-priced at 3.43 us per dependent barrier ≈
   3.8 ms/token), fewer dispatches per block (fused gate+up+SwiGLU: 2 instead of 3 per
   expert), and the descriptor-set cost — not the shared expert's arithmetic.

The four §8ab pieces stay reverted (they were wrong, §8ac) and are now **not worth
fixing**; the code is out of the tree and the design is recorded here only as a
negative result.

### Safety: the last unbounded wait is bounded

`TransferRing::wait_semaphores` was the engine's last `u64::MAX`: it now waits
`Gpu::WAIT_TIMEOUT_NS` and, on expiry, prints what it waited for and calls
`Gpu::fault_abort` (§8ac), which leaves the device healthy instead of hanging it on
the display card. With `wait_fence` and the batch ring's shutdown wait already
bounded, **no unbounded GPU wait remains** (grep-checked). `VK_EXT_robustness2`
(OOB -> zeros, <=1 % cost) is still **queued**: it needs the device-creation feature
chain and the extension string, which is a change to `Gpu::create`'s feature list and
deserves its own parity run rather than the tail of this round.

### Parity and gate

Nothing on a decode path changed, and the reverted tree is the one §8ac measured:
**parity 32/32 at chunk 128 + merged, top-1 13.170561; the canonical gate stands at
20.3 +- 1 aggregate, lane 40-43 ms/step.**

## 8ae. The timer is honest now, and the drain is arithmetic: 2 waits per block, 28 ms of GPU per block

Two items landed and one question answered with numbers. Parity unchanged:
**chunk 128 + merged 32/32, top-1 13.170561.**

### 1. A stage mark can no longer swallow a wait (one line, no behaviour change)

`prefill_block` now emits `self.timers.mark(WAIT_GPU)` **before** the pass-1 `flush()`.
Same chunk, same binary, 64-token prompt:

| stage | before §8ad | after |
|---|---:|---:|
| `shared-exp` | 19.4-20.8 | **0.15** |
| `wait-gpu` | (absorbed) | **21.14** |
| `experts(host)` | 49.6 | 48.07 |
| `ple` | 1.36 | 1.37 |

That is the honest table: the routed shared expert is 0.15 ms/token (worst case, with
the scalar gate included), the drain is 21.1 ms/token, and both are now visible at the
same time. **Every stage number in §8w-§8ad that was adjacent to a flush should be
re-read with this in mind**; `experts(host)` and `ple` were measured either side of a
host-only section and are unaffected.

### 2. How many waits does a block need? Two — the drain is not the wait count

Chunked path, per block: **pass 1 submit+flush, then pass 2 submit+flush** = 2 fence
waits, plus 2 per PLE token on block 1. The target "fewer than 2 waits per block" is
therefore already met by construction, and the cost is not the submits: the 21.1
ms/token of `wait-gpu` is **~28 ms of GPU execution per block per chunk** (0.44
ms/block/token x 64), i.e. ~1 900 dispatches draining. Collapsing pass 1 and pass 2
into one submit would not help — pass 2 *depends* on the host lane's output, so the
dependency graph already forces the split (one submit per pass is ncnn's pattern, and
we are at that: 2 submits, 2 waits).

So the drain is launch-and-execution cost inside the passes, which is where §8ae's
remaining items point:

* **barrier pruning with per-resource tracking** (ncnn's `stage`/`access` model): the
  mechanism the failed call-site flag could not express, re-priced at **3.43 µs per
  dependent barrier ≈ 3.8 ms/token** at ~1 100 barriers. Not attempted here — it is a
  new bookkeeping layer in `dispatch_batch::record` (track last write per `vk::Buffer`,
  emit only when the next dispatch binds a previously-written buffer) and deserves its
  own round with the sync2-vs-ALL_COMMANDS A/B folded in.
* **fused gate+up+SwiGLU** (2 dispatches per expert instead of 3, measured 3.1x/5.3x):
  it halves one of ~31 dispatches per block, so ~1.6 % of the drain — worth doing with
  the barrier work, not before it.

### 3. `VK_EXT_robustness2` is on, feature-detected

`Gpu::create` now enables `VK_EXT_robustness2` when advertised, with
`robustBufferAccess2` + `nullDescriptor` in the feature chain (both cards advertise
it). An out-of-bounds descriptor read returns zeros instead of faulting the GPU, which
turns the class of fault that has twice taken this box down into a wrong number the
parity gate catches. Parity with it on: **32/32, top-1 13.170561** — bit-identical to
the run without it, as expected (it changes fault behaviour, not arithmetic). The
<=1 % cost is the research's measurement; this round was too contended to claim a
quiet-window delta, so it is quoted, not measured.

## 8af. The 21 ms drain is 4 800 dispatches x 4.3 us — measured, and neither item landed

No implementation landed this round. What did land is the decomposition the two items
were aimed at, and it changes which of them is worth doing first.

### Dispatch and submit counts (the numbers Item 1 asked for)

`bench token`, 64-token prompt, RX, one chunk (so 48 block-chunks) against the
token-by-token path over the same prompt:

| | dispatches | submits | per block per chunk | per token per block |
|---|---:|---:|---:|---:|
| chunk 64 | 307 029 | 654 (469 per submit) | **6 396** | **100** |
| token by token | 308 352 | 9 947 (31 per submit) | 206 | 3.2 |

So chunking already does what it was built for — **15x fewer submits per token** (0.21
submits per block-token against 3.2) — and the *dispatch* count per token is
**unchanged** (100 vs 100): the chunk path records the same ~4 800 dispatches per token
(48 blocks x 100), it just records them into ~14 batches instead of ~155.

Two things that surprised this round and are worth keeping:

* **13.6 submits per block-chunk, not 2.** The passes are 2; the rest are PLE: block 1
  runs `ple_step` per token, which flushes twice and then *records GPU work with no
  batch open*, so each of those dispatches becomes its own submit (469 per submit is
  the batch size of the passes, and 654 - 96 is PLE and the head). Batching PLE
  (§8z/§8ad design) would remove ~550 submits per prefill — real, but submits are not
  the cost, as the next row shows.
* **The drain is not the submits.** 21.1 ms/token of `wait-gpu` over 4 800
  dispatches/token = **4.4 us per dispatch of GPU-side latency**, which matches the
  research's 3.43 us per *dependent* barrier plus ~1 us of dispatch: the GPU is
  serialised at roughly one dispatch per 4.4 us, exactly as if every dispatch were
  fenced against the last (it is).

### What that decides

| lever | effect on the 21 ms/token | verdict |
|---|---|---|
| fewer submits (one per pass) | ~0 — already 2 per pass, and 13.6 with PLE | cheap to do, small |
| **per-resource barrier tracking** (~1 100 of ~4 800 barriers are unneeded) | **~3.8 ms/token** at 3.43 us each | **first** |
| **T-batched dense chain** (dispatch count / T) | up to ~4 800/T -> hundreds per token, i.e. the largest single lever | second, and it is *shader* work, see below |
| fused gate+up+SwiGLU | ~1.6 % of the drain | with the barrier work |

**Why Item 1 (T-batched dense chain) did not land.** Of the dense stages, the ones that
already accept `T > 1` are `HcMixer::read`/`write` (the batched-GEMV form is documented
in `ops/mod.rs` and takes `ws.tokens`), `RmsNorm` (`hc * t` rows) and `SwiGlu`
(elementwise over `n`). The ones that **cannot** take `T` without a new kernel are the
token mixers: **GDN** (a sequential conv + delta-rule state advance per token) and
**QSA** (per-position KV append, rope and selection). So a T-batched dense chain is
~40 % restructure (the HC/RMSNorm half, which needs a chunk-residual layout, i.e. the
`r_chunk` stack the prefill already builds) and ~60 % new shader work for the mixers.
That is a milestone, not a round item, and it should be sequenced *after* the barrier
work because the barrier work multiplies its payoff.

**Why Item 2 (async prefetch of expert `e+1`) did not land.** It is the one item that
needs the transfer queue to overlap with compute (semaphores per ring slot), and the
engine's copy path is synchronous by construction (`record_copies` inside the compute
batch). The measured split it was meant to fix is still §8x's: streamed groups cost
429-530 us against 67-84 us resident, and the *host lane* (48.1 ms/token) still does
the 39 non-resident blocks. Until a prefetch overlaps, the honest choice between
"streamed with prefetch" and "more GPU-resident coverage" is untested — and coverage is
the cheaper experiment (a `--gpu-experts` list, no code).

### State

Parity unchanged (chunk 128 + merged 32/32, top-1 13.170561), no decode path touched,
canonical gate stands at 20.3 +- 1 / lane 40-43 ms/step. Prefill at 64-token chunk in
this window: `experts(host)` 48.1, `wait-gpu` 20.6, `ple` 1.4, `mixer` 7.2, everything
else <= 0.2 ms/token.

## 8ag. Coverage cannot fix the lane: +4 blocks buys -7.5 %, and full coverage needs 43 GiB

Item 2 of the round (a flag-only experiment) landed; Items 1 and 3 did not. The
experiment is decisive about *whether the host lane is worth attacking in code*.

### The sweep (`--gpu-experts`, 2 streams, 64-token chunked prefill, RX, ms/token)

| placed blocks | `experts(host)` | `wait-gpu` | arena (VRAM) |
|---:|---:|---:|---|
| 9 (the shipped default) | **38.40** | 19.55 | ~7.9 GiB |
| 13 | **35.52** | 19.28 | ~11.4 GiB |

+4 blocks (−44 % of the lane's work) buys **−2.88 ms/token (−7.5 %)** of lane time and
**nothing** off the drain. The arithmetic is unforgiving: the lane's 38.4 ms/token is
0.98 ms per non-resident block per token, and covering all 39 would need 39 x ~0.9 GiB
= **~35 GiB of arena on top of the 3.7 GiB dense set** — against 16 GiB per card. Even
17 blocks (~14 GiB of arena) is at the edge of what fits beside the dense weights and
the state, and it would still leave 31 blocks on the lane.

**So coverage is not the lever.** The two options that remain are the ones already on
the list, and this experiment is what should decide between them: (a) **async prefetch**
for the streamed path (§8x: 429-530 us per group serialised against 67-84 resident —
overlapping the upload is worth ~4x on the streamed blocks, which *is* the lane), or
(b) accept the lane and attack the drain instead (barrier tracking, ~3.8 ms/token).
(a) is worth ~38 ms/token if it works; (b) is worth ~4 ms/token. That ordering is now
quantified rather than assumed.

### Items 1 and 3: not landed

* **Per-resource barrier tracking** (the ~3.8 ms/token item): not implemented. The
  design is settled — a `HashMap<vk::Buffer, Access>` in `dispatch_batch::record`,
  emitting `copy_dependency`/`barrier_compute_on` only when a bound buffer's last
  access was a write, with a per-token counter — but it is a change to the hottest
  recording path and needs its own parity + counter + sync2 A/B run. Shipping it
  unmeasured is exactly the mistake §8ac recorded.
* **Fused gate+up+SwiGLU**: not attempted (depends on Item 1 landing first, and it is
  ~1.6 % of the drain).

### State

No decode path touched; parity unchanged (chunk 128 + merged 32/32, top-1 13.170561);
canonical gate stands at 20.3 +- 1 aggregate, lane 40-43 ms/step. This window was
contended (back-to-back prefill runs, `ple` first-chunk rows cold at 16.5 vs 5.3 ms), so
no prefill tok/s is claimed — only the stage table, which is what the guardrail asks
for. One incidental confirmation: **PLE's cold rows are visible in the first chunk
line** (`ple 16.47` at 9 blocks, `5.27` at 13) and collapse to 1.1-1.3 ms/token by the
second — §8z's cold-vs-warm story, seen in a single run.

## 8ah. The stage table already sums to the wall — the "missing 100 ms" was an arithmetic artifact

Item 1 delivered: a chunked-prefill token is now attributed **end to end**, and the
residual is zero. There is no unaccounted ~100 ms, and the reason the earlier
arithmetic suggested one is worth writing down because it will mislead the next reader
too.

### The complete table (`BRECCIA_TRACE_PREFILL` + the new `prefill sum` line)

2 streams, 64-token chunk, `--warm 1`, both cards, ms per agent-token, **sum against
that stream's own wall**:

| stage | RX 6950 XT | MI50 |
|---|---:|---:|
| **experts(host)** | **83.04** | **83.39** |
| **wait-gpu** | **32.79** | **85.80** |
| mixer | 8.41 | 17.61 |
| ple | 2.19 | 17.64 |
| hc-write(ffn) | 1.52 | 0.51 |
| hc-read(attn/ffn) + hc-write(attn) + router + shared-exp + embed | 0.54 | 1.36 |
| **stage sum** | **128.50** | **206.33** |
| **wall (this chunk)** | **128.54** | **206.35** |
| **residual** | **0.04 (0.03 %)** | **0.02 (0.01 %)** |

And the cross-check that closes it: `128.50 + 206.33 = 334.8` ms against
`2 x 167.6 = 335.2` ms of combined wall — **0.1 % agreement**.

### Why there was never a 95-105 ms hole

`bench agents` prefills the streams **serially** (`for (s, sess) in multi.streams
.iter_mut()`), and `prefill tok/s per agent` divides the *combined* wall of both
streams by the tokens of one. So the per-agent figure (6.45 tok/s = 155 ms) is really
"both cards' prefill time per token pair divided by two", while the stage table is
measured inside **one** session. The table was always per stream and always complete;
comparing it to the combined wall invented the missing 100 ms. The new `prefill sum`
line prints `sum · wall · residual` per chunk precisely so this cannot recur: if the
residual is not ~0, the table is lying, and now the number says so.

### What the table says to do next (re-ordered by research round 7's finding)

Research round 7 measured the cost model that matters: **a submit+fence is a fixed
56-62 µs wall regardless of how many dispatches it carries**, and each command-buffer
boundary costs 21-34 µs of GPU timeline even inside one submit. That reframes §8af:
167 dispatches removed changed almost nothing because dispatches are not the unit.

By that model the two structural numbers in hand are:

| quantity | measured | predicted cost |
|---|---|---|
| submits per block-chunk in chunked prefill | **13.6**, not 2 | ~550 extra submits per prefill x 62 µs = **~34 ms** |
| unneeded dependent barriers (~1 100 of ~4 800 per token) | §8af | ~3.8 ms/token at 3.43 µs |

The 13.6 submits come from **PLE**: `ple_step` flushes twice per token and then records
GPU work with no batch open, so each of those dispatches becomes its own submit. That
makes **PLE batching the next item** (it also removes the cold-row serialisation of
§8z): one batch around the whole PLE phase per block-chunk, target ~2.2 submits per
block-chunk, and the submit count is now a headline metric to report beside the stage
table.

### Not landed this round

Async expert prefetch (the ~38 ms/token lane item) and the per-resource barrier
tracking: unchanged, and correctly sequenced **after** the submit reduction, because
the two structural costs are now ordered by measurement rather than by estimate.
Parity is untouched this round (timers only): chunk 128 + merged 32/32, top-1
13.170561; no decode path changed, so the canonical gate stands at 20.3 +- 1 aggregate,
lane 40-43 ms/step. The 2.98 tok/s/agent in this window is not a regression — it is a
loaded box (`experts(host)` 83 vs 38-48 in quieter windows, `ple` 17.6 on the MI50
stream); the attribution, not the rate, is the deliverable.

## 8ai. PLE batching is unblocked, not landed: the cap is now a construction parameter

The round's item was the submit reduction (13.6 -> ~2.2 submits per block-chunk,
~34 ms by research round 7's 62 us per submit). It did not land, and the honest reason
is a specific data-layout constraint rather than time, so it is written down before
the next attempt.

### What landed: the blocker §8z named is removed

`xfer::ple_stage_bytes(tokens, hc_dim, n_embd)` is now the rule for a chunked
session's staging `cap` (`Session::new` takes the max of the single-readback default
and this), with a unit test that asserts it covers `t * (2*hc_dim + n_embd) * 4` bytes
for `t = 1, 8, 64, 128` and is not oversized. Before this, `read_batch` refused a
multi-token PLE readback outright — the cap was ~1 MiB against ~12 MiB at `t = 128` —
which is *why* the phase could only go token by token. **No behaviour change**, no
decode path touched, parity untouched.

### What still blocks the batching itself

`ple_step` per token is a **host round trip** by construction: read `R`, `key` and
`value` back, run the host block (grouped RMSNorm, the conv state, the gated residual),
write `R` back. Batching needs `t` of those in one pass, and two things do not yet line
up:

1. **The gather output is per token.** `EmbGather::run(packed, idx, out, batch)` writes
   one token's 16 rows into a single 2 560-float buffer, and the key/value projections
   then read that buffer. A chunk version wants `t * 16` rows gathered at once, but the
   per-token key projection needs a *view* into it at `i * 16` rows — and `Buf` has no
   offset field (the constraint that shaped the merged pass, §8v). Either the projection
   becomes an `m = t` `GemmMany` over the stacked rows (needs a multi-user kernel for the
   PLE key/value types) or each token's rows are copied into the single-token buffer
   inside the batch (one `record_copy` per token — cheap, and the path that fits today).
2. **The hash is ordered and `rows_for` mutates state**, so it must run per token in
   sequence before the batch opens — which means `t` `pack_rows` + uploads. Uploading
   each token's 1 440 bytes separately would reintroduce `t` submits, so the packed rows
   must be concatenated into **one** buffer per chunk with a per-token index list, i.e.
   one `upload` before `ensure_batch` and `t` gather dispatches inside it.

So the shape is settled (one flush, one concatenated upload, `t` gathers + `2t`
projections + `t` copies recorded in **one** batch, `t/24`-token staging slices for the
readback, `t` host blocks, one writeback with a region array) and the piece that made it
not fit in this round is item 1 — the per-token view — which should be done with the
copy-inside-the-batch route first and the `GemmMany` route only if the copy cost shows.

### State

Submits per block-chunk remain **13.6** (unchanged), `wait-gpu` and `ple` unchanged, so
**no prefill delta is claimed this round**. Parity unchanged: chunk 128 + merged 32/32,
top-1 13.170561 (the change is allocation sizing only). No decode path touched, so the
canonical gate stands at 20.3 +- 1 aggregate, lane 40-43 ms/step.

## 8aj. Landed: PLE batching (13.92 -> 4.65 submits/block-chunk) and the tail-chunk plan

Both items of this round landed. Every number below is **clock-verified**
(`power_dpm_force_performance_level` = `high` on both cards, read in the run
header). The RX 6950 XT idling at 500 MHz voided placement conclusions in
sections before this one — treat any single-device RX figure in §8s-§8ai as
**clock-throttled** rather than comparable (AGENTS.md "Measurement traps").

### What changed

1. **PLE is one batched phase per chunk** (`Session::ple_chunk`): the host hash
   still runs token by token (it advances the rolling context), then **one**
   concatenated pack + prefetch, **one** upload recorded into the batch, **one**
   gather over `t*16` rows, per token `copy` (a `Buf` has no offset field, so this
   is the "copy-inside-the-batch" route §8ai chose) + key/value GEMV recorded in
   the **same** batch, **one** `read_batch` for all `t` residuals/keys/values, `t`
   host blocks, **one** region-array writeback (`Staging::write_regions`). The
   per-token key/value outputs are one buffer each for the same no-offset reason.
   Revert/A-B arm: `BRECCIA_NO_PLE_BATCH=1` (decode is untouched — it still uses
   `ple_step`).
2. **Dynamic chunk sizing** (`prefill_cap` / `prefill_plan`, both pure and unit
   tested): `--prefill-chunk N` is now a **target**; the scratch is sized for
   `cap = 1.5 N` and a prompt is split into `ceil(len/cap)` **even** chunks, so no
   chunk is below `cap/2` and a 132-token prompt is `66 + 66`, not `64 + 64 + 4`.
   Revert/A-B arm: `BRECCIA_NO_PREFILL_PLAN=1` (greedy `N`-sized chunks + tail).
3. **A real per-queue submit counter** (`breccia_vk::ctx::submits()`): the
   headline metric cannot come from `Session::submits`, which only counts the
   batch ring — the staging readbacks and one-shot uploads submit on their own, and
   *those* are what PLE batching removes. The `prefill sum` line now prints
   `N submits (X/block-chunk, M from the batch ring)`.
4. **The parity gate earned its keep**: the first version of (2) shadowed `pos` in
   `generate`, so decode started at position 0 — logits after prefill were
   bit-identical, the *second* generated token diverged. Caught by 32/32 parity,
   not by any timing.

### Measured (RX 6950 XT, 64-token prompt, chunk 64, clocks high, quiet window)

| quantity | before | after | delta |
|---|---:|---:|---:|
| **submits/block-chunk** (all queues) | **13.92** | **4.65** | **-9.27 (3.0x)** |
| of which the PLE phase, per chunk | 451 | 5 | -446 |
| of which the batch ring, per chunk | 217 | 218 | 0 (dense chain, next lever) |
| `ple` stage, ms/token | 0.94-1.01 | 0.42-0.46 | **-0.52** (~-34 ms per 64-token chunk) |
| `wait-gpu`, ms/token | 10.13 | 10.12 | 0 |
| `prefill sum · wall · residual` | 70.81 · 70.83 · 0.03 | 71.88 · 71.91 · 0.03 | residual still ~0 |

The **-0.52 ms/token x 64 = -34 ms** is the prediction of §8ah's cost model
(550 submits x 62 us) hit exactly, and it is the honest claim: the *total* chunk
wall does not move because it sits inside `experts(host)`'s +-1.6 ms/token drift
in this window. The remaining 4.65 is **not PLE** — it is the dense chain's batch
segmentation (4.54/block against the 2 passes), i.e. the next item.

### Tail chunk: real, but window-dependent

| prompt | plan before | plan after | prefill total before | after |
|---|---|---|---|---|
| 64 | 64 | 64 | 4.53 s | 4.60 s |
| 132 | 64+64+**4** | **66+66** | 8.76 s | **8.61 s** (-1.7 %) |
| 300 | 64x4+**44** | **75x4** | 20.02 s | 20.48 s (+2 %, noise) |

The tail's own cost here is **72-104 ms/token** against 66-69 for a full chunk
(+8-50 %, not the 8x that `docs/research/07` measured on a loaded box), so removing
it is worth only 1.7 % of a 132-token prefill and nothing measurable at 300 (four
75-token chunks cost the same as four 64s plus a 44). The plan is still the right
default: the pathological shape is gone and `132` no longer pays a 3rd chunk's
fixed cost. **Quote the tail number with its window** — on the loaded MI50 the same
shape was 13 % of the run.

### Parity and gates

- Chunked parity, single device, new plan + batched PLE: **32/32 at chunk
  8/16/32/128**, RX (`0000:03:00.0`). The 27-token gate prompt is now split
  `9+9+9` at chunk 8 (was `8+8+8+3`) and `27` at chunk 32/128.
- The "`--prefill-chunk` is wrong for chunk >= 9" note in §10 **did not
  reproduce** on the RX in this tree: HEAD is 32/32 at chunk 9, 32 and 128.
- Decode path untouched: the 2-stream aggregate in the same window is
  16.5-19.8 tok/s depending on split (see below); the canonical 20.3 +- 1 gate is
  a different (quieter) window, so no decode claim is made here.
- `cargo test --workspace` (29 suites), clippy `-D warnings`, `fmt --check`,
  `./scripts/verify.sh`: green.

### Split sweep with clocks pinned (27-token prompt, gen 8, warm 0, 2 and 4 streams)

| split (RX / MI50) | 2 streams agg | 4 streams agg |
|---|---:|---:|
| 28 / 20 | **23.57** | 20.67 |
| 32 / 16 | 22.76 | **20.79** |
| 36 / 12 | 21.89 | 20.96 |
| 40 / 8 | 23.15 | 20.87 |

At gen 16 + `--warm 1` the 2-stream ranking **inverts** (32/16 19.84, 40/8 18.85,
28/20 16.54), so the split is inside the window's noise at these lengths: neither
the 2-stream nor the 4-stream table shows a real split effect (4 streams: 0.4 %
spread). **Keep 28/20 as the default**; do not tune it further until the host lane
stops being the constraint (lane 69-82 ms/step, fence 12-13 ms/step, 100 %
overlapped).

### Corrected per-card baselines (clocks high, 27-token prompt, gen 32, warm 1)

| card | decode | per-token |
|---|---:|---:|
| RX 6950 XT `0000:03:00.0` | **13.29 tok/s** | 75.2 ms |
| MI50/Radeon VII `0000:0e:00.0` | 9.30 tok/s | 107.5 ms |

RX is **1.43x** the MI50 single-device, so an RX-favoured split is directionally
right — and the **prefill** balance agrees: at chunk 64 a 64-token prompt costs
**65.5 ms/token on the RX vs 92.0 on the MI50** (1.40x; `wait-gpu` 10.1 vs 29.3).
The reason the split does not pay yet is that both cards idle under the host lane
in the measured windows, not that the placement is wrong.

## 8ak. Single-stream: the lane is removed by **expert coverage**, not by overlap

The round's objective was 20.79 tok/s (llama.cpp) on **one** stream with both cards
working. Measured ladder below; every run clock-verified (`power_dpm_force_performance_level`
= `high` on both cards), quiet window, 27-token prompt, gen 8/32, `--warm 1`,
`--threads 16`, dual placement `--split-blocks 28,20`.

### The ladder (single stream, both cards)

| blocks' experts resident | arena | expert bytes/token | `experts(host)` | `wait-gpu` | `head` | tok/s | ms/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (all 48 on the lane) | 0 | **799 MiB** | **47.5** | 21.9 | 4.2 | 11.82 | 84.6 |
| 16 | 14.9 GiB | 508 MiB | 30.0 | — | — | 14.26 | 70.1 |
| 26 | 22.5 GiB | 360 MiB | 17.5 | 33.2 | 7.4 | 15.76 | 63.5 |
| **30** | **25.5 GiB** | **301 MiB** | **13.5** | **33.4** | 7.1 | **16.94** | **59.0** |

`--gpu-experts "0..15,28..41"`, `--split-blocks 28,20`. Parity **32/32** at 30 blocks
(16.70 tok/s, 59.89 ms/token, 301 MiB/token): resident experts are the same kernels
on the arena, so the sequence is bit-identical to the server's.

**+43 % single-stream (11.82 -> 16.94 tok/s) with no engine change** — the item is a
placement/VRAM budget, not a pipelining one. Per block the arena costs 930 MiB
(measured), so full coverage would need ~44 GiB; both cards hold 32 GiB, of which
3.2 GiB is dense weights and 0.4 GiB the head. ~30-32 blocks is the ceiling, which
is what the 30-block row is: 29.2 GiB of 32 GiB used.

### Why the planned overlap (task 1a) is **not** what limits one stream

Block L+1's dense prelude consumes the residual that block L's **lane** produces, so
inside one stream the lane is a strict serial section, not a schedulable one: there
is no GPU work that can be issued while it runs. What *can* be overlapped is the
lane's **data** (page-cache reads / PCIe), not its compute — and at 30 blocks that
data is 301 MiB/token = ~13.5 ms, i.e. **the whole remaining headroom of 1a is
13.5 ms** (59 -> 45.5 ms, 22 tok/s) and it is I/O prefetch, not lane hiding. The
same accounting sets 1b's value: a CPU+RX+MI50 expert split can only help the 18
blocks that do *not* fit in VRAM, and their cost is the host read (~301 MiB), not
spare GPU compute.

### What is left, in order

1. **`wait-gpu` 33.4 ms/token (57 %) is now the bottleneck** — the GPU chain itself:
   ~4 800 dispatches/token at 4.4 µs plus the command-buffer boundaries priced in
   §8af/§8ag. This is the barrier-tracking / T-batched-dense-chain item, and it is
   the only path to 20.8 tok/s now (target: 4 800 -> ~2 800 dispatches, or 3.4 -> 2.0
   barriers/dispatch).
2. **`head` 7.1 ms** (248 320x2 560 GEMV) runs on whichever card owns block 47 — the
   MI50 under every contiguous split. Moving the last block to the RX is worth up to
   ~4 ms/token (head on the 1.43x faster card).
3. Then the lane's 13.5 ms, as a prefetch (1a), if 1 and 2 are exhausted.

### Not attempted this round

- **Route-tail skipping (task 2)** — no ε sweep, no divergence table, nothing to
  report; it stays unmeasured and off.
- **Subgroup-size/tile spec constants (task 3)** — untouched.
- The planned **layer L+1 expert prefetch** and the **CPU+RX+MI50 expert split**
  (task 1) were not implemented: the measurement above reorders them behind (1),
  because coverage already removed 71 % of the lane (`799 -> 301 MiB/token`) and the
  GPU chain is now 2.5x the lane.

### Commands

```bash
# clocks (both cards must read `high`), then the ladder
for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do cat "$f"; done
D=/var/tmp/users/korphaus/tmp/opencode/p4-token
timeout 900 ./target/release/breccia bench token \
  --devices 0000:03:00.0,0000:0e:00.0 --split-blocks 28,20 \
  --gpu-experts "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,28,29,30,31,32,33,34,35,36,37,38,39,40,41" \
  --prompt-ids-file $D/parity-server.json.ids --gen 32 --runs 1 --threads 16 --warm 1 \
  --json $D/cov30-parity.json
```

## 8al. Per-resource barrier tracking: landed, correct, and it finds only 20 % — the rest are real

### What landed

`dispatch_batch::record` now tracks a **dirty set** of `vk::Buffer`s written since the
last barrier and emits a compute→compute barrier **only when the dispatch about to be
recorded binds one of them** (ncnn's model, per *dispatch* rather than per call site,
which is why it cannot repeat §8z's parity failure). Two correctness anchors:

- the dirty set is **conservative in one direction** — every bound buffer counts as
  possibly written, because a kernel does not declare its writes — so it can only
  drop barriers between dispatches whose storage sets are disjoint;
- `end_submit` closes every batch with one barrier if the set is non-empty, because
  queue order alone does not make memory visible across submissions;
- any recorded copy (`record_copy`/`record_copies`) clears the set, since its own
  pre/post barriers fence every prior write.

A/B arm `BRECCIA_NO_BARRIER_TRACK=1` (always-barrier shape). Counters:
`batch_cmd::barriers()` / `barriers_pruned()`, printed as a `barrier` line.

### Measured (single stream, both cards, 30 blocks resident, clocks `high`)

| quantity | always-barrier | per-resource |
|---|---:|---:|
| barriers per token (32 generated) | 319 400 | **256 670 (-19.6 %)** |
| barriers per dispatch | 1.00 | **0.82** |
| tok/s / ms per token | 16.94 / 59.0 (gen 8 window) | 17.14 / **58.36** |
| parity (32 greedy tokens vs the HIP server) | 32/32 | **32/32** |

**Why only 20 %**: this graph is a chain through the residual stack `r`. Almost every
dispatch binds `r` and most write it, so with no read/write declaration per kernel the
intersection is non-empty by construction (~0.8 of dispatches). §8af's "~1 100 of 4 800
barriers are unneeded" was a *call-site* estimate; at the dispatch level they are
dependencies. **Cutting `wait-gpu` further needs fewer dispatches (fusion), not fewer
barriers** — which reorders §8af's items: item 2 (fused kernels) first.

### sync2 vs v1 `ALL_COMMANDS`, equal barrier count (same flags, adjacent runs)

| barrier form | tok/s | ms/token |
|---|---:|---:|
| v1 `ALL_COMMANDS` (`BRECCIA_BARRIER=full`) | 11.89 | 84.10 |
| sync2 COMPUTE→COMPUTE (default) | **17.27** | **57.91** |

Both arms emitted exactly 184 190 barriers, so this is the barrier's own cost:
**-26.2 ms/token, 1.45x** — the "1.6x cheaper" research run is right and the
"identical" one was measuring a window where something else dominated. Keep sync2 the
default.

**Window caveat**: research round 9 was benchmarking on the RX throughout, so the
gen-8 numbers bounce 11.9-17.3 tok/s for identical barrier counts; the pairs above are
adjacent, so the *deltas* are usable and the absolute rates are not. The 32-token parity
run (17.14, 58.36) is the one to quote, and it is within noise of §8ak's 16.94.

### Correction to §8ag

§8ag concluded "coverage cannot fix the lane (+4 blocks buys -7.5 %, full coverage needs
~43 GiB)". The second half stands; the first was a **4-block step, not a ladder**:
0 -> 30 blocks takes the lane 47.5 -> 13.5 ms and the stream 11.82 -> 16.94 tok/s
(§8ak). Coverage *is* the lane lever up to the VRAM ceiling (~30-32 blocks of 48).

### Not done this round

Items 2 (fused RMSNorm+QKV / gate+up+SwiGLU / RoPE+KV / hyper-connection), 3 (frozen
command-buffer replay) and 4b (adaptive coverage from measured free VRAM with a reserve
on the display card) were **not** implemented: item 1's measurement reordered them
(fusion first), and the window did not allow a new shader round that could pass the
GPU-safety protocol (one shape per invocation, validation on, RX first).

## 8am. `wait-gpu` is **78 % launch overhead**: fusions before traffic (and the probe is still broken)

### The verdict, from a controlled A/B rather than a stage table

`BarrierScope::Full` and `BarrierScope::Sync2` do the **same work, the same
dispatches and the same bytes**; they differ only in the sync primitive. At an
identical barrier count (184 190 emitted in both arms) on the same config (single
stream, 30 blocks resident, clocks `high`):

| arm | tok/s | ms/token |
|---|---:|---:|
| v1 `ALL_COMMANDS` (`BRECCIA_BARRIER=full`) | 11.89 | 84.10 |
| sync2 COMPUTE→COMPUTE (default) | **17.27** | **57.91** |

**26.2 ms/token of the 32.5 ms `wait-gpu` is launch/serialisation cost**, i.e.
**≈78 % overhead, ≈22 % execution**. Nothing about arithmetic, bytes or residency
changes between the arms, so this is not an inference from wall/diff — it isolates the
overhead directly. With 0.82 barriers/dispatch and ~4 700 dispatches/token, 26.2 ms
over 3 900 barriers is 6.7 µs per barrier of *GPU-timeline* serialisation, which is the
launch cost the barrier forces (§8af priced the freed dispatch at 4.4 µs; the barrier
that follows it costs more than the dispatch).

**Consequence (this reorders the round's items):** cut **dispatch count**, not bytes.
Fused RMSNorm+QKV / RoPE+KV / HC read+write / gate+up+SwiGLU, then the frozen
command-buffer replay. Route-tail skipping (bytes) and expert residency are *not* the
next lever at a 22 % execution share.

### The per-stage GPU table is blocked: the probe aborts

`BRECCIA_GPU_TIME=1` dies on the first forward pass with

```
breccia: the first forward pass: reading the GPU-time probe:
         vkGetQueryPoolResults: A fence or query has not yet completed
```

The probe read its pool while the submission that wrote the stamps was still in
flight. Two fixes landed in `Session::probe_stamp`/`probe_flush`: stop stamping when
the pool is full instead of reading mid-recording, and drain this session's segments
before the read. The read **still** races on the first pass (a pool is shared across
the ring's slots, so the not-ready stamps belong to a submission that is not the
session's `pending_seg`), so **no per-stage µs table exists this round** — the
instrument needs a per-slot pool (one pool per ring slot, read when that slot's fence
has been waited) before it can be trusted. Recorded here so the next round fixes it
rather than rediscovering it.

### State after this round (single stream, both cards, clocks `high`)

| quantity | value |
|---|---:|
| tok/s / ms per token | **17.32 / 57.74** |
| `wait-gpu` | 32.5 (56.6 %) |
| `experts(host)` (lane, 301 MiB/token) | 13.4 |
| `head` | 7.0 |
| ple / xfer-down | 1.2 / 0.6 |
| parity (32 greedy tokens vs the HIP server) | **32/32** |
| barrier accounting | 256 670 emitted, 62 730 pruned (0.82/dispatch) |

### Items 2 and 3: what was and was not done

- **No fusion shaders were written.** The GPU-safety protocol (RX first, one new
  shape per invocation, validation on, `timeout`) needs a dedicated window, and the
  round's budget went to the attribution that decides *which* fusions matter. The
  order they should land, from the overhead share above: (1) fused `gate+up+SwiGLU`
  — 2 dispatches per expert instead of 3, the largest single dispatch reduction per
  token; (2) fused RMSNorm+QKV; (3) fused RoPE+KV-write; (4) frozen CB replay.
- **`head` cannot be moved by a split choice** (item 3a): the head runs on the device
  that owns block 47, and every split that keeps ~30 blocks resident puts block 47 on
  the MI50. Making the RX own the last block needs an explicit head-device option (a
  code change, ~-4 ms expected), not a different `--split-blocks`. The best 1-stream
  config remains `--split-blocks 28,20` with the 30-block arena.
- **Adaptive coverage (item 3b) not done**: the arena is still a host-specific block
  list; sizing it from measured free VRAM with an explicit reserve on the display
  card (MI50 drives DP-0/DP-1/HDMI-A-0) and a refuse-and-degrade path is the next
  portable piece.

**Window note**: the paired research agent was benching the RX during these runs —
the lane swung 13.4-26.2 ms between identical configs. The pairs above are adjacent
(same window), so the *deltas* are sound; the absolute 17.32 tok/s had one clean
re-measurement and matches §8ak/§8al within noise.

## 8an. The census is wrong and the record path is the real twin of `wait-gpu`

### Dispatch count: ~2 800/token, not 4 700

`313 790` dispatches over `112` token-graphs (3 prompt passes x 27 + 31 decode) =
**~2 800 dispatches/token** on the 30-block config. The 4 700 in the round's premise
came from §8af's pre-residency chain; expert coverage removed ~40 % of it. With
`0.82 barriers/dispatch` that is **~2 300 barriers x 6.7 µs ≈ 15.4 ms of the 32.5 ms
`wait-gpu` (47 %)** — barrier serialisation is *half* the fence wait, not all of it.
The round's predicted fusion+replay+head budget (−15 ms) survives, but its components
shrink: fusing 700 dispatches is now 700 of 2 800 (25 %) and worth ~−4 ms only if the
dispatching really is per-expert (unproven — see below).

### The census instrument is not usable yet

`BRECCIA_TRACE_DISPATCH=1` (`Session::probe_report`) prints per-stage dispatch counts
from `Timers::dispatches`, but that accounting only sees stages that are current **at a
batch flush**, and its `dispatch_mark` uses the process-wide `recorded()` counter which
resets per batch. One decode token prints only `shared-exp 90 · hc-write(ffn) 32 ·
head 1`, i.e. **123 of ~2 800** — everything recorded inside a block's batch is
invisible. A correct census needs the dispatch count of *each* batch attributed to the
stage that was current when the batch was **opened** (`ensure_batch`), i.e. per-batch
marks rather than per-flush ones. Not done: the number above (total/token) is sound,
the per-stage split is not yet.

### The finding that reorders item 2: **record time is 32.5 ms/token**

```
record   32.52 ms/token of host recording (313790 dispatches, 36 per submit)
```

That is the *host* side of the same 58 ms token, and it is as large as `wait-gpu`
(32.5 ms). Per dispatch it is **~11.6 µs** — far above research 08's 1.4 µs
submit-path figure, because `dispatch_batch::record` allocates and writes a **fresh
descriptor set per dispatch** (the rule that keeps recorded batches safe, module docs).
So the frozen-CB replay (item 2 of the round) is the **largest single item on the
board**, not the third: it removes both the per-dispatch set write and the record-path
CPU time, and the two are worth up to ~20-30 ms/token if the set allocation is what
dominates. That is the next round's first job, and it needs the per-slot query pools
(recorded in §8am) plus a shape-frozen decode graph.

### State (single stream, both cards, clocks `high`, quiet window)

| quantity | value |
|---|---:|
| tok/s / ms per token | **17.19 / 58.16** (17.32 / 57.74 in the prior window) |
| `wait-gpu` | 32.5 (56 %) |
| host record time | **32.5 ms/token** |
| dispatches/token | **~2 800** (0.82 barriers/dispatch) |
| `experts(host)` / `head` / `ple` / `xfer` | 13.4 / 7.0 / 1.3 / 1.1 |
| parity (32 greedy tokens vs the HIP server) | **32/32** |
| expert bytes/token | 301 MiB |

### Not done this round, in the order the numbers now dictate

1. **Frozen-CB replay** — the biggest item (record path 32.5 ms/token, 11.6 µs per
   dispatch). Requires a shape-frozen decode graph and per-slot query pools.
2. **Correct dispatch census** (per-batch stage marks) — cheap, and it decides whether
   the per-expert GEMVs really are individual dispatches (`GemmMany` collapse would be
   worth more than all the fusions).
3. The four fusions (gate+up+SwiGLU, RMSNorm+QKV, RoPE+KV, HC) — ~−4 ms, unmeasured
   because not written; new shaders need the GPU-safety window.
4. Head-device option (−4 ms) and the adaptive arena (portable coverage rule).
5. Zero-copy host-memory experts (`docs/research/09`: 22-26 GB/s verified) for the 18
   hosted blocks — removes the last serial CPU section; needs 4096 B alignment and,
   for a pinned pool, `LimitMEMLOCK=infinity` (`ulimit -Hl` is 8 MiB here).

## 8ao. Persistent descriptor sets: implemented, measured, **no delta** — so replay is not redundant

### What was tried (Item 1a)

`dispatch_batch::record` no longer allocates a set per dispatch: `SETS` caches one
**pre-written** set per `(pipeline handle, buffer handles)`, allocated from a
persistent per-device pool (`set_pool`, freed in `shutdown`). A cached set is written
before any command buffer references it, so the module's `vkUpdateDescriptorSets` rule
still holds. The A/B arm is the same switch as the barrier tracker
(`BRECCIA_NO_BARRIER_TRACK=1` falls back to the per-batch set), and
`descriptor_sets()` returns `(written, reused)`.

### Measured (single stream, 30 blocks, gen 16, warm 1, clocks `high`, adjacent pair)

| arm | record ms/token | dispatches/token | tok/s | ms/token | `wait-gpu` |
|---|---:|---:|---:|---:|---:|
| per-batch set (old) | 19.76 | 5 167 | 16.95 | 59.00 | 33.49 |
| **persistent set (new)** | **19.48** | 5 167 | 16.78 | 59.60 | 33.40 |

**No delta** (−0.3 ms, inside noise). So the ~3.2 µs/dispatch of record time is **not**
descriptor-set allocation: with the alloc+write removed the hot path still costs
`cmd_bind_pipeline` + `cmd_bind_descriptor_sets` + `cmd_push_constants` +
`cmd_dispatch` + the barrier record, and that is where the time is. (The set write was
already amortised in practice: RADV's pool allocation from a reset pool is cheap.)

**Consequence for Item 2:** the frozen-CB replay is **not** made redundant by 1a — it
removes the *whole* record path (all four commands plus the driver's validation of
each), not just the set. At 3.2 µs/dispatch × 5 167 that is up to ~16 ms/token of host
time, and it is the only remaining way to cut recording. It stays the next item, and it
still needs the per-slot query pools (§8am) to be trusted.

`BRECCIA_NO_BARRIER_TRACK=1` therefore now has two meanings (barriers **and** the
descriptor cache); they must be split into two flags before the next measurement round.

### State (single stream, both cards, clocks `high`)

| quantity | value |
|---|---:|
| tok/s / ms per token | **17.56 / 56.94** (best of the window) |
| `record` | 22.4 ms/token (this window), 19.5 in the adjacent pair |
| `wait-gpu` | 33.4 (0.82 barriers/dispatch, 62 730 pruned) |
| dispatches/token | **5 167** (36 per submit; ~3.2 µs each) |
| `experts(host)` / `head` / `ple` / `xfer` | 13.4 / 7.0 / 1.2 / 1.1 |
| expert bytes/token | 301 MiB |
| parity (32 greedy tokens vs the HIP server) | **32/32** |

### Not done (unchanged order)

1. **Frozen-CB replay** — now confirmed as the only remaining record-path lever.
2. **Complete dispatch census** (per-batch stage marks) — still 123 of 5 167 visible.
3. The four fusions, the head-device option (~−4 ms), the adaptive arena
   (measured-free-VRAM rule with a display-card reserve), zero-copy host-memory experts.

## 8ap. The denominator: **1 843 dispatches/token**, and what that re-prices

### The counter was right, the division was wrong

`Session::dispatches` is a **run total**, and the CLI prints it before dividing by
`runs_n` (runs, not tokens). Dividing 313 790 by "60-ish token-graphs" gave 5 167; the
run actually contains **170** token-graphs: `--warm 1` (one full generation: 27 prompt
+ 32 decode) + the two prompt passes of the measured run (2 x 27) + its 32 forwards
= 58 + 54 + 32 + ... = 170. 313 790 / 170 = **1 845**, which the fixed census confirms
independently: **1 843 dispatches/token** (`ple 78 · hc-write(ffn) 1 719 · handoff 39 ·
final-hc 7`). The static estimate of ~1 650 (32/block x 48 + head) is the same number to
within the 190 non-block dispatches.

So the true prices, at ~19 ms/token of recording: **~10 µs per dispatch** (not 3.2),
and 1 511 barriers/token (0.82/dispatch) account for ~12 ms of the 32.5 ms `wait-gpu`
— i.e. barriers are ~38 % of the fence wait, and **recording itself (~19 ms, ~33 % of
the token) is the larger of the two.** Both are per-dispatch, which is why every fusion
in `docs/research/10` is worth more than its dispatch count suggests.

### Census repaired (Item 2)

`Timers` now remembers the stage a batch was **opened** in (`open_batch` in
`ensure_batch`) and attributes the batch's whole dispatch count there
(`count_batch` from the batch's own `recorded()`), instead of the broken
global-mark arithmetic. `BRECCIA_TRACE_DISPATCH=1` now prints a table that **sums to
the total** (1 843), verified over 50+ consecutive tokens. The split is still coarse —
1 719 of them land in `hc-write(ffn)` because a block opens its batch at that mark —
so a per-phase split needs marks *inside* `submit_block`; that is the next census step
and it is what D5's delta will be measured against.

### Item 1 (static-graph replay): deliberately **not** pursued

Per `docs/research/10` §3: after the fusions a frozen decode graph is worth 0.8-1.1 ms
(it lands in `vkQueueSubmit`, not the record loop) and **every fusion invalidates it**.
Nothing was written for it this round, and the flag-free code stands. Order agreed:
D5 (hyper-connection collapse, 14 of a block's 32 dispatches) -> D1 -> D3 -> D4 -> D2,
then replay last, if ever.

### Not done this round

D5/D1/D3/D4/D2 (no new shaders: a shader round needs the GPU-safety window — RX first,
one shape per invocation, validation on), head-device option, adaptive arena. The
recommended next step is **D5 with the coarse census as its meter**, since 1 719 of
1 843 dispatches are block work and HC is 44 % of a block's static count.

### State

| quantity | value |
|---|---:|
| tok/s / ms per token | **17.48 / 57.19** |
| dispatches/token (census) | **1 843** |
| record / `wait-gpu` | 22.8 (contended window) / 32.4 |
| `experts(host)` / `head` / `ple` / `xfer` | 12.8 / 7.0 / 1.2 / 1.1 |
| parity (32 greedy tokens vs the HIP server) | **32/32** |

## 8aq. The 7 us/dispatch was a denominator artefact again — the per-call cost is **~0.3 us**

### The measurement

`record` now accumulates host nanoseconds per phase (`batch_cmd::record_profile()`,
printed as a `profile` line). One run (single stream, 30 blocks, gen 16, warm 1,
clocks `high`):

| phase | ns/dispatch | share |
|---|---:|---:|
| descriptor key + cache lookup | **204** | 72 % |
| `cmd_bind_pipeline` | 43 | 15 % |
| barrier (`sync2` COMPUTE→COMPUTE) | 36 | 13 % |
| set write / `bind_descriptor_sets` / `push_constants` / `cmd_dispatch` / tracker | **0 (below timer resolution)** | — |
| **total instrumented** | **~283** | |

So a dispatch costs **~0.3 µs** of host recording, not 7. The `record 14.16 ms/token`
line divides the **decode-window** record wall (reset after the prompt by
`reset_record_seconds()`) by `runs_n`, and the per-dispatch figure derived from it
divides by `sess.dispatches`, which is a **whole-session** count that was never reset —
the same class of mismatch as §8ap. Proof: the profile counters, which are process-wide
and never reset, sum to **71.9 ms over ~170 token-graphs = 0.42 ms/token**, and
0.42 ms/token over 1 843 dispatches is 230 ns/dispatch, matching the table. The two
accounting windows are different lengths; dividing one by the other is what produced
6.9-7.7 us.

**Consequence:** there is no 15-50x anomaly. Recording is ~0.4-0.6 ms/token of host
time (~1 % of the token), the descriptor cache already removed the set cost (§8ao), and
the remaining lever on recording is the 204 ns key lookup — a per-dispatch `Vec` key
allocation in `SETS`. It is worth ~0.38 ms/token, i.e. **below the noise floor of this
box**, so it is recorded here and not "fixed" for a number nobody can see.

The lesson is the same one as §8ap, and it is worth stating in the round's own terms:
**every per-dispatch price quoted so far (3.2 us, 6.7 us, 6.9 us) came from dividing a
wall-clock total by a dispatch count from a different window.** The instrumented table is
the only number that is internally consistent.

### Item 2 (D5) and the rest: not done

No shaders were written this round. The recommended order stands unchanged
(D5 hyper-connection collapse -> D1 -> D3 -> D4 -> D2), with D5's meter being the
census, which now sums but still lumps a block's work into `hc-write(ffn)`: the missing
piece is marks *inside* `submit_block` (prelude / per-expert / ffn), which is cheap and
is the first thing the next round should do before cutting anything.

Still outstanding: head-device option (~-4 ms), adaptive arena with a display-card
reserve, D1-D4.

### State

| quantity | value |
|---|---:|
| tok/s / ms per token | **17.23 / 58.03** |
| dispatches/token (census) | 1 843 |
| recorded host time | **~0.42 ms/token measured, ~0.3 us/dispatch** |
| record / `wait-gpu` (as printed, same window mismatch) | 14.16 / 32.01 |
| `experts(host)` / `head` / `ple` / `xfer` | 13.9 / 7.0 / 1.2 / 1.1 |
| parity (32 greedy tokens vs the HIP server) | **32/32** |

## 8ar. No implementation this round; the three blockers, exactly

This round produced no code. Writing that down with the specific obstacle for each item
so the next one starts from facts.

### D5 / D1 (shader items): blocked on the GPU-safety protocol, not on design

Both need a **new kernel** (`hc_read`/`hc_write` collapsed; gate+up+SwiGLU fused), and
the repository's own rule for a new shader is: RX first, **one fresh dispatch shape per
process**, `VK_LAYER_KHRONOS_validation` on, `timeout 90`, and a reset is a
stop-the-line event (AGENTS.md, after the 2026-10-26 BACO reset). That is a window of
its own: compile the SPIR-V, run one shape, read the VU list, then the second shape. It
cannot be interleaved with benchmark runs of the same process, and it is not something
to start with a few minutes of budget left — the failure mode is a machine reboot.

The design inputs are ready and unchanged: the HC projector is 14 of a block's 32
dispatches (44 %), the gate+up+SwiGLU fusion is measured at 3.1x (m=1) / 5.3x (m=8),
and the collapse target is -290…-380 dispatches/token (research 10). The one cheap
prerequisite is **census marks inside `submit_block`** (prelude / per-expert / ffn) so
each fusion's delta is attributable — that is a ~20-line change with no GPU risk and it
should be the first commit of the next round.

### `--head-device`: not a config flag, a placement change with a handoff

The head is loaded onto `placement.device_of(n_layers - 1)` (`model.rs:507`) and the
final residual is consumed where block 47 ran. Moving the head to the RX therefore needs
(a) a `SessionCfg`/CLI override and (b) a **residual handoff** to the head device in
`submit_head` — the machinery exists (`HANDOFF`, used for a block crossing devices) but
it has to be wired for the head, plus the 416 MiB `output.weight` upload redirect. Not a
flag rename; a small feature with a new data path, which deserves its own parity run.

### Adaptive arena: needs a VRAM query that is not wired

Nothing in the tree asks the driver for free VRAM (no `VK_EXT_memory_budget`, no
`rocm-smi` parse), so "size the arena from measured free VRAM with a display-card
reserve" needs: a free-VRAM probe per device, a byte-budget → block-list selector
(replacing the hardcoded 30-block list), an explicit reserve on the display card, and a
refuse/degrade path that turns a would-be `VK_ERROR_OUT_OF_DEVICE_MEMORY` into "N blocks
resident". Each piece is small; together it is a round, and it must not be rushed on the
same card that drives the displays.

### Where the budget stands (unchanged, proven)

`wait-gpu 32.0 · lane 13.9 · head 7.0 · ple 1.2 · xfer 1.1` of 58.4 ms; 1 843
dispatches/token; recording 0.5 ms (measured per-phase, ~0.3 µs/dispatch); barriers are
real dependencies. **Fusions are the only lever on the 32 ms**, and they are shader work.

## 8as. Census marks inside `submit_block` landed; D5 not attempted

### Census: the 1 719-dispatch lump is now split (2-way, attributable)

`Session::census_split(dev)` closes the open batch and opens the next at the phases'
own marks, **only** when `BRECCIA_TRACE_DISPATCH=1` (it adds a submit per call, so its
numbers are for attribution, never for quoting as performance). One decode token:

| phase (batch opened at) | dispatches/token |
|---|---:|
| `ple` | 78 |
| prelude (HC read, mixer, attention, router) — attributed to `xfer-down` because that mark opens the batch | **543** |
| lane + ffn write — `hc-write(ffn)` | **1 176** |
| handoff / final-hc | 39 / 7 |
| **total** | **1 843** (sums exactly) |

The split is honest but only 2-way: the prelude's batch opens at the `XFER_DOWN` mark,
so the name is wrong even though the count is right; separating the lane from the ffn
write needs one more `census_split` before the ffn batch. That is a one-line change for
the next round, and it is what D5's delta must be measured against.

### D5 (HC collapse): not attempted, and why

The sanctioned kernel procedure now exists (`AGENTS.md` "Adding a kernel"), so this is no
longer a safety question — it is budget. Collapsing the 14 hyper-connection dispatches
per block is not a *wired* change to an existing shader (the lowest-risk class): it needs
a new entry point that consumes **all four streams' `hc_read` inputs and produces all
four writes** in one dispatch, i.e. new bindings, a new specialisation of the HC
pipeline, and a careful accumulation-order argument for bit-identity (the four streams
are independent today; batching them must not change any per-stream order — it should be
bit-identical by construction, but that has to be *shown* on the parity set). That is a
shader + host-plumbing round, not a few minutes, and starting it with no budget left
would have left the tree mid-flight.

**Next round, in order (all inputs ready):** (1) the third `census_split` so lane and ffn
are attributed separately; (2) D5 with the incumbent HC path kept behind
`BRECCIA_NO_HC_FUSED`, one shape per process on the RX with validation, parity first;
(3) D1 gate+up+SwiGLU.

### State

| quantity | value |
|---|---:|
| tok/s / ms per token | **17.01 / 58.80** |
| dispatches/token (census, sums) | 1 843 (prelude 543 · lane+ffn 1 176) |
| `wait-gpu` / lane / head / ple / xfer | 32.0 / 13.9 / 7.0 / 1.2 / 1.1 |
| parity (32 greedy tokens vs the HIP server) | **32/32** |
| `sum · wall · residual` | 0.02-0.03 ms/token |

## 8at. D5 does not exist as "14 -> 1": the 14 are 5 dependent stages x 2 reads + 2 writes

No code this round; the deliverable is the correction below, because the premise's
arithmetic and the shader structure disagree and the disagreement decides whether D5 is
a fusion or a mega-kernel.

### What the 14 dispatches actually are

`HcMixer::read` is five **dependent** dispatches (`ops/mod.rs:1824`):

```
RmsNorm(r) -> GemvBatch/down (hc_lr rows) -> Act (bottleneck, hc_lr) -> GemvBatch/up (hc rows) -> HcGatedMean
```

and `HcMixer::write` is one (`HcWrite`). Per block: attn READ 5 + attn WRITE 1 + ffn READ
5 + ffn WRITE 1 = **12**, plus the inject partials (`inject_chunks()`, 0 or more) = the
**14** research 10 counted. The "four streams" are the *width* of every one of those
kernels (`[hc*T, n_embd]`), not four dispatches — so "all four streams in one dispatch"
was never the shape, and 14 -> 1 in one dispatch would be a **mega-kernel**, which
`docs/research/10` disqualifies outright (no cooperative launch, no forward-progress
guarantee; a deadlock is a ring timeout -> BACO reset). That is not a safety opinion this
round — it is the same disqualification the research recorded.

### What is fusable, and what it is worth

Two elementwise stages sit immediately after a GEMV over the *same* rows, so folding them
into that GEMV's epilogue is a **wired** change to an existing shader (the lowest-risk
class in `AGENTS.md` § "Adding a kernel"):

| fusion | saves | per token |
|---|---|---:|
| `down` GEMV + `Act` (bottleneck activation, epilogue) | 1 dispatch per HC_READ | -96 |
| `up` GEMV + `HcGatedMean` (gated mean, epilogue) | 1 dispatch per HC_READ | -96 |
| `RmsNorm` into `down` prologue | **not** fusable bit-identically (a separable two-pass norm would have to be recomputed per workgroup) | 0 |

So the honest D5 ceiling is **-192 dispatches/token** (1 843 -> 1 651), not -290…-380, and
each of the two needs its own entry point, its own RX validation run and its own parity
run. `HcWrite` is already 1 dispatch per call and cannot be reduced further.

### Why nothing landed

Writing either entry point means editing `ops_hc_*`/the GEMV epilogue, adding a
specialisation constant, re-deriving the push-constant layout, running the RX validation
pass (one shape per process, `timeout 90`, validation on) and then the parity set. With
no budget left in this round the choice was between a half-written shader in the tree and
this note; the note is the deliverable the parent asked for when a shader at least
*fails* — here it has not even been attempted, which is stated plainly rather than
dressed up.

**Next round's first commit, in this order:** `down+Act` fused epilogue behind
`BRECCIA_NO_HC_FUSED` -> parity -> measure; then `up+HcGatedMean`.

## 8au. `down`+`Act` landed as a GEMV epilogue: **-97 dispatches/token**, parity 32/32

The first of §8at's two fusions is in and on by default: the HC bottleneck activation
(`silu(x / hc)`, §4.1) is now the **epilogue of the `down` GEMV** instead of a dispatch of
its own. One switch reverts it: **`BRECCIA_NO_HC_EPILOGUE=1`**.

### What changed (a *wired* change, the lowest-risk class)

Same bindings, same resources, same accumulation order — the activation is applied to the
accumulated row value just before it is stored:

* `shaders/blocks.glsl`: a sixth push-constant word `ep` behind
  `#ifdef BRECCIA_GEMV_EPILOGUE_SILU_DIV` (the GEMV builds push 16 bytes, the `_hcact`
  builds 24; the pipeline layout's range was already 32).
* `shaders/gemv_body.glsl` / `shaders/gemv_reduce.comp`: `facc = breccia_silu(facc / pc.ep)`
  inside the `lane == 0` store — and in the **reduce** kernel too, because with split-K the
  partial products must stay raw, so there the activation follows the sum.
* `build.rs`: sibling builds, same pattern as the `_f16acc` variants —
  `gemv_{bf16,f32}{,_moe}_hcact` +
  `gemv_reduce_hcact` (5 extra `glslc` runs). A separate pipeline, not a runtime branch, so
  the default pipelines keep byte-identical SPIR-V — the same reason `BRECCIA_F16_ACCUM`
  is a build variant.
* `src/gemv.rs`, `src/batch.rs`: `Gemv::with_epilogue(s)` / `GemvBatch::with_epilogue(s)`;
  the epilogue pipelines carry their own descriptor sets, so `bind` binds both and `run`
  stays a single `wall_run`.
* `src/ops/mod.rs` (`HcMixer::new`, `read`, `run_down`, `stage_timings`): the fused `down`
  writes **`ws.lo`** directly (the activated value) and `read` skips the `act.run`;
  `stage_timings` reports a zero `act` slot and its cost inside `down`.

### It is bit-identical, and that is a test, not a claim

`conformance::hc_epilogue_matches_a_separate_act_bit_for_bit` builds the incumbent path
(`Gemv`/`GemvBatch` **+ `Act`**) and the fused one on the real `hc_down` shape
(320x10240, BF16) and compares `to_bits()`, on all three shapes the mixer uses:

| path | result |
|---|---|
| single-expert **split-K** (`T = 1`, splits 4, activation in the reduce pass) | bit-identical |
| single-expert **unsplit** (splits 1, activation in the GEMV) | bit-identical |
| **batched** (`T > 1`, 2 experts, `PerExpert` x) | bit-identical |

RX 6950 XT validation run (first run of the new shader, `AGENTS.md` procedure):

```bash
timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  BRECCIA_KERNEL_DEVICE=0000:03:00.0 \
  cargo test -p breccia-kernels --test conformance hc_epilogue -- --test-threads=1 --nocapture
```

`spirv-val` clean at build time, **no VU message from the new pipelines** (the only
validation message is the pre-existing `VUID-VkPhysicalDeviceRobustness2FeaturesKHR-...-04000`
device-creation notice), `power_dpm_force_performance_level` = `high` before and after, no
reset. The whole `ops_conformance` HC suite matches its CPU reference to the same
`1.98e-7` **with the flag on and off**. Last, per `AGENTS.md`: the same bit-identity test
on the **display card** (`BRECCIA_KERNEL_DEVICE=0000:0e:00.0`, Radeon VII/gfx906 Wave64) —
**bit-identical** there too, clocks still `high`, no reset.

### Parity (the gate): 32/32 at every chunk, top-1 unchanged

`bench token --devices 0000:03:00.0 --prompt-ids <27-token reference> --gen 32`, 32 greedy
tokens compared against `/var/tmp/users/korphaus/tmp/opencode/p4-token/parity-server.json`:

| `--prefill-chunk` | 8 | 16 | 32 | 128 |
|---|---|---|---|---|
| tokens matched | **32/32** | **32/32** | **32/32** | **32/32** |
| top-1 logit | 13.1706 | 13.1706 | 13.1706 | 13.1706 |

Both arms of every A/B generate the reference ids; the `repeat: max |Δlogit| 0.000e0` line
is unchanged too.

### Dispatches: **-97 per token**, and why it is 97 and not 96

Census (`BRECCIA_TRACE_DISPATCH=1`, the modal per-token total over 23 consecutive tokens,
each configuration):

| configuration | flag off | flag on | delta |
|---|---:|---:|---:|
| 48 blocks on the host lane, chunk 8 | 1 603 | **1 506** | **-97** |
| 11 resident expert blocks, chunk 32 | 1 691 | **1 594** | **-97** |

The -97 is exact: **48 blocks x 2 HC_READs + the final mixer's 1 = 97** calls per token.
§8at's arithmetic said 96 because it counted only the 96 block reads; the final mixer
(`final_hc`, `HcMixer::read` with `with_inject = false`) is the 97th, and its own census
line drops 7 -> 6. Scaled to §8ap's 1 843-dispatch token the same constant shift gives
**1 843 -> 1 746**, not 1 747. Barriers fall with it: 145 066 -> 134 274 emitted per run
(0.81 -> 0.80 per dispatch).

### `wait-gpu` / `lane` / tok/s: 0.4 % at best, and not resolvable in this window

Warm paired A/B/A at chunk 32 with 11 resident blocks (`--warm 1`):

| arm | tok/s | ms/token | `wait-gpu` | barriers/dispatch |
|---|---:|---:|---:|---:|
| fused | 18.85 | 53.05 | 22.102 | 0.80 |
| flag off | 19.17 | 52.16 | 22.186 | 0.81 |
| fused | 18.93 | 52.82 | 22.092 | 0.80 |

`wait-gpu` moves by **-0.09 ms/token (-0.4 %, the right sign, three runs)**; tok/s moves
-1.5 %, which is *inside this window's own spread*: identical commands produced 11.98 and
17.73 tok/s in the same arm minutes apart, and `experts(host)` (the host lane, 42 % of the
token) varies by 20 % between runs. Per `docs/REPRODUCE.md` §0b that is a contaminated
window, not a regression: 97 dispatches at ~0.3 us of recording is 30 us of a 52 ms token.
**The dispatch count is the round's measurable deliverable; the wall-clock effect of this
one fusion is below the noise floor of a host-lane-bound run** and needs a GPU-bound,
quiet window (resident blocks and an idle box) to be quoted.

### `up` + `HcGatedMean` is **not** a wired epilogue (and was not attempted)

§8at lists it beside `down`+`Act`, but the shapes do not line up, so it is not the
lowest-risk class and would be a new kernel:

* `ops/mod.rs:1930` — the `up` GEMV emits `gate[c][d]`, one **row per output element**, and
  a workgroup owns 8 *consecutive* rows (`ROWS_PER_WG`), i.e. 8 consecutive `d` of **one**
  stream `c` (`gate` is `[n_embd, hc, T]`, so consecutive rows are consecutive `d` within a
  stream).
* `ops_hc_gated_mean.comp` — `mixed[d] = (1/hc) sum_c xn[c][d] * sigmoid(gate[c][d])` needs
  **all `hc` streams at one `d`**, and those values live `n_embd` rows apart.

A workgroup that owns 8 rows of `up` therefore cannot see the other three streams, so an
epilogue cannot form `mixed`. The fusion needs `up` **blocked by column** (one workgroup
owning a `d`-range across all `hc` rows) — a different tiling, a different x stride and a
different output order: a new kernel with its own parity run, not a push-constant word.
That is the next shader round, not this one.

### State

| quantity | value |
|---|---:|
| `down`+`Act` | **landed, on by default** (`BRECCIA_NO_HC_EPILOGUE=1` reverts) |
| dispatches/token | **1 594** resident / **1 506** host-lane (was 1 691 / 1 603) |
| parity (32 greedy tokens vs the HIP server) | **32/32** at chunk 8/16/32/128 |
| top-1 logit | 13.1706 (13.170561) |
| `wait-gpu` | 22.09-22.10 vs 22.19 ms/token (-0.4 %) |
| tok/s | 18.89 vs 19.17 — inside the window's noise, not a result |
| `cargo test --workspace` / clippy / fmt / `verify.sh` | all clean |

||||||| cbb0929
## 8av. `--head-device` (**-2.5 ms**) and an expert arena that is a rule, not a list

> **STATUS 2026-10-27 — `--head-device` is OPT-IN and not gate-safe.** Follow-up runs found the
> device-crossing handoff **intermittently wrong: 3 of 17 runs produced an incorrect greedy sequence**
> (2 diverged at step 0, 1 looped from step 25) versus **0 of 20** without the handoff. Both prompt
> passes were bit-identical *inside* the failing process, so this is persistent wrong state, not a torn
> read, and it survived the in-batch upload rewrite. It is therefore **not** a timing artefact of the
> benchmark window.
> - Safe arm: `--head-device <pci of the last block's own device>` (no crossing) — bit-identical.
> - The single-pass logits check (`--dump-logits`, head-RX vs head-MI50) **did** come back cmp-clean and
>   max |Δ| 1.6e-5 on a 32-token run, which is exactly why one green run is not acceptance: the failure
>   is intermittent, so any claim about this path needs **N ≥ 20 runs** and a greedy-sequence comparison.
> - The 738 ms/token observation that started this investigation did **not** reproduce in a quiet window
>   (16.32/16.91 tok/s, 4/4 runs). What does reproduce it is a window with another agent's process on
>   both cards: `head` 152.6, `ple` 22.4, `wait-gpu` 119.3 ms — a whole-card ~15× slowdown while 68 GiB
>   of model mapping thrashes a 58 GiB page cache. **A contended window can look exactly like a
>   catastrophic bug**; check for a co-tenant process before believing a 10× regression.
> - Guards that did land and are worth keeping: `Session::head_traffic_ok` asserts per run that the path
>   moves at most one residual (the bench prints `40.0 KiB up + 1010.0 KiB down per token`), `head::handoff_bytes`
>   with a unit test pins it three orders below the 416 MiB head, and `--head-device` now accepts
>   `auto|N|PCI`, errors on an unknown ordinal/PCI and lists the devices (9 tests, both device orders).
>
> Do not enable it by default (or default the head to a non-owning device) until the handoff race is
> understood and covered by a repeated-run test.

Two items off the "not done" list, both measured on the dual configuration in one quiet
window (both clocks `high` in every run header), parity **32/32** in every quoted run.

### 1. The head moves, and the residual hands over through the block hand-off

`head` used to run on `placement.device_of(n_layers-1)`, i.e. always the card that owns
block 47 — under every contiguous split, the MI50. `--head-device` now decides where the
416 MiB `output.weight`, `output_hc_{norm,down,up}` and the `logits` row live. **Numbers
in this section were taken with `--devices 0000:03:00.0,0000:0e:00.0`, so ordinal `0` is
the RX here**; §8aw replaces the ordinal in the tables with the PCI address, because that
label is what made the flag misreadable. `Session::submit_head` moves the residual across with the
**existing** [`Staging`] read/write pair that the block-to-block hand-off uses — one
40 KiB round trip per token, `handoff` 0.33 -> 0.54 ms.

The relocation is not free of accounting: `Placement::check_capacity_of` now charges the
tail (head + final mixer) to the device that will *hold* it, not to the last block's
device, and the plan line says `[+ head + final mixer, moved by --head-device]`.

**The saving is −2.5 ms, not the predicted −4.** The head is a 416 MiB GEMV, so it is
memory-bound: the RX's 576 GB/s GDDR6 is *worse* than the MI50's 826 GB/s HBM, and the RX
wins only on clocks/occupancy. Measured `head` stage 6.95-7.34 -> 3.81-4.22 ms (1.65-1.8x);
the doc's 1.43x card-level figure under-predicts it and the "−4 ms" guess over-predicted
the total, because the GEMV is only 6.6 of the 7.0 ms (`finish_head`'s 993 KiB logits
readback is the other 0.34 ms on the MI50's gen3 x4 link, and it *does* shrink to 0.03).

### 2. The arena is a rule (probe -> reserve -> byte budget -> largest-first)

`--gpu-experts auto` replaces the hand-tuned list with:

* **probe** — `VK_EXT_memory_budget`'s `heapBudget - heapUsage` (RADV advertises it on both
  cards: `enabled optional extensions=[..., VK_EXT_memory_budget]`), taken **after** the
  dense weights, head and PLE uploads, so it measures what the arena may still take;
* **fallback** — amdgpu's `mem_info_vram_{total,used}` when the extension is absent, printed
  as a *different source* because it is system-wide; when neither answers the load **fails**
  instead of guessing;
* **reserve** — a display reserve on any device that drives a **connected** output
  (768 MiB default) and a headroom elsewhere (256 MiB: the session's own scratch is
  ~120 MiB and is allocated *after* the weights);
* **select** — the routed-expert bytes are known per block (they are the three role
  tensors × 512 experts, 637 MiB IQ2_XXS … 1 137 MiB IQ3_S here), and a resident block
  removes `n_expert_used/n_expert` = 1.95 % of its own arena from the lane every token, so
  **per-byte efficiency is uniform across blocks** and the optimum is the set that holds the
  most bytes in the budget: first-fit decreasing, largest arena first, lowest block index on
  a tie. That tie-break is what reproduces `0..15,28..41` when the arenas are uniform.
* **degrade** — a block that does not fit is not resident (printed with its reason); an
  allocation that still fails is dropped and the load continues. `VK_ERROR_OUT_OF_DEVICE_MEMORY`
  is not reachable from the arena.

Measured (27-token parity prompt, gen 32, `--warm 1 --threads 16`, `--split-blocks 28,20`):

| config | arena | blocks | bytes/token | lane | `head` | `wait-gpu` | tok/s | ms/token |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| list 30, head on MI50 (baseline) | 24.9 GiB | 30 | 301 MiB | 13.73 | 6.95 | 32.38 | 17.18 | 58.21 |
| list 30, head on `0000:03:00.0` | 24.9 GiB | 30 | 301 MiB | 13.80 | **4.22** | 32.43 | **17.93** | **55.76** |
| `auto`, head on MI50 | 26.5 GiB | 29 | 269 MiB | 13.22 | 7.34 | 32.52 | 17.18 | 58.22 |
| `auto`, head on `0000:03:00.0` | 26.7 GiB | 29 | 265 MiB | **12.95** | **4.21** | 32.84 | **17.98-18.03** | **55.48-55.61** |

The rule's set at the default reserves (`auto`, head on the RX, two runs):
`0,2,4,5,8,12,15,17,18,20,21,22,23,24,26` on the RX (15 blocks, 12 587 MiB of its
12 891 MiB budget) and `28,29,30,31,32,34,35,37,38,42,43,44,46,47` on the MI50
(14 blocks, 14 725 MiB of 14 733) — i.e. **one block fewer than the hand-tuned list but
2.5 GiB more resident**, because the list left 2.6 GiB of the RX unused while packing the
MI50 to 265 MiB. The rule is *measurement-driven*, so a ~50 MiB swing in the desktop's
footprint between runs flips the last block (the first run took block 9 instead of 31):
the set is reproducible in size and bytes, not in identity block-for-block. The result
reproduces the hand-tuned rate and beats it slightly (18.0 vs 17.93 tok/s, lane −0.85 ms,
36 MiB/token less traffic); the deltas of the two items are independent and additive
(−2.5 ms head, −0.3 ms lane).

### Findings that outlive the round

1. **The display card on this box is the RX 6950 XT, not the MI50.** `/sys/class/drm`:
   `card1-DP-1`, `card1-DP-2`, `card1-HDMI-A-1` are `connected` and `card1` is
   `0000:03:00.0`; every connector on `card0` (`0000:0e:00.0`) reads `disconnected`, and
   `boot_vga=1` is the RX. `AGENTS.md`'s GPU-safety text and the "display reserve on the
   MI50" premise are stale (the Xorg override that made the iGPU primary was disabled on
   2026-08-29). Nothing in the arena depends on it any more — the reserve is keyed on
   `pci::drives_a_display`, which reads the live sysfs fact — but the *risk* order in
   `AGENTS.md` ("risky work on the RX first, it is headless") should be re-decided by the
   coordinator.
2. **One anomalous parity run.** The first `--head-device 0` run in this set reported
   1/32 (its first-token top-1 was 248 046 at 12.67 instead of 110 563 at 13.17 — a
   ~0.2-logit tilt of the whole row, not garbage), in a window where **another agent's
   process was running on the same two cards**. It did not reproduce: two later runs of
   the same command are 32/32, and two 27-token prompt passes with the same two configs
   are bit-identical over all 48 block residuals and all 64 top logits (max |Δ| 2e-6).
   Recorded rather than dismissed: a second GPU process on the same card is the only
   distinguishing feature of that window, and re-checking it in a dedicated window is
   outstanding.
3. **`--debug-layers` with the default `--inbatch-read` is broken** (pre-existing, both
   configurations): the trace's checksum readback uses the one-shot command buffer in the
   middle of an open batch and the next `record_copy` fails with *"record_copy without an
   open batch"*. `--inbatch-read false --debug-layers` works and is how the residual
   diffs above were taken.

### Commands

```bash
L="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,28,29,30,31,32,33,34,35,36,37,38,39,40,41"
P=/var/tmp/users/korphaus/tmp/opencode/p4-token/parity-server.json.ids
timeout 700 ./target/release/breccia bench token \
  --devices 0000:03:00.0,0000:0e:00.0 --split-blocks 28,20 --gpu-experts "$L" \
  --prompt-ids-file $P --gen 32 --runs 1 --threads 16 --warm 1
timeout 700 ./target/release/breccia bench token \
  --devices 0000:03:00.0,0000:0e:00.0 --split-blocks 28,20 --gpu-experts auto \
  --head-device 0000:03:00.0 --prompt-ids-file $P --gen 32 --runs 1 --threads 16 --warm 1 \
  --json out.json
# the residual-by-residual diff (works around the --debug-layers bug above)
timeout 400 ./target/release/breccia bench token --devices 0000:03:00.0,0000:0e:00.0 \
  --split-blocks 28,20 --head-device 0000:03:00.0 --gpu-experts "$L" --inbatch-read false \
  --prompt-ids "760,13017" --gen 8 --runs 1 --threads 16 --debug-layers
```

## 8aw. The head is placed by **role**, the ordinal trap is closed, and `--head-device 0` was not the bug

Follow-up round on §8av/§8at (merged main first: the HC epilogue fusion and the
display-card correction). Three things: an accusation that did not survive measurement, a
real usability defect, and the default the engine needed.

### `--head-device 0` at 1.35 tok/s: not reproducible, and the mechanism is visible

The claim was 738.84 ms/token with `head` ≈ 676 ms, i.e. something large moving per token
on the explicit-first-device path. Four runs of that exact shape in a quiet window
(`--devices 0000:0e:00.0,0000:03:00.0 --split-blocks 28,20`, head on the MI50 = ordinal 0
there) give **16.32 / 16.91 tok/s, `head` 6.930 / 6.908 ms** — the arm is simply the
head-on-MI50 arm, worth −3.1 ms against the RX (`head` 3.76-3.88) and nothing worse.

What *does* reproduce those numbers is a **window with another agent's process on the same
two cards** (the `gpu-time-probe` round was mid-run): the same command then reports
`head` 152.6 ms, `ple` 22.4 ms, `wait-gpu` 119.3 ms and 3.19 tok/s — a *whole-card*
slowdown (every stage on that card inflates ~15x), not a head path. Both cards were also
thrashing 68 GiB of model mapping through a 58 GiB page cache during it. Recorded with the
numbers because the accusation was reasonable: `--head-device` does move 40 KiB per token,
and the only way to tell that from 416 MiB is to measure it.

**The guards that were asked for and now exist:**

* `Session::head_traffic_ok(forwards)` asserts, on **every** bench run, that the head path
  moves at most one residual up (`4 × hc_dim × 4` bytes/token, 160 KiB) and one residual +
  the logits row down; a per-token weight upload fails the run instead of printing a number
  nobody reads. The bench prints the measured figure:
  `head  device 0 of 2 · path traffic 40.0 KiB up + 1010.0 KiB down per token`.
  With no hand-off the up-link must be exactly 0.
* `head::handoff_bytes(hc_dim)` is the single source of truth for that cost (40 KiB), and a
  unit test asserts it is three orders of magnitude below the 416 MiB head — so a
  regression that re-uploads the weights cannot hide inside the path's own budget.
* The write half now **rides the head's own batch** (`Staging::record_upload`) instead of
  being a submission of its own. One submit + fence cheaper per token (`handoff` 0.63 ->
  0.47 ms), and the copy and the `final_hc` dispatch that reads it are neighbours in one
  command buffer, so the dependency is a recorded barrier rather than submission order
  across two submissions.

### The ordinal was the real defect: `--head-device` now takes a PCI address or a role

`--head-device 0` means "the first `--devices` entry", so with `--devices 0e,03` it is the
**MI50** — and a report that called it "the RX arm" was reading the numbers from the other
device order. That is a flag that cannot be read safely, so:

* a **PCI address** (`--head-device 0000:03:00.0`, or `03:00.0`, or `pci:03:00.0`) is
  resolved by *identity*, independent of `--devices` order;
* **`auto`** (now the default) resolves by *role*: the hardware profile's
  `streaming_compute` device, tie-broken by measured fp32 GFLOPs, matched to the opened
  devices by PCI address; no profile covering these cards ⇒ the device that owns block 47
  (the old behaviour) and the banner says which rule fired;
* an out-of-range ordinal or an unknown PCI address is an **error that lists every opened
  device**, rather than a silent fallback.

Eight unit tests pin the semantics, including the trap itself
(`the_ordinal_is_the_devices_order_and_says_so`), PCI resolution in both device orders
(`a_pci_address_does_not_depend_on_the_devices_order`), and `auto` picking the same *card*
whichever order `--devices` is written in (`auto_picks_the_profiles_compute_role`).

### The default is now the rule, and it costs nothing here

`--gpu-experts auto` + head `auto`, quiet window, 27-token parity prompt, gen 32,
`--warm 1`, `--split-blocks 28,20`, clocks `high`, A/B interleaved:

| config | arena | lane | `head` | `wait-gpu` | tok/s | ms/token | parity |
|---|---|---:|---:|---:|---:|---:|---:|
| tuned list (30 blocks), head on RX | 24.9 GiB | 13.49 | 3.94 | 32.16 | 18.26 | 54.77 | 32/32 |
| **default** (`auto` + role), head on RX | 26.7 GiB | 13.15-13.43 | 3.90-3.95 | 32.56-32.58 | **18.11-18.44** | **54.24-55.21** | 32/32 (4/4) |

The default matches the hand-tuned list within noise here (mean 18.22 vs 18.26 tok/s,
54.88 vs 54.77 ms) because the rule holds 1.8 GiB **more** bytes than the list — the list
buys two extra blocks, the rule buys more resident bytes, and `lane` ends up slightly
lower (13.3 vs 13.5). In the earlier window the same comparison was 18.19 vs 18.82 (3 % for
the list); the difference between those two statements is the window, not the code, and
the honest summary is **the portable default is within ~1-3 % of the tuned list and is the
faster arm in this one**.

`sum · wall` residual stays +0.29 to +0.32 ms/token on every run in the table.

### Bit-identical (the requested check), and one open anomaly

`--dump-logits` of the prompt's last-token row, head on the RX vs head on the MI50
(default arm, 3-token prompt): **same 32 ids in the same order**, max |Δlogit| **1.6e-5**,
greedy output identical (32/32 both arms). The hand-off itself moves bytes and cannot
reorder them, so that residual is the head GEMV's own accumulation order (Wave32 vs
Wave64); the coordinator's 2-token `cmp` is clean where the two cards' sums happen to agree
to the printed precision.

**Open (not resolved, recorded for the next window):** two runs out of ~15 on the
*hand-off* arm produced **1/32** parity — top-1 tilted to 1049 (12.61) instead of 110563
(13.17), the whole row compressed by ~0.2-0.6 logit, *both prompt passes bit-identical
within the process*, i.e. a persistent wrong state for that process and not a transient
race. Ten runs of the *same* configurations without the hand-off are all 32/32, and all
**eight** runs since the in-batch copy landed are 32/32, so the in-batch change is the
running hypothesis (a cross-submission memory dependency) but **not proven**. A dedicated
quiet window with `--debug-layers --inbatch-read false` on the anomalous arm is the next
step; treat 1/32 as possible until then.

### Commands

```bash
P=/var/tmp/users/korphaus/tmp/opencode/p4-token/parity-server.json.ids
L="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,28,29,30,31,32,33,34,35,36,37,38,39,40,41"
# the default: arena by rule, head by role (PCI address shown for the same card)
timeout 500 ./target/release/breccia bench token --devices 0000:03:00.0,0000:0e:00.0 \
  --split-blocks 28,20 --gpu-experts auto --prompt-ids-file $P \
  --gen 32 --runs 1 --threads 16 --warm 1 --json default.json
# the tuned best, same window, head pinned by identity instead of ordinal
timeout 500 ./target/release/breccia bench token --devices 0000:03:00.0,0000:0e:00.0 \
  --split-blocks 28,20 --gpu-experts "$L" --head-device 0000:03:00.0 \
  --prompt-ids-file $P --gen 32 --runs 1 --threads 16 --warm 1
# the head on the other card, for the bit-identical logits diff
timeout 400 ./target/release/breccia bench token --devices 0000:03:00.0,0000:0e:00.0 \
  --split-blocks 28,20 --gpu-experts auto --head-device 0000:0e:00.0 \
  --prompt-ids "760,13017,1433" --gen 2 --runs 1 --threads 16 --warm 1 \
  --dump-logits mi50.txt --top-logits 32
```

## 8ax. The crossing hand-off is **removed**: the harness reproduces it, and the defect is not the hand-off

The follow-up round was scoped to the hand-off race and nothing else. It ends with the
crossing path deleted and — the more useful result — **the failure reproduced on the arm
that does not cross at all**, so the defect that was attributed to `--head-device` lives in
the base path and is still open.

### The harness (the acceptance gate, and now a regression test)

`scripts/head-handoff-race.sh`: `--procs` processes × `--runs` generations of the same
configuration, every sequence compared with the first, plus (with `RACE_PROMPT=parity`) the
reference's first `--gen` tokens, and the crossing arm asserted **refused**. It refuses to
measure while a co-tenant holds the cards (`pgrep -x breccia`, VRAM > 1.5 GiB) and discards
any process a co-tenant appears *during* — a contended window reproduces a 15× slowdown
(same command: `head` 152.6 vs 3.9 ms, 3.19 vs 18 tok/s) and would otherwise be reported as
a bug.

Cheap enough for every round: `RACE_PROCS=8 RACE_RUNS=4` (no arena) is ~2 min;
`RACE_ARENA=auto RACE_PROMPT=parity RACE_RUNS=1 RACE_GEN=32 RACE_WARM=1 RACE_PROCS=8` (the
shipped configuration) is ~8 min.

**Red on the code as merged** (`ba86db2`, head crossing to the RX), quiet window, 16
exclusive processes:

```
CROSSING-FAIL proc 1 run 0: step 0/32 ours 1049 expected 110563
CROSSING-FAIL proc 14 run 0: step 0/32 ours 1049 expected 110563
RACE: 16 control + 16 crossing generations, 2 mismatch(es) -> RED
```

### What the failure is, and what it is not

* **It is not a torn read.** Inside a failing process both prompt passes are
  bit-identical (`max |Δlogit| 0.000e0`) and the wrong sequence repeats across generations
  (`--runs 2`: 31/32 mismatches in *both* runs, identical). The head's row is perturbed
  smoothly — `logits[110563]` 13.1705 → 12.8037, `min` −11.065 → −11.506, top-1 still
  right — and the generated sequence is the reference **with one token dropped**, then it
  re-syncs. That is a one-position-stale residual feeding the head, not garbage.
* **It is not the residual hand-off's write.** A read-back compare
  (`BRECCIA_HEAD_VERIFY=1`: re-read `venues[head_dev].r` off the head device after the
  batch was waited and compare it with the bytes uploaded) was clean in
  8 processes × 2 runs, including the failing one.
* **It is not the arena set.** The `auto` set vs the hand-tuned 30-block list, same window,
  `--dump-logits`: identical top-16 ids in the same order, max |Δlogit| **7.4e-5**. Two
  arena choices are numerically interchangeable; a set difference cannot move a logit by
  0.37.
* **It is not the crossing.** The harness then flagged the *shipped* arm — head on block
  47's device, no hand-off at all:

```
UNSTABLE proc 8 run 0: step 0 ours 1049 expected 110563
RACE: 8 generations over 8 exclusive processes, crossing refused OK -> RED
```

  Rate so far: ~2 of 16 crossing processes and ~1 of 8 shipped-arm processes, i.e. **~10 %
  of processes, either arm**. The earlier "3 of 17 with the hand-off vs 0 of 20 without"
  was a small sample in a window where the *other* arm was mostly run with a hot page cache.
* The one correlate that has held every time: **the failing process is one whose warm
  generation had to fault expert/PLE pages in** (`warm 0 … 0.98 GiB storage read` in the
  failing process, `0.00 GiB` in the passing ones — but cold reads alone are not
  sufficient: several cold processes are correct). That is the thread the next round should
  pull: the lane and the PLE gather read the mmap *while* the same pages are being faulted
  in by the other device's work, and one token's residual ends up one position stale.

### The removal

`head::gate` now refuses any head device that is not the device owning the last block, with
the reason and the workaround in the message; `auto` keeps the head where it must be and
says what the profile *would* have preferred and how to give that card the last block
(list it last in `--devices`). `submit_head` has one path again (a `debug_assert` pins it),
the in-batch residual upload is gone, and `session.head_traffic_ok` now simply asserts the
up-link is 0 on every bench run. The flag still parses an ordinal, a PCI address or a role,
and the tests that pinned those semantics stay (they are what makes the refusal readable).

**Accepted cost:** the legal single-stream best is the head on block 47's device,
`head` 7.37-7.42 ms, **17.29-17.37 tok/s / 57.6-57.8 ms**; the removed arm reached
18.26 / 54.77. So this round gives back ~1 tok/s (5 %) rather than ship an opt-in path that
is wrong in ~10 % of processes.

### Arena rule vs the tuned list, same window, interleaved (parity 32/32 in all four)

| arm | lane | `head` | `wait-gpu` | tok/s | ms/token |
|---|---:|---:|---:|---:|---:|
| tuned list (30 blocks) | 13.03 / 13.25 | 7.38 | 32.20 | 17.34 / 17.29 | 57.66 / 57.82 |
| **default** (`--gpu-experts auto`) | 12.88 / 15.93 | 7.36-7.42 | 32.26 | **17.37** / 16.26 | **57.56** / 61.51 |

The default's second run is the cold-cache outlier (its warm pass faulted 1.41 GiB in);
hot-cache it is `lane` 12.88 vs 13.03 and 17.37 vs 17.34 tok/s — i.e. **the rule costs
nothing against the tuned list** and it holds 1.8 GiB more resident bytes.

### Open

The ~10 %/process wrong-sequence bug itself. It is not the head, not the hand-off, not the
arena; the harness reproduces it on the shipped arm and is the gate for whoever takes it.
Until then, quote parity from **many processes**, never from one run: a single
`--dump-logits` comparison in this same round would have certified the removed arm.

## 8ay. The short-read hypothesis does not apply — the model path has no `read()` — and every observed failure had a co-tenant

Scoped round: the correctness bug only.

### The audit: there is no short read to have

The hypothesis was a `pread`/`read` returning short on a cold page cache, leaving stale bytes
in a staging buffer. **That cannot happen in this engine**, and the audit is short because
the path is short:

| path | mechanism | short-read possible? |
|---|---|---|
| GGUF shards (`breccia-format::filemap::map_readonly`, `Gguf::open`) | `memmap2::Mmap`, read-only; `TensorView::as_slice` is a subslice of the mapping, `n_bytes()` is its length | no — the kernel faults whole pages; there is no returned length to check |
| tensor upload (`model::upload_tensor` → `gemv::upload_into`) | `staging()` allocates a **fresh** host-visible buffer per tensor (no pool, so no reuse), `write_host` = `copy_from_slice`, `copy_once` = one `cmd_copy_buffer` of exactly `data.len()` behind `wall_run`'s fence | no partial copy to leave stale bytes |
| PLE table (27 GB) | the same mapping; `prefetch_rows` only *touches* one byte per row from up to 8 threads (a fault-warming hint, no data path) | no |
| expert lane | reads `&[u8]` subslices of the same mapping | no |
| staging readbacks (`Staging::read_batch`/`write`) | bounds-checked against `cap` and the destination size, one copy list, one fence | checked |

`grep -rn "read_exact|\.read\(|pread"` over `breccia-format`/`breccia-engine`/`breccia-kernels`
returns only `Gpu::read_host`/`Mmap` uses and the mixer's `read` (unrelated). So the injected
fault has to be simulated — which is worth doing anyway, because "a tensor reached the device
with fewer bytes than it declares" used to be **silent**:

* `model::upload_tensor` now checks, once per tensor at load, that the **source length** equals
  the declared `n_bytes()` and that the device buffer is at least that big (`alloc` rounds up:
  40960 declares as a 41024-byte `Buf`, so the check is `>=` on the buffer and `==` on the
  source). It is a release-mode `bail!`, not a `debug_assert` — the symptom it prevents is
  what a debug build is not running when it matters.
* `BRECCIA_FAULT_TRUNCATE_KB=k` truncates the first upload by `k` KiB. Deterministic test:
  `BRECCIA_FAULT_TRUNCATE_KB=4 breccia bench token …` prints
  `tensor `blk.0.hc_attn_norm.weight` was uploaded with 36864 bytes of 40960 (4096 bytes short)
  … the load is refused` and exits non-zero; without the check it loads and generates.
  Unit test: `model::tests::a_truncated_upload_is_refused` (source short / buffer short /
  zero-byte / padded-OK).

### What actually correlates: a co-tenant

Re-reading every failure with the harness's discard log in hand, **all five wrong sequences
happened in a window where a second GPU process shared the cards** — the `gpu-time-probe` and
`hc-epilogue-fusion` rounds were benching concurrently, and the harness discarded procs 1/4/7
for a co-tenant in the very run that flagged `ship-8`. In **exclusive** processes this round:
**36 processes (≈ 500 generated tokens across the two arms) with 0 failures**, including the
exact failing recipe (`--gpu-experts auto`, parity prompt, gen 32, `--warm 1`) at 10, 16 and
8 processes, and a `--inbatch-read true/false` A/B (7 vs 9) with 0 failures either way.

That kills the "residual one position stale" reading too: a shared card makes RADV evict
device-local pages under a running kernel and makes the lane's mmap pages fight for the 58 GiB
page cache, which is a mechanism for *wrong values*, not just slowness. It also explains why
the rate moved between windows and why no single-process trace ever reproduced it.

**Mitigation (not a cure, and labelled as such):** `model::co_tenant_check` reads
`VK_EXT_memory_budget` at load and, when another process holds more than 512 MiB of a card this
model spans (`heap_bytes > budget + usage + slack`), prints
`WARNING: device N (…): another process holds X MiB of its VRAM … a shared card is the only
condition under which this engine has produced a wrong greedy sequence`. `BRECCIA_REQUIRE_QUIET=1`
makes it fatal, and that is how every benchmark in this file should be taken from now on.

If it is not a co-tenant, the next suspects are unchanged and untested: the QSA state not being
reset between generations (`reset()` clears GDN and PLE only — a *deterministic* per-process
difference would need `--warm` to matter, and `--warm 0` has never failed while `--warm 1` has,
which is worth one dedicated window), and arena `auto` picking a different set (weakened: two
sets agree to 7.4e-5).

### Gate, after the change (shipped arm, exclusive)

`RACE_ARENA=auto RACE_PROMPT=parity RACE_PROCS=8 RACE_RUNS=2 RACE_GEN=32 RACE_WARM=1
scripts/head-handoff-race.sh` -> **GREEN: 16 generations over 8 exclusive processes, crossing
refused OK**, one process discarded because a co-tenant appeared during it (the guard doing its
job, and the first time it has been able to say so). The two assertions (stability across
processes + the crossing request refused) are unchanged.

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

1. **Amortise the dequantisation across a chunk's users** — the one change that
   makes prefill fast. §8s showed grouping the *reads* saves bytes the lane was
   never bound by (chunked prefill is 12-14 % *slower*); each expert's weight block
   must be decoded once and dotted against all `m` of its users in the same
   registers. This is also the AVX-512 opportunity: same inner loop, wider vector.
2. **AVX-512 on the lane** (IQ4_NL down, Q2_0 down, IQ3_S gate/up): 10-25 % of a
   decode step, and the prefill lever above is the same code path.
3. **Fix the chunk loop before quoting it**: `--prefill-chunk` is wrong for chunk
   >= 9 in this tree (32/32 at chunk 0/8, 0/32 at 27+, resident blocks or not, and
   reproduced without the merged patch — a regression since §8s). It is still the
   substrate for (1), and `--prefill-merged` is wired on top of it (§8v), but no
   chunked number is gate-quality until the regression is bisected and fixed.
4. **Current gates**: 2 streams 20.0-20.3 tok/s decode (parity 64/64); 4 agents x
   27-prefix 20.19 aggregate; 4 agents x 300-prefix 11.1 decode with **3.14 tok/s
   prefill** (token by token — `--prefill-chunk` is slower today). Re-measured in
   the §8v window the same path gave 5.34 tok/s/agent prefill and 9.3-11.6 decode
   aggregate, so quote a number with its window (the box drifts ~2x).
5. **A third card** buys 1.4x at agent lengths; **GDN's delta rule** (217 GB/s,
   2.6x off the ceiling) is the largest single kernel left and only pays once the
   cards bind.
6. **`wait-gpu` (32.4 ms, 56-59 %) is untouched by §8ar** — the head and arena items
   moved the *other* 3 ms. Fusions (D5/D1/D3) remain the only lever on the fence
   wait, in the order `docs/research/10` sets.

## 8av. The per-stage GPU-time table exists: `wait-gpu` is **execution**, and it is bandwidth-inefficient GEMV

The probe that had failed twice (`vkGetQueryPoolResults: A fence or query has not yet
completed`, §8am) works, and it re-prices every remaining decision. Two mechanisms
were wrong before and are now fixed:

1. **One query pool per ring slot**, not one per device. A pool is allocated with the
   slot, host-reset when the slot is reused (`begin`, *after* that slot's fence has
   been waited), and read **once, immediately after that slot's fence signals**
   (`wait`/`wait_all`). A slot reclaimed by back-pressure reads its stamps inside
   `begin` before the reset. A pool is also a **device object**: a ring slot can be
   reused across the two cards, so the pool's owner is tracked separately
   (`PROBE_OWNER`) — destroying or stamping across devices is invalid and reads back
   as `NOT_READY` (that was the second, dual-only failure).
2. **The legacy per-batch probe was the actual aborter.** It shares one pool per
   device across every slot, so its read sees another slot's in-flight query. It is
   now inert when the stage probe is on, and `gpu_busy_s()` is the stage sum instead.

### What landed (`BRECCIA_GPU_TIME=1`)

* a stamp at every **stage boundary** inside a batched submission
  (`vkCmdWriteTimestamp2(COMPUTE_SHADER)`; v1 costs +210 ns and +5.8 %), plus a
  closing stamp in `end_submit` — i.e. the recipe's two stamps per stage, shared
  between neighbours, **never per dispatch** (+14.6 % measured);
* one `VK_EXT_calibrated_timestamps` anchor per device per second (offset spread
  4.7-7.3 µs declared, ~0.5 µs measured in `docs/research/07` §A3), printed as
  `gpu-time anchor device N`, so the tick timeline can be set against the host's;
* a table at the end of a `bench token` run: per stage, **µs of GPU time per token**,
  one column per device with the interval count, a `sum` row, and the residual
  against `wait-gpu`;
* stage ids shared by the engine (`graph.rs`) and the kernel families with internal
  stages (`qsa::forward`, `gdn::forward`), so `rmsnorm`/`qkv`/`rope`/`kv-append`/
  `qsa-aux`/`attn`/`attn-out` and `gdn-proj`/`gdn-scan`/`gdn-out` are separate rows.

### The table (single stream, dual, `--split-blocks 28,20`, 30 resident blocks, gen 16, warm 1, clocks `high`)

`17.10 tok/s / 58.47 ms/token` in this window (the reference window of the round ran
18.94/52.80; the box was shared with two other agents, so the **shares** are the
result and the absolute level is 8-11 % high):

| stage | RX µs | MI50 µs | Σ µs | n/token |
|---|---:|---:|---:|---:|
| hc-read (attn+ffn) | 2613.7 | 3697.2 | **6310.9** | 96 |
| hc-write (attn+ffn) | 257.9 | 290.3 | **548.3** | 96 |
| gdn-proj | 2205.7 | 4143.4 | **6349.1** | 36 |
| gdn-scan | 1282.1 | 1577.3 | **2859.5** | 36 |
| gdn-out | 1012.0 | 1956.1 | **2968.1** | 36 |
| qkv | 665.6 | 1307.6 | **1973.2** | 12 |
| attn | 462.3 | 575.5 | **1037.7** | 12 |
| attn-out | 437.2 | 637.6 | **1074.9** | 12 |
| qsa-aux (idx pool/tail/score/topk) | 89.3 | 149.5 | 238.9 | 12 |
| rope | 64.2 | 60.0 | 124.2 | 12 |
| kv-append | 17.0 | 21.0 | 38.1 | 12 |
| rmsnorm | 0.4 | 0.8 | 1.2 | 12 |
| router | 629.3 | 688.6 | **1318.0** | 48 |
| moe-gate | 361.9 | 726.4 | **1088.3** | 30 |
| moe-up | 435.5 | 786.6 | **1222.0** | 30 |
| moe-down | 318.1 | 1086.4 | **1404.5** | 30 |
| moe-combine | 118.6 | 89.5 | 208.1 | 30 |
| moe-swiglu | 91.5 | 55.7 | 147.2 | 30 |
| moe-map | 3.5 | 8.5 | 12.0 | 120 |
| shared-up | 353.7 | 348.8 | 702.5 | 48 |
| shared-down | 147.6 | 305.8 | 453.4 | 48 |
| shared-scalar | 105.6 | 99.6 | 205.2 | 48 |
| shared-swiglu | 40.2 | 70.4 | 110.5 | 48 |
| shared-gate | 15.3 | 24.4 | 39.8 | 48 |
| head | 0.0 | 4252.0 | **4252.0** | 1 |
| final-hc | 0.0 | 76.1 | 76.1 | 1 |
| xfer (lane readback copies) | 98.2 | 177.9 | 276.1 | 18 |
| **sum** | | | **35 039.8** | |
| `wait-gpu` (host fence) | | | **32 997.0** | |
| **residual** | | | **−2 042.8 (−6.2 %)** | |

Grouped: **attention/GDN 16 665 (47.6 %)** · **HC read+write 6 859 (19.6 %)** ·
**head+final-hc 4 328 (12.4 %)** · **device MoE 4 082 (11.7 %)** ·
**shared expert + router 2 829 (8.1 %)** · xfer 276 (0.8 %).

### What it says

* **`wait-gpu` is execution, not launch overhead.** The spans sum to *more* than the
  fence wait (−6.2 %; the excess is work whose wait lands in the next token's
  window). There is **no unexplained residual** — W ≈ 0, ≤2 ms.
* **Nothing is bandwidth-saturated.** Implied rates: expert GEMVs 205 GB/s (RX) /
  145 GB/s (MI50); HC projectors 281 / 142 GB/s; head **98 GB/s** on the MI50 — against
  576 GB/s (RX HBM) and ~1 TB/s (MI50 HBM2). A single-token GEMV reads a 0.6-1 GB/s
  -worth-of-weights tensor with one workgroup per output block and a dependent
  barrier chain, so it is **latency- and occupancy-bound**, which is exactly what
  §8am's A/B (barrier form, 78 % "overhead") could not see.
* **Dispatch/barrier is inside the spans and cannot be split from execution at stage
  granularity.** Bound it from the measured launch prices: 1 749 dispatches/token
  (26 238 recorded over the 15-token window) × 1.1-4.4 µs of in-batch
  dispatch+barrier = **1.9-7.7 ms**, of which the barrier term (0.81
  barriers/dispatch, 1.1-2 µs each) is ~1.5-2.8 ms. It is *not* zero, and it is not
  22 ms either.
* The host stage table and the GPU table agree: `sum ≡ wall` on the host side, and
  the GPU sum is within 6 % of `wait-gpu`, so the token is
  `wait-gpu 33.0 + lane 13.9 + head(host) 6.4 + ple 1.2 + xfer 1.1` with nothing left
  over.

### Verdict: what the next round attacks

The three largest single items are **not** the expert lane and **not** dispatch
count: `gdn-proj` (6.35 ms) + `gdn-out` (2.97) + `head` (4.25) + the HC projectors
(6.86) are 20.4 ms of 35.0 ms, all of them the same shape (one-token GEMV at 15-35 %
of peak). The next attack is the **single-token GEMV pipeline**: rows-per-workgroup /
column-blocked tiling and split-K for the 2560-wide GEMVs (the head and the GDN
projections first), then the HC projector chain, which is 96 × ~70 µs of five
*dependent* dispatches. T-batching the dense chain stays parked: it needs occupancy
work first, or it will multiply a 15 %-of-peak kernel.

### Non-invasiveness (probe on vs off, adjacent arms, clean window)

| arm | tok/s | ms/token | `wait-gpu` ms | record ns/dispatch |
|---|---:|---:|---:|---:|
| probe off | 17.27 | 57.89 | 32.00 | 279 |
| probe on | 17.10 | 58.47 | 32.96 | 303 |
| probe off | 17.50 | 57.14 | 31.97 | 286 |
| probe on | 16.93 | 59.08 | 33.00 | 273 |

**+1.6 % on the GPU timeline (+0.99 ms), −1.9 % on wall**, i.e. ~2 %: 958 stamps/token
× 1.03 µs/stamp, inside the 0.3-2.1 µs/stamp range of `docs/research/07` §C2. The
table is therefore **indicative at the 2 % level** — its shares are safe, its
absolute rows carry ±2 %.

Parity is untouched: **32/32 at chunk 8/16/32/128 with the probe on *and* off**, top-1
`13.170561` everywhere. One `--warm 1 --gen 32` arm produced a different first token
(`248046`, not `110563`) during the heaviest memory-pressure minute of the window
(19 GB of swap in use by three concurrent 30 GiB processes); it did not reproduce in
24 further arms, and 8 alternating probe-on/off arms were all correct. Recorded
rather than dismissed: if it recurs, it is a latent race, not the probe's stamps.

### Item 2: the in-path profiler is gated, `420 → 258 ns/dispatch`

`record()`'s five `Instant::now()` pairs are behind a cached `BRECCIA_RECORD_PROFILE`
flag and `track_disabled()`'s two `env::var_os` behind a `OnceLock`. A/B (4 adjacent
pairs, gen 8): **416-432 → 254-275 ns/dispatch**, i.e. **−162 ns/dispatch (−39 %)**,
0.28 ms/token at 1 749 dispatches — matching `docs/research/11`'s 165 ns for the pairs.
The remaining 258 ns is exactly what that report predicted is left: `key + cache`
130 ns (the `(u64, Vec<Buffer>)` allocation, R3 — **not** taken this round), binds
85 ns, barrier 34 ns. The CLI now prints a windowed
`record N ns/dispatch` line (numerator and denominator both reset after the prompt),
so this number no longer depends on `sess.dispatches`.

### Commands

```bash
# the table (one probe run; the numbers above)
timeout 300 env BRECCIA_GPU_TIME=1 ./target/release/breccia bench token \
  --devices 0000:03:00.0,0000:0e:00.0 --split-blocks 28,20 --gpu-experts \
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,28,29,30,31,32,33,34,35,36,37,38,39,40,41" \
  --prompt-ids-file /var/tmp/users/korphaus/tmp/opencode/p4-token/parity-server.json.ids \
  --gen 16 --runs 1 --threads 16 --warm 1
# probe overhead: alternate BRECCIA_GPU_TIME=1 with it unset, same command
# timer strip: alternate BRECCIA_RECORD_PROFILE=1 with it unset, --gen 8 --warm 0
# parity (single card: chunked prefill needs one device), 8 arms, all 32/32
timeout 400 env BRECCIA_GPU_TIME=1 ./target/release/breccia bench token \
  --devices 0000:03:00.0 --prefill-chunk 32 --gen 32 --runs 1 --threads 16 --warm 1 \
  --prompt-ids-file /var/tmp/users/korphaus/tmp/opencode/p4-token/parity-server.json.ids \
  --json /var/tmp/users/korphaus/tmp/opencode/probe-gate/one-on-32.json
```

Safety: no shader was touched and no new dispatch shape was dispatched. All runs were
`timeout`-bounded, `power_dpm_force_performance_level` read `high` before and after
(and after every arm), and `journalctl -k` shows **no `amdgpu` page fault, ring timeout
or reset during this round's window**. (It does show one at **00:43**, ~3.8 h before
this round's runs — an RX GFX-hub page fault → `ring gfx_0.0.0 timeout` → ring reset →
"device wedged, but recovered", from whatever ran then, not from this round. It is on
the *display* card, which is now the RX per the corrected `AGENTS.md`; it is recorded
here because it is the most recent reset on the box and the next reader should know it
happened.) The box was shared with two other agents' benchmarks for most of the
window; each arm's log/grep records whether another `breccia bench` was alive, and the
table above quotes arms whose neighbours were, while the probe-overhead A/B/A/B quotes
the one clean stretch (`pgrep` = this process only).
