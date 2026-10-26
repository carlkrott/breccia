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
4. **The expert lane's structural lever** (§8b): `-ot`-style GPU expert blocks,
   worth ~20 ms/token on the host side, and the page prefetch across the block
   boundary. Together with batching that is where the remaining 100 ms lives.
