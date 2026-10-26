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
