# 08 — submits per block-chunk, priced — plus DeepSpeed-Inference

Round 8 (2026-10-27). Paired research round. Part A prices every mechanism that could
take chunked prefill from **13.6 submits per block-chunk to ~2**, on the headless RX
6950 XT alone; Part B audits **DeepSpeed-Inference** (rotation #7).

**Read-only outside this file** — no engine code was touched. Scratch harness:
`<SCRATCH>/opencode/research-harnesses/r08-submits/` (ash 0.38 + a C
probe for `VK_EXT_device_generated_commands`), never inside the repo.

## TL;DR

1. **The submit count is not what makes chunked prefill slow, and the round-7 cost model
   was applied to the wrong unit.** Measured: a fence-waited submit costs **33.2 µs**
   (+23.8 µs of it is the wait, +8.6 the submit), a *queued* submit **8.6 µs**, and a
   command-buffer boundary **1.7 µs** of GPU timeline. A chunk-64 prefill records **654
   submits** (§8af) — i.e. **21.6 ms per chunk ≈ 0.34 ms per token**, not §8ah's ~34 ms/
   token. Merging PLE (13.6 → 2.2 per block-chunk) is worth **~0.2–0.35 ms/token of
   ~155 ms**, one tenth of one percent. It is still correct engineering — it is just not
   the lever, and the round's headline has to be that it is not.
2. **Round 7's "command-buffer boundary costs 21–34 µs of GPU timeline" does not
   reproduce: it is 1.7 µs.** Secondary and primary CBs cost the same. r07's `chain` arm
   wrote overlapping query slots (`base` 0 and 3 with `per = 4`), so its "gap" was really
   *three dispatches plus a boundary*. Re-measured with disjoint slots and A/B/A
   interleaving: `+1.75 µs` per boundary, confirmed by two independent methods.
3. **Where the boundary money actually is: the submit *packet*, and the re-recorded CB.**
   An extra submit costs **+8.6 µs wall / +8.8 µs GPU**; submitting a *re-recorded* CB
   costs ~**1.4 µs/dispatch more** than replaying the same frozen CB (0.8/17.1/57.7 µs at
   4/12/32 dispatches) — the host-side record timer (~115–200 ns/dispatch) does not see it.
4. **So the same machinery should be aimed at the *stall*, not the count.** With an
   800 µs host slice per block (our expert lane), `submit → host work → wait` (one timeline
   value) measures **905 → 811 µs per block**: it recovers ~94 µs/block ≈ **4.5 ms/token**,
   ten times the whole submit-count item. The PLE phase is 5 stalls/token with nothing
   overlapping — that, not 33 µs, is why `ple` is 2.2 ms/token on the RX stream and
   17.6 ms/token on the MI50 stream.
5. **Secondary CBs are cheap and legal, with one hard rule**: a barrier *inside* a
   secondary does order across secondaries on RADV (12/12 dependent increments correct,
   validation-silent), a barrier in the primary does too (12/12), and with no barrier the
   answer is wrong (11/12 and 0/12). Never rely on CB-boundary implicit ordering.
6. **`VK_EXT_device_generated_commands` COMPUTE: creation works, execution does not.**
   Feature, indirect execution set, 2-token layout and a 6 656 B preprocess requirement all
   succeed; submitting `vkCmdExecuteGeneratedCommandsEXT` produces a `[gfxhub] page fault`
   + `ring gfx_0.0.0 timeout` that the driver recovers from by resetting the ring (twice,
   before and after fixing buffer-device-address usage in my probe). And structurally it
   could never help this question: it is a *recorded command*, so it removes neither a
   submit nor a CB boundary — at best the ~1.4 µs/dispatch submit-path cost of item 3.
7. **Incidental, and it invalidates every absolute µs in this file's siblings: the RX runs
   at 500 MHz.** `power_dpm_force_performance_level = auto` never leaves DPM level 0 under
   headless compute load (polled 5× during a run), while round 6 measured the shared
   expert at **5.09 TFLOP/s** = **100 % of the RX's FP32 peak at 500 MHz** and 28 % of its
   1801 MHz peak. The kernels are not slow; the card is idling. See Top-3 #1.
8. **DeepSpeed-Inference** is a tensor-parallel *training-engine family* with an inference
   facade: what transfers is ZeRO-Inference's **NVMe/CPU weight streaming** (pinned ring,
   1 MiB blocks, queue depth 8 — independently matching our §03 measurement), its
   **reserve-then-admit scheduler** (`can_schedule` returns an explicit refusal reason),
   and v2's **device row-map / atom-builder** pattern. What does not: CUDA graphs, GDS,
   cutlass/NVFP4, all collectives. Ranked list in §B5.

---

# Part A — how many submits a block-chunk actually needs

## A0. The arithmetic the round was asked to check

| quantity | measured | source |
|---|---|---|
| submits per block-chunk (chunked prefill) | **13.6** (654 submits for a chunk-64) | §8af/§8ah |
| the "not 2" part of it | PLE on block 1: 2 flushes + ~3 unbatched records **per token** | `graph.rs:2813–2889`, `:1739–1744` |
| predicted cost at 62 µs (round 7) | ~550 × 62 µs = **34 ms per prefill** | §8ah table |
| **that cost, per token of a chunk-64** | **34 ms / 64 = 0.53 ms/token** of a 155 ms token = **0.34 %** | this round |

`ple.layers = [1]` (one block of 48), so the 13.6 is *per block per chunk* and the round's
own table (`docs/phase-4/01-forward-pass.md:2612–2615`) already says what it means:
**6 396 dispatches and 654 submits per block-chunk** against 100 dispatches / 3.2 submits
per block-token. A chunk-64 prefill is ~9.9 s of wall, and 654 submits at my measured
33.2 µs is **21.6 ms of it**. The sentence "13.6 submits … = ~34 ms" was written per
*chunk* and then compared against per-*token* stage rows; §8af had already concluded the
right thing ("the drain is not the submits") three sections earlier.

So Part A's job is not "is it big" (it is not) but "what does each mechanism cost, and what
*is* the stall worth" — because the PLE phase's 5 submits/token are also 5 **host–GPU
round trips that overlap nothing**, and that is a different, larger question.

## A1. Method, and the clock trap that would have hidden everything

Measured on `0000:03:00.0` (RADV NAVI21, mesa 26.2.4), one process at a time,
`timeout`-wrapped, validation off for timing / on for the first run of every new path.

**Trap 1 — the card is at 500 MHz.** The same 12-dispatch CB measured **145.8 µs** in one
block of a run and **68.1 µs** in another; `pp_dpm_sclk` polled during a run shows
`0: 500Mhz *` throughout, and `auto` never promotes (idle reads 0 MHz). Every arm is
therefore measured **inside the same rep loop** and reported as a **per-rep median
difference against the inline arm**, so clock state cancels; absolute µs in this document
are 500 MHz numbers and must not be compared with a boosted-clock run.

**Trap 2 — overlapping query slots.** r07's `chain` arm stamped slots
`base + i` with `base ∈ {0, 3}` and `per = 4`, so slots 3 and 4 were written by *both*
CBs and `vkGetQueryPoolResults` returns the last write: the reported "CB1 end → CB2 start"
gap was really *3 dispatches + the boundary*. This round uses disjoint slots per CB and
confirms each boundary two ways (interior stamps and a whole-span difference).

```bash
cd <SCRATCH>/opencode/research-harnesses/r08-submits
cargo build --release --offline
timeout 90  ./target/release/r08-submits --device NAVI21 caps        # queue family, CB slots
timeout 90  ./target/release/r08-submits --device NAVI21 --n 12 dep  # ordering across CBs
timeout 200 ./target/release/r08-submits --device NAVI21 --n 12 topo # layouts + submissions
timeout 200 ./target/release/r08-submits --device NAVI21 --n 12 replay --rounds 12
timeout 120 ./target/release/r08-submits --device NAVI21 --n 12 cbup # re-record vs replay
timeout 120 env HOST_US=800 ./target/release/r08-submits --device NAVI21 --n 12 overlap
```

## A2. Q1 + Q5 — secondary CBs, and what a boundary really costs

Same 12 dispatches (256 KiB touch, 64 fma, engine's `ALL_COMMANDS` barrier after each),
**one submission**, interleaved across 6 layouts, 15 reps, per-rep delta vs inline:

| layout | GPU span | boundary (stamp) | Δ vs inline (per-rep median) | per boundary |
|---|---:|---:|---:|---:|
| inline, 1 CB | 30.8 µs | — | — | — |
| 2 secondary CBs | 32.3 | 1.52 µs | +1.6 µs | **+1.60 µs** |
| 4 secondary CBs | 35.5 | 1.51 | +4.9 | **+1.63** |
| 12 secondary CBs | 48.3 | 1.50 | +17.6 | **+1.60** |
| 2 **primary** CBs | 32.3 | 1.44 | +1.6 | **+1.60** |
| 4 **primary** CBs | 35.4 | 1.45 | +4.6 | **+1.55** |

(A second run 20 minutes earlier gave 32.5 / 36.1 / 50.0 / 32.5 / 35.9 with the same
1.7–1.8 µs per boundary — run-to-run spread ≤ 8 %, so quote **1.5–1.8 µs, call it 1.7**.)

Three conclusions, all cheap and all reusable:

* **A CB boundary is 1.6–1.8 µs of GPU timeline — 0.7× a dispatch+barrier (2.57 µs), not
  5–8×.** Splitting a block into 4 CBs inside one submit therefore costs ~5 µs, not 85.
* **Secondaries and primaries cost the same.** RADV merges secondaries into the primary
  stream, so there is no reason to prefer secondaries for *boundary* reasons — but also no
  penalty, and a secondary is the natural unit for "pre-record once, replay" (A4).
* **A 12-dispatch CB at 500 MHz is 30.8 µs, of which 12 × 2.57 = 30.8 is dispatch+barrier.**
  That rate is a *lower bound* for the engine: the stage table's `wait-gpu`
  (32.8 ms/token RX / 85.8 ms/token MI50 over ~4 800 dispatches) implies **6.8 µs per
  dispatch**, i.e. at or above round 7's **5.7 µs footprint knee** for a ≥1 MiB dirty set —
  which is exactly what streaming an expert's weights through each dispatch produces.
  This harness touches 256 KiB, so it lands at 2.57 µs; the two numbers agree on the
  *mechanism* and differ by the footprint, which is the round-7 result holding up.

**Recording cost is a non-issue** (Q2's other half): 12 dispatches inline record in
**1.1–1.4 µs (≈100 ns/dispatch)**; 12 secondaries cost 8.4 µs because each `begin`/`end`
pair is ~0.7 µs. So *recording many small secondaries is worse than recording one primary*
if you record them per token — but see A4, which is about not recording them at all.

## A3. Q1/Q4 — barrier scope in secondaries (the RADV gotcha), measured

A genuinely dependent chain (`b[i] += 1` per dispatch, 12 dispatches, correct answer 12),
validation (`VK_LAYER_KHRONOS_validation` + `VK_LAYER_VALIDATE_SYNC=1`) silent:

| placement of the barrier | b[0] | verdict |
|---|---:|---|
| inline, barrier after each | 12 | correct |
| **inside each secondary** | **12** | correct on RADV — and sync-validation raises nothing |
| in the primary between `executeCommands` | 12 | correct (the spec-safe form) |
| nowhere (secondaries) | 11 | **wrong** — one update lost |
| nowhere (inline) | 0 | **wrong** — all twelve overlapped |

So: the driver does honour a barrier recorded inside a secondary across the following
secondary, but the *only* form that is safe by the letter of the spec is the barrier in the
primary (or at the top of the next secondary). Both cost exactly one barrier packet, so
**take the spec-safe form** — it is free. Two practical rules fell out:

* **Re-submission works** with only a descriptor-set swap (r07 measured 19.6 ns/set;
  re-measured here at ≈0 added cost, and the frame never needs `ONE_TIME_SUBMIT`), but the
  secondary must be recorded with `SIMULTANEOUS_USE` if it can be executed from more than
  one in-flight submission. Without it, `vkBeginCommandBuffer`/`vkResetCommandBuffer` on a
  pending CB is a validation error and, at depth ≥2, a real hazard.
* **Timestamps in secondaries are fine** (`vkCmdWriteTimestamp2` with `COMPUTE_SHADER`);
  `vkCmdResetQueryPool` in one is not.

## A4. Q2 — pre-recorded CBs: the one place the submit path pays

`replay`, 12 dispatches, 12 rounds, sequential (fence each):

| arm | per round |
|---|---:|
| record + submit + wait | 99.6 µs |
| **replay (same CB) + submit + wait** | **60.5 µs** |
| descriptor-set update + replay + submit + wait | 60.6 µs |

**39 µs per round** is not CPU recording — recording the same 12 dispatches measures
1.4 µs. It is the *submit path*: RADV uploads the IB (and its state bundle) at
`vkQueueSubmit`, and a CB that was re-recorded must be re-uploaded, while an unchanged CB
hits a cached path. `cbup` isolates it against CB size:

| dispatches | replay | re-record+submit | record alone | **submit-path delta** |
|---:|---:|---:|---:|---:|
| 4 | 43.6 µs | 44.5 | 1.8 | **+0.8 µs (0.2/dispatch)** |
| 12 | 87.1 | 104.1 | 5.6 | **+17.1 µs (1.4/dispatch)** |
| 32 | 295.3 | 353.0 | 9.1 | **+57.7 µs (1.8/dispatch)** |

At 4 dispatches there is no penalty, so treat the slope as a trend, not a law — but if it
holds at engine scale it is **~1.4 µs × ~4 800 dispatches = 6.7 ms/token of prefill that no
host-side timer attributes**, because it happens inside `vkQueueSubmit`, not in
`record_seconds()`. That is the largest single number in this document, it is cheap to
confirm (`cbup` at 128 and 256 dispatches, and `record_seconds()` vs `submits` in the
engine's own summary), and if it is real it *is* the argument for frozen geometry — the
first argument for it that has survived measurement (round 4 priced the same idea at
−0.3 ms/token on the CPU side alone).

## A5. Q4 — the submission strategies, and the interleave cost

Same 12 dispatches, pre-recorded so that only submission is being measured, interleaved:

| strategy | wall | GPU span | marginal |
|---|---:|---:|---|
| S1 fence wait after **each** submit | 400.9 µs | 371.0 µs | — |
| S2 all queued, **one** fence at the end | 155.4 | 125.3 | **S1−S2 = +22.3 µs per fence wait** |
| S3 **one** submit for all 12 | 62.9 | 30.8 | **S2−S3 = +8.4 µs wall / +8.6 µs GPU per extra submit** |

(Second run: 420.7 / 159.3 / 64.6 wall, 388.6 / 127.8 / 30.8 GPU, +23.8 and +8.6 — so
**+22–24 µs per fence wait, +8.4–8.6 µs per submit**.)

Decomposed: a serialized submit+wait is **33–35 µs**, of which 8.5 is the submit and 23 is
the wait; queuing makes the wait disappear and leaves 8.5. The fence wait also shows up in
the **GPU timeline** (S1 371.0 → S2 125.3 µs = **+22.3 µs of GPU span per fence wait**):
waiting for idle is not free on the device either, which is why the same 12 dispatches
cost 30.8 µs queued and 371 µs serialized. Note this is lower than r07's 56 µs round trip
for an *empty* CB on an *idle* GPU; the difference is that here the queue is already busy,
so the submit→start latency is partly overlapped. Both are true numbers for different
states, and 33 µs is the one that applies to a busy engine.

**The interleave (Q4)** — "submit pass 2 of block N while pass 1 of block N+1 records" —
priced two ways:

| scheme (12 chunks, 800 µs host slice = one block's expert lane) | wall/chunk |
|---|---:|
| submit → **wait** → host work (the engine's `flush()` order today) | 905.2 µs |
| submit → **host work** → wait (one timeline value, depth 1) | **811.3 µs** |

and on a bare re-record loop, `depth-2` through a timeline semaphore is **17.3 µs/submit
against 33.2 µs** serialized. The cost of the scheme is the extra in-flight submit
(+8.6 µs host, +8.8 µs GPU if it is a *new* submit); the benefit is that the 24 µs wait
disappears behind work the host has to do anyway. **At an 800 µs slice it is worth
~94 µs/block ≈ 4.5 ms/token at 48 blocks** — an order of magnitude more than the entire
submit-count item, and it needs only a reordering plus timeline semaphores (the engine
already has `Timeline::wait_host` for the marker path).

## A6. Q3 — `VK_EXT_device_generated_commands`: present, creatable, and not usable

`vulkaninfo` (unchanged from round 4): `deviceGeneratedCommands = true`,
`maxIndirectCommandsTokenCount = 128`, `maxIndirectSequenceCount = 1 048 576`,
`supportedIndirectCommandsShaderStagesPipelineBinding = { COMPUTE }`,
`…ShaderBinding = { COMPUTE }`. `probe/dgc.c` (built against the system
`vulkan_core.h`, which *does* have the EXT structs — ash 0.38 only ships the **NV**
variant, which is why round 4 said "no types at all"):

| step | result |
|---|---|
| instance + device with `VK_EXT_device_generated_commands` (+`VK_KHR_maintenance5`, required) | **ok** |
| `vkCreateIndirectExecutionSetEXT` (pipeline binding, 1 pipeline) | **ok** |
| `vkCreateIndirectCommandsLayoutEXT` (2 tokens: PUSH_CONSTANT@0, DISPATCH@4, stride 16) | **ok** |
| `vkGetGeneratedCommandsMemoryRequirementsEXT` (64 sequences) | **ok**, 6 656 B, align 256 |
| `vkCmdExecuteGeneratedCommandsEXT(isPreprocessed = VK_FALSE)` recorded | **ok** |
| submit + fence | **hangs** → `[gfxhub] page fault … Process dgc` at `0x0000800000000000`, then `ring gfx_0.0.0 timeout`, `device wedged, but recovered through reset` |

Two attempts: the first before, the second after adding `bufferDeviceAddress` and
`VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` to every buffer whose address is handed to the
driver (a real omission — the fault address is a null-ish GPU VA, which is what a bad
indirect/preprocess address looks like). The fault is identical, so either the EXT path is
broken in this mesa for compute, or my layout/execution-set configuration still misses a
VUID. **Verdict: "not yet", and not worth another round on its own merits** — because of
the structural point: `vkCmdExecuteGeneratedCommandsEXT` is a *recorded command*. It cannot
remove a submit or a CB boundary. Its only possible win here is item A4's re-upload cost,
and there is a cheaper way to get that (frozen geometry with a masked direct dispatch, which
round 4 already specified and which needs no new API).

**Crash-safety note (new incident-log row):** two `gfxhub` page faults and two ring resets,
`0000:03:00.0` (headless RX), process `dgc`, both recovered by the driver;
no `Illegal opcode`, box healthy afterwards, all subsequent harness runs bit-identical.
This is the *wrong* card for the same experiment and the right one for it — the fault was a
null VA, not an OOB table walk.

## A7. Q5 — so what *does* reduce the boundary cost?

| lever | measured effect | verdict |
|---|---|---|
| fewer **primary** CBs per submit | −1.7 µs each (GPU) | free, do it by accident |
| secondaries instead of primaries | 0 µs — identical | use them for **replay**, not for boundaries |
| fewer **submits** | −8.6 µs wall / −8.8 µs GPU each | real, and already ≥2/block |
| don't fence every submit | −24 µs each | **the one that pays** (A5) |
| larger submits (more dispatches per submit) | 0 additional (fixed cost) | already done |
| a second queue / dedicated compute family | not measured | **no** — one queue is 1 queue family; two families would put the barrier back |
| DGC | structurally cannot | closed (A6) |

## A8. Recommendation — reaching ~2 submits per block-chunk, and what it is worth

The target is reachable with three changes, all inside `ple_step_rows` and
`prefill_block`, none of which touch arithmetic (so parity is by construction):

1. **Batch the PLE phase across the chunk's tokens** (`prefill_block`, `graph.rs:1739`).
   All `t` residuals for block 1 are already in the venue's `r` before the block starts
   (`swap_r_in(dev, i)`), and the host PLE block cannot feed back into the GPU half within
   the token — the rolling context advances *on the host* from `rows_for(tok)`. So:
   one `ensure_batch` → `t` gathers + `2t` key/value dispatches → **one** `submit_batch`,
   then **one** `read_batch` of `[t, hc_dim]*3`, then the `t` host blocks in a loop, then
   one host write of `[t, hc_dim]`. That is **2 submits per PLE block-chunk** instead of
   `2 + 3t`, and it is the only structural change the 13.6 → 2.2 target actually needs.
2. **Replace the per-submit fence with one timeline wait per block.** `flush()` today
   submits *and* waits, in that order, before the host does anything. Swap to
   `submit → host lane → wait(value)`; the measured gain is 94 µs/block at our 800 µs
   slice (A5), and the pathological case (nothing to overlap) costs one extra in-flight
   submit, +8.6 µs.
3. **Bound the stalls, then count them.** Add `stalls/token` next to `submits/token` in the
   step summary — a flush before a host readback is a stall, a `submit_batch` is not. The
   PLE phase is 5 stalls/token; target 1 (the readback is unavoidable while the rolling
   context lives on the host, and moving it to a shader would change arithmetic).

**Expected effect, in the right units:** 13.6 → ~2.2 submits per block-chunk removes
~11.4 × 48 ≈ 550 submits per chunk. At 33.2 µs each that is **18.3 ms per chunk-64 chunk =
0.29 ms/token**, i.e. **0.19 % of a 155 ms prefill token** (0.02 % if the remaining submits
are queued rather than fenced). Item 2 is worth **~4.5 ms/token** on the same window.
That ordering — *reorder before you merge* — is the round's actionable result.

---

# Part B — DeepSpeed-Inference audit

Shallow clone OK: `Reference Projects/deepspeed`, `1fcac82` (2026-10-27), 452 MB.
Two generations live side by side: **v1** (`deepspeed/inference/`, `module_inject/`,
`ops/transformer/inference/` — kernel injection into HF modules) and **v2**
(`deepspeed/inference/v2/` — a from-scratch inference engine with its own model
implementations, a ragged-batch abstraction and a block KV allocator). ZeRO-Inference is
not a directory: it is **ZeRO-3's partitioned-parameter swapper plus DeepNVMe**, driven
from the normal `init_inference` path.

## B1. Tensor-parallel and KV partitioning

* **AutoTP** (`module_inject/auto_tp.py`, `tp_shard.py`): a per-architecture *parser*
  returns an injection list of `(client_module, policy_tuple)` pairs; the policy tuple names
  the four weights to slice (`qkvw`, `attn_ow`(+b), `mlp_1w`, `mlp_2w`). Slicing is
  `ReplaceWithTensorSlicing(mp_group, mp_size, ...)` with per-role rules: QKV is split
  `3 × hidden` then per-head-shard (`get_shard_size_list` handles non-divisible head
  counts), `attn_ow` is split on its *input* dim and all-reduced, `mlp_1w` on its output,
  `mlp_2w` on its input. Fused-QKV checkpoints are re-split by name
  (`fusedqkv_utils.require_tp_fused_qkvw`, `split_by_qkvlist_and_refuse`).
* **KV cache is partitioned by head, not by block**: each rank keeps
  `num_heads / tp_size` heads of `layer_past` for every layer, max length fixed at
  creation (`max_out_tokens`). There is **no paging and no per-sequence allocator in v1**.
  v2 replaces that with a real block allocator (`inference/v2/allocator.py`,
  `state_manager`, `free_blocks` per KV cache, `get_kv_requirements(seq, n_new_tokens,
  max_new_blocks)`) — block-granular, per-sequence, with an exact pre-admission check.
* **MoE**: expert-parallel *and* expert-slicing (`docs/_tutorials/mixture-of-experts-
  inference.md`): `ep_size` partitions experts across ranks; if `n_expert < world_size`
  each expert is sliced vertically/horizontally instead. Non-expert params stay
  data-parallel + tensor-sliced. **Our placement problem (which card holds which resident
  expert) is not in this design at all** — they assume every expert is either on a rank or
  replicated, never partially resident.

## B2. Kernel injection and the fusion list

`replace_with_kernel_inject=True` swaps the HF block for `DeepSpeedTransformerInference`,
whose forward is a **fixed 5-kernel sequence per layer**:

| # | kernel | fusion it performs |
|---|---|---|
| 1 | `qkv_func` (`LinearLayer` → `DeepSpeedSelfAttention.linear_func`) | **LayerNorm + QKV GEMM**, one kernel (`pre_layer_norm`) |
| 2 | `compute_attention` | attention (+ KV append, optional alibi) |
| 3 | `attn_ow` | output projection **+ residual add** |
| 4 | `mlp_1w` = `[2·inter, hidden]` + activation | **gate+up in one GEMM + GELU/SiLU, one kernel** (this is exactly round 6's 3.1×/5.3× fused gate+up+SwiGLU) |
| 5 | `mlp_2w` | down projection **+ residual + (optional) all-reduce** |

The "fusion list" is therefore not a graph pass — it is a **hand-written layer class per
(model family, dtype) pair** (`inference_policy` dicts in `replace_policy.py`:
`transformer_policy` with `injection_policy` naming `attn_qkvw/attn_qkvb/attn_ow/attn_ob/
mlp_1w/mlp_1b/mlp_2w/mlp_2b`). Cost of that choice: each new architecture needs a policy,
and anything the policy does not name stays eager. Benefit: five dispatches per layer with
no graph machinery, which is the number our own ~100 dispatches/block should be compared
against.

**v2 goes further and is the interesting half:** one `ragged` batch for the whole step,
one **blocked-flash attention kernel over all sequences** (no per-sequence launch), a
`linear_blocked_kv_rotary` kernel that does **QKV GEMM + KV write into paged blocks + RoPE
in one dispatch**, fused `top_k_gating`, and device-side `moe_gather`/`moe_scatter` to
compact tokens per expert before a grouped GEMM. `AtomBuilder` builds the attention tile
map `[max_atoms, 8]` **on the host in C++** from the ragged batch (a `q_block × kv_block`
grid per sequence) and uploads it once — the same "one launch over a device row map"
pattern rounds 4 and 6 converged on, applied to attention instead of MoE.

## B3. ZeRO-Inference offloading — the part that maps onto 71 GB on 58 GB

Three layers, all CUDA-agnostic in shape:

1. **Partitioned parameter store** (`runtime/swap_tensor/partitioned_param_swapper.py`):
   parameters are partitioned to CPU (or, with `use_gds`, kept device-side) and streamed
   back in a **ring of pinned staging buffers** — `buffer_count` buffers × `buffer_size`
   elements, allocated with `new_cpu_locked_tensor` so the DMA target never migrates.
   `swap_in_tensor(param, async_swap, dest_buffer)` + `available_buffer_ids`, with
   `require_swap_buffer` skipping the copy when the destination is already pinned.
2. **DeepNVMe** (`ops/op_builder/aio`, `AsyncIOBuilder`): `libaio` with
   `block_size = 1 MiB`, `queue_depth = 8`, `intra_op_parallelism = 1`,
   `single_submit = false`, `overlap_events = true` (constants.py) — i.e. **8 MiB of
   read in flight, 1 MiB per request**, which is the same operating point our own §03
   measurement found (queue-limited at `nr_requests × max_sectors ≈ 32 MiB`, prefetch
   depth > 2 buys nothing, `read_ahead_kb` on the btrfs BDI is the real knob).
3. **Their measured result** (`blogs/deepnvme/08-2024`): LLaMA3-70B, **weights on NVMe**,
   1×A100-80GB, prompt 512, gen 32, **batch 96** → **6–7 tok/s aggregate** on 4 NVMe
   drives (≈13 GB/s), GDS +10–18 %. That is a *throughput* design point (≈14 s per step);
   they say so explicitly ("well suited to throughput-oriented applications").

Read against our box: we are in the **CPU tier, not the NVMe tier** (71 GB model, 58 GB
RAM, page-cache warm path at 58 GB/s per §03) — so their NVMe numbers bound *our cold*
case only, and their architecture offers **no residency policy**: there is no frequency
ranking, no hot/cold split, no per-expert decision. `zero.Init()` + the swapper is a
*pure* "partition everything, stream what the current module needs" design.

## B4. Scheduling

* **v1**: a single-sequence `generate` loop with an optional **whole-model CUDA graph**
  (`enable_cuda_graph`, `cuda_graph_created`) — explicitly *not* continuous batching.
* **v2**: a real scheduler. `can_schedule(uids, lengths)` does a **dry run** over the batch:
  it walks each sequence, asks the model `get_kv_requirements(seq, length, free_blocks)` and
  subtracts from `free_blocks` as it goes, and returns a **typed refusal**
  (`SchedulingResult.BatchSequenceLimitExceeded | KVCacheLimitExceeded |
  EngineSequenceLimitExceeded | BatchTokenLimitExceeded`). `max_ragged_sequence_count`,
  `max_ragged_batch_size`, `max_tracked_sequences` are the three admission limits.
* `get_remaining_block_capacity(uid)` exposes the partially-filled last block, so a caller
  can pack a new token into an already-allocated block instead of allocating one.

## B5. Ranked adaptations for Breccia

| # | adaptation | maps to | expected effect | effort |
|---|---|---|---|---|
| 1 | **Reserve-then-admit with a typed refusal** (v2 `can_schedule`): predict arena + KV + chunk buffers for the request against *live* free memory, return `KVCacheLimitExceeded`/`BatchTokenLimitExceeded` **before** the first token, with the numbers that produced the refusal in the error | our residency thrash (§8x/§8z) and 6-agent bursts | today a request that does not fit is discovered mid-token; this makes it a door decision. Measurable: p99 tok/s per agent and zero residency replans over a 6-agent burst | S |
| 2 | **One readback + one write per block-chunk, not per token** (the swapper's "swap the module's whole param set as one unit" idea, applied to our PLE residual/context) | chunked prefill, `ple` 2.2/17.6 ms per token | 2 submits per PLE block-chunk instead of 2+3t, and one stall/token instead of five — part A's A8 items 1–2, **~0.3 ms/token + ~4.5 ms/token from the wait reordering** | S–M |
| 3 | **Pinned staging everywhere** (`new_cpu_locked_tensor` ≡ `mlock`): audit whether our persistent staging buffer is locked; a page that migrates during a copy is a silent 10× on that copy | the lane (38–48 ms/token) and PLE gathers | check + one `mlock` — measurable in the `lane` stage and in `ple`'s cold rows | S |
| 4 | **Fused `qkv+rope+kv-write` in one dispatch** (`linear_blocked_kv_rotary`) — our QSA path records append/rope/select separately | dispatch count per block (~100) | small on its own; it is the *shape* the T-batched dense chain (§8af item 1) needs, since QSA is one of the two mixers that blocks it | M–L |
| 5 | **Host-built tile map for the chunked expert pass** (the `AtomBuilder` pattern): compute the `(token, expert) → workgroup` map in a plain loop per chunk and upload it once, instead of deriving it per launch | chunked MoE, ~130 distinct experts/block | removes per-launch host derivation; the same map `--prefill-merged` already needs | S |
| 6 | **`buffer_count` as an explicit prefetch depth** rather than a constant, chosen from measured queue depth (ours: queue-limited at ~1.7 blocks) | the lane's prefetch | documents why depth > 2 does nothing, and makes the number a *setting* with a measurement behind it | S |

**Not adaptable / CUDA-only:** CUDA graphs and `torch.cuda.CUDAGraph` capture (our Vulkan
answer is a pre-recorded CB — priced in A4); **GDS / GPUDirect Storage** (no AMD
equivalent; our `O_DIRECT` and page-cache paths are already measured and dominated, §03);
cutlass/NVFP4/mixed-GEMM, MoQ int8, `deep_gemm`; all tensor/expert-parallel *collectives*
(NCCL, `all_gather_coalesced`, DeepEP) — we have no interconnect problem, we have a
*placement* problem they never model; their MoE expert-parallel design assumes
fully-resident experts. Their KV story (head-partitioned, fixed max length; or v2's blocks)
is behind our Phase-6 plan (paged blocks + swap maps + prefix checkpoints from round 4).

---

## Top-3 next experiments

1. **Settle the GPU clock before any more performance work.** Poll
   `/sys/class/drm/card1/device/pp_dpm_sclk` during a bench (this round's harness prints
   it), then run the *same* 2-stream decode with
   `power_dpm_force_performance_level = high` and restore it. Round 6's 5.09 TFLOP/s is
   exactly the 500 MHz FP32 peak (80 CU × 64 lanes × 2 × 500 MHz = 5.12 TFLOP/s) and 28 %
   of the 1801 MHz peak, so if the clock is real the GPU half is 3.6× larger than we think
   and the whole bottleneck ordering changes. **Do this with the power-limit profile
   checked first** (the 850 W cap), and record what was changed and restored.
2. **Price the submit-path upload (`cbup`) at engine scale.** Extend `examples/
   dispatch_overhead` with 128/256-dispatch CBs, replay vs re-record, and print
   `submits`, `record_seconds()` and the *fence-wait* separately. If the ~1.4 µs/dispatch
   slope holds, it is ~6.7 ms/token of prefill that no current timer attributes, and it
   makes frozen geometry (round 4 item 2) the first structural item instead of the third.
3. **Reorder before merging: `submit → host lane → wait`.** Replace `flush()`'s
   submit-then-wait with `submit_batch()` + one `Timeline::wait_host` value per block,
   leaving the PLE batching for afterwards. Falsifiable: with the lane at 800 µs/block the
   exposed wait should fall from ~33 µs to ~0 (measured 905 → 811 µs/block here), i.e.
   `wait-gpu` in the stage table should drop by ~1.5 ms/token on the RX stream and ~4 ms on
   the MI50 stream. Gate: bit-identical logits; one flag; revert is `flush()` unchanged.

## Provenance

* Harness `r08-submits` (ash 0.38, `glslangValidator` + `spirv-val` in `build.rs`; modes
  `caps|dep|topo|replay|cbup|overlap`), probe `probe/dgc.c` against system
  `vulkan_core.h`. Everything under
  `<SCRATCH>/opencode/research-harnesses/r08-submits/`.
* Device: RX 6950 XT `0000:03:00.0` only (headless). The MI50/Radeon VII `0000:0e:00.0`
  drives the displays and was **never** used this round.
* Window: `gpu-manager`, `qwen-dual.service`, `strata-rx.service`, `d1-mi50.service` all
  inactive before and after (state as found). Another agent's short `breccia`/`git`
  processes appeared intermittently; because the protocol is per-rep-difference, the
  numbers reproduced to <0.5 % across runs on either side of them (30.8/32.5/50.0 µs GPU
  spans identical in two runs 20 minutes apart).
* GPU faults: two `gfxhub` page faults + two ring resets on `0000:03:00.0`, both from
  `probe/dgc` (null-ish GPU VA `0x0000800000000000`), both recovered by the driver;
  `journalctl -k` shows no `Illegal opcode`, and every subsequent run reproduced its
  pre-fault values. Proposed incident-log row: *2026-10-27, `dgc` (DGC EXT compute
  MWE), RX, gfxhub page fault → ring timeout → recovered by reset; cause: DGC execute path
  with an unresolved VA, not a kernel defect.*
