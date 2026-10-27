# 07 — GPU-side attribution on RADV, and Ollama

Round 7 (2026-10-27). Paired round against the live puzzle from §8ac: a
correct-looking chunk-level shared-expert restructure moved the stage
**20.74 → 18.05 ms/token (−13 %)** while per-dispatch pricing said the routed GEMMs
are ~0.9 ms/token. So **~17 ms of that stage is unattributed** and we are guessing
with coarse host-side timers. Part A answers "what can actually attribute GPU time
here, and how do we instrument 20 ms across ~20 stages without changing the
number?" Part B audits Ollama (rotation #9).

**Everything in Part A is measured on this box**, on the headless RX 6950 XT
(`0000:03:00.0`, RADV NAVI21, mesa 26.2.4, kernel 6.x amdgpu), one process at a
time, `timeout`-wrapped, validation off except where noted. Nothing was run on the
display card (Radeon VII / MI50, `0000:0e:00.0`).

---

## TL;DR

1. **Three mechanisms work and are enough**: `vkCmdWriteTimestamp2` query pools
   (+`timestampPeriod`), `VK_EXT_calibrated_timestamps` (to put GPU ticks on the
   host clock), and `VK_AMD_buffer_marker` (progress only, no clock).
   A fourth is advertised and **broken** (`VK_KHR_performance_query`: 17 counters,
   enumerates fine, `vkCmdBeginQuery` segfaults in `libvulkan_radeon.so`).
   RGP is impossible on this kernel (no RGP support in amdgpu).
2. **A timestamp costs 306 ns** (RX). Per-*stage* instrumentation is free
   (40 stamps = 12 µs on a 20 ms token = 0.06 %); per-*dispatch* instrumentation is
   not (3 018 stamps ≈ 0.92 ms ≈ 4.6 %).
3. **The missing gain is a boundary problem, not a dispatch problem.** Measured on
   the RX: in-batch dispatch+barrier = **9.5 µs**, but a submit+fence round trip has
   a **fixed 56 µs** on top of whatever the GPU does, and a command-buffer boundary
   *inside one submit* costs **21–34 µs of GPU timeline**. Removing 167 of 192
   dispatches can only ever recover ~2.7 ms — which is exactly what was measured.
   The stage is paid for per *boundary*, and the restructure removed no boundaries.
4. The recipe that ends the guessing: **one query pool per device, 2 stamps per
   stage boundary, printed as a three-column table (host wall / GPU span /
   host−GPU)**. Any stage whose `host − GPU` exceeds ~70 µs contains a submit+fence;
   `s.submits` per token is already counted, so the boundary count is one division
   away.
5. **The relaunch (Part C) found that the *next* puzzle was not an attribution problem.**
   The "~95–105 ms of a 155–166 ms agent-token is unattributed" claim does not survive
   measurement: `sum ≡ wall` to ≤0.13 ms on six chunk traces, and the 155–166 ms is
   **2 × the mean per-stream prefill wall** (`bench agents` prefills the streams
   sequentially and divides by `n`) plus a tail chunk that runs at 8× the per-token cost.
   Read **C1** before instrumenting anything, and **C2** for the re-priced tool table.
6. **Ollama** is now a Go supervisor over a `llama-server` subprocess per model
   (77 MB shallow clone at `6c15b26`). Everything about *kernels* is
   llama.cpp-inherited; what Ollama adds is a **runner scheduler** (ref-counted
   runners, keep-alive expiry, LRU eviction, max 3 models/GPU), a **fit-then-degrade
   load loop** (predict VRAM → retry with smaller auto context/batch on OOM), and
   **backpressure** (bounded request queue). Ranked adaptations in §B4; the top one
   is a residency *lease* with idle expiry for the expert pool.

---

# Part A — attributing GPU time on RADV

## A1. What actually works here

| mechanism | status | where it helps | minimal setup |
|---|---|---|---|
| `vkCmdWriteTimestamp2` + TIMESTAMP pool | **works** | per-stage *and* per-dispatch GPU spans | 1 pool/device, `query_count = stages*2`, `reserved=0`; `vkCmdWriteTimestamp2(COMPUTE_SHADER)` |
| `vkCmdWriteTimestamp` (v1, TOP/BOTTOM_OF_PIPE) | works, **costlier** | legacy only | same pool; **+210 ns/stamp, +5.8 %** vs v2 on a 32-dispatch chain — don't use for new code |
| `VK_EXT/KHR_calibrated_timestamps` | **works** | placing GPU ticks on the host timeline (host gap vs GPU busy) | `vkGetCalibratedTimestampsEXT` with `CLOCK_MONOTONIC`; see the unit trap in A3 |
| `VK_AMD_buffer_marker` | **works** | "which dispatch has the GPU reached" (hang forensics, OOB bisect) | `vkCmdWriteBufferMarkerAMD(BOTTOM_OF_PIPE, host-visible buf, off, u32)`; markers verified in host memory (`[1,2,3,4]`) |
| `VK_KHR_performance_query` (HW counters) | **advertised, broken** | would have given VALU busy / L2 hit / VRAM bytes per dispatch | 17 counters exist on **qf0 only**; `vkCmdBeginQuery` on a PERFORMANCE_QUERY pool **SIGSEGVs** in `libvulkan_radeon.so` |
| Radeon GPU Profiler (RGP / SQTT) | **impossible** | wave-level timeline, the real answer to "why 19 ms" | needs amdgpu RGP support: **0 `rgp` symbols in `/proc/kallsyms`, no `/dev/rgp`** |
| `umr`, `radeontop`, `amdtop` | not installed | register/occupancy inspection; coarse GPU busy | Arch packages; passwordless sudo available, `debugfs/dri/0000:03:00.0` reachable as root |
| `perf` (linux-tools) | **not installed** | host-side: which thread burns the 17 ms (recording vs waiting) | `perf_event_paranoid = 2` (per-process OK); installing `perf` is the single highest-value host-side add |
| `sysprof` / `perfetto` | not installed; no Vulkan layer present | cross host+GPU timeline | would need a build; the calibrated-timestamp recipe replaces it for our purpose |
| `RADV_DEBUG` | works | `shaderstats` (per-pipeline VGPR/wave/instruction counts), `info`, `checkir`, `syncshaders`, `hang`, `pipeline`, `startup`, `noopt`, `nocompute` | `RADV_DEBUG=shaderstats ./bench …` (options read from `libvulkan_radeon.so`; `RADV_DEBUG=bogus` is silently ignored, there is no help list) |
| `ACO_DEBUG` | works | `validateir`, `validatera` only in this build | `ACO_DEBUG=validateir` |
| `/sys/class/drm/card*/device/gpu_busy_percent` | works, **card-wide only** | occupancy sanity while a bench runs; cannot split two streams or two passes | already used by `harnesses/bench_attrib.sh` |

What this means in practice: on this box there is **no hardware-counter and no
profiler path**, so attribution must come from (a) engine-side timestamps and
(b) scratch-harness A/B ablations of the exact shapes. Both are cheap; both are
reliable; neither requires new tooling.

### A1.1 Device facts worth writing down

| | RX 6950 XT (NAVI21) | Radeon VII (VEGA20) |
|---|---|---|
| `timestampPeriod` | **10 ns/tick** | **40 ns/tick** |
| `timestampValidBits` | 64 (qf0 graphics+compute, qf1 compute, qf3 video-encode); 0 on video-decode / sparse | same shape |
| `timestampComputeAndGraphics` | true | true |
| `VK_EXT_calibrated_timestamps` | rev 2 | rev 2 |
| `VK_AMD_buffer_marker` | rev 1 | rev 1 |
| `VK_KHR_performance_query` | **advertised**, 17 counters on **qf0 only** (0 on qf1) | **not advertised** (the counter enumerator still answers 17 on qf0 — driver inconsistency, moot: record path crashes) |
| `VK_KHR_pipeline_binary`, `VK_EXT_descriptor_buffer`, timeline semaphore, hostQueryReset | yes | yes |
| `VK_KHR_cooperative_matrix` | no | no |

Note the queue-family trap if anyone ever revisits counters: **a dedicated
compute-only family (`qf1`) reports zero counters**; the engine already takes the
*first* family containing `COMPUTE`, which is `qf0`, so a counter build would run on
the right family. The blocker is the crash, not the family.

## A2. What a timestamp costs, and the perturbation budget

Measured on the RX: 256 back-to-back `vkCmdWriteTimestamp2` in one command buffer
span **77.96 µs** → **305.7 ns per query**, flat from 8 to 256 queries (the first
pair reads 440–480 ns because the first stamp also pays the CB's start).

Perturbation of a real workload — *n* dispatches (256 KiB touch + full
`ALL_COMMANDS` barrier each), priced with 2 outer stamps versus 2 stamps per
dispatch:

| dispatches | 2 stamps | 2n stamps | delta |
|---:|---:|---:|---:|
| 4 | 43.9 µs | 46.1 µs | **+5.1 %** (+560 ns/dispatch) |
| 16 | 148.9 µs | 158.6 µs | **+6.5 %** (+605 ns/dispatch) |
| 64 | 568.6 µs | 610.2 µs | **+7.3 %** (+650 ns/dispatch) |

So per-stamp ≈ 306 ns and the price of *instrumenting a dispatch* is ~600 ns —
i.e. **6 % of a 9.5 µs dispatch+barrier**. That is the whole story for the budget:

* **Per stage (recommended)**: 20 stages × 2 stamps = 40 × 306 ns = **12 µs on a
  20 ms stage = 0.06 %.** Invisible. Do this always, in the shipping build.
* **Per dispatch (attribution runs only)**: ~1 509 dispatches/token × 2 stamps =
  3 018 × 306 ns = **0.92 ms ≈ 4.6 % of a 20 ms token** — over the 2 % rule, so it
  needs a flag, a single-block scope, and a correction of −306 ns/stamp when
  comparing against the uninstrumented number.
* **Max queries in flight is not the limit** (maxQueryCount is effectively
  unbounded here; 512-query pool, `RESET_COMMAND_BUFFER` pool, `hostQueryReset`
  available). The limit is *readback*: `vkGetQueryPoolResults(WAIT)` on a query the
  GPU never wrote **blocks forever** — this hung the research harness until
  `timeout` killed it. Always bound it (`device_wait_idle()` first, or read without
  `WAIT` after a known fence), which is the same discipline as `WAIT_TIMEOUT_NS`.

## A3. The unit trap: calibrated ticks are **not** query ticks

`vkGetCalibratedTimestampsEXT(CLOCK_MONOTONIC)` on RADV returns the device-domain
clock in **nanoseconds**, while query-pool timestamps are the **raw 100 MHz ticks**
that need `× timestampPeriod`. Getting this wrong is a silent **10× error on the RX
and 40× on the MI50**. Verified against `/proc/uptime`:

```
/proc/uptime            = 1736.91 s   (booted 28.9 min ago)
calibrated raw value    = 1737753435877   -> 1737.8 s   (already ns; DO NOT ×10)
query tick × 10         = 1731249xxxx×10  -> 1731 s     (ticks; MUST ×10)
```

With the correct anchoring the calibration is excellent: 20 samples give a
sample-to-sample offset **spread of 521 ns** and a declared `maxDeviation` of
52–143 ns. So `host_ns(gpu_tick) = tick × period + (host_now − calibrated_now)`
holds to well under a microsecond across a benchmark window — which is what makes
the three-column table in A6 possible.

## A4. Why "fewer dispatches" bought 13 % and not two orders

Four costs, all measured on the RX with the same 256 KiB dependent workload:

**(1) A synchronous submit+fence has a fixed 56 µs that has nothing to do with the
GPU's work.** Decomposed (median of 15; record / submit-return / fence-wait / total
in µs, plus the GPU span from timestamps):

| payload | record | submit | wait | total | GPU span |
|---|---:|---:|---:|---:|---:|
| empty CB | 0.1 | 7.2 | **56.1** | 63.4 | 0.4 |
| 1 dispatch (256 KiB) | 0.4 | 7.8 | 73.9 | 82.3 | 17.6 |
| 12 dispatches (256 KiB) | 2.7 | 8.2 | 174.2 | 185.4 | 113.8 |
| 12 dispatches (4 MiB) | 2.7 | 8.2 | 355.5 | 366.6 | 295.2 |

`wait ≈ GPU span + 56 µs` in every row. Two consequences: **batching 12 dispatches
costs the same fixed overhead as batching 1**, and any code path that submits and
waits per unit pays 56 µs per unit no matter how small the unit is.

**(2) It is not a host wake-up.** Replacing `vkWaitForFences` with a spin on
`vkGetFenceStatus` changes the empty-CB round trip from 57.6 → 54.5 µs. Spinning
does not help, so the 56 µs is inside the submit→signal path (RADV submit +
doorbell + GPU start when idle), not the scheduler. There is nothing to win by
busy-waiting.

**(3) A command-buffer boundary costs 21–34 µs of GPU timeline, even inside one
submit.** Two CBs, four dispatches each, submitted once, no host wait between them:
the gap from CB1's last dispatch-end stamp to CB2's first stamp is **33.3–34.4 µs**,
while an in-CB dispatch+barrier is **4.4 µs**. With the barriers removed the gap is
still **20.8–21.1 µs** — so ~21 µs is the CB prologue/epilogue itself and ~12 µs is
the trailing `ALL_COMMANDS` barrier's drain. *Pass boundaries are ~5–8× more
expensive than a dispatch.*

**(4) Extra submits add both wall time and GPU timeline.** 12 dispatches in 1, 2 and
4 submits (fence after each): wall **185 / 263 / 424 µs**, GPU span **113.8 / 122.5 /
140.3 µs**. Each extra boundary ≈ **+62–65 µs wall, +9 µs GPU**.

**(5) A barrier's price tracks the dirty footprint, not the barrier flavour.**
Four dispatches with versus without a barrier, per-dispatch barrier cost:
256 KiB → **3.56 µs**, 1 MiB → **5.36 µs**, 4 MiB → **5.71 µs**, 16 MiB → **5.68 µs**
(full `ALL_COMMANDS` and `COMPUTE→COMPUTE` v2 are within 0.1 % of each other here).
Round 6's 3.43 µs at 256 KiB reproduces exactly; the useful new fact is that the
**knee is at ~1 MiB** — anything that streams MBs of weights per dispatch pays ~5.5 µs
per barrier, so "1 509 barriers → ~400" is worth ~6–8 ms/token, not 1 ms.

### The arithmetic that closes §8ac

The restructure removed ~167 of 192 shared-expert dispatches and bought 2.7 ms.
2.7 ms / 167 = **16 µs per dispatch removed** — which matches the measured
per-dispatch cost on this workload (9.5 µs dispatch+barrier in-batch; 15.4 µs when
each dispatch drags a submit boundary with it). The §8ab prediction of "two orders"
came from *kernel time alone* (0.03 ms/token) and silently assumed the *launch*
cost was per-dispatch. It is per-**boundary**:

```
stage_time ≈  Σ dispatches × (kernel + barrier)        (9.5–15 µs each)
            + Σ submits    × 56 µs                      (host, fixed)
            + Σ CB/submit boundaries × 21–34 µs         (GPU timeline)
            + host recording                            (0.2 µs/dispatch here;
                                                         ~0.2–1 µs in the engine,
                                                         which allocates a fresh
                                                         descriptor set per dispatch)
```

For 18 ms/token, either there are ~300 synchronous boundaries per token, or the
handful of remaining stages are 100× their priced kernel time. Those two
possibilities are distinguished in one run by the table in A6 — no guessing.

Two more candidates worth naming because they are Vulkan-visible, not arithmetic:

* **implicit sync between our phases.** `pass1_flush` waits a timeline marker, not
  just a fence; a marker wait that lands *after* the GPU already finished costs the
  same 56 µs class of latency, and a marker that lands *before* stalls the host for
  the whole segment. Both show up as `host − GPU` in the table.
* **shape-change/cache effects.** A restructure that reads 4× more weight bytes per
  dispatch trades dispatch count for barrier price (3.56 → 5.7 µs) and for L2
  misses on the expert slab. The `footprint` protocol in A7 measures that directly.

## A5. Hardware counters: what is there, and why it does not help (yet)

`VK_KHR_performance_query` on the RX enumerates **17 counters, all
`VK_QUERY_SCOPE_COMPUTE`, on queue family 0 only**:

```
GPU active cycles, Waves, Instructions, VALU Instructions, SALU Instructions,
VMEM Load Instructions, SMEM Load Instructions, VMEM Store Instructions,
LDS Instructions, GDS Instructions, VALU Busy, SALU Busy,
VRAM read size, VRAM write size, L0/L1/L2 cache hit ratio
```

Exactly what one would want for "is the shared expert compute-, memory- or
occupancy-bound". It cannot be used: creating the pool succeeds, the profiling lock
succeeds, and then **`vkCmdBeginQuery` on that pool segfaults inside
`libvulkan_radeon.so`** (reproduced twice, once from C with raw loader
`vkGetInstanceProcAddr` pointers, once from `ash`):

```
profiling lock acquired
rec: reset perf  /  rec: reset ts  /  rec: ts0  /  rec: begin perf
Segmentation fault  ->  libvulkan_radeon.so + 0x... (SIGSEGV, core dumped)
```

The GPU is untouched by this (it is a driver-side crash in the host process; the box
stayed healthy, no `amdgpu` messages). Re-test after any mesa bump: it is a
one-line check (`probe/perfq2.c`). Until then, treat counter-based attribution as
unavailable and use shape ablation instead.

## A6. Recommended instrumentation recipe (the deliverable)

Goal: attribute 20 ms across ~20 stages with **< 2 %** distortion, and tell host
time from GPU time. Measured budget says this costs 0.06 %.

**Step 1 — one query pool per device, per-token stage stamps.** The plumbing already
exists (`breccia_vk::ctx::query_pool/timestamps/reset_queries`, `run_timed`); what is
missing is *stamps at stage boundaries inside a real decode submission*.

```rust
// once per device
let pool = gpu.query_pool(64)?;            // 32 stages × 2, wraparound per token
// inside the block/segment recording, after the stage's last dispatch:
unsafe { dev.cmd_write_timestamp2(cmd,
    vk::PipelineStageFlags2::COMPUTE_SHADER, pool, slot) };
```

Stamp **both** ends of each of: `shexp_gate`, `gate`, `up`, `swiglu`, `down`,
lane-copy/`pf.shared` handoff, and every `flush()`/`end_submit` boundary
(segment start + segment end). Use `cmd_write_timestamp2` with `COMPUTE_SHADER`
(v1 `TOP/BOTTOM_OF_PIPE` costs +210 ns and +5.8 % on a dispatch chain).

**Step 2 — host stamps at the same points, on the same timeline.**
`Instant` around each stage gives host wall; `VK_EXT_calibrated_timestamps` (one
sample per window, `CLOCK_MONOTONIC`) converts GPU ticks into that same host clock —
**remembering A3: calibrated values are ns, query values are ticks.**

**Step 3 — print three columns per stage, per token** (shapes below are illustrative;
the point is the *pattern*, the numbers come from the run):

| stage | host µs | GPU µs | host − GPU µs | what it means |
|---|---:|---:|---:|---|
| `shexp_gate` | 78 | 0.4 | 77.6 | **contains a submit+fence** (56 µs) |
| `gate` | 12 | 9.0 | 3.0 | recorded in-batch, healthy |
| `down` | 14 | 10.5 | 3.5 | healthy |
| `pass1_flush` | 61 | 4.0 | 57.0 | pure boundary cost |

Interpretation rule, taken straight from A4: `host − GPU < ~10 µs` → the stage is
GPU/recording; `≈ 56–70 µs` → it owns a synchronous boundary; `≥ 100 µs` → it owns
several, or a marker wait. Summing `host` must reproduce the 18 ms; if it does not,
the missing time is *between* stages (recording outside the stamped region), which
is itself the answer.

**Step 4 — count boundaries, don't guess them.** `Session.submits` and
`Session.dispatches` already exist per run. One division gives
`boundaries/token`; multiply by 56 µs (host) and 21–34 µs (GPU) and compare with the
stage. If `boundaries × 56 µs` ≈ the unattributed time, stop touching kernels.

**Step 5 — only then, per-dispatch stamps (attribution-only build).** 2 stamps per
dispatch for **one block of one token** (≈ 8 dispatches), with the −306 ns/stamp
correction; never for a measured throughput number.

**Rules that came out of building this** (each one cost a debugging round):

* Bound every `vkGetQueryPoolResults` — `WAIT` on an unwritten query blocks forever.
* Reset the pool *inside* the command buffer (`vkCmdResetQueryPool`) if a token can
  be cancelled, or a reused pool returns the previous token's stamps.
* Never mix calibrated values with query values without the unit check (A3).
* Timestamps are 64-bit on every family we use; the transfer/video-decode family has
  `timestampValidBits = 0` — a stamp recorded in a CB submitted to that queue is
  undefined. The engine's `Qc::X` (transfer) queue must not carry stamps.

## A7. Scratch protocols (in `/var/tmp/users/korphaus/tmp/opencode/research-harnesses/r07-attrib`)

```bash
cd /var/tmp/users/korphaus/tmp/opencode/research-harnesses/r07-attrib
cargo build --release --offline          # ash 0.38, glslangValidator + spirv-val in build.rs
timeout 90 ./target/release/r07-attrib --device NAVI21 caps
timeout 90 ./target/release/r07-attrib --device NAVI21 query       # per-stamp cost + perturbation
timeout 90 ./target/release/r07-attrib --device NAVI21 calib       # round-trip decomposition
timeout 90 ./target/release/r07-attrib --device NAVI21 chain       # CB boundary, one submit
timeout 90 ./target/release/r07-attrib --device NAVI21 footprint   # barrier vs dirty bytes
timeout 90 ./target/release/r07-attrib --device NAVI21 submit      # 1 vs 2 vs 4 submits
timeout 90 ./target/release/r07-attrib --device NAVI21 tsver       # v1 vs v2
timeout 90 ./target/release/r07-attrib --device NAVI21 marker      # VK_AMD_buffer_marker
cd probe && gcc -O1 -o perfq  perfq.c  -lvulkan && timeout 60 ./perfq    # counter enumeration
             gcc -O1 -o perfq2 perfq2.c -lvulkan && timeout 60 ./perfq2   # record path (SIGSEGV = still broken)
```

Two measurement traps this harness had to fix, worth carrying into the engine:
a *single* dispatch's measured span includes ~0.7 µs of CB start-up (so price
marginals from `chain(n)`–`chain(2n)`, never from `single × n`), and a "time N
dispatches in a loop" protocol overlaps them (round 6 saw 137 ns vs 1.07 µs
dependent — the same trap, still armed).

---

# Part B — Ollama audit

Shallow clone OK: `Reference Projects/ollama`, `6c15b26` ("llama.cpp: version bump
b11901"), 77 MB. Architecture has changed since the versions most write-ups
describe: **there is no in-process runner any more.** `llm/llama_server.go` (3 030
lines) launches a **`llama-server` subprocess per model** on a random localhost port
and talks HTTP to it; `server/sched.go` (1 849 lines) is a scheduler over those
processes. `mlxrunner/` exists for Apple only.

## B1. What Ollama actually adds (vs llama.cpp-inherited)

| concern | Ollama-specific | inherited from llama.cpp |
|---|---|---|
| batching | nothing beyond choosing `-np`/`-b`/`-ub` | continuous batching, slot handling, prompt cache |
| KV/context | `-c numCtx × numParallel`, `--context-shift`, `--keep`, `--kv-unified`, `--cache-ram 0` for one decision mode | KV cache types, context shift, keep-tokens |
| model management | **scheduler, ref-counts, keep-alive timers, LRU eviction, max 3 models/GPU, reload-on-option-change** | — |
| memory fit | **predict-then-degrade load loop, per-GPU VRAM accounting scraped from the child's log** | `-ngl` layer offload, mmap |
| backpressure | **bounded pending queue (`OLLAMA_MAX_QUEUE`) → `ErrMaxQueue`** | — |
| store/distribution | manifest/blob store, pull/upload, registry | — |

## B2. The scheduler (the part worth reading)

`server/sched.go`:

* **Runners are ref-counted and identified by model + resolved options.**
  `needsReload` (line 1423) compares the *request's* resolved `Runner` options
  against the loaded runner's: any difference in `NumCtx` (after clamping to the
  model's train context), `NumBatch`, `NumGPU`, mmap mode, adapters, projectors or
  `contextShift` triggers a **full reload** rather than a reconfigure. Health is a
  `Ping` with a 10 s timeout (2 min while loading).
* **Keep-alive is a per-runner timer, resettable per request.** Default
  `OLLAMA_KEEP_ALIVE = 5 min`; `< 0` means "never expire" (`MaxInt64`); a request
  may override it, and `keep_alive: 0` with an empty prompt is the documented
  *unload now* RPC. When a runner goes idle: if `sessionDuration <= 0` it is
  expired immediately, else `time.AfterFunc` with `expiresAt = now + duration`, and
  the timer is **reset** (not recreated) if the runner goes idle again.
* **`OLLAMA_MAX_LOADED_MODELS` defaults to auto = 3 × GPU count**
  (`defaultModelsPerGPU = 3`, line 121) with the comment "loading many small models
  on a large GPU can cause stalling" — i.e. the cap is a *latency* guard, not a
  memory guard. When the cap is hit, `findRunnerToUnload` picks a victim
  (idle/LRU) and the pending request is retried in a loop.
* **Admission control**: `pendingReqCh` is a channel of size `OLLAMA_MAX_QUEUE`
  (default 512); a full queue is answered with an error, not queued.
* **`OLLAMA_NUM_PARALLEL` defaults to 1** and is forcibly 1 for model families that
  cannot batch across slots (mllama, qwen3-vl/moe, qwen3-next, lfm2, nemotron_h, …).
  The subprocess gets `-c numCtx × numParallel` and `-np numParallel`, so the KV
  budget is *pre-divided* by parallelism.

## B3. Load-time fit logic (the most transferable thing here)

Two loops, both in `sched.go`:

1. **`updateFreeSpace`** (line 1335): predicts each loaded runner's per-GPU usage
   via `VRAMByGPU(deviceID)` and subtracts it before admitting the next model.
2. **`reduceAutoNumCtxForLoadOOM`** (line 811): if the load failed (or the predicted
   context would not fit), it **degrades exactly the auto knobs** — `NumCtx` and
   `NumBatch`, *not* user-specified values — and retries once. `numCtxAuto` /
   `numBatchAuto` flags record which values came from the auto-tier so that a later
   `needsReload` comparison does not treat them as user intent (line 1441).

The accounting itself is honest about its own fragility: `VRAMByGPU` reads
`s.vramByDevice[name]`, filled by **parsing `buffer size` lines out of the child's
`--log-verbosity 4` output**, with a startup check that logs
`"llama-server VRAM tracking: no per-device buffer sizes were parsed…"` if the log
format ever changes (`llama_server.go:1106`). It is a pragmatic "measure the real
allocator, not the estimate" design — with a log-format coupling that is one
refactor away from silently reporting zero.

Also load-time: `maybeDisableMmapForHostPressure` (line 1236) decides mmap vs
`--no-mmap` from predicted size versus **host memory pressure** — the same trade we
just settled from the other side in `docs/research/03` (page cache is our friend;
`read_ahead_kb` was the bug), and a useful reminder that a shipping runtime treats
"mmap" as a *policy*, not a constant.

## B4. Ranked adaptations for a 2–6 agent service

1. **Residency *lease* with idle expiry (highest value).** Ollama's
   `sessionDuration` + `expireTimer` + `expiresAt` + `maxRunners` is 40 lines that
   solve a problem we still pay for every token: which expert/weight set stays
   resident and for how long. Adapt as a `Lease { model_key, opts_hash, refcount,
   expires_at }` on the expert pool: pin residency across a conversation, release
   on idle timeout, and *reload rather than re-plan* when the option hash changes
   (`needsReload` semantics). Measurable: lanedecode tok/s with a 30 s idle gap
   between two requests, before/after (today the pool re-plans per token).
2. **Predict-then-degrade admission instead of failing mid-token.** Before a request
   is admitted, predict VRAM (arena + KV + chunk buffers at the requested
   `ctx × USERS_MAX`) and, if it does not fit, degrade one *named* auto knob
   (context, then chunk size) once and retry; otherwise reject. Pair it with a
   bounded pending queue (Ollama's `ErrMaxQueue`) so a 6-agent burst is
   backpressured at the door instead of thrashing residency. Measurable: p99
   tok/s per agent and zero OOM/restart events over a 6-agent burst.
3. **Engine-local VRAM accounting from the allocator's own report.** Adopt the
   *idea* (not the log scraping): every allocation site already knows its size —
   emit one structured `alloc` line per arena/workspace/ring and reconcile it
   against `/sys/class/drm/card*/device/mem_info_vram_used` at load. That is the
   check that would have caught the "planned vs actual residency" class of bug
   (and it is cheap: one line per allocation, one read per load).
4. Honourable mentions, in order: `--kv-unified`/`--cache-ram 0` for the "many
   one-shot decisions" mode (our `prefill_merged` analogue); `-c = ctx × parallel`
   pre-division (we do the same with `USERS_MAX`, but Ollama makes the trade
   explicit in one place); `-np 16` for short decision prompts, which is the same
   "small requests want slots, long requests want context" split we get from
   `prefill_chunk`; and `--flash-attn auto|on|off` resolved once at load, never per
   request.

What does **not** transfer: everything kernel-level (it is llama.cpp), the
subprocess-per-model architecture (we have one process and a dual-GPU placement
problem Ollama does not attempt), the JSON/manifest store, and the HTTP API shapes.

---

## Top-3 next experiments

1. **Attribute the 18 ms in one run (Part A recipe, steps 1–4).** Add the stage
   query pool to `record_dense_prelude` + the shared-expert stage, print
   `host / GPU / host−GPU` per stage over 64 decode tokens (2 streams), and divide
   `Session.submits` by tokens. **Falsifiable prediction**: ≥ 60 % of the 18 ms
   appears in `host − GPU` on stages that own a submit/fence, and the boundary count
   is ≥ 150/token. If instead `host − GPU < 5 µs` everywhere, the time is GPU-side
   and the next tool is `RADV_DEBUG=shaderstats` plus shape ablation (A4/A7).
2. **Price the boundary, then merge it.** Take the segment count per block (2 today:
   pre-router + shared-expert) to 1 with a single `end_submit` per block, and measure
   with the `chain` protocol: expect **−21…−34 µs GPU and −56…−62 µs host per
   boundary removed**, i.e. −2…−5 ms/token at ~96 boundaries/token. This is a
   one-flag experiment and it is worth strictly more than another dispatch fusion.
3. **Install `perf` and attribute the host half.** `perf` is absent on this box
   (`perf_event_paranoid = 2`, passwordless sudo available) and the host is where
   the 56 µs and the recording cost live. `perf record -g` on one 32-token decode
   run settles "recording vs waiting vs lane copies" without touching the engine —
   the one measurement that needs zero code changes.

## Rotation

Ollama is **done** (#9). Next: **DeepSpeed-Inference (#7)** — chosen over the
exllamav2 follow-up because its tensor/KV *partitioning* machinery is the closest
published analogue to our standing dual-GPU placement policy (which card holds which
resident block), whereas exllamav2's remaining surface (bits_prop, CUDA graphs) was
already priced as non-portable in `docs/research/03`. Keep exllamav2 as a follow-up
if a KV-quantisation path ever becomes admissible.

---

# Part C — relaunch addendum (second pass of round 7, same day)

The first pass (Part A/B above) answered "what can attribute GPU time here". This pass
was relaunched against a **different, live puzzle** — *"the chunked-prefill stage table
accounts for only ~60 ms of a ~155–166 ms agent-token; ~95–105 ms is unattributed"* — and
re-measured the same tools independently from a second harness
(`/var/tmp/users/korphaus/tmp/opencode/research-harnesses/rl7-attrib`, own crate, copies
`harnesses/vk-probe/src/ctx.rs` verbatim). Same box, same card, same discipline.

It adds three things and confirms one:

* **C1 — the ~95–105 ms is a denominator artefact, not missing work.** The host stage
  table is complete to ≤0.13 ms; the 155–166 ms figure is 2× the mean per-stream wall
  because the bench prefills the streams sequentially, plus tail-chunk inflation.
* **C2 — an independent re-measurement of the tool costs**, and a reconciliation with
  A2's 306 ns/stamp (a stamp is a *dependency edge*: its marginal cost tracks the work in
  flight, 0.3 µs dense → 2.1 µs after a real dispatch).
* **C3 — what else is free here** (`gpu_busy_percent` as a 20–40 ms per-card duty-cycle)
  and what is not even packaged (`umr`, `rgp`, `amdtop`).
* **C4 — four Ollama details** that Part B's B4 did not name.

## C1. The live puzzle: `sum ≡ wall`, so the missing time is in the denominator

### C1.1 Measured: the coarse stage table has no residual

`BRECCIA_TRACE_PREFILL=1` already prints `sum · wall · residual` per chunk
(`graph.rs`, end of `prefill()`). Six chunk traces, two cards, warm and cold, `t=64` and
`t=4`:

| run | stream | chunk `t` | stage sum | wall | residual (ms/token) |
|---|---|---:|---:|---:|---:|
| `--streams 1`, RX | 0 | 27 | 84.03 | (first chunk) | — (0 % needed a 2nd chunk) |
| `--streams 2`, prefix 132 | 0 (RX) | 64 | 87.68 | 87.71 | **+0.03** |
| | 1 (MI50) | 64 | 107.06 | 106.93 | **−0.13** |
| | 0 (RX) | 4 | 660.87 | 658.20 | −2.67 (0.4 %, the tail chunk) |
| | 0 (RX, warm) | 64 | 167.59 | 167.61 | **+0.01** |
| | 1 (MI50) | 64 | 178.84 | 178.71 | **−0.13** |
| `--streams 2`, prefix 64 | 0 (RX) | 64 | 67.62 | 67.65 | **+0.03** |
| | 1 (MI50) | 64 | 316.50 | 316.51 | **+0.01** |

`sum ≡ wall` to 0.01–0.13 ms. Every stage mark is on the critical path and every
millisecond of the chunk lands in a stage. There is no hidden ~100 ms of host work in the
record path.

### C1.2 Mechanism: the bench divides by `n` over a *sequential* prefill

```rust
// crates/breccia-cli/src/main.rs:2080 — stream 0's whole prompt, then stream 1's
if a.prefill_chunk > 0 {
    for (s, sess) in multi.streams.iter_mut().enumerate() {
        for part in padded[s].chunks(a.prefill_chunk) { logits[s] = sess.prefill(part, pos)?; }
    }
}
// main.rs:2234 — and then:
"... {:.2} tok/s per agent", prompt_tokens as f64 / prompt_seconds / n as f64
```

The two prefills never overlap, so the reported per-agent rate is **half** the
single-stream rate, i.e. **per-agent ms/token = 2 × mean per-stream wall**. Measured on
one run (`--streams 2 --prefix 64 --prefill-chunk 64`, one chunk each):

```
stream 0 (RX)    67.62 ms/token      stream 1 (MI50)   316.50 ms/token
reported         2.60 tok/s per agent  →  385 ms/agent-token
check            (67.65 + 316.51)/2 × 2 = 384                      ✓
```

### C1.3 Second contributor: the tail chunk is 8× the per-token cost

`prefix mod chunk` tokens pay the chunk's *fixed* cost (48 blocks × PLE, cold rows, one
drain each) divided by a tiny `t`:

| chunk | t | stage sum (ms/token) | dominant stages |
|---|---:|---:|---|
| 1st | 64 | 87.68 | `experts(host)` 61.8, `wait-gpu` 15.9 |
| 2nd | 64 | 107.06 | `experts(host)` 81.6 |
| **tail** | **4** | **660.87** | `experts(host)` 569.3, `wait-gpu` 73.1, `ple` 9.7 |

`132 = 64 + 64 + 4`, so the default gate prefix hits it every run: 4 × 0.66 s = 2.6 s of
a ~20 s prompt (**13 %**) at 8× the per-token rate.

**So**: 60 ms own + ~60–100 ms the other stream (serialised, in the denominator) + tail
inflation = 155–166 ms. No engine defect, and **nothing for the attribution tooling to
find** — which is exactly why three previous rounds of "the table must be lying" produced
nothing.

### C1.4 If a residual ever does appear, look here first

1. The one un-instrumented `flush()` per chunk: `wall0` is taken *before*
   `self.flush()` and `timers.start()` *after* it (`graph.rs:1588-1595`), so one pass-1
   drain per chunk is in `wall` and in no stage.
2. `timers.clear()` at the end of a traced chunk — a per-token average can mix windows.
3. **GPU idle *inside* a batch** (host recording slower than the GPU executes). This is
   the only place a batched submission can hide time, it is 0.3 % in the synthetic case
   (C2.2), and it needs the timestamp + calibration recipe to see at all.

### C1.5 The three experiments this decides

1. **Interleave the two streams' chunked prefill** — the same block-pair interleave the
   decode path already uses (`multi.rs::token_all`). Today the RX's host lane (38–48
   ms/token) runs with the RX GPU idle *and* the MI50 prefill not yet started. Overlap of
   stream A's lane/PLE with stream B's GPU is worth up to `min(lane, gpu)` per block-pair,
   and it makes the per-agent metric honest. ~40 lines in the CLI bench.
2. **Print `gpu_busy` per chunk.** The probe (`BRECCIA_GPU_TIME`) is reset after the
   prompt and only reported once a token in decode (`graph.rs:1145 probe_report`), so
   **prefill has no GPU-side number at all**. Without it, "4.4 µs per dispatch of drain"
   (§8af) is `wall / dispatches`, an inference; with it, GPU-busy vs wall *per pass*
   separates "the GPU is executing 1 900 dispatches" from "the GPU is waiting for the
   host to record them" — opposite fixes (barrier pruning vs T-batching).
3. **Kill the tail chunk**: chunk sizes that divide the prefix, or append the tail rows to
   the previous chunk. Free, and it removes 13 % of a 132-token run.

## C2. Independent re-measurement of the tools (same card, second harness)

### C2.1 What works, one measured example each

Device: `RX 6950 XT (RADV NAVI21)`, radv 26.2.4, Vulkan 1.4, `timestampPeriod 10.0 ns`,
`timestampValidBits 64` on both compute families (0 on decode/sparse-only).

| tool | status here | measured example | verdict |
|---|---|---|---|
| `vkCmdWriteTimestamp2`, 1 per region | **works** | 20 regions / 600 dependent dispatches, 1 submit: **17.72 ms** vs **17.65 ms** with no queries = **+0.36 %** | use it — backbone of the recipe |
| `vkCmdWriteTimestamp` v1 | works, same | 4.303 ms vs 4.292 ms (`WriteTimestamp2`) vs 4.293 (stage `NONE`) for 258 stamps | keep v1 only for compatibility |
| per-dispatch stamps (2 × 600) | works, **expensive** | **20.22 ms** = **+14.6 %** over 17.65 ms; marginal **2.10 µs/stamp** | **no** |
| stamp resolution | 40 ns between back-to-back stamps (p50, max 240 ns over 64) | tick 10 ns | fine for 100 µs stages |
| `VK_AMD_buffer_marker` | advertised + enabled | 256 markers **4.328 ms** vs 258 stamps **4.348 ms** (+0.5 %) | works, **no advantage** on one queue; the only option for a second device's timeline |
| `VK_EXT/KHR_calibrated_timestamps` | enabled; domains `DEVICE`, `CLOCK_MONOTONIC`, `CLOCK_MONOTONIC_RAW` | host−GPU offset spread **4.9–5.1 µs over 0.45 s** (unpinned sampler; A3's anchored figure is 521 ns) | use it — this is what puts host marks on the GPU timeline |
| `VK_KHR_performance_query` | advertised, 17 counters, **crashes** | enumerates (`GPU active cycles` CYCLES, `Waves`, `Instructions`, `VALU/SALU Busy`, `LDS/SMEM/VMEM`, `VRAM read/write size`, `L0/L1/L2 hit ratio`, all `COMMAND` scope, `CONCURRENTLY_IMPACTED`), then **SIGSEGV at `libvulkan_radeon.so+0x12d0b3`** — 6/6, five of them from A7's `perfq2` | **do not use** (independent confirmation of A5) |
| `gpu_busy_percent` (sysfs) | **free, no install** | 69 189 reads/s (0.01 ms/read); during a run the value changed **82× in 0.79 s ≈ every 20–40 ms**, spanning 0→99 % | the only zero-cost per-card sampler; 20–40 ms resolution |
| validation layer | installed | first run: one VU about un-reset queries (harness bug, fixed), **no** VU for `stage NONE` stamps or markers | keep first-running under it |

Availability, checked rather than assumed (`pacman -Ss`, no installs): `radeontop 1.4`,
`perf 7.2`, `sysprof 51`, `apitrace 14`, `renderdoc 1.46` are packaged; **`umr`, `rgp`,
`amdtop`, `rocm-smi`, `gfxreconstruct` are not in any configured repo**, and no RGP or
Perfetto capture layer is installed (`/usr/share/vulkan/explicit_layer.d` = validation
only; implicit = MangoHud, gamescope, anti-lag, device-select).

### C2.2 The engine-shaped recipe, priced end to end

20 stages × 30 dependent dispatches (barrier after each — a block-chunk's shape), A/B/A,
best of 3. Whole sweep: **0.79 s**.

| variant | wall | vs reference | GPU span sum | host/record residual |
|---|---:|---:|---:|---:|
| A untimed, one submit+fence per stage (today's engine) | 18.41 ms | — | — | — |
| B 20 GPU regions, still 20 submits | 18.43 ms | **+0.09 % vs A** | 17.66 | 0.77 ms (4 %) |
| C0 one submit, no queries | 17.65 ms | reference | — | — |
| **C 20 GPU regions, one submit** | **17.72 ms** | **+0.36 % vs C0** | 17.66 | **0.059 ms (0.3 %)** |
| D 1 200 queries, per dispatch | 20.22 ms | **+14.6 %** | — | — |

Consequences beyond A6: **batching submissions is worth more than the instrumentation**
(B → C0 = −4.2 %, i.e. **39 µs of host time per stage boundary**), and **once a pass is one
submit, `wall − GPU busy` is a trustworthy 0.3 % measurement**. Query budget is pure
arithmetic: `queries × 2.1 µs` — 40/token = 84 µs (0.06 % of 150 ms), 192/token (2 per
pass per block) = 0.4 ms (0.26 %), 4 800/token = 10 ms (**7 %**).

### C2.3 Reconciling 306 ns with 2.1 µs (both are right)

A2 measured 306 ns/stamp in a dense run; C2 measured 2.10 µs/stamp when each stamp sits
after a 2 000-FMA dispatch plus a barrier. A stamp is a **pipeline-flush edge**, so its
marginal cost grows with the work in flight in front of it — the same footprint law round
6a found for dependent barriers (1.6 µs at `bpr=2`, 5.6 µs at `bpr=10`). Budget rule that
covers both: **0.3 µs (floor) to 2.1 µs (worst case)**, and price *your* stamp density
with one 20-line A/B before adopting a per-stage table.

## C3. Ollama: four details Part B did not name

Same clone (`6c15b26`), same conclusion (Go supervisor over `llama-server` per model,
`llm/llama_server.go:365`). Additions to B4:

1. **A bypass queue, not just a scheduler** (`server/sched.go:518 useLoadedRunner`): a
   request whose model is *already loaded* never enters the pending queue — `refCount++`,
   stop the expire timer, hand back the runner. For 6 agents on one model this is the
   difference between "no head-of-line blocking" and "every turn queues behind a load".
2. **The admission threshold is a number: 80 % of free VRAM**, checked *before* spawning
   (`sched.go:576-600`), with the ladder `num_ctx` 256k→32k→4k (`nextLowerAutoNumCtx`),
   then `num_batch` tier (`generationBatchForContext`), then **one** evict-all retry
   (`oomRetryAttempted`) and fail fast.
3. **The ledger is scraped from the child, not predicted**: `memoryParsingWriter`
   (`llama_server.go:2862-2980`) parses the child's own `model buffer size / KV buffer
   size / compute buffer size` lines *and per-device free memory*, then uses those numbers
   for the next placement. B4's item 3 said "emit one line per allocation"; Ollama shows
   the stronger form — **read the runtime's own report back into the planner**.
4. **`num_parallel` is hard-forced to 1** for `qwen3next/qwen35moe/mllama/…`
   (`sched.go:546-552`) because those families have sequential state, and context is
   `-c = numCtx × numParallel` (`llama_server.go:393`) — per-slot context, never shared.
   Independent confirmation of §8af: our GDN/QSA state is sequential, so **slots buy
   nothing and the batching win must come from the chunk dimension** (T-batched dense
   chain), which is the item already parked behind barrier pruning.

## C4. Top-3 for the next implementation round (this pass's ranking)

1. **Interleave chunked prefill across streams** (C1.5-1) — the only change that both
   improves throughput *and* repairs the metric, on the path the gate already exercises.
2. **`gpu_busy` per chunk + calibrated-timestamp anchoring** (C1.5-2), then choose barrier
   pruning vs T-batching from the *measured* GPU-idle fraction.
3. **Chunk-size shaping** (C1.5-3) and a re-baseline: `132 → 64+68` (or `192 → 64×3`), and
   re-read every prefill number in §8z–§8ag for tail contamination before quoting it.

## C5. Safety log (this pass)

* One crash: `rl7-attrib perf` **SIGSEGV in `libvulkan_radeon.so+0x12d0b3`** on the
  `VK_KHR_performance_query` path, deterministic (five prior `perfq2` segfaults at the
  same offset in `journalctl -k`). **No `amdgpu` ring/reset/page-fault lines**, both cards
  idle at 0 % afterwards. Driver bug, not a kernel defect. This is the *third* independent
  reproduction; treat the extension as unusable on mesa 26.2.4.
* All harness runs on the **RX (`0000:03:00.0`)**, one process at a time, under
  `timeout`, first run with `VK_LAYER_KHRONOS_validation`.
* `bench agents --streams 2` (MI50 as the second stream) is the shipped gate path: two
  runs, ~25 s and ~40 s, no faults.

```bash
# C2 numbers (0.79 s per full sweep)
cd /var/tmp/users/korphaus/tmp/opencode/research-harnesses/rl7-attrib
timeout 120 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./target/release/rl7-attrib tools
timeout 90 ./target/release/rl7-attrib perf        # CRASHES by design of the driver bug

# C1 numbers
BRECCIA_TRACE_PREFILL=1 timeout 300 ./target/release/breccia bench agents \
  --streams 2 --devices 0000:03:00.0,0000:0e:00.0 --prefix 64 --prefill-chunk 64 --gen 1 --warm 1
BRECCIA_TRACE_PREFILL=1 BRECCIA_TRACE_SHARED=1 timeout 300 ./target/release/breccia bench agents \
  --streams 1 --devices 0000:03:00.0 --prefix 132 --prefill-chunk 64 --gen 1 --warm 1
```
