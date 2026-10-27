# 02 — Vulkan follow-ups (measured) + PowerInfer audit

Phase 4 paired research, round 2. Date: 2026-10-26. **Read-only**: no engine code changed, no
engine shader touched. Everything in Part A is measured on the two discrete cards with a
standalone harness that lives **outside the repo** (`<SCRATCH>/vkround2/`,
≈600 lines of C + 2 GLSL shaders, source listed in §A0.3 — an implementation agent may move it to
`harnesses/vk-micro/` if it proves useful). Part B is read out of the cloned PowerInfer tree with
`file:line` references.

Bottlenecks this round is priced against (from `docs/phase-4/01-forward-pass.md`):

| label | bottleneck | number today |
|---|---|---|
| **(a)** | prefill = per-token dense chain, ~98 % of the time (chunked prefill being T-batched) | 6.13 tok/s/agent |
| **(b)** | GPU half of decode, 1 509 dispatches, barrier scope under test | **70.8 ms/token** |
| **(c)** | host expert lane, ALU-bound | **38–50 ms/token @ 3–5 GB/s effective** |
| **(d)** | merged MoE only for resident blocks; non-resident experts streamed once per expert on the host lane | 799 MiB/token = 45 GB expert set |

---

# Part A — measured answers

## A0. Method, environment, and the three numbers that matter

### A0.1 Environment

| | |
|---|---|
| driver | `radv` / `Mesa 26.2.4-arch3.1`, ICD `1.4.354`, loader `1.4.363` |
| device resolution | by **PCI bus** via `VK_EXT_pci_bus_info` (`0000:0e` = MI50/Radeon VII, `0000:03` = RX 6950 XT) — never by ordinal |
| clock state | not pinned (short runs, best-of-N, see the note in A0.4) |
| validation | `VK_LAYER_KHRONOS_validation` on the **first** run of each new shader, on the MI50: **0 messages**; then the RX. No GPU fault, no `amdgpu` page fault in `journalctl -b`. |

### A0.2 What each arm actually does

```c
// mode 0 "full"   (today's engine, ctx.rs:953 Gpu::barrier_all)
vkCmdPipelineBarrier(ALL_COMMANDS -> ALL_COMMANDS, MEMORY_READ|MEMORY_WRITE -> same);
// mode 1 "sync2"  (the proposed change)
VkMemoryBarrier2{ COMPUTE_SHADER, SHADER_WRITE -> COMPUTE_SHADER, SHADER_READ|SHADER_WRITE };
vkCmdPipelineBarrier2(...);
// mode 3 "events" vkCmdSetEvent2 after dispatch i, vkCmdWaitEvents2 before dispatch i+1, same scope
// mode 4/5        barrier only, no dispatch (isolates the barrier packet itself)
```

Each arm records **N dispatches (or N barriers) into one command buffer**, brackets it with two
`vkCmdWriteTimestamp2` calls, submits once and fences once; GPU time is the timestamp delta, and
12 independent recordings/submits are taken with the minimum reported (best-of-12 is the right
statistic here: it removes scheduler jitter without hiding a real cost).

### A0.3 Reproduce

```bash
cd <SCRATCH>/vkround2
glslangValidator -V --target-env vulkan1.3 -o touch.spv touch.comp
glslangValidator -V --target-env vulkan1.3 -o lat.spv   lat.comp
gcc -O2 -o vkmicro vkmicro.c -lvulkan

timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./vkmicro barrier 0e 128 1
timeout 90 ./vkmicro barrier 0e 128 4096          # 1 MiB working set per dispatch
timeout 90 ./vkmicro submit  0e 512               # per-submit gap
timeout 90 ./vkmicro lat     03                   # wave32 vs wave64
timeout 180 env SRC_HOST=1 ./vkmicro copy 03 compute  130 1746000
timeout 180 env SRC_HOST=1 RADV_EXPERIMENTAL=transfer_queue ./vkmicro copy 03 transfer 130 1746000
timeout 180 env SRC_HOST=1 RADV_DEBUG=nodma ./vkmicro copy 03 compute 130 1746000
./vkmicro info 0e; ./vkmicro info 03               # feature dump, no GPU work
```

All numbers below are **µs of GPU time**, best-of-10/12 unless stated.

### A0.4 The three results that change the plan

1. **A pipeline barrier packet costs ~0.01 µs — effectively nothing.** 512 sync2 barriers with no
   dispatch between them: 1.12 µs total (2.2 ns each). So "narrow the barrier scope" is *not*
   about the barrier; it is about what the barrier forces: a full **kernel-complete →
   kernel-launch serialization** of ~1.1–2.7 µs (RX) / 2.7–4.4 µs (MI50) per dispatch.
2. **That serialization is flat in grid size** (1 → 1 024 workgroups: 2.69 → 3.09 µs on the MI50).
   It is a launch-latency floor, not a drain proportional to the kernel's work — so it is the same
   46.9 µs-average problem whether a kernel is small or large.
3. **A queue submission boundary costs 11–18 µs** (`vkQueueSubmit2` + driver kick), flat in the
   number of submissions. At ~96 segments/token that is 1.1–1.5 ms/token on its own.

---

## A1. Barrier / queue-submit micro-cost (item 1) — measured, both cards

### A1.1 Tiny dispatches (1 workgroup × 64 threads), n = 128 and n = 512

| arm | MI50 (n=128) µs/op | MI50 (n=512) µs/op | RX (n=128) µs/op | per-100 on RX | note |
|---|---:|---:|---:|---:|---|
| **full** `ALL→ALL` + `MEMORY_READ\|WRITE` (today) | **4.406** | 4.323 | **5.219** | 521.9 | `ctx.rs:953` |
| **sync2** `COMPUTE→COMPUTE` | **2.715** | 2.702 | **1.135** | 113.5 | proposed |
| **none** (disjoint buffers) | 0.050 | 0.048 | **0.017** | 1.7 | the floor |
| **events** (`SetEvent2`/`WaitEvents2`) | 3.611 | 3.666 | 1.464 | 146.4 | **worse** |
| barrier only, sync2 (no dispatch) | 0.009 | 0.002 | 0.011 | 1.1 | packet is free |
| barrier only, full (no dispatch) | 0.026 | 0.007 | 0.020 | 2.0 | packet is free |

**Read it this way.** On the MI50, going full → sync2 pays back **1.69 µs/dispatch**; on the RX the
same change pays back **4.08 µs/dispatch** — the RX's `ALL_COMMANDS` + `MEMORY_READ|WRITE` scope
costs 2.2× more than a compute-scoped dependency, and 300× the no-barrier floor. Every µs of that
is the barrier *forcing a serialized relaunch*, not the packet.

**Event-based waits lose.** `vkCmdSetEvent2`/`vkCmdWaitEvents2` measured 1.46 µs/op on the RX vs
1.135 for a plain `vkCmdPipelineBarrier2` with the same scope (and 3.61 vs 2.72 on the MI50). They
buy nothing on a single queue and cost a CP round trip. **Do not use them for our producer/consumer
chains.**

### A1.2 Grid-size scaling — the serialization is a fixed launch cost

MI50, µs/op (full / sync2 / none):

| grid (workgroups) | touched/op | full | sync2 | none |
|---|---:|---:|---:|---:|
| 1 | 0 | 4.41 | 2.72 | 0.050 |
| 64 | 16 KiB | 4.40 | 2.69 | 0.062 |
| 256 | 64 KiB | 4.59 | 2.81 | 0.176 |
| 1 024 | 256 KiB | 4.82 | 3.09 | 0.604 |
| 4 096 | 1 MiB | **7.14** | **5.15** | 2.314 |

RX, same shape: 1 MiB working set → full **4.458**, sync2 **2.807**, none 1.540, events 2.844.
(The RX's `none` arm is L2/Infinity-Cache bound: 2 MiB moved in 1.54 µs ≈ 1.36 TB/s.)

So the dependency cost stays ~2.7 µs (MI50) / ~1.1–2.8 µs (RX) from grid 1 to grid 4 096, and only
starts to include the kernel's own work at 1 MiB. **A wide kernel is not "cheaper to barrier"; it
just hides the barrier behind its own runtime.** For our decode chain (kernels of 20–40 µs of
measured work), the barrier is hidden — which is exactly why it cannot be the 21–29 ms of
unattributed time doc 01 was hunting.

### A1.3 Submission boundaries

512 tiny dispatches, no barriers, wall clock to fence, split across k submissions:

| k submissions | MI50 µs/dispatch | gap/submit | RX µs/dispatch | gap/submit |
|---:|---:|---:|---:|---:|
| 1 | 0.119 | — | 0.065 | — |
| 8 | 0.330 | 15.5 µs | 0.222 | 11.4 µs |
| 32 | 1.014 | 14.8 µs | 1.172 | 18.3 µs |
| 96 | 2.962 | 15.3 µs | 2.133 | 11.1 µs |

**≈13–15 µs per submit, flat.** 96 segments/token = **1.3–1.5 ms/token**; the 6-slot ring already
allows fewer segments, so this is a bounded, cheap win (halving segments ≈ −0.6…−0.8 ms/token).

### A1.4 Priced against our token (item 1 verdict)

| change | RX µs/dispatch | × 1 509 dispatches | MI50 |
|---|---:|---:|---:|
| full barrier → sync2 scope | **−1.65 … −4.08** | **−2.5 … −6.2 ms/token** | −1.7 → −2.6 ms |
| delete the barrier on a provably disjoint pair | −1.1 … −2.8 | −1.7 … −4.2 ms per 1 509 pairs | −2.7 → −4.1 ms |
| halve the number of segments (96 → 48) | — | −0.6 … −0.8 ms | −0.7 … −0.8 ms |

**The 21–29 ms attribution in doc 01 §Lever 1 is refuted.** Barriers + submits can account for at
most ~4–8 ms of the 70.8 ms GPU half. What remains is either (i) dispatch count itself — the
serialized relaunch floor of ~1.1–2.7 µs × 1 509 that is *not* hidden behind real kernels, or
(ii) something the kernel-time sum does not include (pipeline/descriptor transitions between the
~130 distinct expert pipelines per block are the prime suspect, since we re-bind a *different*
pipeline for every projection of every block). The next step is not a barrier change; it is the
instrumented real-token sweep the implementation round is already running. **The ceiling on lever 1
is 4–8 ms, and it should be taken as a 4–8 ms item, not a 30 % GPU-half item.**

## A2. Transfer queue and region-array copies (item 2) — two corrections

### A2.1 `RADV_EXPERIMENTAL=transfer_queue` — verified, and it is **not** free

`vulkaninfo` with the env var, all three devices:

```
without:  [0] 1 q  0x0f G|C|T|S   [1] 4 q 0x0e C|T|S   RX also [2] video-decode [3] video-encode  [4] 1 q 0x08 S
with:     [0] 1 q  0x0f G|C|T|S   [1] 4 q 0x0e C|T|S   [2] 1 q 0x20 dec [3] 2 q 0x04 T|S  [4] 1 q 0x40 enc [5] 1 q 0x08 S
```

So yes: a **TRANSFER-only family with 2 queues** appears on **both** discrete cards (MI50 family 2,
RX family 3) and on the iGPU (1 queue). `ctx.rs`'s `dedicated_x` branch will fire with no code
change. But measure what it does before switching it on:

| copy (216.5 MiB = 130 × 1.746 MB blobs) | MI50 | RX |
|---|---:|---:|
| VRAM → VRAM, compute-family queue (today) | **423 GB/s** (0.57 ms) | **218 GB/s** (1.04 ms) |
| VRAM → VRAM, TRANSFER-only family | **49 GB/s** (4.63 ms) — **8.6× slower** | not run (same engine) |
| system RAM → VRAM, compute-family queue | **3.0 GB/s** (75.8 ms) | **27.8 GB/s** (8.16 ms) |
| system RAM → VRAM, TRANSFER-only family | **3.3 GB/s** (69.1 ms, **+10 %**) | **28.6–28.7 GB/s** (7.92 ms, **+3 %**) |
| system RAM → VRAM, compute queue + `RADV_DEBUG=nodma` | — | **27.9 GB/s** (no change) |

**Correction 1 to doc 01 §Lever 2.** "Arm A needs no code at all" is true but *dangerous*: the
env var re-routes **every** transfer on the device, and on the MI50 the on-card path collapses
**8.6×** (423 → 49 GB/s). If any staging/VRAM-to-VRAM copy shares that lane, the flag is a
regression. Enable it for the **expert H2D lane only** (per-queue selection, not a process-wide env
var), where the measured gain is +3 % (RX) / +10 % (MI50) — real but small, and it does **not**
change the RX's expert-stream budget: 1.746 MB/expert still costs 61 µs.

**Correction 2.** `RADV_DEBUG=nodma` changes **nothing** on the RX H2D path (27.8 → 27.9 GB/s):
the GFX-ring blit and the DMA engine are indistinguishable at the wire, i.e. the RX expert stream
is PCIe-limited in both branches, and doc 01's "28.7 GB/s is the wire" ceiling is confirmed twice.
On the MI50 there is nothing to win either (3.0 → 3.3 GB/s).

### A2.2 Region arrays: no GPU gain, a **13–18× host-record** gain

`vkCmdCopyBuffer2` with a `VkBufferCopy2[130]` vs 130 separate `vkCmdCopyBuffer` calls, one command
buffer, one submit, GPU time from timestamps and wall time to fence (they agree to 1 %):

| | GPU time | GB/s | host record |
|---|---:|---:|---:|
| MI50, 130 × 1.746 MB, separate | 0.574 ms | 395 | **184 µs** |
| MI50, 130 × 1.746 MB, array2 | 0.569 ms | 399 | **16 µs** |
| RX, 130 × 1.746 MB, separate | 1.040 ms | 218 | **197 µs** |
| RX, 130 × 1.746 MB, array2 | 1.040 ms | 218 | **28 µs** |
| MI50, 508 × 4 KiB, separate | 0.061 ms | 34 | **301 µs** |
| MI50, 508 × 4 KiB, array2 | 0.059 ms | 35 | **84 µs** |

**≈0.6–1.5 µs of host CPU time per `vkCmdCopyBuffer` call, vs ≈0.13–0.22 µs per region with one
array call.** GPU time is identical (the CP is not the bottleneck at these sizes), so the entire
value is on the *host* — which is bottleneck (c).

Priced: the merged-prefill path issues ~130 expert blobs × 48 blocks = **6 240 copies per 128-token
chunk**. At 1.0–1.5 µs each that is **6.2–9.4 ms of host time per chunk**, → **0.9–1.4 ms** with
region arrays. **−5…−8 ms of host record per chunk** for a one-call code change, plus it removes
6 240 CP packets from the dispatch stream that A1 is trying to shorten.

### A2.3 One methodological warning from doing this

Every copy number here is taken **both** ways (GPU timestamps *and* wall-clock-to-fence). They agree
to 1 %. The first iteration of this harness used wall clock only and reported a **423 GB/s**
"system→VRAM" transfer, because a handle-comparison bug silently placed the source buffer in VRAM.
**Any copy-bandwidth number in a gate report must name the memory *type index* it actually
allocated** (`VKDEBUG=1 ./vkmicro copy …` prints it: type 0 = device-local VRAM, type 2 =
system/HOST_CACHED, type 3/4 = ReBAR). Doc 01 §A4's type table is the right reference; the mistake
mode is silent and produces a plausible-but-impossible number.

## A3. Wave32 measurement kit (item 3) — measured, and the answer is "no"

What the capability surface says (both cards, queried):

| | RX 6950 XT | MI50 |
|---|---|---|
| `subgroupSize` / `min…max` | 64 / **32…64** | 64 / **64…64** |
| `requiredSubgroupSizeStages` | **0xf0** (frag, task, mesh, compute) | **0x0** |
| `VK_EXT_subgroup_size_control` + `computeFullSubgroups` | yes | yes (but unusable) |
| `subgroupSupportedOperations` | 0x6ff | 0x6ff |

**Correction to doc 01.** It says the MI50 "must be a no-op — it will refuse the required size".
It does not refuse: the pipeline is **created successfully and the size is silently ignored**
(`requiredSubgroupSizeStages = 0`). That matters for experimental hygiene — an MI50 "wave32" arm
will report bit-identical results *because nothing changed*, which would look like a passing parity
gate. Always confirm the wave size changed: `RADV_DEBUG=shaderstats` shows the VGPR count move
(8 → 16 for my probe kernel), and that is the cheapest possible occupancy kit — **it gives
VGPR/SGPR/LDS/scratch/code size per pipeline with no API code at all**, so
`VK_KHR_pipeline_executable_properties` (which is present on both, and does give the same fields
plus the ACO disassembly) is optional rather than the first step.

**The measurement.** A dependent-load chain (32 serialized `v[idx]` loads per thread, 4 096
workgroups × 64 threads = 262 144 threads, 4 MiB table) — the shape of the GDN delta rule and the
HC read chain:

| | time | effective BW |
|---|---:|---:|
| RX, wave64 | 158.84 µs | 211.3 GB/s |
| RX, wave32 (required size 32) | 158.72 µs | 211.4 GB/s |
| **RX wave32 / wave64** | **1.001×** | |
| MI50, "wave32" (silently ignored) | 231.52 µs | 144.9 GB/s |

`RADV_DEBUG=shaderstats` confirms the two RX pipelines really were compiled differently
(VGPRs 8 vs 16, SGPRs 108 both) — **so this is a real wave32 run, and it buys 0.1 %.**

**Verdict: doc 01 left 3's 5–15 % on the GDN delta rule and the HC read chain is not supported by
the cheapest available probe.** gfx1030's problem on those kernels is not resident-wave count.
Every candidate fix priced off "latency-bound" (wave32, LDS restructure, split-K) should therefore
be re-priced off *measured* VGPR/LDS/occupancy (`RADV_DEBUG=shaderstats` on the real kernels, one
line of env, no code) before spending parity risk. Do the shaderstats dump in the next round; do
**not** schedule a wave32 A/B on the strength of the extension being present.

## A4. Feature surface the first round missed (item 4) — all present, almost all worthless

Queried with `vkGetPhysicalDeviceFeatures2` on both cards (identical answers):

| feature | RX | MI50 | verdict for (a)–(d) |
|---|---|---|---|
| `VK_EXT_robustness2` `nullDescriptor` | 1 | 1 | **small, free (c)**: bind a null SSBO for an absent/streamed expert instead of swapping descriptor sets → fewer host rebinds in the recoder. Take only if the recorder is touched anyway. |
| `VK_EXT_pipeline_robustness` | 1 | 1 | **0 today**: it buys back bounds-check elimination, but our engine does not enable `robustBufferAccess` (`ctx.rs:593` builds a default `PhysicalDeviceFeatures2`), so there is nothing to remove. |
| `VK_EXT_shader_demote_to_helper_invocation` | 1 | 1 | 0 for us: no fragment shaders; `demote` is a no-op in compute. |
| `VK_KHR_shader_terminate_invocation` | 1 | 1 | **~0**: could free a wave early on a padded tile tail. Our tails are ≤ 1 workgroup per dispatch — immeasurable. |
| `VK_KHR_maintenance6` | 1 | 1 | 0: descriptor-buffer / bound-API maintenance. Nothing compute-relevant. |
| `VK_KHR_shader_float_controls` | 1 | 1 | present and **useable**: `denormBehaviorIndependence = ALLOW_ALL`, `shaderDenormPreserveFloat32 = 1`, `shaderDenormFlushToZeroFloat32 = 1`, `shaderSignedZeroInfNanPreserveFloat32 = 1` |
| `VK_KHR_shader_float_controls2` | 1 | 1 | same, per-instruction |
| `shaderExpectAssume` (1.4) | 1 | 1 | 0–2 % on the dequant loops (already in doc 01) — the only remaining "free" shader flag |
| `VK_KHR_zero_initialize_workgroup_memory` | 1 | 1 | µs-scale, already in doc 01 |
| `timestampPeriod` | **10 ns** | **40 ns** | measurement: MI50 cannot resolve sub-1 µs kernels; all MI50 sub-µs numbers above are quantised at 0.04 µs (harmless at n = 128–512). |

**The `shader_float_controls` answer, and it is a "no":** both `DenormPreserve` and
`DenormFlushToZero` are selectable for f32, so we *could* drop a `DenormFlushToZero` execution mode
into the dequant/GEMV inner loops. It would be worth ~0 because GCN/RDNA has **no denormal
penalty** — the hardware does not stall on denormals the way some other ISAs do. There is no
fast-math / no-contraction switch to gain on either card. **Do not spend a parity risk on denormal
modes.**

**Nothing in either extension list (245 RX / 218 MI50) remains that plausibly helps (a)–(d) beyond
what round 1 already ranked.** The one genuine surprise is *negative*: the transfer queue and the
wave32 extension, the two "free" levers round 1 recommended, each measured ~0 (or worse) here.

## A5. Priced summary — every operation this round measured

µs per operation, and what it is worth against a 1 509-dispatch / 96-segment token.

| operation (measured) | MI50 | RX 6950 XT | where it lands | ms/token |
|---|---:|---:|---|---:|
| `vkCmdPipelineBarrier2` COMPUTE→COMPUTE, no dispatch | 0.009 | 0.011 | — | 0 |
| `vkCmdPipelineBarrier` ALL→ALL, no dispatch | 0.026 | 0.020 | — | 0 |
| tiny dispatch, no barrier (CP streams it) | 0.050 | 0.017 | floor | — |
| tiny dispatch behind **sync2** barrier | 2.715 | **1.135** | (b) | — |
| tiny dispatch behind **full** barrier | 4.406 | **5.219** | (b) today | — |
| tiny dispatch behind **events** | 3.611 | 1.464 | (b) — don't | — |
| 1 MiB dispatch behind sync2 / full / none | 5.15 / 7.14 / 2.31 | 2.81 / 4.46 / 1.54 | (a) prefill | — |
| one queue submission boundary | 15.2 | 11–18 | (b) | 1.3–1.5 |
| `vkCmdCopyBuffer` host record, per call | 1.0–1.4 | 1.0–1.5 | (c) | — |
| `vkCmdCopyBuffer2` region, host record | 0.13–0.16 | 0.13–0.22 | (c) | — |
| 1.746 MB expert H2D (per expert) | **583** | **62.7** | (d) | 29.2 (10×48 experts) |
| 216.5 MiB H2D stream | 75.8 ms | 8.16 ms | (d) | — |
| VRAM→VRAM 216.5 MiB | 0.57 ms | 1.04 ms | staging | — |
| wave32 vs wave64, dependent-load chain | n/a (ignored) | **−0.1 %** | (b) | 0 |

**Net margin available from Part A alone:**
`full→sync2 barrier scope` **−2.5…−6.2 ms/token** (RX; −2.6 on the MI50) ·
`region-array copies in merged prefill` **−5…−8 ms of host record per chunk** ·
`fewer submission segments` **−0.6…−0.8 ms/token**.
Everything else measured zero or negative.

---

# Part B — PowerInfer audit

## B1. What was cloned

```
Reference Projects/powerinfer   shallow --depth 1, HEAD 8bd56d6 (2026-05-11), MIT
  llama.cpp       10 956 lines  the hybrid hot/cold engine (their ggml fork, single file)
  ggml.c          20 570 lines  CPU-side sparse ops (ggml_mul_mat_idx, ggml_axpy)
  ggml-cuda.cu     9 636 lines  neuron-aware sparse CUDA/HIP kernels
  powerinfer-py/               the *placement solver* (solver.py, export_split.py)
  smallthinker/                their 2026 MoE engine (llama.cpp fork + fused sparse MoE)
```

## B2. The mechanism, exactly

### B2.1 Hot/cold is a **per-neuron bitmask + a compaction table**, not a per-tensor or per-block flag

`llama.cpp:1374-1378` puts two tensors on every layer:

```c
struct ggml_tensor * gpu_idx;     // 0/1 per FFN neuron — the hot mask
double               gpu_offload_ratio;              // [0,1] per layer
struct ggml_tensor * gpu_bucket;  // double index from GPU split neuron to original neuron
```

`export_split.py:29-46` builds both: `torch.topk(activation, select_count)` → `gpu_idx` (the mask)
and `gpu_bucket` (the **sorted** list of chosen indices = the compaction/indirection table). At
load time `llama.cpp:2898-2963 create_striped_mat_to_gpu` **physically copies only the selected
rows** into a compacted GPU tensor (`ffn_gate_gpu/ffn_up_gpu/ffn_down_gpu`), and
`llama.cpp:2850-2858` special-cases the two degenerate ratios (0 or 1 = no hybrid at all).

Two consequences worth naming:

* The GPU tensor is **not** a slice of the model and **not** a whole block; it is a striped,
  compacted, per-layer subset, addressed through `gpu_bucket`. The CPU keeps the full tensor and
  skips the rows the GPU owns (`gpu_index` is passed to the CPU op).
* The unit of placement is the **neuron** (one row of `ffn_up`/`ffn_gate`/`ffn_down`), i.e. it is
  already "per-expert-within-a-block" granularity in our terms.

### B2.2 The kernel reads the residency map *inside* the row loop

`ggml-cuda.cu:4685-4700` (`dequantize_mul_mat_vec_sparse`):

```c
int gpu_row = blockIdx.y*blockDim.y + threadIdx.y;
if (gpu_row >= nrows) return;
int row = lst ? lst[gpu_row] : gpu_row;          // bucket: compacted row -> global neuron
if (idx[row] < dev_sparse_threshold) return;      // per-row skip, warp-uniform
... dequantize + dot over the row ...
```

That is the whole "sparse kernel" trick: the launch is over the **resident** row count, the global
row index comes from a **table**, and the *per-token* selection is a **threshold test on the
predictor's output** that costs one branch. `dev_sparse_threshold` is a `__constant__` pushed once
from the hparams (`ggml-cuda.cu:9635`), so it costs no per-dispatch upload. Batched variants
(`…_batch_sparse`, `ggml-cuda.cu:4607`, `:4758`) do the same with a `[n_tokens, n_expert_used]`
index.

### B2.3 The predictor is a 2-layer MLP, and it is *not* the router

`llama.cpp:1372-1373` loads `mlp_pre_w1` (`n_embd × hidden`) and `mlp_pre_w2` (`hidden × n_ff`) per
layer, and `llama.cpp:4678-4690` builds the graph:

```c
idx = ggml_mul_mat(ctx, pre_w1, pred_inpl);   // hidden  = W1 · x
idx = ggml_relu(ctx, idx);
idx = ggml_mul_mat(ctx, pre_w2, idx);         // scores  = W2 · hidden   -> one score per neuron
```

then everything downstream uses `idx` as the sparse index list. **It runs once per layer and
replaces the dense FFN**, so its own cost is two small GEMVs (n_embd×hidden + hidden×n_ff) against a
dense FFN of n_embd×n_ff×3 — cheap by construction.

**The training code is not released** (`README.md:298`: `- [ ] Release predictor training code`).
The published method (PowerInfer, SOSP'23) is a per-layer MLP trained offline on a calibration set
to regress the *magnitude* of each neuron's activation, with the threshold selecting the top-k; the
weights ship separately on HuggingFace (`README.md:183-186`: `<model>-Predictor` repos) and are
merged into the GGUF by `convert-hf-to-powerinfer-gguf.py`. For us: no code to port, but the design
is trivially reproducible (we already have a router whose output is the ground truth).

### B2.4 Placement is an **ILP**, not a heuristic

`powerinfer-py/powerinfer/solver.py:1-90` — one integer program per model:

* variables: `x[layer, neuron] ∈ {0,1}` plus one slack per layer `x[layer]` (`solve_gpu_split`);
* objective `c`: maximize total activation weight covered by the resident set;
* constraint 1: total bytes ≤ VRAM capacity (per-layer neuron sizes × `x`);
* constraint 2/3: a *min/max residency per layer* band, so no layer is starved (the
  `int(threshold/batch)` and `-1000000` rows at `solver.py:55-66`);
* solved with `ilp` + 30 s timeout, then `aligned_lst[i] = sum(ans[...]) * batch` is the per-layer
  neuron count that gets exported.

Real invocation, from their own log (`docs/token_generation_performance_tips.md:37`):

```
solve_gpu_split(neuron=13824, layer=40, batch=256, threshold=0,
                vram_capacity=13 357 166 592)          # 12.4 GiB
→ llm_load_gpu_split: offloaded 12577.50 MiB of FFN weights to GPU   # of 16825.94 MB total
```

**~75 % of the FFN resident in 12.4 GiB**, chosen per neuron, with a *global* byte budget and a
*per-layer* fairness band. That is the piece we do not have: we choose residency at **block**
granularity, by hand, in `-ot`.

### B2.5 The overlap: two independent graph nodes, joined by one `add`

`llama.cpp:4545-4590 llm_build_sparse_mul_mat` and `:4593-4645 llm_build_sparse_axpy` build, for
each of `up`, `gate`, `down`:

```c
out     = ggml_mul_mat_idx(up,      inp, idx, gpu_index);            // CPU, skips GPU rows
if (up_gpu) {
  out_gpu = ggml_mul_mat_idx_upscale(up_gpu, inp, idx, gpu_bucket);  // GPU, resident rows only
  out     = ggml_add(ctx, out, out_gpu);                             // single join
}
```

`out` and `out_gpu` are **siblings with a common parent** — nothing in the graph orders the CPU node
after the GPU node. The GPU node is enqueued (async, `ggml_cuda_assign_buffers_no_alloc` defers the
synchronisation to the first host read), and the CPU node runs on the host thread pool while the GPU
executes. The only synchronisation point in the layer is the `ggml_add` merge. Their own HIP build
even carries a comment about model-size edge cases in this path (`llama.cpp:4559-4565`), which is
evidence of how narrow the coupling is: one pointer.

Also note `full_gpu` (`gpu_offload_ratio >= 1.0`) is a straight fast path with `lst = NULL` — the
degenerate case stays branch-free.

### B2.6 The 2026 successor (SmallThinker) applies the same machinery to **MoE**

`smallthinker/src/llama-graph.cpp:960 build_moe_sparse_ffn_smallthinker` → `ggml_fused_sparse_moe`
(`ggml/src/ggml.c:5256`): router weights → `top-k` → **one fused op** for the whole MoE. Their
README states the two mechanisms explicitly: *"leveraging the sparsity induced by the ReLU
activation to skip certain computations during the UP/DOWN calculation of each expert based on the
GATE output, as well as using a predictor to perform sparse computation"*, and *"leveraging the
router's pre-selection mechanism"*. Reported outcome: 21B-A3B in **8 GB** at 20.3 tok/s on a
desktop CPU, 21B at 11.47 GiB / 30.2 tok/s unconstrained.

## B3. What does and does not map onto us

### B3.1 The ReLU assumption does not transfer — say it plainly

PowerInfer's whole sparse story is built on **exact zeros from ReLU**. Our Qwen3.8-Flash-Next experts
are **SwiGLU** (`silu(gate)·up`), which has *no exact zeros*: `silu` is never 0 except at exactly 0.
So §B2.2's per-neuron threshold skip, and SmallThinker's sparse up/down, **do not port**. Do not
spend a round on "expert-internal sparsity" — there is none to find.

### B3.2 Everything about *placement* and *overlap* does transfer

Our sparsity source is different and stronger: **discrete top-k routing** (10 of 512 experts per
block). Substitute "expert" for "neuron" and the three transferable items map cleanly:

| PowerInfer | ours |
|---|---|
| `gpu_idx` per-neuron hot mask (B2.1) | per-**(block, expert)** resident mask, instead of whole-block `-ot` |
| `gpu_bucket` compaction (B2.1) | our `moe_map`/slot map already *is* this — one more level: an `expert_slot[]` table for the resident arena |
| `lst[gpu_row]` + threshold test (B2.2) | a single `resident[expert]` test at the top of the merged-MoE kernel, launched over all experts |
| ILP over VRAM budget (B2.4) | ILP over `Σ_blocks Σ_experts freq(b,e)·bytes ≤ VRAM`, with a per-block floor |
| CPU node ∥ GPU node, `ggml_add` join (B2.5) | non-resident-expert host lane ∥ resident-expert GPU batch, joined at the fan-in |

### B3.3 The arithmetic that makes it worth doing

From `docs/phase-4/01-forward-pass.md:1597-1621`:

* one expert blob (gate+up+down, IQ3_XXS) = **1.746 MB**
* top-10 × 48 blocks = **799 MiB streamed per token** if nothing is resident
* the full expert set is 25 769 draws × 1.746 MB = **41.9 GiB** — bigger than both cards' combined VRAM

At the RX's measured 28.7 GB/s (A2.1), 799 MiB/token is **29.2 ms/token of pure PCIe**, which is
most of bottleneck (c)'s 38–50 ms. And *nothing* in the expert set fits: 41.9 GiB vs ~28 GB of
usable VRAM means **per-expert residency is the only way to move that number**, because whole-block
residency can never reach it.

Two independent levers act on the same 29 ms:

* **residency** removes the bytes (PowerInfer's ILP, ~67 % of the set fits at 28/41.9 GiB),
* **overlap** hides the bytes that remain behind the GPU half (PowerInfer's node structure).

## B4. Ranked adaptations

Impact is stated against (a)–(d) from the round brief; every item names the bench that proves it.

| # | adaptation | source | (a) | (b) | (c) | (d) | effort | bench |
|---|---|---|---|---|---|---|---|---|
| **1** | **Host lane ∥ GPU resident batch, joined at the fan-in.** Start the non-resident-expert host work the moment routing lands; do not wait for the GPU. Single join at the slot-map fan-in (already one-writer-per-slot, so no atomics). | `llama.cpp:4545-4645` (`ggml_mul_mat_idx` ‖ `…_upscale` → `ggml_add`) | 0 | 0 | **removes up to 38–50 ms/token from the critical path** (bound: `max(70.8, 50)` instead of `70.8+50`) | −0 (structure) | M | `bench token` with a host-lane/GPU-busy timeline: measure the *offset* between lane start and GPU idle, then wall ms/token |
| **2** | **Per-(block,expert) residency from a routing histogram + ILP budget.** Replace whole-block `-ot` with a resident mask + `expert_slot[]` table; solve `Σ freq·bytes ≤ VRAM` with a per-block floor. | `solver.py:1-90`, `export_split.py:29-46`, `create_striped_mat_to_gpu` | 0 | 0 | **−direct**: 799 MiB/token → ~264 MiB at a 67 % resident fraction ⇒ **29.2 → ~9.6 ms** of PCIe | **this is (d)** | M | `bench agents --profile-routes` histogram → `--resident-per-expert` plan → `bench token` lane ms/step (target 38–50 → < 15) |
| **3** | **Predictor as a one-block-ahead prefetch oracle.** A 2-layer MLP (`n_embd→hidden→512`) on block N's output block, predicting block N+1's top-10; use it to prefetch the non-resident blobs one block (~1.5 ms) early, which is enough for 10 × 61 µs = 610 µs of transfer. | `mlp_pre_w1/w2` (`llama.cpp:1372`, `:4678`) | 0 | 0 | hides the *residual* stream after #2 — up to **9.6 ms fully hidden** | complements #2 | M–L (training harness) | offline: routing-recall@10 of the predictor vs the real router on the eval set; online: `bench token` PCIe-wait histogram (`--profile-xfer`) → target ~0 exposed µs |
| 4 | **Residency-aware row skip inside the merged-MoE kernel** (`resident[expert]` test at kernel top, launch over all experts, threshold pushed once as a push constant). | `ggml-cuda.cu:4685-4700`, `:9635` | 0 | removes per-expert host dispatch decisions from the recorder | feeds (d) | **lets one dispatch cover resident + non-resident** at zero traffic | S | `bench token` dispatch counter (per-block: ~130 → ~3) + parity 24/24 |
| 5 | **Fused single-op MoE** (`ggml_fused_sparse_moe`): routing + all experts + fan-in in one op, `selected_experts` from the router. | `smallthinker/src/llama-graph.cpp:1001`, `ggml/src/ggml.c:5256` | 0 | −per-expert dispatch entirely | 0 | converges with 4 | M | same as 4; dispatch counter is the metric |
| 6 | **Batch-size-adaptive sparse threshold** (`LLAMA_SPARSE_PRED_THRESHOLD`, `llama.cpp:1269`): the threshold is a runtime knob, so the hot set can shrink at high batch where sparse gather stops paying. | `llama.cpp:1268-1269` | 0 | — | 0 | small | S | `bench agents --streams 4` at 2 thresholds |

**Do not copy from PowerInfer:**

* its **ReLU/exact-zero sparsity** (§B3.1) — our experts are SwiGLU, there is nothing to skip;
* its **predictor as a replacement for the FFN** — for us the router is already a 512-way
  classifier costing one small GEMV; a predictor only earns its keep as a *prefetch oracle* (#3);
* its **`BLIS` dense-CPU-GEMM prefill path** — `docs/BLIS.md` documents building BLIS against
  llama.cpp, but no BLIS call sites exist in this HEAD's `llama.cpp`; there is no code to read, so
  nothing is claimed here;
* its **HIP "hack"** at `llama.cpp:4559-4565` (`up_gpu->data = up_gpu`) — a workaround for a
  >40 B-model ROCm bug, in exactly the hybrid path we would be adapting.

---

# Part C — top 3 next experiments, each with the bench that proves it

Ordered cheapest-falsification-first, and chosen to be independent (different files, different
subsystems) so they fit one grouped round.

### 1. Barrier scope: `sync2` + delete barriers between provably disjoint dispatches

Change `dispatch_batch` (`crates/breccia-kernels`) from `Gpu::barrier_all` (`ctx.rs:953`) to
`vkCmdPipelineBarrier2` `COMPUTE_SHADER → COMPUTE_SHADER` / `SHADER_WRITE → SHADER_READ|SHADER_WRITE`,
and drop the barrier on dispatch pairs that touch disjoint buffers (bit-identical by construction).

* **Proves it:** `examples/dispatch_overhead` batched mode, three arms (full / sync2 / none) — the
  harness in §A0 already produces these numbers, port it as the reference; then
  `bench token --streams 2` with parity 24/24 unchanged.
* **Number to move:** **−2.5…−6.2 ms/token** (measured A1.4), not the 3–12 ms doc 01 hoped for.
* **Do not** bundle the transfer-queue or wave32 experiments with it: both measured ~0/worse (§A2,
  §A3) and would muddy the attribution.

### 2. Host lane ∥ GPU resident batch (adaptation #1) — the largest single item available

Restructure the decode step so the non-resident-expert host lane is *started* as soon as routing is
known and the GPU resident-expert batch runs concurrently, with the single join at the fan-in.

* **Proves it:** `bench token` with a two-lane timeline (`--profile-lanes`: host lane start/end vs
  GPU busy intervals). The metric that matters is **exposed lane time** (lane time not overlapped),
  then wall ms/token. If exposed lane time does not fall, the dependency analysis is wrong and the
  GPU half is the serialising side — that is a useful answer in one window too.
* **Number to move:** wall ms/token; the bound is `max(GPU 70.8, lane 38–50)` vs today's sum.
* Cheap falsification: a *sleep-injection* run (add a known 5 ms stall to the lane and to the GPU
  half in turn) tells you which half the token is actually waiting on, in two short runs.

### 3. Region-array copies + routing-histogram residency plan (adaptations #2 and §A2.2)

`vkCmdCopyBuffer2` with a region array everywhere we currently issue one `vkCmdCopyBuffer` per blob,
and a `bench agents --profile-routes` histogram feeding an ILP residency plan (per-block expert
floor, like `solver.py`'s per-layer band).

* **Proves it:** `bench agents --streams 4 --prefix 300 --prefill-chunk 128` for the copy half
  (**−5…−8 ms host record per chunk**); `bench token` lane ms/step for the residency half
  (target 38–50 → < 15 ms with 67 % of the expert set resident).
* **Number to move:** merged-prefill prompt seconds, and the decode lane.
* Note: `-ot` currently selects blocks; the plan needs the *same* slot map with a new mask level, so
  it lands in the existing `MoeMap` rather than a parallel mechanism.

**Deferred, with the measurement that defers it:** wave32 (0.1 % on the probe, §A3),
event-based sync (1.3–2.5× worse than a plain barrier2, §A1.1), the dedicated transfer queue as a
process-wide env var (8.6× slower on-card copies on the MI50, +3 %/+10 % only on the H2D lane,
§A2.1), `VK_EXT_pipeline_robustness` (nothing to remove — `robustBufferAccess` is off), denormal
modes (no AMD penalty), `nullDescriptor` (only worth it if the recorder is touched anyway).
