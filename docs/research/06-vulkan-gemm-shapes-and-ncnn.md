# 06 — small-M quantised GEMM shapes, fused SwiGLU, barrier bookkeeping; audit of ncnn

**Round:** paired research 6 (2026-10-26/27). **Read-only:** no engine crate was touched
(`git status` at the end of the round shows only `docs/research/*` from this agent).

Measurement program: a throwaway standalone harness (own `VkDevice`, `ash`, no engine crate) in
opencode's scratch space, per AGENTS.md — it compiled the engine's own
`crates/breccia-kernels/shaders/blocks.glsl` into every shader, so the kernel it measured *is* the
Q4_K decode path we ship. **The scratch source was not retained** (the concurrent implementation
round reaped the scratch tree mid-round), so §0.1 and Appendix A carry everything needed to
rebuild it in one sitting; the numbers below are the record. Every table in this doc is inline, so
nothing depends on a surviving file.

> The paired implementation round owns the shape/barrier harness that *is* kept
> (`vk-gemm-shapes`, with `sweep` and `barrier` subcommands and PCI-targeted device selection);
> this round's numbers came from an independent program so that neither agent's edits could move
> the other's baseline.

## 0. What was measured, and how (read this before quoting a number)

| | |
|---|---|
| Shapes | shared expert **gate/up = 640 rows × 2560 K**, **down = 2560 rows × 640 K** (`bpr` 10 and 3, so `down` exercises the partial-tail block) |
| Quant | **Q4_K** (QK 256, 144 B/block, 4.5 bit/weight) — the shared expert's gate/up type |
| Users | m = 1 and m = 8 (the decode chunk), users ride in `pc.splits` exactly as `gemm_body.glsl` does |
| Cards | RX 6950 XT (`0000:03:00.0`, headless) full sweep; MI50 (`0000:0e:00.0`, **display**) 4 shapes per process, per the ≤4-fresh-shapes rule |
| Timing | queue timestamps around 64 iterations inside one command buffer, **each pipeline pre-warmed with one un-timed dispatch**, min of 2–3 process repeats |
| Reproducibility | median spread across 3 repeats **1.5 %**, worst fully-occupied shape 3 % |
| Correctness | all geometries **bit-identical** (`max \|Δ\| = 0.0`) vs. an independent CPU Q4_K dequant + f64 dot: **max rel 2.7 × 10⁻⁶** over 2 560 terms |

Three measurement traps this round hit, all worth knowing:

1. **Cold clocks cost 5×.** Without the pre-warm the *first* shape in a process measured
   gate/up m=1 at 12.4 µs; warm and at steady state the same dispatch is **2.39 µs**. Under-filled
   shapes (80 workgroups) cannot ramp the card on their own, so a single un-warmed dispatch is
   indistinguishable from a slow kernel. Every number below is pre-warmed + 64 iterations.
2. **Back-to-back identical dispatches overlap.** 64 identical dispatches with *no* barrier
   measure 137 ns of added cost each (they pipeline); the same dispatches in a **dependent** chain
   with one `COMPUTE→COMPUTE` barrier cost **1 074–1 118 ns** each (§3). A throughput number taken
   without a dependency understates a real graph leg by ~4×.
3. **Sub-occupied workgroups duplicate work.** With one 32-lane group per row, `rows_per_wg <
   local_size_x/32` makes idle lane groups process rows the *next* workgroup also owns: 2× and 4×
   penalties measured (`down` m=1: **3 440 ns at `ls256/r8` vs 13 137 ns at `ls256/r2` and
   26 181 ns at `ls256/r1`**). Only `rows_per_wg == local_size_x/32` variants are fair comparisons.

## 1. Part A1 — workgroup/occupancy sweep at N=640, K=2560 (and the `down` shape)

All fully-occupied variants (one 32-lane group per row). `ns/disp` is one dispatch; `GB/s` is
weight bytes / time (constant across m — that is the point); `GFLOP/s` counts 2·m·N·K.

**RX 6950 XT, Q4_K**

| shape | m | ls/rows/WG | ns | GFLOP/s | GB/s |
|---|---:|---|---:|---:|---:|
| gate/up 640×2560 | 1 | ls64/r2 | 2 524 | 1 298 | 365 |
| | 1 | ls128/r4 | 2 539 | 1 290 | 363 |
| | 1 | **ls256/r8** (shipping shape) | 2 389 | 1 372 | 386 |
| | 1 | ls512/r16 | 2 391 | 1 371 | 386 |
| | 1 | ls1024/r32 | 2 458 | 1 333 | 375 |
| | 1 | **ls256/r8 + unroll2** | **2 138** | **1 533** | **431** |
| | 8 | ls128/r4 | 5 497 | 4 769 | 168 |
| | 8 | **ls256/r8** | 5 228 | 5 014 | 176 |
| | 8 | **ls256/r8 + unroll2** | **4 834** | **5 423** | **191** |
| | 8 | ls256/r8 + unroll4 | 5 199 | 5 042 | 177 |
| down 2560×640 | 1 | ls256/r8 | 3 440 | 953 | 321 |
| | 1 | ls512/r16 | 3 319 | 987 | 333 |
| | 1 | **ls256/r8 + unroll2** | **3 146** | **1 041** | **352** |
| | 8 | **ls256/r8** | **9 143** | **2 867** | **121** |
| | 8 | ls512/r16 | 9 276 | 2 826 | 119 |
| | 8 | ls256/r8 + unroll2 | 9 357 | 2 802 | 118 |
| | 8 | ls256/r8 + unroll4 | 10 981 | 2 387 | 101 |

**MI50 (display card, 4 shapes per process)**

| shape | m | ls/rows/WG | ns | GFLOP/s | GB/s |
|---|---:|---|---:|---:|---:|
| gate/up | 1 | ls64/r2 … ls512/r16 | 7 150–7 540 | 437–458 | 122–129 |
| gate/up | 8 | ls64/r2 … ls512/r16 | 20 940–21 105 | 1 242–1 252 | 43.7–44.0 |
| down | 1 | ls64/r2 … ls256/r8 | 11 060–11 285 | 290–296 | 98–100 |
| down | 8 | ls64/r2 … ls256/r8 | 44 980–45 205 | 580–583 | 24.5–24.6 |

**Reading**

* **Workgroup *size* is not a lever.** 64 → 512 threads (2 → 16 rows/WG) is flat within **5 %** on
  both cards for every shape and both m. The current `local_size_x = 256 / ROWS_PER_WG = 8` shape is
  at the top of that plateau; `ls512/r16` is 0–3 % better on `down` and 0–1 % worse on gate/up. There
  is no geometry win to collect here, so the chunk-level shared-expert pass should **not** be
  re-tuned around launch shape.
* **`down` (K=640, 3 blocks/row, 320 workgroups) reaches 321–352 GB/s at m=1** — that is the
  *healthy* number for this kernel family. **gate/up (K=2560, 10 blocks/row, 80 workgroups) only
  reaches 365–431 GB/s** with the same bytes-per-weight, and the gap is not bandwidth: it is the
  10-deep dependent block chain per row with 80 workgroups to hide it.
* **Unroll 2 blocks in flight is the only cheap win: −9 % gate/up m=1, −8 % gate/up m=8, −9 %
  down m=1** (neutral-to-−3 % on down m=8 where `bpr = 3`). Unroll 4 is worse than 2 everywhere
  (register pressure). A `bpr`-specialised pipeline (§4.2) reaches the same effect more strongly,
  because it lets the compiler unroll the *whole* loop rather than 2 iterations of it.
* **m = 8 buys 3.7× on gate/up and 3.0× on `down`**: 5 228 ns (or 4 834 with unroll-2) for eight users, against
  8 × 2 389 ns for eight separate m=1 calls. This is the quantitative case for the chunked shared-expert pass
  that is already being built: at m=8 the shared expert's three GEMMs cost **gate 4.83 + up 4.83 + down 9.14 ≈
  18.8 µs per block**, i.e. **0.90 ms/token** over 48 blocks — against **~5.8 ms/token** in the T=1 trace, where
  the same work is issued as dependent m=1 dispatches.
* **The MI50 is 3.0–5.2× slower than the RX on exactly these shapes** (gate/up m=8 44 GB/s vs 176,
  down m=8 24.6 vs 121). Its per-dispatch latency boundary (round 4: 4.4× the RX) shows up directly
  in throughput here; fine-grained shared-expert work belongs on the RX, and the MI50 should get
  resident bandwidth-bound blocks only.
* **Forced subgroup size does not pay.** The RX reports `subgroupSize = 64` and its default already
  measures identical to `requiredSubgroupSize = 64`; forcing **32 is 5–8 % slower** on every shape
  (e.g. `down` m=8 `ls256/r8`: 9 143 → 10 716 ns). `VK_EXT_subgroup_size_control` is therefore only
  useful to *pin* 64 against a driver change, not to explore.

## 2. Part A2 — fused SwiGLU (+ down): what fits, what does not

**LDS budget, measured:** `maxComputeSharedMemorySize = 64 KiB` on the RX (the harness prints it
per device; the MI50 matches, per round 4). The shared expert's intermediate is `n_ff = 640` f32 =
**2.5 KiB per user**, so m=8 needs **20 KiB** — it fits, and that is why the experiment below was
run with a 20 KiB (`LDS_VECS = 1280`) staging window.

**2.1 Staging the activation rows in LDS is a loss — everywhere.** `down` reads `m × 640` f32
(20 KiB at m=8) once per workgroup instead of from L2/Infinity Cache:

| variant | down m=1 | down m=8 | gate/up m=1 | gate/up m=8 |
|---|---:|---:|---:|---:|
| global X (`ls256/r8`, shipping) | 3 440 ns | 9 143 ns | 2 389 ns | 5 228 ns |
| LDS-staged X (20 KiB) | 3 441 ns | 9 739 ns | 2 657 ns | n/a (> window) |
| LDS-staged X (`ls128/r4`) | 4 930 ns | 13 530 ns | 3 765 ns | n/a |
| LDS-staged X (`ls64/r2`) | 8 957 ns | 25 444 ns | 6 732 ns | n/a |

Staging costs a cooperative copy + `barrier()` per workgroup and buys nothing at m=8 (+6 %) and
loses badly at small workgroups (2.6× at `ls64/r2`) because the copy is not amortised over enough
rows. **Do not spend LDS on X.** That also means `VK_KHR_workgroup_memory_explicit_layout` has
nothing to alias here.

**2.2 gate + up + SwiGLU in one dispatch: measured, and it is the round's biggest single win.**
`shaders/gateup_swiglu.comp` gives one workgroup row `j` of *both* matrices, decodes both block
streams in the same lane, and writes `silu(gate_j·x) * (up_j·x)` — 3 dependent dispatches → 1.

RX, Q4_K, identical buffers, pre-warmed, 64 iterations, two repeats (`d`/`e`, they agree to 1 %):

| m | gate alone | gate+up (2 disp.) | gate+up+SwiGLU (3 disp.) | fused (1 disp.) | fused vs 3 disp. |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 787 ns | 10 728 ns | 15 633 ns | **4 690–5 018 ns** | **3.1–3.3×** |
| 8 | 5 611 ns | 26 816 ns | 50 738 ns | **9 291–9 531 ns** | **5.3–5.5×** |

The mechanism is trap 2: a *dependent* 640×2560 m=1 GEMM costs 5.35 µs as the second dispatch in a
chain but 2.79 µs when dispatched back-to-back, and the fused kernel issues both matrices' block
loads in the same warp, so it pays the dependency latency once for twice the bytes. **Gate+up+SwiGLU
is worth ~3× at m=1 and ~5× at m=8 on the dependent path** — far more than the "one dispatch saved
plus a 20 KB round-trip" estimate that motivated it. `ls256/r4` is exactly 2× the `ls256/r8` number,
which is the duplication penalty of §0.3, not a real shape effect.

**2.3 SwiGLU + *down* in one dispatch is not feasible, and the reason is structural.** The `down`
projection needs the whole 640-wide intermediate for each output row, and the intermediate can only
be *produced* by the same workgroup if that workgroup owns all 640 gate rows for all m users —
1.84 MB of weights streamed through **one** CU (~1/60 of the card's bandwidth), i.e. ~60× slower.
The grid cannot be reduced below ~60 workgroups without losing the card, and Vulkan compute has no
device-wide barrier (no grid sync; it is not a feature of any extension present on these cards).
Therefore:

* **Feasible, measured:** gate + up + SwiGLU → one dispatch (§2.2).
* **Feasible, cheap, unmeasured:** fuse the scalar shared-expert gate multiply and/or the residual
  add into the `down` epilogue (one dependent dispatch removed per block, same shape as the
  existing `ops_swiglu` fusion).
* **Not feasible:** gate/up/SwiGLU/down in one dispatch. The "chunk-level shared-expert pass" should
  therefore mean **2 dispatches per block** (fused gate+up+silu, then down), not 1.
* Headroom check: LDS could hold the intermediate (2.5 KiB × m) *if* a future pass ever wants a
  K-chunked `down` with the intermediate pinned in shared memory — 20 KiB at m=8 fits with room for
  the double-buffered weight tile the decode would need.

## 3. Part A3 — `vkCmdPipelineBarrier2` bookkeeping (unblocks the parked item)

Dependent chain of 64 dispatches over a 256 KiB buffer bound as both source and destination, so
dispatch *k+1* really reads what *k* wrote. RX, 4 repeats of 64-dispatch chains:

| barrier form | ns/dispatch | Δ vs none |
|---|---:|---:|
| no barrier (race; timing only) | 137 | — |
| one barrier every 4th dispatch (race) | 129 | ≈ 0 |
| **v2 `COMPUTE→COMPUTE`, `SHADER_WRITE→SHADER_READ`** | **1 116** | **+979** |
| v2 `COMPUTE→COMPUTE`, `READ\|WRITE→READ\|WRITE` | 1 118 | +981 |
| v2 `COMPUTE→COMPUTE` + `BY_REGION` | 1 074 | +937 |
| v1 `ALL_COMMANDS→ALL_COMMANDS`, memory R\|W | 1 749 | +1 612 |

**Smallest correct granularity, measured:** `vkCmdPipelineBarrier2` with
`srcStage = dstStage = COMPUTE_SHADER` and an *exact* `srcAccessMask`/`dstAccessMask` pair is
**1.6× cheaper** than the v1 `ALL_COMMANDS` form we emit per dispatch, and the minimal
`write→read` mask is identical to the defensive `read|write` mask (no reason to be defensive).
`BY_REGION` is a further 4 % — worth taking since we always know the written range, but it is noise
next to the 8× gap between "one barrier" and "no barrier".

So the parked barrier-pruning item is worth, at most, **≈0.94 µs per barrier removed** on the RX
(1509 → ~400 barriers ⇒ **−1.0 ms/token**, consistent with round 4's −0.7…−3.2 ms estimate), and it
does not depend on knowing the call site *if* the dependency is tracked per resource instead:

**ncnn's pattern is the answer to the "call-site ≠ dispatch" blocker.** `VkCompute::barrier_readwrite`
(`src/command.cpp:2214-2252`) keeps `stage_flags` + `access_flags` **on the buffer's memory
metadata** and emits a barrier only when the previous access was a write (or the stage differs),
then updates the metadata. One `if` decides correctness for any caller, because the *resource*
remembers what happened to it. Applied to our graph this is a mechanical change in
`dispatch_batch::record`: for each bound range, if `last_access & SHADER_WRITE`, emit a
`COMPUTE→COMPUTE` barrier with `srcAccess = last_access` and set `last_access = READ|WRITE`;
otherwise emit nothing and OR in `READ`. It subsumes both the parked item and the "disjoint resource
sets need no barrier" case, and it is bit-identical by construction (it is exactly the dependency
the spec requires).

RADV gotchas confirmed this round: barriers are free as *packets* but their **serialisation is the
cost** (no-barrier chains overlap, barrier chains do not); batching dispatches without barriers is
only legal when the tracked flags say so (they overlap, which is why the race rows above are 8×
cheaper); and nothing about `vkCmdPipelineBarrier2` changes inside secondary command buffers
(round 4's replay result stands: descriptor-set swap 19.6 ns/set, secondary-CB replay ≈ −0.3
ms/token).

## 4. Part A4 — remaining Vulkan surface that touches small-M GEMM

1. **`VK_EXT_subgroup_size_control` — settled, not a lever.** Default = 64 on both cards; pinning 64
   is a no-op, pinning 32 costs 5–8 % (§1). Keep it enabled only as a guard.
2. **Shape-specialised pipelines (`VkSpecializationInfo`) — the untried one that fits us best.**
   Freezing `bpr`, `cols`, `users` (and `base`) as specialisation constants lets the driver fully
   unroll the block loop and fold every address. ncnn does exactly this for shapes *and* tile
   geometry (`src/layer/vulkan/convolution_vulkan.cpp:499-548`, 15–18 specialisation words per
   pipeline) and keys the pipeline cache on
   `(spv, specializations, local_size_xyz, subgroup_size)` (`src/pipelinecache.h:39-56`). Our
   shaders already take these values as push constants, so the change is: one pipeline per
   `(quant, bpr, cols, users, local_size)` and push constants only for the *data* offset. Expected
   effect ≥ the measured unroll-2 win (−9 %) because the compiler gets the real trip count; needs
   its own measurement, and it multiplies pipeline count (see §4.5).
3. **`VK_KHR_pipeline_binary` (present on both cards) — removes the cold-start compile that made
   §0.1 a 5× measurement trap**, and reduces the number of *fresh shapes* a test process must
   compile (the 2026-10-26 reset correlated with 24 fresh dispatch shapes in one process). Cache
   compiled binaries on disk keyed like §4.2 and a sweep no longer compiles anything.
4. **`VK_EXT_descriptor_buffer` + `VK_KHR_maintenance5` (both present) — descriptor updates without
   pools.** Round 4 measured `vkAllocateDescriptorSets` per dispatch as the dominant record cost
   (212 ns vs 68 ns/dispatch); a descriptor buffer turns the write into a mapped memcpy and removes
   the pool entirely. Complementary to, not a replacement for, tracked barriers.
5. **`VK_KHR_shader_subgroup_uniform_control_flow` (present).** Our block loop *is* uniform; the only
   divergence is the tail mask. Declaring it lets the compiler assume reconvergence-free subgroup
   ops. Cheap to try (one `#extension` + `[[subgroup_uniform_control_flow]]` on the loop), expected
   small; do not spend a round on it alone.
6. **`VK_KHR_cooperative_matrix` is *not* present** on either card under Mesa 26.2.4/RADV (0
   occurrences in `vulkaninfo`), which rules out ncnn's coopmat GEMM path for us and settles the
   question round 1 left open. `VK_EXT_shader_tile_image` is graphics-only; `VK_KHR_maintenance7`
   robustness has no throughput effect for us.
7. **`VK_KHR_shader_subgroup_rotate` / `_extended_types`** — a 32-lane reduction is 5 shuffle-adds
   either way; nothing to gain at this size.

## 5. Part B — audit: ncnn's Vulkan compute runtime

Clone: `Reference Projects/ncnn` @ `9fc8106e` (2026-10-26), 131 files under `src/layer/vulkan`, no
build attempted (nothing here needs to run). This is the most *Vulkan-native* engine in the
rotation: no CUDA anywhere, so its solutions are the ones shaped by the same constraints we have.

### 5.1 Patterns worth taking

1. **Tracked per-resource dependency barriers** (`src/command.cpp:2214-2348`). Covered in §3 — the
   single most valuable transfer, and it dissolves our parked blocker.
2. **Pipelines keyed on `(spirv, specializations, local_size_xyz, subgroup_size)` with a persistent
   cache** (`src/pipelinecache.h`, `VkCompute`/`PipelineCache::get_pipeline`). The local size is not
   baked into the shader: `VulkanDevice::compile_shader_module(spv, lx, ly, lz)`
   (`src/gpu.h:461`) **patches `local_size` into the SPIR-V** so one module serves every geometry.
   That is why ncnn can afford per-shape pipelines: they cost a spec-constant variation, not a
   shader compile. Binaries can be saved/loaded to disk (`save_cache`/`load_cache`), the same idea as
   `VK_KHR_pipeline_binary`.
3. **Descriptor update templates, with `VK_KHR_push_descriptor` preferred when available**
   (`create_descriptor_update_template`, `vkCmdPushDescriptorSetWithTemplateKHR`, and the
   `support_VK_KHR_push_descriptor()` branches throughout `command.cpp`). Per-dispatch descriptor
   cost collapses to a template apply. Combined with deferred record replay (below) it is the
   closest thing to a "graph" a pure-Vulkan engine can have without indirect dispatches.
4. **Deferred record list replayed at submit** (`VkComputePrivate::record`, `delayed_records`): each
   `record_*` appends a small struct, and only `submit_and_wait` walks the list into the command
   buffer. That decouples *what the layer asked for* from *what is recorded*, which is what allows
   patterns 1 and 3 to be decided at replay time with full knowledge of the neighbouring ops. Our
   `dispatch_batch` records eagerly; a deferred list is the structural prerequisite for
   resource-tracked barriers.
5. **Two allocators, block-suballocated.** `VkBlobAllocator` sub-allocates from **16 MiB** blocks
   aligned to `buffer_offset_alignment` with a per-block free list (`budget`); `VkWeightAllocator`
   uses **8 MiB** blocks and can prefer host memory for weights (`src/allocator.h:308-334`,
   `allocator.cpp:696-823`). Our per-dispatch `vkAllocateMemory` is fine for a probe but not for a
   1 500-dispatch graph.
6. **Quantised-weight kernels keep the decode in registers and pack 4/8 outputs per invocation**
   (`elempack`): the int8 winograd/gemm path fuses dequant into the GEMM, mirroring our
   `gemm_body.glsl`. No m-batching (ncnn's "batch" is a CNN image batch), so nothing to take for
   small-M beyond confirmation that decode-in-registers is the right structure.
7. **`NCNN_COVERAGE` shader coverage mode** (`gpu_coverage.cpp`): records which shaders a run
   actually executed and barriers between them. Cheap analogue for us: assert that a quant type's
   pipeline was reached at least once, which would have caught several of our "kernel not wired in"
   bugs.

### 5.2 Mobile assumptions that will NOT transfer

* **fp16 as the default compute type.** ncnn gates everything on `support_fp16_storage`,
  `support_fp16_packed`, `support_fp16_arithmetic` (`src/gpu.cpp:2092-2107`) because Adreno/Mali
  are 2× fp16 and often *only* fast in packed fp16. Our cards do have 2× packed fp16
  (`V_PK_FMA_F16`), so the *idea* transfers — but the shader-level packing assumptions
  (`elempack` 4/8 = SIMD over channels) are CNN-shaped and irrelevant to GEMM.
* **Storage-image compute paths** (`support_image_storage`, `VkImageMat`,
  `shape_support_image_storage`) exist because Mali's buffer path is slow and its image path is
  not. RDNA is the opposite; do not adopt `VkImageMat`.
* **Per-driver tile/unroll workarounds.** `convolution_vulkan.cpp` is littered with them: *"adreno
  driver fails with multiple M tiles per subgroup"* (lines 229, 948, 1185), *"adreno int8 register
  allocation fails with subgroup tile unrolling"* (2599, 2667), *"adreno miscomputes int8 im2col
  with multiple K tiles"* (2670), *"prefer 32-column tiles for adreno int8 winograd"* (2547). These
  are negative-knowledge for us: they encode a driver's bugs, not a machine's strengths.
* **Winograd** (`convolution_winograd_23/43`) and im2col GEMM tiling are convolution transforms; no
  analogue in a transformer's dense GEMM.
* **`max_workgroup_count` clamping / `set_optimal_local_size_xyz`** (`src/pipeline.cpp:70-103`)
  halving tile dims until the product fits `max_workgroup_invocations` — useful hygiene, but our
  measured plateau (§1) says the answer for our shapes is simply "one lane group per row, fully
  occupied".

## 6. Ranked adaptations for Breccia

| # | Adaptation | Evidence | Measurable sketch |
|---|---|---|---|
| 1 | **Fuse gate+up+SwiGLU into one dispatch** for the shared expert (and the same pattern for HC_READ's `norm`/`gated`) | §2.2: 15.6 → 4.7–5.0 µs (m=1), 50.7 → 9.3 µs (m=8) | one flag, one shader; end-to-end ms/token at m=8 chunk, bit-identical when `silu` is the same expression |
| 2 | **Resource-tracked barriers** (ncnn `barrier_readwrite` ported into `dispatch_batch`) | §3: 1.07 µs per barrier removed; 1509 → ~400 ⇒ −1.0 ms/token | one flag, `max \|Δlogit\| = 0`, dispatch-count counter shows the drop |
| 3 | **Shape-specialised pipelines** (`bpr`/`cols`/`users` as spec constants, `local_size` patched into SPIR-V) | §1 unroll-2 = −9 % is the floor of what full unrolling gets; ncnn §5.1.2 | per-shape spike first on gate/up m=8 (4 834 → target ≤4 300 ns) |
| 4 | **Descend on `UNROLL=2` for `bpr ≥ 4` only** (gate/up yes, `down` no) | §1: −9 %/−8 %/−9 % on gate/up m=1/m=8, down m=1; +2 % on down m=8 | one flag; GEMV parity unaffected (different kernel) |
| 5 | **`VK_KHR_pipeline_binary` + pipeline pre-warm in every test process** | §0.1: cold clock/pre-compile = 5× error; 2026-10-26 reset correlated with 24 fresh shapes | cache dir on disk; a sweep's first-vs-second shape variance should collapse to <3 % |
| 6 | **Descriptor buffer / update templates** for the per-dispatch descriptor write | round 4: 212 → 68 ns/dispatch when allocation is removed; ncnn §5.1.3 | host-lane ms/chunk, no GPU change |
| 7 | **Do not** stage X in LDS; **do not** re-tune workgroup size; **do not** force subgroup 32 | §2.1, §1, §4.1 | negative results — record and move on |

## 7. Top-3 next experiments (each with its bench)

1. **Fused gate+up+SwiGLU in the real graph (shared expert, m = chunk).**
   `timeout 90 env BRECCIA_KERNEL_DEVICE=0000:03:00.0 cargo test -p breccia-kernels --test conformance`
   then the two-agent gate:
   `timeout 90 ./target/release/breccia --model … --threads 16 --tokens 32` ×3 alternating binaries.
   Success = **≥2 ms/token** at m=8 chunk and bit-identical greedy tokens; kernel-level target
   9.3 µs/block against today's 15.6 µs chain (measured, §2.2).
2. **Resource-tracked barriers in `dispatch_batch`.** One flag
   (`BRECCIA_BARRIER_TRACKED=1`), assert \|Δlogit\| = 0 on both cards, count dispatches-with-barrier
   (expect ~400/1 509). Gate: **ms/token −0.7…−1.0**, and *no* barrier on weight→x-only dispatches
   (weights are read-only for the whole token — with tracked flags those need zero barriers, which
   the call-site heuristic could never prove).
3. **`bpr`/`cols`/`users` specialisation on the gate/up GEMM.** Spike in the harness first:
   rebuild `gemm_q4k.comp` with `-DBPR=10 -DCOLS=2560 -DUSERS=8 -DUNROLL=10` (full unroll) and
   compare against the measured 4 834 ns; if it lands ≤4 300 ns, move to real `VkSpecializationInfo`
   + a keyed pipeline cache, and re-measure `pipeline_binary` cold-start.

## 8. Open questions this round did not settle

* **Why a dependent 2-dispatch chain costs 4× the sum of its parts** (10.7 µs vs 2 × 2.8 µs at m=1).
  The barrier is only 1.1 µs of it. Candidates: wave-drain per dispatch at 80 workgroups on 60 CUs,
  or the descriptor-set rebind. This matters because it is the *same* factor that makes the
  engine's shared expert cost ~120 µs/block while the work is ~8.5 µs — worth one measurement with
  a 3 200-row shape (320 WGs) to see whether the drain disappears at high occupancy.
* **`VK_KHR_cooperative_matrix` is absent**, so a WMMA path for prefill (m ≥ 64) is off the table;
  fp32 packed-fp16 (`V_PK_FMA_F16`, 2×) remains the only arithmetic-rate lever, and only if the
  decode can be expressed in fp16 without a parity break.
* The 20.9–55.3 ms chunked shared expert in the trace is **not** explained by kernel time at m=8
  (§1 says 0.90 ms/token over 48 blocks). Whatever else is in that number, it is dependency and
  dispatch structure — which is exactly what adaptations 1 and 2 attack.

## Appendix A — rebuilding the sweep (exact defines)

`gemm_body.glsl`'s geometry lifted into compile-time defines, plus one extra body variant. Base:
`crates/breccia-kernels/shaders/gemm_q4k.comp` = `#define QK 256u` / `#define BLOCK_WORDS 72u` /
`#include "blocks.glsl"` / `#define DEQUANT_UNIT dequant_q4_k_unit`, then `gemm_body.glsl`'s `main`
verbatim with these substitutions:

```glsl
layout(local_size_x = LSX) in;               // -DLSX=64|128|256|512|1024
#define ROWS   ROWS_MACRO                    // -DROWS=1|2|4|8|16|32, kept <= LSX/32
#if LDS_VECS > 0                             // -DLDS_VECS=1280 for the 20 KiB X-staging variant
shared vec4 lds_x[LDS_VECS];                 // cooperative load + barrier() before the early-out
#endif
```

* **`-DUNROLL=n`**: decode `n` blocks before any dot product so all `n` block loads are in flight:

  ```glsl
  for (; b + UNROLL <= bpr; b += UNROLL) {
      float vv[UNROLL][8];
      [[unroll]] for (uint t = 0u; t < UNROLL; ++t)
          DEQUANT_UNIT(row_base + (b + t) * BLOCK_WORDS, u, vv[t]);
      [[unroll]] for (uint t = 0u; t < UNROLL; ++t) { /* dot all m users, acc0/acc1 by (b+t)&1 */ }
  }
  ```

  Accumulator parity must stay keyed on the absolute block index (`(b+t)&1`, not `t`), otherwise
  the result is not bit-comparable with `UNROLL=1`.

* **Fusion prototype** (`gateup_swiglu.comp`): `#define BRECCIA_OWN_BINDINGS 1` so the four SSBO
  bindings and a 24-byte push constant (`rows, cols, bpr, base_gate, base_up, users`) can be
  declared locally, then the same `DEQUANT_UNIT` twice per block (`vg[8]`, `vu[8]`) and one
  per-user `dot` pair; epilogue `O[j*rows + row] = silu(Σgate) * Σup` after a 32-lane shuffle
  reduce of both sums.

* **Fixture** (this is what makes the numbers comparable): every Q4_K block has `d = f16 0x2400`
  (2⁻⁶), `dmin = f16 0x2000` (2⁻⁷ — both finite and small), and pseudo-random bytes for the scale
  and `qs` fields, so no fixture can produce inf/NaN scales while every decode path (scale unpack,
  nibble select, tail mask) is exercised. Correctness gate: all geometries bit-identical and a CPU
  f64 dequant+dot within 3 × 10⁻⁶ relative.

* **Command lines** (RX = `--device 0`, MI50 = `--device 1`, all wrapped in `timeout 90`):

  ```bash
  <bin> --device 0 --mode verify                                  # correctness gate
  <bin> --device 0 --mode all    --json rx.json                   # sweep + barrier + fuse
  <bin> --device 1 --mode sweep  --shape gate --m 8 --sub none \
        --only ls64/r2,ls128/r4,ls256/r8,ls512/r16                # display card: <=4 fresh shapes
  ```
