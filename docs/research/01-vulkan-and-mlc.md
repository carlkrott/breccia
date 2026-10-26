# 01 — Vulkan capability audit (both discrete devices) + MLC-LLM audit

Phase 4 paired research, round 1. Date: 2026-10-26. **Read-only**: no engine code was changed, no
GPU benchmark was run (capability queries only). Everything below is either read out of the live
driver with `vulkaninfo` / the ICD binary, or read out of the cloned reference tree with a `file:line`.

Scope targets, from the current measurement set (`docs/phase-4/01-forward-pass.md`):

| label | bottleneck | number today |
|---|---|---|
| **(a)** | GPU half of decode: tiny dispatches | **1 509 dispatches at 46.9 µs avg = 70.8 ms/token** (§8l) |
| **(b)** | host expert lane, ALU-bound | **37–50 ms/token at ~3–5 GB/s effective** (§8p) |
| **(c)** | merged prefill being wired (multi-user GEMM + fan-in) | **6.13 tok/s/agent** baseline; budget 72.3 MiB/token (§8u/§8v) |
| **(d)** | f32 storage atomics for the fan-in — *"this device has none"* | claim in §8v item 1 / §8c |

---

# Part A — Vulkan capability audit

## A0. Method and environment (reproducible)

```bash
env -u MESA_VK_DEVICE_SELECT vulkaninfo > vkinfo.txt     # full dump, both devices, 7 889 lines
# extension lists, per device (the "Device Extensions: count = N" blocks)
awk 'NR>=2805 && NR<=3374' vkinfo.txt | awk '{print $1}' | grep '^VK_' > ext_rx.txt    # 245
awk 'NR>=4763 && NR<=5218' vkinfo.txt | awk '{print $1}' | grep '^VK_' > ext_vega.txt  # 218
comm -23 <(sort ext_rx.txt) <(sort ext_vega.txt)   # 28, RX only
comm -13 <(sort ext_rx.txt) <(sort ext_vega.txt)   #  1, MI50 only
strings -n 3 /usr/lib/libvulkan_radeon.so | awk '$1>=7494600 && $1<=7495400'   # the RADV knob tables
```

| | value |
|---|---|
| loader / instance | **Vulkan 1.4.363** (`vulkan-icd-loader 1.4.363.0-1.1`, headers `1:1.4.363.0-1`) |
| ICD / device `apiVersion` | **1.4.354** (`radeon_icd.json`, `vulkan-radeon 3:26.2.4-1`) |
| driver | `radv`, `Mesa 26.2.4-arch3.1` |
| RX 6950 XT | GPU0, `AMD Radeon RX 6950 XT (RADV NAVI21)`, gfx1030, **245** device extensions |
| Radeon VII / MI50 | GPU1, `AMD Radeon VII (RADV VEGA20)`, gfx906, **218** device extensions |
| (iGPU, unused) | GPU2, `RAPHAEL_MENDOCINO`, 245 — same extension set as the RX |

The 1.4.363 vs 1.4.354 gap is loader-vs-ICD, not a capability difference: every extension and
feature cited below is what the **ICD** reports.

Extension-set diff is almost entirely graphics: the RX is **+28** — 14 `VK_KHR_video_*` /
`VK_VALVE_video_encode_rgb_conversion` entries, 7 ray-tracing (`VK_KHR_ray_*`,
`VK_KHR_acceleration_structure`, `VK_KHR_fragment_shader_barycentric`), `VK_EXT_mesh_shader`,
`VK_EXT_host_image_copy`, `VK_KHR_performance_query`, `VK_EXT_pipeline_library_group_handles`,
`VK_EXT_post_depth_coverage`, `VK_EXT_border_color_swizzle`, `VK_EXT_image_sliced_view_of_3d`,
`VK_KHR_fragment_shading_rate` — and the MI50 is **+1**: `VK_AMD_rasterization_order`. **No
compute-relevant extension exists on one card only.** That is the single most useful result of the
diff: capability work is symmetric, placement work is not.

## A1. The f32-atomics question, answered with feature bits (§8c/§8v item 1)

`VkPhysicalDeviceShaderAtomicFloatFeaturesEXT` and `...Float2FeaturesEXT`, both devices:

| feature | RX 6950 XT | MI50 | meaning |
|---|---|---|---|
| `shaderBufferFloat32Atomics` | **true** | **true** | f32 **storage-buffer atomics exist** (`atomicExchange` on a `float` in an SSBO) |
| `shaderBufferFloat32AtomicAdd` | false | false | **no `atomicAdd(float)` on storage buffers** |
| `shaderBufferFloat32AtomicMinMax` | **true** | false | RX only (from `VK_EXT_shader_atomic_float2`) |
| `shaderSharedFloat32Atomics` | true | true | LDS f32 atomics |
| `shaderSharedFloat32AtomicAdd` | **true** | **true** | **LDS `atomicAdd(float)` works on both cards** |
| `shaderImageFloat32AtomicAdd` | false | false | (we have no images) |
| `VK_EXT_shader_atomic_float` in device extension list | yes | yes | — |

**Verdict: the recorded claim "this device has no f32 storage atomics" is wrong as written, and right
in the one way that mattered.**

* Wrong: storage-buffer f32 atomics *are* advertised on both cards (`atomicExchange`, and on the RX
  also `atomicMin/Max`), and **shared-memory f32 `atomicAdd` is advertised on both**. A fan-in can
  therefore be built either as a two-phase (per-workgroup LDS accumulate → one global store per
  output) or with LDS atomics inside a token's workgroup.
* Right in effect: a global `atomicAdd(float)` scatter — the naive "each (expert, user) pair adds into
  `out[user]`" — is **not** available. Neither is a CAS loop idiomatic here: `atomicCompSwap` is
  integer-only, so it would need a `uint`-aliased buffer plus a loop, and it is *slow under
  contention* (the fan-in's contention is exactly worst-case: 10 writers per output row).
* **And it is moot**: §8v's landed `moe_fanin.comp` resolves `(token, expert) → slot` through a
  host-built slot map, so each output row has exactly one writer and needs no atomics at all. Keep
  that design. The correct one-line record for the report is: *"no `atomicAdd(float)` in storage
  buffers on either card; shared-memory f32 atomics are available; the fan-in avoids both."*

The hardware does have `global_atomic_add_f32` (gfx8+), so this is a RADV exposure choice, not a
silicon limit — worth re-checking after a Mesa upgrade and **not** worth a CAS emulation today.

## A2. Feature surface that matters to (a)–(d)

Both devices unless stated. "Used?" = does `crates/breccia-vk` or `crates/breccia-kernels` use it today.

| feature / property | RX | MI50 | used? | relevance | how we would use it | expected | effort |
|---|---|---|---|---|---|---|---|
| `subgroupSize` | 64 | 64 | yes | — | — | — | — |
| `minSubgroupSize` … `maxSubgroupSize` | **32 … 64** | **64 … 64** | queried only | **(a)** latency-bound kernels | `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` + `subgroupSizeControl`/`computeFullSubgroups` (both already enabled in `ctx.rs`) | see lever 3 | S |
| `requiredSubgroupSizeStages` | **FRAGMENT, COMPUTE, TASK, MESH** | **None** | no | **(a)** | wave32 is **RX-only**; on the MI50 the driver says "you cannot require anything" (min=max=64) | — | — |
| `shaderSubgroupRotate` (1.4) | true | true | no | (a) | one-instruction rotate for the reduction tree / `subgroupShuffleUp` ladders | ~0 today (reductions are already cheap vs the drain) | S |
| `shaderSubgroupExtendedTypes` | true | true | no | (a) | 8/16/64-bit shuffles | ~0 | S |
| `shaderIntegerDotProduct` | true | true | **no** | **(c)** | `OpSDot`/`OpUDot` (`i8vec4` · `i8vec4` → i32, packed 4-in-32-bit); 8-bit and 16-bit accumulator-saturating forms accelerated | see lever 5 | L |
| `integerDotProduct4x8BitPackedSignedAccelerated` | **true** | **true** | — | (c) | dp4a is real on both; **mixed signedness is NOT accelerated** on either | pins activation quant to signed int8 ↔ signed weights | — |
| `integerDotProduct16BitSignedAccelerated` | true | true | — | (c) | an int16 dot path exists if int8 precision is not enough | — | — |
| `integerDotProduct32/64Bit…Accelerated` | false | false | — | — | 16-bit multiples only | — | — |
| `VK_KHR_cooperative_matrix` | **absent** | **absent** | no | (c) | **confirmed absent**; also no matrix ISA on either chip (no MFMA/WMMA in gfx906 or gfx1030) | n/a | — |
| `shaderFloat16` / `shaderInt8` / `storageBuffer8BitAccess` | true | true | partly | (c) | real `int8_t`/`float16_t` in SSBOs → dp4a without bit packing; fp16 halves operand bytes (register/LDS pressure), it does **not** raise MAC rate on RDNA2 | — | S |
| `computeFullSubgroups` | true | true | no | (a) | required for `subgroupBarrier` after LDS hand-off in a wave32 kernel | prerequisite of lever 3 | — |
| `timestampPeriod` / `timestampValidBits` | 10 ns / 64 | **40 ns** / 64 | yes | measurement | MI50 timestamps quantise to 40 ns — kernels under 1 µs are not measurable there | — | — |
| `maxComputeWorkGroupInvocations` | 1024 | 1024 | yes | (a) | — | — | — |
| `maxComputeSharedMemorySize` | **65 536** | **65 536** | — | (a)(c) | 64 KiB LDS is available on both; a fused GDN pass or a fan-in staging buffer can use more than the 32 KiB often assumed | — | — |
| `maxStorageBufferRange` / `maxMemoryAllocationSize` | 4 GiB−1 | 4 GiB−1 | yes | (c) | one arena per card < 4 GiB is fine; a *single* buffer > 4 GiB is not (we already split) | — | — |
| `maxPushConstantsSize` | 256 | 256 | yes (5 words) | (c) | `base_words` fits; 60 more words free for a fused-kernel descriptor | — | — |
| `bufferImageGranularity` | 1 | 1 | — | (c) | no padding needed between buffer ranges in one arena | — | — |
| `vulkanMemoryModel` + `…DeviceScope` | true | true | no | (d) | explicit-scope atomics; only needed if we ever do device-scope f32 exchange | — | — |
| `bufferDeviceAddress` (+`CaptureReplay`) | true | true | no | (c)(d) | BDA lets a shader hold a *pointer* to an expert blob / slot row instead of `base_words` arithmetic (`GL_EXT_buffer_reference`) | cosmetic now that `base_words` works | S |
| `VK_KHR_shader_untyped_pointers` (1.4) | true | true | no | (c) | same, in SPIR-V form: pointer-typed indirection for the slot map | cosmetic | S |
| `VK_KHR_zero_initialize_workgroup_memory` | true | true | no | (a) | deletes explicit LDS zeroing in every reduction prologue | ~µs | S |
| `shaderExpectAssume` (1.4) | true | true | no | (a) | `expectEXT` on the hot uniform branches of the dequant loops (ACO can then drop the reconvergence) | 0–2 % | S |

## A3. API/runtime surface (the part that touches dispatch and copy cost)

| extension | RX | MI50 | used? | relevance | how we would use it | expected | effort |
|---|---|---|---|---|---|---|---|
| `VK_KHR_synchronization2` (core 1.3) | yes | yes | yes (`vkQueueSubmit2`) | **(a)** | `vkCmdPipelineBarrier2` with `COMPUTE_SHADER → COMPUTE_SHADER`, `SHADER_WRITE → SHADER_READ\|WRITE` instead of today's `ALL_COMMANDS → ALL_COMMANDS` + `MEMORY_READ\|WRITE` in `Gpu::barrier_all` | **lever 1** | S |
| `VK_KHR_copy_commands2` (core 1.3) | yes | yes | **no** | **(c)** | `vkCmdCopyBuffer2` with a **region array** (508 × 1.746 MB per block in one command) instead of one `vkCmdCopyBuffer` per blob | **lever 2** | S |
| `VK_KHR_copy_memory_indirect` | yes (`indirectMemoryCopy=true`, `indirectMemoryToImageCopy=true`) | yes | **no** | (c)(d) | `vkCmdCopyMemoryIndirect` reads the copy list **from a device buffer** → the miss-split lane can copy the experts the *GPU* routed, with no host round trip. Caveat, measured: `supportedQueues = GRAPHICS \| COMPUTE` — **not** the transfer-only family | medium | M |
| `RADV_EXPERIMENTAL=transfer_queue` (driver knob, see A5) | adds a **TRANSFER-only family, 2 queues** | same | no | **(c)** | makes `ctx.rs`'s `dedicated_x` branch fire → `XferMode::DistinctFamily`, SDMA engine | **lever 2** | **zero code** |
| `VK_EXT_device_generated_commands` | **`deviceGeneratedCommands=true`**, `dynamicGeneratedPipelineLayout=true`, `maxIndirectCommandsTokenCount=128`, `…StagesPipelineBinding = COMPUTE` | **same** | **no** | (a)(c) | `vkCmdExecuteGeneratedCommandsEXT` over a **device-built** token stream (`EXECUTION_SET` + `PUSH_CONSTANT` + `DISPATCH` tokens, `VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT` is present) — the dispatch list *and its push constants* come from a buffer the GPU just wrote | **lever 6** (deferred, see below) | M–L |
| `VK_EXT_descriptor_buffer` / `VK_EXT_descriptor_heap` | true | true | no | (a) | measured non-starter (§9 of `02-untried-levers.md`) | 0.3 % | — |
| `VK_EXT_mutable_descriptor_type` / `VK_VALVE_…` | true | true | no | (a) | only helps re-binding one set to different buffer types; we already reset pools | ~0 | — |
| `VK_EXT_push_descriptor` + templates | yes (`pushDescriptor=true`) | yes | **no** | (a) | `vkCmdPushDescriptorSetWithTemplateKHR` bakes binding data into the command stream → no `vkUpdateDescriptorSets`, no "fresh set per recorded dispatch" hazard. This is what TVM does (`vulkan_wrapped_func.cc:72`) | 0.3 % (already measured); **enables re-recordable command buffers** | S |
| `VK_KHR_maintenance4` | true | true | no | (c) | `vkGetDeviceBufferMemoryRequirements` before create → exact arena sizing | trivial | S |
| `VK_KHR_maintenance5/6` | true | true | no | — | `vkGetRenderingAreaGranularity`, push-descriptor fixes; nothing compute-relevant | 0 | — |
| `VK_KHR_maintenance7…11` | true | true | no | — | 7 = render-pass/query, 8 = 64-bit `VkAccessFlags3`, 9 = default vertex attribute values, 10 = attachment flags, 11 = image transfer granularity. **Graphics only** | 0 | — |
| `VK_KHR_map_memory2` | true | true | no | — | `vkMapMemory2` with flags; cosmetic | 0 | — |
| `VK_EXT_host_image_copy` | **RX only** | absent | no | — | images only; we have none | 0 | — |
| `VK_KHR_pipeline_binary` (`pipelineBinaries=true`) | true | true | **no** | start-up | `vkGetPipelineBinaryData` → cache compiled pipelines across runs, keyed by the driver's UUID | cuts cold start (N specialisations), **0 runtime** | S |
| `VK_EXT_shader_module_identifier` | true | true | **no** | start-up | `vkGetShaderModuleIdentifierEXT` → look up before compiling | same | S |
| `VK_EXT_pipeline_creation_cache_control` (core 1.3) | true | true | no | start-up | `CREATE_PIPELINE_FAIL_ON_PIPELINE_COMPILE_REQUIRED` to never block on compile | same | — |
| `VK_KHR_pipeline_executable_properties` | true | true | **no** | **(a)** | per-shader **VGPR / SGPR / LDS / scratch / code size + ACO disassembly** (`VK_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_AMD`) | **lever 4** | S |
| `VK_AMD_shader_core_properties` / `2` | true | true | no | (a) | `shaderEngineCount`, `computeUnitsPerShaderArray`, `simdPerComputeUnit`, `wavefrontsPerSimd` → exactly how many waves/SIMD a kernel *achieves* vs how many it *needs* | pairs with lever 4 | S |
| `VK_EXT_calibrated_timestamps` / `VK_KHR_calibrated_timestamps` | true | true | no | measurement | map GPU ticks to `CLOCK_MONOTONIC` → attribute the **host-lane/GPU overlap** honestly instead of comparing two unsynchronised clocks | better attribution of (b) | S |
| `VK_EXT_memory_budget` | true | true | **no** | (c) | `vkGetPhysicalDeviceMemoryProperties2` with `MemoryBudgetPropertiesEXT` → pick residency from the live budget instead of a static `-ot` list | makes `-ot` adaptive | S |
| `VK_EXT_memory_priority` (`memoryPriority=true`) | true | true | **no** | (c) | priority 1.0 on the resident-expert arena so RADV never evicts it under pressure | low, free | S |
| `VK_EXT_external_memory_host` | true | true | **yes** | (b) | already wired; zero-copy BAR reads measured negative (T22) | — | — |
| `VK_EXT_pci_bus_info` | true | true | (we resolve by sysfs) | — | could replace the sysfs walk in `pci.rs` | cosmetic | — |
| `VK_KHR_performance_query` | **RX only** | absent | no | (a) | present, but RADV's counter-set support is thin; treat as unavailable until probed | ~0 | — |
| `VK_EXT_shader_object` | true | true | no | — | replaces pipelines; per-dispatch gains already timed as noise | 0 | — |
| `VK_EXT_graphics_pipeline_library` | true | true | no | — | graphics | 0 | — |
| `VK_KHR_shader_constant_data` / `_abort` / `_quad_control` / `_maximal_reconvergence` / `_terminate_invocation` / `_float_controls2` / `_fma` | true | true | no | — | compute-side but no measurable cost in our profile | 0 | — |

## A4. Memory: ReBAR is live, and the recorded flag table needs one correction

Both cards expose **11 memory types in the same 2-heap layout** — heap 0 = 29.22 GiB system (GTT),
heap 1 = **15.98 GiB device-local VRAM**. Because a 16 GiB BAR is present, the driver also advertises
**device-local memory types that the CPU can map**:

| type | heap | flags | note |
|---:|---|---|---|
| 0, 1 | VRAM | `0x0001` | device-local only — what TVM picks for compute buffers |
| 3 | VRAM | `0x0007` | DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT (image formats listed) |
| 4 | VRAM | `0x0007` | same flags, **no image formats** → the buffer-only host-visible VRAM type |
| 9 | VRAM | `0x00c7` | + DEVICE_COHERENT_AMD + **DEVICE_UNCACHED_AMD** |
| 5, 6 | system | `0x000e` | HOST_VISIBLE + HOST_COHERENT + **HOST_CACHED** |
| 7 | VRAM | `0x00c1` | DEVICE_LOCAL + device-coherent + **device-uncached**, *not* host-visible |
| 8, 10 | system | `0x00c6` / `0x00ce` | host-visible coherent, uncached / + host-cached |

**Correction to the addendum in `02-untried-levers.md`.** It records type 9 as
*"+ HOST_CACHED + DEVICE_COHERENT/CACHED"*. `0x00c7` has **no** `HOST_CACHED` bit; it has
`DEVICE_UNCACHED_AMD`. So the VRAM mappings are **uncached from the CPU's point of view** (write-combined
at best): fine for host→GPU writes, and *bad* for any CPU readback (expect hundreds of MB/s, not tens
of GB/s). The recommendation in that addendum still stands, but the split must be:

* VRAM types 4 / 9 → **uplink** buffers only (the 10 KiB `mixed` activation, the mailbox writes). Never
  the router-id readback.
* system types 5/6/10 (`HOST_CACHED`) → **downlink** (router ids, logits, any CPU-consumed result).
* TVM's independent implementation ranks memory the same way (`vulkan_device.cc:307-330`: "prefer not
  host visible" for compute, "`rank += HOST_CACHED`" for staging), which is a second opinion that the
  VRAM host-visible types are not for CPU reads.

Also measured: every *queue family* has `present support = true`, and the only transfer-capable
families by default are the graphics family (1 queue) and the compute family (**4 queues**).

## A5. RADV driver knobs that exist on this Mesa (read out of the ICD, then tested)

Read from `/usr/lib/libvulkan_radeon.so` (option tables at file offset ≈ 7 494 600):

| knob | values present | our use | verdict |
|---|---|---|---|
| `RADV_EXPERIMENTAL=transfer_queue` | `transfer_queue` | **not used** | **measured**: adds a dedicated **TRANSFER-only family with 2 queues** on both cards (RX family 3, MI50 family 2). This is the only way to get real async copies. Also `bfloat16`, `sparse`, `hic`, `localbos`, `emulate_rt`, `msrtss`, `video_decode/encode` |
| `RADV_DEBUG=nodma` | `nodma` | not used | diagnostic: turns off the SDMA path for buffer copies → **A/B to prove whether our 28.7 GB/s expert stream is already going through SDMA** |
| `RADV_DEBUG=syncshaders` | `syncshaders`, `fullsync` | not used | diagnostic: forces a wait between consecutive dispatches → **measures how much overlap our current 1 509-dispatch stream actually has** |
| `RADV_DEBUG=nocache`, `nircache` (`PERFTEST`) | both | not used | reproducibility: `~/.cache/mesa_shader_cache` must be identical/warm across A/B arms, or explicitly disabled in both |
| `RADV_DEBUG=shaderstats`, `psocachestats` | both | not used | prints per-pipeline VGPR/code stats **without** writing API code — cheap predecessor to lever 4 |
| `RADV_DEBUG=zerovram`, `nobinning`, `nongg(c)`, `nogpl`, `nort`, `nomeshshader`, `noeso`, `novideo`, `nofmask`, `nohiz`, `nodcc` | graphics | — | no compute effect |
| `RADV_PROFILE_PSTATE=standard\|peak\|min_sclk\|min_mclk` | present | **not used** | **pin the clocks for A/B windows** — removes the PSU/thermal variance that AGENTS.md's measurement discipline is trying to control (needs the sysfs `pp_*`/perf-level permission) |
| `RADV_PERFTEST=` | `dccmsaa, cswave32, pswave32, gewave32, nosam, rtwave64, dmashaders, nogttspill, rtcps, nofastclears, nohiz, nocachecompat, shaderstats, …` | not used | nothing in this list reduces per-dispatch cost or helps GCN/RDNA2 small-kernel latency (`cswave32/pswave32/gewave32` are graphics stages) |
| `RADV_QUEUE_DISABLE=compute` | present | — | diagnostic only |
| `MESA_SHADER_CACHE_{DIR,DISABLE,MAX_SIZE,SHOW_STATS}`, `MESA_VK_TRACE` | present | `MESA_VK_DEVICE_SELECT` known | use `SHOW_STATS` when auditing compile cost |

**There is no RADV knob that makes a 1509-dispatch token fast.** The knobs that matter are (i) getting
the SDMA family, (ii) diagnosing what the copies and dispatches already do, and (iii) clock pinning for
honest A/B. Everything else in the table is graphics.

## A6. Ranked: the top 5 unused Vulkan levers

Ranked by expected movement of (a)–(d) per unit of effort, and each with the experiment that settles it.
Brutal honesty up front about the two ceilings I measured:

* **RX expert streaming is at the PCIe wire limit.** 28.7 GB/s measured against a PCIe 4.0 ×16 link's
  ~30 GB/s usable → **no Vulkan lever improves copy bandwidth**. What is left on (c) is *bytes*
  (residency) and *overlap*, never speed.
* **There is no matrix hardware.** `VK_KHR_cooperative_matrix` is absent on both, and neither gfx906
  nor gfx1030 has a matrix ISA. dp4a is the ceiling for int8 (≈2× f32 per cycle, see lever 5).

### Lever 1 — per-dispatch barrier scope (targets (a), the 70.8 ms GPU half)

**What we do.** `breccia_kernels::dispatch_batch` records one dispatch and then
`Gpu::barrier_all` (`crates/breccia-vk/src/ctx.rs:953`) = `srcStage ALL_COMMANDS → dstStage ALL_COMMANDS`,
`srcAccess MEMORY_READ|WRITE → dstAccess MEMORY_READ|WRITE`, after **every one of the 1 509 dispatches**.

**Why it is the top item.** The GPU half is 70.8 ms. `§8l`'s own table sums the *individually measured*
kernel times to **42–49 ms/token** on the slower card. So **21–29 ms/token is not arithmetic** — it is
37–41 % of the GPU half, and the only two candidates are (i) the drain/flush each `ALL_COMMANDS` barrier
forces and (ii) inter-submission gaps between the ~96 segments/token. The synchronous protocol is not
the cost any more (host record is 0.32 ms/token after the pool fix, and a *synchronous* dispatch costs
36 µs (MI50) / 64 µs (RX) per `dispatch_overhead`), so the money is in the barrier.

**Experiment (one afternoon, MI50 first, validation on).** Extend
`crates/breccia-engine/examples/dispatch_overhead.rs` with a batched mode: 512 identical tiny
dispatches in **one** command buffer, three arms — (i) today's `barrier_all`, (ii)
`vkCmdPipelineBarrier2` with `COMPUTE_SHADER→COMPUTE_SHADER` / `SHADER_WRITE→SHADER_READ|WRITE`, (iii)
no barrier between dispatches that touch disjoint buffers. Print µs/dispatch. Then the same on a real
block (`bench token`, parity gate unchanged) with the stage-minimal barrier, which is bit-identical by
construction.

**Expected.** (i)→(ii) is a flush-scope reduction: expect 2–8 µs/dispatch off = **3–12 ms/token**.
(ii)→(iii) shows the remaining drain floor and is what a dependency-tracking recorder would buy.
If the three arms come out equal, the 21–29 ms is inter-segment gap and the answer is deeper batching
(the 6-slot ring already allows it), not barrier work — either way we learn where 30 % of the GPU half
goes, which we currently do not know.

**Risk.** Low. `SYNCHRONIZATION2` is core, `vkQueueSubmit2` is already the submission path, and the
narrow barrier is a strict subset of the current one. Keep the full barrier for the
`record_barrier_to_host` path only.

### Lever 2 — a real SDMA transfer queue, and region-array copies (targets (c), and the copy half of (a))

**What we do.** `RingConfig`/`TransferRing` (`crates/breccia-vk/src/transfer.rs`) submits
`vkCmdCopyBuffer` **one region per call** on the *second queue of the compute family*
(`XferMode::SameFamilySecondQueue`) — a compute-family queue, not a transfer-only one. Whether RADV
routes those copies to the SDMA engine internally is exactly what `RADV_DEBUG=nodma` (A5) settles, and
it has never been checked here.

**What is available.** (i) `RADV_EXPERIMENTAL=transfer_queue` adds a TRANSFER-only family (2 queues,
measured above), and `ctx.rs:498-527` **already prefers it** (`dedicated_x` → `XferMode::DistinctFamily`),
so arm A needs **no code at all**. (ii) `vkCmdCopyBuffer2` takes a `VkBufferCopy2` array — the
"508 regions in one command" design of `02-untried-levers.md` §2 is a one-call change, and today's
`vkCmdCopyBuffer(&[region])` cannot express it.

**Experiment (no code, then small code).** `bench transfer --regions 130,508 --bytes 1746000` ×
`{default, RADV_EXPERIMENTAL=transfer_queue}` and record the reported `xfer_mode`. Then repeat on the
real path: `bench agents --streams 4 --prefix 300 --prefill-chunk 128`, since merged prefill is the
copy-heavy path.

**Expected.** Bandwidth flat (wire limit). The gain is **overlap and per-region overhead**: 130 region
copies at ~1–2 µs of CP-packet setup each on the GFX ring is 130–260 µs *of GFX time* per block per
chunk, ×48 blocks ≈ **6–12 ms per 128-token chunk**, and it also removes those packets from the
dispatch stream that lever 1 is trying to shorten. Diagnostic companion: `RADV_DEBUG=nodma` tells us
whether today's copies are already SDMA-backed (if `nodma` changes nothing, they are not, and the
gain is real).

**Risk.** Low. A documented experimental flag; gate it behind a report line that records the mode.

### Lever 3 — wave32 on the RX (targets (a); the latency-bound 34 % of it)

**What we measured.** RX: `minSubgroupSize 32`, `maxSubgroupSize 64`, and `requiredSubgroupSizeStages`
**includes COMPUTE**. MI50: `64…64`, stages **None** — so this is an **RX-only** lever, and the
`VK_EXT_subgroup_size_control` extension being present on the MI50 means nothing.

**Why it is worth an afternoon, but only that.** The bandwidth-bound GEMVs are at 83–106 % of RX DRAM
and cannot gain. The kernels that *can* gain are the ones already measured below the roof and
latency-limited: the **GDN delta rule (217 GB/s vs 596 GB/s for a comparable GEMV, 12.9 ms of the
70.8 ms)**, the **HC read chain (half of 596 GB/s, 11.4 ms)**, and the **shared expert (81–161 GB/s,
5.8 ms)**. gfx1030 is a SIMD32 machine: wave64 issues as 2 cycles per instruction and needs *one wave
per SIMD to fill*; wave32 doubles the resident wave count per SIMD, which is exactly the resource
those three kernels are short of.

**Experiment.** A `--subgroup 32|64` flag on `bench gemv`/`bench ops`/`bench token`, MI50 first (must be
a no-op — it will refuse the required size), then RX. Parity: a 32-lane tree is not a 64-lane tree, so
if logits move, keep a **fixed 64-wide LDS reduction inside the wave32 kernel** (deterministic, and
`computeFullSubgroups` + `VK_KHR_zero_initialize_workgroup_memory` are both available to do it cheaply).

**Expected.** 0 % on bandwidth-bound kernels; **5–15 % on the delta rule and HC read**, i.e. **1–3 ms/token**
of the 70.8 ms if the three latency-bound families behave as their numbers suggest. Cheap to falsify.

**Risk.** Low; reduction order is the only correctness surface.

### Lever 4 — make the occupancy question measurable (`VK_KHR_pipeline_executable_properties`)

**Why.** 34 % of the GPU half (GDN 12.9 + HC read 11.4 ms) is described in `§8l` as "latency-bound",
which is a symptom, not a cause. The cause is one of: too few resident waves (SGPR/VGPR pressure), LDS
over-allocation, scratch spills, or an `s_waitcnt`-bound dependency chain. We currently cannot tell
which, and every candidate fix (wave32, LDS restructure, split-K, register-blocking) is priced blind.

**What.** With `VK_KHR_pipeline_executable_properties` + `VK_AMD_shader_core_properties`, dump per
pipeline: VGPR/SGPR counts, LDS bytes, scratch bytes, code size, and the ACO disassembly
(`VK_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_AMD`). Compute achieved waves/SIMD from the VGPR
count and compare with `wavefrontsPerSimd`. This is a ~60-line probe in `breccia-vk/src/probe.rs`
(already exists) and no shader changes.

**Expected.** Not a speed-up by itself; it is the cheapest way to *choose* between the three fusions
that §8l already priced (`−2.5 ms`, `−1.0 ms`, `−3.0 ms`) and to know whether wave32 (lever 3) helps
before spending the parity risk. Do it in the same window as lever 3.

### Lever 5 — dp4a for the chunk's dense GEMM (targets (c))

**What we measured.** `shaderIntegerDotProduct` on both; **4×8-bit **packed signed** and **unsigned**
forms accelerated on both; **mixed signedness NOT accelerated** on either; 8-bit and 16-bit
integer dot accelerated; 32/64-bit not. `VK_KHR_cooperative_matrix` absent (no fallback to find).

**Where it pays.** Not decode (our dense chain is latency/bandwidth-bound at T=1, 81–161 GB/s). It pays
in the **merged chunk's dense chain after lever-2/§8u land**, where the projections become real GEMMs
with 128–256× the MAC count and maximal reuse — and in the expert GEMM's inner loop only if the
dequantisation is restructured to produce `int8` operands (a numerics change).

**Honest rate.** 4 int8 MACs per lane per instruction against 1 f32 FMA — but these chips issue
**2 f32 FMA/lane/cycle** (RX: 5 120 SP × 2 × 2.25 GHz ≈ 23 TFLOP/s; MI50: 3 840 × 2 × 1.8 ≈ 13.8), so
dp4a is **≈2× the f32 MAC rate, not 4×**. `02-untried-levers.md` rank 10 says "~4× f32"; that is
per-instruction, not per-cycle. Still worth it at chunk scale, still L effort, still **not** parity-safe.

**Constraint to design around.** Because mixed-signedness dot is not accelerated, activations must be
quantised **signed** (`q8_1` with symmetric range, exactly what Strata's `act_quant_q8_1` produces) and
weights signed — `v_dot4_i32_i8` form only.

## A7. Lever 6 — device-generated command streams (real, deferred on driver maturity)

`VK_EXT_device_generated_commands` is on **both** cards, and unlike most of the extension list it is
not graphics-only: the properties list `SHADER_STAGE_COMPUTE_BIT` under `supportedIndirectCommandsShaderStages`
and, decisively, `supportedIndirectCommandsShaderStagesPipelineBinding` and `…ShaderBinding` are
**COMPUTE_BIT only**, with `VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT` in the token enum
(`/usr/include/vulkan/vulkan_core.h:25373`).

That means a **dispatch sequence can be built on the device and executed with one host call**, push
constants included (`PUSH_CONSTANT_EXT` token; `maxIndirectCommandsTokenCount = 128` per sequence,
`maxIndirectSequenceCount = 1 048 576`). For §8v's orchestration this is the missing piece: today the
host must read the router's ids back and then record ~130 dispatches per block; with DGC the *routing*
(the router's own output) becomes the token stream, so a block's MoE becomes one
`vkCmdExecuteGeneratedCommandsEXT` recorded **before** the routing happens. Pair it with
`vkCmdCopyMemoryIndirect` (A3) and the whole per-block expert path — copy + dispatch — stops touching
the host.

**Why it is not in the top 5:** RADV's DGC is a newer path, and the honest expected gain on our current
numbers is small in the *decode* case, where host record is already 0.2 µs/dispatch (0.32 ms/token).
Its value is (i) prefill orchestration (§8v item 3, ~130 dispatches × 48 blocks per chunk with a host
readback between each) and (ii) it is the only Vulkan mechanism that removes dispatch *control* from the
host without merging kernels. Effort M–L, and it must be validated on the MI50 with
`VK_LAYER_KHRONOS_validation` before the RX (a malformed indirect stream is a page-fault class bug, per
AGENTS.md).

## A8. Measured-dead, with the reason (do not spend time here)

* **Queue multiplicity for intra-card overlap.** The compute family advertises **4 queues**, we use 1 —
  but on AMD every queue in a family is submitted to the *same GFX ring*, so extra queues
  time-slice packets, they do not run concurrently. Intra-card concurrency has to come from *one*
  submission containing independent work (or from the SDMA ring, lever 2), not from more queues. Cost
  to prove: one A/B on `bench token --streams 2`; expectation ≈0.
* **`VK_EXT_descriptor_buffer` / mutable descriptor type / shader objects.** Already timed: record is
  0.32 ms/token = 0.3 %. Push-descriptors-with-template is the only one worth touching, and only as an
  enabler for re-recordable command buffers.
* **Copy bandwidth.** 28.7 GB/s is ~96 % of a PCIe 4.0 ×16 link. Nothing in Vulkan changes the wire.
  (The MI50's 2.9 GB/s is 74 % of gen3 ×4 — there, `transfer_queue` might reach ~3.6 GB/s, which is
  still irrelevant: hot experts must be resident on that card.)
* **Pipeline binary / shader-module-identifier / creation-cache-control.** Start-up only (they matter
  for a cold `bench` and for a server restart, not for tok/s).
* **`VK_KHR_performance_query`.** RX-only and thin in RADV; lever 4 is the practical substitute.
* **Everything in the 28 RX-only extensions** except `performance_query`: ray tracing, mesh shaders,
  video codecs, `host_image_copy` (images), fragment shading rate, border-colour swizzle. Zero compute
  relevance, and the engine has no images.

---

# Part B — MLC-LLM audit (Vulkan backend focus)

## B1. What was cloned

```
Reference Projects/mlc-llm      shallow (--depth 1), sparse: python/ cpp/ docs/ 3rdparty/, HEAD 8978ea9
Reference Projects/mlc-llm/3rdparty/tvm   submodule 201fa00 (mlc-ai/relax), full tree, 46 MB
```
The audit therefore reads the **actual Vulkan runtime** (`3rdparty/tvm/src/backend/vulkan/runtime/`)
and the LLM machinery (`cpp/serve/`, `python/mlc_llm/`), not upstream docs. No network fallback needed.

## B2. What their Vulkan backend *is* — and the first useful fact

MLC's Vulkan target is declared with **`thread_warp_size: 1`**
(`python/mlc_llm/support/auto_target.py:539-553`): thread-warp size **1** means *no subgroup primitives
at all*. Every reduction in their generated code goes through **shared memory + `threadIdx` barriers**.
Their own docs describe Vulkan as the backend for "less typical environments (e.g. SteamDeck)"
(`docs/get_started/introduction.rst:317`).

Consequence for us: **MLC's Vulkan path is a portability target, not a performance target**, and its
kernels are structurally *worse* than ours on AMD for reduction-heavy work (we use subgroup reductions;
they pay an LDS round trip + barrier per output row). So the transferable value is in their
**compiler-side planning and their MoE/batching data structures**, not in their kernel bodies. That
reframes the whole audit and is why the ranked list below contains no schedule ports.

Runtime shape, for contrast with ours:

| | MLC/TVM Vulkan | Breccia |
|---|---|---|
| submission | one command buffer per CPU thread, **one submit + one fence per `StreamSync`**, kernels deferred as closures (`vulkan_stream.cc:100-170`) | 6-slot ring, one submit per segment, ~96 segments/token, no `vkQueueWaitIdle` |
| barrier after a dispatch | `COMPUTE_SHADER → TRANSFER\|COMPUTE_SHADER`, `SHADER_READ\|WRITE → …` (`vulkan_wrapped_func.cc:99`, `:166`) | **`ALL_COMMANDS → ALL_COMMANDS`** + `MEMORY_READ\|WRITE` (lever 1) |
| descriptor binding | `vkCmdPushDescriptorSetWithTemplateKHR` in the immediate path, deferred `vkUpdateDescriptorSets` otherwise (`vulkan_wrapped_func.cc:72`, `:110`) | fresh pool-reset set per recorded dispatch |
| memory | one `VkBuffer` + one `VkDeviceMemory` **per tensor**, no pooling (`vulkan_buffer.cc:56-80`); reuse comes from **compile-time planning** | arena buffers, hand-pooled |
| pipeline cache | **none** — no `vkCreatePipelineCache` anywhere in the runtime | RADV disk cache, unused API support |
| quantisation | group-quant (q4/q3 int + scale), decode kernels specialised per storage dtype | GGUF IQ/K-quants, `GemmMany`/fused GEMV |

## B3. Differences that are worth something to us

**B3.1 `indptr`-driven group GEMM: the whole MoE in ONE dispatch.**
`python/mlc_llm/op/moe_matmul.py:386 group_gemm` / `:560 dequantize_group_gemm`: a **fixed grid**
(`CTA_COUNT = 1024`, `BLK_M = 8`, `BLK_N = 128`, `BLK_K = 32`) where each CTA strides over
`(expert, tile)` pairs selected by an `indptr[num_experts+1]` array — `x[indptr[a] : indptr[a+1]]` is
expert `a`'s user range. Their decode twin (`:13 gemv`, `:78 dequantize_gemv`) uses the same
`indptr[0, slot]` indirection with `experts_per_tok` as `blockIdx.y`.

This is the same *data* design as our `moe_map`/`MoeMap` + `GemmMany`, but the *dispatch* design is
the opposite of §8v's plan: they do **not** issue one dispatch per distinct expert. Our §8v
orchestration is "~130 experts × 48 blocks = 6 240 dispatches per 128-token chunk, 0.22 ms/token at
4.5 µs". Theirs is **one group GEMM per projection role per block** — gate, up, down, so ~150
dispatches per 48-block chunk — with the expert loop inside each kernel and the SwiGLU folded into the
matmul epilogue by `compiler_pass/fuse_dequantize_matmul_ewise.py`. It also gives the fan-in a natural
home (the epilogue writes by *user* row index, so each output row has one writer — **no atomics**,
which is the same conclusion we reached through the slot map).

Note their `BLK_M = 8` independently matches our measured `GemmMany` sweet spot (m = 8: **5.1× RX /
3.4× MI50** per-user, §8u).

**B3.2 Compile-time memory planning, with a pre-flight estimate.**
`python/mlc_llm/compiler_pass/lift_global_buffer_alloc.py` + `estimate_memory_usage.py` +
`3rdparty/tvm/src/runtime/workspace_pool.{h,cc}`: the whole graph's buffers are planned **at compile
time** into a few arenas with a computed peak, and the engine refuses to start if the plan does not
fit. Reuse is a schedule property, not a runtime allocator property. (Their Vulkan-level allocation is
naive — one alloc per tensor — so the plan is what keeps the alloc count static and small.)

For us the transferable piece is the **pre-flight peak estimate** that gates residency: we choose
`-ot` by hand and discover the ceiling by failing. A static plan that prints "resident N experts →
peak X MiB, fits/does not fit" is what makes adaptive residency (from `VK_EXT_memory_budget`, A3)
tractable.

**B3.3 One function, batch-bucketed schedules.**
`python/mlc_llm/compiler_pass/low_batch_specialization.py:1-60`: for a symbolic matmul they build
`gemv(bucket 2)`, `gemv(bucket 4)` and a general `gemm`, then emit

```python
if symbolic_n <= 2:   <gemv bucket-2 body>
elif symbolic_n <= 8: <gemv bucket-4 body>
else:                 <gemm body>
```

inside **one** compiled function. The runtime batch size picks the schedule *on device*, with no extra
kernel selection on the host and no extra dispatch. We do the same thing with a push constant and a
host-chosen geometry, and a re-dispatch when the path changes. Buckets {2, 4, 8} are a cheaper shape
than our continuous `m`.

**B3.4 Prefix reuse as a radix tree with copy-on-write sequence forking.**
`cpp/serve/radix_tree.cc`, `prefix_cache.cc`, `new_request_prefill.cc:287-340` (`MatchPrefixCache`),
`model.h` `ForkSequence`. The KV blocks of a matched prefix are **shared, not copied**: a new sequence
forks the parent's page table and prefill runs only on the unmatched tail.

This is our `02-untried-levers.md` rank 5 with the mechanism filled in, and it is the highest-value
item in this audit for (c). Our version is harder in one respect (the GDN conv/recurrence, QSA indexer
and PLE rolling state are *mutable per-sequence state*, not page-table-able KV), so the fork has to
copy a state blob for those, not just a page table — but that blob is small and fixed-size, and the
page-table half transfers directly (our Phase-6 page table is already identity-only, per
`docs/phase-3/06` §3.1).

**B3.5 Verification as one batched device op.**
`python/mlc_llm/op/batch_spec_verify.py` + `cpp/serve/engine_actions/batch_verify.cc`: the verify
window is a **single batched forward** with the mask applied in-kernel and an explicit
`logit_positions` list, and the accept/reject walk for a *tree* of drafts is a device kernel over
`token_tree_parent_ptr / first_child / next_sibling` arrays — no host loop per candidate, no host-side
mask materialisation.

This is the shape our MTP (rank 6) should be built in: one dispatch chain per verify window, decision
on device, host only at the sampled-token boundary.

**B3.6 Small but free: padded tiles and alignment-driven vector width.**
`3rdparty/tvm/python/tvm/s_tir/dlight/gpu/low_batch_gemv.py` (`pad_einsum`, `auto_vectorize`,
tile search with `TS × TR ≤ max_num_threads`, rf-reduce then LDS cross-thread reduce). Our K values
(2 560, 640) and activation rows are not all power-of-two aligned; their discipline is "pad the tile so
the inner loop has no bounds check, and choose vec width from the *measured* alignment". Worth a look
at our non-quantised passes; not worth porting their reduction.

## B4. Ranked adaptations (impact on (a)–(d), effort, and the bench that proves it)

| # | adaptation | source | (a) | (b) | (c) | (d) | effort | bench |
|---|---|---|---|---|---|---|---|---|
| 1 | **Prefix radix tree + COW sequence fork** (skip prefill on the matched prefix; fork the state blob for GDN/QSA/PLE) | `cpp/serve/prefix_cache.cc`, `new_request_prefill.cc:287`, `radix_tree.cc` | 0 | 0 | **−40…60 % prompt seconds when agents share a prefix** (item 5 of the 02 doc, now with a mechanism) | 0 | M | `bench agents --streams 4 --prefix 300 --shared-prefix 200`, prompt seconds |
| 2 | **`indptr` group MoE per chunk** (expert loop *inside* one kernel, epilogue writes by user row → no atomics) | `op/moe_matmul.py:386`, `:560` | 0 (decode path untouched) | 0 | removes ~6 200 dispatches/chunk; ≈**0.2–0.4 ms/token** of the merged path, and deletes the per-expert SwiGLU pass | −1 (removes the need entirely) | M | `--prefill-chunk 128` with dispatch counter: 6 240 → ~150; parity 24/24 |
| 3 | **Verify window as one batched op with in-kernel mask + device accept walk** | `op/batch_spec_verify.py`, `engine_actions/batch_verify.cc` | removes the per-candidate host sync in the verify window | 0 | 0 | 0 | M | `bench token --mtp … --spec 3`: ms/window vs 108 ms; parity 64/64 |
| 4 | **Pre-flight memory plan + `VK_EXT_memory_budget`-driven residency** | `lift_global_buffer_alloc.py`, `estimate_memory_usage.py`, `workspace_pool.cc` | 0 | 0 | **+residency headroom** → feeds 02-doc ranks 2/3 | 0 | S | `bench agents --streams 4`: lane ms/step 98 → target 75–85, with a printed "fits" plan |
| 5 | **Batch-bucketed schedule in one function** (`if n<=2 gemv elif <=8 gemv else gemm`) | `low_batch_specialization.py` | 0 | 0 | small (removes host-side m bucketing) | 0 | S | shape sweep in `bench gemv`/`GemmMany` m = 1…8 |
| 6 | **Alignment-driven vec width + tile padding** audit of the non-quantised passes | `dlight/gpu/low_batch_gemv.py` | 0–2 % | 0 | 0–2 % | 0 | S | `bench ops` per-pass µs |

**Do not copy from MLC** (each would be a regression here):
* their **reduction strategy** (LDS instead of subgroups) — we are ahead;
* their **Vulkan memory layer** (one allocation per tensor, no pooling) — we are ahead;
* their **`compile-then-ship-a-.so` model** — a 48-block graph with our specialisation count would
  compile for minutes, and the engine's whole point is to be a runtime;
* their **quantisation set** — no GGUF IQ1/IQ2/IQ3 path, so nothing to reuse for our expert types;
* their **lack of a pipeline cache** — we should keep using RADV's, and additionally adopt
  `VK_KHR_pipeline_binary` for cold start (A3).

---

# Part C — combined top 3 "do this next", each with the bench that proves it

Chosen to be independent (different files, different subsystems) so they fit one grouped round per the
repo rule, and ordered so the cheapest falsification comes first.

1. **Barrier-scope measurement + fix (Part A lever 1).** Extend
   `examples/dispatch_overhead.rs` with the 512-dispatch batched arms (full / stage-minimal / none),
   then switch `dispatch_batch` to `vkCmdPipelineBarrier2` `COMPUTE_SHADER→COMPUTE_SHADER`.
   **Proves it:** µs/dispatch in the three arms, then `bench token --streams 2` with the parity gate
   unchanged. **Number to move:** GPU half 70.8 ms → **< 60 ms** if the projection holds; if the arms
   are equal the 21–29 ms is inter-segment gap and we go to a deeper ring instead.
   *Why first:* 30 % of the GPU half is currently unattributed, and this is a 1-file change on the
   MI50 with zero numerics risk.

2. **Dedicated SDMA family + region-array copies (Part A lever 2).** Arm A is
   `RADV_EXPERIMENTAL=transfer_queue` with **no code** (`ctx.rs` already prefers `dedicated_x`); arm B
   is `vkCmdCopyBuffer2` with the 130/508-region array.
   **Proves it:** `bench transfer --regions 508 --bytes 1746000` × {default, env} printing
   `xfer_mode`, then `bench agents --streams 4 --prefix 300 --prefill-chunk 128`.
   **Number to move:** merged-prefill prompt seconds, and the 6–12 ms/chunk of GFX-ring copy packets.
   *Why second:* it is the copy leg of (c) that §8u's budget assumes (2.6 ms/token of PCIe), and the
   first arm costs nothing but an env var.

3. **Occupancy diagnosis of the 34 % latency-bound families, then wave32 on the RX (Part A levers 3+4).**
   One window, one probe: `VK_KHR_pipeline_executable_properties` + `VK_AMD_shader_core_properties`
   VGPR/LDS/scratch/occupancy dump for the GDN delta rule, the HC-read `down` GEMV and the shared
   expert; then `--subgroup 32` on the RX only (MI50 refuses by capability), with a fixed 64-wide LDS
   reduction if parity moves.
   **Proves it:** per-shader achieved-vs-required waves, then `bench gemv`/`bench ops`/`bench token`
   A/B on **both** cards, parity 24/24 (or the LDS-reduction variant if not).
   **Number to move:** GDN 12.9 + HC read 11.4 + shared expert 5.8 ms; target **−1…−3 ms/token**.

Deferred by this audit, with reasons on record: dp4a (only pays after the chunk's dense chain is a real
GEMM, and it is a numerics change — Part A lever 5), descriptor/push-descriptor work (0.3 % measured),
pipeline binary caching (start-up only), queue multiplicity (one GFX ring per family — Part A §A8),
device-generated command streams until the copy leg is device-driven (Part A §A7), and
every f32-atomics variant of the fan-in (the slot map already removed the need — Part A §A1).

---

## Appendix A — complete RX 6950 XT device extension list (245)

```
  VK_AMD_buffer_marker                             VK_AMD_device_coherent_memory                    VK_AMD_draw_indirect_count                       VK_AMD_gcn_shader
  VK_AMD_gpu_shader_half_float                     VK_AMD_gpu_shader_int16                          VK_AMD_memory_overallocation_behavior            VK_AMD_mixed_attachment_samples
  VK_AMD_shader_ballot                             VK_AMD_shader_core_properties                    VK_AMD_shader_core_properties2                   VK_AMD_shader_early_and_late_fragment_tests
  VK_AMD_shader_explicit_vertex_parameter          VK_AMD_shader_fragment_mask                      VK_AMD_shader_image_load_store_lod               VK_AMD_shader_trinary_minmax
  VK_AMD_texture_gather_bias_lod                   VK_EXT_4444_formats                              VK_EXT_attachment_feedback_loop_dynamic_state    VK_EXT_attachment_feedback_loop_layout
  VK_EXT_border_color_swizzle                      VK_EXT_buffer_device_address                     VK_EXT_calibrated_timestamps                     VK_EXT_color_write_enable
  VK_EXT_conditional_rendering                     VK_EXT_conservative_rasterization                VK_EXT_custom_border_color                       VK_EXT_custom_resolve
  VK_EXT_depth_bias_control                        VK_EXT_depth_clamp_control                       VK_EXT_depth_clamp_zero_one                      VK_EXT_depth_clip_control
  VK_EXT_depth_clip_enable                         VK_EXT_depth_range_unrestricted                  VK_EXT_descriptor_buffer                         VK_EXT_descriptor_heap
  VK_EXT_descriptor_indexing                       VK_EXT_device_address_binding_report             VK_EXT_device_fault                              VK_EXT_device_generated_commands
  VK_EXT_device_memory_report                      VK_EXT_discard_rectangles                        VK_EXT_display_control                           VK_EXT_dynamic_rendering_unused_attachments
  VK_EXT_extended_dynamic_state                    VK_EXT_extended_dynamic_state2                   VK_EXT_extended_dynamic_state3                   VK_EXT_external_memory_acquire_unmodified
  VK_EXT_external_memory_dma_buf                   VK_EXT_external_memory_host                      VK_EXT_fragment_shader_interlock                 VK_EXT_global_priority
  VK_EXT_global_priority_query                     VK_EXT_graphics_pipeline_library                 VK_EXT_hdr_metadata                              VK_EXT_host_image_copy
  VK_EXT_host_query_reset                          VK_EXT_image_2d_view_of_3d                       VK_EXT_image_compression_control                 VK_EXT_image_compression_control_swapchain
  VK_EXT_image_drm_format_modifier                 VK_EXT_image_robustness                          VK_EXT_image_sliced_view_of_3d                   VK_EXT_image_view_min_lod
  VK_EXT_index_type_uint8                          VK_EXT_inline_uniform_block                      VK_EXT_legacy_vertex_attributes                  VK_EXT_line_rasterization
  VK_EXT_load_store_op_none                        VK_EXT_map_memory_placed                         VK_EXT_memory_budget                             VK_EXT_memory_priority
  VK_EXT_mesh_shader                               VK_EXT_multi_draw                                VK_EXT_mutable_descriptor_type                   VK_EXT_nested_command_buffer
  VK_EXT_non_seamless_cube_map                     VK_EXT_pci_bus_info                              VK_EXT_physical_device_drm                       VK_EXT_pipeline_creation_cache_control
  VK_EXT_pipeline_creation_feedback                VK_EXT_pipeline_library_group_handles            VK_EXT_pipeline_robustness                       VK_EXT_post_depth_coverage
  VK_EXT_present_timing                            VK_EXT_primitive_restart_index                   VK_EXT_primitive_topology_list_restart           VK_EXT_primitives_generated_query
  VK_EXT_private_data                              VK_EXT_provoking_vertex                          VK_EXT_queue_family_foreign                      VK_EXT_robustness2
  VK_EXT_sample_locations                          VK_EXT_sampler_filter_minmax                     VK_EXT_scalar_block_layout                       VK_EXT_separate_stencil_usage
  VK_EXT_shader_atomic_float                       VK_EXT_shader_atomic_float2                      VK_EXT_shader_demote_to_helper_invocation        VK_EXT_shader_image_atomic_int64
  VK_EXT_shader_module_identifier                  VK_EXT_shader_object                             VK_EXT_shader_replicated_composites              VK_EXT_shader_stencil_export
  VK_EXT_shader_subgroup_ballot                    VK_EXT_shader_subgroup_vote                      VK_EXT_shader_uniform_buffer_unsized_array       VK_EXT_shader_viewport_index_layer
  VK_EXT_subgroup_size_control                     VK_EXT_swapchain_maintenance1                    VK_EXT_texel_buffer_alignment                    VK_EXT_tooling_info
  VK_EXT_transform_feedback                        VK_EXT_vertex_attribute_divisor                  VK_EXT_vertex_input_dynamic_state                VK_EXT_ycbcr_2plane_444_formats
  VK_EXT_ycbcr_image_arrays                        VK_EXT_zero_initialize_device_memory             VK_GOOGLE_decorate_string                        VK_GOOGLE_hlsl_functionality1
  VK_GOOGLE_user_type                              VK_INTEL_shader_integer_functions2               VK_KHR_16bit_storage                             VK_KHR_8bit_storage
  VK_KHR_acceleration_structure                    VK_KHR_bind_memory2                              VK_KHR_buffer_device_address                     VK_KHR_calibrated_timestamps
  VK_KHR_compute_shader_derivatives                VK_KHR_copy_commands2                            VK_KHR_copy_memory_indirect                      VK_KHR_create_renderpass2
  VK_KHR_dedicated_allocation                      VK_KHR_deferred_host_operations                  VK_KHR_depth_clamp_zero_one                      VK_KHR_depth_stencil_resolve
  VK_KHR_descriptor_update_template                VK_KHR_device_address_commands                   VK_KHR_device_fault                              VK_KHR_device_group
  VK_KHR_draw_indirect_count                       VK_KHR_driver_properties                         VK_KHR_dynamic_rendering                         VK_KHR_dynamic_rendering_local_read
  VK_KHR_extended_flags                            VK_KHR_external_fence                            VK_KHR_external_fence_fd                         VK_KHR_external_memory
  VK_KHR_external_memory_fd                        VK_KHR_external_semaphore                        VK_KHR_external_semaphore_fd                     VK_KHR_format_feature_flags2
  VK_KHR_fragment_shader_barycentric               VK_KHR_fragment_shading_rate                     VK_KHR_get_memory_requirements2                  VK_KHR_global_priority
  VK_KHR_image_format_list                         VK_KHR_imageless_framebuffer                     VK_KHR_incremental_present                       VK_KHR_index_type_uint8
  VK_KHR_internally_synchronized_queues            VK_KHR_line_rasterization                        VK_KHR_load_store_op_none                        VK_KHR_maintenance1
  VK_KHR_maintenance10                             VK_KHR_maintenance11                             VK_KHR_maintenance2                              VK_KHR_maintenance3
  VK_KHR_maintenance4                              VK_KHR_maintenance5                              VK_KHR_maintenance6                              VK_KHR_maintenance7
  VK_KHR_maintenance8                              VK_KHR_maintenance9                              VK_KHR_map_memory2                               VK_KHR_multiview
  VK_KHR_performance_query                         VK_KHR_pipeline_binary                           VK_KHR_pipeline_executable_properties            VK_KHR_pipeline_library
  VK_KHR_present_id                                VK_KHR_present_id2                               VK_KHR_present_wait                              VK_KHR_present_wait2
  VK_KHR_push_descriptor                           VK_KHR_ray_query                                 VK_KHR_ray_tracing_maintenance1                  VK_KHR_ray_tracing_pipeline
  VK_KHR_ray_tracing_position_fetch                VK_KHR_relaxed_block_layout                      VK_KHR_robustness2                               VK_KHR_sampler_mirror_clamp_to_edge
  VK_KHR_sampler_ycbcr_conversion                  VK_KHR_separate_depth_stencil_layouts            VK_KHR_shader_abort                              VK_KHR_shader_atomic_int64
  VK_KHR_shader_clock                              VK_KHR_shader_constant_data                      VK_KHR_shader_draw_parameters                    VK_KHR_shader_expect_assume
  VK_KHR_shader_float16_int8                       VK_KHR_shader_float_controls                     VK_KHR_shader_float_controls2                    VK_KHR_shader_fma
  VK_KHR_shader_integer_dot_product                VK_KHR_shader_maximal_reconvergence              VK_KHR_shader_non_semantic_info                  VK_KHR_shader_quad_control
  VK_KHR_shader_relaxed_extended_instruction       VK_KHR_shader_subgroup_extended_types            VK_KHR_shader_subgroup_rotate                    VK_KHR_shader_subgroup_uniform_control_flow
  VK_KHR_shader_terminate_invocation               VK_KHR_shader_untyped_pointers                   VK_KHR_spirv_1_4                                 VK_KHR_storage_buffer_storage_class
  VK_KHR_swapchain                                 VK_KHR_swapchain_maintenance1                    VK_KHR_swapchain_mutable_format                  VK_KHR_synchronization2
  VK_KHR_timeline_semaphore                        VK_KHR_uniform_buffer_standard_layout            VK_KHR_variable_pointers                         VK_KHR_vertex_attribute_divisor
  VK_KHR_video_decode_av1                          VK_KHR_video_decode_h264                         VK_KHR_video_decode_h265                         VK_KHR_video_decode_queue
  VK_KHR_video_decode_vp9                          VK_KHR_video_encode_h264                         VK_KHR_video_encode_h265                         VK_KHR_video_encode_intra_refresh
  VK_KHR_video_encode_quantization_map             VK_KHR_video_encode_queue                        VK_KHR_video_maintenance1                        VK_KHR_video_maintenance2
  VK_KHR_video_queue                               VK_KHR_vulkan_memory_model                       VK_KHR_workgroup_memory_explicit_layout          VK_KHR_zero_initialize_workgroup_memory
  VK_MESA_image_alignment_control                  VK_NV_compute_shader_derivatives                 VK_VALVE_mutable_descriptor_type                 VK_VALVE_shader_mixed_float_dot_product
  VK_VALVE_video_encode_rgb_conversion
```


## Appendix B — complete MI50 / Radeon VII device extension list (218)

```
  VK_AMD_buffer_marker                             VK_AMD_device_coherent_memory                    VK_AMD_draw_indirect_count                       VK_AMD_gcn_shader
  VK_AMD_gpu_shader_half_float                     VK_AMD_gpu_shader_int16                          VK_AMD_memory_overallocation_behavior            VK_AMD_mixed_attachment_samples
  VK_AMD_rasterization_order                       VK_AMD_shader_ballot                             VK_AMD_shader_core_properties                    VK_AMD_shader_core_properties2
  VK_AMD_shader_early_and_late_fragment_tests      VK_AMD_shader_explicit_vertex_parameter          VK_AMD_shader_fragment_mask                      VK_AMD_shader_image_load_store_lod
  VK_AMD_shader_trinary_minmax                     VK_AMD_texture_gather_bias_lod                   VK_EXT_4444_formats                              VK_EXT_attachment_feedback_loop_dynamic_state
  VK_EXT_attachment_feedback_loop_layout           VK_EXT_buffer_device_address                     VK_EXT_calibrated_timestamps                     VK_EXT_color_write_enable
  VK_EXT_conditional_rendering                     VK_EXT_conservative_rasterization                VK_EXT_custom_border_color                       VK_EXT_custom_resolve
  VK_EXT_depth_bias_control                        VK_EXT_depth_clamp_control                       VK_EXT_depth_clamp_zero_one                      VK_EXT_depth_clip_control
  VK_EXT_depth_clip_enable                         VK_EXT_depth_range_unrestricted                  VK_EXT_descriptor_buffer                         VK_EXT_descriptor_heap
  VK_EXT_descriptor_indexing                       VK_EXT_device_address_binding_report             VK_EXT_device_fault                              VK_EXT_device_generated_commands
  VK_EXT_device_memory_report                      VK_EXT_discard_rectangles                        VK_EXT_display_control                           VK_EXT_dynamic_rendering_unused_attachments
  VK_EXT_extended_dynamic_state                    VK_EXT_extended_dynamic_state2                   VK_EXT_extended_dynamic_state3                   VK_EXT_external_memory_acquire_unmodified
  VK_EXT_external_memory_dma_buf                   VK_EXT_external_memory_host                      VK_EXT_fragment_shader_interlock                 VK_EXT_global_priority
  VK_EXT_global_priority_query                     VK_EXT_graphics_pipeline_library                 VK_EXT_hdr_metadata                              VK_EXT_host_query_reset
  VK_EXT_image_2d_view_of_3d                       VK_EXT_image_compression_control                 VK_EXT_image_compression_control_swapchain       VK_EXT_image_drm_format_modifier
  VK_EXT_image_robustness                          VK_EXT_image_view_min_lod                        VK_EXT_index_type_uint8                          VK_EXT_inline_uniform_block
  VK_EXT_legacy_vertex_attributes                  VK_EXT_line_rasterization                        VK_EXT_load_store_op_none                        VK_EXT_map_memory_placed
  VK_EXT_memory_budget                             VK_EXT_memory_priority                           VK_EXT_multi_draw                                VK_EXT_mutable_descriptor_type
  VK_EXT_nested_command_buffer                     VK_EXT_non_seamless_cube_map                     VK_EXT_pci_bus_info                              VK_EXT_physical_device_drm
  VK_EXT_pipeline_creation_cache_control           VK_EXT_pipeline_creation_feedback                VK_EXT_pipeline_robustness                       VK_EXT_present_timing
  VK_EXT_primitive_restart_index                   VK_EXT_primitive_topology_list_restart           VK_EXT_primitives_generated_query                VK_EXT_private_data
  VK_EXT_provoking_vertex                          VK_EXT_queue_family_foreign                      VK_EXT_robustness2                               VK_EXT_sample_locations
  VK_EXT_sampler_filter_minmax                     VK_EXT_scalar_block_layout                       VK_EXT_separate_stencil_usage                    VK_EXT_shader_atomic_float
  VK_EXT_shader_atomic_float2                      VK_EXT_shader_demote_to_helper_invocation        VK_EXT_shader_image_atomic_int64                 VK_EXT_shader_module_identifier
  VK_EXT_shader_object                             VK_EXT_shader_replicated_composites              VK_EXT_shader_stencil_export                     VK_EXT_shader_subgroup_ballot
  VK_EXT_shader_subgroup_vote                      VK_EXT_shader_uniform_buffer_unsized_array       VK_EXT_shader_viewport_index_layer               VK_EXT_subgroup_size_control
  VK_EXT_swapchain_maintenance1                    VK_EXT_texel_buffer_alignment                    VK_EXT_tooling_info                              VK_EXT_transform_feedback
  VK_EXT_vertex_attribute_divisor                  VK_EXT_vertex_input_dynamic_state                VK_EXT_ycbcr_2plane_444_formats                  VK_EXT_ycbcr_image_arrays
  VK_EXT_zero_initialize_device_memory             VK_GOOGLE_decorate_string                        VK_GOOGLE_hlsl_functionality1                    VK_GOOGLE_user_type
  VK_INTEL_shader_integer_functions2               VK_KHR_16bit_storage                             VK_KHR_8bit_storage                              VK_KHR_bind_memory2
  VK_KHR_buffer_device_address                     VK_KHR_calibrated_timestamps                     VK_KHR_compute_shader_derivatives                VK_KHR_copy_commands2
  VK_KHR_copy_memory_indirect                      VK_KHR_create_renderpass2                        VK_KHR_dedicated_allocation                      VK_KHR_deferred_host_operations
  VK_KHR_depth_clamp_zero_one                      VK_KHR_depth_stencil_resolve                     VK_KHR_descriptor_update_template                VK_KHR_device_address_commands
  VK_KHR_device_fault                              VK_KHR_device_group                              VK_KHR_draw_indirect_count                       VK_KHR_driver_properties
  VK_KHR_dynamic_rendering                         VK_KHR_dynamic_rendering_local_read              VK_KHR_extended_flags                            VK_KHR_external_fence
  VK_KHR_external_fence_fd                         VK_KHR_external_memory                           VK_KHR_external_memory_fd                        VK_KHR_external_semaphore
  VK_KHR_external_semaphore_fd                     VK_KHR_format_feature_flags2                     VK_KHR_get_memory_requirements2                  VK_KHR_global_priority
  VK_KHR_image_format_list                         VK_KHR_imageless_framebuffer                     VK_KHR_incremental_present                       VK_KHR_index_type_uint8
  VK_KHR_internally_synchronized_queues            VK_KHR_line_rasterization                        VK_KHR_load_store_op_none                        VK_KHR_maintenance1
  VK_KHR_maintenance10                             VK_KHR_maintenance11                             VK_KHR_maintenance2                              VK_KHR_maintenance3
  VK_KHR_maintenance4                              VK_KHR_maintenance5                              VK_KHR_maintenance6                              VK_KHR_maintenance7
  VK_KHR_maintenance8                              VK_KHR_maintenance9                              VK_KHR_map_memory2                               VK_KHR_multiview
  VK_KHR_pipeline_binary                           VK_KHR_pipeline_executable_properties            VK_KHR_pipeline_library                          VK_KHR_present_id
  VK_KHR_present_id2                               VK_KHR_present_wait                              VK_KHR_present_wait2                             VK_KHR_push_descriptor
  VK_KHR_relaxed_block_layout                      VK_KHR_robustness2                               VK_KHR_sampler_mirror_clamp_to_edge              VK_KHR_sampler_ycbcr_conversion
  VK_KHR_separate_depth_stencil_layouts            VK_KHR_shader_abort                              VK_KHR_shader_atomic_int64                       VK_KHR_shader_clock
  VK_KHR_shader_constant_data                      VK_KHR_shader_draw_parameters                    VK_KHR_shader_expect_assume                      VK_KHR_shader_float16_int8
  VK_KHR_shader_float_controls                     VK_KHR_shader_float_controls2                    VK_KHR_shader_fma                                VK_KHR_shader_integer_dot_product
  VK_KHR_shader_maximal_reconvergence              VK_KHR_shader_non_semantic_info                  VK_KHR_shader_quad_control                       VK_KHR_shader_relaxed_extended_instruction
  VK_KHR_shader_subgroup_extended_types            VK_KHR_shader_subgroup_rotate                    VK_KHR_shader_subgroup_uniform_control_flow      VK_KHR_shader_terminate_invocation
  VK_KHR_shader_untyped_pointers                   VK_KHR_spirv_1_4                                 VK_KHR_storage_buffer_storage_class              VK_KHR_swapchain
  VK_KHR_swapchain_maintenance1                    VK_KHR_swapchain_mutable_format                  VK_KHR_synchronization2                          VK_KHR_timeline_semaphore
  VK_KHR_uniform_buffer_standard_layout            VK_KHR_variable_pointers                         VK_KHR_vertex_attribute_divisor                  VK_KHR_vulkan_memory_model
  VK_KHR_workgroup_memory_explicit_layout          VK_KHR_zero_initialize_workgroup_memory          VK_MESA_image_alignment_control                  VK_NV_compute_shader_derivatives
  VK_VALVE_mutable_descriptor_type                 VK_VALVE_shader_mixed_float_dot_product
```

