# 09 — Zero-copy host memory, transfer concurrency, KV quant + ggml-vulkan audit

Round 9. Paired with the implementation round that owns the single-stream lane.
Engine audited: **llama.cpp `ggml-vulkan`** (the engine that beats us on this exact hardware).
Everything below was measured this round unless marked *(cited)*.

Run header (every number in Part A):

```
GPU        : AMD Radeon RX 6950 XT (RADV NAVI21), 0000:03:00.0 — HEADLESS card
sclk       : 2695 MHz, power_dpm_force_performance_level=high  (checked before AND after every run)
PCIe       : 16.0 GT/s x16 (gen4 x16); MI50 8.0 GT/s x16 (gen3 x16) — not touched this round
driver     : mesa 26.2.4, RADV; Vulkan 1.4.354
mlock      : soft 8 MiB / hard 8 MiB for the login user ⇒ pinned arms run under `sudo -n` (root holds CAP_IPC_LOCK)
host       : 58 GB RAM (44 GB available), 32 threads, single NUMA node
harness    : <SCRATCH>/opencode/research-harnesses/r09-zerocopy
             (`r09-zerocopy` ash harness + `c/nvme.c` liburing harness; results/ has the raw matrices)
safety     : every run `timeout 90`, validation on for first runs of each path, ≤4 dispatched
             shapes per process, one process at a time, MI50 never used
```

All Part A runs on the RX only. The MI50 is the display card and was left alone (see §A2.4 for
what the numbers imply for it, which is an inference, not a measurement).

---

## A1. Can the GPU execute host-resident experts? — YES, at 22–26 GB/s

This is the round's headline: **a compute shader reading host RAM runs at the PCIe wire rate.**
The earlier "zero-copy BAR reads measured negative" (capy T22, and our own round 5 at
46.9–64.6 ns/row ≈ 1.4–1.9 GB/s) was a *90-byte-record gather* pathology, not a property of
GPU-initiated host reads. With coalesced 16-byte loads the ceiling is the link, not the path.

### A1.1 Capability (RX 6950 XT, RADV 26.2.4)

| Item | Value |
|---|---|
| `VK_EXT_external_memory_host` | listed **and** enabled |
| `minImportedHostPointerAlignment` | **4096 B** |
| importable memory types for a host pointer | `0x20` → type 5 = `HOST_VISIBLE \| HOST_COHERENT \| HOST_CACHED` (system heap, 29 916 MiB) |
| buffer memory types | `0x7ad` (intersection with `0x20` = type 5) |
| import works on | `mmap(MAP_ANONYMOUS)` + `mlock`, `mmap` unlocked/**pageable**, and driver-allocated `HOST_VISIBLE` memory |
| note | `HOST_VISIBLE \| DEVICE_LOCAL` (types 3/4, 16 368 MiB) is **VRAM** (ReBAR on) — still what round 5 found; the host path is type 5 only |

**Harness trap, worth recording**: `ash`'s `Entry` must outlive the instance. Dropping it after
`create_device` leaves `vkGetDeviceProcAddr` pointing at unloaded code — the first
`vkGetMemoryHostPointerPropertiesEXT` call then segfaults *identically to an unimplemented
extension*. The same code under plain C works, which is how the bug was isolated (`c/hostptr.c`).

### A1.2 Measured throughput

Shader streaming read, 512 MiB buffer, 2048 workgroups × 256 threads, grid-stride `uvec4` XOR
(kernel cost is pure load issue; nothing is computed from the data):

| Source of the bytes | Sustained read | vs PCIe gen4 x16 (32 GB/s theoretical) |
|---|---|---|
| VRAM (device-local) | **585 GB/s** | reference: VRAM is not the constraint |
| **pinned host RAM** (`mlock`, imported) | **28.7–29.7 GB/s** | 90–93 % |
| **pageable host RAM** (driver `HOST_VISIBLE`, not pinned) | **28.7 GB/s** | 90 % |
| **unlocked anon host RAM** (imported, never mlocked) | **29.5 GB/s** (warm; 8.6 GB/s first touch) | 92 % |

Dequant GEMV, m = 1, one expert's gate/up shape (640 rows × 2560 in, 256-thread WGs, 8 rows/WG,
dependent barrier per dispatch, min of 4 rounds). The dequant loop does **not** slow the host path —
it is PCIe-bound, exactly like the copy engine:

| Weight source | layout | bytes | GPU time | GB/s |
|---|---|---|---|---|
| VRAM | f16 (16 B/32) | 3.1 MiB | 23.1 µs | 141.7 |
| VRAM | q8 (5.6 bpw eq.) | 1.8 MiB | 15.5 µs | 118.6 |
| VRAM | q4 (5.0 bpw eq.) | 1.0 MiB | 14.9 µs | 68.6 |
| **pinned host** | f16 | 3.1 MiB | **124.3 µs** | **26.4** |
| **pinned host** | q8 | 1.8 MiB | **72.2 µs** | **25.5** |
| **pinned host** | q4 | 1.0 MiB | **45.7 µs** | **22.4** |
| pageable host | f16 / q8 / q4 | | 131.9 / 71.8 / 45.7 µs | 24.9 / 25.7 / 22.4 |

So the GPU executes host-resident experts at **22.4–26.4 GB/s with dequant in the kernel** —
86–92 % of the 28.7 GB/s the copy engine gets for the same bytes (phase 2 measured 25.8 GB/s
zero-copy DMA, 28.7 GB/s ring DMA; both reproduce here).

### A1.3 The crossover, in the units of the live problem

One token moves **799 MiB** of expert weights (10 of 512 experts × 48 blocks × ~1.66 MB).

| Path | GPU-visible time | Host lane | Notes |
|---|---|---|---|
| today: CPU lane + fence | lane **47.5 ms** exposed, fence 21.9 ms | 47.5 ms of CPU GEMM+page-cache reads | single stream: `lane + GPU ≈ 85 ms` |
| today: stage → DMA → GPU GEMM | 27.8 ms of PCIe (28.7 GB/s) | **+ ~40 ms of host-side copy** into the staging ring (799 MiB at ~20 GB/s) | copy and lane are both on the host |
| **zero-copy: GPU reads host RAM directly** | **32 ms** (799 MiB / 25 GB/s) | **0 ms** — no staging, no CPU GEMM | the 32 ms is PCIe-bound work that can co-schedule with the block's other maths |

**Break-even is at 32 ms of exposed lane; we are at 47.5 ms.** Consequences:

* Even with *zero* overlap, zero-copy is **−15 ms/token** against today's lane, and it deletes the
  staging copy (a further ~40 ms of host time that today competes with the lane for memory bandwidth).
* With the PCIe reads spread across the block's independent work (attention/SSM/shared expert are
  not PCIe-bound), the realistic single-stream target is **55–65 ms/token (15–18 tok/s)** against
  84.9 ms today. The remaining serial part is the *dependency chain*, not the bytes.
* The bytes are the same bytes: zero-copy does not create PCIe traffic, it removes the *second*
  traversal of the same bytes through the host (page cache → pinned staging) and the CPU's maths.
* Pinning is not required for the read to work (pageable measured 28.7 GB/s), but **is required for
  correctness**: an imported pageable range can be migrated/evicted underneath the GPU. The 8 MiB
  `ulimit -l` makes `LimitMEMLOCK=infinity` (Strata already has the drop-in) a hard prerequisite for
  every arm that pins more than a few MiB. Root has `CAP_IPC_LOCK` and mlock'd 1 GiB here without
  touching the limit.
* VRAM at the same shapes is 5.4× faster (141 GB/s) — host-residency is a *capacity* play that also
  happens to remove the lane, not a way to make expert maths faster. Resident blocks stay on the
  card; only the miss/overflow set should be host-resident.

---

## A2. Transfer-queue concurrency — one stream already saturates the link

### A2.1 There is no dedicated transfer family by default

| | default | `RADV_EXPERIMENTAL=transfer_queue` |
|---|---|---|
| queue families | compute = 0, **no TRANSFER-only family** | transfer = **family 3, 2 queues** |
| consequence | copies and compute share one family/queue ⇒ submitted copies serialize with dispatches | copies and compute run concurrently |

Phase 2's "distinct transfer family, 4 queues" was measured **with that experimental flag set**; it
is not the default on this driver. Round 1's "free SDMA family" is free only if the env var is set.
Today's engine therefore cannot overlap its streaming lane's copies with compute unless that flag is
on — a one-env-var prerequisite for every pipelining experiment below.

### A2.2 Sustained rates (256 MiB cell, 5 submits, transfer queue unless noted)

| Cell | GB/s | reading |
|---|---|---|
| H2D 1 stream, compute queue | 21.7–24.6 | the compute queue is the slower copy path (8–18 % penalty) |
| H2D 1 stream, transfer queue | **25.4–26.5** | one stream ≈ link ceiling |
| H2D 2 streams (2 queues) | 25.7–26.5 | **+1 % — no scaling** |
| H2D 4 streams | 25.6–26.6 | +1 % |
| D2H 1 stream | 25.2–26.1 | symmetric |
| D2H 2 streams | 25.3–25.9 | no scaling |
| **H2D ‖ D2H (one stream each)** | **17.9 up + 17.9 down = 35.9 aggregate** | full duplex is real: +40 % aggregate, each direction drops to ~70 % |
| H2D ‖ heavy compute | **21.3–25.6** (74–100 % of alone) | copies barely notice compute |
| compute ‖ H2D | compute **0.99–1.02×** of its solo time | compute does **not** wait for the copy engine |

**Answer**: the RX sustains **one** ~26 GB/s H2D stream; queue depth buys nothing (the link is the
limit), a second *direction* is free (+40 % aggregate) and concurrent compute costs copies ≤26 % while
costing compute ~0. So the streaming lane should use **one deep copy stream, on the transfer family,
with copies issued as early as the dependency allows** — not more streams. Earlier ad-hoc numbers
(e.g. 3.2× compute slowdown under copies) were a broken metric: waiting for device idle measures
`max(compute, copy)`, not compute. The fence-per-submit version above is the honest one.

### A2.3 What this changes for the lane

The lane's problem is not PCIe bandwidth (28.7 GB/s is already used) and not stream count — it is
that the bytes cross the host **twice** (page cache → staging → VRAM) and are *also* GEMV'd on the
CPU. A1 removes both. A2's remaining contribution is that the copy stream and the GPU maths are
genuinely independent, so the *residual* H2D traffic (resident-set refresh, KV) can be hidden.

### A2.4 The MI50 (inference, not measured)

At gen3 x16 (16 GB/s theoretical, phase 2 measured 2.84 GB/s DMA *only because that harness used a
single queue* — that number is a harness artefact, not the link) a zero-copy read arm would land
around 9–13 GB/s. Not measured this round: it is the display card, and a reset there reboots the box
(2026-10-26 incident). Any MI50 zero-copy arm is a separate, gated run after the RX arm passes.

---

## A3. NVMe direct tier — worth it only as a *miss* path, at QD ≥ 8

`c/nvme.c`, liburing, `O_DIRECT`, random offsets over the 47 GB shard (read-only), medians of 3.

| IO size | QD 1 | QD 8 | QD 32 |
|---|---|---|---|
| 4 KiB (PLE record block) | 55 MB/s · 14 k IOPS | 316 MB/s · 81 k IOPS | **1 394 MB/s · 357 k IOPS** |
| 128 KiB | 944 MB/s · 7.6 k | 4 867 MB/s · 39 k | 5 606 MB/s · 45 k |
| 460 KiB (expert gate/up, page-aligned) | 2 041 MB/s · 4.5 k | **5 822 MB/s · 13 k** | 5 656 MB/s · 12.6 k |
| 900 KiB (whole expert) | 2 854 MB/s · 3.2 k | **5 937 MB/s · 6.8 k** | 5 616 MB/s · 6.4 k |
| 460 KiB, **page cache** (buffered) | — | 12.5 GB/s (io_uring copy-bound; 58 GB/s with 32-thread pread *(cited r03)*) | — |

**Measurement trap (new)**: `462 KiB = 473 088 B` is **not** page-aligned (115.5 pages). Read that
way, btrfs silently serves part of the request from cache and the same cell reports
**11.3–13.5 GB/s — 2× the device**. Every O_DIRECT cell must use a page-multiple (460 KiB /
900 KiB above). This invalidated a first pass of these numbers; the table is the corrected one.

Other findings:

* **The knee is QD 8.** QD 1 is 2.0–2.9 GB/s (2.8× worse); QD 32 adds nothing and 900 KiB at QD 32
  is *slower* than QD 8 (5.62 vs 5.94). One io_uring ring with 8–16 entries is the whole recipe.
* **PLE pattern**: 4 KiB records at QD 32 cost **2.9 µs/row** vs 63 µs at QD 1 (round 5's number) —
  a 22× gain from queue depth alone, no layout change. At 48 rows/token that is 0.14 ms/token even
  fully cold.
* **`<STORAGE_ROOT_A>` is a SATA SSD, not NVMe** (`sdb`, ext4): cold 460 KiB 332–390 MB/s, 4 KiB
  62 k IOPS / 243 MB/s. As a Tier-3 expert store it is **15× slower** than the model's own NVMe
  volume. The phase-0 note "Tier-3 belongs on agent-storage-a" is right for *capacity*, wrong for
  *expert streaming*: expert misses are cheaper served from the 990 PRO (5.8 GB/s) than from sdb1
  (0.39 GB/s).
* **`read_ahead_kb` on `btrfs-1` is back to 8192 KiB.** Round 3's 128 KiB fix is not persisted
  (no sysctl/udev rule); it was re-read at 8192 during this round. Any cold-path comparison must
  re-apply it first, or it measures 9× read amplification again.

**Verdict.** For a 71 GB model on 58 GB RAM the working set cannot fit, so a cold tier is
unavoidable — but O_DIRECT is **not** a replacement for the page cache (5.8 GB/s vs 12.5–58 GB/s
warm) and it deletes the hot set by construction. It is worth building as: (a) the *miss* path, with
a QD ≥ 8 ring and page-aligned 460/900 KiB requests; (b) the PLE record path, where 2.9 µs/row at
QD 32 makes the 4 KiB pathology disappear without a layout change. Budget: at 5.8 GB/s a **5 % miss
rate costs 6.9 ms/token**; 10 % costs 13.8 ms; a full miss costs 137 ms. So the direct tier only pays
*after* the LRU cache has bounded misses — it does not remove the need for the LRU.

---

## A4. Tiered KV quantisation — the price and the slots

KV geometry for this model: 12 full-attention blocks (`compress_ratio` 4 at indices 3,7,…,47) ×
2 KV heads × key/value 256 → **24 KiB per token per slot** at f16. The 36 SSM blocks carry recurrent
state, not KV.

| context | f16 | q8_0 (8.5 bpw) | q4_0 (4.5 bpw) |
|---|---|---|---|
| 32 K | **786 MiB** | 419 MiB | 222 MiB |
| 64 K | **1.5 GiB** | 805 MiB | 426 MiB |

### A4.1 Time (measured, gfx1030)

Dot-shaped streaming kernel over the real K/V shape (32 768 rows × 512 elements per layer, dependent
barrier per pass, VRAM source, min of 4):

| dtype | bytes/pass | time/pass | GB/s |
|---|---|---|---|
| f16 | 32.0 MiB | 486.8 µs | 68.9 |
| q8_0 (padded to 9 B/32) | 18.0 MiB | 157.4 µs | 119.9 |
| q4_0 (padded to 5 B/32) | 10.0 MiB | 88.9 µs | 118.0 |

Full attention per token = K pass + V pass × 12 layers:

| dtype | measured (this kernel shape) | byte floor at 585 GB/s (VRAM streaming) |
|---|---|---|
| f16 | **11.7 ms/token** | 1.4 ms |
| q8_0 | **3.8 ms** | 0.7 ms |
| q4_0 | **2.1 ms** | 0.4 ms |

So KV quantisation is worth **3–10 ms/token (4–12 % of a token)** depending on how close attention
gets to the byte floor — the 8.4× gap on the f16 row is our kernel (512-wide rows, software
half-decode), not physics. The *comparative* price is what gates the decision: q8 is 0.32× and q4
0.18× of f16's attention time.

### A4.2 Quality (numeric simulation, head_dim 256, 4 096 keys, heavy-tailed row scale)

| variant | dot-product rel. error | mean cos | value-output rel. error |
|---|---|---|---|
| f16 (reference) | 2.1e-4 | 1.000000 | — |
| **q8_0** | **5.4e-3** | 0.999986 | — |
| **q4_0** | **9.3e-2** | 0.995620 | 9.2e-2 |
| q4_0 + Hadamard rotation | 9.5e-2 | 0.995601 | 9.0e-2 |

q8 is essentially free (0.5 % error). q4 is 9 % per-dot — survivable but must be gated on a real
end-to-end check, not this proxy. **Rotation bought nothing here**, because the synthetic data is
isotropic: rotation's published benefit is on *outlier channels*, and whether this model's KV has
them is a measurement we have not made (it needs real activations, not a simulator). Treat
"4-bit + rotation is near-lossless" as **unverified for this model**.

### A4.3 What it buys

At 64 K, f16 needs 1.5 GiB per slot. The RX has 16 GB minus the resident expert set, so f16 admits
**1–2** concurrent 64 K slots; q8 admits 3–4; q4 (or 4-bit rotated) admits **5–7**. That is exactly
the "gates 4–6 agents" claim, and it is a **capacity** argument, not a speed argument — the 3–10 ms
saving is a bonus, the slots are the point. Recommendation: **q8_0 KV as the default** (0.5 % error,
0.32× time, 1.9× slots), q4_0 only behind a quality gate and only for long-context slots.

---

## A5. Other Vulkan surface for a single stream

Ranked by value to the exposed lane, all priced this round or already settled:

1. **`VK_EXT_external_memory_host` + host-resident experts** (§A1) — removes the lane *and* the
   staging copy. Highest value by a wide margin. Prerequisite: `LimitMEMLOCK`.
2. **`RADV_EXPERIMENTAL=transfer_queue`** (§A2.1) — the only way to get a second queue family at all
   on this driver; without it all pipelining collapses to one serial queue.
3. **Issue copies as early as possible in the same submission**, since a queued submit costs
   8.6 µs wall / +8.8 µs GPU *(cited r08)* and a fence-waited one 33 µs — 48 blocks × 2 = ~1.6 ms/token
   of submission overhead if every block waits, vs ~0.4 ms if copies are pipelined. Small next to 32 ms,
   but free.
4. **Multi-queue compute submission: not available.** Family 0 has **one** queue; there is no async
   compute queue to put the CPU lane's consumer on. "Run the CPU lane as another queue's consumer"
   has no Vulkan representation here — the lane must be removed (A1) or overlapped on the host side.
5. **`VK_KHR_shader_subgroup_*`** — the RX's default subgroup is 64 and forcing 32 costs 5–8 %
   *(cited r06)*. Nothing in the subgroup surface moves a PCIe-bound kernel.
6. **`VK_KHR_performance_query`** still segfaults *(cited r07)*; timestamps remain the only clock.
   This round re-confirms the recipe: `TOP_OF_PIPE` on the first stamp and `BOTTOM_OF_PIPE` on the
   second is mandatory — two `BOTTOM_OF_PIPE` stamps measure zero because both execute at the end of
   the dispatch. That single mistake produced three bogus "0.040 µs" tables before it was caught.

---

## Part B — `ggml-vulkan` audit (llama.cpp `7aca0d2`, 2026-10-26)

Clone: `Reference Projects/llama.cpp-ggml-vulkan` — shallow clone **succeeded** (`7aca0d2ea8d9`,
2026-10-26). `ggml/src/ggml-vulkan/` is 16 778 lines of `ggml-vulkan.cpp` + 188 shaders, with the
buffer/alloc split into `ggml-vulkan-buffers.cpp` (866 lines).

### B1. Buffer policy: they ship "tensors in system RAM, computed on the GPU"

`ggml_vk_create_buffer_device()` (`ggml-vulkan-buffers.cpp:220-252`) picks memory property flags from
four policy switches:

| env var | effect |
|---|---|
| `GGML_VK_PREFER_HOST_MEMORY` | device buffers are allocated **`HOST_VISIBLE\|HOST_COHERENT`** (system RAM) first, `DEVICE_LOCAL` as fallback — shaders then read/write host memory directly |
| `GGML_VK_ALLOW_SYSMEM_FALLBACK` | allow the sysmem fallback when VRAM is exhausted |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` | refuse ReBAR VRAM for host-visible buffers |
| default | prefer `DEVICE_LOCAL\|HOST_VISIBLE\|HOST_COHERENT` (ReBAR VRAM), fall back to pure device-local |

This is the same capability §A1 prices at 22–26 GB/s, shipped as a supported mode. It matters
because it proves the *pattern* (import/allocate in system RAM, dispatch against it, no staging,
no CPU compute) is production-viable on RADV — it is not a research idea.

**But be precise about what they use for MoE offload**: `-cmoe`/`--n-cpu-moe`/`-ot` route the expert
tensors to the **CPU buffer type** (`src/llama-model.cpp` tensor-buft overrides, per our
`docs/specs/fork-flags-semantics.md`), i.e. llama.cpp **GEMVs offloaded experts on the CPU**, exactly
like our lane. Their Vulkan-from-sysmem path is a UMA/fallback mechanism, not their MoE strategy.
So: the capability is validated by them; the *use* we want is ours to invent. That asymmetry is the
single biggest opportunity in this audit.

### B2. Where they merge submissions, and how they keep the GPU busy

* `device->max_nodes_per_submit = 100` (`:4287`), overridable by `GGML_VK_MAX_NODES_PER_SUBMIT` —
  they submit at least every 100 graph nodes regardless of dependencies, i.e. many small submits,
  which r08 prices as cheap (8.6 µs queued).
* `submit_pending` / `almost_ready_fence` (`:12967`, `:13515`, `:13564-13591`) — a *pipelined*
  submission: a command buffer is submitted as soon as its pipelines are ready, while later nodes are
  still being recorded, and an "almost ready" fence lets the host proceed before the GPU finishes.
* `ggml_vk_graph_optimize()` (`:14865`) — a **whole-graph reordering pass** run before execution:
  it moves independent nodes as early as their inputs allow (with a full view/aliasing hazard check,
  `is_src_of`/`base`), while *pinning* recognised fusion patterns so they are not disturbed. This is
  r04's "prefetch-before-wait" and r08's "submit → host work → wait" interleave implemented as a
  compiler pass instead of a scheduler hack.
* Op fusion: `ctx->num_additional_fused_ops` — a mat-vec dispatch consumes the **next one or two
  nodes** (bias/scale) via push-constant `MAT_VEC_FUSION_FLAGS_BIAS0/1` (`:6935-6964`), with the
  output written to the *later* node's tensor. `add_rms_fusion` does the same for add+RMSNorm.
* Descriptor reuse (`GGML_VK_DISABLE_DESCRIPTOR_REUSE`) and per-device command-buffer recycling
  (`:1120-1130`, `:16255`) — the same conclusion r08 reached from the other direction (replay is
  cheaper than re-record).

### B3. Small-batch quantised matmul: two families, chosen per shape

* `mul_mat_vec_q` (MMVQ) for m = 1–few: one row per workgroup, `row_split` = 1 or 4 chosen from
  `n_rows`/`hsk`, `d_split` bounded by subgroup size, with `row_split` used *specifically* so
  synchronisation stays inside a subgroup and "avoids barriers" (`:1215-1222`) — an independent
  confirmation of r06's barrier-footprint finding.
* `mul_mmq` (MMQ) for larger m, with **`BK_STEP = 1` for `MUL_MAT_ID`** vs 4 otherwise (`:1530`) —
  i.e. the MoE-indexed matmul gets its own tile schedule, and there are separate warptile tables for
  `mmq`, `mmq_int`, `mmq_cm1`, `mmq_cm1_int`, `mmqid`, `mmqid_int`, plus per-size-class (`l/m/s`)
  variants (`:1700-1751`). 15 references to `GGML_OP_MUL_MAT_ID`: MoE is a first-class op with its
  own kernels, not a loop of small matmuls.
* `GGML_VK_DISABLE_MMVQ` / `GGML_VK_FORCE_MMVQ` (`:4941`) — the m=1 vs m>1 crossover is treated as
  a *tunable heuristic per device*, which is the property we lack (we have one geometry).

### B4. KV quantisation + flash attention

`ggml_vk_fa_scalar_uses_mmq(device, k_type, v_type)` (`:1659`) selects an **integer-dot FA path for
`Q4_0/Q4_1/Q5_0/Q5_1/Q8_0` keys**, gated on `integer_dot_product && subgroup_clustered`, and the FA
pipeline state carries `(k_type, v_type)` (`:1391`) so dequantisation happens **inside** the flash
attention kernel — no separate dequant dispatch, no materialised f16 KV. That is exactly the shape
§A4 says we want, and it means `-ctk q8_0 -ctv q8_0` is a *kernel-level* feature there, not a
host-side conversion.

### B5. `--split-mode row` on Vulkan

`make_gpu_buft_list()` (`src/llama-model.cpp:1211-1240`) asks the backend for
`ggml_backend_split_buffer_type(dev_index, tensor_split)`; row-split is therefore a **buffer type**
that interleaves one tensor's rows across devices, letting a single matmul span both cards. If the
backend does not provide it, llama.cpp throws "device does not support split buffers". Note this is
unrelated to the FA `row_split` in §B3 (same word, different mechanism). For us: a row-split buffer
type is the only way one expert's GEMV could use both cards, and it needs a cross-device reduction
(the partials must come back) — with 28.7 GB/s of PCIe each way, a reduction of one 640-wide
partial per expert is ~nothing, but *every* expert would pay it. Per-expert placement (what we do)
avoids that cost; row-split is only interesting for the *dense/shared* tensors.

### B6. Scheduling for `-np` slots

llama.cpp's continuous batching lives above the backend: all `-np` sequences are packed into **one**
`llama_batch` per step and executed as **one graph**, so the backend sees a single submission with
M = n_slots. That is how it gets cross-stream overlap without any stream machinery — the opposite of
our "one graph per stream". Our measured cross-stream hiding (`lane 47.5 ms` hidden only
*cross-stream*) is the manual version of the same thing; the structural version is one graph per
step with all streams in the batch dimension.

### B7. Ranked adaptations, by value to us

1. **Host-resident expert execution for the miss set (§A1 + B1).** 22–26 GB/s measured; removes the
   47.5 ms lane and the staging copy. Everything else in this list is a few ms. *Measurable sketch*:
   import each expert block's page-cache range once (`minImportedHostPointerAlignment` 4096, mlock'd),
   GEMV directly out of it, flag-gated, parity on a 64-token prefix, expect 85 → 55–65 ms/token.
2. **A graph-level reorder pass + trailing-op fusion (§B2).** r08's 10× interleave result
   (905 → 811 µs/block) becomes a compiler pass: emit independent nodes early (attention of block L
   beside the expert GEMV of block L−1) and fold the trailing bias/scale into the mat-vec via
   push-constant flags instead of a dispatch. Sketch: 1 509 dispatches → ~1 100 with the same maths;
   measure in *submits* (r08), not dispatches.
3. **KV quantisation with dequant-inside-attention (§A4 + B4).** q8_0 default: 0.5 % dot error,
   0.32× attention time, 1.9× slots; q4_0 for long-context slots behind a quality gate (9 % dot
   error — needs a real-activation check, and rotation is *unverified* here). Sketch: add q8/q4 KV
   cells + FA-style dequant to the attention kernel; gate on 64 K attention time and slot count.
4. **A `MUL_MAT_ID`-style expert dispatch (§B3).** One dispatch over an expert-indexed weight array
   with its own tile schedule (`BK_STEP = 1`), instead of per-expert launches. Composes with item 1
   (the array can live in host memory) and with r04's fused-MoE item.
5. **Per-shape pipeline selection with `m = 1` vs `m > 1` crossover as a tunable (§B3, §B2).**
   The MMVQ/MMQ split is a *device-level heuristic*; we have one geometry for both regimes.
6. **One graph per step with all streams in the batch dimension (§B6).** Structural fix for the
   per-stream graph cost; bigger change, lower urgency than 1–3.
7. **Row-split buffer type (§B5)** — only for dense tensors; per-expert placement already avoids the
   reduction cost.
8. **Not applicable / already settled**: cooperative matrix (absent on both cards, r06),
   `vkCmdDispatchIndirect` over pre-recorded CBs (r03-r04 refuted), their `GGML_VK_ASYNC` +
   transfer-queue preference (we measured the same 25–26 GB/s), their sync loggers (we have
   timestamps).

### B8. Portability flags

* The integer-dot FA path needs `integer_dot_product && subgroup_clustered` (`:1661`) — worth a
  capability probe on gfx1030 before promising q4/q8 KV in-kernel dequant.
* `coopmat`/`coopmat2`/`OCP_FP4`/`bfloat16` warp paths are all compiled per-device and gated;
  on gfx1030 they are off (no cooperative matrix, r06).
* Nothing in the top 3 adaptations is ROCm/HIP-specific or CPU-GEMM-only; item 1 is RADV-specific
  only in that it needs `VK_EXT_external_memory_host` (present on both our cards, phase 2 verified).

---

## Top-3 experiments for the next round

1. **Zero-copy host-resident experts, end-to-end, one flag.** Import the miss set's page-cache
   ranges (`VK_EXT_external_memory_host`, mlock, `LimitMEMLOCK=infinity`), GEMV them on the GPU, drop
   the CPU lane and the staging copy for those experts. Parity gate: bit-identical logits on a
   64-token prefix vs the device-resident path **at the same weights** (import must be a pure address
   change). Report ms/token single-stream + 2-stream, plus per-block lane time. Expected 85 → 55–65 ms
   single stream; a negative result would show up as host-read GEMV falling below 15 GB/s under real
   concurrency, which §A1 says should not happen (22–26 GB/s with the copy engine idle).
2. **Transfer-queue pipelining with the copies issued early** (`RADV_EXPERIMENTAL=transfer_queue`,
   dedicated transfer family, copies for block L+1 submitted before the wait on block L). A/B/A in one
   pass, measured in *submits*. Ceiling from §A2: 25.6 GB/s beside compute at 1.00× compute cost, so
   any residual H2D traffic is free once it is off the serial path. Independent of item 1 (different
   subsystem: submission vs memory source) — safe to group.
3. **KV quantisation cell: q8_0 (and q4_0 behind a gate), dequant inside attention.** Measure 32 K and
   64 K attention time/token and slot count, gate q4 on a real 4 K-prefix comparison (not the
   isotropic simulator in §A4.2). Expected 3.8 ms/token (q8) and 1.9× slots; this is the item that
   decides whether 4–6 agents fit.

Housekeeping that must land with the above: re-apply `read_ahead_kb=128` to `btrfs-1` **persistently**
(it is 8192 again), and keep every O_DIRECT request a page multiple (§A3 trap).

---

### Sources / raw data

* Harness: `<SCRATCH>/opencode/research-harnesses/r09-zerocopy/`
  (`src/main.rs`, `shaders/read.comp`, `shaders/gemv.comp`, `c/hostptr.c`, `c/nvme.c`,
  `results/nvme.txt`, `results/nvme_aligned.txt`, `kvsim.py`).
* Repro, capability probe: `timeout 60 ./target/release/r09-zerocopy caps`
* Repro, zero-copy read: `timeout 90 sudo -n env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ./target/release/r09-zerocopy read 512 mlock` (arms: `dev`, `mlock`, `drv`, `page`)
* Repro, expert GEMV from host: `... gemv 640 80 1 mlock 16`
* Repro, transfer concurrency: `timeout 120 sudo -n env RADV_EXPERIMENTAL=transfer_queue \
  ./target/release/r09-zerocopy h2d 256`
* Repro, NVMe: `./c/nvme bench <shard> 471040 8 2000` (page-aligned sizes only)
* Engine clone: `Reference Projects/llama.cpp-ggml-vulkan` @ `7aca0d2ea8d92b56911233b822db01f9fbb1ae50`
