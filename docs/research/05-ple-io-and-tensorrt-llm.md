# 05 — The PLE gather, priced (Vulkan + IO) — plus a TensorRT-LLM audit

Round 5. Read-only research: no engine code was changed. Two questions:

* **Part A** — our PLE lookup is a gather of **90-byte records out of a 28.8 GiB tensor**
  (320 001 536 records × 90 B, `per_layer_token_embd.weight`). Batching it is the next
  implementation item, so this round prices the *candidate mechanisms* on the actual cards and
  the actual filesystem, and says which one to build.
* **Part B** — audit TensorRT-LLM (not covered before): iteration-level scheduling, KV block
  management, launch amortisation, speculative orchestration, and what is CUDA-only.

Environment for every measurement: kernel `6.18.55-2-cachyos-lts`; RADV **Mesa 26.2.4**,
device API 1.4.354; cards `0000:0e:00.0` (Radeon VII / MI50, gfx906) and `0000:03:00.0`
(RX 6950 XT, gfx1030); `qwen-dual`, `strata-rx`, `d1-mi50` **inactive before and after** (verified
`systemctl is-active`); no bench ran while another GPU process was active; every run wrapped in
`timeout 90`/`timeout 120`; MI50 first, RX only after MI50 came back clean. `read_ahead_kb` is
**128** on `/sys/class/bdi/btrfs-1/` (round 3's fix is still in place — this is the new baseline).

---

# Part A — what a PLE gather actually costs

## A.0 The answer first

1. **`vkCmdCopyBuffer2` region arrays are the wrong tool**: **~300–360 ns per region on the MI50,
   110–140 ns on the RX, almost independent of region size**. For 6 144 rows that is
   **2 316 µs (MI50) / 857 µs (RX)**. A compute shader does the same work in **45.7 µs / 18.5 µs**
   at the byte-exact 90-byte layout, and **7.7 µs / 2.5 µs** if records are padded to 96 B.
2. **The gather cost is dominated by load-instruction count, not bytes.** Byte-exact 90 B (45 ×
   `uint16` loads per record) = **7.4 ns/row (MI50) / 2.3 ns/row (RX)**; the same data padded to
   **96 B and read as 6 × `uvec4`** = **1.0 / 0.34 ns/row** — a **7.4× / 6.8×** win for **6.7 %**
   more bytes. Padding is the single cheapest change available.
3. **Where the source lives decides everything.** VRAM-resident gather: 1.0–3.0 ns/row. Same
   gather with the source in **system RAM over PCIe**: **46.9 ns/row (MI50) / 64.6 ns/row (RX)**.
   ReBAR-allocated buffers are *VRAM*, not host memory — they measure like VRAM (this is the trap
   in the obvious experiment; see A.4).
4. **The 48× btrfs over-read is one 4 KiB block, not a compressed extent.** FIEMAP: **2 213
   extents of 8–16 MiB, 1:1 logical→physical (uncompressed)**; a random 90-B `pread` pulls
   **4.07 KiB of device traffic** and costs **63 µs** at QD1. Nothing at the syscall/filesystem
   layer can fix a 90-byte stride — but **co-locating a token's 48 rows (4 320 B contiguous)**
   takes the same read to **8.18 KiB for 48 rows (1.9×, not 46×)** and cuts cold PLE for a
   128-token chunk from **15.6 ms to 0.45 ms (34×)**.
5. **Best overall**: repack PLE **token-major, 16-byte-padded** (48 × 96 B = 4 608 B per token),
   gather with a `uvec4` compute shader, keep a VRAM hot set for the active vocabulary, and serve
   misses with an 8–32-thread host gather of 4 608-B records + one H2D. That is the design to
   build; everything else in this part is the evidence for it.

## A.1 Method (and one measurement that had to be thrown away)

`harnesses`-style scratch probe, `<SCRATCH>/opencode/p5ple/vkgather/`
(reuses `vk-probe`'s `ctx.rs` plumbing verbatim: instance/device setup, buffers, descriptor sets,
`vkCmdWriteTimestamp` bracketing). Two shaders (`gather_b16.comp`, `gather_v4.comp`,
`gather_b16ilp4.comp`), plus the copy-engine arms.

* **Source tensor**: 2 Mi records (180 MiB at 90 B / 192 MiB at 96 B), pseudo-random bytes
  (xorshift; incompressible so nothing flatters the numbers), device-local.
* **Timing**: GPU timestamps, `iters` scaled to a ≥30 ms window, `per_iter` = one gather of N rows.
  Payload = N × record size (90 B or 96 B).
* **The trap I fell into first, and the fix**: the initial version replayed *one* index set for
  every iteration. A 1 024-row gather touches ~128 KiB of cache lines, which fits L2, so the probe
  measured cache bandwidth. The shipped probe carries **32 rotated index sets** and each timed
  iteration uses a different one, so successive gathers touch different rows of the 180 MiB tensor.
  (Conclusions did not change — `b16` and `v4` came out at the same ns/row — but the numbers are
  now DRAM numbers, and `v4` at N=6 144 did get 20 % slower once rotation was in, which is the
  signature of a real memory path rather than a cached one.)
* Validation: MI50 pass ran with `VK_LAYER_KHRONOS_validation` → **0 validation errors**
  (after fixing two harness bugs: a missing `TRANSFER_DST` usage bit and an overlapping-destination
  region in the coalesce arm). RX pass ran clean without layers.
* IO probe: `p2io.c` — random `pread` with `posix_fadvise(DONTNEED)` per arm (drops *that file's*
  clean pages only; no global `drop_caches`, no root), device traffic from
  `/sys/block/nvme0n1/stat` sector deltas, 1/8/32 threads. One 4-arm batch was discarded because a
  1 GiB `dd` was running in the parallel shell and contaminated the disk counters (it showed
  136 KiB/read for a 4 320-B read; the clean repeat shows 8.18 KiB, stable over 3 runs).

## A.2 The gather table (measured)

Device-local source, warm in VRAM, **µs per gather** (ns/row and payload GB/s at the widest N):

### RX 6950 XT (`0000:03:00.0`)

| path | N=1 | 16 | 64 | 256 | 1024 | 6144 | ns/row @6144 | GB/s payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `copy2` regions 90 B | 0.39 | 6.09 | 11.24 | 19.52 | 116.68 | **857.40** | 139.6 | 0.64 |
| `copy1` regions 90 B (legacy call) | – | – | – | – | 111.35 | – | 108.7 | 0.83 |
| compute `b16` (45 × u16, 90 B) | 0.02 | 0.05 | 0.11 | 0.52 | 2.36 | **18.46** | 3.0 | 30.0 |
| compute `b16` ×4 ILP, 90 B | – | – | – | – | 3.00 | 20.00 | 3.3 | 27.7 |
| **compute `v4` (6 × uvec4, 96 B)** | 0.01 | 0.01 | 0.01 | 0.07 | 0.34 | **2.52** | **0.41** | **234** |
| `copy2` regions 96 B | – | – | – | – | 110.68 | 665.86 | 108.4 | 0.89 |
| `copy2` coalesced (merged runs) | – | – | – | – | 98.64 | 609.35 | 99.2 | 0.91 |
| `copy2` token-major 4 320 B (48 rows) | 0.95 | – | 3.81 | 15.24 | – | – | 59.5 | 72.6 |
| `copy2` token-major 14 400 B (160 rows) | 1.50 | – | 5.97 | 23.90 | – | – | 93.4 | 154 |

### MI50 / Radeon VII (`0000:0e:00.0`)

| path | N=1 | 16 | 64 | 256 | 1024 | 6144 | ns/row @6144 | GB/s payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `copy2` regions 90 B | 0.30 | 5.27 | 19.22 | 77.12 | 369.35 | **2316.03** | 377.0 | 0.24 |
| `copy1` regions 90 B (legacy call) | – | – | – | – | 368.68 | – | 360.0 | 0.25 |
| compute `b16` (45 × u16, 90 B) | 0.08 | 0.12 | 0.50 | 1.93 | 7.62 | **45.67** | 7.4 | 12.1 |
| compute `b16` ×4 ILP, 90 B | – | – | – | – | 8.75 | 49.49 | 8.1 | 11.2 |
| **compute `v4` (6 × uvec4, 96 B)** | 0.08 | 0.07 | 0.07 | 0.26 | 1.03 | **7.66** | **1.2** | **77** |
| `copy2` regions 96 B | – | – | – | – | 360.49 | 2168.63 | 353.0 | 0.27 |
| `copy2` coalesced (merged runs) | – | – | – | – | 342.97 | 2035.97 | 331.4 | 0.27 |
| `copy2` token-major 4 320 B | 1.67 | – | 6.99 | 26.59 | – | – | 104 | 41.6 |
| `copy2` token-major 14 400 B | 1.67 | – | 6.65 | 26.62 | – | – | 104 | 138 |

Reading the table:

* **The copy engine costs ~100 ns per region as a floor** (14 400 B regions hit 104 ns on both
  cards, i.e. bandwidth-bound), **plus ~200 ns of dependent latency per small region** on the MI50
  (360 vs 110 ns between the two cards is a latency difference, not a bandwidth one). Region count,
  not region size, is what you pay for — the arrival rate is ~2.8–9 M regions/s at best.
* **Coalescing is worthless for PLE indices.** 1 024 random rows over 7.1 M blocks merge into
  **1 021 regions** (6 144 → 6 121): random 90-byte rows essentially never share a 4 KiB block, so
  the two-stage "sort, merge, copy" idea buys 3 % while adding a sort.
* **Token-major is where the copy engine becomes respectable**: one 4 320-B region per token is
  59.5 ns (RX) / 104 ns (MI50) — so a 128-token chunk of PLE costs **3.8 µs (RX)** by copy or
  **~1 µs** by compute shader. Neither matters once the *bytes are local*; the point is that after
  a token-major repack the copy engine is no longer pathological.

## A.3 Why the 90-byte case is slow, and why 96 is enough

The mechanism is per-record *load-instruction count*, not bytes fetched. All 45 `uint16` loads of
one record are contiguous, but inside a warp the 32 threads are pointing at 32 different records,
so every one of the 45 instructions generates 32 scattered accesses — 1 440 line requests per
warp for ~64 unique lines. The 16-byte-aligned variant needs **6** instructions per record, and
the measured ratio lands exactly on the instruction ratio (45/6 = 7.5 vs 7.4× measured on MI50).

Consequences:

* **Two-byte loads are 2-byte *aligned*** (90 = 2 mod 4, so `uint16` is always legal, `uint32` is
  not) — this is why the byte-exact path has to use u16 at all. There is no legal 4-byte or
  16-byte load at a 90-byte stride, on any Vulkan driver.
* **Padding to 96 B** costs **+6.7 % storage** (28.8 → 30.7 GiB) and buys 7×. Padding to 128 B
  (one full cache line per record) would cost +42 % and is **not** needed — the win is alignment,
  not line-exclusivity.
* **ILP does not help**: 4 records interleaved per thread is *slower* (8.1 vs 7.4 ns on MI50),
  because it cuts the thread count by 4 on a path that is already request-rate-bound. Do not build
  a wider-ILP gather; pad the layout.

## A.4 Where the source lives (the arm everyone conflates)

Same `b16` gather, same 180 MiB source, three memory types on each card:

| source memory | MI50 ns/row (N=1024) | RX ns/row (N=1024) |
|---|---:|---:|
| `DEVICE_LOCAL` (VRAM) | 7.4 | 2.3 |
| `HOST_VISIBLE \| DEVICE_LOCAL` ("ReBAR") | 7.4 | 2.3 |
| `HOST_VISIBLE` (system RAM, real PCIe) | **46.9** | **64.6** |
| `copy2` 90 B regions from system RAM | 950 (N=1024) | 445 (N=1024) |

**ReBAR is VRAM.** A `HOST_VISIBLE|DEVICE_LOCAL` allocation on both cards is device memory mapped
into the host BAR; GPU reads of it are VRAM reads. Only the plain `HOST_VISIBLE` type is system
RAM (the 29 GiB heap[0]), and there the gather is **6–28× slower** and the copy engine
**~3× slower** again.

That third row is the important comparison for the PLE design, because it is the "GPU gathers
straight out of the page cache" proposal:

* GPU gather over PCIe: **46.9 ns/row (MI50)**, **64.6 ns/row (RX)**.
* Host gather, 32 threads, warm page cache: **~16 ns/row** (0.5 µs single-thread ÷ 32; and the
  measured 2.53 µs/row cold at 32 threads is the cold case).
* Host gather, 32 threads, *cold*: 2.53 µs/row — i.e. 40–50× worse than either.

So: **a GPU gather only wins once the rows are in VRAM.** Moving the gather onto the GPU while the
data is still in host RAM is a 3–4× regression against a warm host gather. The H2D of the gathered
block is the cheap part: 553 KiB = **8.1 µs (RX)** / 4.4 µs pipelined (MI50); 3.6 MiB = 130 µs
(28.3 GB/s, matching round 1's PCIe number for the RX) and 1 365 µs on the MI50 (gen3x4, 2.7 GB/s
— one more reason fine-grained PLE work belongs on the RX lane).

## A.5 The IO side: where the 48× actually comes from

FIEMAP on the real PLE shard (read-only; nothing was written to `/srv`):

```
$ filefrag -v .../Qwen3.8-...-00002-of-00002.gguf | head
   0:  0..4095:      299217264..299221359:  4096:            <- 16 MiB, physical == logical
   1:  4096..6143:   352749696..352751743:  2048:  299221360  <- 8 MiB,  physical == logical
...
filefrag: 2213 extents found
```

**2 213 extents averaging 13 MB, logical length == physical length ⇒ the tensor is *not*
compressed** (and a buffered copy of a 1 GiB slice through `compress=zstd:1` also stayed 1:1 —
this data does not compress, so compression is not a variable here at all).

Random-read measurement on that file, 2 GiB span, page cache dropped per arm:

| read size | µs/read (1 thread) | device KiB/read | amplification |
|---|---:|---:|---:|
| **90 B** (one PLE row) | **63.4** | **4.07** | **46.3×** |
| 4 KiB (aligned) | 70.0 | 7.98 | 2.0× |
| 8 KiB | 75.6 | 12.62 | 1.6× |
| 64 KiB | 122.7 | 67.95 | 1.1× |
| 128 KiB | 172.0 | 127.77 | 1.0× |
| **4 320 B** (48 rows, token-major) | **70.0** | **8.18** | **1.9×** |

So the "48×" is **one block (4 KiB) plus a little readahead**, which is the minimum a block device
can do for an isolated 90-byte read at a random offset. It is *not* a compressed extent, and it is
not fixable by any advice, flag or prefetcher:

* **FIEMAP-informed coalescing is a dead end.** The extents are 13 MB, so 6 144 random rows land
  in ~2 213 extents (≈2.8 rows per extent). Reading the *mapped extent* would mean reading 13 MB to
  use 2.8 rows (≈380× amortised amplification). Coalescing at block granularity is what the kernel
  already does, and it merges 3 of 1 024 regions (A.2). There is nothing to coalesce.
* **The marginal cost of a bigger read is tiny**: 90 B → 128 KiB is 1400× the bytes for 2.6× the
  time. The lever is *useful rows per read*, which is a **layout** decision, not an IO one.

## A.6 The repack, priced

Same file, `pread` of 4 320 B at a random offset (i.e. exactly the access pattern a token-major
`48 × 90 B` record would produce), threads varied:

| threads | 90-B rows (now) | 4 320-B token records (repacked) |
|---|---:|---:|
| 1 | 63.4 µs → 15.8 k rows/s | 70.0 µs → 14.3 k rec/s |
| 8 | 8.53 µs → 117 k rows/s | 10.1 µs → 98.8 k rec/s |
| 32 | **2.53 µs → 395 k rows/s** | **3.51 µs → 285 k rec/s** |

Per 128-token chunk (128 tokens × 48 layers = 6 144 rows):

| | reads | time @32 threads | device bytes |
|---|---:|---:|---:|
| now (90-B rows) | 6 144 | **15.6 ms** | 24.9 MiB |
| repacked (4 320-B tokens) | 128 | **0.45 ms** | 1.05 MiB |

**34× faster, 24× fewer bytes.** Sanity check against the engine: round 3 recorded the chunked PLE
path at **14.7 ms/token-chunk** and attributed it to a single-threaded cold gather of 6 144 rows;
the model here predicts 15.6 ms at 32 threads and ~393 ms single-threaded, so the current engine is
already using parallelism — it is simply issuing 48× more reads than it needs to.

Note the thread scaling: 8 threads reach **62 %** of the 32-thread rate, so the background gather
should use 8–16 threads and leave the rest of the machine to the host lane.

## A.7 Vulkan feature check for gather staging

| feature | MI50 | RX | verdict |
|---|---|---|---|
| `VK_KHR_copy_commands2` (core 1.3, no feature bit) | present | present | **measured worst option** for <1 KiB regions (A.2); fine ≥4 KiB |
| `VK_KHR_map_memory2` | present | present | irrelevant: it only removes a `pNext` chain; the cost is the map itself |
| `VK_EXT_host_image_copy` | **absent** | present | no use: buffers, not images, and it would be slower than a compute gather |
| `VK_EXT_map_memory_placed` | present | present | the nearest analogue to TRT-LLM's VMM paging (Part B.3) — relevant later, not to PLE |
| `VK_EXT_memory_budget` | present | present | useful: the RX reports an 8.4 GiB budget on a 16 GiB heap, so a PLE hot set must be sized from `heap_budget`, not `heap_size` |

Map/unmap churn on a host-visible buffer (1024 × 4 KiB):

| | per map+unmap | persistent map + 1 024 writes |
|---|---:|---:|
| MI50 | 3.41 µs | 8.1 µs total (8 ns/write) |
| RX | 2.83 µs | 6.4 µs total (6 ns/write) |

A map per gather costs **2.8–3.4 µs**, i.e. 3× the cost of gathering 1 024 rows on the RX. Whatever
holds the gathered block must be **persistently mapped**; `map_memory2` does not change that.

Host-side bookkeeping: building a 6 144-entry `VkBufferCopy2` array is **1 ns/region** (6 µs for
the chunk), so on the PLE path the host is not the problem — the GPU-side region serialisation is.

## A.8 Recommendation (build this, in this order)

1. **Repack the PLE tensor token-major with 96-byte-padded rows** (`48 × 96 = 4 608 B` per token —
   4 320 B of payload). Fixes both pathologies at once: IO amplification 46× → 1.9×, and gather
   7.4 → 1.0 ns/row. Storage 28.8 → 30.7 GiB (+6.7 %). This is a *new artifact*, not a rewrite of
   the reference file (the reference stays read-only; the tool writes a derived copy).
2. **Gather with a compute shader using `uvec4` loads and a rotated index SSBO** (one dispatch per
   chunk, `local_size_x = 128`, one thread per record). Never `vkCmdCopyBuffer2` region arrays for
   sub-kilobyte regions.
3. **Residency**: keep a VRAM hot set sized from `VK_EXT_memory_budget` (~1–2 GiB = 230–460 k
   tokens), gather from it at 0.4 ns/row; misses go through an **8–16-thread host gather of
   4 608-B records + one H2D of the block** (553 KiB ≈ 8 µs on the RX). Do **not** gather over
   PCIe from system RAM — 47–65 ns/row is worse than the warm host gather it replaces.
4. **Persistently map** the staging/gathered block; ignore `map_memory2` and `host_image_copy`.
5. If a repack is impossible for now, the byte-exact fallback is the `b16` shader
   (**2.3 ns/row RX / 7.4 ns/row MI50**) — still 40–100× better than region arrays.

## A.9 What I did not measure

* A full 28.8 GiB repack (write volume and time) — the amplification and gather numbers are
  per-read and layout-parametric, so the repack cost is a byte-count estimate, not a measurement.
* `VK_EXT_map_memory_placed`, `VK_EXT_external_memory_host` zero-copy of a page-cache range into a
  Vulkan buffer — the mechanism that would make "GPU gathers from page-cache memory" fast, and the
  one thing that could invalidate A.4's PCIe conclusion (round 3 listed it as unmeasured too).
* The real token→row mapping of `(160, 320 001 536)`; the repack design assumes 48/160 rows per
  token. If a token maps to 512 rows (round 3's open question), the same argument gets *stronger*
  (more rows per contiguous record), not weaker.
* `io_uring`/`SQPOLL` gather — the per-read cost here is device latency at QD1, and 32 threads
  already reach 395 k reads/s, so an async ring buys little.

---

# Part B — TensorRT-LLM audit

Shallow clone OK: `Reference Projects/tensorrt-llm`, **392 MB**, `cfae4519`, `--depth 1`.

## B.1 What the checkout is (and is not)

TRT-LLM is now **PyTorch-first**: `tensorrt_llm/_torch/` is the model/execution path, with a
`cpp/` core that still owns batching, KV and kernels (`cpp/tensorrt_llm/{batch_manager,executor,
kernels,runtime}`). There is no longer a single C++ "TRT engine" driver class to read; the
in-flight-batching *policy* is C++ and the *execution* is a traced torch graph with piecewise CUDA
graphs. Plan accordingly: the transferable material is in `batch_manager/`, not in the kernels.

## B.2 Iteration-level scheduling — how the batch is composed

`MicroBatchScheduler::operator()` (`cpp/include/tensorrt_llm/batch_manager/microBatchScheduler.h:62`)
takes `(activeRequests, inflightReqIds, maxBatchSizeRuntime, maxNumTokensRuntime)` and returns
**two vectors: context requests and generation requests**. Facts worth stealing:

1. **Two budgets, not one**: a *request count* and a *token count* per iteration, both re-supplied
   every iteration (`maxBatchSizeRuntime`, `maxNumTokensRuntime`). Nothing is reserved for the
   request's lifetime.
2. **The token budget counts *compute* tokens, net of reuse.** Comment at
   `microBatchScheduler.cpp:405`: *"batchNumTokens tracks COMPUTE tokens only (excluding reusable
   cached tokens)"*, and `reuse_adjusted_compute()`
   (`microBatchScheduler.cpp:67`) subtracts the radix-tree-reusable prefix from the first chunk
   only. A request with a 4 k cached prefix costs the budget almost nothing.
3. **`chunkUnitSize` = one KV block (or a multiple)**, explicitly *"to avoid fragmentation"*
   (`microBatchScheduler.h:28-40`). Context chunks are quantised to the block size, so the block
   table never needs a partially-filled block per request.
4. **Three chunking policies** (`microBatchScheduler.cpp:333`): `kEQUAL_PROGRESS` (every request
   advances one `chunkUnitSize` per loop until the budget is gone — lock-step fairness),
   `kFIRST_COME_FIRST_SERVED` (greedy), `kFORCE_CHUNK` (advance to the next snapshot point).
   Draft tokens are **discarded** if they do not fit the current chunk unit
   (`fitDraftTokens`, `microBatchScheduler.cpp:90`) — the speculative window is subordinate to the
   block accounting.
5. **Requests are ordered by LoRA id** before scheduling (`microBatchScheduler.cpp:31-52`), with
   finished-context requests partitioned to the end so the sampler can group them. Cheap
   "same-weight work together" ordering, no fancy cost model.

`CapacityScheduler` (`capacityScheduler.h`) is the admission layer above it: `MaxRequestsScheduler`
(trivial) and `MaxUtilizationScheduler`, which takes the **KV cache manager** and may **pause
already-started requests** to admit a higher-value one, has a **two-step look-ahead**
(`capacityScheduler.cpp:165`) that reserves for *two* iterations so a request does not thrash
between admit/pause, and has **prefix-aware scheduling** enabled by default
(`enablePrefixAwareScheduling = true`).

## B.3 KV cache block management

Two generations coexist. `kvCacheManager.cpp` (v1) is the familiar paged cache:
`maxBlocksPerSeq = ceilDiv(maxTokensHeld, tokensPerBlock)` (`:642`), blocks per window for
sliding-window layers, `enableBlockReuse` (radix/collision-based sharing), `maxTokens`,
`freeGpuMemoryFraction`, `hostCacheSize`, `crossKvCacheFraction` as the sizing knobs
(`executor/kvCacheConfig.cpp:25-56`).

**`kv_cache_manager_v2/` is the interesting one** (and it is what the newest commits are about):

* **Three tiers**: `CacheTier { GPU_MEM, HOST_MEM, DISK }` (`common.h:54`), each with a **byte
  quota** (`config.h`). The host tier reserves its address range **once at `maxQuota`** so that
  growing the tier "only commits more of it and never moves the base address" — a paging design
  that never has to re-pin or re-upload on growth.
* **Digest-keyed block radix tree** (`blockRadixTree.h`): per-block key = **SHA-256 chain over the
  token ids within a `ReuseScope`** (per-request/tenant namespace), incremental — "on first
  mismatch, skip the remaining hashing" (`:126`). Prefix match is decided **purely by digest**,
  with an explicit note that matching on token ids alone would allow cross-request reuse of
  another tenant's blocks (a *correctness/security invariant*, not an optimisation).
* **Eviction**: `LRUEvictionPolicy` (a `std::list` of pages with stable `NodeRef` iterators) and
  `PrioritizedEvictionPolicy` — per-priority LRU sub-queues in a `std::map`, **lowest priority key
  evicted first, LRU within it** (`evictionController.h:95-149`); one
  `PerLevelEvictionController` **per cache level**, one policy per pool group.
* **Cold-page codec**: demoting a page to host/disk can transcode it
  (`IKvCacheColdPageCodec`, `coldPageCodec.h:78`) — compression on the way down, e.g. an nvfp4
  codec for cold KV.
* Copy machinery: `batchedPageCopy.cu` (batched page D2H/H2D), `copyEngine`, `stagingBuffer`,
  `cudaVirtMem` (VMM), `eventManager`/`streamingEventSink` for cache events.

For 2–6 streams this is over-engineered as *KV*, but the **shape** is directly reusable for our
expert residency problem: quota per tier + priority-LRU per level + demote-with-transcode.

## B.4 Launch amortisation: piecewise CUDA graphs

`tensorrt_llm/_torch/compilation/piecewise_optimizer.py` captures CUDA graphs **per piece**, where
the pieces are cut at the attention op
(`_piecewise_boundary_ops()` = `attn_custom_op_inplace`, `mla_*`, `gdn_*`, `mamba2_*`,
`minimax_m3_attn_*`, `:32-44`), and the replay set is **bucketed by token count**
(`capture_num_tokens: List[int]`, sorted; `PiecewiseRunner` builds one graph per bucket,
`backend.py:60-74`). Two facts matter for us:

* The graph exists to **freeze geometry** (a fixed token count per bucket), exactly as round 4
  concluded for LMDeploy. Attention stays outside the captured region because its shapes/pointers
  change per iteration.
* **This is CUDA-only** in mechanism (`cudaStreamBeginCapture`/`cudaGraphExec`): Vulkan has no
  graph object. The portable residue is: *bucket the token dimension, pre-record per bucket, and
  keep the truly dynamic part outside the replay*. Our secondary-command-buffer replay (round 4:
  works on RADV, 19.6 ns per descriptor-set swap, worth −0.3 ms/token) is the same idea with a much
  weaker payoff, so the **bucketing discipline** is the part to copy, not the replay.

## B.5 Speculative decoding orchestration

`tensorrt_llm/_torch/speculative/` is a zoo: `mtp.py`, `eagle3.py`, `dflash.py`, `dspark.py`
(+`dspark_planner/schedule/verify`), `ngram.py`, `suffix_automaton.py`, `pard.py`,
`dynamic_tree_ops.py` + `eagle3_dynamic_tree.py`/`mtp_dynamic_tree.py` + `spec_tree_manager.py`,
`spec_sampler_base.py`, `accept_stats.py`, `ragged_helpers.py`. The transferable structure:

* **Draft slots are a fixed-size resource** (`mtp.py:48`: `num_draft_slots = config.max_draft_len`)
  with a slot pool and a shared resource manager between draft and target engines (`mtp.py:108`);
  the draft KV is a separate cache (`use_separate_draft_kv_cache`).
* **Verification is a tree**, not a chain, in the dynamic-tree variants: the draft tokens form a
  tree, `spec_tree_manager` packs it and attention reads a *tree mask*, then a single
  target forward verifies the whole tree; `accept_stats.py`/`acceptance_rate` is what closes the
  loop.
* **`SpeculationGate`** (`speculation_gate.py:8-60`): a rolling average of the true acceptance rate
  over the last N speculation-enabled iterations; if it falls below a threshold, speculation is
  **permanently disabled**. ~20 lines, and it is the thing that prevents a speculative path from
  silently costing 30 %.
* The scheduler treats draft tokens as **budget items that can be dropped** (B.2 item 4), so
  speculation never breaks block accounting.

## B.6 What maps onto our 2–6 stream agent case

Our current state: **20.3 ± 1 tok/s aggregate** on 2 streams (lane 40–43 ms/step, `--threads 32`),
prefill **6.45 tok/s/agent**, PLE gather being batched, barrier pruning 1 509 → ~400, four
multi-user GEMMs + T-batched shared expert pending, merged MoE works for the 9 resident blocks
while the streamed non-resident path loses on serialisation (429–530 µs/group vs 67–84 resident).

| TRT-LLM mechanism | our equivalent | what to copy |
|---|---|---|
| compute-token budget net of reusable prefix | we budget per stream/prompt length | count **compute** tokens; a cached prefix must not consume the iteration's T budget (this is the T-batched shared expert's sizing rule) |
| `chunkUnitSize` = KV block | our KV/attention tile sizes | quantise work units to the cache block so nothing is partially filled |
| `kEQUAL_PROGRESS` chunking | we serve streams FIFO per step | lock-step advance for 2–6 streams; removes the long-prefill starvation that shows up as p99 TTFT |
| 3-tier KV + priority-LRU per level | 9 resident expert blocks + streamed rest | treat expert residency as a **tiered cache with priority-LRU**, instead of a static 9 |
| digest-keyed radix tree (SHA-256 chain, `ReuseScope`) | we have radix reuse from round 1 | add the digest/scope invariant so reuse is provably safe across agents |
| token-count-bucketed graph replay | secondary-CB replay | bucket T, pre-record per bucket; keep attention dynamic |
| `SpeculationGate` | no spec path yet | if MTP/draft lands, ship the gate with it |

## B.7 Ranked adaptations (TensorRT-LLM → Breccia)

1. **Reuse-adjusted compute-token budget + block-quantised chunks** (B.2). Directly sizes the
   pending "T-batched shared expert" and the four multi-user GEMMs, and it is the only scheduler
   idea here that pays on *every* step. Measurable: 2-stream decode tok/s at fixed lane time, plus
   p99 TTFT when one stream issues a 300-token prefill against a cached prefix.
2. **Expert residency as a priority-LRU tier** (B.3's eviction shape applied to MoE, not KV). Our
   9 resident blocks are a static choice; `PrioritizedEvictionPolicy` per level is the natural
   replacement, with the streamed pool as the second level. Measurable: hit rate of resident
   blocks and the streamed-path share of the *"429–530 µs/group vs 67–84 resident"* cost.
3. **Token-bucket geometry freezing** (B.4, portable residue only). Bundle with the barrier-pruning
   work: after pruning, the remaining ~400 dispatches have fixed geometry per T-bucket, so one
   pre-recorded secondary CB per bucket is testable. Ceiling from round 4 is −0.3 ms/token — worth
   doing only as a rider on barrier pruning, never as its own round.
4. **`SpeculationGate`** (B.5) — 20 lines, only if/when a draft path exists. Deferred, but recorded
   so we do not ship speculation without it.

## B.8 Explicitly CUDA-only, with no Vulkan equivalent

`cudaGraph`/`cudaGraphExec` (capture + replay; our nearest is secondary CBs, priced at −0.3 ms),
`cudaVirtMem`/VMM paging (nearest is `VK_EXT_map_memory_placed`, *present on both cards*, plus
`vkAllocateMemory` — a real but unbuilt analogue), `cudaMemPool`, NCCL/`deep_ep`/NVLS multicast,
TMA/wgmma/`deep_gemm`/cutlass sm90+ paths, `torch.compile`/Inductor, the nvfp4 cold-page codec (our
artifact is IQ3_XXS/IQ3_S — the *idea* of transcoding on demotion survives, the codec does not).

---

# Top-3 next experiments, each with its bench

**E1 — PLE repack + 4 608-B record gather** (Part A.6). Build a token-major 96-byte-padded derived
artifact for the PLE tensor, then gather 8–16 threads (misses) / compute shader (hits). Expect the
chunked PLE path to go **14.7 ms → <1 ms**.
```
bench:  engine --profile ple, 64-token prefix + 2 streams, warm; A/B/A in one window
gate:   embeddings bit-identical to the 90-B path (compare gathered block bytes)
metric: ms/chunk for PLE; device bytes/chunk via /sys/block/nvme0n1/stat
```

**E2 — compute-token budget + equal-progress chunking** (B.2/B.7-1). Replace FIFO stream service
with a per-iteration budget that counts compute tokens net of cached prefix, chunks quantised to
the KV block, `kEQUAL_PROGRESS` across 2–6 streams.
```
bench:  2-stream decode, 8-16 steps, --threads 32; plus one 300-token prefill racing a decode
gate:   aggregate decode tok/s must not regress from 20.3 ± 1
metric: tok/s/stream, spread between streams, p99 TTFT of the prefill stream
```

**E3 — MoE residency as a priority-LRU tier** (B.3/B.7-2). Replace the static 9-block resident set
with quota + priority-LRU (resident tier / streamed tier), keeping the merged MoE kernel for
resident blocks.
```
bench:  merged-MoE micro-bench (per-group µs) + end-to-end 2-stream decode
gate:   merged-path output bit-identical; resident hit rate recorded
metric: µs/group resident vs streamed, streamed share per token, tok/s
```

---

# Surprises / corrections

1. **Region-array copies are the *slowest* gather** — ~300–360 ns/region on the MI50 and
   110–140 ns on the RX, ~independent of region size. The instinct "batch the 90-byte copies into
   one `vkCmdCopyBuffer2`" is 40–350× worse than a compute shader on this workload.
2. **Gather cost tracks load-instruction count, not bytes**: 45 × u16 = 7.4 ns/row vs 6 × uvec4
   (96 B padded) = 1.0 ns/row on the MI50. +6.7 % storage for 7.4× speed.
3. **ReBAR is VRAM, not host memory.** My first PCIe arm measured identical to VRAM because
   `HOST_VISIBLE|DEVICE_LOCAL` is device memory in the BAR. Only a plain `HOST_VISIBLE` allocation
   crosses PCIe (46.9 / 64.6 ns/row) — and it *loses* to a warm 32-thread host gather (~16 ns/row).
4. **The 48× over-read is one 4 KiB block.** FIEMAP: 2 213 extents of 8–16 MiB, uncompressed, 1:1.
   There is no extent-level fix, no compression to blame, and coalescing merges 3 of 1 024 regions.
   Only the layout is at fault.
5. **32 threads is past the knee** for the IO gather (8 threads already reach 62 % of it), so the
   PLE gather should be 8–16 threads, not 32 — the opposite of the decode lane's `--threads 32`.
6. **TRT-LLM's scheduler knows about prefix reuse**: the per-iteration budget is charged in
   *compute* tokens with the cached prefix subtracted (`microBatchScheduler.cpp:405,67`). That is a
   structural idea we do not have, and it is directly the sizing rule for the pending T-batched
   shared expert.

---

# Appendix — provenance

Probe sources and raw logs (scratch, not committed):
`<SCRATCH>/opencode/p5ple/` — `vkgather/` (Rust, Vulkan; `src/main.rs`, copied
`src/ctx.rs` from `harnesses/vk-probe`, shaders `gather_b16.comp`, `gather_b16ilp4.comp`,
`gather_v4.comp`), `p2io.c`, `run-mi50.log`, `run-rx.log`, `run-mi50-validation.log`.

Reproduce:

```bash
cd <SCRATCH>/opencode/p5ple/vkgather && cargo build --release
timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ./target/release/vkgather --pci 0000:0e:00.0     # MI50 first, always
timeout 90 ./target/release/vkgather --pci 0000:03:00.0   # RX only after MI50 is clean

cd ../ && gcc -O2 -pthread -o p2io p2io.c
PLE=/srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
./p2io --file $PLE --n 6000 --rec 90   --threads 32 --span-mb 2000
./p2io --file $PLE --n 6000 --rec 4320 --threads 32 --span-mb 2000

filefrag -v $PLE | head -25          # 8-16 MiB extents, physical == logical
```

Reference clone: `Reference Projects/tensorrt-llm` @ `cfae4519` (shallow, 392 MB, read-only).
