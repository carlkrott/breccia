# 31 — Host memory tier: what the host lane's mapping can and cannot get

Round 31, 2026-10-28. Read-only research on base f4800c2. No build, no GPU program, no benchmark, nothing committed. "Derived" marks arithmetic on cited measurements. "UNVERIFIED" marks claims not checked here. A bare path is in this repo at f4800c2; `:` gives a line. Doc 30's "Doc 08" means `docs/phase-3/08-cpu-expert-lane.md`, not `docs/research/08-submits-and-deepspeed.md` (conflict h). This doc writes full paths.

Parent context (as doc 30): two-GPU decode, Qwen3.8 IQ3_XXS, 70.6 GiB mapped, 30 GiB memory cap, lane 41 ms at 32 threads. Host blocks per stream are 18 or 39 depending on which residency applies; 48 is the byte basis of the phase-3 and phase-4 figures (conflict j). Gate 0 decides which.

## 1. Verdict

1. **The lane's mapping has no memory-tier advice.** `crates/breccia-format/src/filemap.rs:9–13` is a bare `Mmap::map`: no madvise, no MAP_POPULATE, no mlock, no hugepage hint. Every memory-tier effect the lane sees comes from kernel defaults and sysctls.
2. **Transparent hugepages do not reach this mapping.** THP is enabled=always, yet the live model mapping shows FilePmdMapped 0 and file_thp 0. A MAP_SHARED file mapping on btrfs without large folios stays 4 KiB-granular, so MADV_HUGEPAGE does nothing here (§6). The hugepage half of phase-4 untried lever 4(a) is dead for the GGUF mapping.
3. **Bandwidth shows no memory roof at 18 host blocks.** 15.3 GB/s is 50–68% of the 16-thread standalone band. At 39 blocks, 41 ms needs 33.2 GB/s, which is the measured 32-thread in-graph ceiling (phase-4, on 48-block bytes), so the lane runs at that ceiling and not beyond it. The 30 GiB cap also leaves a steady-state storage miss stream of about 2.5 GB/s under uniform routing, which routing skew would lower (§2). Gate 0 decides which case is live.
4. **The best new candidate is a counter-first fix to the PLE table (card 1), not a placement change.** Derived: each token faults 16 random 90-byte PLE rows, and each fault reads a readahead window. At the live btrfs readahead (128 KiB) that is up to 2 MiB of device reads and page cache per token for 1,440 useful bytes, while the file's miss counter stays at or below 100 (§4, card 1). Doc 03 says to restore 8192 KiB, which would make each fault read 8 MiB (conflict e).
5. **Hugepage-backed host import (Q4, §7) turns on one unknown:** whether RADV or libdrm sets `AMDGPU_GEM_USERPTR_ANONONLY`. If it does, doc 09:385's page-cache import fails at the kernel, and only anonymous memory remains (card 6). Alignment is not a constraint; 4 KiB passes at every layer.

No card is a decode-rate claim. Every expected effect is UNVERIFIED.

## 2. Q1: bandwidth ceiling, and bandwidth- or compute-bound at 32 threads

Geometry (doc 30 §2, §4; doc 29 §1): 512 experts per block, 10 routed per token per block, 48 blocks, mean expert 1,746,133 B (4,915,200 weights, 0.355 B/weight). Uniform routing. The 41 ms is the phase-4 §8y lane at 32 threads (`docs/phase-4/01-forward-pass.md` §8y, 41.2 ms).

| host blocks per stream | MB per stream-token | MB per pair step | GB/s at 41 ms | GB/s per physical core (16) | streamed set | vs 30 GiB cap |
|---|---:|---:|---:|---:|---|---|
| 18 (doc 30 §3; doc 29 C7) | 314.3 | 628.6 | 15.3 | 0.96 | 16.09 GB = 14.99 GiB | fits, about 15 GiB before PLE pages |
| 39 (CLI default, `crates/breccia-cli/src/main.rs:1662–1664`) | 681.0 | 1,362.0 | 33.2 | 2.08 | 34.87 GB = 32.47 GiB | 2.47 GiB over |
| 48 (all blocks; the byte basis of phase-3 §6.2 and phase-4 §8p, conflict j) | 838.1 | 1,676.3 | 40.9 | 2.56 | 42.91 GB = 39.97 GiB | 9.97 GiB over |

**Ceilings measured.** Standalone lane (`docs/phase-3/08-cpu-expert-lane.md` §6.2, lines 281–297; token bytes at 48 blocks, line 25): 1 thread 3.09 GB/s; 8 threads 20.4; 16 threads 22.5–30.5; single-thread kernel ceiling 2.9–3.1 GB/s. No standalone row exists at 32 threads; the table stops at 16. In-graph (`docs/phase-4/01-forward-pass.md`, lines 436–446): 15.0 at 8 threads, 20.8 at 16. The 33 GB/s figure for 32 threads appears in doc 03 line 11 (`docs/research/03-io-and-exllamav2.md:11`, no row behind it) and as the lane's ceiling in phase-4's 4-stream in-graph step (`docs/phase-4/01-forward-pass.md` lines 1179–1192: 98 ms per step, 4 × 799 MiB; lines 1316–1323; `docs/phase-4/02-untried-levers.md` lines 17, 189). Phase-4 cites 33 GB/s for that step; its own arithmetic (4 × 799 MiB = 3.35 GB over 98 ms) is 34.2 GB/s. This doc's 33.2 is derived at 41 ms, not read from that step.

**Reading.**
- 18 blocks: 15.3 GB/s is 50–68% of the 16-thread standalone band, 75% of the 8-thread figure, and matches the in-graph 8-thread number. No measured roof is reached.
- 39 blocks: 33.2 GB/s is above the highest standalone row (30.5, at 16 threads) and sits at the one 32-thread figure in the repo (34.2 by phase-4's arithmetic, above). So the lane runs at that in-graph ceiling, not beyond it. The gap to the standalone band is the byte accounting or the residency (doc 30 conflict i; phase-3 and phase-4 price a token at 48 blocks, conflict j).
- 48 blocks (the byte basis of phase-3 §6.2 and phase-4 §8p): the streamed set is 42.91 GB = 39.97 GiB, which matches the routed-expert mapping in `crates/breccia-engine/src/model.rs` (1–20). Under the same uniform model 24.9% of bytes miss, about 10.2 GB/s of storage demand at 41 ms, against doc 03's 4.7–5.4 GB/s cold rate at 128. Phase-4's 4-stream step ran on page-cache reads at about 34 GB/s (4 × 838 MB over 98 ms), so either the cap in force then (not recorded here) was larger, or that step was not steady state.
- Compute: 16 physical cores, two SMT threads each (CCD0 = CPUs 0–7 and 16–23, doc 30 card 1). Per-core demand is 31–33% of one core's single-thread kernel rate at 18 blocks, and 67–72% at 39 blocks. Two siblings share one core's issue width, so 39 blocks is plausibly issue-bound; 18 blocks is not.
- Miss stream at 39 blocks: with 30 GiB of cache and no other resident pages, 7.6% of expert bytes miss under independent uniform references (an LRU-like cache keeps C/N of them: 32.21 / 34.87 = 92.4%). PLE pages share the same cap (card 1), so the real rate is higher. Doc 29 C14 records routing-skew readings that do not reproduce (302 vs 579.5 MiB per token at T = 32); skew would lower the rate. That is about 2.5 GB/s of storage reads at 33.2 GB/s demand. Doc 03's cold rates are 0.69–1.10 GB/s at read_ahead 8192 and 4.7–5.4 GB/s at 128 (`docs/research/03-io-and-exllamav2.md` lines 110–124). At 8192 the 39-block case cannot run at 41 ms; at 128 it can, only if the disk is otherwise idle. At 18 blocks the set fits, so any steady-state storage read comes from something other than the expert set (PLE churn, card 1).
- Page touches: 76,733 4 KiB pages per stream-token at 18 blocks (4,796 fault-around units); 166,260 at 39; 204,625 at 48.

**Answer.** At 32 threads, the 18-block lane sits under every measured ceiling. The closest measurement of bound type is phase-4 `docs/phase-4/01-forward-pass.md:440`: "It scales with threads, so it is compute-bound, not fault-bound." It was taken at 8 and 16 threads on 48-block bytes, with the sampled expert pages already warm in the page cache (the source's own inference from run-to-run timing, lines 440–443). The 18-block set is the regime that would be fully resident after warm-up. My stall-bound reading (per-core demand 31–33%, 76,733 page touches per stream-token) is inference and conflicts with line 440 (conflict l); card 0's IPC and stall counters decide it. The 39-block lane is compute-leaning, which agrees with line 440, and its 41 ms is not steady state without about 2.5 GB/s of misses. At 48 blocks the miss stream is about 10 GB/s. A 41 ms step cannot describe both residencies. Card 0 decides which one ran.

## 3. Card 0: Gate 0 extension (measurement, no code)

Extend doc 30 §3; do not add a second Gate 0. Add to the same single output, over the same window:
- `memory.stat` deltas in bytes: file, file_mapped, workingset_refault_file, pgmajfault, pgsteal, pgscan.
- `/proc/<pid>/smaps_rollup` FilePmdMapped and AnonHugePages; `mincore` split between the expert range and the PLE range.
- `perf stat` IPC, dTLB-load-misses, and the box's stall and DRAM events (names from `perf list` at run time; not pinned here), with both SMT siblings loaded.

Divide every cumulative counter by the window's token count before comparing. None of them is a decode rate.
- **Gate:** all of the above appear beside lane ms, fence and read_bytes. Without them, cards 1–6 stay ungated.
- **Expected effect:** none (measurement).
- **Risk:** `perf_event_paranoid` may block counters; check first.

## 4. Ranked cards (flag default OFF for all)

**Card 1: MADV_RANDOM on the PLE table range only.**
- **Change:** `madvise(MADV_RANDOM)` on the PLE range (26.82 GiB; `crates/breccia-engine/src/model.rs:1–20`). Never on expert ranges. Flag `PLE_RANDOM`, default OFF.
- **Mechanism (derived; mm/filemap.c at linux-6.18.y 1253a6aa, lines 3206–3330):** a sync fault on a mapping without `VM_RAND_READ` reads a window of `ra_pages` around the fault (start = pgoff - ra_pages/2, size ra_pages). Live ra_pages is 128 KiB (the btrfs BDI's `read_ahead_kb`). 16 rows per token gives 2 MiB of device reads and cache fill for 1,440 useful bytes, while the file is below the throttle. RANDOM reads one 4 KiB page per fault: 64 KiB per token.
- **Throttle (same source, lines 3305–3313):** the file's `mmap_miss` counter rises on each sync miss and falls on cache hits (lines 3381–3383), capped at 1000. Above `MMAP_LOTSAMISS` (100), read-around is skipped and a fault reads only its own page. The counter lives in the open file (`file->f_ra`), so PLE and expert faults share it. A miss-heavy PLE stream can hold the counter above 100, which pulls the 2 MiB bound toward 4 KiB per fault. RANDOM ranges return before the counter is touched, so card 1 also takes PLE misses out of the counter that gates expert read-around; the gate's expert-refault check covers that effect. UNVERIFIED on this workload: no counter card 0 lists exposes the live `mmap_miss` value.
- **Gate:** steady-state read_bytes per token with the PLE range split out (`mincore`), and expert workingset_refault_file and pgmajfault deltas over 200 tokens after warm-up. Pass: PLE read bytes fall by at least 1 MiB per token, and expert refaults do not rise.
- **Expected effect:** UNVERIFIED. If the mechanism holds, the cache stops churning and the 18-block case keeps its headroom.
- **Risk:** doc 03 measured RANDOM as negative for expert regions (conflict a). This card does not touch experts; applying it there repeats that loss.

**Card 2: MADV_COLLAPSE on hot 2 MiB expert ranges (void unless PMD folios exist).**
- **Change:** after warm-up, `MADV_COLLAPSE` over 2 MiB-aligned ranges covering the hottest expert blobs. Expert ranges only; never PLE.
- **Void unless** the page cache can hold PMD-size folios for this file. btrfs large folios need CONFIG_BTRFS_EXPERIMENTAL, which is absent here (6.18.55 status UNVERIFIED), and file_thp is 0.
- **Density caveat:** a 2 MiB folio holds 1.2 experts (1,746,133 B each). A collapse maps about one expert per PMD, so the TLB gain per byte is bounded by density. Hot-first layout (doc 29 card 4) is the only way to fill a PMD with hot experts.
- **Gate:** smaps FilePmdMapped moves above 0 after collapse, and the lane reaches at most 38 ms at `--threads 32` (3 ms cut, about +0.6 tok/s by doc 30 §2). The fence is reported with it.
- **Expected effect:** UNVERIFIED.
- **Risk:** collapse copies pages (cost unmeasured) and can fail silently. Check FilePmdMapped rather than the call's return value.

**Card 3: WILLNEED per range, two blocks ahead at most (replaces doc 29 card 1's RANDOM on experts).**
- **Change:** WILLNEED on the predicted next block's expert ranges, issued before the current block's phase 1. Depth ≤ 2 blocks.
- **Predictor:** doc 23 (expert prediction and placement) and doc 24. Doc 30 §8 rejects cross-block prefetch because no predictor is measured, and the route trace (doc 30 OQ4) does not exist yet.
- **Measured history:** `docs/research/03-io-and-exllamav2.md:120` WILLNEED one block ahead 1.23 GB/s (superseded by the 128 KiB sysctl); `:121` two blocks ahead with 32 threads 3.6–5.0 GB/s, unstable; `:122` four blocks ahead 1.06 GB/s, amplification 4.1, negative (queue limit).
- **Gate:** steady-state read_bytes and major faults per expert both fall, with the route trace in hand first.
- **Expected effect:** UNVERIFIED; capped by the two-block measurement.
- **Risk:** queue saturation past two blocks (doc 03:122); wasted reads when the prediction is wrong.

**Card 4: MemoryLow on the engine unit (co-tenant case only).**
- **Change:** a `MemoryLow=` drop-in on the engine unit. Not `MemoryHigh` (rejected, §8). Flag: drop-in, OFF until measured.
- **Gate:** with a co-tenant workload running (another model server or ComfyUI), the engine cgroup's workingset_refault_file and pgmajfault deltas over the window, with and without MemoryLow. Alone, no effect is expected.
- **Expected effect:** UNVERIFIED.
- **Risk:** memory.low is best effort under global pressure. It does not protect the engine above its own memory.max.

**Card 5: anonymous THP copy of the hot streamed set (alternative to doc 29 card 2's in-place mlock).**
- **Change:** copy the streamed set (18 blocks, 14.99 GiB) into an anonymous `MADV_HUGEPAGE` region, mlock it, then drop the source file pages (doc 29 card 1 path). With defrag=defer+madvise, madvised regions take direct reclaim and compaction on fault (kernel docs, not re-read), so the hint can work for anonymous memory. Doc 09 A1.2 measured first touch of unlocked anonymous host memory at 8.6 GB/s.
- **Gate:** AnonHugePages at least 90% of the region; pswpin and pswpout both zero over the window; lane ms and dTLB misses against the in-place mlock build; anon plus file footprint within the cap.
- **Expected effect:** UNVERIFIED.
- **Risk:** the copy doubles footprint until the source pages are dropped. Anonymous memory is swappable to zram, which holds other tenants' data here (§6). Compaction stalls on a fragmented box are unmeasured.

**Card 6: hugepage-backed anonymous host import (doc 29 card 5, with the page-size question added).**
- **Change:** none yet. A future GPU-qualification probe, not run here: on the RX, call `vkGetMemoryHostPointerPropertiesEXT` and then `vkAllocateMemory` with `VkImportMemoryHostPointerInfoEXT` on (a) a page-cache range of the GGUF mapping and (b) a 2 MiB-aligned anonymous `MADV_HUGEPAGE` region. Record the VkResult for each.
- **Flag:** OFF.
- **Gate:** (a) succeeds, so doc 09:385 stays alive. (b) succeeds, so card 5 can feed the RX import path. Any imported path must then pass doc 27's gate, which doc 29 §0 item 5 names. Precedent: doc 09 A1.2 measured anonymous mlocked and unlocked imports read at 28.7–29.7 GB/s; no file-backed import is measured there.
- **Expected effect:** UNVERIFIED.
- **Risk:** ANONONLY decides (§7). THP-backed anonymous memory is the intended path; MAP_HUGETLB mappings are believed to carry a vm_file and fail (UNVERIFIED).

## 5. Q2: how CPU serving stacks handle the memory tier

| Stack (source as read) | Mapping advice | Page size | NUMA and affinity | Notes |
|---|---|---|---|---|
| llama.cpp (`src/llama-mmap.cpp:488, :492, :519, :903`, as cited by doc 03 lines 81–98; that clone is not pinned to 8ca4060b) | `posix_fadvise` SEQUENTIAL on the file; `MAP_POPULATE` when prefetch is on; `posix_madvise` RANDOM on lazy ranges; `llama_prefetch` is MADV_WILLNEED over merged ranges. Fork knob `LLAMA_MMAP_CPU_MADVISE=random` adds per-tensor RANDOM, which doc 03 measured as the wrong call for these region sizes | none | `--numa` sets prefetch 0 and RANDOM over the mapping (earlier read, not re-verified) | The only stack with mapping-hint code, and its hints point the opposite way from what cold expert reads need. |
| vLLM CPU backend (docs as read) | none | none | VLLM_CPU_OMP_THREADS_BIND=auto per NUMA node; TCMalloc or Intel OMP via LD_PRELOAD | No mmap or mlock path found. |
| xFasterTransformer (README as read) | none | none | numactl NUMA rank per socket; libnuma | Multi-socket recipe; one NUMA node here. |
| fastllm (README as read) | none | none | `--auto` planner only | No memory-tier mechanism found. |
| KTransformers (README as read) | none | none | "NUMA-aware memory management" (claim only) | No mechanism found in what was read. |
| OpenVINO (search snippets, 2025.2 notes) | mmapped tensors; delayed constant weights | no info | NUMA-cloned read-only weights (CNN) | Snippets only; no code read. |

None of the six uses MADV_HUGEPAGE, MADV_COLLAPSE, hugetlbfs or file THP. The README and snippet rows are weak evidence: absence in a README is not absence in code.

## 6. Q3: page cache, THP, cgroup, zram and MGLRU under a 70 GiB mapping and a 30 GiB cap

Live snapshot (cumulative counters from the exited bench cgroup; not rates): memory.max 32,212,254,720; file 23.5 GB; file_mapped 22.9 GB; anon 19.9 MB; swap.current 13.6 MB; workingset_refault_file 6,955,607; pgmajfault 311,693; pgsteal 11,077,144; anon_thp 6.3 MB; file_thp 0; model mapping FilePmdMapped 0, THPeligible 1.

- **The cgroup reclaimed its own file pages.** It sat near its cap (pgsteal, workingset refaults). Its anonymous memory was negligible (19.9 MB).
- **zram does not hold the experts.** Expert and PLE pages are clean, MAP_SHARED file pages. Clean file pages are dropped without writeback and never reach zram. The zram device (sysfs mm_stat for the zram device: orig 23.8 GB, compr 9.27 GB, mem_used 9.5 GB, about 2.5:1; phase-0/04:11 recorded 2.6:1) holds other tenants' anonymous data. Card 5 would move experts into zram's reach. zram's charging to the cgroup is UNVERIFIED.
- **THP.** enabled=always, shmem=never, defrag=defer+madvise; CONFIG_READ_ONLY_THP_FOR_FS=y (khugepaged file collapse is built in; its conditions not re-read here). File THP needs huge folios from the filesystem, and btrfs has them only with CONFIG_BTRFS_EXPERIMENTAL, which is absent. Kernel docs (transhuge; "latest", not pinned) say MADV_COLLAPSE ignores the sysfs enabled setting. Live FilePmdMapped is 0, which fits 4 KiB folios.
- **Readaround.** A sync file mmap fault reads a window of ra_pages around the fault, unless the file's miss counter is above 100 (mm/filemap.c at linux-6.18.y 1253a6aa, lines 3206–3330; §4 card 1). ra_pages comes from the btrfs BDI's read_ahead_kb, 128 KiB live. Doc 03 measured the effect on expert regions: amplification 5.9–9.0 at 8192 and 1.05–1.14 at 128 (`docs/research/03-io-and-exllamav2.md` lines 110–124). For random 90-byte PLE rows the same window is almost pure waste (card 1).
- **memory.high vs memory.max.** memory.high throttles below the hard limit; memory.stat is in bytes.
- **MGLRU** is enabled (lru_gen 0x0007). It orders eviction by generation age and does not enlarge the cap. 14.99 GiB (18 blocks) can be the hot set under 30 GiB; 32.47 GiB (39 blocks) cannot.

## 7. Q4: hugepage-backed host import

- **Alignment.** minImportedHostPointerAlignment is 4096 on this box (doc 09:39; doc 22:22). The Vulkan refpage for VK_EXT_external_memory_host was read in its Issues section only; the address and size rule is not re-quoted here. `crates/breccia-vk/src/transfer.rs:691` aligns to max(minImported, 4096). A 2 MiB-aligned region passes.
- **amdgpu userptr.** The ioctl checks `offset_in_page(addr | size)`, which is 4 KiB (`drivers/gpu/drm/amd/amdgpu/amdgpu_gem.c`, torvalds/linux @ 65ba8f34). Neither the alignment nor the size stops hugepage-backed anonymous memory.
- **ANONONLY.** Flag value 1<<1 (`include/uapi/drm/amdgpu_drm.h`; READONLY 1<<0, VALIDATE 1<<2, REGISTER 1<<3). In `amdgpu_ttm_tt_get_user_pages()` (`drivers/gpu/drm/amd/amdgpu/amdgpu_ttm.c`), with ANONONLY set and `vma->vm_file` non-NULL, the VALIDATE path returns -EPERM. THP-backed anonymous memory has no vm_file, so it passes.
- **Userspace: not checked.** Whether RADV's host-import path sets ANONONLY is UNVERIFIED (searches in §10).
- **Consequences.** If ANONONLY is set, doc 09:385's page-cache import fails, and only anonymous memory remains (card 5, then card 6). If it is not set, doc 09:385 stays alive, but the page-cache pages still need mlock and give no hugepage benefit.

## 8. Rejected

1. **MADV_HUGEPAGE on the GGUF mapping** (the hugepage half of phase-4 untried lever 4(a), and of doc 09's plans). No file THP without huge folios; live FilePmdMapped is 0.
2. **btrfs large folios.** Experimental config absent here; 6.18.55 status UNVERIFIED; not a change this round can make.
3. **NUMA first-touch, mbind, numactl placement.** One NUMA node. The lane's cost is not remote memory.
4. **memory.high below memory.max.** Throttles the engine's own faults; the cap is already the hard limit; no measured gain.
5. **llama.cpp `--numa` mapping advice** (RANDOM, prefetch 0). Doc 03 measured RANDOM negative on expert regions.
6. **hugetlbfs.** Believed to carry vm_file and fail ANONONLY (§7, UNVERIFIED). Needs a separate reservation for no measured gain.
7. **zswap.** Compresses anonymous pages only; the expert set is clean file pages.
8. **zram writeback and swap tuning** (swappiness 10, vfs_cache_pressure 50). Neither touches clean file pages.
9. **vLLM and xFasterTransformer NUMA recipes.** Multi-socket knobs; one NUMA node here.
10. **fastllm `--auto` and the KTransformers NUMA claim.** No mechanism found.
11. **OpenVINO mmap of constants.** Same default path as llama.cpp, without the hints.
12. **mlock of the whole 70 GiB mapping.** Larger than the 30 GiB cap. Only the streamed set is lockable (doc 29 card 2).
13. **Huge folios or mlock on the PLE table.** Random 90-byte rows; a 2 MiB unit per row is pure waste.
14. **Raising the cap.** A deployment decision, outside this round. At any cap below 33 GiB, the 39-block set does not fit.

## 9. Conflicts

a. **Doc 29 card 1** sets MADV_RANDOM on streamed expert ranges. `docs/research/03-io-and-exllamav2.md` lines 55–68 and 110–124 measured RANDOM at 0.32–0.77 GB/s against 0.69–1.10 for the default (RANDOM disables fault-around and readahead). Doc 03 also measured SEQUENTIAL negative (0.25–0.60), so the replacement is per-range WILLNEED (card 3), not SEQUENTIAL. RANDOM stays right for PLE (card 1).
b. **Doc 09:385** plans to import each expert block's page-cache range (4096, mlock'd). The kernel's ANONONLY check on the VALIDATE path (§7) decides it at runtime. Doc 09 A1.2 measured only anonymous imports (28.7–29.7 GB/s mlocked; 29.5 GB/s unlocked, warm). Card 6 settles the file-backed case.
c. **Phase-4 untried lever 4(a)** (`docs/phase-4/02-untried-levers.md:195`, `:425`): MADV_HUGEPAGE plus mlock on the hot experts, target 33 → 45–50 GB/s. The hugepage half is a no-op on a MAP_SHARED file mapping. The mlock half survives as doc 29 card 2. The 45–50 GB/s target has no mechanism here.
d. **Doc 30 §2–§3** (41 ms; 33 GB/s at 39 blocks) against the cap (32.47 GiB streamed > 30 GiB) and the standalone band (highest row 30.5 at 16 threads). Doc 30 flags this too (its conflict i). Resolution: 41 ms at 39 blocks needs about 2.5 GB/s of steady-state misses under uniform routing. Gate 0 read_bytes settles it. Doc 29 C7 and Q7 record the same 18-vs-39 split (citing doc 27:133), and doc 29 C14 records routing-skew readings that do not reproduce.
e. **Doc 03 `docs/research/03-io-and-exllamav2.md` lines 20–28** says "this box: 8192 ← the problem" and gives "restore: echo 8192". The live btrfs BDI value is 128, the recommended value. Doc 29 C9 cites a timer that pins 128 (doc 29's citation at C9, not checked here), and doc 09:440 says the fix was not persisted. Restoring 8192 would make each PLE fault read an 8 MiB window (derived: up to 128 MiB of device reads per token, at 16 faults). Do not run that restore line.
f. **Phase-0/04:11** (zram 20.5 GB data in 7.8 GB, 2.6:1) against the live 9.5 GB physical at about 2.5:1. Snapshot drift; no card depends on it.
g. **Doc 03 lines 70–74 region sizes** (462.5 + 462.5 + 900 KiB = 1,825 KiB = 1,868,800 B, equal to the sampled blk.0 blob in doc 29 §1) against the mean 1,746,133 B used here (doc 29 §1, doc 30 §2). The mean gives 7% fewer bytes per token. This doc uses the mean, as doc 30 does.
h. **Numbering.** Doc 30 §3, §5 and §7 say "Doc 08" for the lane ceilings and geometry. That is `docs/phase-3/08-cpu-expert-lane.md`. `docs/research/08-submits-and-deepspeed.md` is a different document.
i. **Doc 29 §0 item 2 vs card 5.** Doc 29 §0 item 2 pins the streamed set in place with mlock on the file mapping, not an anonymous copy; doc 29 §5 Q3 asks whether that is acceptable. Card 5 is the anonymous-copy alternative. Both need the same cap decision.
j. **Byte basis.** Doc 31's rows use 18 and 39 host blocks (18 from doc 30 §3 and doc 29 C7; 39 from the CLI default, `crates/breccia-cli/src/main.rs:1662–1664`). The standalone bench (`docs/phase-3/08-cpu-expert-lane.md` line 25, §6.2) and phase-4's 4-stream step (`docs/phase-4/01-forward-pass.md` lines 1179–1192) price a token at 48 blocks, 838.1 MB. The GB/s rows in §2 are bytes over time, so they stand as bandwidth comparisons; the block count changes only the bytes. The 33 GB/s ceiling is a 48-block figure, and Gate 0 must record the block count.
k. **Single-core stream.** `docs/phase-4/02-untried-levers.md` line 189 cites a 53.9 GB/s single-core stream on this host and 54–65 GB/s from other engines. Not re-measured here. Doc 31 uses the lane's own single-thread kernel rate (2.9–3.1 GB/s) for per-core demand, not a stream rate.
l. **Bound type.** Phase-4 `docs/phase-4/01-forward-pass.md:440` calls the lane compute-bound because it scales with threads (in-graph table, lines 436–446: 8 and 16 threads, 48-block bytes, expert pages warm in the page cache per lines 440–443). §2 reads the 18-block lane as stall-bound by inference from per-core demand and page-touch count. The two conflict. The 39-block reading agrees with line 440, but line 440 is a warm-cache run, so it does not cover the cap-limited case. Card 0's IPC and stall counters decide.

## 10. Open questions

1. Steady-state read_bytes at 18 and at 39 blocks, with the live readahead of 128 (card 0). Decides which residency is live and whether 41 ms is steady state.
2. Does PLE readaround displace expert pages, and is the file's miss counter above 100 during decode? No counter exposes it; card 1's gate is the test.
3. At 39 blocks, is the 32-thread lane SMT-limited? (card 0 IPC.)
4. Does RADV or libdrm set ANONONLY? (card 6 probe; a GPU program, so a future card.)
5. Does MADV_COLLAPSE raise FilePmdMapped on this file and kernel, and at what cost? (card 2.)
6. Does linux-6.18.y give btrfs large folios without CONFIG_BTRFS_EXPERIMENTAL? (UNVERIFIED.)
7. Does the 32-thread in-graph figure (33 GB/s cited, 34.2 by the step's own arithmetic) hold standalone and at 39 blocks? Its only source is phase-4's 4-stream step on 48-block bytes (conflict j); no standalone row exists past 16 threads.
8. Route histogram (doc 30 OQ4): needed for card 3's predictor.
9. Cap split: how much of the 30 GiB does PLE churn take, and is an anonymous copy (card 5) acceptable to the owner (doc 29 §5 Q3)?

## 11. What was not checked

- No build, run, benchmark, perf session or GPU program. The live snapshot is cumulative from an exited cgroup.
- Userspace ANONONLY (RADV, libdrm): DeepWiki is not indexed for either. Code search does not index mesa/mesa, so its zero hits are inconclusive. UNVERIFIED.
- Kernel behaviour is from kernel documentation ("latest", not pinned to 6.18.55) and from amdgpu at torvalds/linux master 65ba8f34. The read-around and miss-throttle path is read at linux-6.18.y 1253a6aa, not at the running 6.18.55; whether 6.18.55 matches is UNVERIFIED. The live `mmap_miss` value is not exposed by any card 0 counter, so whether the throttle is active on this workload is UNVERIFIED. Doc 03's measured amplification is the evidence that readahead applies to mmap faults.
- No Zen 4 per-CCD fabric figure was verified; no two-CCD bandwidth ceiling is used as a gate.
- Other stacks: READMEs and docs as read, no commit pinned. OpenVINO: search snippets only.
- llama.cpp line numbers are doc 03's citations of its own clone, not re-read at 8ca4060b. The `--numa` row is from an earlier read and is not corroborated by doc 03.
- hugetlbfs vm_file behaviour: UNVERIFIED.
- No GGUF header read. The 90-byte PLE row and expert counts come from `crates/breccia-engine/src/model.rs` and doc 29.

## 12. Sources

Repo at f4800c2 (path, lines as read):
- crates/breccia-format/src/filemap.rs (9–13)
- crates/breccia-engine/src/model.rs (1–20, 1340–1362)
- crates/breccia-kernels/src/cpu/expert.rs (301; 389–412; 463–548)
- crates/breccia-engine/src/graph.rs (3132–3144)
- crates/breccia-vk/src/transfer.rs (691)
- crates/breccia-cli/src/main.rs (1662–1664)
- docs/research/03-io-and-exllamav2.md (11; 20–28; 55–74; 81–98; 110–124)
- docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md (39; 42; 58–60; 385; A1.2)
- docs/research/22-vulkan-driver-surface-tiering.md (22)
- docs/research/23-model-agnostic-drafting-and-expert-prediction.md; docs/research/24-moe-offload-mixed-precision-experts.md
- docs/research/29-model-loading-and-checkpoint-io.md (§0 items 1–5; §1; §2; §5 Q3, Q7; C7, C9, C14)
- docs/research/30-cpu-lowbit-expert-kernels.md (§2, §3, §8, §9, §10)
- docs/phase-0/04-host-memory.md (11, 25)
- docs/phase-3/08-cpu-expert-lane.md (line 25; §6.2–6.3, lines 281–316)
- docs/phase-4/01-forward-pass.md (§8y, about lines 2109–2175; 436–446; 1179–1192; 1316–1323)
- docs/phase-4/02-untried-levers.md (17, 189, 195, 425)

Upstream (repo, path, commit; no URLs):
- ggml-org/llama.cpp (clone cited in doc 03 lines 81–98; not pinned here): src/llama-mmap.cpp:488, :492, :519, :903; src/llama-model-loader.cpp:1738–1752 (fork patch per doc 03)
- torvalds/linux @ 65ba8f34: drivers/gpu/drm/amd/amdgpu/amdgpu_gem.c (userptr ioctl); drivers/gpu/drm/amd/amdgpu/amdgpu_ttm.c (get_user_pages, ANONONLY); include/uapi/drm/amdgpu_drm.h (flags); Documentation/admin-guide/mm/transhuge.rst, multigen_lru.rst, cgroup-v2.rst (read as kernel docs, "latest")
- linux stable tree, branch linux-6.18.y, commit 1253a6aa: mm/filemap.c (`MMAP_LOTSAMISS` at 3206; `do_sync_mmap_readahead` 3260–3330; `do_async_mmap_readahead` 3361–3386; fault call site 3501); fetched from a stable-tree mirror and identified by commit
- Khronos Vulkan refpage for VK_EXT_external_memory_host (Issues section)
- vLLM, xFasterTransformer, fastllm, KTransformers: READMEs and docs as read; OpenVINO: search snippets
