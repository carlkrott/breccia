# 22 — Vulkan driver surface for tiering: queues, host import, budget, global priority, iGPU

Round 22 of the paired rotation. Research only: no code was changed, and no GPU, binary, `vulkaninfo` or benchmark was run. Every speed number is cited, marked `derived` (arithmetic on cited inputs), or marked `UNVERIFIED - requires the GPU qualification card`. RADV and kernel source are not on the machine, so driver behaviour is UNVERIFIED unless a repo file or the Khronos registry on disk states it.

## 1. Summary

- **Sync is the blocker, not the queue.** The RX copy and FMA compute overlap (102.3 %, `docs/phase-0/02-vulkan-raw-probe.md:236-258`), but the expert arena weights are created `EXCLUSIVE` (`crates/breccia-engine/src/model.rs:683-700` → `crates/breccia-kernels/src/gemv.rs:992-1018` → `crates/breccia-vk/src/ctx.rs:1062`). A promotion copy on the transfer family into them needs an ownership transfer the code does not do (Q1).
- **The default queue mode is in doubt.** The code picks `DistinctFamily` on family 1 (`ctx.rs:724-742`); doc 01 `:247` and doc 09 `:110-116` say `SameFamilySecondQueue`. The runtime `xfer_mode` print decides which. Neither is a TRANSFER-only (SDMA-style) family (Q1).
- **Decode cannot be prioritised over promotion today.** Both cards advertise `VK_KHR_global_priority` and `VK_EXT_global_priority` (doc 01 `:569-595`, `:637-661`), but no global-priority struct is chained, and every queue's relative priority is 0.0 (`ctx.rs:756`, `pQueuePriorities`). Realtime privilege is UNVERIFIED (Q2).
- **Per-expert host import is the wrong unit.** One import per streamed block (18 total, each about 0.89 GiB, derived) is the granularity to test. Per-expert import is 9,216 `VkDeviceMemory` objects (derived), and the count limit is UNVERIFIED (Q3).
- **A general LRU suballocator is not needed.** Expert blobs are one size per model (`graph.rs:254-255`), so a fixed slot table with an expert→slot index does the same job (derived). `VK_EXT_memory_budget` sizes the arena once, at load (Q4).
- **Radeon 610M: about 0 marginal.** It is `DisplayOnly` and excluded (`crates/breccia-profile/src/role.rs:12`, `:92`), and the crate docs at `crates/breccia-vk/src/lib.rs:19-20` describe an iGPU pin that hides both discrete GPUs (`Host::new` unsets it). It shares DDR5 with a host lane at about 5.9 GB/s (derived) (Q5).
- **Cooperative matrix: settled negative, re-checked.** `VK_KHR_cooperative_matrix` is absent on both cards (probe `:109`, doc 01 `:101`), and the registry has no `VK_KHR_cooperative_matrix2` (Q6).

## 2. Breccia state

- **Queue choice.** `Gpu::create` takes `fam_c` as the first COMPUTE family (`crates/breccia-vk/src/ctx.rs:706`) and `other_x` as the first other family with TRANSFER (`ctx.rs:724`). Family 1 on both cards matches, so `family_x` is 1 with `XferMode::DistinctFamily` (`ctx.rs:737-742`). Relative queue priorities (`pQueuePriorities`, not global priority) are `[0.0; 2]` (`ctx.rs:756`). Nothing in `crates/` names `VkDeviceQueueGlobalPriorityCreateInfo` or a memory-priority struct.
- **Ring.** `TransferRing` uses fixed slots of one expert blob and two timelines, `arrive` and `release` (`crates/breccia-vk/src/transfer.rs:6-27`, `:45-52`). Its slots are CONCURRENT across the transfer and compute families (`ctx.rs:1040-1044`, `:1045-1066`; `transfer.rs:138-139`), so the ring needs no ownership transfers. The ring is not on the engine path: only `breccia-vk` (the module and `transfer_bench.rs`) uses it, and `crates/breccia-kernels/src/batch.rs:14` names it in a doc comment. Engine copies go through `breccia_kernels::batch_cmd::record_copy` (`crates/breccia-engine/src/xfer.rs:267-268`, `:338`), and streamed experts are a CPU memcpy into host-visible slots (`crates/breccia-engine/src/graph.rs:254-255`, `:553`). The ring's arrive/release rule is the model to adopt; the decode path does not use it today.
- **Arena.** Each resident block holds gate, up and down as one buffer per tensor, all 512 experts (`model.rs:683-700`). Those buffers come from `gemv::upload` → `alloc` → `gpu.buf` (`gemv.rs:992-1018`), which passes no families, so they are EXCLUSIVE (`ctx.rs:1062`).
- **Addressing.** Resident experts are addressed as `(*e as u32) * wgate` into the 512-expert buffer; streamed slots use base 0 (`crates/breccia-engine/src/graph.rs:2766-2792`).
- **Budget.** `Gpu::mem_budget` sums heapBudget minus heapUsage over DEVICE_LOCAL heaps (`ctx.rs:987`). The extension is enabled at `ctx.rs:790-792`. Residency is chosen once at load (`crates/breccia-engine/src/arena.rs:1-39`, `model.rs:1398-1420`).
- **Host import.** `probe_host_memory` aligns to `max(minImportedHostPointerAlignment, 4096)` (`crates/breccia-vk/src/transfer.rs:691`). `probe_import` makes one range and one import (`transfer.rs:750-803`). The pinning and `LimitMEMLOCK` rule is `docs/BRIEF.md:144`.
- **Name check.** The round brief named `crates/breccia-engine/src/host_arena.rs`. It does not exist. The host lane is in `graph.rs:553` and `model.rs`.

Conflicts, reported and not resolved here:

- (a) Default queue mode: doc 01 `:247` and doc 09 `:110-116` say `SameFamilySecondQueue`; code `ctx.rs:737-742` and probe `:70-74` give family 1 (COMPUTE|TRANSFER, 4 queues). Gate: runtime `xfer_mode` print.
- (b) `transfer_queue` family index: doc 01 `:186` and doc 09 `:116` say family 3 with 2 queues. The default-env probe shows family 3 as "(none)", 1 queue (`probe:73`). The repo holds no probe recorded under the env knob.
- (c) DRAM bandwidth: `docs/BRIEF.md:16` gives 576 GB/s (spec), `:123` gives 810 GB/s measured, and probe `:143` measures 462.2 GB/s. This doc uses them as a range.
- (d) Host-to-device rate: `docs/BRIEF.md:126` gives 17.9 up + 17.9 down; doc 09 `:125-133` gives 25.4–26.5 H2D against a 28.7 GB/s link.

## 3. Derived numbers

- Host lane rate: 301 MiB × 18.59 tok/s = 5.87 GB/s, approximate. 301 MiB/token is the 30-block residency (`docs/BRIEF.md:79`, `:159`); 18.59 tok/s is a later row (`:73`), so the product mixes two rows of one configuration.
- Copy share of DRAM: 28.7 ÷ 462 = 6.2 % (÷576 = 5.0 %, ÷810 = 3.5 %).
- Expert size: 301 MiB ÷ (18 blocks × 10 routed experts, top-10 per `docs/BRIEF.md:21`) = 1.67 MiB; arena 26.7 GiB ÷ (30 × 512) = 1.78 MiB. The 6 % gap is unexplained.
- Per-block arena: 26.7 GiB ÷ 30 = 0.89 GiB. Per-expert imports: 18 × 512 = 9,216. Streamed bytes if imported: 9,216 × 1.78 MiB = 16.0 GiB.
- Alignment waste at 4 KiB: 4 ÷ (1.67 × 1024) = 0.24 %.
- Efficiency check: 1 − (1.040 − 1.060) ÷ 0.884 = 102.3 %, which reproduces the probe.
- Expert lookups per token: 48 blocks × top-10 (`docs/BRIEF.md:21`) = 480.

## 4. Findings by question

**Q1 — async queues and sync.** Family 0 is the graphics family (1 queue) and family 1 is a 4-queue COMPUTE|TRANSFER family on both cards (`probe:70-77`). RX: compute 1.060 s, copy 0.884 s (about 28.2 GB/s), both 1.040 s, efficiency 102.3 %, repeat 100.0 %. Radeon VII: efficiency 0.8 %, effectively serial (`probe:236-258`). The compute leg is an FMA kernel, which is ALU-bound. It is not a decode-like DRAM-bound load, so DRAM interference from a promotion copy is UNVERIFIED. Derived upper bound on the copy's DRAM share: 28.7 ÷ 462 = 6.2 % (÷576 = 5.0 %; ÷810 = 3.5 %). Inputs: 462.2 GB/s measured and 576 GB/s spec (`docs/phase-0/02-vulkan-raw-probe.md:143`), 810 GB/s (`docs/BRIEF.md:123`), 28.7 GB/s H2D (probe `:213`). Sync: a copy on family 1 that family 0 later reads needs either CONCURRENT buffers across {0, 1} (`buf_with(..., Some(&[0, 1]))`, as `TransferRing`'s slots do in the bench path) or a release/acquire ownership pair with a timeline wait. Spec prose is not on disk (only `vk.xml` and `validusage.json` are), so the ownership rule is stated from general Vulkan practice and is UNVERIFIED as written here. CONCURRENT cost on RADV is UNVERIFIED.

Promotion sequence (design, not measured; the engine has no such path today, and this follows `TransferRing`'s rule): (1) the copy on family 1 writes a free slot and signals `arrive` at value v (`submit_upload`, `transfer.rs:259`); (2) family 0 waits on `arrive >= v` before reading the slot (`submit_compute`, `transfer.rs:354`); (3) the slot is reused only after family 0 signals `release >= v` (`transfer.rs:6-27`); (4) an arena weight written on family 1 and read on family 0 needs a release/acquire pair or a CONCURRENT buffer, and neither exists in the code today.

**Q2 — global priority.** Both cards list the global-priority extensions (doc 01 `:569-570`, `:595`, `:637-638`, `:661`). The enum has REALTIME at `vk.xml:13437`, with `_EXT` and `_KHR` aliases at `:24799` and `:24947`. `VK_ERROR_NOT_PERMITTED_KHR` is aliased at `vk.xml:24942`. Which call returns it, and the privilege REALTIME needs, are UNVERIFIED (no spec prose or driver source on disk). The code requests no priority (`ctx.rs:756`). Priority is set per queue at device creation, so decode on HIGH and copy on LOW is a two-queue test on the existing layout. Whether RADV honours it on these cards is UNVERIFIED.

**Q3 — VK_EXT_external_memory_host.** The alignment is 4096 B and the import type is 5, HOST_VISIBLE|HOST_COHERENT|HOST_CACHED (doc 09 `:36-46`). Import works on anonymous+mlock memory and on pageable mmap (doc 09 `:42`). Maximum single allocation is 4 GiB − 1 (doc 01 `:107`). `maxMemoryAllocationCount` is named at `vk.xml:2185` with no device value on disk: UNVERIFIED. Streamed experts today are memcpy'd from mmap into slots (`graph.rs:553`; 301 MiB/token, `docs/BRIEF.md:79`). Importing them instead needs about 16.0 GiB of anonymous RAM (derived), against a roughly 30 GB budget. Importing the file-backed mmap directly is not in doc 09's list: UNVERIFIED. Recommendation: one import per streamed block (18 imports, each under 1 GiB, derived), never per expert.

**Q4 — budget and LRU.** `VK_EXT_memory_budget` is read at load (`ctx.rs:987`) and drives residency once (`arena.rs:1-39`). It is not a runtime signal. `VK_EXT_memory_priority` is listed on both cards (doc 01 `:574`, `:642`). Doc 01 `:140` plans priority 1.0 on the arena, and `crates/` has no use of it; its effect on RADV eviction is UNVERIFIED. `VK_EXT_pageable_device_local_memory` is absent from both appendices (doc 01 `:554-681`). The registry lists it as depending on `VK_EXT_memory_priority` (`vk.xml:28338`). So, per the recorded appendices, driver paging of device-local memory is not advertised. Design: a slot table with N = arena bytes ÷ slot bytes (derived: 1.67–1.78 MiB per expert), an expert→slot index, and LRU over slots. A slot is reused only after `release >= v` for its last reader, the rule in `transfer.rs:6-27`. A general suballocator adds nothing for one blob size (derived). The index adds one lookup per routed expert: 48 × top-10 = 480 per token (derived). Its cost is UNVERIFIED.

Eviction rule (design, not measured): a slot becomes free only after the last compute that read it has signalled `release` at or above the value of its last upload (`transfer.rs:6-27`). The LRU order lives in host memory as one entry per slot, so each routed-expert lookup is O(1) (derived). A miss costs one upload of one expert blob, 1.67-1.78 MiB (derived), which is the unit the host lane already moves.

**Q5 — Radeon 610M (gfx1036).** Not a compute device: `DisplayOnly` (`role.rs:12`, `:92`, "not benchmarked"). The crate docs at `lib.rs:19-20` record the iGPU pin, `MESA_VK_DEVICE_SELECT=1002:164e!`, and that `Host::new` unsets it. Compute ceiling is 2 CUs × flops/clock/CU × clock; the flops figure and the clock are UNVERIFIED. The memory path is the constraint: the host lane pulls 301 MiB × 18.59 tok/s ≈ 5.87 GB/s (derived, `docs/BRIEF.md:73`, `:79`) from the same DDR5. DDR5 bandwidth is UNVERIFIED. Verdict: about 0 marginal.

**Q6 — cooperative matrix and new compute extensions.** `VK_KHR_cooperative_matrix` is absent on both cards (probe `:109`, `:123`, `:312`; doc 01 `:101`), and neither appendix lists a cooperative extension. In the registry, `VK_NV_cooperative_matrix2` depends on `VK_KHR_cooperative_matrix` (`vk.xml:31193`), and `VK_NV_cooperative_matrix_decode_vector` depends on `VK_NV_cooperative_matrix2` (`vk.xml:32414`). No `VK_KHR_cooperative_matrix2` is in the registry. RADV newer than the installed 26.2.4 is UNVERIFIED: no device query is allowed. Verdict: negative, unchanged.

## 5. Top 3 (with gates)

1. **Give the arena weights a sync contract before any promotion copy.** What: create arena buffers CONCURRENT for families {0, 1}, or add a release/acquire pair per promotion. Moves: correctness of copy-then-read (no speed claim). Evidence: `model.rs:683-700`, `ctx.rs:1045-1066`, `ctx.rs:1062`, probe `:236-258`. Gate: the qualification card prints `xfer_mode` on both cards, reads back a known expert after a promotion copy, and measures decode with a concurrent copy. UNVERIFIED - requires the GPU qualification card.
2. **Fixed-slot tiering with an expert→slot index and timeline-gated reuse.** What: replace the 512-expert buffer per tensor with N slots and an index table. Moves: the 301 MiB/token host lane; each hit saves one streamed expert of 1.67–1.78 MiB (derived). Evidence: `graph.rs:254-255`, `:2766-2792`, `transfer.rs:6-27`, `docs/BRIEF.md:79`, `:159`. Gate: the qualification card measures index cost and hit rate on a real routing trace. UNVERIFIED - requires the GPU qualification card. `VK_EXT_memory_priority` on the slot pool is a hint to test after that, not a lever.
3. **Two-priority split, measured before any use.** What: request HIGH on the decode queue and LOW on the copy queue at device creation. No REALTIME. Moves: decode-versus-copy contention; the copy's DRAM share is at most 6.2 % (derived). Evidence: doc 01 `:569-595`, `vk.xml:13437`, `:24947`, `ctx.rs:756`. Gate: decode tok/s with and without a concurrent promotion copy. UNVERIFIED - requires the GPU qualification card.

Rejected: per-expert host import (9,216 allocations and 16.0 GiB if imported); a general LRU suballocator; the Radeon 610M (about 0); cooperative matrix (settled).

## 6. Overlaps

- Doc 01 Lever 2 (`:245-258`) names the same `xfer_mode` test; Top 3 #1 reuses its test.
- Doc 09 A2 (`:110-135`) makes the default-queue claim in conflict (a); its H2D numbers are not re-costed here.
- Doc 13 (admission and eviction) and doc 19 (MLX and LLM-in-a-flash) are the likely overlaps for Top 3 #2; not read this round.

## 7. UNVERIFIED list

- Device `maxMemoryAllocationCount` value (`vk.xml:2185` names the limit only).
- REALTIME privilege, and which call returns `VK_ERROR_NOT_PERMITTED`.
- RADV behaviour for global priority, memory priority and CONCURRENT buffers; the ownership-transfer rule's wording.
- Whether a file-backed mmap can be imported as host memory (not in doc 09 `:42`).
- Conflicts (a) and (b) until the `xfer_mode` print runs; conflict (c) DRAM rate.
- Radeon 610M flops/clock, clock and DDR5 speed.
- RADV 26.2.x extensions newer than 26.2.4.
- Copy interference with DRAM-bound decode: the phase-0 compute leg is ALU-bound, so this is untested.

## 8. Sources

- `crates/breccia-vk/src/ctx.rs`: 702 `Gpu::create`; 706, 724, 735-742 queue choice; 756 priorities; 790-792 budget extension; 987 `mem_budget`; 1028 `buf`; 1045-1066 `buf_with`.
- `crates/breccia-vk/src/transfer.rs`: 6-27 ring header; 120 `TransferRing`; 259 `submit_upload`; 354 `submit_compute`; 691 alignment; 750-803 `probe_import`.
- `crates/breccia-engine/src/model.rs` 683-700, 1398-1420; `crates/breccia-kernels/src/gemv.rs` 992-1018; `crates/breccia-engine/src/graph.rs` 254-255, 271, 553, 2766-2792; `crates/breccia-engine/src/arena.rs` 1-39.
- `crates/breccia-profile/src/role.rs` 12, 92; `crates/breccia-profile/src/plan.rs` 344-345; `crates/breccia-vk/src/lib.rs` 19-20.
- `docs/phase-0/02-vulkan-raw-probe.md` 63-82 (heaps and families), 109, 123, 236-258 (overlap), 312.
- `docs/research/01-vulkan-and-mlc.md` 101, 107, 123, 139-140, 186, 245-258, 554-681; `docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md` 27-135.
- `docs/BRIEF.md` 16, 73, 79, 123, 126, 144, 159.
- Khronos registry `vk.xml` 2185, 13437, 24799, 24942, 24947, 28338, 31193, 32414.

## 9. Not read

- Mesa RADV and kernel DRM/amdgpu source (not on the machine).
- Khronos spec prose (only `vk.xml` and `validusage.json` are on disk).
- `crates/breccia-engine/src/arena.rs` beyond the header and the cited lines; `graph.rs` and `transfer.rs` beyond the cited lines; `crates/breccia-vk/src/transfer_bench.rs`; `crates/breccia-vk/src/bench.rs`.
- Unmerged worktree branches, names only: `feat/vk-ext-memory-priority`, `feat/memory-priority-optin`, `fix/host-import-arena-cap`, `feat/vk-pipeline-cache`.
- Doc 01 beyond the cited lines; README rows 23-27 in full.
- `harnesses/vk-probe`.
