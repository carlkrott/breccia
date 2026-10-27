# D3: Host expert direct import (VK_EXT_external_memory_host)

Status: design only, offline. This section names no GPU run. It builds on D1 (8182899, `--host-expert-import`, default OFF, no consumer) and D2 (d7ed5e0, `HostArena` in `crates/breccia-engine/src/host_arena.rs`).

## Goal

For the 18 blocks that stream through the host lane (about 301 MiB per token, per the brief), have the GPU read the expert bytes straight from host memory imported with VK_EXT_external_memory_host. This removes the CPU GEMV and the staging copy for those blocks. Bit-identical output is required: the import is an address change, not a numerics change.

## Research constraints (docs/research/09)

- Import works on anonymous mlock'd memory: 28.7 to 29.7 GB/s pinned, 92 to 93% of PCIe (A1.2). Dequant in-kernel runs at 22.4 to 26.4 GB/s (A1.2 table).
- An imported pageable range can be migrated or evicted under the GPU, so pinning is required for correctness (A1.3). This is why D2 locks the arena.
- `minImportedHostPointerAlignment` is 4096. The only importable host memory type is type 5 (HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, system heap) (A1.1).
- Plan-set facts, not re-verified in this stage: file-backed mmap is rejected by amdgpu_ttm.c, libdrm hard-codes ANONONLY, and llama.cpp disables AMD host import because of re-fault cost. This is why the arena is anonymous and D2 copies the blobs out of the GGUF file with pread.

## Engine file map

- `crates/breccia-engine/src/host_arena.rs` (D2, done): `HostArena::new` (alloc_zeroed plus mlock), `fill` (pread from `BlobSource`), `blob(i)`. This is the import source. Its base is 4096-aligned, which satisfies the import alignment. Doc drift: the module comment says `MAP_PRIVATE` + `MAP_LOCKED`, but the code uses `std::alloc::alloc_zeroed` plus `libc::mlock`. Both are anonymous and locked, so behaviour is correct. The comment should be corrected in the D3 commit.
- `crates/breccia-vk/src/ctx.rs`: `Gpu.external_memory_host` (about line 446), extension enabled at about line 772, `find_mem_type` at about line 590. Expected to need no change; the import memory type is chosen from `memory_type_bits` of `vkGetMemoryHostPointerPropertiesEXT`.
- `crates/breccia-vk/src/transfer.rs`: `probe_import` (about line 750) already runs `vkGetMemoryHostPointerPropertiesEXT` and `vkImportMemoryHostPointerInfoEXT` on a probe allocation. D3 generalises this to import an existing `HostArena` range (a new function, not a change to the probe). It returns a `Buf` whose lifetime is bounded by the arena.
- `crates/breccia-engine/src/model.rs`: arena build at about lines 679 to 712 (a block whose allocation fails drops to the host lane) and the arena rule at about line 844. D3 adds: when `--host-expert-import` is set, the host-lane blocks get a `HostArena` fill and an import, instead of the CPU path.
- `crates/breccia-engine/src/graph.rs`: CPU lane call at about line 999 (`cpu::path()`), `ChunkRoute` at about line 2413, `fill_stream_slot` at about line 2516 (the staging copy that imported blocks must bypass), expert blob word-count and base push constant at about lines 2585 to 2595. D3 adds a branch: an imported blob is read from the imported `Buf` at `blob.start`, with no staging.
- `crates/breccia-kernels`: the GEMM must take a base byte offset into an imported buffer. Read `gemm.rs` before deciding. If the change is a wired change (same bindings, base offset from push constant), it is the lowest-risk class under AGENTS.md. Otherwise it is a new entry point and needs the adding-a-kernel protocol.
- `crates/breccia-flags/src/plan.rs` (about lines 70, 288) and `parse.rs` (about line 177): the flag exists, default OFF. No change.

## LimitMEMLOCK

- The box limit is 8 MiB soft and hard (research/09 header). Any arena above a few MiB fails `mlock`. D2's error text already names LimitMEMLOCK and the streaming fallback.
- Requirement: `LimitMEMLOCK=infinity` in the service unit (plan set D5), or an arena size bounded to the granted limit. The engine must not be run as root to get CAP_IPC_LOCK.
- Size: pinned bytes = page-rounded sum of the imported blobs. This document does not measure it. D3 must print `pin_bytes` at load. Placement should limit imports to the miss set (research A1.3), not the whole host lane, because a 58 GB host should not carry a large locked commitment by default.

## Parity plan

- Gate: 32/32 greedy tokens and top-1 13.170510 at chunk 8, 16, 32 and 128 (docs/phase-4/01-forward-pass.md). The imported path must match the host-lane path bit for bit with identical weights.
- Order of work: (1) compile-only check offline (`cargo check -p breccia-engine`). (2) First GPU run on the MI50 (headless, PCI 0000:0e:00.0), one process, `timeout 90`, validation layer on, at most 4 fresh dispatch shapes. (3) RX 6950 XT (display card, 0000:03:00.0) only after the MI50 passes, and only with the supervisor's window. Any GPU step first needs a `GPU WINDOW NEEDED:` comment stating what runs, which device, the time box and the abort criteria.
- Measurement: short form, a 64-token prefix and 8 to 16 decode steps, A/B/A with the flag.

## Abort criteria

- Any GPU fault line in the kernel log during a run: stop, capture the journal, report. Do not re-run.
- `mlock` failure at load: degrade to the host lane (D2 behaviour). Do not abort.
- Parity mismatch: the flag stays OFF. Do not enable it by default.

## Open items for the D3 implementation

- Confirm the plan-set facts about amdgpu_ttm.c, libdrm ANONONLY and llama.cpp, which this design takes as given.
- Choose which blocks are imported: the 18 host-lane blocks, or the miss set only.
- Correct the `host_arena.rs` module comment (`MAP_LOCKED` wording) in the same commit as the first D3 code change.
- Not measured here: pinned bytes for the expert set, the import throughput on this engine's GEMM, and any parity result.
