# 15 — LMCache and Mooncake: tiered KV caching and transfer, audited against Breccia's KV layout

Round 15. Closes README row 21. Section A is the Vulkan capability half of this round. Sections B–E are the upstream anatomy and the Breccia adaptation.

**Engines audited (read-only, pinned).** LMCache at `6993f46c91f182def4e73b7882177e8881671301`; Mooncake (kvcache-ai) at `2be3d4a07ceb9af895053c0098888962d19d581e`. Upstream paths are repo-relative to those trees; Breccia paths are repo-relative to this repo. Nothing was built or run on a GPU, and no benchmark was run.

**Label note.** "Cheap-offline" means CPU-only arithmetic or a unit check. "GPU-window" means it needs the card in a quiet window. Anything that needs the card is marked `UNVERIFIED - requires the GPU qualification card`.

**Headline.** Six findings decide the design.

1. **Breccia's two cache classes have different granularities, so LMCache's 256-token chunk cannot be the unit for GDN state.** Causal KV uses 16-cell pages (`crates/breccia-kernels/src/qsa/mod.rs:89`). GDN state is one fixed snapshot per layer: 3 MiB of delta-rule state plus a 120 KiB conv history (`crates/breccia-kernels/src/gdn/mod.rs:79–83`). LMCache hashes and stores per 256-token chunk (`lmcache/v1/multiprocess/token_hasher.py:85`) and gives recurrent state its own object group (`lmcache/v1/kv_layer_groups.py:218, 689–716`). At 256 tokens a 100K prefix needs 390 GDN snapshots, 42.7 GiB, against 673 MiB at I=16K (doc 13 C5, `docs/research/13-batch-manager-admission-eviction.md:135–136`). The snapshot interval must be set separately from the page size.

2. **The in-tree q8_0 V page is 9 KiB per layer, which is not a multiple of 4 KiB.** A 16-cell V q8_0 page per layer is 16 × 2 heads × 8 blocks × 36 B = 9,216 B (`qsa/mod.rs:169–172, 175–187`), which is 2.25 × 4096. The K f16 page (16 KiB) and the GDN snapshot (798 × 4096 B per layer) are aligned. Mooncake uses 4096 as the O_DIRECT alignment for file offset, size and buffer (`mooncake-store/include/storage_backend.h:1106`; applied per I/O at `mooncake-store/src/storage_backend.cpp:1863–1867`), and an imported host base pointer needs 4096 (`docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md:39`). The fix is one of three: padding V to 12 KiB per layer-page (+33 % V bytes); group 4 pages (64 tokens) per object, which is 36,864 B = 9 × 4096 B and needs identity page mapping inside the group (`qsa/mod.rs:27`); or write page-major records, one per 16-cell page holding all 12 QSA layers, so V q8_0 is 110,592 B = 27 × 4096 B and K f16 plus V q8_0 is 300 KiB = 75 × 4096 B, with no padding. Whether page-major needs a kernel change depends on the copy-offset rule (Layout constraints, UNVERIFIED). If copies need 4096-aligned offsets, the VRAM pool must also go page-major, which needs a layer-stride change in the kernel addressing (`qsa/mod.rs:26`). If they do not, page-major host records work with the per-layer VRAM pool and the kernel unchanged.

3. **A restorable prefix is the GDN state plus conv history, the indexer's pooled keys and tail, and the KV pages. KV pages alone are not enough.** The 112.2 MiB snapshot does include conv: 36 × (3,145,728 + 122,880) B = 117,669,888 B = 112.2 MiB (`gdn/mod.rs:101–102`). The indexer keeps pooled keys and a 3-row tail, not raw keys (`qsa/mod.rs:37–39`), so the pooled keys are state and cannot be recomputed from KV pages. They cost 128 B per cell in f32, about 146.5 MiB per 100K tokens across 12 QSA layers (`qsa/mod.rs:197–199`).

4. **LMCache's disk path falls back to buffered I/O silently on write and after a warning on read, so its disk numbers can measure the page cache.** `write_file` drops to a plain buffered write with no log line when the size is unaligned or `use_odirect` is off (`lmcache/v1/storage_backend/local_disk_backend.py:781–783`). `read_file` warns at `:803–805` and then reads buffered at `:810–811`. Its own doc says direct I/O is "Required to measure real disk bandwidth" (`docs/source/mp/l2_storage/fs_native.rst:21–22`). Breccia must fail loudly on misalignment.

5. **Neither project's tier runs on this machine without a torch, CUDA/HIP or RDMA runtime, except Mooncake's file store, which is portable as ideas only.** Mooncake's `mooncake-store/src/posix_file.cpp` includes only POSIX headers and glog. Its transfer engine guards CUDA/MUSA/MACA (`mooncake-transfer-engine/include/transfer_engine.h:296`) and prefers local memcpy over TCP loopback for same-host transfers (`transfer_engine.h:312–318`). LMCache's tier code imports torch. Mooncake's RDMA and TCP paths have no single-box analogue here. Port nothing; adopt storage-layer ideas only.

6. **The one design worth building is a snapshot-boundary prefix cache with a file tier. Its prize is UNVERIFIED end-to-end.** The NVMe read of a 100K-token restore set (q8 KV, GDN snapshot and pooled keys, 2.19 GB) is about 0.37 s at the BRIEF 5.9 GB/s (C4), against 4.3–4.6 h of re-prefill (doc 13 C5). The host-to-device step depends on the card. The q8 KV set uploads in 66–87 ms on the RX (22–29 GB/s, RX only), but in about 0.66 s on the MI50 at about 2.9 GB/s (`docs/BRIEF.md:126`), and the per-card layer split is UNVERIFIED. The region-count overhead, the real disk read and the hit rate are `UNVERIFIED - requires the GPU qualification card`.

**Standing verdicts respected.** Quantised GEMV is instruction-issue bound (doc 12). KV quantisation buys capacity, not speed (`docs/BRIEF.md` §7). The RX 6950 XT executes host-resident experts at 22–29 GB/s (doc 09 measures the RX only). Imported host pointers need 4096-byte alignment (doc 09:39), and pinning needs `LimitMEMLOCK=infinity` (doc 09:99–101). Neither card has `VK_KHR_cooperative_matrix` (doc 06:238,352). The MI50 is gen3 x4 (`docs/BRIEF.md` §5, rule 6). The SATA SSD is 0.39 GB/s and the system NVMe about 5.9 GB/s at QD8 (`docs/BRIEF.md` §6). Weights never go on NVMe. T3 never shares gpu-manager's 4d kv-slots/ directory, whose prune deletes `*.bin`.

## A. Vulkan capability angle

What the GPU side already measures (doc 09, as cited in doc 14 A): GPU dequant GEMV reading host-pinned experts runs at f16 26.4, q8 25.5, q4 22.4 GB/s; copy-engine ring DMA at 28.7 GB/s; pinned host at 28.7–29.7 GB/s (`docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md:58, 71–78`).

**Decisive check.** One check decides whether either project's storage tier can run beside Breccia's Vulkan path without a GPU runtime: list every dependency of the T2/T3 code path. Done this round as a grep over the pinned trees (cheap-offline, no GPU):

1. Mooncake file store. `mooncake-store/src/posix_file.cpp` includes only POSIX headers (`<cerrno>`, `<sys/uio.h>`, `<unistd.h>`) and glog. The store headers (`storage_backend.h`, `eviction_strategy.h`, `local_hot_cache.h`) have no CUDA, HIP or RDMA include. The grep hits for those names inside `storage_backend.h` are substring false positives (e.g. `ShardMaps`).
2. Mooncake transfer engine. The CUDA guard sits at `transfer_engine.h:296`, and the RDMA, TCP and CXL transports are behind the multi-protocol switch. The HIP transport (`mooncake-transfer-engine/include/transport/hip_transport/hip_transport.h`) is AMD HIP, not Vulkan. The engine has no Vulkan transport.
3. LMCache tiers. `local_disk_backend.py`, `local_cpu_backend.py`, `lazy_memory_allocator.py`, `kv_codec/asym_k16_v8.py` and `kv_codec/encoded_kv.py` all import torch. The CPU tier imports torch and gates its hot cache on torch devices (`lmcache/v1/storage_backend/local_cpu_backend.py:10, 57, 63`).

**Result.** The Mooncake file store is C++ with no GPU dependency, so its ideas port to Rust but no code is reused. The transfer engine adds nothing to the single-box path beyond memcpy. The LMCache tiers cannot be used as they stand. Verdict: port nothing, adopt the storage ideas (chunk identity, object groups, O_DIRECT discipline, owner lock).

The Vulkan-specific constraint that neither project addresses is host-import alignment. A T2 slab that the Vulkan side imports must be 4096-aligned (doc 09:39). Neither project has a Vulkan import path, so Breccia has to enforce this itself. The end-to-end check of an O_DIRECT-filled slab imported into Vulkan is `UNVERIFIED - requires the GPU qualification card`.

## B. Upstream anatomy (pinned SHAs)

**B1 LMCache hashing.** `TokenHasher` defaults to `chunk_size=256` and blake3 (`lmcache/v1/multiprocess/token_hasher.py:85`). Each chunk hash chains the prefix hash with the packed chunk (`:34–55`). Only complete chunks are hashed: `num_complete = effective_len - effective_len % chunk_size` (`:267`), so a partial trailing chunk is dropped. Session asserts chunk alignment (`lmcache/v1/multiprocess/session.py:102–106, 120–124`). Lookup does not assert; it floor-divides the start into a chunk index (`modules/lookup.py:66`).

**B2 LMCache object groups.** `KernelGroupInfo` carries `recurrent_state` (`lmcache/v1/kv_layer_groups.py:159, 218`). Defaults are `lmcache_tokens_per_chunk=256` and `separate_object_groups=False` (`:339–340`). The full-sliding-window shortcut excludes recurrent groups (`:622`). `_detect_object_groups` states that "Recurrent pages and SW attention KV never share an object" (`:715–716`). Chunk divisibility is enforced: `lmcache_tokens_per_chunk % tokens_per_block != 0` raises (`:774–777`). The older single-group path is deprecated with the message "It does not have hybrid model support" (`:545`).

**B3 LMCache disk.** `use_odirect` defaults to off in the local disk backend (`lmcache/v1/storage_backend/local_disk_backend.py:167, 170`); the O_DIRECT open is at `:785` (write) and `:814` (read). The newer distributed config defaults direct I/O on (`lmcache/v1/distributed/config.py:418`) and `align_bytes` to 4096 (`:108, 312`). The two defaults disagree, so the distributed path is the one to read if LMCache's disk tier is revisited.

**B4 LMCache codecs.** `kv_codec/asym_k16_v8.py` keeps K native and quantises V to FP8 e4m3fn, with per-tensor, per-layer-head or per-page-head scales (docstring `:2–13`; `compute_v_scales` `:69`). `kv_codec/encoded_kv.py:131–153` (`CodecHashes`) binds model revision, tokenizer, RoPE config, attention backend and KV layout into the cache identity, with the "cache-poisoning gates" note at `:173`.

**B5 LMCache blend index.** `lmcache/v1/mp_coordinator/blend_index.py:1–7` describes a strided rolling-hash probe whose candidates are "verified token-exact". The verifier is `matcher.py`, which was not read, so cross-prefix KV exactness is UNVERIFIED. `BlendIndex` is at `:97`, with `chunk_size=256, probe_stride=1` at `:105`.

**B6 Mooncake file store.** `FileStorageConfig` (`mooncake-store/include/storage_backend.h:206`): `total_size_limit` defaults to 2 TB (`:224–225`); `use_uring=false` (`:235`); disk watermark eviction is on, high 0.90 and low 0.80 (`:242–244`). `BucketStorageBackend` (`:760`) sets `kDirectIOAlignment = 4096` (`:1106`), applied per I/O to the aligned buffer, the read range and the write size (`mooncake-store/src/storage_backend.cpp:1540, 1863–1867, 2541–2554`), an owner lock fd (`:1119`), and `.bucket` and `.meta` suffixes (`:1120–1121`). Its aligned-buffer constant is `32 * 1024 * 1024` with the comment "16MB" (`:1125`). The comment and the value disagree, so do not copy the constant without re-checking it.

**B7 Mooncake master and cache.** `MasterConfig` (`mooncake-store/include/master_config.h:95–222`): `offload_on_evict = false` (`:185`), `promotion_admission_threshold = 2` (`:193`), `promotion_max_per_heartbeat = 1` (`:200`). `LocalHotCacheConfig` (`mooncake-store/include/local_hot_cache.h:22–26`): 16 MiB blocks and `admission_threshold = 2`. The SSD prefetcher is best-effort and triggered out of band by exist probes (`mooncake-store/include/ssd_prefetcher.h:1–3`), with 4 threads (`:109`). LRU eviction pops the list tail (`mooncake-store/include/eviction_strategy.h:67–75`).

**B8 Mooncake transfer engine (no single-box analogue).** `TransferRequest` intent values are `kUnspecified`, `kForegroundGet`, `kBackgroundPrefetch` and `kMigration` (`mooncake-transfer-engine/include/transfer_engine.h:41–46`). `priority` is ignored by the classic engine (`mooncake-transfer-engine/include/transport/transport.h:60–79`). "RDMA success means queued on the notification QP, not remote delivery" (`transfer_engine.h:277`). "A wait timeout does not cancel the transfer" (`transfer_engine.h:221–223`). These describe RDMA and cross-host behaviour. On one box the engine is a local memcpy, so none of B8 applies to Breccia's T2 or T3.

## C. Breccia adaptation sketch

**C1 Sizes in the tree** (cheap-offline, from the constants at `qsa/mod.rs:71–81, 89, 100–101, 161–220` and `gdn/mod.rs:59–83`).

Per token, across the 12 QSA layers: K f16 is 12 KiB, V f16 is 12 KiB, and V q8_0 is 6.75 KiB (2 heads × 8 blocks × 36 B × 12). K f16 plus V f16 is 24 KiB per token; K f16 plus V q8_0 is 18.75 KiB per token.

| Object | Per layer | Per 16-cell page (12 layers) | Per 100K tokens (6,250 pages) | 4096-aligned |
|---|---|---|---|---|
| K f16 page | 16 KiB | 192 KiB | 1.14 GiB | yes |
| V f16 page | 16 KiB | 192 KiB | 1.14 GiB | yes |
| V q8_0 page | 9 KiB (9,216 B) | 108 KiB | 0.64 GiB | no (2.25 × 4096) |
| Indexer pooled keys (f32) | 512 B per 4-cell block | 2 KiB | 146.5 MiB (12 layers) | no per page; yes per 32 cells |
| GDN snapshot (state + conv) | 3,268,608 B (798 × 4096) | per sequence, not per page | 112.2 MiB per snapshot; 673 MiB at I=16K | yes |

Totals per 100K tokens: f16 K plus V is 2.29 GiB (2.46 GB); K f16 plus V q8_0 is 1.79 GiB (1.92 GB). The pooled keys add 146.5 MiB to either.

**Layout constraints.**
- Alignment, as far as the tree shows. Mooncake's storage backend uses 4096 for the O_DIRECT aligned buffer, the read range and the write size (`mooncake-store/src/storage_backend.cpp:1540, 1863–1867, 2541–2554`; `mooncake-store/include/storage_backend.h:1106`). An imported host base pointer needs 4096 (doc 09:39). Copy-region offsets inside an already-imported buffer have no alignment statement in the tree: the local Vulkan registry (not in the repo) gives only "Specified in bytes" for `VkBufferCopy`, lines 1528–1531, and the spec text is not here. That rule is UNVERIFIED against the spec. Until it is checked, align every copy offset to 4096; under that rule page-major host records need a page-major VRAM pool, which is the kernel change.
- The V q8_0 layer page fails. Option (a) pads each layer-page to 12 KiB: +3 KiB per layer-page, which is +219.7 MiB per 100K tokens (+33 % of V q8_0 bytes). Option (b) groups 4 pages (64 tokens) per object: 36,864 B per layer, aligned with no padding, but it needs identity page mapping inside the group (`qsa/mod.rs:27`). Option (c) writes page-major records: one record per 16-cell page holding all 12 QSA layers. V q8_0 is 110,592 B = 27 × 4096 B, and K f16 plus V q8_0 is 300 KiB = 75 × 4096 B, aligned with no padding. Each record is one page, so it needs no grouping. Whether it also needs the kernel change at `qsa/mod.rs:26` depends on the copy-offset rule above (UNVERIFIED): if sub-offsets must be 4096-aligned, the VRAM pool must be page-major too, which needs the layer-stride change; if not, the per-layer VRAM pool and the kernel stay as they are.
- Pooled keys in 32-cell groups (8 blocks, 4096 B) are aligned. Per-page groups are not.
- Object boundaries must be multiples of the 16-cell page and of the 4-cell indexer block. The snapshot interval must satisfy I ≡ 0 (mod 16). LMCache's equivalent check is `kv_layer_groups.py:774–777`.

Cheap-offline check: a unit test asserting `% 4096 == 0` on each record slab size and offset, for the padded, grouped and page-major layouts. Page-major's per-layer sub-offsets (L × 9,216 B) fail that test, so the test must state whether it checks sub-offsets, which is the copy-offset question (UNVERIFIED). No GPU.

**C2 What to take from each project (proposed; no flag or module exists in tree).**
- Chunk identity: a page-chain hash over 16 token ids, `hash(prev, 16 token ids)` (doc 13 C5), seeded with model revision, tokenizer, RoPE config, KV layout and V layout (`CodecHashes`, `encoded_kv.py:131–153`). Hash only at snapshot boundaries.
- Write at the boundary, not at eviction. Mooncake defaults `offload_on_evict` to false (`master_config.h:185`), and Breccia should not make eviction the save point.
- Admission: write a boundary to T2 only after its second hit (`local_hot_cache.h:26`; `master_config.h:193`), so one-off prefixes do not fill the ~30 GB pinned tier. The value 2 is a borrowed default; the right threshold for Breccia's workload is `UNVERIFIED - requires the GPU qualification card`.
- Fail loud: any misaligned or short O_DIRECT read or write is an error, not a buffered fallback (B3).
- T3 layout: one object per snapshot, in its own directory (`crates/breccia-kv` as planned in `docs/reference-engines/02-kv-and-context.md` §11.3, or a directory name chosen then). It never shares gpu-manager's 4d kv-slots/ directory. Use a suffix other than `.bin`, and an owner lock fd like Mooncake's (`storage_backend.h:1119–1120`), so a second process cannot share the directory.
- Prefetch: a best-effort read started when the request is queued, the SSD prefetcher idea (`ssd_prefetcher.h:1–3`).

**C3 Sketch.** T1 VRAM (in tree) holds the live pages and state. T2 pinned host (planned, about 30 GB) holds boundary snapshots: a header, GDN slabs, V pages in the layout chosen in R2 (padded, grouped or page-major), pooled keys and the page-table identity. T3 NVMe (planned, 100 GiB cap) holds the same record shape, written at boundaries and read with O_DIRECT. A hit restores GDN state plus conv, the pooled keys and tail, and the KV pages up to the boundary; the rest of the prompt is prefilled as today. A hit needs an exact boundary match. There is no splice (section E).

**C4 Numbers.** Every row is cheap-offline arithmetic from C1 and the cited inputs. The 22–29 GB/s rows are the RX 6950 XT's measured range (doc 09, `docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md:58, 71–78`); the MI50 rows use the BRIEF link figure (`docs/BRIEF.md:126`, not re-measured). The per-card layer split is UNVERIFIED: `crates/breccia-engine/src/placement.rs:6–8` assigns each block's dense state and KV to its device, and the production split is not in the tree (`--split-blocks` at `crates/breccia-engine/src/model.rs:304, 394–399`; default `Placement::balanced` at `model.rs:401`). Every end-to-end figure is `UNVERIFIED - requires the GPU qualification card`.

| Term | Input | Result |
|---|---|---|
| KV restore, K f16 + V q8_0 (1.92 GB) to the RX | 22–29 GB/s (doc 09, RX only) | 66–87 ms |
| KV restore, f16 (2.46 GB) to the RX | 22–29 GB/s (doc 09, RX only) | 85–112 ms |
| GDN snapshot (112.2 MiB) to the RX | 22–29 GB/s (doc 09, RX only) | about 4.1–5.3 ms (matches doc 13 C5) |
| Pooled keys (146.5 MiB) to the RX | 22–29 GB/s (doc 09, RX only) | 5.3–7.0 ms |
| KV restore, K f16 + V q8_0 (1.92 GB) to the MI50 (if all layers sit there) | ~2.9 GB/s, MI50 gen3 x4 (`docs/BRIEF.md:126`) | about 0.66 s |
| KV restore, f16 (2.46 GB) to the MI50 (if all layers sit there) | ~2.9 GB/s | about 0.85 s |
| GDN snapshot (117.7 MB) to the MI50 (if all layers sit there) | ~2.9 GB/s | about 41 ms |
| Full restore set (2.19 GB) to the MI50 (if all layers sit there) | ~2.9 GB/s | about 0.76 s |
| T3 read of the q8 record (1.92 GB) | system NVMe, 5.9 GB/s at QD8 (BRIEF §6) | 0.33 s |
| T3 read of the full restore set: q8 KV 1.92 GB + GDN 117.7 MB + pooled keys 153.6 MB = 2.19 GB | system NVMe, 5.9 GB/s | 0.37 s |
| T3 read of the f16 record (2.46 GB) | system NVMe, 5.9 GB/s | 0.42 s |
| T3 read of the q8 record on SATA | 0.39 GB/s (BRIEF §6) | 4.9 s |
| Per-region copy overhead if each (layer, pool, page) is one region | 150,000 regions per 100K tokens; 110–140 ns RX and 300–360 ns MI50 per region (`docs/research/05-ple-io-and-tensorrt-llm.md:25–26`) | RX 16.5–21 ms; MI50 45–54 ms |
| Re-prefill of 100K tokens | doc 13 C5 | 4.3–4.6 h |

The region count is 6,250 pages × 12 layers × 2 pools (K and V, separate pools at `qsa/mod.rs:279–281`) = 150,000. Grouping 4 V pages cuts the V regions by four. The per-region cost is doc 05's figure (`docs/research/05-ple-io-and-tensorrt-llm.md:25–26`) and is not measured for this copy shape: `UNVERIFIED - requires the GPU qualification card`. The NVMe bandwidth term alone is under half a second. With page-major records and a page-major VRAM pool, the copy count falls to one region per page per pool: 6,250 per 100K if K and V are one record in one VRAM buffer, 12,500 if they are two. With the per-layer VRAM pool in tree, each page still copies 12 layer-pages per pool, so the count stays near 150,000 (R2).

## D. Ranked recommendations (five; no sixth earned)

**R1 — Snapshot-boundary prefix cache, with the GDN interval set apart from the 16-cell page (cheap-offline design; GPU-window for hit rate).**
- Effect: prefix reuse at 16K boundaries (673 MiB per 100K prefix) instead of 42.7 GiB at 256-token chunks. This is the "more prefill via prefix reuse" target.
- Measure: cheap-offline. For I in {4K, 16K, 64K}, assert I % 16 == 0 and print snapshot count and bytes for a 100K prefix. No GPU. GPU-window: prefill tokens saved on a real repeated-prompt workload, `UNVERIFIED - requires the GPU qualification card`.
- Risk: the hit rate depends on prompt boundaries. Restore must be exact (state, conv, pooled keys and tail); the test is that a restored run matches a cold run token for token. No parity risk if restore is exact.
- Touches: `docs/research/13-batch-manager-admission-eviction.md:167–168` (design item 5); `crates/breccia-kernels/src/gdn/mod.rs:79–102`; `crates/breccia-kernels/src/qsa/mod.rs:197–199`; planned `crates/breccia-kv`.
- Gated on: nothing for the design. The GPU-window hit-rate measure needs R2.

**R2 — Alignment-correct record layout: grouped V, 32-cell pooled keys, or page-major records (cheap-offline for sizes; GPU-window for copy count).**
- Effect: each option removes the padding. Grouped V (4 pages) cuts the V regions by four. Page-major (one record per 16-cell page, all 12 QSA layers) is aligned without grouping, but its copy-count gain needs a page-major VRAM pool: 6,250 regions per 100K if K and V are one record in one VRAM buffer, 12,500 if they are two, against 150,000 at per-layer granularity. With the per-layer VRAM pool in tree, a restore still copies 12 layer-pages per page per pool, so the copy count stays near 150,000 and only the host side gains, provided per-layer sub-offsets need no 4096 alignment (UNVERIFIED).
- Measure: cheap-offline unit test asserting every record slab size and offset is a multiple of 4096, for each I in R1's set and for each of the three layouts. Compare against the padding alternative, +219.7 MiB per 100K tokens. GPU-window: copy count and restore time per layout, `UNVERIFIED - requires the GPU qualification card`.
- Risk: 64-token object granularity needs identity page mapping inside the group (`qsa/mod.rs:27`). A non-identity page table must fall back to single-page objects, and that fallback needs its own test. Page-major with a page-major VRAM pool changes the attention index formula (`qsa/mod.rs:26`); page-major host records with the per-layer pool do not, if sub-offsets need no alignment (UNVERIFIED). Either way page-major needs a parity run before it lands. Grouped V leaves the kernel unchanged.
- Touches: `crates/breccia-kernels/src/qsa/mod.rs:26` (index formula, page-major VRAM pool only) and `:161–220` (`sizes` module); the planned record writer.
- Gated on: nothing for grouped V or the padded layout; page-major needs its parity run, and the kernel change only if the VRAM pool goes page-major.

**R3 — T3 NVMe snapshot tier: own directory, fail-loud O_DIRECT, identity-sealed hash (cheap-offline for directory and hash tests; GPU-window for restore time).**
- Effect: the NVMe read of a 100K-token restore set is about 0.37 s (C4), against 4.3–4.6 h of re-prefill. The host-to-device step adds 66–87 ms for the q8 KV set on the RX (doc 09, RX only), or about 0.66 s for the q8 KV set and 0.76 s for the full set if the layers sit on the MI50 (BRIEF link figure; per-card split UNVERIFIED). End-to-end is `UNVERIFIED - requires the GPU qualification card`.
- Measure: cheap-offline. Test that the T3 root is never equal to or inside gpu-manager's kv-slots/ directory, that suffixes avoid `.bin`, and that a misaligned read is an error. GPU-window: restore time per 16K boundary, including region overhead, `UNVERIFIED - requires the GPU qualification card`.
- Risk: write wear from writes at every boundary (C2 limits writes to boundaries). The 100 GiB cap should evict on Mooncake-style watermarks (0.90 high, 0.80 low; `storage_backend.h:243–244`) rather than hard-stop.
- Touches: planned `crates/breccia-kv`; `docs/reference-engines/02-kv-and-context.md` §11.2–11.3.
- Gated on: R1 and R2 cheap-offline checks passing.

**R4 — Keep V q8_0 as the capacity lever; do not add an FP8 codec (byte table done in C1; GPU-window for the context ceiling).**
- Effect: at fixed VRAM, V q8_0 saves 5.25 KiB per token across 12 layers (12 KiB f16 V minus 6.75 KiB q8_0 V), about 0.50 GiB per 100K tokens (C1: 2.29 GiB f16 K plus V, against 1.79 GiB). Cross-check: `docs/BRIEF.md:169` measured 932 → 764 MiB per agent at 32K, a drop of 168 MiB, which equals 5.25 KiB × 32,768 tokens. That is capacity, not speed (BRIEF §7).
- Measure: GPU-window: the largest context that fits with f16 V and with q8_0 V, `UNVERIFIED - requires the GPU qualification card`.
- Risk: q8_0 V changes numerics relative to f16 V (reference at `crates/breccia-kernels/src/qsa/cpu.rs:55–78`). The parity gate for that path is not covered here.
- Touches: `crates/breccia-kernels/src/qsa/mod.rs:110–126` (`VLayout`, `BRECCIA_KV_Q8V`).
- Gated on: nothing.

**R5 — Write snapshots at the boundary, admit on second hit, prefetch on queue (GPU-window).**
- Effect: keeps one-off prefixes off the ~30 GB pinned tier and hides the T3 read behind queue wait.
- Measure: cheap-offline: replay a prefix list through an LRU at admission thresholds 1 and 2 to get hit counts, if a prompt log exists (not in tree; UNVERIFIED). GPU-window: T2 hit rate and queue-to-restore time, `UNVERIFIED - requires the GPU qualification card`.
- Risk: the threshold of 2 is borrowed from Mooncake and LMCache, not measured for Breccia.
- Touches: planned `crates/breccia-kv`; the scheduler design items in doc 13.
- Gated on: R3.

## E. What not to copy

1. **LMCache's 256-token chunk as the GDN unit** (`lmcache/v1/multiprocess/token_hasher.py:85`; `lmcache/v1/kv_layer_groups.py:339`). It costs 42.7 GiB per 100K prefix at I=256, against 673 MiB at I=16K.
2. **CacheBlend-style non-prefix splicing** (`lmcache/v1/mp_coordinator/blend_index.py:1–7`). Recurrent GDN state cannot be spliced at a non-prefix offset without recomputing from there, and that breaks bit parity. Cross-prefix exactness is UNVERIFIED (`matcher.py` not read). Reject.
3. **Silent or warned buffered fallback for O_DIRECT** (`local_disk_backend.py:781–783, 803–811`). Fail loud (B3).
4. **Torch-based tier code** (`lmcache/v1/storage_backend/local_cpu_backend.py:10, 57, 63`; `lazy_memory_allocator.py`). Breccia has no torch, so these tiers are not portable.
5. **The FP8 K16/V8 codec** (`lmcache/v1/kv_codec/asym_k16_v8.py:2–13, 69`). Breccia has no FP8 path, FP8 hardware on either card is UNVERIFIED, and a second scale tensor would reintroduce the V alignment problem (C1). KV quantisation gives capacity, not speed (BRIEF §7).
6. **Mooncake's transfer-engine transports** (`mooncake-transfer-engine/include/transfer_engine.h:296, 312–318`; `transport/hip_transport/`). RDMA, TCP, CXL, NVLink and HIP have no single-box analogue on this machine, and the same-host path is local memcpy.
7. **Mooncake's default sizes** (`mooncake-store/include/storage_backend.h:224–225`, 2 TB; `master_config.h:95–222` for the master). Breccia's cap is 100 GiB.
8. **The `kAlignedBufferSize` constant** (`storage_backend.h:1125`, value 32 MiB, comment "16MB"). Do not copy it until the mismatch is resolved or re-derived.
9. **Sharing gpu-manager's 4d kv-slots/ directory, or a `.bin` suffix.** Its prune deletes `*.bin`.
10. **Offload on eviction as the save point** (`master_config.h:185`). Breccia writes at boundaries (C2).
