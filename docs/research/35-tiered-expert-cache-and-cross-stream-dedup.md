# 35 — Tiered expert cache and cross-stream dedup: the 30 GiB cap binds under uniform routing, the page hint is clamped, and cross-stream dedup is small

Round 35. Docs only: no code changed, no build, no GPU or model process run. Base commit a6c53f6. Every gain below is UNVERIFIED until the GPU qualification card runs. Arithmetic is labelled derived where it is one, and a bound where it assumes uniform routing. Commits named as off base were read with `git show`.

Citations are repo-relative; "doc NN" means the `docs/research` file with that number unless a path is given. Kernel facts come from the Linux v6.18 source text fetched in this round; line numbers are given only where the fetched text showed them. GB is 10^9 bytes; GiB is 2^30 bytes.

## 0. Verdict

1. **Whether the 30 GiB cap binds depends on routing skew, which is unmeasured for decode.** The 39 streamed blocks total 34,589,900,800 B (32.21 GiB, 34.59 GB) by GGUF tensor sizes, against a cap of 32,212,254,720 B (30 GiB). Under uniform routing no policy beats the cached fraction: 93.13% hits and 6.87% misses at zero headroom, which is 46.4 MB of disk reads per stream-token (a bound, derived). One prefill point (about 130 of 512 experts per block for a 128-token chunk; `docs/phase-4/02-untried-levers.md` `:472–473`) implies a window working set of 8.18 GiB, inside the cap. The uniform figure is a bound, not an expected value.
2. **The page hint is clamped, so it is ineffective, not thrashing.** In v6.18, `force_page_cache_ra` caps each WILLNEED at `max(bdi->io_pages, ra->ra_pages)` pages. The btrfs bdi has `read_ahead_kb` 128, so `ra_pages` is 32 pages (128 KiB); `io_pages` is not visible in sysfs. The 117 WILLNEED calls per Model therefore read 128 KiB each at least, and 2 MiB each only if `io_pages` were 512 (unverified): 15.3 MB to 245.4 MB per Model, 0.04% to 0.71% of the 34.59 GB the calls name. The DONTNEED half drops nothing while the mapping holds the pages. The hint is off by default (commit 0c2c8e6, not in base).
3. **Cross-stream dedup is small.** Under independent uniform routing the two streams share 1.95% of their expert picks per block. That saves 13.2 MB per token pair, 0.98% of 1,351.2 MB. Both Models map the same shards, so the page cache already shares disk reads; a shared reader would save only lane DRAM passes. The saving reaches 25% of pair bytes only at 50% shared picks, and 50% only at identical routing.
4. **Frequency pinning and LFU tie LRU under uniform routing.** They can win only if a decode routing trace shows skew, so card C1 (the trace) comes first.
5. **Residency +4 blocks (C2) is the only lever here that fits under uniform routing.** It leaves 0.28 GiB of cap spare at zero headroom, and its VRAM cost is not measured in this round.
6. **Ranked flag-OFF cards: C1 to C6** (§6). None has a measured gain.

## 1. As built (base a6c53f6)

- **Two streams, one Model each.** The bench token path (`crates/breccia-cli/src/main.rs:1686–1700`) loads one Model per stream: `devices: vec![dev.clone()]` and `ArenaSpec::List(expert_sets[s].clone())` at :1695. The agents path loads a Model per stream index (`main.rs:2003–2011`; `ArenaSpec::List(groups[s].clone())` at :2011).
- **Each Model maps its shards.** `Gguf::open` (`crates/breccia-format/src/gguf.rs:79`) maps each file with memmap2 `Mmap::map` (`gguf.rs:98`). `map_readonly` (`crates/breccia-format/src/filemap.rs:9–13`) is the same call, used for the MTP files (`crates/breccia-format/src/mtp.rs:139–140`). Two Models therefore hold two mappings of the same shard files, and the kernel page cache holds one copy of each page. The ShardSet internals were not traced this round.
- **Default residency.** `main.rs:1664`: `vec![vec![0, 1, 2, 4, 8, 18, 20, 22, 23]; n_streams]`. The comment at :1660–1663 cites §8p (+2.5 tok/s, about 17 ms per step) and says the set does not fit once two streams share a card, so it is not used above two streams. The agents path applies the same list only for two streams on two devices (`main.rs:1990`). Nine blocks are resident per stream; 39 are streamed.
- **Shared host lane.** The comment at `crates/breccia-engine/src/multi.rs:25` says the host expert lane is the only shared object. The comment and doc 32 disagree on the pool's thread count (16 against 32; §8). `Multi::new` shares the pool at `multi.rs:155–156`. `token_all` (`multi.rs:178`) reaches `covered` (:196) and `finish_block` (:199); the mechanics were not traced in detail.
- **Decode lane.** `moe_topk` runs at `crates/breccia-engine/src/graph.rs:3137` under the pool lock; `lane_s` is updated at :3140 and `lane_covered_s` at :3142. The CPU kernel is `crates/breccia-kernels/src/cpu/expert.rs:463`.
- **Prefill.** `fill_stream_slot` is at `graph.rs:2516`, `prefill_block_merged` at :2566, and `STREAM_SLOTS = 8` at :271. `BRECCIA_TRACE_ROUTE` (`graph.rs:2432`) prints distinct counts only.
- **Upload reads from the mapping.** `upload_tensor` (`crates/breccia-engine/src/model.rs:1185`) reads `t.as_slice()` (:1188), a slice of the mapping, so uploaded pages stay mapped for the life of the Model.
- **Residency picks by bytes.** `crates/breccia-engine/src/arena.rs:202` documents first-fit decreasing per device, largest arena first, with a metric that is "deliberately bytes held, not block count"; `select` is at :213. Heat and sensitivity are not inputs.
- **No advice calls in base.** A grep of `crates/` at a6c53f6 finds no madvise, fadvise, WILLNEED, DONTNEED or MADV_ call. Mlock and MEMLOCK appear only in `crates/breccia-profile`, where the RLIMIT_MEMLOCK probe is at `system.rs:159–163` (profile reporting).
- **Off base.** `crates/breccia-vk/src/pagecache.rs` on branch `feat/r29-fadvise` has `drop_range` (:40) and `willneed_range` (:46); its note says fadvise does not drop pages a mmap still holds. Commit 0c2c8e6 (R29-1, "wire --page-hint-trim engine hooks", default off; not an ancestor of a6c53f6) adds `LoadOptions.page_hint_trim` (`model.rs:331`, default false at :345), the call at :758, and `page_hint_experts` at :1145 (§4). Commit d7ed5e0 ("anon mlock'd host expert arena") is also off base. Commit de45780's subject is a validator change (`validate(D5)`); the limit setting itself was not traced (§6, C3). The string "units need LimitMEMLOCK=infinity" is already in base at `crates/breccia-profile/src/plan.rs:168`.
- **Cgroup snapshot.** Doc 31 (`docs/research/31-host-memory-tier.md:111`): memory.max 32,212,254,720; file 23.5 GB; file_mapped 22.9 GB; anon 19.9 MB; workingset_refault_file 6,955,607. These are cumulative counters from an exited bench cgroup, with no token count.

## 2. Cross-stream sharing (Part 1)

**Bytes per stream per token.** Each token picks 10 experts per block, and an expert is 1/512 of its block's bytes. Averaged over the 39 streamed blocks (§9), one stream reads 10 × 34,589,900,800 / 512 = 675,584,000 B = **675.6 MB**. Two streams read 1,351.2 MB per token pair. Doc 31 `:24` gives 681.0 MB and 1,362.0 MB which matches multiplying 39 by the 48-block mean (§8, conflict 1). At 48 blocks the figure is 838.1 MB.

**Shared fraction under independent uniform routing.** Each stream picks a given expert with p = 10/512 = 0.01953 per block per token. The expected number of experts both streams pick per block is 512 × p² = 100/512 = 0.1953, which is 1.95% of one stream's 10 picks. Bytes saved per token pair = (100/512) × 67,558,400 B = **13,195,000 B = 13.2 MB**, or 0.98% of the pair total. Derived, with expert sizes averaged over the 39 streamed blocks.

**Saving as a function of overlap f.** f is shared picks divided by one stream's picks. Rows below the first are parameters, not measurements. Saved bytes per pair = f × 675.6 MB.

| f | Saved per pair | % of pair bytes | Basis |
|---|---|---|---|
| 1.95% | 13.2 MB | 0.98% | Derived, independent uniform routing |
| 10% | 67.6 MB | 5.0% | Parameter |
| 25% | 168.9 MB | 12.5% | Parameter |
| 50% | 337.8 MB | 25.0% | Parameter |
| 100% | 675.6 MB | 50.0% | Bound: identical routing |

**Where high overlap could come from.** The 100% row is reachable only if both streams route identically. The generate loop (`multi.rs:233–235`) sends the same token to every stream during prefill (`token_all(&vec![t; ns], pos)`), so prefill is identical across streams by construction. Decode diverges once sampled tokens differ. Whether the bench token path calls generate was not traced. The agents path gives agent 0 the parity reference prompt (`main.rs:1962`); the prompts for the other agents were not traced, so f for that path is not stated. No measured decode overlap was found in the docs read this round; the one skew measurement is a prefill point.

**Cost the saving does not count.** A shared reader that waits for both streams' routing before starting block b would change the overlap that `covered` (`multi.rs:196`) accounts for. The mechanics were not traced in detail.

**What the page cache already does.** Both Models map the same shards, so one page cache holds one copy of each page and disk reads are already shared. A shared reader would save lane DRAM passes, not disk I/O.

## 3. Hot-set policies under the 30 GiB cap (Part 2)

**Setup (derived).**

- Cap: C = 30 × 2^30 = 32,212,254,720 B (memory.max, doc 31 `:111`).
- Working set: S = 34,589,900,800 B for the 39 streamed blocks (32.21 GiB). Per block, the five smallest are 668,467,200 B each (blocks 3, 6, 7, 11 and 13, printed by the fit script). The six smallest sum to 4,010,803,200 B, six times 668,467,200, so the sixth is also 668,467,200 B (its index is not printed). The largest of the 48 blocks is 1,192,755,200 B (index not printed). Block 0 is 956,825,600 B.
- Overflow: S − C = 2,377,646,080 B (2.21 GiB). Doc 31 `:24` gives 2.47 GiB (§8, conflict 2).
- One cache. Both Models read the same files, so the working set counts once: S, not 2S. This holds while both streams keep the same residency list (the default).
- Headroom H is a parameter (0, 1 GiB, 2 GiB) for anon memory and PLE pages, which are file pages in the same cgroup. Anon was 19.9 MB at the doc 31 snapshot. PLE is 26.82 GiB (`model.rs:1–20`) and is read per token; doc 31 `:57` gates its read bytes. The resident share of PLE is not measured here, so H = 0 is optimistic.
- Fit fraction c = (C − H)/S: H = 0 gives 93.13%; H = 1 GiB gives 90.02%; H = 2 GiB gives 86.92%.

**Uniform routing (bound).** Under an independent reference model with equal probabilities, a cache holding a fraction c of the expert bytes hits about c of references, and the Che characteristic-time approximation agrees. It ignores the size spread across blocks, so count-based and byte-based fractions differ slightly. Per stream and token p = 10/512 = 0.01953. The characteristic time solves (1 − p)^T = 1 − c, so T = ln(m)/ln(1 − p), with m = 0.068738, ln m = −2.6775 and ln(1 − p) = −0.019725. That gives **T = 135.7 stream-tokens**. With two streams, the per-pair reference probability is q2 = 1 − (1 − p)² = 0.038681, so ln(1 − q2) = 2 ln(1 − p) = −0.039449 and **T_pair = 67.9 token pairs**. The threshold halves; the miss fraction does not change.

**Expected misses (bound).** Per stream-token there are 39 × 10 = 390 expert references, and 390 × 0.068738 = 26.8 misses. An expert is three slices (gate, up, down); their 39-block mean is 34,589,900,800 / 19,968 = 1,732,267 B, or 422.9 pages of 4 KiB. Misses are 26.8 × 422.9 = 11,338 pages = **46.4 MB per stream-token**, or 92.9 MB per token pair. The doc 31 refault counter (6,955,607, §1) is cumulative with no token count, so it cannot be turned into a per-token figure here.

**Measured skew (one point).** `docs/phase-4/02-untried-levers.md` `:472–473` records about 130 of 512 experts per block for a 128-token prefill chunk; doc 27 `:164` repeats that point and notes there is no histogram. The uniform comparison D(128) = 471.0 is doc 27 `:53`. The window working set is then 130/512 × 34,589,900,800 = 8,782,592,000 B = **8.18 GiB**, inside the cap with 21.8 GiB to spare. This is one prefill point. Decode is unmeasured, which is why the uniform miss is only a bound.

**Policies.** Steady-state effects are derived unless marked.

| Policy | Mechanism | Bytes in cap | Steady-state effect | Verdict |
|---|---|---|---|---|
| A. Kernel LRU/MGLRU (base) | None | S = 34.59 GB | Uniform: m = 6.87% (bound). Skewed: unmeasured | Baseline |
| B. Frequency-pinned mlock set (C3) | mlock the most-referenced experts | Pinned ≤ C − H | Uniform: hit rate equals the pinned share, the same as LRU, so no gain. Skewed: gain = f_m − pinned/S, with f_m unmeasured | Conditional on C1 |
| C. Anon mlocked LFU arena (d7ed5e0, off base) | Copy experts to anon memory and mlock them | Same budget; the page-cache copy must be dropped, or the bytes count twice | Uniform: no gain. Mlocked pages cannot be reclaimed, so a pinned set above C − H forces pressure | Reject unless C1 shows skew |
| D. Frequency-ordered sidecar | Reorder experts by heat in a sidecar file | Same bytes | Each expert is three contiguous slices (about 136, 136 and 155 pages at the 48-block mean). Reordering changes locality, not bytes; zero pages saved per miss | Reject: no byte effect |
| E. fadvise WILLNEED over the set (0c2c8e6) | §4 | 15.3 to 245.4 MB per Model at load | About zero in steady state | Ineffective |
| F. DONTNEED on uploaded blocks (0c2c8e6) | §4 | Zero while the mapping is alive | Load time only | No-op |
| G. One shared Mmap; drop resident pages after upload (C5) | Remove the PTEs once, then MADV_DONTNEED or munmap | Up to 8.32 GB of resident-block pages released at load instead of under pressure (derived; kernel reclaim not read) | About zero (derived): those pages are never touched again | Warm-up only |
| H. Residency +4 blocks (C2) | Move blocks 3, 6, 7 and 11 to the GPU | S − 2,673,868,800 = 31,916,032,000 B = 29.72 GiB | Fits at H = 0 with 0.28 GiB spare, so the overflow is removed and m = 0 under uniform routing | Only lever that fits under uniform routing; VRAM headroom unverified |

**Residency arithmetic.** Four smallest blocks: 4 × 668,467,200 = 2,673,868,800 B. S becomes 31,916,032,000 B = 29.72 GiB, which fits at H = 0 with 296,222,720 B (0.28 GiB) to spare. That spare must also hold anon memory and PLE pages, so the fit is not established. At H = 1 GiB six blocks are needed: the six smallest sum to 4,010,803,200 B, which is six times 668,467,200. S becomes 30,579,097,600 B = 28.48 GiB, leaving 0.52 GiB to spare under the 1 GiB headroom.

## 4. Page-hint risk check (Part 4)

**Kernel text, v6.18 (WebFetch of the raw source; the fetched text carried these lines).**

- `mm/readahead.c`, `force_page_cache_ra`: `max_pages = max_t(unsigned long, bdi->io_pages, ra->ra_pages);` then `nr_to_read = min_t(unsigned long, nr_to_read, max_pages);`. A WILLNEED reads at most `max(io_pages, ra_pages)` pages.
- `mm/fadvise.c`, `POSIX_FADV_WILLNEED`: `nrpages = end_index - start_index + 1;`, passed to `force_page_cache_readahead(mapping, file, start_index, nrpages)`.
- Not seen: the body of the `force_page_cache_readahead` wrapper that forwards `nrpages` to `force_page_cache_ra`. That link is UNVERIFIED.

**Bdi values on this host.** The btrfs bdi (sysfs name `btrfs-1`) has `read_ahead_kb` = 128, so `ra_pages` = 32 pages (sysfs read). `io_pages` is not exposed in sysfs (UNVERIFIED), so each WILLNEED reads at least 32 pages and its upper bound is unknown.

**What commit 0c2c8e6 asks for.** `page_hint_experts` (`model.rs:1145` on that commit) runs only when `page_hint_trim` is set. For each block b in 0..n_layers (48) and each role (gate, up, down), it takes the tensor's whole byte range: if the block is placed it calls `drop_range`, otherwise `willneed_range`. Per Model that is 39 × 3 = 117 WILLNEED calls and 9 × 3 = 27 DONTNEED calls. Both Models issue identical calls on the same files. The gate tensor alone is 13,670,809,600 B, which is 3.34 M pages.

**What the kernel reads.** Each WILLNEED reads `min(nrpages, max(io_pages, 32))` pages from the start of its tensor.

| io_pages | Per call | Per Model (117 calls) | Share of 34,589,900,800 B |
|---|---|---|---|
| 32 or fewer (lower bound) | 131,072 B (128 KiB) | 15,335,424 B = 15.3 MB | 0.044% |
| 512 (2 MiB; only if io_pages were 512, UNVERIFIED) | 2,097,152 B | 245,366,784 B = 245.4 MB | 0.71% |

The bytes read are the start of each tensor. In block 0, expert 0's gate slice is 473,600 B, so 128 KiB is about 28% of it. Expert 0 is one of 512 experts, which the router picks with probability 10/512 per token. The hint therefore does not target the experts the router picks.

**The claim's scale.** WILLNEED over 39 of 48 blocks names 34,589,900,800 B, which is 80.6% of the 42,912,972,800 expert bytes (81.25% by block count). That is 34.59 GB (32.21 GiB), above the 30 GiB cap (32.21 GB). The clamp means the hint reads only 0.04% to 0.71% of those bytes. The hint does not thrash, because it never sweeps the set. **Verdict: ineffective.**

**DONTNEED half.** The 27 DONTNEED calls per Model (9 placed blocks × 3 tensors) drop nothing while the pages stay mapped. The branch's own note (`pagecache.rs` on `feat/r29-fadvise`) says fadvise does not drop pages a mmap still holds. The fetched `mm/truncate.c` text describes `mapping_evict_folio` as refusing mapped folios (no line numbers; the exact path from fadvise DONTNEED is UNVERIFIED). Breccia uploads through `t.as_slice()` (`model.rs:1188`) and keeps the Mmap alive, so placed-block pages stay mapped and DONTNEED drops nothing (derived).

**What a working fix needs.** The PTEs must go first, in both Models: one shared Mmap, or munmap after upload (card C5). A targeted hot-set prefetch would use `MADV_POPULATE_READ` over the chosen ranges, which faults through the normal fault path rather than the readahead clamp (6.18 behaviour not read; UNVERIFIED). Derived: under uniform routing the steady-state gain of any prefetch is zero, because the miss fraction is fixed by c.

## 5. Literature (Part 3)

Cited, not re-audited. Sources marked "search snippet" were not fetched in full. Numbers are as stated in the named doc, not verified here. llama.cpp and the other engines in the README's already-covered list are not re-audited.

| Source | What it does | Status | Maps to Breccia's cache policy |
|---|---|---|---|
| MoE-Lightning, arXiv 2411.11217 | CGOPipe CPU-GPU-I/O pipelining over paged weights | Doc 21 `:72` (10.3× peak throughput over offloading systems, Mixtral 8x7B on one 16 GB T4; abstract as cited there); doc 27 `:119–121` (Table 4 as summarised gives about 4.7× and 8.6×, against the stated 3.5× and 6.7×) | Doc 27 `:98` judges paged weights redundant with the existing slot ring (`STREAM_SLOTS = 8`, `graph.rs:271`). No new cache policy |
| Fiddler, arXiv 2402.07033 | Per-expert choice between moving weights and computing on the CPU | Doc 27 `:101` (cited there; source not re-read here); doc 24 `:139` (move 68.1–72.8 µs against 75.0 µs on the CPU) | The host lane already computes on the CPU. Changes miss cost, not miss rate |
| HOBBIT, arXiv 2411.01433 | Low-precision copies stand in for missed experts; three-tier hierarchy | Doc 24 `:48`, `:99` (S1 −25.2%, S2 −20.8% per token); `:93` (bytes per miss); `:119` (top-1 about 96%); `:140` (three tiers; NVMe 296 µs) | A miss can be served from a 2-bit shadow (doc 24 `:7`). Changes bytes per miss, not the miss rate |
| ProMoE, arXiv 2410.22134 | Expert prefetching | Doc 24 `:118` (84.7% against 58.3%, as cited) | Prefetch needs a predictor and an overlapped copy path. Not a cache policy |
| AdapMoE, arXiv 2408.10284 | Per-layer expert-count allocation | Doc 24 `:137` (25% fewer experts, 1.35×; abstract figures, not verified here; needs per-layer sensitivity sweeps); id at doc 24 `:201` | Per-block residency allocation. `select()` (`arena.rs:213`) is bytes-only, with no heat or sensitivity input |
| MoE-Infinity, arXiv 2401.14361 | Not re-read this round | Already covered: doc 24 `:143`, doc 03 `:167`; id at doc 24 `:201` | A frequency pin (C3) would need a routing trace like C1. Not re-audited |
| Pre-gated MoE, arXiv 2308.12066 | Not fetched | Doc 23 `:83` and `:120` mention it; id present there | Cross-block prediction needs a predictor (doc 30 `:185`). Whether it needs a trained gate change is not verified here |
| EdgeMoE, arXiv 2308.14352 | Non-expert weights resident, experts fetched on use, per-expert bit-width (as the search snippet describes) | Search snippet only; id from the snippet, UNVERIFIED; no numbers | Same bit-width mechanism as HOBBIT. No new policy |
| SwapMoE, arXiv 2308.15030 | Pool of virtual experts within a tunable memory budget (as the search snippet describes) | Search snippet only; id from the snippet, UNVERIFIED; numbers not verified | A fixed resident budget per block, which C2 (residency +4) already is |
| llama.cpp `--override-tensor` | Per-tensor placement override | Contrast only; not re-audited; no source read | Breccia maps with memmap2 `Mmap::map` (`gguf.rs:98`) and uploads from the mapping (`model.rs:1188`) |

Vulkan host import: doc 09 `:92` records a zero-copy saving of 15 ms per token. Host import was not traced, and §3 counts file-backed pages only; host-visible copies of the same bytes would need their own accounting before they enter the §3 arithmetic.

## 6. Ranked flag-OFF cards

Each card carries its own full gate line. C1 ranks first because its measurement decides whether C2 and C3 are worth building and supplies the overlap f for C6.

**C1. Decode routing trace (measurement; ranked first).** Flag OFF `BRECCIA_TRACE_EXPERTS` (proposed name; base has only the prefill distinct-count trace `BRECCIA_TRACE_ROUTE`, `graph.rs:2432`). It writes, per token and block, the 10 (expert, weight) pairs in the format proposed in doc 24 `:103`. From them it gives a per-block reference histogram D(W) (distinct experts per block over W decode tokens, W = 1 to 512) and the two-stream overlap f. Trace overhead counts in tok/s.

Gate: refaults (primary): expert `workingset_refault_file` and `pgmajfault` deltas over 200 tokens after warm-up; pass when expert refaults do not rise (doc 31 `:57`; its PLE clause is not gated here, since no card splits PLE reads). tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents`, flag ON and OFF interleaved. PCI ids: 0000:03:00.0 (RX 6950 XT, s1) and 0000:0e:00.0 (MI50, s0), both in every run (doc 05 `:13`, `:79`, `:93`). Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output (O32 as defined in doc 34 `:234`; "exact, against O32" at `:250`).

Gain: UNVERIFIED - requires the GPU qualification card

**C2. Residency +4 blocks (fits only at H = 0).** Flag OFF: a residency-list override with the default unchanged, adding blocks 3, 6, 7 and 11 to the list at `main.rs:1664` (proposed to be wired as a flag; flag name not verified). Per stream that moves 2,673,868,800 B (2.67 GB) onto the GPU, and the streamed set falls to 29.72 GiB, which fits the cap at H = 0 with 0.28 GiB spare. H = 1 GiB needs six blocks (4.01 GB per stream) and leaves 0.52 GiB. Anon and PLE pages are not in those spare figures. VRAM headroom was not checked in this round, so the card is blocked until a VRAM budget is measured.

Gate: refaults (primary): expert `workingset_refault_file` and `pgmajfault` deltas over 200 tokens after warm-up; pass when expert refaults do not rise (doc 31 `:57`; its PLE clause is not gated here, since no card splits PLE reads). tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents`, flag ON and OFF interleaved. PCI ids: 0000:03:00.0 (RX 6950 XT, s1) and 0000:0e:00.0 (MI50, s0), both in every run (doc 05 `:13`, `:79`, `:93`). Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output (O32 as defined in doc 34 `:234`; "exact, against O32" at `:250`).

Gain: UNVERIFIED - requires the GPU qualification card

**C3. Frequency-pinned mlock (conditional on C1).** Flag OFF `--pin-experts <list>` (proposed). Pinned bytes must stay at or below C − H. It runs only if C1 shows the top experts carry a share f_m well above pinned bytes / S; under uniform routing it gains nothing (§3, policy B). It needs the memlock limit raised (`LimitMEMLOCK` in the unit file; the RLIMIT_MEMLOCK probe is at `system.rs:159–163` in the `breccia-profile` crate). Commit de45780 is off base; its subject is a validator change (`validate(D5)`), and the limit setting itself was not traced. The string "units need LimitMEMLOCK=infinity" is already in base at `crates/breccia-profile/src/plan.rs:168`.

Gate: refaults (primary): expert `workingset_refault_file` and `pgmajfault` deltas over 200 tokens after warm-up; pass when expert refaults do not rise (doc 31 `:57`; its PLE clause is not gated here, since no card splits PLE reads). tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents`, flag ON and OFF interleaved. PCI ids: 0000:03:00.0 (RX 6950 XT, s1) and 0000:0e:00.0 (MI50, s0), both in every run (doc 05 `:13`, `:79`, `:93`). Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output (O32 as defined in doc 34 `:234`; "exact, against O32" at `:250`).

Gain: UNVERIFIED - requires the GPU qualification card

**C4. Warm-up populate in place of the clamped WILLNEED.** Flag OFF `--hint-populate` (proposed). `MADV_POPULATE_READ` over the C1-hot ranges at load, sized at most C − H. Its 6.18 behaviour was not read. Steady-state gain is zero under uniform routing (derived); it only shortens the first tokens.

Gate: refaults (primary): expert `workingset_refault_file` and `pgmajfault` deltas over tokens 1 to 200 after load (the warm-up is the change under test); pass when expert refaults do not rise (doc 31 `:57`; its PLE clause is not gated here, since no card splits PLE reads). tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents`, flag ON and OFF interleaved. PCI ids: 0000:03:00.0 (RX 6950 XT, s1) and 0000:0e:00.0 (MI50, s0), both in every run (doc 05 `:13`, `:79`, `:93`). Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output (O32 as defined in doc 34 `:234`; "exact, against O32" at `:250`).

Gain: UNVERIFIED - requires the GPU qualification card

**C5. One shared Mmap across both stream Models.** Flag OFF `--shared-shard-map` (proposed). One mapping of the shards serves both Models. After upload, `MADV_DONTNEED` or munmap on the uploaded resident ranges would release up to 8,323,072,000 B of resident-block pages at once rather than waiting for pressure (derived; kernel reclaim was not read). The cap arithmetic in §3 is unchanged by this, since normal reclaim already unmaps cold file pages. This is the change the DONTNEED half in §4 needs. Steady-state gain derived as about zero, because those pages are not read again.

Gate: refaults (primary): expert `workingset_refault_file` and `pgmajfault` deltas over 200 tokens after warm-up; pass when expert refaults do not rise (doc 31 `:57`; its PLE clause is not gated here, since no card splits PLE reads). tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents`, flag ON and OFF interleaved. PCI ids: 0000:03:00.0 (RX 6950 XT, s1) and 0000:0e:00.0 (MI50, s0), both in every run (doc 05 `:13`, `:79`, `:93`). Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output (O32 as defined in doc 34 `:234`; "exact, against O32" at `:250`).

Gain: UNVERIFIED - requires the GPU qualification card

**C6. Shared lane reader across the two streams (ranked last).** Flag OFF `--shared-lane-reader` (proposed): one reader per block and one staging buffer for both streams. Saves f × 675.6 MB per token pair: 13.2 MB (0.98%) at the independent-routing f, 25% of pair bytes only at f = 50%, and 50% only at identical routing. It may change the overlap that `covered` (`multi.rs:196`) accounts for; the mechanics were not traced in detail.

Gate: refaults (primary): expert `workingset_refault_file` and `pgmajfault` deltas over 200 tokens after warm-up; pass when expert refaults do not rise (doc 31 `:57`; its PLE clause is not gated here, since no card splits PLE reads). tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents`, flag ON and OFF interleaved. PCI ids: 0000:03:00.0 (RX 6950 XT, s1) and 0000:0e:00.0 (MI50, s0), both in every run (doc 05 `:13`, `:79`, `:93`). Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output (O32 as defined in doc 34 `:234`; "exact, against O32" at `:250`).

Gain: UNVERIFIED - requires the GPU qualification card

## 7. Unverified

- `io_pages` for the btrfs bdi (not in sysfs). The clamp's upper bound is therefore unknown; the 512-page row is a hypothesis.
- The `force_page_cache_readahead` wrapper body (not in the fetched text).
- Line numbers for `mapping_evict_folio`, and the exact call path from fadvise DONTNEED on 6.18.
- `MADV_POPULATE_READ` behaviour on 6.18 (not read).
- Decode routing skew. The only measurement is prefill (`docs/phase-4/02-untried-levers.md` `:472–473`; doc 27 `:164`).
- Per-token denominators for doc 31's refault counters.
- Headroom H (anon and PLE share the cgroup) and PLE per-token read bytes.
- VRAM headroom for C2, and the 0.28 GiB H = 0 spare.
- EdgeMoE and SwapMoE: search snippets only.
- The memlock setting for C3: de45780 is a validator change, not the setting; its source was not traced.
- Whether the bench token path calls the generate loop, and the prompts of the bench agents (§2).
- The 4 KiB page assumption: THP does not reach the mapping (doc 31 `:10`).
- The indices of the largest block (1,192,755,200 B) and of the sixth-smallest block (668,467,200 B, from the six-block sum).

## 8. Conflicts

1. **Doc 31 `:24` against the GGUF sums.** Doc 31 gives 39 blocks = 34.87 GB (32.47 GiB) and 681.0 MB per token per stream. The per-block GGUF sums give 34.59 GB (32.21 GiB) and 675.6 MB. Doc 31 multiplies 39 by the 48-block mean, 42,912,972,800 / 48 = 894,020,266.7 B, giving 39 × 894,020,266.7 = 34,866,790,400 B. The nine resident blocks are larger than average: 8.32 GB against 9 × the mean = 8.05 GB. Not corrected here; doc 31 needs the fix.
2. **Overflow.** Doc 31 `:24` says 2.47 GiB over the cap. This round gives 2.21 GiB (§3).
3. **Commit 0c2c8e6 is not in base.** Read with `git show`. The clamp in §4 is a kernel property and applies whatever the flag is set to.
4. **Uniform routing against the one skew point.** Doc 27 `:53` gives D(128) = 471.0 under uniform routing; against the measured ~130, the ratio is 3.6 (doc 27 `:131`). The source text (`docs/phase-4/02-untried-levers.md` `:472–473`) says 2.7×. Unresolved. The uniform miss in §3 is therefore a bound. Doc 29 `:113` (C14) also records this conflict.
5. **Pool thread count.** The comment at `multi.rs:25` (and `graph.rs:439–441`) says 16 threads. Doc 32 (`docs/research/32-hetero-gpu-balance-and-determinism.md` `:11`) reports the lane pool at `default_threads()` clamped to 32, which is 32 on the host doc 31 `:33` describes. Not resolved here.
6. **GGUF down tensors.** 18 down tensors are IQ4_NL, with exact bytes 8,493,465,600 B. The other 30 use ggml type id 42, for which only offset deltas are available: 7,077,888,000 B. Together they give 15,571,353,600 B, matching the delta total for the down role. The delta method is used throughout.

## 9. Arithmetic

**Tensor sizes.** Offset deltas, with each tensor's end at the next tensor's offset (last tensor at the data-region end). Read-only walk with a `python3 -I` script over both shards.

- Shard 1: 47,039,860,096 B, data region from 11,024,256. Shard 2: 28,800,138,432 B, data region from 192.
- Data-region totals: 47,028,835,840 + 28,800,138,240 = 75,828,974,080 B, equal to the sum of the deltas.
- Expert tensors (48 blocks × gate, up, down): gate 13,670,809,600; up 13,670,809,600; down 15,571,353,600. Total 42,912,972,800 B = 39.97 GiB (doc 31 `:5`, `model.rs:1–20`).
- Per expert, 48-block mean: 42,912,972,800 / (48 × 512) = 1,746,133 B.
- Per block, 48-block mean: 894,020,266.7 B. Block 0 = 242,483,200 + 242,483,200 + 471,859,200 = 956,825,600 B. Block 3 = 216,268,800 + 216,268,800 + 235,929,600 = 668,467,200 B.
- Resident nine (0, 1, 2, 4, 8, 18, 20, 22, 23): 8,323,072,000 B.
- Streamed 39: 42,912,972,800 − 8,323,072,000 = 34,589,900,800 B = 32.21 GiB.
- Per expert across three roles, 39-block mean: 34,589,900,800 / 19,968 = 1,732,267 B.
- Groups from the same walk: expert 42,912,972,800 B; PLE 28,800,138,240 B = 26.82 GiB (matches `model.rs:1–20`); token embedding 273,152,000 B; other non-expert tensors 3,842,711,040 B. The four sum to 75,828,974,080 B, the data-region total.

**Lane bytes.** 10 × 34,589,900,800 / 512 = 675,584,000 B (675.6 MB) per stream-token. Pair: 1,351,168,000 B. At 48 blocks: 10 × 42,912,972,800 / 512 = 838.1 MB.

**Cross-stream.** (100/512) × 67,558,400 = 13,195,000 B. Divided by 1,351,168,000 = 0.98%. Rows of §2: f × 675,584,000 B per pair.

**Cap and fit.** C = 32,212,254,720 B. S − C = 2,377,646,080 B = 2.21 GiB. c = C / S = 0.931262; m = 0.068738. H = 1 GiB: (C − 2^30) / S = 0.900220. H = 2 GiB: 0.869178.

**Thresholds.** ln(0.068738) = −2.6775. ln(1 − 10/512) = −0.019725. T = 135.7 stream-tokens. q2 = 1 − (1 − 10/512)² = 0.038681; ln(1 − q2) = −0.039449; T_pair = 67.9 token pairs.

**Misses (bound).** 390 × 0.068738 = 26.8 misses per stream-token. Expert = 1,732,267 B = 422.9 pages. 26.8 × 422.9 = 11,338 pages; 11,338 × 4,096 = 46,440,448 B = 46.4 MB per stream-token. Pair: 92.9 MB.

**Measured skew (one point).** 130 × 67,558,400 = 8,782,592,000 B = 8.18 GiB. Share 130/512 = 25.4%. Headroom 30 − 8.18 = 21.8 GiB.

**Fit by residency.** Four smallest blocks: 2,673,868,800 B removed; S becomes 31,916,032,000 B = 29.72 GiB; spare at H = 0: 296,222,720 B = 0.28 GiB. H = 1 GiB: six smallest blocks, 4,010,803,200 B removed; S becomes 30,579,097,600 B = 28.48 GiB; spare 559,415,296 B = 0.52 GiB.

**WILLNEED.** Per call min(nrpages, 32) = 32 pages = 131,072 B. 117 calls = 15,335,424 B = 15.3 MB per Model. Ceiling at 512 pages (unverified io_pages): 117 × 2,097,152 = 245,366,784 B = 245.4 MB. Shares of 34,589,900,800 B: 0.044% and 0.71%.

**Expert share named.** 34,589,900,800 / 42,912,972,800 = 80.6% of expert bytes; 39/48 = 81.25% by block count.

## 10. Hygiene

Public-repo rules for this file: no absolute paths, no IP addresses, no host or user names, no personal names, no secrets, and none of the words the brief bars. PCI bus ids are the only hardware addresses; they are already public in doc 05. The grep result is recorded in the final report, not here.
