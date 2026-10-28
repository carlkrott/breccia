# 36 — Niche Vulkan engines: no weight paging fits the cap, Fastllm's O_DIRECT miss path is the one transferable idea, and `--gpu-experts auto` with two streams silently runs zero resident blocks

Round 36. Docs only: no code changed, no build, no GPU or model process run. Base commit 612fb67. Every gain below is UNVERIFIED until the GPU qualification card runs. Arithmetic is labelled derived where it is one, and a bound where it assumes uniform routing. GB is 10^9 bytes; GiB is 2^30 bytes.

> **Status on `port/expert-import` HEAD (c66622f), checked after writing:** the `bench token --streams N` half of the trap is already fixed by b1037e1 (`multi_arena_specs` routes every group through `ArenaSpec::parse`, `auto` becomes `Auto`, bad tokens bail), which is not in base 612fb67. `bench agents` still has the trap at HEAD: its own `filter_map` parse turns `auto` into `[[]]` and both streams run zero resident blocks. Card A therefore reduces to `bench agents` reusing `multi_arena_specs`. Line numbers below are at the base commit.

Citations are repo-relative for Breccia. "doc NN" means the `docs/research` file with that number. External engines were read through the GitHub API at the pinned SHAs below, not cloned. Each external citation names its SHA. Spot-checks at those SHAs are listed per engine; anything not spot-checked is marked agent-reported.

## 0. Verdict

1. **No engine here pages weights in a way that fits the 30 GiB cap without a measurement Breccia still lacks.** MNN's Vulkan path keeps weights resident and declines int2 and int3 (`VulkanIndexedLinear.cpp:74`), so it cannot run this format. Its mmap path is plain MAP_SHARED OS paging (`MNNFileUtils.cpp:289–293`). Breccia already maps its shards the same way (`crates/breccia-format/src/gguf.rs:98`).
2. **Fastllm is the only engine with a disk-expert policy, and its O_DIRECT bounded cache is the one idea that transfers.** Breccia can take a miss-only version (card B). Card B conflicts with doc 03 §A.3 and touches doc 35 option C, so it needs the doc 03 owner's sign-off and a hit-fraction baseline first. Under uniform routing the misses are 46.44 MB per stream-token (a bound, doc 35 §9). At the two-stream aggregate in `graph.rs:4148–4155`, that is 16.0% to 16.5% of the 5.86 GB/s device ceiling (derived).
3. **Trap in both bench commands: `--gpu-experts auto` silently gives every stream zero resident blocks.** `bench token --streams 2` and `bench agents` share the parse, and the nine-block default needs an empty parsed list. Card A. Against the 9-block default, the all-streamed set is 42.91 GB: +162.56 MB per stream-token (+24.06%) and 9.97 GiB over the cap (derived). Until card A lands, pass no `--gpu-experts` flag for two streams. A two-stream run with `auto` on this code is a zero-block run. Past runs were not checked against the code they used.
4. **The thread default is stated three different ways.** `bench cpu-experts` defaults to 8 threads (`bench.rs:114`, `:238–239`). The engine defaults to 32 (`graph.rs:4155`). Two comments say the shared pool is 16 threads (`multi.rs:25`, `graph.rs:441`). Card C makes the bench default follow the engine's rule (a copy, because the kernels crate cannot call the engine). The engine's own gate comment reports 14.7 tok/s at 8 threads against 20.2–20.8 at 32 (`graph.rs:4148–4155`, cited, not re-measured here).
5. **WebGPU and wgpu add no kernel card that Breccia lacks.** The persistent (expert, tile) walk in mlc-llm is already doc 11 D2. Ratchet's quantized GEMV falls through to `todo!()`. Dawn's limits are WebGPU limits.
6. **Fastllm's visits-below-two admission gate and its touch weight are not new as admission rules.** Second-hit admission is in doc 15 (`:84`) and doc 17 (`:103`). The touch weight is a scoring detail for doc 35 C3 and is not carded here.

Ranked flag-OFF cards: A, C, B (§6). None has a measured gain.

## 1. As built (base 612fb67)

- **Multi-stream expert list.** `bench_token_multi` (`crates/breccia-cli/src/main.rs:1636–1676`) splits `--gpu-experts` on `;`, then on `,`, and drops every token that does not parse as an integer (`filter_map`). The 9-block default applies only when the parsed group list is empty and `n_streams == 2 && devs.len() >= 2` (`main.rs:1659–1664`). `bench_agents` uses the same parse (`main.rs:1974`). Its default is keyed the same way, on `g.is_empty()` (`main.rs:1988`), and it expands the groups with `get(i).or_else(last)` (`main.rs:1995`). It passes `ArenaSpec::List(groups[s].clone())` (`main.rs:2011`). With `auto`, `g` is `[[]]`, so both streams get an empty list and nothing panics.
- **Single-stream path.** `main.rs:2446–2455` maps `auto` to `ArenaSpec::Auto` with `display_reserve` and `headroom` from the CLI. The multi-stream path builds `ArenaSpec::List` directly at `main.rs:1695` and `:2011`, so it never reaches `Auto`.
- **An empty list is not an error.** `Model::load` maps `ArenaSpec::List` (`crates/breccia-engine/src/model.rs:651–670`) by pushing each in-range block. Only an out-of-range block bails (`:654–655`). An empty list pushes nothing and returns Ok. `ArenaSpec::parse` (`crates/breccia-engine/src/arena.rs:46–67`) returns `Auto` for `auto` and `Off` for an empty string, but the multi-stream path never calls it.
- **Host expert lane.** `ExpertPool::new` (`crates/breccia-kernels/src/cpu/expert.rs:388–401`) spawns named workers (`breccia-expert-{id}`) with no affinity. A grep of `crates` for `affinity`, `sched_set`, `core_affinity` and `cpuset` returns nothing.
- **Engine thread default.** `graph.rs:990` uses `default_threads()` when `cfg.threads == 0`. That function (`graph.rs:4155–4165`) counts `processor` lines in the kernel's cpuinfo file, clamps to 1–32, and falls back to 8.
- **Bench thread default.** `crates/breccia-kernels/src/cpu/bench.rs:114–115` returns `(physical_cores() / 2).clamp(1, 16)`. `run_cpu_expert_bench` uses it when `opts.threads == 0` (`bench.rs:238–239`). The CLI flag defaults to 0 (`crates/breccia-cli/src/main.rs:1512`), and its help reads "physical cores/2, capped at 16" (`main.rs:277`).
- **Page touch for PLE rows.** `prefetch_rows` (`graph.rs:4063–4080`) spawns scoped threads per call, with `rows.len().min(8)` at `:4069`. Its doc comment says "up to four threads" (`:4063–4064`). Callers: `graph.rs:2300` (timed into `pf_ple_prefetch_s` at `:2301`), `:3528` (per-token PLE rows; the comment at `:4063–4064` cites 16 faults), and `:3627` (timed at `:3628`).
- **Grouped prefill expert GEMM.** `graph.rs:2676–2710` groups the chunk's users per distinct expert and runs the three GEMMs per group of up to eight. The grouped host lane is at `graph.rs:2381`. `STREAM_SLOTS = 8` is at `graph.rs:271`.
- **Shaders.** `crates/breccia-kernels/shaders/*.comp` already dequantise in-shader (for example `gemv_iq2_s_moe.comp`, `gemv_iq3_s_moe.comp`, `gemm_iq3_s.comp`, `dequant_iq3_s.comp`).

## 2. MNN-LLM (Vulkan)

Pinned: `alibaba/MNN@ce2f41ec8cd3` for the LLM and loader paths. Doc 26 reads shader files at `35a20ca0995b`. Spot-checked at `ce2f41ec8cd3`: `VulkanIndexedLinear.cpp:74`, `llmconfig.hpp:409–415`, `MNNFileUtils.cpp:289–293`.

| Technique | MNN source | Breccia status | Verdict |
|---|---|---|---|
| Vulkan linear keeps all weights resident; int2 and int3 declined | `VulkanIndexedLinear.cpp:74` | Breccia's IQ3_XXS path is its own shaders (`crates/breccia-kernels/shaders/`) | Not transferable: MNN's Vulkan path cannot take this format. |
| OS-paged weights via `use_mmap` and `use_cached_mmap` | `llmconfig.hpp:409–415` | Breccia maps shards (`gguf.rs:98`); the cap analysis is doc 35 §3 | No card. `use_cached_mmap` is a cache option. Its identity issue applies only if Breccia adds a disk cache, which it does not have. |
| MAP_SHARED file mapping | `MNNFileUtils.cpp:289–293` | Same mapping (`gguf.rs:98`); 4 KiB granularity (doc 31 `:10`) | Already in base. |
| Chunked staging at load | agent-reported | Load path only | No steady-state effect. Not carded. |
| Token bucketing per expert | agent-reported | Grouped prefill already exists (`graph.rs:2676–2710`) | Covered. Not carded. |
| `gemv_dequant_int3` (`subgroupAdd`, `constant_id`), coop-matrix conv1x1, `EagerBufferAllocator`, fused projections | doc 26 `:84`, `:127`, `:165`, `:166` | Doc 26 and doc 22 | Covered. |
| Vulkan OOM handling (`VulkanRuntime.cpp:20–23`) | agent-reported | Not checked | Dropped (§7). |

MNN verdict: no weight-paging scheme that fits 30 GiB. Its paging is OS paging of MAP_SHARED files, which doc 35 §3 already analyses for Breccia's mapping.

## 3. Fastllm (hybrid CPU and GPU MoE)

Pinned: `ztxz16/fastllm@8b0431b9333f`. Spot-checked at that SHA: `diskdevice.cpp:160–169`, `:979–984`, `:2472–2477`, `:2528`, `:2570–2571`, `:4165–4167`; `tools/fastllm_pytools/cli.py:35`; `hybrid_auto.py:1019–1022`; `numasdevice.cpp:77–79`; `numas.cpp:439–441`. No Vulkan backend was found in the CMake file or the tree. The repo code search returned 403, so that negative is not exhaustive.

| Technique | Fastllm source | Breccia status | Verdict |
|---|---|---|---|
| Bounded expert cache read with O_DIRECT, so the cache is not "another unbounded copy in the OS page cache" | `diskdevice.cpp:160–169` | Streamed experts read through the mapping (doc 35 §1) | Card B, miss-only variant. |
| fadvise WILLNEED skipped under direct I/O | `diskdevice.cpp:979–984` | Doc 35 §4 found the WILLNEED clamp | Cross-reference only. Breccia has no O_DIRECT path to skip it on. |
| Touch weight `clamp(ceil(8*routes/tokens), 1, 8)` | `:2472–2477` | Residency picks by bytes (`crates/breccia-engine/src/arena.rs:202–205`) | Scoring detail for doc 35 C3. Conditional. Not carded. |
| Visits-below-two gate | `:2528`, `:2570–2571` | Second-hit admission: doc 15 `:84`; doc 17 `:103` (store threshold 2) | Covered. |
| Prefetch depth 8 when parts run in parallel | `:4165–4167` | Doc 35 C4; `STREAM_SLOTS` (`graph.rs:271`) | Covered. |
| One shared expert pool ("a second spinning pool would compete") | `numasdevice.cpp:77–79` | One shared pool (`multi.rs:22–27`) | Covered. |
| L3-spread placement, keyed on L3 not NUMA | `numas.cpp:439–441` | Doc 30 §6.2 and doc 08 §6.3 (negative on this host, 12.8–13.0 GB/s) | Not carded: already tested negative. |
| Static device-layer split `--moe_device_layers` | `hybrid_auto.py:1019–1022` | Residency is hand-set too (`main.rs:1664`) | Not new. Doc 35 C2 is the residency lever. |
| Fixed thread count tied to a host (`FASTLLM_NUMA_THREADS=27`) | `tools/fastllm_pytools/cli.py:35` | Engine default is 32 (`graph.rs:4155`) | Not transferable as a number. |

Doc 31 row 103 says fastllm has "No memory-tier mechanism found", and line 139 repeats it. Fastllm has both a disk-expert policy (`diskdevice.cpp:160–169`) and a visits gate (`:2528`). Doc 31's row is wrong in part (§8).

## 4. WebGPU, Dawn, ratchet, mlc-llm, crabml

Pinned: `huggingface/ratchet@21ac3a5ff2cb`, `google/dawn@106704a9ac3f`, `mlc-ai/mlc-llm@8978ea9aec62`. Spot-checked: ratchet `ops/matmul/mod.rs` match arms and `workgroup_gemv.rs:162`, `:255`; Dawn `Limits.cpp:62`, `:136`; mlc `python/mlc_llm/op/moe_matmul.py:449–461, 511`.

- **ratchet.** Quantized GEMV in `ops/matmul/mod.rs` falls through to `_ => todo!()`, so ratchet has no quantized expert GEMV on WebGPU. Nothing to take. Its Q8_0F vec4 load pattern is agent-reported and falls in the family doc 26 covers. Not carded.
- **Dawn.** `Limits.cpp:62` gives a 128 MiB binding limit in the compatibility profile. `:136` gives a 256 B storage-offset alignment. Both are WebGPU limits. Breccia targets Vulkan's device limits, not this profile's; its buffer sizes were not checked against them (§7). Whether its binding offsets meet the device's `minStorageBufferOffsetAlignment` was not checked (§7).
- **mlc-llm.** `moe_matmul.py:449–461, 511` walks a flattened (expert, tile) space with persistent blocks. Doc 11 D2 already names this (`moe_matmul.py:386–480`), and doc 01 covers the Vulkan side. Not re-carded here. It belongs to the prefill owner (doc 27).
- **WebLLM.** Not read separately this round. It runs MLC-compiled kernels, so the mlc citations are the only kernel source checked (§7).
- **crabml.** Its mmap, WillNeed and mlock use is agent-reported only. It maps onto doc 35 C3 to C5 and docs 31, 12 and 26. Not spot-checked.

WebGPU verdict: the subgroup-free techniques that survive (in-shader dequant and tiled GEMV) are already in Breccia's shaders. No card.

## 5. Not carded, and why

- MNN bucketing, chunked load staging, `use_cached_mmap`: covered or load-only (§2).
- Fastllm visits gate, touch weight, prefetch depth, shared pool, L3 spread, static device split: covered, conditional, or tested negative (§3).
- mlc persistent (expert, tile) walk: doc 11 D2, prefill owner (§4).
- Dawn limits and ratchet GEMV: WebGPU-only (§4).
- `prefetch_rows` thread spawns per token (`graph.rs:3528`): the cost is not measured. The existing `pf_ple_prefetch_s` counter (`graph.rs:2301`, `:3628`) already times it, so read that counter in the GPU qualification run before any card. No card now.
- Pool thread-count comments at `multi.rs:25` and `graph.rs:441` say 16 threads, against the 32 the code gives (§8.3). A comment fix only; not carded.

## 6. Ranked flag-OFF cards

Card A ranks first because a two-stream `auto` run on this code measures the zero-block path, so such a number does not describe the configuration it names. Card C ranks second because it changes what `bench cpu-experts` measures. Card B ranks last because it changes the page-cache policy and conflicts with an existing rule.

Warm-up, used by every gate in this section: the buffered path runs until the resident page count of the streamed expert ranges, taken with a `mincore` sweep, has not grown for 20 consecutive tokens. Gates count 200 tokens from that token. The 20-token plateau is a choice, not measured.

**A. Multi-stream `--gpu-experts auto` reaches the Auto rule (flag OFF `BRECCIA_MULTI_ARENA_AUTO`, proposed name).**

Trap, in code. `main.rs:1645–1657` turns `auto` into one group with no integers, so `groups` is `[[]]`. The `groups.is_empty()` test (`:1659`) is then false, so the 9-block default is skipped, and each stream gets `ArenaSpec::List(vec![])` (`main.rs:1695`, `:2011`). `bench_agents` does the same: its `g.is_empty()` test (`main.rs:1988`) is false, and `get(i).or_else(last)` (`main.rs:1995`) hands `[]` to both streams. `model.rs:651–670` accepts the empty list without error.

Change, flag ON only: the multi-stream path parses the spec with `ArenaSpec::parse` and passes `Auto` with the same `display_reserve` and `headroom` the single-stream path reads (`main.rs:2446–2455`). Flag OFF: unchanged. Not in this card: a non-integer entry other than `auto` is still dropped silently. That needs its own bail on parse failure.

Arithmetic (derived, §9): the flag-OFF `auto` run streams all 48 blocks, 42,912,972,800 B, against the default 34,589,900,800 B. Lane bytes per stream-token are 838.14 MB against 675.58 MB, +162.56 MB (+24.06%). Overflow of the 30 GiB cap is 10,700,718,080 B (9.97 GiB) against 2,377,646,080 B (2.21 GiB).

Baseline: the flag-OFF run with `--gpu-experts` omitted (the 9-block default). Flag-OFF `auto` is the zero-block path, so it is not a valid baseline.

Gate: refaults (primary): expert `workingset_refault_file` and `pgmajfault` deltas over 200 tokens after warm-up; pass when expert refaults do not rise (doc 31 `:57`). tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents`, flag ON and OFF interleaved against the 9-block baseline. Both GPUs in every run. Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output (the 9-block baseline) is expected only if the CPU lane and the GPU expert shaders give identical bits for the same expert. Flag ON changes which blocks sit on the GPU, so some experts move between the host lane and a GPU shader, and the output can change. The repo does not establish that identity: doc 32 card 5 checks the CPU lane against its own sequential reference, not against the GPU. Doc 32 card 6 is the house precedent for a host-to-GPU expert move, gated on a 64-token output bit-identical to baseline. If the flag-ON output differs, record the first differing token and fail the card; do not waive it.

Gain: UNVERIFIED - requires the GPU qualification card

**C. Bench thread default follows the engine (flag OFF `BRECCIA_BENCH_ENGINE_THREADS`, proposed name).**

Change, flag ON only: `default_threads()` in `crates/breccia-kernels/src/cpu/bench.rs:114–115` returns the logical CPU count clamped to 1–32, and falls back to 8 when the processor list cannot be read. That is the engine's rule (`graph.rs:4155–4165`), copied rather than called: `crates/breccia-engine/Cargo.toml:10` makes the engine depend on kernels, so kernels cannot depend on the engine. A shared helper in kernels would remove the copy, but it moves engine code, which this card does not do. `run_cpu_expert_bench` (`bench.rs:238–239`) keeps using `default_threads()` when `threads == 0`. Flag OFF: unchanged, 8 on this host. The help text at `main.rs:277` must name the rule in use. No engine path changes.

Arithmetic (derived from the figures cited in `graph.rs:4148–4155`, not re-measured): 14.7 tok/s at 8 threads is 27.2% to 29.3% below 20.2–20.8 at 32 threads (1 − 14.7/20.2 and 1 − 14.7/20.8). The bench at 8 threads measures the configuration the comment calls the regression.

Gate: refaults (primary): not affected, since no engine path changes; run it anyway to confirm. tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents` (the engine path) flag ON and OFF interleaved, and `bench cpu-experts` at the default in each state. Both GPUs in every run. Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output is expected from the comment at `crates/breccia-kernels/src/cpu/expert.rs:290-300` (the result does not depend on thread count). The 32-thread case is untested (doc 32 card 5), so this card waits for that test.

Gain: UNVERIFIED - requires the GPU qualification card

**B. Miss-only O_DIRECT for streamed expert slices (flag OFF `BRECCIA_MISS_DIRECT`, proposed name).**

Idea. Fastllm reads its bounded disk-expert cache with O_DIRECT so that the bytes are not duplicated in the page cache (`diskdevice.cpp:160–169`). Breccia takes only the miss path. For each expert slice: call `mincore` on the slice's enclosing page range. If every page is resident, read through the mapping as today (a hit). If any page is not resident, `pread` with O_DIRECT into a per-worker aligned staging buffer, and compute from that buffer (a miss). Misses never enter the page cache, so they cannot evict hits.

Why this is not doc 35 option C. Option C keeps a second, anonymous copy of the cached bytes. Card B keeps no miss, so no byte is held twice. The staging buffer is transient.

Costs and risks (stated, not measured):

- Warm-up is required (the rule is in the §6 intro). Without it, the flag ON from token 0 finds no expert page resident, every streamed pick reads through O_DIRECT, nothing is inserted, and the hit fraction stays near zero. Each streamed pick then costs its full bytes: 675.58 MB per stream-token, which is 13.65 GB/s at the 20.2 tok/s aggregate, 2.33 times the 5.86 GB/s ceiling (derived; at 20.8 tok/s, 14.05 GB/s, 2.40 times).
- The page-cache set freezes after warm-up, because misses never insert. Under independent uniform references, a frozen set and LRU of the same size give the same expected hit fraction, so under uniform routing the expected hit fraction is the same 0.931 (derived, doc 35 §3). Under skewed routing, a frozen set loses adaptation. Card B therefore needs the C1 trace (doc 35) before it is trusted under skew.
- Eviction after the freeze. PLE row reads (`prefetch_rows`, `graph.rs:4063–4080`) and any other mapped read still insert pages, and those inserts can evict frozen expert pages. Misses never reinsert, so the resident expert set can only shrink after the freeze, and the hit fraction can only decay. A single `mincore` reading is not enough; the hit fraction must be sampled across the 200 tokens.
- The refault counters do not see card B's misses. O_DIRECT reads never enter the page cache, so a miss does not register as a workingset refault or a major fault. A flat refault delta does not show misses.
- O_DIRECT needs page-multiple offsets and lengths (doc 09 `:441`). Each read covers the enclosing page range, which adds at most two pages per miss: 219,610 B per stream-token, 0.47% of the miss bytes (derived).
- Staging memory is anonymous and counts against the cap. At most 32 workers (`graph.rs:990` default) each hold a buffer sized to the largest expert slice plus 8 KiB: 74,809,344 B, 0.23% of the cap. That displaces page cache, so it is a headroom cost (doc 35 H).
- `mincore` runs per slice and per role: 3 × 390 = 1,170 calls per stream-token (derived). Its cost is not measured.

Arithmetic (derived, §9):

- Miss bytes per stream-token: 46,440,448 B (doc 35 §9, bound under uniform routing).
- Requests: buffered readahead at 128 KiB (doc 35 §4) is 354 requests per stream-token. O_DIRECT is one request per miss, 26.8 per stream-token, before any kernel split (`max_sectors_kb` was not read).
- Device time if the miss bytes ran alone at 5.86 GB/s (doc 03 `:124`): 7.92 ms per stream-token. At the two-stream aggregate of 20.2 to 20.8 tok/s (`graph.rs:4148–4155`), the pair rate is 10.1 to 10.4 pair-tokens per second, so miss traffic is 0.94 to 0.97 GB/s, or 16.0% to 16.5% of that ceiling.
- Per-byte speed: doc 03 `:136` gives O_DIRECT at 5.9 GB/s against ra=128 at 5.0 GB/s, a ratio of 1.18. The card's expected gain is in refaults and request count, not bytes.

Conflict. Doc 03 §A.3 says do not switch to O_DIRECT and that O_DIRECT "deletes the hot set by construction". Card B is a narrower exception: it never deletes the hot set, because hits still use the mapping. It still needs the doc 03 owner to accept it. Doc 03's first-touch warming exception (`:176`) is unaffected.

Gate: hit fraction and misses per stream-token (primary, flag ON). The refault counters do not see O_DIRECT misses (see the risks above), so they are secondary for this card. Pass when the `mincore` hit fraction at token 200 after warm-up is not below its value at the freeze, and expert `workingset_refault_file` and `pgmajfault` deltas do not exceed the flag-OFF run (doc 31 `:57`). The uniform-routing expectation is 0.931 (derived). tok/s (secondary): aggregate from `bench token --streams 2` and `bench agents`, flag ON and OFF interleaved. Both GPUs in every run. Cap: MemoryMax=30G (32,212,254,720 B). Parity: exact against the flag-OFF output is expected, since the miss path feeds the same bytes to the same code. Verify it by output comparison; it is not measured here.

Gain: UNVERIFIED - requires the GPU qualification card

## 7. Unverified

- Fastllm: "no Vulkan backend" rests on the CMake file and the tree. The repo code search returned 403. The meaning of `routes` in the touch formula was read by an agent and not re-derived here.
- MNN: the semantics of `use_cached_mmap`, the load-time chunking, the bucket routing, and the Vulkan OOM claim (`VulkanRuntime.cpp:20–23`). The OOM claim was not verified and is dropped.
- Ratchet's Q8_0F vec4 pattern, crabml's contents, and WebLLM's kernels: agent-reported or not read.
- Whether Breccia's binding offsets meet the device's `minStorageBufferOffsetAlignment`: not checked.
- `mincore` per-call cost; `max_sectors_kb` on the btrfs device; O_DIRECT behaviour on 4 KiB-granular mappings (doc 31 `:10`).
- Card A: whether two Models probing the same device give a sound Auto budget. The probe order (`probe_budgets`, `model.rs`) was not traced.
- Card C: whether any gate in the repo reads `bench cpu-experts` at its default.
- Pool thread count in the multi-stream path. The comments say 16. `default_threads()` gives 32 on this host. Which one the shared pool gets depends on `cfg.threads` at construction, and that was not traced.
- The 99 ms and 96 ms per stream-token figures are derived from a code comment's aggregate. They are not measured.
- Per-token cost of the `prefetch_rows` thread spawns (§5).
- Whether the bench token path calls the generate loop (doc 35 §7).
- Card B: the 20-token plateau that ends warm-up, and the eviction effect of PLE reads on the hit fraction, are not measured.
- Parity for cards A, B and C: CPU and GPU expert bit-identity is not established in the repo (§6, card A). Card B's expectation rests on the miss path using the same bytes and code.

## 8. Conflicts

1. **Doc 31 against fastllm.** Doc 31 row 103 and line 139 say fastllm has "No memory-tier mechanism found". Fastllm has an O_DIRECT bounded disk-expert cache (`diskdevice.cpp:160–169`) and a visits gate (`:2528`). The row is wrong in part; the doc 31 owner should correct it.
2. **Engine default against bench default against help text.** The engine defaults to 32 (`graph.rs:4155`). The bench defaults to 8 (`bench.rs:114`). The help text says "physical cores/2, capped at 16" (`main.rs:277`). Card C.
3. **Pool thread count.** `multi.rs:25` and `graph.rs:441` say the shared pool is 16 threads. The code gives `default_threads()` clamped to 32 (`graph.rs:989–990`, `:4155`), and doc 32 `:11` states that count. Doc 35 §8.5 records the same conflict. Not resolved here.
4. **`prefetch_rows` comment and code.** The doc comment says "up to four threads" (`graph.rs:4063–4064`). The code uses `rows.len().min(8)` (`:4069`). No runtime change is proposed.
5. **Doc 03 §A.3 against card B.** Doc 03 says do not switch to O_DIRECT. Card B is a miss-only variant and needs the doc 03 owner's decision.
6. **Doc 35 option C against card B.** Option C needs the page-cache copy dropped or bytes count twice. Card B never inserts misses, so no byte is counted twice. It still freezes the page-cache set (§6, card B risks). This is a different trade, not a contradiction.

## 9. Arithmetic

Breccia values (doc 35 §9 and the code citations above).

- Streamed set, default: 34,589,900,800 B. All 48 blocks: 42,912,972,800 B. Difference: 8,323,072,000 B, the nine resident blocks.
- Lane bytes per stream-token, default: 10 × 34,589,900,800 / 512 = 675,584,000 B (675.58 MB). All 48: 10 × 42,912,972,800 / 512 = 838,144,000 B (838.14 MB). Difference: 162,560,000 B, +24.06%.
- Cap: C = 30 × 2^30 = 32,212,254,720 B. Default overflow: 34,589,900,800 − C = 2,377,646,080 B (2.21 GiB). All 48: 10,700,718,080 B (9.97 GiB).
- Uniform routing, per stream-token: picks per block = 10; streamed picks = 39 × 10 = 390. Miss fraction m = 0.068738 (doc 35 §3). Misses = 390 × 0.068738 = 26.81. Miss bytes = 11,338 pages × 4,096 B = 46,440,448 B (46.44 MB), which is the doc 35 §9 bound. Hit fraction 1 − m = 0.931262 (93.13%).
- Alignment, card B worst case: 26.81 misses × 2 pages × 4,096 B = 219,610 B. Share of miss bytes: 219,610 / 46,440,448 = 0.47%.
- Requests, card B: 46,440,448 / 131,072 (128 KiB) = 354.3 buffered readahead requests per stream-token. O_DIRECT: 26.8 requests per stream-token before any kernel split.
- Device time, card B: 46,440,448 / 5.86e9 B/s = 7.92 ms per stream-token if alone.
- Aggregate, card B: at 20.2 tok/s the pair rate is 10.1 pair-tokens/s, so miss bytes are 10.1 × 2 × 46,440,448 = 938.1 MB/s, 16.01% of 5.86 GB/s. At 20.8 tok/s: 10.4 × 2 × 46,440,448 = 966.0 MB/s, 16.48%. Per stream-token time: 1000/10.1 = 99.0 ms and 1000/10.4 = 96.2 ms.
- Per-byte ratio, card B: 5.9 / 5.0 = 1.18 (doc 03 `:136`).
- Staging, card B: 32 × (2,329,600 + 8,192) = 74,809,344 B, 0.23% of C. The 2,329,600 B is the largest block's all-role expert total, 1,192,755,200 / 512, used as a loose per-role bound.
- `mincore` calls, card B: 3 roles × 390 picks = 1,170 per stream-token.
- Card C: 14.7 / 20.2 = 0.728 (27.2% below); 14.7 / 20.8 = 0.707 (29.3% below). The reverse ratios are 1.374 and 1.415.
- Card B without warm-up: 675,584,000 B per stream-token × 20.2 aggregate tok/s = 13.65 GB/s, 2.33 times the 5.86 GB/s ceiling; at 20.8 tok/s, 14.05 GB/s, 2.40 times.

## 10. Hygiene

Public-repo rules applied to this file and the README index: no absolute paths, IP addresses, hostnames, personal names, or the words the brief bars. The grep result is recorded in the round 36 report, not here.
