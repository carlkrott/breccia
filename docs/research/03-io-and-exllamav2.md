# 03 — IO / page-cache strategy for a model larger than RAM, and an exllamav2 audit

Paired research round 3. Date: 2026-10-26. **Read-only with respect to engine code** — no file under
`crates/**` was touched. Everything numeric below was measured on this box in this round unless it is
explicitly marked *cited* or *projected*.

Bottlenecks this round was scoped against (from the spawn brief and `docs/phase-4/02-untried-levers.md`):

| | value |
|---|---|
| (a) host expert lane | 72–75 % of prefill, ~30 % of decode; 32 threads at 20–33 GB/s, single core 2.5–3.8 GB/s (ALU-bound) |
| (b) model > RAM | 43.8 GB expert shard + 28.8 GB PLE shard = **72.6 GB on 58 GB of RAM**; cold runs re-read **~21 GB** from NVMe |
| (c) GPU decode half | 70 ms over 1 509 dispatches; barrier now `sync2` (−2.5…−6.2 ms) |
| (d) per-(block,expert) residency | queued behind a routing histogram |

---

# Part A — IO / page-cache strategy

## A.0 The answer first

**One host configuration change is worth more than every other IO idea in this document combined:
set the btrfs BDI's readahead to 128 KiB.**

```bash
cat /sys/class/bdi/btrfs-1/read_ahead_kb        # this box: 8192  ← the problem
sudo sh -c 'echo 128 > /sys/class/bdi/btrfs-1/read_ahead_kb'
# restore: sudo sh -c 'echo 8192 > /sys/class/bdi/btrfs-1/read_ahead_kb'
```

Measured on our real access pattern (48 blocks × top-10 experts × 3 role-regions, cold, 8 worker
threads touching the mmap — i.e. exactly what `ExpertPool` does), 16 blocks / 233 MiB useful per run,
clean page cache dropped before every rep:

| btrfs-1 `read_ahead_kb` | useful GB/s | device bytes / useful bytes |
|---:|---:|---:|
| 64 | 2.57 | 2.00 |
| **128 (== `max_sectors_kb`)** | **4.7 – 5.4** | **1.05 – 1.14** |
| 192 | 3.33 | 1.79 |
| 256 | 2.60 | 2.39 |
| 384 | 2.33 | 2.63 |
| **8 192 (current)** | **0.69 – 1.10** | **5.9 – 9.0** |

**≈ 6–7× on every cold expert read, and ~7× less NVMe traffic, with no code change, no new syscall,
no copy, and the page cache still doing the caching.** It is reversible in one line and it is the
difference between a 28 s and a 4 s cold run.

Everything else in Part A — `O_DIRECT`, prefetch depth, `MADV_*`, `posix_fadvise` — is either neutral
or negative *once the readahead is right*. The detailed measurements follow, because the negative
results are as load-bearing as the positive one (three of them contradict advice we would otherwise
have copied from llama.cpp).

## A.1 Why the knob that matters is not the one anybody documents

Three layers set readahead, and only one of them is in play here:

1. **`/sys/block/nvme0n1/queue/read_ahead_kb`** (8 192 KB on this box) — the *block device* BDI.
   Model files are opened on **btrfs** (`/dev/nvme0n1p6`, mounted `compress=zstd:1,noatime,discard=async`),
   and btrfs files are backed by the filesystem's own BDI, **`/sys/class/bdi/btrfs-1/`**, not by the
   block device's. Writing the block-device knob changed *nothing* in my first sweep (128 → 8192
   all measured 0.42–0.43 GB/s, identical amplification), which is what sent me looking.
   `/sys/class/bdi/259:0` (the `@srv` subvolume number) also does nothing — verified.
2. **`posix_fadvise(fd, …)`** — file-scope, `POSIX_FADV_RANDOM` sets `file->f_ra.ra_pages = 0`.
   On btrfs this is **ineffective**: measured 0.91 GB/s with `FADV_RANDOM` against 1.10 without, and
   the over-read did not move (amp 6.8 vs 5.9). btrfs's own extent-map readahead dominates.
3. **`madvise(MADV_RANDOM)`** — VMA-scope, sets `VM_RAND_READ`, which disables *fault-around*
   (16-page/64 KiB mapping per fault) and readahead for those pages. It does change behaviour —
   in the wrong direction: 0.32–0.77 GB/s versus 0.69–1.10 for the default.

Why 128 KiB is exactly right: our access unit is a **contiguous 462.5 KiB (gate/up) or 900 KiB (down)
expert region**, and `max_sectors_kb` on this device is **128 KiB**. Readahead smaller than the region
can never over-read and always issues full-size requests; readahead larger than the region reads
neighbouring experts that the router will not ask for. 128 KiB = "one device-sized request, no waste"
is the mathematical optimum for a region ≫ 128 KiB, and the measurements agree to within 1.1×.

### What llama.cpp's two env knobs actually do (source read, not guessed)

`Reference Projects/llama.cpp` (upstream) and the qwen4exp fork `/home/korphaus/llama.cpp-qwen4exp-mmap`
(the binary our parity gate runs against):

* **`LLAMA_MMAP_FADVISE=normal|random`** — a *local fork patch* (`src/llama-mmap.cpp:488-497`), not
  upstream. It replaces the default `posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)` with
  `POSIX_FADV_NORMAL` or `POSIX_FADV_RANDOM` **on the whole fd at open time**. There is no per-tensor
  granularity and no MADV at all. Upstream llama.cpp hardcodes `POSIX_FADV_SEQUENTIAL`
  (`Reference Projects/llama.cpp/src/llama-mmap.cpp:488`) and additionally calls
  `posix_madvise(POSIX_MADV_RANDOM)` on its **"lazy ranges"** (`:519`) and `MAP_POPULATE` when
  prefetching is on (`:492`).
* **`LLAMA_MMAP_CPU_MADVISE=random|normal|willneed`** — also a fork patch
  (`src/llama-model-loader.cpp:1738-1752`). It calls `posix_madvise(beg, end, POSIX_MADV_RANDOM)` on
  **each CPU-resident tensor** as it is allocated, i.e. exactly our expert tensors, with the comment
  *"random stops read-around on page faults"*. This is the knob whose intent matches our problem —
  and it is measurably the wrong call **for our region size**: it buys 7–8× less device traffic at
  the cost of 1.4–2× less throughput, while `read_ahead_kb=128` buys both at once (amp 1.05 *and*
   5 GB/s). It is right for llama.cpp because their `-cmoe` path faults in per-expert slices through
  ggml's backend in a way that benefits from fewer speculative pages, and because their host-lane
  kernel is ALU-bound at ~3 GB/s so device traffic is the scarcer resource. Ours is the same
  ALU-bound shape, so the trade is closer than it looks — but amp 1.05 beats amp 1.0-with-0.9-GB/s.
* **`llama_prefetch`** (`:903`) is `madvise(MADV_WILLNEED)` over merged ranges — the ancestor of the
  prefetch experiment in A.4.

**Conclusion for us: do not inherit those two env vars into Breccia.** They are patches for a
per-tensor mmap layout on a machine whose filesystem readahead was never looked at.

## A.2 Priced option table (measured)

Cold = clean page cache dropped (`echo 1 > /proc/sys/vm/drop_caches`) immediately before the rep;
"useful GB/s" = expert-region bytes ÷ wall time; "amplify" = device sectors read (from
`/sys/block/nvme0n1/stat`) ÷ useful bytes. 3 reps/cell, spread shown as a range. 8–16 blocks
(117–233 MiB) per cell; single-token-scale work, seconds per cell.

| option | useful GB/s (cold) | amplify | cost/benefit |
|---|---:|---:|---|
| **mmap + touch, 8–32 threads, `read_ahead_kb=8192` (today)** | **0.69 – 1.10** | 5.9 – 9.0 | baseline |
| mmap + touch, `read_ahead_kb=128` (**recommended**) | **4.7 – 5.4** | **1.05 – 1.14** | one sysctl |
| `madvise(MADV_RANDOM)` on the expert ranges | 0.32 – 0.77 | 1.0 – 2.1 | **negative**; what `LLAMA_MMAP_CPU_MADVISE=random` does |
| `madvise(MADV_SEQUENTIAL)` | 0.25 – 0.60 | 3.8 – 26 | negative |
| `posix_fadvise(FADV_RANDOM)` | 0.24 / 0.91 | 6.8 – 15.6 | **no effect on btrfs** |
| `posix_fadvise(FADV_SEQUENTIAL)` | 0.16 | 26 | negative |
| `madvise(MADV_POPULATE_READ)` (sync, 1 thread) | 3.9 – 4.0 | 1.0 | good, but blocking; needs 5.14+ (have 6.18) |
| `MADV_WILLNEED` 1 block ahead, inline | 1.23 | 1.0 | the zero-copy prefetch; superseded by the sysctl |
| `MADV_WILLNEED` 2 blocks ahead + 32 threads | 3.6 – 5.0 | 1.0 – 1.6 | near-ceiling, unstable |
| `MADV_WILLNEED` 4 blocks ahead | 1.06 | 4.1 | **negative** — queue limit, see A.4 |
| buffered `pread`, 8–32 threads | 4.8 – 5.9 | 0.96 – 1.04 | needs a staging copy |
| `O_DIRECT` `pread` 128 KiB, 32 threads | **5.86 – 5.95** | 1.0 | device ceiling, most reproducible (±0.9 %) |
| sequential read of 400 MiB, 1 thread | 2.29 | 1.0 | device ceiling for *one* thread |
| warm mmap touch (page-cache hit), 32 threads | **58** | – | today's steady state; nothing beats it |

Device ceiling on this box: **≈ 6 GB/s** (O_DIRECT 32 threads, and buffered 32 threads both land
there; gen4 x4 NVMe). Every cold-path number is a fraction of that; the fractions are set by
over-read alone.

### What this is worth per token

Per-token expert bytes are **799 MiB** (cited, `docs/phase-4/02` §1). Cold cost of a token:

| miss fraction | bytes | today (0.9 GB/s) | `ra=128` (5.0 GB/s) | `O_DIRECT` (5.9) |
|---:|---:|---:|---:|---:|
| 100 % | 838 MB | **1 117 ms** | 168 ms | 142 ms |
| 20 % | 168 MB | 223 ms | 34 ms | 28 ms |
| 5 % | 42 MB | 56 ms | 8.4 ms | 7.1 ms |
| 0 % (resident) | — | 0 | 0 | 0 |

And the two whole-run numbers that matter:

* **Cold run**: 21 GB re-read → **28 s today → 4.2 s** at `ra=128`. (This is the "every comparison is
  in-window, the box drifts 2×" problem in `docs/phase-4/01` §9: the drift *is* this knob's variance,
  amplified by cache state.)
* **Merged chunked prefill (untried-levers rank 1)**: its byte floor is the distinct experts of the
  chunk — 40.9 GiB for 4 × 300 tokens, read once. At today's 0.9 GB/s that floor is **48 s**, i.e.
  **21 tok/s/agent**; the item is *never* going to reach 20–45 tok/s/agent with this knob where it is.
  At 5.0 GB/s the floor is **8.6 s → 140 tok/s/agent**, and the PCIe path (28.7 GB/s = 1.5 s) becomes
  the binding constraint, which is the ranking the untried-levers doc already assumed.

**This is the single highest-value finding of the round: ranked experiment #1 in the plan is gated on
a host sysctl nobody had measured.**

## A.3 `O_DIRECT` vs the page cache — do not switch

`O_DIRECT` (128 KiB requests, 32 threads) hits the device ceiling **5.86–5.95 GB/s with ±1 %
reproducibility**, better than buffered pread, and it never pollutes the page cache. It is still the
wrong choice for the engine, for one reason that dominates everything: **it throws away the only
mechanism that makes steady state fast.** 58 GB/s (warm mmap, zero-copy, 32 threads) versus 6 GB/s
(cold) is a **10× cliff**, and the page cache is what keeps us on the right side of it for ~95 % of
expert reads. A model that is 72.6 GB on 58 GB of RAM survives *only* because the hot set is
resident; `O_DIRECT` deletes the hot set by construction.

What the other engines do with SSD experts (cited, `docs/phase-4/02` §9 and the clones): MoE-Infinity
keeps experts on SSD behind an explicit request queue with prefetch across *requests* (it batches
many user requests so that each expert fetch is amortised across requests); colibri and PowerInfer
keep experts in host RAM and treat the SSD only as the load source; capy-flash and Basalt stream from
host RAM over PCIe. **None of them streams per-token expert weights from SSD in the steady state** —
that is a statement about the economics (6 GB/s SSD vs 33 GB/s host lane vs 28.7 GB/s PCIe), and our
numbers reproduce it: even at the fixed 5 GB/s, a 100 %-miss token costs 168 ms against the host
lane's 24.5 ms.

The one place `O_DIRECT` earns its keep: **first-touch warming at startup** (`llama-server`-style
preload), where there is no reuse to lose and the 6 GB/s deterministic rate beats the fault path.
Even there, `MADV_POPULATE_READ` at 3.9 GB/s with zero code is within 1.5× and needs no buffer
management.

## A.4 Prefetch shape and depth — priced, and mostly not worth building

`MADV_WILLNEED` on the *next block's* 30 regions, issued inline before touching the current block:

| depth (blocks) | useful GB/s | amplify | note |
|---:|---:|---:|---|
| 0 (none) | 0.24 | 15.6 | single-thread fault path |
| 1 | **1.23** | **1.0** | 5× today, and it reads *exactly* the useful bytes |
| 2 | 0.77 | 1.0 – 1.6 | with 32 threads: 3.6 – 5.0 |
| 4 | 0.41 | 4.1 | worse than none — queue saturation |

The physics of the depth limit is in sysfs and worth writing down, because it bounds every prefetch
scheme we could build:

```
nr_requests (nvme0n1)     = 256        # in-flight tags
max_sectors_kb            = 128        # largest request
→ max useful bytes in flight = 32 MiB
one block's expert set    = 18.7 MB (10 experts × 1.78 MiB)
→ optimum depth ≈ depth 1.7 blocks; depth 4 asks for 75 MB > 32 MiB → thrash (measured 1.06 GB/s)
```

Consequently:

* With `read_ahead_kb=128`, the plain fault path already reaches 4.7–5.4 GB/s, i.e. **86–92 % of the
  6 GB/s device ceiling**, so a prefetch layer buys ≈ nothing (168 → ~150 ms for a fully cold token).
  **Do not build it.**
* The prefetch idea we *would* have built (PowerInfer/colibri L+1 router prediction to warm the next
  block) is already a measured negative in this project (`docs/phase-4/02` §9, capy T25 −2.7 %) and
  my depth-4 result explains *why*: the predictor's extra bytes are not free because the in-flight
  budget is only 32 MiB, so prediction and demand reads compete for the same tags.
* If we ever do want the last 8 %: a single `readahead(fd, off, len)` (or `MADV_WILLNEED`) on the
  next block from one dedicated thread, depth 2. `readahead(2)` returns in **1.9 ms** for 200 MiB
  (it only queues), so it costs the caller nothing; it is the cheapest async read there is and it
  needs no staging buffer.
* "9.6 ms of misses" is consistent with **≈ 7 % miss fraction at today's rate** (0.07 × 799 MiB ÷
  0.9 GB/s ≈ 65 ms… which is too big; at 5 GB/s it is 11 ms). Read the other way: the 9.6 ms figure is
  what a *near-fully-resident* expert mmap costs when ~1–2 % of reads miss at today's slow rate, and
  it is what ~7 % of misses would cost after the fix. **Miss *rate* is the only thing that moves this
  number once readahead is right** — which is exactly the (d) residency item, not an IO item.

## A.5 The PLE mmap is a different problem, and it is the worst IO pattern in the model

The second shard is a **single tensor**: `per_layer_token_embd.weight`, `(160, 320 001 536)`,
type IQ4_NL → **90 bytes per 160-element record, 90-byte stride, 28.8 GiB total**. Every PLE lookup is
a **90-byte gather into a 28.8 GiB array**. Measured (`pleprobe`, 5 000 pseudo-random rows, cold):

| mode | µs / row | rows/s | device bytes / row |
|---|---:|---:|---:|
| cold, `pread` 90 B | **64.8** | 15 400 | 3.9 KiB (**48×** over-read) |
| cold, `FADV_RANDOM` + 90 B | 64.3 | 15 600 | 5.7 KiB (71×) — **no help** |
| cold, `pread` 4 KiB around the row | 79.4 | 12 600 | 67 KiB |
| **warm** (page-cache hit) | **0.5 – 0.9** | 1.1 – 2.0 M | – |

Three things follow:

1. A cold PLE row is **latency-bound at ~65 µs** no matter what you do with the file — 90 B ÷ 65 µs =
   1.4 MB/s per thread. The only lever is **rows in flight**: 32 threads → ~500 k rows/s. Our lane must
   never do a cold PLE row on a critical path single-threaded.
2. The 48–71× over-read is **btrfs extent-level, not readahead** (it survives `FADV_RANDOM` and it is
   ~68 KiB per random 4 KiB read: one compressed extent + metadata). Nothing at the syscall layer
   fixes it; only *locality* does.
3. Warm PLE is 0.5–0.9 µs/row, so **residency dominates**: keeping the *active-vocabulary* rows of the
   PLE resident is worth 100× on any lookup. At 90 B/row, 1 M active rows = 90 MiB. This is a far
   better use of the RAM budget than a random share of 28.8 GiB.
4. `docs/phase-4/01` §9 measures `ple` at 1.3 ms/token (token-by-token) but **14.7 ms/token** in the
   chunked path — an 11× regression that this measurement explains: 128 tokens × 48 layers = 6 144
   rows, at 2.4 µs/row average = 14.7 ms. **The chunked PLE path is a single-threaded cold gather.**

**Open question to hand to whoever owns the spec:** how a token maps to rows in `(160, 512·625 003)` is
not documented in `docs/specs/`; `320 001 536 = 2^9 × 625 003` and the 512 could be an expert/group
axis. If a token needs 512 × 90 B = 45 KiB spread over 625 003 strides, the prefetch shape is
completely different from "48 rows, one per layer" and the fix is a contiguous re-layout, not a
prefetcher. Worth 30 minutes of reading `qwen4exp-architecture.md` before building anything for PLE.

## A.6 Recommended IO configuration (the one to test)

```bash
# 1. the knob (reversible; verified restored to 8192 after this round)
sudo sh -c 'echo 128 > /sys/class/bdi/btrfs-1/read_ahead_kb'

# 2. everything else: leave at defaults
#    - no O_DIRECT, no posix_fadvise, no MADV_RANDOM, no MADV_SEQUENTIAL, no prefetch thread
#    - keep the mmap + zero-copy read path (warm 58 GB/s is unbeatable)
#    - keep 8-32 fault threads; thread count does not matter once amplification is 1.0
```

Engine-side, the only change worth landing is a **diagnostic, not a behaviour change**: have
`breccia-profile` read `inode_to_bdi()`'s `read_ahead_kb` for the model's filesystem (plus
`max_sectors_kb`) and print a warning when `read_ahead_kb > max_sectors_kb` and the model does not fit
in RAM. Five lines, no privileges, and it would have caught this.

Optional, only if we later bind on PLE or cold-start: a **32-thread parallel 4 KiB gather** for PLE rows
(and never a 90-byte `pread`), issuing in bulks of ≥ 256 rows so btrfs's per-row extent read is
amortised.

## A.7 What I did not measure, and why

* **A full cold model load** (72.6 GB) — the brief forbids long full-model loads; every number above is
  a 117–233 MiB window, 3 reps, seconds per cell. The extrapolations are linear in bytes and labelled.
* **`VK_EXT_external_memory_host` / ReBAR zero-copy reads** — GPU, out of scope this round.
* **io_uring with `IORING_SETUP_SQPOLL`** — would only matter if we replaced the fault path with an
  explicit async COMIT ring; the sysctl makes that unnecessary.
* **The `zstd:1` compression factor on the expert shards** — the PLE over-read is consistent with
  incompressible data still paying per-extent reads, but I did not confirm whether btrfs actually
  stored the expert extents compressed (`compsize` needs root and is slow on 44 GB). If it *did*
  compress them, a `chattr +c`-free copy would trade disk for speed; not worth it at 5 GB/s.

Environment notes for reproduction: kernel `6.18.55-2-cachyos-lts`; `/srv` on btrfs
(`compress=zstd:1,noatime,discard=async`); `read_ahead_kb` 8192 on both `nvme0n1` and `btrfs-1`;
`max_sectors_kb` 128; `nr_requests` 256; scheduler `kyber`; THP `always`; `vm.swappiness=10`;
64 MiB L3 per CCD; 58 GB RAM. No GPU service was running or touched (`qwen-dual`, `strata-rx`,
`d1-mi50` all inactive before and after); `drop_caches` was used between reps and the page cache was
allowed to refill (it self-heals). The probe sources and raw outputs are in
`/var/tmp/users/korphaus/tmp/opencode/p5io/` (`ioprobe.c`, `pleprobe.c`, `layout.py`, `sweep*.sh`,
`out*.txt`).

---

# Part B — exllamav2 audit

Clone: `Reference Projects/exllamav2` @ `7dc12af3a81f34ac3f27cd7602ed539b638933ca` (2026-03-04),
`git clone --depth 1 https://github.com/turboderp-org/exllamav2` — the org in the brief is canonical and
still resolves; the repo is alive but **EXL3 (the current format) lives in `exllamav3`, not here** —
`grep -rl EXL3` finds one test file. So this audit covers **EXL2 only**; that is still the right target
because the *method* (measure-then-allocate) is identical and EXL3 only changes the codebook.

## B.1 Quantisation format design — the part worth stealing (as a method)

* **Per-module, not per-model, bit allocation.** `conversion/qparams.py` describes a candidate as
  `(group_size, bits[], bits_prop[], scale_bits)` — a *list* of bitrates with proportions
  (`bits_prop` = the fraction of groups at each rate, so a single tensor can be 60 % 4-bit / 40 % 6-bit),
  4-bit scales, group size 32 by default. Note that this is *finer than per-layer*: it is per
  `/`-matrix with a mix of rates inside one matrix.
* **The measurement** (`conversion/measure.py`): for each module in turn, quantise it with each
  candidate, run the module forward on **real calibration activations** (c4_sample, 2048 tokens), and
  score against the *f16 module's own output*:
  `accuracy = 1 − mean(‖x_q − x_ref‖_F / ‖x_ref‖_F)` (`test_error`, `:126-142`). Two things matter:
  the error is **module-output**, not weight MSE; and the activations come from the *real* residual
  stream, so a module that sees large activations is penalised automatically.
* **The allocation** (`conversion/optimize.py`): every `(module, candidate)` becomes a slot
  `(total_bits, 1 − accuracy)`, and **simulated annealing** (`ext_c.sim_anneal`, 3 stages × 80 samples,
  `norm` temperature swept over `[1.5, 3.5]` to trace the Pareto front) picks one candidate per module
  under a hard **total bit budget** `numel × target_bpw`. It reports both `sum(log(err))` and
  `max(err)`, so you can buy average quality or worst-layer quality deliberately.
* **Why per-layer matters, quantitatively:** a uniform bitrate is optimal only if every module has the
  same sensitivity, and they differ by an order of magnitude (attn `q/k/v/o` vs `down_proj`, early vs
  late layers). At a fixed average bitrate the optimiser moves bits from insensitive modules to
  sensitive ones; that is what lets EXL2 4.0 bpw sit near uniform 4.65 bpw at equal KLD.

**Applicability to us: near zero for weights, moderate for KV.** Our weights are a fixed artifact
(the fork's GGUF already carries a per-layer type assignment: IQ2_XXS for 9 layers' gate/up, IQ2_XS
10, IQ2_S 10, IQ3_XXS 6, IQ3_S 13, `down` in IQ4_NL 18 / Q2_0 30 — i.e. the *producer* already ran an
EXL2-style allocation, cited `docs/phase-3/08` §2), and our gate is bit-identical logits against the
production server, so re-quantising is out of scope twice over. Where the method transfers is
**Phase 6's KV quantisation**: we will have to choose bits per (layer, K vs V) under a 2 048 B/cell
budget, and the right way to choose is exactly this harness — measure per-layer K/V output error on
real prompts, anneal a budget, and gate on KL/top-1 rather than on bit count.

## B.2 Cache / KV strategy

* Three cache classes (`exllamav2/cache.py`): fp16, `Cache_8bit` (fp8 K and V with per-block scales),
  and `Cache_Q` — grouped RTN with **independent `weights_per_byte_k` and `weights_per_byte_v`**, i.e.
  K and V can hold different bitrates (the same idea as ik_llama.cpp's `-ctk`/`-ctv`).
* **PagedAttention is native, not bolted on**: every `get_kv_state`/`store_kv_state` takes
  `page_size`, `block_table`, `cache_seqlens` and the dequant kernel (`q_to_fp16_kv`) indexes through
  them (`cache.py:472-516`). Their batch decode is a page-table gather in the attention kernel.
* **Quant block aligned to whole tokens** (`cache.py:452-458`): `q_block` is the smallest number of
  tokens such that `kv_dim × q_block` is a multiple of 512, and the cache length is rounded up to it.
  Consequence: a decode step *never* has to dequantise a partial block, and every read is block-aligned.
  Cheap trick, real win, and we will hit exactly this problem in Phase 6.
* `fixed_device` per cache: the KV cache (and its temp fp16 shadows) can be pinned to a chosen GPU
  independently of the model's module placement.

## B.3 CUDA graphs + stream structure — and its Vulkan analogue

* Graph capture is **per module, inside the fused handle**: `make_q_mlp(..., not cfg.no_graphs)`
  (`mlp.py:203-224`) and the same for attention (`attn.py:329`). Each MLP/attention owns a captured
  graph; replay is one call. `EXLLAMA_NO_GRAPHS` disables it.
* **One C++ call per module** is the real structural point: `make_q_mlp` fuses
  `rmsnorm → gate → up → SiLU → down → residual` into a single handle, i.e. one host call and one
  graph covering ~5 kernels. Their per-layer launch count is therefore ~2 (one attention handle, one
  MLP handle). **Ours is ~31 dispatches per block** (1 509 ÷ 48). That is the (c) bottleneck expressed
  as a ratio.
* Kernel selection is **static per shape**, 3 families × 2 variants chosen on `max_m`
  (`cuda/comp_units/kernel_select.cu`): `1a/1b` = GEMV (m = 1), `2a/2b` = split-K for small m,
  `3a/3b` = plain GEMM (`a`/`b` = with/without output-weight handling). Nothing exotic, and we
  already have the same three shapes (`gemv`, `gemv_*_moe`, `GemmMany`).
* **Streams: there is no overlap structure to copy.** Single default compute stream; no PDL, no
  handshakes, no multi-stream pipelining. `parallel_decoder.py` looks like parallelism but is an
  *architecture* fusion (models whose block runs attention and MLP in parallel off one norm), not two
  queues. `stloader` (below) is the only place with a second thread.

**What this means for (c):** exllamav2's 70 ms-equivalent advantage is bought by *fewer, bigger
launches*, which for us is untried-levers item 7 (fuse the device MoE into one row-mapped dispatch) —
not by graph replay. Their graph replay buys ~0.3 ms/token at our dispatch cost, which is what we
already measured (`docs/phase-4/02` §8, "recording is 0.32 ms/token").

**But there is one genuinely new idea here.** A CUDA graph is replayable because its *geometry* is
frozen while *buffer contents* vary. Our per-block command buffers cannot be replayed because
`dispatch_batch` re-records geometry (workgroup counts derived from the routing row map) into a ring
of 6 buffers, resetting each before use (`crates/breccia-vk/src/ctx.rs:1082-1094`,
`BATCH_CMDS = 6`). **The Vulkan equivalent of a graph is a pre-recorded command buffer + indirect
dispatch**: put the per-(block, expert) group counts in a buffer, record `vkCmdDispatchIndirect`
once per block, and reuse the recording for every token. That removes the "a pending buffer cannot be
reset" constraint that forces the 6-deep ring, and it makes the *geometry* a device-side detail.
Measurable: recording time (0.32 ms/token → ~0) and ring depth 6 → 2.

## B.4 Batching / scheduling

`exllamav2/generator/dynamic.py` + `server/websocket.py`: a fixed `max_batch_size` decided at load,
`max_input_len × max_batch_size` scratch preallocated up front (the same compile-time memory-planning
lesson MLC gave us in round 1), one cache slot per batch row, and a fill-then-run loop. No chunked
prefill, no continuous batching across a request's decode, no radix/prefix sharing — this is a
*single-user-first* engine with a small batching concession. Nothing here beats what we already have
from vLLM/SGLang in `docs/phase-4/02` §8 (chunked prefill interleave) — **skip**.

## B.5 `stloader` — arrives at Part A's answer independently

`exllamav2/exllamav2_ext/ext_stloader.cpp`: **8 reader threads, each `pread`-ing 1 MiB blocks strided
8 MiB apart**, pushing completed blocks onto a deque; one copy worker merges adjacent blocks and issues
`cudaMemcpyAsync` per merged run. `STLOADER_BLOCK_SIZE = 1 MiB`, `STLOADER_THREADS = 8`, both hardcoded.

That is exactly the shape my Part A probe found optimal *without* knowing about it (8–32-thread
explicit `pread` = 4.8–5.9 GB/s cold against 0.7–1.1 for the fault path), and `pread` rather than
mmap-touch for the loading path. Their loader is used only to fill VRAM at startup — they never had to
solve the > RAM case — but the corroboration is useful: an engine with far more IO experience than us
chose "8 threads × 1 MiB pread", not "mmap and hope".

## B.6 What maps onto our four bottlenecks

| our bottleneck | what exllamav2 offers | verdict |
|---|---|---|
| (a) host expert lane 72–75 % of prefill | **nothing** — their MoE is fully device-resident, torch fallback for m > 4, no CPU experts in the fast path. | no adaptation. |
| (b) model > RAM | `stloader`'s 8 × 1 MiB pread pattern (corroboration only); no residency policy, no SSD tier. | see Part A; the sysctl is the lever, not their code. |
| (c) 70 ms over 1 509 dispatches | per-module fused handle + per-module graph; 3 kernel variants chosen on m; `q_moe_mlp_forward_` = one launch for routing + gather + all experts when `m ≤ 4` and `experts ∈ {4,8,16,128}` (`moe_mlp.py:240-247`). | **fused single-dispatch MoE (item 7) confirmed**, plus the indirect-dispatch idea in B.3. |
| (d) per-(block, expert) residency | each expert is its own `ExLlamaV2Linear` with its own `device_idx` (`moe_mlp.py:211-221`) — per-expert placement *exists*, but it is static config, not frequency-ranked. | no adaptation; our PowerInfer plan is ahead of them. |
| KV / streams | independent K/V bitrates, page-table-native dequant, token-aligned quant blocks, per-cache `fixed_device`. | Phase 6 (KV) and a cheap stream-placement lever. |

## B.7 Explicitly CUDA-only, with no Vulkan equivalent

* **Graph capture itself** — `cudaStreamBeginCapture`/`cudaGraphLaunch` has no Vulkan command. The
  analogue is a pre-recorded `VkCommandBuffer` (+ indirect dispatch for varying geometry), which is
  weaker: no cross-buffer node dependencies without timeline semaphores, and RADV re-validates
  nothing so our gain is only the host recording cost, which is already 0.32 ms.
* **`cudaMemcpyAsync` on a copy engine inside the loader** — we use a transfer queue; measured, that
  is fine *except* the MI50, where the transfer queue as a global flag was 8.6× slower on-card
  (research 02). Nothing to port.
* **fp8 (`e4m3`) KV with hardware conversion** — `Cache_8bit` relies on `__nv_fp8` conversion in the
  dequant kernel. There is no fp8 storage format in Vulkan core; we would store q8 as int8 + scale,
  which is what `Cache_Q` does anyway. Not a blocker, just a different container.
* **`__syncthreads`-free small-m kernel families** (`comp_units`) — CUDA-specific warp-synchronous
  tricks; Vulkan needs `subgroupBarrier`/`memoryBarrierShared`, and our measured barrier cost is
  already ~0 as a packet (research 02). Skip.
* **`bits_prop` (mixed bitrates inside one matrix)** — this needs a per-group bitrate index that the
  kernel reads at run time; expressible in GLSL, but it is a *format* feature, and our format is fixed.
  Not applicable.

## B.8 Ranked adaptations (exllamav2 → Breccia)

| # | adaptation | maps to | expected effect | effort |
|---|---|---|---|---|
| 1 | **Indirect-dispatch replay**: freeze per-block dispatch geometry (`vkCmdDispatchIndirect`, counts in a buffer), so one recorded command buffer serves every token and block | (c) | recording 0.32 ms/token → ~0; ring 6 → 2; unblocks deeper pipelining. Also removes the per-token geometry re-derivation from the host critical path — the part that *is* 55–71 % of the step | M |
| 2 | **Fused single-dispatch MoE** (their `q_moe_mlp_forward_`, m ≤ 4): row-mapped gather + all experts + SwiGLU in one dispatch | (c), and it is the multiplier for (d) | 2.0 → 1.0–1.2 ms/block = **−7 ms/token** at 9 resident blocks, and it moves the residency crossover (untried-levers item 7) | M |
| 3 | **Measure-then-allocate harness for KV bits** (module-output error on real prompts + annealed budget), with token-aligned quant blocks and independent K/V bitrates | Phase 6, stream count | at 300 tokens KV is 24 KiB/token (0.9 % of expert bytes) so ~0 today; it is the *right* way to spend the VRAM lever at ≥ 2 k context | M |
| 4 | **Per-cache `fixed_device`**: pin stream *i*'s KV + GDN state to the MI50 while the RX owns experts | streams | KV+state is small today (~28 MB at 4 × 300) — park until streams are VRAM-bound; it is free to implement as an arena flag | S |

Deliberately not ranked: their batching/scheduling (weaker than what we already have), their
`stloader` (Part A already covers the ground), and their graph-per-module (we have the equivalent).

---

# Top-3 next experiments, with the bench for each

**1. BDI readahead A/B, one window, cold.** The highest value-per-minute experiment in this document.

```bash
# warm the process, then alternate A/B/A inside one window with the cache dropped between arms
for arm in 8192 128 8192 128; do
  sudo sh -c "echo $arm > /sys/class/bdi/btrfs-1/read_ahead_kb"
  sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'
  ./target/release/breccia bench agents --streams 4 --prefix 300 --prefill-chunk 256 \
      --json /var/tmp/users/korphaus/tmp/opencode/ra-$arm-$(date +%s).json
done
sudo sh -c 'echo 8192 > /sys/class/bdi/btrfs-1/read_ahead_kb'   # restore
```
Numbers to move: prompt seconds (today 95.6 s for 4 × 300), and `bench cpu-experts` first-pass vs
best-pass battery ratio (today "cold 42.4 ms, best 41.1 ms" — that gap *is* the miss cost at `ra=8192`).
Parity gate untouched (IO cannot change results). Expected: prefill floor 21 → 140 tok/s/agent, cold
run 28 → 4 s.

**2. PLE gather: parallelise it, and check the layout.** 14.7 ms/token in the chunked path against
1.3 ms/token in the token-by-token path is an 11× regression that should not exist.

```bash
BRECCIA_TRACE_PREFILL=1 timeout 180 ./target/release/breccia bench agents --streams 2 --prefix 128 \
    --prefill-chunk 128 2>&1 | grep -E "^ple|stage"
# then: issue the chunk's PLE rows as one 32-thread bulk of 4 KiB preads (never 90-byte readahead reads)
```
Number to move: `ple` 14.7 → ≤ 2 ms/prompt token. Also settle the `(160, 512·625 003)` indexing
question before choosing a prefetch shape.

**3. Indirect-dispatch replay (exllamav2's graph, in Vulkan).**

```bash
timeout 180 ./target/release/breccia bench token --devices 0000:0e:00.0 --gen 32 --runs 3 \
    --record-replay on   # new flag; geometry in a buffer, vkCmdDispatchIndirect
timeout 180 ./target/release/breccia bench token --devices 0000:03:00.0 --gen 32 --runs 3 \
    --record-replay on
```
Number to move: recording time per token (0.32 ms) → ~0.02, ring depth 6 → 2, and the p50 of the
host-side gap between block submissions. Parity gate: 64/64 must stay bit-identical (same kernels,
same order). MI50 first (AGENTS.md), validation layer on the first run.

---

## One-line verdicts (for the README)

* **IO.** One sysctl beats every code change: `read_ahead_kb=128` on the **btrfs BDI** (not the block
  device's) takes cold expert reads from **0.7–1.1 → 4.7–5.4 GB/s** at **1.05×** amplification instead
  of 9×. `O_DIRECT`, `FADV_RANDOM`, `MADV_RANDOM`, prefetch depth > 2 and `madvise` advice in general
  are neutral-to-negative once that is right; llama.cpp's `LLAMA_MMAP_*ADVISE` knobs are the wrong
  trade for our 462–900 KiB regions and must not be inherited. PLE is a separate pathology: 90-byte
  records, 65 µs cold, 48× over-read, 0.5 µs warm — residency and parallelism, not advice.
* **exllamav2.** Its per-module *measure-then-anneal* bit allocation is the best-articulated version of
  an idea we can only use for KV (our weights are a fixed artifact with a bit-identical gate); its one
  structural lesson for (c) is *one fused call per module* (≈2 launches/layer against our 31), whose
  Vulkan form is **indirect dispatch over a pre-recorded command buffer** plus the fused MoE we
  already had queued. No residency, batching or stream-overlap ideas to take.
