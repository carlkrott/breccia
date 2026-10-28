# 37 — Lossless expert compression against the 30 GiB cap

Round 37. Docs only: no code changed, no build, no GPU or model process run, and the model shards were not loaded. Base commit 0c74704. Every gain below is UNVERIFIED until the GPU qualification card runs. Arithmetic is labelled derived where it is one, and a bound where it assumes uniform routing. Entropy figures are measured on the sample named in each line; a sample figure is not a model figure. GB is 10^9 bytes; GiB is 2^30 bytes.

## 0. Verdict

| Quantity | Bytes | Ratio to streamed set | Saving | Over the cap after the tier |
|---|---|---|---|---|
| Streamed set S (39 blocks, gate, up and down) | 34,589,900,800 | 1.00000 | 0 | 2,377,646,080 B (2.21 GiB) before any tier |
| Cap C (MemoryMax=30G) | 32,212,254,720 | 0.931262 | 6.874% required | 0 |
| Measured Huffman, integer code lengths, no table cost | 33,781,251,066 | 0.97662 | 2.34% | 1.46 GiB |
| Ideal order-0, as coded | 33,391,778,908 | 0.96536 | 3.46% | 1.10 GiB |
| Ideal plus d-coding bound | 32,941,664,722 | 0.95235 | 4.77% | 0.68 GiB |
| Required with 1 GiB of host headroom | 31,138,512,896 | 0.900220 | 9.98% | not reached |
| Required with 2 GiB of host headroom | 30,064,771,072 | 0.869178 | 13.08% | not reached |

1. **Order-0 lossless coding does not bring the model shards under the cap.** The cap needs a ratio of 0.931262. The best measured ratio is 0.95235, and that uses the optimistic bound on the d fields. The shortfall is 2.1 points at the bound, 3.4 points at ideal order-0 and 4.5 points at Huffman. Headroom for the host makes the gap larger.
2. **The savings are in the small fields, and the big fields are close to full entropy.** The IQ3_XXS grid is at 7.803 of 8 bits, and its sign field is at 6.9998 of 7 bits. The sign field is 28.6% of an IQ3_XXS block's bits and does not compress at order 0. The largest gains are on the down role: Q2_0 codes sit at 1.805 of 2 bits, and IQ4_NL nibbles at 3.911 of 4.
3. **The miss savings are real, but decode of the hit bytes is unlikely to pay.** Under uniform routing the tier removes 15.1 MB (Huffman) to 31.5 MB (d-bound) of the 46.44 MB of miss bytes per stream-token (32% to 68%, derived). To do it, the tier decodes 644 MB to 661 MB of hit bytes per stream-token. Break-even needs decode at least 21 to 43 times the refault byte rate. On the 0.94 GB/s stand-in that is a floor of 20 to 40 GB/s, about 10 to 19 cores at the zstd proxy speed of 2.09 GB/s per core. The lane those cores would take from is already CPU-bound (doc 30: 85.7 ms per step at 8 threads, 41.2 ms at 32), and decode adds 0.31 s per stream-token to the 0.18 to 0.27 s of dequant, so lane CPU work at least doubles. The sustained decode demand for the pair is about 13 GB/s, which is about 6 single-core equivalents at the zstd proxy speed (derived).
4. **Card order.** Card A measures the refault baseline under MemoryMax=30G, and that measurement decides whether card B is needed. Card B is a bench-only decode test with a kill test. Card C is built only on a pass.
5. **Lossy options are not carded.** Doc 24 2-bit shadows, requantization and the NeuZip lossy mode are rejected (§6).

Ranked flag-OFF cards: A, B, C (§5). None has a measured gain.

## 1. As built (base 0c74704)

- **Expert layout.** The model is a 48-block MoE with 512 experts and top-10 routing, stored in two GGUF shards. The header parse found shard 1 data at 11,024,256 B with 1,223 tensors and shard 2 data at 192 B with one tensor. Alignment is 32 B.
- **Quant geometry** (`crates/breccia-format/src/quant.rs`). IQ3_XXS is 256 values per 98 B. IQ4_NL is 32 per 18 B. Q2_0 is 64 per 18 B (fork-specific type 42). IQ2_XXS is 256 per 66 B, IQ2_XS 256 per 74 B, IQ2_S 256 per 82 B and IQ3_S 256 per 110 B. The IQ3_XXS field layout (fp16 d, 64 grid bytes, 8 sign words, 4-bit scales) was checked against `crates/breccia-kernels/src/cpu/scalar.rs:160–190`.
- **Type census: the gate and up roles are not IQ3_XXS.** Per block, gate and up use IQ3_S in 13 blocks, IQ2_S in 10, IQ2_XS in 10, IQ2_XXS in 9 and IQ3_XXS in 6. Down uses Q2_0 in 30 blocks and IQ4_NL in 18. Streamed bytes per role (gate and up carry the same figures each):

| Role | Type | Blocks | Streamed bytes |
|---|---|---|---|
| gate, and up | IQ3_S | 13 | 2,883,584,000 |
| gate, and up | IQ2_S | 10 | 2,686,976,000 |
| gate, and up | IQ2_XS | 10 | 1,697,382,400 |
| gate, and up | IQ2_XXS | 9 | 1,730,150,400 |
| gate, and up | IQ3_XXS | 6 | 1,926,758,400 |
| down | Q2_0 | 30 | 5,662,310,400 |
| down | IQ4_NL | 18 | 7,077,888,000 |

  The total is 34,589,900,800 B, the same streamed total as doc 35. The brief's framing of the gate and up roles as IQ3_XXS does not match these types (§8).
- **Resident and streamed sets.** Each stream keeps blocks 0, 1, 2, 4, 8, 18, 20, 22 and 23 resident (8,323,072,000 B). The other 39 blocks are streamed. All 48 blocks would be 42,912,972,800 B (doc 36 §9).
- **Two streams share the shards.** Both streams map the same shards (`crates/breccia-format/src/gguf.rs:98`), so each expert byte sits once in the page cache.
- **The disk bytes are the model bytes.** The shard extents are not compressed by the filesystem. Filefrag flags show no encoded extents, and a positive control on a scratch volume with compression enabled showed the encoded flag. A refault therefore reads the model's own bytes.
- **Host lane.** The CPU expert kernel is at `crates/breccia-kernels/src/cpu/expert.rs:463`, and the engine's `moe_topk` is at `crates/breccia-engine/src/graph.rs:3137`. Doc 30 §8y gives the lane at 41.2 ms per step at 32 threads and 85.7 ms at 8. Doc 03 gives single-core lane decode at 2.5 to 3.8 GB/s, and a device ceiling of 5.86 GB/s (doc 03 `:124`).

## 2. Per-stream entropy on a sample

The sample is six experts from blocks 47, 45, 6, 19, 24 and 29. The brief asked for three or four experts. Six were used so that every gate and up type appears. That gives 18 expert tensors, measured on 9.96 MiB sample (10,444,800 B). The IQ3_XXS figures come from four expert tensors in blocks 47 and 45, measured on 2.39 MiB sample (2,508,800 B, 25,600 blocks), because those are the only IQ3_XXS tensors in the sample. The IQ4_NL figures come from the two IQ4_NL down tensors, measured on 1.76 MiB sample (1,843,200 B, 102,400 blocks of 18 B). The Q2_0 figures come from the four Q2_0 down tensors, measured on 1.76 MiB sample (1,843,200 B, 102,400 blocks of 18 B).

Field-level entropy and per-field compression, with the field width in bits:

| Stream | Field width | Entropy (bits) | zstd -1 ratio | zstd -3 ratio | Notes |
|---|---|---|---|---|---|
| IQ3_XXS grid (64 B per block) | 8 | 7.803 (measured on 2.39 MiB sample) | 0.9801 | 0.9819 | lz4 -1 gives no gain (ratio about 1.00) |
| IQ3_XXS signs (28 B per block) | 7 | 6.9998 (measured on 2.39 MiB sample) | 1.0000 | not run | incompressible |
| IQ3_XXS scales (4 B per block) | 4 | 3.353 (measured on 2.39 MiB sample) | 0.8348 | not run | |
| IQ3_XXS d (2 B per block) | 16 | 10.21 (measured on 2.39 MiB sample; 16-bit plug-in, biased low) | 0.7654 | not run | |
| IQ4_NL nibbles (16 B per block) | 4 | 3.911 (measured on 1.76 MiB sample) | 0.9911 | 0.9878 | |
| IQ4_NL d | 16 | 11.38 (measured on 1.76 MiB sample; 16-bit plug-in) | 0.8223 | not run | |
| Q2_0 codes (16 B per block) | 2 | 1.805 (measured on 1.76 MiB sample) | 0.9072 | not run | |
| Q2_0 d | 16 | 11.05 (measured on 1.76 MiB sample; 16-bit plug-in) | 0.8089 | not run | |

Whole-type zstd -1 ratios over all of each type's streams: IQ3 0.9755, IQ4 0.9723, Q2 0.8962. The gap to the order-0 ideal comes from the small scale and d streams, which zstd compresses well and the order-0 ideal leaves uncoded (§8).

Ideal and Huffman ratios per stream type, with the byte-class split used where the type has no field table:

| Type (role) | Coding in the ideal | Sample | Ideal (order-0) | Huffman (integer code lengths) |
|---|---|---|---|---|
| IQ3_XXS (gate, up) | fields, d raw | measured on 2.39 MiB sample (2,508,800 B) | 0.9773 | 0.9804 |
| IQ4_NL (down) | fields, d raw | measured on 1.76 MiB sample (1,843,200 B) | 0.9801 | 0.9913 |
| Q2_0 (down) | fields, d raw | measured on 1.76 MiB sample (1,843,200 B) | 0.9134 | 0.9572 |
| IQ2_XXS (gate, up) | byte classes d, grid, sign_scale; d coded | measured on 0.81 MiB sample (844,800 B, one expert) | 0.98132 | 0.9852 |
| IQ2_XS (gate, up) | byte classes d, qs_lo, qs_hi, scales; d coded | measured on 0.90 MiB sample (947,200 B, one expert) | 0.96874 | 0.97181 |
| IQ2_S (gate, up) | byte classes d, grid_lo, signs, qh, scales; d coded | measured on 1.00 MiB sample (1,049,600 B, one expert) | 0.96926 | 0.97135 |
| IQ3_S (gate, up) | byte classes d, qs, qh, signs, scales; d coded | measured on 1.34 MiB sample (1,408,000 B, one expert) | 0.97504 | 0.97779 |

The IQ2 and IQ3_S rows are byte-class plug-in entropies on one expert's two tensors per type. They are a sample limitation, and they are the only rows where d is coded in the ideal. The IQ3_XXS ideal is the bit-weighted sum of its fields over 784 bits per block: (512 × 0.97538 + 224 × 0.99997 + 32 × 0.83825 + 16 × 1.0) / 784 = 0.9773.

Decode speed, one pinned core with a load average near 8, one run each (noisy). zstd -1 on the grid stream compresses at 1560 MB/s and decompresses at 2090 MB/s. zstd -3 gives 982 and 2187 MB/s. IQ4 codes at zstd -1 give 2045 and 4239 MB/s. lz4 -1 does not compress the grid stream, and its decode rate is memory-copy speed, so it is not a decoder speed.

Weighted over the whole streamed set, these per-type ratios give the §0 figures: ideal 0.96536, Huffman 0.97662, ideal plus d bound 0.95235. The IQ2 and IQ3_S rows are the split ideals above. Recomputing from 5-digit per-type ratios moves the third decimal by less than 0.00002.

## 3. Literature and engines

Every citation is pinned to a commit, a paper ID or a kernel tag. Items marked UNVERIFIED were not confirmed at the pin.

- **ZipNN** (arXiv 2411.05239v2). For BF16 Llama-3.1, compressed size is Zstd 77.7%, EE+Zstd 68.8% and ZipNN 66.4%. GPTQ and AWQ compress to 85% to 91%. The paper says the fraction and sign bits "hardly compress at all", and says other GGUF-quantized models "do not compress at all". The GGUF statement covers the quants the paper tested. Which quants those were is not confirmed, so the statement is not carried to IQ3_XXS here.
- **DFloat11** (arXiv 2504.11651v3). Exponent-only Huffman on BF16 weights, about 70% of the original size, bit-identical output. Reported token-generation throughput is 2.3 to 46.2 times that of CPU offload, and kernel decompression is up to 20.97 times faster than nvCOMP. It codes BF16 exponents, not block-quantized IQ formats.
- **Huff-LLM** (arXiv 2502.00922v1). An FPGA prototype with up to 32% smaller size, 31% lower latency and 26% lower energy, as reported by the literature pass from the abstract, not re-read here. Llama-3-8B bits per parameter are 10.96 from an FP16 base and 11.68 from a BF16 base, with 6.13% area overhead. The simulated latency figures are not verified.
- **NeuZip** (arXiv 2410.20650v1). Its lossless 7-bit mode uses 10.95 units of memory against 15.08 for vanilla, with the same perplexity (9.89 for both). The lossy mode is rejected here. Table units are as printed; GiB was not rechecked.
- **nvCOMP** (commit f0f982e1, 2026-07-13). Version 6.0.0 reports ANS compression up 50% and decompression up 30%. Version 2.5.1 reports GDeflate decompression up to 2 times. No absolute GB/s is given. CUDA only.
- **DirectStorage GDeflate** (commit 32f9411b). GDeflate is a reformatted DEFLATE stream with the same ratio, split into 32 sub-streams. It is not Deflate-compatible. Windows API.
- **zstd** (README at commit c9d3af9e). Level -1 decompresses at 1550 MB/s on Silesia, on an i7-9700K at 4.9 GHz. The README does not state whether this is single-thread. The same table lists lz4 1.10.0 at 3850 MB/s.
- **lz4** (README at commit f42fdc17, 2026-10-18). Single-thread mode. Version 1.9.0 gives 4970 MB/s, and an old zstd 1.4.0 level -1 row gives 1380 MB/s. This disagrees with the zstd README on lz4 speed. Neither is a Breccia number.
- **rANS** (reference repository README at commit ecb7935b). Scalar byte rANS runs at 137.5 to 147.1 MiB/s, interleaved at 227 to 236 MB/s. The 64-bit variant runs at 165 to 175 MiB/s, and interleaved at 288 to 306 MB/s. No clock is stated.
- **FSE and Huff0** (README at commit 4cf72b44, Core i7-5600U). Huff0 decompression runs at 1350 MB/s (Proba80), 860 (Proba14) and 555 (Proba02). Threading is not stated.
- **zswap** (v6.18 tree, commit 7d0a66e4). `zswap_store` is at `mm/zswap.c:1495`, load at `:1601`, and invalidate at `:1661`. Its caller is `swap_writeout` in `mm/page_io.c` (defined at `:240`, called at `:275`). It stores anonymous swap-out pages only.
- **zram** (v6.18 tree; SHA not recorded this round). The huge-object threshold is at `drivers/block/zram/zram_drv.c:1871`, with `zs_malloc` at `:1813`.
- **memory.zswap.max** (v6.18 tree). `Documentation/admin-guide/cgroup-v2.rst:1881–1887`. It caps the zswap pool of a cgroup.
- **cleancache.** Removed by commit 0a4ee518185e902758191d968600399f3bc2be31 ("mm: remove cleancache", 2022-01-22, shipped in v5.17). `mm/cleancache.c` is absent at 7d0a66e4.
- **VK_EXT_memory_decompression** (Vulkan-Docs commit 560d4305e8a6466fa7d07c83ca09429e4d010ff9, tag v1.4.363). The local Vulkan registry lists only the GDEFLATE method for this extension. The RADV advertisement is UNVERIFIED. The mesa-26.2.4 tag exists on GitLab, but there is no SHA and the extension list was not located. The extension name and the decompression entry point appear in the installed RADV binary, but a string in a binary does not show that the driver advertises the extension.
- **Dropped.** A rANS throughput figure from arXiv 1402.3392 was not verified, and it is not used.

## 4. Arithmetic

**4.1 Cap and overflow (exact).** S = 34,589,900,800 B. C = 32,212,254,720 B. The overflow is S − C = 2,377,646,080 B (2.21 GiB), which needs a ratio of C/S = 0.931262, a saving of 6.874%. A tier at ratio r removes (1 − r) × S of that overflow. Measured removal: 808,649,734 B (0.75 GiB) at Huffman, 1,198,121,892 B (1.12 GiB) at ideal, and 1,648,236,078 B (1.54 GiB) at the d bound.

**4.2 Compressed resident tier against the page cache.** Assumption: the whole cap holds expert bytes. This is an upper bound on both columns. PLE pages and the host headroom in doc 35 §9 lower both.

| Tier | Ratio r | Fraction of expert bytes resident, C/(rS) | Miss bytes per stream-token | Saved against 46.44 MB | Hit bytes decoded per stream-token |
|---|---|---|---|---|---|
| Page cache, no tier | 1.00000 | 0.93126 | 46.44 MB | 0 | 0 |
| Huffman | 0.97662 | 0.95356 | 31.38 MB | 15.06 MB (32.4% of misses) | 644 MB |
| Ideal order-0 | 0.96536 | 0.96468 | 23.86 MB | 22.58 MB (48.6%) | 652 MB |
| Ideal plus d bound | 0.95235 | 0.97786 | 14.96 MB | 31.48 MB (67.8%) | 661 MB |

Miss bytes are derived under uniform routing: 10 picks per block, 390 streamed picks per stream-token, and the streamed fraction of 0.068738 (doc 35 §3). Per stream-token, lane bytes are 675,584,000 B.

**4.3 Decode time per stream-token.**

- Hit bytes of 644 to 661 MB, at the zstd proxy speed of 2.09 GB/s, take 0.31 s single core per stream-token. The lane dequantizes its 675.6 MB of lane bytes per stream-token at 2.5 to 3.8 GB/s single core, which takes 0.18 to 0.27 s (doc 03). Decode would be added to dequant, not swapped for it.
- Refault time removed at the 5.86 GB/s ceiling is 2.6 ms (Huffman), 3.9 ms (ideal) and 5.4 ms (d bound) per stream-token. This is device time at the ceiling, not a measured refault cost.
- Decode bytes per saved byte, the break-even ratio, is 42.8 (Huffman), 28.9 (ideal) and 21.0 (d bound). For decode to pay, its throughput has to exceed that ratio times the refault read rate. At the derived uniform pair miss rate of 0.94 GB/s (20.2 aggregate tok/s), break-even is 40 GB/s for Huffman, 27 GB/s for ideal and 20 GB/s for the d bound. The 0.94 GB/s rate is a lenient stand-in: it is the average miss rate over wall time, so it treats all wall time as fault stall, which favours the tier. Card A measures the real figure. Hits per saved byte equal 1/(1-r) when the compressed bytes are fully resident, so the ratio does not depend on how much of the cap is compressed, and a partial or hybrid tier faces the same break-even.
- Sustained demand: 2 × 644 MB × 10.1 pair-tokens/s = 13.0 GB/s for the pair, which is about 6.2 single-core equivalents at 2.09 GB/s.

**4.4 Headroom.** With 1 GiB held back for the host, the required ratio is 0.900220 (9.98% saving). With 2 GiB, it is 0.869178 (13.08%). No tier in the §4.2 table reaches either.

## 5. Ranked flag-OFF cards

Card letters are local to this doc. Gain is not claimed on any card. Every card names both gate GPUs, RX 6950 XT (s1) and MI50 (s0), and every card uses MemoryMax=30G (32,212,254,720 B) for its cap. Warm-up is the rule in doc 36 §6: it ends when the resident page count of the streamed expert ranges (a `mincore` sweep) has not grown for 20 consecutive tokens, and the gate counts 200 tokens from that point. The 20-token plateau is a choice, not measured.

**A. Refault baseline for the model shards under MemoryMax=30G (measurement only, no flag).**

Change: none to code. Run the current build at the 9-block default, with flags OFF, and sample the expert `workingset_refault_file` and `pgmajfault` counters over 200 tokens after warm-up. Also read the device's read-byte counters over the same window. Repeat three times, interleaved with `bench token --streams 2`, to measure run-to-run noise.

Baseline: the run itself. Primary metric: the expert `workingset_refault_file` and `pgmajfault` deltas over 200 tokens after warm-up, converted to bytes per stream-token as R = pages × 4096 B. Secondary: aggregate tok/s, reported and not gated.

Decision rule: a tier can remove at most s × R of the expert refault bytes per stream-token, where s is the share of misses it removes under uniform routing: 0.324 (Huffman), 0.486 (ideal) or 0.678 (d bound). The time that removal saves per stream-token is s × R divided by the measured read rate. Close the lossless question without card B if R is zero across the three repeats, or if that saved time is below the run-to-run spread of per-stream-token time across the repeats. Otherwise run card B.

Gate: R, the read rate and the noise are recorded on RX 6950 XT (s1) and MI50 (s0) in the two-stream run. Cap: MemoryMax=30G (32,212,254,720 B). Parity: not applicable, no code changed.

Gain: UNVERIFIED - requires the GPU qualification card

**B. Bench-only decode test for a Huffman-split expert tier (flag OFF `BRECCIA_LOSSLESS_BENCH`, proposed name).**

Change, flag ON only: a bench command that encodes the streamed expert bytes of a sample of blocks into a per-field Huffman tier, then times decoding of hit-sized batches on 1 thread and on the engine's thread count. The engine path is not changed.

Baseline: the card A run, which supplies R and the read rate B_n.

Kill test: pass only if the aggregate decode throughput, measured on the built tier, exceeds (hits/saved) × B_n, where hits and saved are recomputed from the measured ratio. At the derived uniform rate of 0.94 GB/s this is 20 to 40 GB/s, a lenient stand-in for the rate card A measures. Fail if the measured throughput is below that, or if the 13.0 GB/s sustained demand (§4.3) is more than the host can spare beside the lane.

Gate: refaults (primary): not affected, since the engine path is unchanged; card A's R and B_n are the reference. tok/s (secondary): not a gate here, because no engine run is made. The GPU condition is the card A run on RX 6950 XT (s1) and MI50 (s0). Cap: MemoryMax=30G (32,212,254,720 B), with the encoded tier counted. Parity: every sampled expert tensor must round-trip bit-exact.

Gain: UNVERIFIED - requires the GPU qualification card

**C. Conditional lossless resident tier (flag OFF `BRECCIA_LOSSLESS_EXPERT_TIER`, proposed name).**

Change, flag ON only: the resident path stores the streamed expert bytes Huffman-split and decodes them into scratch on each hit. Lossless only. Built only if card B passes.

Baseline: the flag-OFF run with the 9-block default under MemoryMax=30G (card A).

Gate: refaults (primary): expert `workingset_refault_file` and `pgmajfault` deltas at least 15% below baseline over 200 tokens after warm-up. The 15% is a choice, set below the derived 32.4% Huffman removal. tok/s (secondary): not more than 2% below baseline. Both are choices, not derived. Measured on RX 6950 XT (s1) and MI50 (s0). Cap: MemoryMax=30G (32,212,254,720 B), with the tier's resident bytes counted.

Parity: outputs bit-identical to flag-OFF over 64 tokens. This follows the house precedent in doc 32 card 6, which gates a host-to-GPU expert move on 64-token bit-identity.

Gain: UNVERIFIED - requires the GPU qualification card

## 6. Rejected list

- **Doc 24 2-bit shadows.** Lossy by design, since the shadow is an approximation of the expert weights. Doc 24 was not re-read this round.
- **Requantization to a smaller type.** Lossy.
- **NeuZip lossy mode.** Lossy. Its lossless mode was not evaluated on block-quantized IQ types, so it is not carried here.
- **zswap on expert pages.** zswap stores anonymous swap-out pages only (`mm/page_io.c:275`). Expert bytes are file-backed mapped pages, so zswap never sees them.
- **zram.** Holds a compressed anonymous copy. The expert bytes would sit in the page cache and in zram at once until the page cache lets go of them. This was not measured.
- **cleancache.** Removed in v5.17.
- **nvCOMP.** CUDA only. Neither the RX 6950 XT nor the MI50 has a CUDA path.
- **DirectStorage GDeflate.** Windows API.
- **Vulkan GDEFLATE through VK_EXT_memory_decompression.** The spec defines only the GDEFLATE method. RADV advertisement is UNVERIFIED, and decode throughput was not measured.
- **GPU Huffman or ANS decoder.** Not carded this round. A grep of the crates for Huffman and ANS decoders found none, and the 21 to 43 times break-even applies to it as well. Card B measures the host decoder first.
- **Q2_0-only lossless.** Saves 0.70% (Huffman) to 1.98% (d bound) of the streamed set. It cannot close the 6.87% gap alone.
- **Doc 35 option C2 (residency +4 blocks).** Not rejected, and not re-derived here. It changes which bytes are resident rather than how they are stored, so it needs no decode. Its VRAM cost is unmeasured here.

## 7. Unverified

- Higher-order context models and joint sub-byte modelling are not measured. Every ratio here is order-0.
- The IQ2 and IQ3_S rows use one expert's two tensors per type, so the split entropies could move with more experts.
- The 16-bit plug-in entropy of the d fields is biased low on a small sample.
- Huffman figures use integer code lengths and leave out table and bit-packing cost. A real Huffman tier would be slightly worse than the figure.
- zstd and lz4 speeds come from one run each, under a load average near 8.
- The Mesa RADV advertisement of VK_EXT_memory_decompression.
- The quants behind the ZipNN GGUF statement.
- The zram and memory.zswap.max citations are pinned to the v6.18 tree, not to a commit SHA.
- The actual refault read rate under MemoryMax=30G. Card A measures it.
- Routing skew. All miss figures assume uniform routing, which is a bound.
- The literature speeds were measured on other hardware and are not comparable to Breccia.

## 8. Conflicts

1. **Brief and model name against the GGUF types.** The brief framed the gate and up roles as IQ3_XXS. The shards use five types across those roles (§1). The IQ3_XXS tensors are a minority, in 6 blocks. The brief's framing is not used.
2. **Q2_0 Huffman loss.** Huffman gives 0.9572 and the ideal gives 0.9134. The average Huffman code length is 1.904 bits against 1.805 bits of entropy. Integer code lengths cost about 0.04 of the ratio on this stream.
3. **zstd beats order-0 on the small streams.** zstd gives 0.8348 on IQ3_XXS scales, 0.7654 on IQ3_XXS d, and 0.8089 on Q2_0 d. The order-0 ideal leaves d uncoded, so the d-coding bound is the fair comparison for these streams.
4. **Earlier generic byte bounds are superseded.** The earlier byte-level bounds for IQ2_XXS, IQ2_XS, IQ2_S and IQ3_S (0.9971 to 0.9983) were upper bounds on the split-stream ratio. They overstated the compressed size, and the split-class entropies replace them. The earlier weighted figure of 0.97630 and its "cannot reach the cap even at ideal" wording are withdrawn. The corrected ideal is 0.96536.
5. **lz4 speed.** The zstd README and the lz4 README give different lz4 figures (§3), and neither is measured on this host. The sampler's lz4 ratio on the IQ3_XXS grid is recorded as about 1.00 in one place and 0.996 in another; lz4 gives no useful gain on that stream either way.
6. **ZipNN against measured ratios.** ZipNN's "do not compress" statement, for the quants it tested, is consistent with the measured grid and sign streams, which barely compress. It is not a measurement of IQ3_XXS.

## 9. Arithmetic appendix

- **Cap.** C = 30 × 2^30 = 32,212,254,720 B. S = 34,589,900,800 B. S − C = 2,377,646,080 B = 2.2144 GiB. C/S = 0.9312618. Saving needed = 1 − C/S = 6.8738%.
- **Overflow after a tier.** Remaining overflow = r × S − C. Huffman: 0.97662 × S − C = 1,568,996,346 B = 1.461 GiB. Ideal: 1,179,524,188 B = 1.099 GiB. d bound: 729,410,002 B = 0.679 GiB.
- **Headroom.** Required ratio with H bytes held back = (C − H)/S. H = 2^30 gives 0.900220. H = 2 × 2^30 gives 0.869178.
- **Per-block field arithmetic.** IQ4_NL: 128 nibble bits and 16 d bits per block, 144 total. Ideal = (128 × 0.97775 + 16) / 144 = 0.9802 with rounded inputs (0.9801 unrounded, §2). With d coded at 11.38 bits, the bound = (125.15 + 11.38) / 144 = 0.9481. Q2_0: 128 code bits and 16 d bits per block. Ideal = (128 × 0.9025 + 16) / 144 = 0.9133 with rounded inputs (0.9134 unrounded, §2). With d coded at 11.05 bits, the bound = (115.52 + 11.05) / 144 = 0.8790. Huffman: (128 × 0.952 + 16) / 144 = 0.9573 with rounded inputs (0.9572 unrounded, §2).
- **Miss bytes, uniform routing.** Lane bytes per stream-token L = 10 × S / 512 = 675,584,000 B. Miss fraction without a tier = 1 − C/S = 0.068738, so misses = 46,440,000 B (46.44 MB). This matches doc 35 §9.
- **Resident fraction with a tier.** f = C / (r × S). Misses with the tier = (1 − f) × L.
- **Saved bytes.** Saved = 46.44 MB − (1 − f) × L. Huffman 15.06 MB; ideal 22.58 MB; d bound 31.48 MB.
- **Hit bytes and decode time.** Hits = f × L: 644.2, 651.7 and 660.6 MB. At 2.09 GB/s single core: 0.308, 0.312 and 0.316 s per stream-token.
- **Break-even.** Decode bytes per saved byte = hits / saved = 42.8, 28.9 and 21.0. Break-even decode throughput = (hits / saved) × B_n. With B_n = 0.938 GB/s (the pair rate of 10.1 pair-tokens/s times 2 times 46.44 MB, derived), that is 40.1, 27.1 and 19.7 GB/s. The 0.938 GB/s stand-in is lenient (§4.3); card A measures the real figure.
- **Sustained demand.** 2 × hits × 10.1 pair-tokens/s = 13.0 to 13.3 GB/s, which is 6.2 to 6.4 single-core equivalents at 2.09 GB/s.
- **Refault time at the ceiling.** saved / 5.86 GB/s = 2.57 ms (Huffman), 3.85 ms (ideal), 5.37 ms (d bound) per stream-token.
- **Lane comparison (doc 03).** 675,584,000 B / 3.8 GB/s = 0.178 s; / 2.5 GB/s = 0.270 s per stream-token single core.
- **Card A decision bound.** f × R with f = 0.324, 0.486 and 0.678 under uniform routing. R is measured in card A, so the decision uses R, not the uniform 46.44 MB.

## 10. Hygiene

- **Model reads.** The three sampler runs read 48,695,168 B in total (46.44 MiB), derived from the counters in the run logs: 27,805,568 B for the header stage (run 1, kept in the header cache) and 10,444,800 B of sample chunks in each of runs 2 and 3. The total is under the 64 MiB cap (67,108,864 B).
- **How the reads were done.** The run logs do not record the CPU or I/O priority the sampler ran at, so none is claimed here. The sampler and the split script do not mmap the model shards.
- **No GPU process and no model load.** No GPU process was started and the model was not loaded.
- **No code changed.** No file under `crates/` was touched. Scratch files are in an untracked scratch directory, and nothing was committed.
- **Final grep.** Hostnames, addresses and absolute paths are checked on this doc and the README diff; the result is in the round report.
