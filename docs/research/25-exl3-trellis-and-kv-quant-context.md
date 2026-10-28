# 25 — EXL3/QTIP trellis weights and KV/state quantization

Round 25, 2026-10-28. Scope: quantization of expert and dense weights (EXL3 / QTIP trellis) and of context state (QSA KV, GDN recurrent state), mapped onto the shipped Breccia engine. Standing rules: no GPU work, no model loads, no benchmarks, no commits. Every speed or capacity number is either measured in the repo (cited to a doc or file) or derived here and marked **derived**. Anything marked **UNVERIFIED** is not confirmed by a measurement or a primary source in this round. Citations: "doc NN" means `docs/research/NN-*.md`, except "doc 02" (`docs/reference-engines/02-kv-and-context.md`) and "phase-4 doc 01" (`docs/phase-4/01-forward-pass.md`).

## 0. Verdict

Three cards, each with a gate in §4.

1. **Card A — K as q8 for the 12 QSA layers.** The repo already has the V q8 layout (36 B per 32 elements). Same layout for K saves 448 B per layer-token, 5,376 B per token, **168 MiB per 32K stream**. Gate: KL and top-1 agreement at positions beyond 2,048 (top-k active), plus the existing +0.3% PPL anchor. The Hadamard option has precedents in QuaRot and exllamav3 (§8).
2. **Card B — GDN recurrent state stored as f16.** Saves **54 MiB per stream** (36 layers × 1.5 MiB). The state is recurrent and the paper gives no precision bound (§2.3); the gate is a long horizon, not a single window.
3. **Card C — offline trellis requant of cold experts below 2.125 bpw**, as a sidecar beside the doc 24 S1 shadow. A uniform format saves bytes against S1 only below 2.125 bpw, and against the shipped expert mix only below 2.84 bpw. Gate: per-expert output error on the CPU path, then full-model KL against the S1 baseline at equal bytes.

Headline: **EXL3 at 3 bpw is not a byte lever** (5.6% more expert bytes than the shipped mix). Its speed case is open and leans toward EXL3 on paper: an ALU-only codebook at 3–4 instructions per weight puts the MI50 at 1,418–1,733 M el/ms against IQ3_XXS's 433, and makes the RX DRAM-bound (§1.2). The decode kernel read is the top follow-up. KV/state is small next to the arena. Each GiB of KV displaces about 1.1 arena blocks, about 0.8 ms per token (§2.6, UNVERIFIED). The owner's context and stream target decides which trade is worth making (§7).

## 1. Question 1 — EXL3 / QTIP trellis for weights

### 1.1 The bound rule, and a conflict in the inherited constants

BRIEF §5 and doc 12 §A2 state the crossover as `I_unit > 8.0·bpw` (MI50) and `I_unit > 53·bpw` (RX). Re-deriving from doc 12's own rates (ALU element rate = 8K/I_unit, DRAM element rate = 8·BW/bpw) gives `I_unit > (K/BW)·bpw`:

| Card | Rate used | K/BW | Threshold per bpw | Doc 12 / brief constant |
|---|---|---|---|---|
| MI50 | K 7.8e12, BW 825 GB/s | 9.45 | 9.45 | 8.0 |
| RX | K 1.59e13, BW 576 GB/s (nominal) | 27.6 | 27.6 | 53 |
| RX | K 1.59e13, BW 445 GB/s (achievable, doc 12 §A2) | 35.7 | 35.7 | 53 |

The brief's constants do not follow from doc 12's stated rates: the stated formula applied to K 7.8e12 and 825 GB/s gives 9.45·bpw for MI50, not 8.0·bpw; 8.0 needs K = 6.6e12, and 53 needs K ≈ 2.4e13 at 445 GB/s. The Q5_K head (I_unit 202, 630 M el/ms on the RX) sits on both laws: the stated RX ALU law (127,000 / I_unit) gives 629, and the 445 GB/s DRAM ceiling for Q5_K gives 647. It cannot separate them. **The RX threshold is UNVERIFIED and needs a re-measurement before any RX verdict rests on it.** The byte verdict in §1.5 does not depend on K or on the RX bound.

Element rates in M el/ms, from doc 12's laws (MI50 62,400/I_unit; RX 127,000/I_unit; DRAM = BW·8/bpw):

| Type | bpw | I_unit | MI50 ALU | MI50 DRAM (825) | RX ALU | RX DRAM (445) |
|---|---|---|---|---|---|---|
| IQ3_XXS (doc 12 §A2 table, derived I_unit) | 3.0625 | 144 | 433 | 2,155 | 882 | 1,162 |
| EXL3 3 bpw, ALU codebook at 3 instr/weight (§1.2) | 3.0 | ≈36 | 1,733 | 2,200 | 3,528 | 1,187 |
| EXL3 3 bpw, ALU codebook at 4 instr/weight (§1.2) | 3.0 | ≈44 | 1,418 | 2,200 | 2,886 | 1,187 |

Under doc 12's constants, IQ3_XXS is ALU-bound on both cards (RX 882 < 1,162; MI50 433 < 2,155). The EXL3 rows use I_unit = 8c + 12 for an ALU-only decode (§1.2), not a kernel count. Under the brief's 53·bpw, the RX is DRAM-bound for IQ3_XXS (I_unit 144 < 162). The verdict in §1.5 does not depend on any of these.

### 1.2 EXL3 per-element ALU cost

QTIP's codebooks are cheap per weight: at most 4 hardware instructions per weight on NVIDIA GPUs (3INST is 3 ALU instructions), and as few as 2 (QTIP §3.1–3.1.1). Its bitshift state, j = (i·2^kV mod 2^L) + c, reads each element at a bit offset without a chain (derived from the formula). The reference code gathers from a precomputed 2^L table (`lib/codebook/bitshift.py` L114–122, L200–201), a lookup per element. The text read does not say whether EXL3 uses 1MAD, 3INST or a table, and the decode kernel was not read.

Units: doc 12's I_unit counts instructions per 8-element unit (doc 12 L117–121), so per element the cost is I_unit/8. With c decode instructions per element, doc 12's dot share of about 12 per unit, and s for window extract and scale (unknown): I_unit = 8c + 12 + s. On the MI50 at 3 bpw the threshold is 28.4, so DRAM-bound needs I_unit below 28.4, that is c = 2 with no extract or scale work (I_unit 28 gives 2,229 M el/ms, just above the 2,200 ceiling). c = 3 gives I_unit 36 (1,733 M el/ms) and c = 4 gives 44 (1,418 M el/ms). The MI50 stays ALU-bound in both, at 3.3–4.0× IQ3_XXS's 433 on paper. On the RX the stated law gives 3,528 and 2,886 against a DRAM ceiling of 1,187 at 445 GB/s, so the RX is DRAM-bound at about 1,190, against IQ3_XXS's 882 (its threshold of about 83–107 is UNVERIFIED, §1.1). A table gather at one lookup per element (8 per unit) falls in doc 12's lookup penalty (IQ4_XS: 8 lookups per unit, 47% efficiency, measured), which would roughly halve these paper rates (UNVERIFIED for EXL3). Integer multiply rates on gfx906 and gfx1030 are not in doc 12 (UNVERIFIED).

### 1.3 GLSL feasibility without cooperative matrix

Cooperative matrix is absent on both cards (doc 09 §8 item 8; doc 22 Q6). Batch-1 decode is a GEMV, so an MMA path is not needed. A trellis dequant-in-shader follows the existing IQ3_S pattern (doc 12 notes its 2 KiB shared grid): a uint buffer, bit-window extracts, one multiply-add per element. Feasible in GLSL on paper; no driver-level test was run this round. EXL3 also applies a 128-point Hadamard with sign vectors to the input and output vectors at run time (`exllamav3/modules/quant/exl3.py` L176–200). That is O(n+m) per token, not per weight, so §1.2 is unchanged, but it is one more shader pass.

### 1.4 Bits per weight vs quality

Not measured in the repo. The repo's baseline is IQ3_XXS at 3.0625 bpw. The S1 2-bit shadow (IQ2_XXS gate/up + Q2_0 down, doc 24) has no quality result in the tree (doc 24 line 96). The exllamav3 docs name only two bpw targets (3.75 and 4.00) and give no quality curve; QTIP's quality-versus-bpw results were not extracted (§8). The EXL3 bpw question below 3 is open.

### 1.5 Expert verdict by type mix

Expert geometry: gate and up are 640×2560 and down is 2560×640, so 4,915,200 elements per expert. Bytes per expert:

| Format | bpw | Bytes per expert | vs shipped mix |
|---|---|---|---|
| Shipped mix, weighted over 48 layers (derived below) | 2.84 | 1,746,133 | 0% |
| S1: IQ2_XXS gate/up + Q2_0 down (doc 24) | 2.125 | 1,305,600 | −25.2% |
| IQ3_XXS uniform | 3.0625 | 1,881,600 | +7.8% |
| EXL3 uniform 3.0 | 3.0 | 1,843,200 | +5.6% |
| EXL3 uniform 2.5 | 2.5 | 1,536,000 | −12.0% |
| EXL3 uniform 2.0 | 2.0 | 1,228,800 | −29.6% |

Verdict: **EXL3 at 3 bpw costs 5.6% more expert bytes than the shipped mix, so it cannot move the expert-byte bound.** The byte lever is below 2.84 bpw uniform. The shipped mix comes from the per-layer types: gate/up (taken as one type per layer) IQ2_XXS ×9, IQ2_XS ×10, IQ3_XXS ×6, IQ3_S ×13, IQ2_S ×10; down Q2_0 ×30, IQ4_NL ×18. Class sizes are in `crates/breccia-format/src/quant.rs` lines 158–177. Gate/up sums to 53,401,600 B and down to 30,412,800 B over 48 layers, so 83,814,400 / 48 = 1,746,133 B per expert. Two cross-checks hold: that matches doc 24 line 30 exactly (83,814,400 B over 48 layers, 1.6652 MiB per use), and the BRIEF's 301 MiB over 180 lane uses implies 1.672 MiB, 0.4% higher, and S1's 1,305,600 B sits 25.2% below it, matching doc 24's 25.2% cut. Expert bytes per block: 512 × 1,746,133 B ≈ 852 MiB, against the 930 MiB arena cost in §2.5.

### 1.6 Where decode runs, and the offline sidecar

Streamed experts are computed on the CPU in `moe_topk` (`crates/breccia-engine/src/graph.rs:3137`, inside `finish_block`; doc 24). So a trellis format for streamed experts needs a CPU decoder. `grep` of `crates/` and `docs/` finds no trellis or EXL3 decoder in the tree. Resident experts would use GLSL (§1.3).

A requant is an offline sidecar file. It reads the expert tensors, writes a new file, and leaves the model GGUF untouched. Allocation can follow doc 03's measure-then-allocate method (per-tensor bpw by measured error; doc 03 line 306 says EXL3 changes only the codebook; §6 item 7 corrects this). Note that ik_llama.cpp's IQK trellis quants are already in the covered list (docs/reference-engines.md line 19). This round adds the byte arithmetic and the S1 comparison, not a second audit of that engine.

## 2. Question 2 — KV and state capacity

### 2.1 Bytes per token (12 QSA layers)

Per layer per token, from `crates/breccia-kernels/src/qsa/mod.rs` (`sizes`) and `qsa/cpu.rs:55-70`:

| Component | Bytes per layer-token | Notes |
|---|---|---|
| K f16 | 1,024 | 2 KV heads × 256 × 2 B |
| K q8 (repo layout) | 576 | 16 blocks × 36 B (32 i8 + f32 scale) |
| V f16 | 1,024 | |
| V q8 (repo layout) | 576 | GGUF Q8_0 is 34 B per 32; repo uses 36 B |
| Indexer key (pooled, f32) | 128 | per cell per layer |

Per token, all 12 layers plus indexer (1,536 B total):

| Layout | KV bytes/token | Incl. indexer | Context at 32K (MiB, all layers) |
|---|---|---|---|
| f16 K + f16 V | 24,576 | 26,112 | 816 |
| K f16 + V q8 (**shipped**) | 19,200 | 20,736 | 648 |
| K q8 + V q8 | 13,824 | 15,360 | 480 |

Measured per-agent totals at 32K are 933 MiB (f16) and 764 MiB (shipped). Against the KV totals above (816 and 648 MiB), that leaves 117 and 116 MiB for GDN and workspace (§2.3). Phase-4 §8bf's 728 MiB for K+V q8 is mis-derived (§6).

### 2.2 The indexer is insulated from K quantization

The indexer pools its own raw keys: `RAW [T, idx_dim]` with idx_dim 128, a separate projection from head_dim 256 (`crates/breccia-kernels/shaders/qsa_idx_pool.comp`, bindings 0–4). Pooled keys are `imrope(rmsnorm(mean(raw[4b..4b+3])))`, one per 4-cell block. Top-k selects 512 blocks, so 2,048 cells per query (`qsa/mod.rs:81-95`). Changing the main K cache therefore cannot change block selection. Indexer keys are read for every query over every pooled block (`qsa_idx_score.comp`), so they stay resident (§3.3).

### 2.3 GDN state per stream

Per GDN layer: state 3 MiB (128 × 48 × 128 f32) plus conv history 120 KiB (3 × 10,240 × 4 B). That is 3,268,608 B per layer. 36 layers give 117,669,888 B = **112.2 MiB per stream**, independent of context (`placement.rs:307-315`, `gdn/mod.rs:79-97`).

Per-token GDN traffic: 2 reads + 1 write of 3 MiB per layer = 339.7 MB per token. At the measured 2,768 µs that is 122.7 GB/s, 14.9% of 825 GB/s (BRIEF states 119 GB/s and 14%; minor difference). The one-pass floor (1R + 1W) is 226.5 MB, 275 µs at 825 GB/s.

**f16 state, derived:** 36 × 1.5 MiB = **54 MiB saved per stream**, and about half the GDN bytes per token (169.9 MB). Whether f16 is numerically safe for a recurrent state is **UNVERIFIED**. The update is a gated delta rule (arXiv 2412.06464, Eq. 10): a decay gate damps old error and a delta term corrects it, so drift need not compound, but the paper gives no precision bound, and fla keeps the state in fp32 on both kernel paths (§8). The gate in §4 B is therefore a long-horizon check, not an assumption.

### 2.4 Per-card placement under the explicit split

Under `--split-blocks 28,20`, placement assigns contiguous block ranges by index (`placement.rs` L96–118):

| Device (ordinal from split order) | Blocks | QSA layers | GDN layers | KV bytes/token (shipped incl. idx) | GDN per stream |
|---|---|---|---|---|---|
| Device 0 | 0–27 | 3,7,11,15,19,23,27 (7) | 21 | 12,096 | 65.5 MiB |
| Device 1 | 28–47 | 31,35,39,43,47 (5) | 15 | 8,640 | 46.8 MiB |

Card names are inferred, not loaded. The canonical command lists `0000:03:00.0` (RX) first (phase-4 doc 01 L3018, L3818). The L3840 arm lists the MI50 first and reports it as ordinal 0; its head is pinned by `--head-device 0`, so the head does not prove the mapping, but the arm does. `placement.rs` L96–118 assigns `counts[i]` to the i-th device. So the RX/MI50 labels in §2.5 and §3 take device 0 = RX (UNVERIFIED at load). Confirm by PCI address at load, never by ordinal (AGENTS.md GPU rule). The auto arena takes the byte-balanced path (`placement.rs` ~181–226), so these per-card QSA counts apply to the explicit split only.

Headroom after the auto arena (measured, phase-4 doc 01 §8av): **RX about 304 MiB, MI50 about 8 MiB**. Arena loads in `Model::load` (`model.rs:672`); `Session::new` (`graph.rs:636-675`) allocates GDN and QSA state afterwards. Default session capacity is 256 cells (`graph.rs:638-641`), so any real context displaces arena blocks.

### 2.5 Maximum context per stream, per card

The RX and MI50 labels take device 0 = RX from §2.4 (inferred, not loaded). Per-card context at S streams with k arena blocks given up on that card (930 MiB each, phase-4 doc 01 L2676–2692; real block sizes are 637–1,138 MiB by type, so read the table ±20%). N is the context per stream. The RX-to-MI50 headroom and GDN are per card; the stream limit is the smaller of the two cards.

| Card | Layout | S | k = 0 | k = 1 | k = 2 |
|---|---|---|---|---|---|
| RX | shipped (K f16, V q8) | 2 | 7,501 | 47,811 | 88,121 |
| RX | shipped | 4 | 913 | 21,068 | 41,223 |
| RX | shipped | 6 | 0 | 12,154 | 25,590 |
| MI50 | shipped | 2 | 0 | 51,244 | 107,678 |
| MI50 | shipped | 4 | 0 | 22,784 | 51,001 |
| MI50 | shipped | 6 | 0 | 13,298 | 32,109 |
| RX | K q8 + V q8 | 2 | 10,127 | 64,545 | 118,964 |
| RX | K q8 + V q8 | 4 | 1,233 | 28,442 | 55,651 |
| RX | K q8 + V q8 | 6 | 0 | 16,408 | 34,547 |
| MI50 | K q8 + V q8 | 2 | 0 | 69,180 | 145,365 |
| MI50 | K q8 + V q8 | 4 | 0 | 30,759 | 68,852 |
| MI50 | K q8 + V q8 | 6 | 0 | 17,952 | 43,348 |

Arithmetic: N_max = (budget_card/S − GDN_card) / (KV_bytes_per_token_card), with budget_card = headroom + k × 930 MiB. 0 means the GDN state alone does not fit. The table is derived, not measured. The session capacity must be set to N for these numbers to apply.

Card A saves 98 MiB per 32K stream on RX (7 × 448 × 32,768 B) and 70 MiB on MI50 (5 × 448 × 32,768 B), 168 MiB in total.

### 2.6 Exchange rate, UNVERIFIED

One streamed block costs 16.7 MiB of lane traffic per token (10 experts × 1.6652 MiB per expert, doc 24 line 30). The BRIEF gives 13.5 ms for 301 MiB per token (doc 24 line 20), and 23.4 GB/s (doc 24 line 30). Derived: 0.75 ms per token per block. One GiB of KV displaces about 1.1 blocks, so **about 0.8 ms per token per GiB, 1.5% of the 53.78 ms token**. This is a linear extrapolation and **UNVERIFIED**; lane time is not strictly proportional to bytes once the cache policy changes.

## 3. Question 3 — Tiered context cache: spilling quantized pages to host RAM

### 3.1 Page geometry and transfer time

`PAGE_CELLS = 16`. Page size across 12 layers: f16 K+V 384 KiB, K f16 + V q8 300 KiB, both q8 216 KiB.

Worst case per token, all 2,048 selected cells miss, KV only (indexer resident, §3.3):

| Layout | MiB per token | RX at 28.7 GB/s (pinned, doc 09 L58–59) | MI50 at 2.9 GB/s (brief; conflicts doc 09 L12) |
|---|---|---|---|
| f16 K + f16 V | 48.0 | 1.75 ms | 17.4 ms |
| K f16 + V q8 | 37.5 | 1.37 ms | 13.6 ms |
| K q8 + V q8 | 27.0 | 0.99 ms | 9.8 ms |

The table is the all-12-layer upper bound. Spill is RX-only (§3.3), so the RX-hosted layers (7 of 12, §2.4) carry 7/12 of each figure: K f16 + V q8 is 21.9 MiB per stream per token, 0.80 ms at 28.7 GB/s, and scales with S. Against a 53.8 ms token, that RX figure is 1.5% if not overlapped. The all-layer MI50 worst case is 25% at the shipped layout (13.6 ms) and 32% at f16 (17.4 ms). These are transfer-only; overlap with compute is **UNVERIFIED**.

### 3.2 Granularity, 1× to 4×

Top-k selects 512 blocks of 4 cells, 2,048 cells per query. A 16-cell page holds up to four selected 4-cell blocks, so the transfer per token ranges from 1× (the selected cells are contiguous; the §3.1 table) to 4× (each selected 4-cell block sits in a different page: 512 pages × 300 KiB = 150 MiB per token for K f16 + V q8).

The fix is a 4-cell spill unit: a 4-cell unit costs the contiguous best case in every pattern (37.5 MiB per token for K f16 + V q8). How selection clusters across real decode is **UNVERIFIED** and needs a CPU-side trace, not a GPU run.

Below 2,048 cells, top-k selects every cell, so spill saves nothing. The spill tier only pays beyond that.

### 3.3 Rules

- **RX only.** The MI50's 5 QSA layers carry 5/12 of the 13.6 ms all-layer worst case, about 5.7 ms per token at 2.9 GB/s (K f16 + V q8), so they stay resident. The RX's 7 layers are the spill candidates.
- **Indexer keys stay resident.** 1,536 B per token, read for every pooled block on every query. Spilling them adds traffic on every token for a saving of about 1.5 KiB per token.
- **Pinned host memory competes with the mmap'd model.** Phase-4 doc 01 §8av records page-cache thrash, and doc 02 caps tier T2 at 6 GiB. Host RAM is shared with the mmap'd model, so pinned spill pages would displace its page cache. Not measured.
- **NVMe spill is out of scope.** The brief scopes host RAM; unmeasured.

## 4. Top three cards with gates

Gate thresholds use only numbers already in the tree. Where the tree has no number, the gate names the baseline to measure against and leaves the threshold open. No threshold is invented here.

**Card A — K q8 (repo layout) for the 12 QSA layers, optional Hadamard rotation of K before quantization.**
- Change: a K-q8 path beside the existing V-q8 pair (`crates/breccia-kernels/build.rs:249-266` KV_Q8V_SHADERS; `qsa/cpu.rs:55-70` to_q8_0). Hadamard on K is an option to measure: QuaRot rotates queries and keys (stage 1d) and exllamav3 applies H32 to K groups before quantization (§8). KIVI's fixed large-magnitude key channels (§8) share a 32-element block scale with their neighbours, so the rotation is the first lever to test (doc 02 §2.2 covers llama.cpp's rotation).
- Gain: 168 MiB per 32K stream (98 MiB on device 0 = RX, 70 MiB on device 1; §2.4). Capacity only, no speed claim.
- Gate: (i) PPL within +0.3% of the shipped K f16 + V q8 path on the 4K holdout (threshold from doc 12 line 337; baseline is the shipped layout, not the f32-activation path); (ii) KL to base logits on a held-out set **at positions beyond 2,048 so top-k is active**, using the perplexity tool path in doc 21 lines 79 and 100 (method only; no KL number is in the tree, so the owner sets it); (iii) top-1 agreement with f16 K over the same positions. Per-kernel drift on the CPU reference (`qsa/cpu.rs`) needs no GPU; the end-to-end gate does.

**Card B — GDN recurrent state stored as f16 (state only; conv history stays f32).**
- Gain: 54 MiB per stream (device 0 = RX 31.5 MiB, device 1 = MI50 22.5 MiB; §2.4). Also about half the GDN state traffic (339.7 → 169.9 MB per token; time effect UNVERIFIED).
- Gate: (i) the existing 32/32 greedy-token parity gate must still pass (BRIEF §4 and §9); (ii) a long-horizon check, at least 4K tokens, PPL within +0.3% of the f32-state path (the doc 12 line 337 threshold reused; baseline is the f32 state, not the f32 activation path); (iii) a per-layer state max-abs error against the f32 CPU reference (`gdn/cpu.rs`) at the end of the horizon. The decay gate damps old error, but the paper gives no precision bound, so (ii) and (iii) are the binding gates, not (i).

**Card C — offline trellis requant of cold experts below 2.125 bpw, as a sidecar beside doc 24 S1.**
- Gain: at 2.0 bpw, 1,228,800 B per expert: −5.9% against S1 and −29.6% against the shipped mix. Against S1 there is no byte gain above 2.125 bpw; against the shipped mix, above 2.84 bpw.
- Gate: (i) CPU-only per-expert relative output error on a recorded route trace (doc 24's route-trace policy), against the S1 expert error on the same trace; (ii) full-model KL to base logits at equal bytes, against the S1 baseline measured with the same harness (doc 24 has no 2-bit quality result, so this baseline must be measured first). Threshold: "beats S1 on KL at equal bytes"; no number is invented here.
- Cost: a CPU trellis decoder is absent from the tree and must be written before (i) can run.

## 5. Rejected and deferred ideas

- **EXL3 at 3 bpw as a byte lever:** 5.6% more expert bytes than the shipped mix (§1.5). Not rejected on speed: the ALU case favours EXL3 on paper and waits for the decode read (§1.2).
- **Cooperative-matrix or MMA EXL3 kernels:** cooperative matrix is absent on both cards; batch-1 decode is a GEMV.
- **KVQuant-style non-uniform LUT for KV:** quality is not the objection (nuq3 with 1% outliers is 5.17 vs 5.12 fp16 on Llama-2-7B, §8). The lookup cost is: doc 12 §A2 shows 47% efficiency with 8 lookups.
- **Unrotated q4 KV (q4_0 / iq4_nl) now:** a q4 K layout (18 B per 32 elements) adds about 108 MiB per 32K stream beyond Card A, and unrotated K is the sensitive side (llama.cpp PR #7412, §8). Deferred, not rejected: rotated K4/V4 is close to fp16 on QuaRot's models (§8). Revisit after Card A's Hadamard gate.
- **KIVI per-channel K with a full-precision residual window:** per-channel grouping targets fixed outlier channels (§8), and the residual window adds full-precision bytes (residual R = 128 in the KIVI setup, not re-checked here). Card A keeps the repo's blockwise layout and tests the Hadamard first.
- **V rotation:** deferred, not rejected on runtime cost. SpinQuant applies R2 to values and the out-projection (§8), and a fixed pair like that can be folded into weights offline (inference). The gain is unmeasured and it needs an attention-weight requant.
- **Indexer keys below f32/bf16:** 1,536 B per token in total; doc 02 says never below bf16.
- **NVMe spill tier:** out of brief scope; unmeasured.
- **MI50 host-page streaming:** 9.8–17.4 ms per token worst case at 2.9 GB/s.
- **Spilling below 2,048 cells:** top-k selects every cell; nothing to save.
- **IsoQuant / RotorQuant code:** no LICENSE (docs/reference-engines.md); algorithm reading only, already in the covered list.
- **TurboQuant for KV:** the paper's rotation is a QR-based random orthogonal matrix, not a Hadamard (§8); its KV design is in `docs/reference-engines/02-kv-and-context.md` §4, not re-audited here.

## 6. Conflicts with earlier docs

1. **Doc 12 §A2 bound constants** (8.0·bpw MI50, 53·bpw RX; BRIEF §5 repeats them) do not follow from doc 12's stated K and BW rates (§1.1). The head measurement (I_unit 202, 630 M el/ms) cannot separate them. Owner decision needed (§7).
2. **Phase-4 §8bf** (docs/phase-4/01-forward-pass.md, ~lines 4540–4564) claims K+V q8 at 32K is about 728 MiB ("0.78 × 933"). The 0.78 ratio is the K-f16/V-q8 pair ratio applied to the whole total. The correct figure is 596 MiB for K q8 + V q8 (480 KV+idx + 116 GDN and workspace).
3. **Doc 02 §11.1**: indexer 12 KiB per token (1,024 B per cell per layer) against code's 128 B per cell per layer, so 1.5 KiB per token. Its q8_0 row (13,056 B per token) uses GGUF's 34 B per 32; the repo uses 36 B.
4. **Doc 09 A4** has a unit error: "32K 786 MiB" should be 768 MiB (24,576 × 32,768). Its q8 rows assume 8.5 bpw for K and V.
5. **Doc 09 L12** says MI50 PCIe is gen3 x16; phase-4 doc 01 §8av and BRIEF say gen3 x4 at about 2.9 GB/s. Doc 09 A2.4 (L154) calls the 2.84 GB/s figure a single-queue harness artefact and estimates 9–13 GB/s (inference, not measured). Tier numbers for the MI50 in §3 depend on this (§7).
6. **State size constant:** `placement.rs` `state_bytes` charges 2,180 B per cell per QSA layer. The `sizes` module gives 2,048 + 128 = 2,176 B. The 4 B difference is unexplained. The f16 assumption is conservative for V q8.
7. **Doc 03 line 306** says EXL3 changes only the codebook. exllamav3 also applies a 128-block Hadamard with sign vectors to the weights, before quantization and at run time (`quantize.py` L1473–1487; `exl3.py` L176–200, VERIFIED from code), so doc 03 understates the format change.

## 7. Owner decisions needed

1. **Context target per stream and stream count (2, 4 or 6).** This sets how many arena blocks each card gives up (§2.5). Without it, neither the spill tier nor Card A's gain has a target.
2. **Blocks for KV trade.** One GiB of KV displaces about 1.1 streamed arena blocks (930 MiB each), about 0.8 ms per token or 1.5% of token time (UNVERIFIED linear rate, §2.6).
3. **MI50 link rate:** gen3 x4 (phase-4 doc 01, BRIEF) or gen3 x16 (doc 09 L12). This decides whether MI50 spill is ever viable.
4. **GPU measurements and a kernel read:** the RX bound (27.6–35.7·bpw from stated K and BW, or 53·bpw in the brief; the head measurement cannot separate them), plus EXL3's decode kernel: codebook form and instruction count per weight (§1.2). The kernel read is the top follow-up because it can move the speed verdict either way. Neither was done this round (GPU time is out of scope).
5. **KL threshold** for Cards A and C. The tree has the method (doc 21 lines 79, 100) but no number.
6. **2-bit baseline:** whether to measure S1 quality (doc 24 gap) before Card C.
7. **Parity constant:** 13.170510 (BRIEF) or 13.170561 (AGENTS.md).

## 8. External sources

Fetched text and raw source files, not the PDFs; check quotes against the PDFs before any public use. Source line numbers come from unpinned default branches, so pin commits before citing. VERIFIED = read; NOT STATED or NOT FOUND = absent from the text read.
- **QTIP (arXiv 2406.11235, §3.1–3.1.1).** Bitshift state j = (i·2^kV mod 2^L) + c. The codebooks are said to need at most 4 hardware instructions per weight on NVIDIA GPUs (3INST: 3 ALU instructions). Lookups per weight NOT STATED. The reference code gathers from a 2^L table (`lib/codebook/bitshift.py` L114–122, L200–201). Changes §1.2.
- **ExLlamaV3 EXL3 (exllamav3 repository).** "A streamlined variant of QTIP" (README); bpw examples 3.75 and 4.00 only (`doc/convert.md`); "a full description of the format is coming" (`doc/exl3.md`, tag v1.4.4). Weights get a 128-block Hadamard with sign vectors before quantization and at run time (`exllamav3/modules/quant/exl3_lib/quantize.py` L1473–1487; `exllamav3/modules/quant/exl3.py` L176–200). A `rocm/` tree has RDNA-named kernels, so "CUDA-only" is not supported. Tail-biting and the full bpw list NOT FOUND. Changes §1.3 and §6 item 7.
- **exllamav3 KV cache (`exllamav3/cache/quant.py` L30; `exllamav3_ext/cache/q_cache_kernels.cuh` L12, L97–98, L225–226).** K and V at 2–8 bits in groups of 32 with fp16 scales; each group is rotated by H32 and scaled by its absmax, on the same path for K and V. Where the inverse is applied is NOT STATED. Precedent for Card A's Hadamard option.
- **KIVI (arXiv 2402.02750) and KVQuant (arXiv 2401.18079).** KIVI: keys per channel, values per token, because keys have "some fixed channels" of large magnitude (§3.2); up to 2% accuracy drop (§4.2.2); GSM8K Llama-2-7B 13.50 to 12.74 at KIVI-2 (Table 3); 2.35–3.47× throughput on one A100 (§4.2.4). KVQuant: non-uniform k-means with per-vector 1% outliers (§3.1–3.4); Llama-2-7B WikiText-2 (Table 13) fp16 5.12, nuq4 5.13, nuq3 5.17; C4 and Table 18 NOT READ. Changes §4 Card A and §5.
- **QuaRot (arXiv 2404.00456) and SpinQuant (arXiv 2405.16406).** QuaRot applies an online head-wise Hadamard to queries and keys (stage 1d); its KV-only K4/V4 WikiText-2 PPL (Table 6) is 5.51 vs 5.47 (7B), 4.91 vs 4.88 (13B), 3.33 vs 3.32 (70B). SpinQuant learns R2, applied to values and the out-projection (§3.2); a fixed pair like that can be folded into weights offline (inference, not stated). SpinQuant KV-only results NOT CHECKED. Changes §5.
- **TurboQuant (arXiv 2504.19874, §3.1, §4.3), IsoQuant (arXiv 2603.28430), RotorQuant README.** TurboQuant's rotation is a random orthogonal matrix from QR of a Gaussian; Hadamard NOT STATED. LongBench, Llama-3.1-8B: full 50.06, KIVI 3-bit 48.50, TurboQuant 2.5 bits 49.44 (vendor-reported, not re-run). IsoQuant reports kernel speed only; real-model KV evaluation is future work. RotorQuant's README (self-reported): iso3/iso3 +4.2% and turbo3 +6.6% PPL. Changes §5.
- **llama.cpp and vLLM.** Quantized V needs flash attention (`src/llama-context.cpp` L570, L4088). PR #7412 (model and context NOT STATED): at matched 10.25 bits, K-only q4_0 adds 0.19 PPL over f16 and V-only adds 0.02 (computed from the PR table). Discussion #5932 (Qwen2.5-Coder-7B, Q6_K): q8_0 K and V add 0.004 PPL. vLLM FP8 KV: per-tensor or per-head scales (per-head needs the FlashAttention backend), defaults 1.0 without calibration; accuracy NOT STATED. Changes §4 Card A and §5.
- **Gated DeltaNet (arXiv 2412.06464, Eq. 10) and fla.** S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ: a decay gate and a delta term. State precision NOT STATED in the paper. fla keeps fp32 accumulators and returns an fp32 final state on both paths (`fla/ops/gated_delta_rule/fused_recurrent.py` L104–106, L213–215; `fla/ops/common/chunk_delta_h.py` L101–115). Qwen3-Next card: 12 × (3 DeltaNet + 1 attention) blocks, the repo's 36 + 12 split. Changes §2.3 and Card B.
- **Open (not read or not found).** qwen.ai blog NOT FOUND; EXL3 decode kernel (not read); QTIP lookups per weight and quality-versus-bpw; KVQuant C4 and Table 18; TurboQuant H100 claims; the vLLM third calibration method.

## 9. Not checked

- No GPU run, no model load, no measured number this round. Capacity numbers are derived from the layouts and the repo's measured arena budgets.
- Not checked: the EXL3 decode cost (§1.2), integer multiply rates on gfx906 and gfx1030, the K-Hadamard effect on this model, f16 GDN drift, the ordinal-to-card mapping (§2.4), and the 78 MiB gap between the 930 MiB arena cost per block and the 852 MiB of expert bytes (not broken down).
