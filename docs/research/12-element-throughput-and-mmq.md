# 12 — What limits elements/ms, what to do about it, and llama.cpp's MMQ/MMVQ

Round 12. Paired with the implementation round that owns the single-token GEMV.
Engine audited: **llama.cpp `ggml-vulkan`** (`Reference Projects/llama.cpp-ggml-vulkan`,
read-only). This is the *second* pass on it: round 09 covered the scheduler/record path,
this one goes into `vulkan-shaders/mul_mmq*` and `mul_mat_vecq*` — the quantised-GEMV
inner loop, which is the thing that beats us on this box.

**Headline.** Quantised decode on this hardware is **instruction-issue bound**, and the
constant is measurable: element rate ≈ **K / instructions-per-element** with
**K = 7.8 × 10¹² instruction slots/s on the MI50 and 1.59 × 10¹³ on the RX** — i.e. the
MI50 kernel is issuing **1.18 wave64 instructions per cycle per CU** (of the one it can
issue) and the RX **2.7 wave32 per cycle** (of four). Two unrelated quant types on the
MI50 (`IQ3_S` 351 M el/ms, `Q5_K` post-fix 315) imply that
same K to **within 2.3 %** (7.77 vs 7.95 × 10¹²), which is the evidence that the model
is right and not a coincidence. The leftover gap in the *other* types is a **dependent
codebook-lookup stall**, and it scales monotonically with lookups per unit
(2 → 100 % of the model, 3 → 69 %, 8 → 47 %). Bytes do not appear in the law at all
until the type is cheap enough to be DRAM bound — and on the MI50 that happens only
below ~1 instruction per bit of weight, i.e. never for a codebook quant.

So the ranked answer is: **cut instructions per element, and the biggest single cut is
the int8 dot (`OpSDot` / `v_dot4_i32_i8`) plus two algebraic tricks that only exist on
the integer path** — the per-block scale applied once per block instead of per element,
and the `min`/offset term collapsed into one multiply against the q8_1 block *sum*.
That is exactly llama.cpp's `mul_mat_vecq.comp` / `mul_mmq.comp` structure (§B).
Predicted: **−3.5…−4.5 ms/token** on the IQ3/IQ2 stages, **−1.1…−1.9 ms** on Q5_K/IQ4_XS,
plus 2.2…2.7 ms from the structural items that need no new extension.

**No GPU was touched this round.** Another agent owns both cards for the concurrent
`VK_KHR_shader_integer_dot_product` probe, so every number below is either (a) a
measurement already recorded in `docs/phase-4/01-forward-pass.md` §8av/§8ax/§8ay, or
(b) a **static** count of SPIR-V instructions compiled with our own build flags (§A1).
The one new measurement is host-only and reproduces in seconds; commands in §A1 and the
"Sources" section.

---

# Part A — what limits elements/ms

## A1. Method: count the actual instruction mix

`build.rs` compiles every kernel with `glslc -O --target-env=vulkan1.3`. The decode unit
is one GLSL function per type (`crates/breccia-kernels/shaders/blocks.glsl`), called once
per 8-element unit per lane, so a probe that calls it once and stores the result isolates
**one unit's instruction cost** with the loop, the dot and the prologue removed. Baseline
(empty unit, same bindings) is 30 instructions, so `D = total − 30`.

```bash
# host only: no device, no dispatch. Harness (not in the repo):
cd /var/tmp/users/korphaus/tmp/opencode/research-harnesses/r12-instr-mix
# probe_common.tmpl = #version 450 + the two extension pragmas + blocks.glsl include,
# layout(local_size_x=32), one DEQUANT_UNIT(...) call, sum the 8 values, one store.
glslc -O --target-env=vulkan1.3 -I crates/breccia-kernels/shaders -I . out/<type>.comp -o out/<type>.spv
spirv-dis --no-color out/<type>.spv -o out/<type>.dis
grep -c '= Op' out/<type>.dis                       # total
grep -cE '= Op(Shift|Bitwise|IAdd|IMul|ISub)' ...   # mix
```

Per **8-element unit, per lane** (i.e. per 256-element block split across 32 lanes):

| type | bpw | block B | **D** (default) | **D** (wide) | of which bit-ops | int add/shift | loads | converts | `select` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IQ2_XXS | 2.06 | 66 | **115** | 115 | 18 | 18 | 12 | 7 | 3 |
| IQ2_XS | 2.31 | 74 | **115** | 115 | 18 | 17 | 10 | 5 | 4 |
| IQ2_S | 2.47 | 82 | 132 | 132 | 27 | 21 | 11 | 7 | 4 |
| IQ3_XXS | 3.06 | 98 | 132 | 132 | 24 | 24 | 12 | 8 | 3 |
| **IQ3_S** | 3.44 | 110 | **165** | 165 | **41** | **31** | 12 | 9 | 4 |
| IQ4_XS | 4.25 | 136 | 104 | **121** | 24 | 19 | 9 | 8 | 2 |
| IQ4_NL (/32 el) | 4.50 | 18 | 69 | 69 | 10 | 10 | 7 | 4 | 2 |
| Q2_0 (/64 el) | 2.25 | 18 | 49 | 49 | 3 | 7 | 6 | 3 | 1 |
| Q4_K | 4.50 | 144 | 142 | 142 | 36 | 26 | 12 | 11 | 3 |
| **Q5_K** | 5.50 | 176 | 162 | **190** | 41→48 | 31→34 | 13→19 | 12→18 | 8→12 |
| Q6_K | 6.56 | 210 | 124 | 124 | 26 | 28 | 8 | 8 | 5 |
| BF16 | 16.0 | 512 | **37** | 37 | 2 | 7 | 5 | 2 | 1 |
| F16 | 16.0 | 512 | 34 | 34 | 1 | 7 | 5 | 1 | 1 |

Three facts fall straight out of the mix:

1. **Bitfield extraction dominates.** `IQ3_S` is 41 bit-ops + 31 int add/shift = **44 %**
   of the unit; `Q5_K` wide is 48 + 34 = **43 %**. FMAs are *not* the cost: the whole
   dot (2 × `OpDot` on `vec4` = 8 FMA) plus 2 `vec4` loads adds ~12 instructions to a
   unit of **165–190**. **The dequant is 92–94 % of the arithmetic; the MAC is 6–8 %.**
   This is the single most important number in the report: *vectorising the dot (half2,
   `v_pk_fma_f16`) can recover at most 4 %.*
2. **The `_wide` build raised `Q5_K` 162 → 190 (+17 %) and yet made it 2.25× faster**
   (§8ax: 140 → 315 M el/ms). So the pre-fix limit was **not** instructions: it was the
   8× sector amplification of 16 two-byte loads at an 8-byte lane stride. Instruction
   count only becomes the limit *after* the memory path is clean. `IQ4_XS` moved
   104 → 121 for **no** gain, which is the same statement read the other way: it is not
   instruction bound either (see §A2).
3. **Per-element instruction counts are 6–8× the floor.** An fp32 dequant-then-FMA needs
   ≥ 3 lane-ops per element irreducible (1 extract, 1 `u2f`, 1 FMA). `IQ3_S`'s unit is
   177/8 = **22.1 lane-ops per element**. `Q5_K` wide is 202/8 = **25.3**.

## A2. The law: element rate = K / instructions-per-element

Machine capacity, and the only calibration freedom is "how many instruction slots per
cycle per CU":

* **MI50 (gfx906)** 60 CU × 64 lanes × 1.725 GHz = **6.62 × 10¹² lane-ops/s**
  (13.25 TFLOP/s ÷ 2 — agrees with the datasheet, one wave64 instruction retires per
  CU per cycle).
* **RX 6950 XT (gfx1030)** 80 CU, 2.3 GHz. Our own round-06 measurement pins
  **5 120 FP32 lanes** for FMA-dense code (5.09 TFLOP/s at the 500 MHz DPM state is
  exactly 5 120 × 2 × 0.5 GHz), but the *measured* element rates below only fit with
  ≈ 2.4 instruction slots per cycle per CU — RDNA2's dual-issue FP32 is real for this
  mix (loads, shifts, masks) while AMD's TFLOPS figure counts only the first issue.
  Calibrated: **K_RX = 1.59 × 10¹³ slots/s**.

Expressed as issue rate rather than lane-ops, since that is what the number means:
K_MI50 = 75 lane-slots/CU/cycle = **1.18 wave64 instructions per cycle per CU** (of the
1 a gfx906 CU can issue — saturated), K_RX = 86 lane-slots/CU/cycle = **2.7 wave32
instructions per cycle per CU** (of the 4 an RDNA2 CU can issue — 67 %). Either way the
kernel is issuing flat out; "occupancy" and "bandwidth" are not the quantities to tune.

Rearranged for the unit form: `rate (M elements/ms) = K / (I_unit/8) / 10⁹`, i.e.

> **MI50: rate ≈ 62 400 / I_unit  ·  RX: rate ≈ 127 000 / I_unit**

where `I_unit` = instructions per 8-element unit (decode `D` + ~12 for the dot). Then:

| type | I_unit | MI50 model | MI50 measured | **efficiency** | lookups per unit |
|---|---:|---:|---:|---:|---:|
| IQ3_S | 177 | 353 | **351** | **100 %** | 2 (grid) |
| Q5_K wide | 202 | 309 | **315** | **102 %** | 0 |
| IQ2_XS | 127 | 491 | 345 | 70 % | 3 (2 grid + sign table) |
| IQ2_XXS | 127 | 491 | 332 | 68 % | 3 |
| IQ4_XS | 133 (wide) | 469 | 221 | **47 %** | **8 (float LUT per element)** |
| Q5_K pre-fix | 174 | 359 | 140 | 39 % | — (sector bound) |
| BF16 | 49 | 1 273 | 405 | DRAM: 413 | 0 |

**Validation, two ways.** (i) The two *different* types that have no stall land on the
same K to 2.3 % (IQ3_S implies 7.77 × 10¹², Q5_K-wide 7.95 × 10¹²) — a shared shape and
accumulator order, two unrelated decoders, one constant. (ii) BF16 does *not* follow the
instruction model: it lands at 405 M el/ms = **98 % of its 413 M el/ms DRAM ceiling**,
which is the independent confirmation that the *memory* half of the model is right too.

**Where the residual efficiency goes: dependent codebook lookups.** The gap is monotone
in lookups per unit — 2 → 100 %, 3 → 69 %, 8 → 47 %. `IQ4_XS` is the pathological case:
`iq4nl_value(q)` is a *per-element* float table load (the GLSL compiles to 8 dependent
global loads per unit), so one lane waits ~8 × L1 latency per unit and only ~47 % of the
issue slots are usable. That is a latency task, not an instruction-count task, and it is
the one place where a codebook move to **LDS** pays (see A3.4).

**And the crossover.** A type is ALU bound rather than DRAM bound iff
`K/(I_unit/8) < BW/(bpw/8)`, i.e.

> **ALU-bound iff `I_unit > (8·BW·10⁹/K) · bpw`**  →  **`I_unit > 8.0 · bpw` (MI50)**,
> **`I_unit > 53 · bpw` (RX)**.

Read that twice. On the MI50, *any* codebook quant is ALU bound by 4–5× (IQ3_S: 177 vs
27.5; Q5_K: 202 vs 44) — bytes are almost irrelevant there. On the RX you must spend
**53 instructions per bit of weight** before ALU matters, so the RX runs out of DRAM long
before it runs out of instruction slots. The head is the worked example and it was already
measured on both cards: `I_unit` 202 vs 44 (MI50, ALU bound → 315) and vs 292 (RX, not
ALU bound → 630 = **97 % of 445 GB/s**).

**This is the quantitative form of "matched to measured strengths"**: instruction-bound
work belongs on the **RX** (1.59 × 10¹³ vs 7.8 × 10¹² slots/s, 2.0×), bandwidth-bound
work belongs on the **MI50** (825 vs 445 GB/s, 1.85×). The engine already shows it:
`attn_qkv` IQ3_S is 2 205 µs on the RX and 4 143 µs on the MI50 for the same work (§8av).

*A falsifier that costs 20 seconds.* The sweep is already scripted (§8ax commands):
`breccia bench gemv --rows 248320 --cols 2560 --type IQ3_S` on each card. Prediction:
**RX/MI50 ≈ 2.0–2.4** (I_unit 177 → 1 064 vs 353 modelled; 630 measured on the RX for
I_unit 202 says the ratio at equal I_unit is 2.03). If the RX comes in near 1.0, the
2.4-slots/cycle calibration is wrong and every prediction below should be halved.

## A3. The candidate fixes, costed and ranked

Ranked by (element-rate gain) ÷ (risk × effort). The arithmetic uses `rate = K/I_unit`
and `I_unit(new) = I_unit(old) − removed`, and each item states what it removes.

### 1. int8 dot + q8_1 activation — the only item that changes the *class* of the kernel

`dotPacked4x8EXT(a,b)` (GLSL `GL_EXT_integer_dot_product` → SPIR-V `OpSDot` packed 4×8 →
`v_dot4_i32_i8`) computes **4 MACs per instruction slot**, and three things disappear that
have nothing to do with the MAC:

| removed | per 8-element unit | why it can go |
|---|---:|---|
| `u2f` converts | 8 | the dot consumes int8 directly |
| `fmul`/`fadd` of the MAC | 8–16 | folded into the dot |
| the per-element `- dm` (Q4_K/Q5_K/Q6_K) | 8 | `Σ(d·q·x) − dm·Σx`: the offset becomes **one** multiply against the q8_1 block **sum** (`ds.y`), not K multiplies |
| the per-element scale multiply | 8 | int accumulation is scale-invariant inside a block, so `d` is applied **once per block** at the epilogue |
| float codebook → int8 codebook (IQ4_NL/IQ4_XS) | 4–8 loads | `iq4nl_to_i8x8()` repacks 8 int8 from 4 words |

New unit costs, from llama.cpp's own repack shapes (§B2) counted the same way:
**Q5_K int8 ≈ 55–65**, **IQ4_XS int8 ≈ 45–55**, **IQ3_S int8 ≈ 90–100**,
**Q6_K int8 ≈ 60**, **Q4_K int8 ≈ 60**.

| type | I_unit | new I | MI50 model | vs measured | capped by DRAM at |
|---|---:|---:|---:|---:|---:|
| Q5_K wide (head) | 202 | ~60 | 1 040 | 315 → **3.3×** | 1 200 (→ 1 040) |
| IQ3_S (GDN, qkv) | 177 | ~95 | 657 | 351 → **1.87×** | 1 919 |
| IQ4_XS (shared) | 133 | ~50 | 1 248 | 221 → **2.7–3×** (also removes the 8-LUT stall) | 1 553 |
| Q6_K | 136 | ~72 | 867 | – | 1 006 |
| IQ2_XS/XXS | 127 | ~100 | 624 | 345 → 1.8× | 2 857 |

**The honest caveat, and it is the concurrent probe's question.** llama.cpp's own
architecture detection distinguishes RDNA3 from RDNA2 by
`integerDotProduct4x8BitPackedMixedSignednessAccelerated`
(`ggml-vulkan.cpp:44-56`) — i.e. **gfx1030 does not report the 4×8 packed dot as
*accelerated***, while gfx906 has the DLOps unit (Vega 20: 26.5 TOPS INT8 = 2× its
13.3 TFLOP FP32). llama.cpp does not gate on that flag for AMD, it only requires the
extension to exist, so the int8 dot is shipping on *both* our cards — but the **gain on
the RX is instruction-count only, not a throughput multiplier**, and the RX is DRAM bound
for Q5_K anyway (630 of a 647 ceiling), so **int8 dot is worth ~0–10 % on the RX and
1.9–3.3× on the MI50.** Rank it accordingly: it is a *MI50* fix. The probe owns the
number; this document owns the decision rule: **port it iff it beats
`dotPacked4x8EXT` → 4× `v_fma_f32` by ≥1.5 slots on the MI50**, and do not expect it to
move the RX.

### 2. The q8_1 producer must cost zero dispatches

The int8 path needs `x` as int8 + per-32-group scale + block sum. llama.cpp pays a
`quantize_q8_1` dispatch per matmul (`quantize_q8_1.comp`, `ggml-vulkan.cpp:6335`);
at our 1.1–4.4 µs per launch that would be **+0.1…+0.4 ms/token** and it must not be
paid. Our advantage: our elementwise producers (`ops_rmsnorm`, `ops_swiglu`, `ops_act`,
`hc_gated_mean`) already reduce a whole row in one workgroup, so emitting the q8_1 form
beside the f32 form is a **wired change to the producer** (one extra output buffer, one
extra reduce over the 32-element absmax) — and one input row feeds several projections
(`attn_qkv` + `attn_gate` + `ssm_beta/alpha` + router all read the same normalised `x`),
so ~12–20 producers per token replace ~96+ quantise calls. Cost: ~2–3 µs per producer row,
i.e. **≤0.05 ms/token**. That fusion is what makes item 1 cheap; without it, item 1 is
worth ~20 % less.

### 3. Packed-nibble unpack for the fp32 path (no new extension) — 1.2–1.6×

Even without a dot, the extraction can be 4 elements per instruction: mask a 32-bit word
with `0x0F0F0F0F`, shift by 4, and you hold 4 quant codes; `unpack8` gives 4 bytes for a
`u8vec4`. Our `_wide` `Q5_K` unit still walks `jj = 0..7` with a select for the 5th bit and
a shift per element (`byte4(jj<4?qa:qb, jj&3)`), which is why it is 190 not ~70. Floor for
fp32: 1 extract + 1 `u2f` + 1 FMA per element ≈ 24–32 per unit, so **I_unit 70–90 is
reachable while staying bit-identical for the MAC order** — the same values, the same fp32
accumulation, just a better unpack. Predicted 202 → ~90 on Q5_K: **330 → 690 M el/ms**
(1.2–1.6× the current wide build after allowing for the fp32 `u2f` floor). This is the
item to do *first* if the int8 dot lands badly, and it composes with it.
`Q4_K`, `Q6_K`, `IQ4_XS` share the pattern; `IQ2/IQ3` do not (their codes are grid
*indices*, consumed by a lookup, not by arithmetic).

### 4. Codebooks from global (SSBO) to LDS — only where the stall is

llama.cpp keeps its IQ3_S grid in **2 KiB of `shared` memory**, filled once per
workgroup: `shared uint32_t iq3s_grid[512]` + `init_iq_shmem()` (`types.glsl:1577`,
`:1749-1757`) — the same conclusion our `blocks.glsl` reached for a different reason
(a dynamically indexed GLSL `const` array spills to private memory on RADV; llama.cpp
declares the const array and then *copies* it to LDS, we upload it to a 7 KiB SSBO).
LDS latency (~20–30 cycles, no descriptor, no generic-address path) versus our L1-resident
SSBO (~40–60). Predicted: **IQ4_XS 221 → 270–300** (the 8-lookup stall), **IQ2 332 → 360**
(the 3-lookup stall), **IQ3_S no change** (already 100 % of its model — it is *not*
latency bound, so this is the cheapest placebo in the list and should be measured on
IQ4_XS only). Worth **0.1–0.3 ms/token**; do it as a rider, not a round.
*(If item 1 lands, the LUT becomes 4 packed int8 words and this becomes moot.)*

### 5. `half2` / packed fp16 (`v_pk_*` on RADV): ~4 %, do not

The dequant output is consumed by an fp32 dot, so packed f16 can only help the MAC and the
scale multiply: 8 FMA → 4, 8 mul → 4, and `v_cvt_pk_f16_f32` converts 2 values per
instruction — but **there is no packed int→float convert on RDNA2 or gfx906**, so the 8
`u2f` (the second-largest removable term) stay. From the mix in §A1: dot+loads are 12 of
165–190 instructions, of which maybe half is recoverable → **3–4 %**, and it costs the
fp16 accumulate path we already measured as an accuracy knob. Not worth a pipeline.

### 6. int16 accumulate: no. Nothing to accumulate into

The int accumulator inside a dot is already **int32 and exact** (256 × 127 × 127 ≈ 4.1 ×
10⁶, no overflow). Outside a dot there is no int path to accumulate. What int16 *would*
buy is register pressure at `m > 1` in `gemm_body` — irrelevant at 16 accumulator
registers. Drop.

### 7. Pre-dequantised weight caching — priced, and it loses twice

Two independent tests.

**(a) Time.** A cached f16/bf16 copy has `I_unit` 49 → model 1 273 M el/ms, so it is
**DRAM bound at 413 M el/ms** on the MI50 (measured 405 = 98 %). It therefore beats
today's quant decode on every type (1.2–1.9×: IQ3_S 351, IQ2_XS 345, IQ4_XS 221,
Q5_K-def 140) **and loses to the int8 path (624–1 248)**. So caching is what you do
*instead of* writing the dot, never alongside it. Per tensor: the Q5_K head (417 MiB)
becomes 1.20 GiB and runs **2.02 → 1.48 ms on the MI50** (1.4×) — but on the **RX** the
same copy is 1.20 GiB at 445 GB/s = **2.7 ms against 1.009 ms today, a 2.7× loss**,
because the RX is DRAM bound. The law in §A2 predicts that asymmetry exactly.

**(b) Space — the decisive one.** Dense set 3.66 GiB at IQ3 ≈ **8.51 G elements** → f16
copy **17.0 GiB**, int8 copy 8.5 GiB. Both discrete cards report **16 GiB**
(`mem_info_vram_total` = 17 163 091 968 B) and each already holds 3.66 GiB dense +
~8 GiB of expert arena + state. **There is no configuration in which a dequantised dense
copy fits.** Break-even in the only form that matters: caching pays iff
`17.0 GiB / BW < N·I_quant/K` **and** the copy fits — the first holds for two tensors, the
second holds for none except the head, and the head is cheaper on the RX at 5.5 bpw.
**Verdict: reject; spend the same effort on item 1.** (A middle option exists — an *int8*
copy at 8.5 GiB — but it is item 1 with the quantisation done offline, and it still does
not fit beside the arena.)

### 8. Not a fix, a placement rule: stop running ALU-bound GEMV on the MI50

From §A2: MI50 7.8 × 10¹² slots/s versus RX 1.59 × 10¹³ (2.0×), and MI50 825 GB/s versus
RX 445 GB/s (1.85×). So for every *codebook* type in this model the MI50 is the wrong card
— it is 2× slower on exactly the shapes it is given, while its bandwidth advantage is
unused because those shapes are ALU bound. Freeing the MI50 for the head (Q5_K, DRAM
bound) and the bandwidth-shaped work, and moving `attn_qkv`/`qkv`/`super` IQ3_S to the RX,
is worth **~1–2 ms/token at zero kernel risk** and needs only the existing
`--split-blocks`/placement knobs. This is already half-visible in §8av (RX 2 205 vs MI50
4 143 for `attn_qkv`) and has never been stated as a rule.

## A4. The accuracy contract

An int8 dot **cannot** be bit-identical, so it needs its own gate. State exactly what is
traded: the *weights* are unchanged (same int codes, same fp16 `d`/`dmin`, same per-block
scale); the new rounding is **the activation**, quantised to `q8_1` — one symmetric int8
code per element with a per-32-element scale `d_x = absmax/127`, plus the block sum
`Σq_x` in the header. The int32 accumulation is exact, so the *only* error source is
`|δx| ≤ d_x/2`, i.e. **≤ 0.39 % of the 32-element absmax**, applied once. Weights keep
their current 3-bit error, which is 5–10× larger than this, so the path is not the
dominant term — but "small" is not a gate.

**Proposed gate (four tests, all cheap, all on existing harnesses):**

1. **Per-stage relative error, the predictive number.** For every matmul stage on the
   parity prompt (prefix 32), `||y_int8 − y_f32||₂ / ||y_f32||₂ ≤ 2 × 10⁻³` and
   `max|Δy| / max|y| ≤ 5 × 10⁻³`. Theory: uniform quantisation error uncorrelated with
   the weights gives a dot error ≈ `(d_x/2)/√3 / x_rms · 1/√1` ≈ 0.11 % RMS, so 2 × 10⁻³
   is a ~2× pass band, not a guess. New harness: dump both stage outputs for one prompt,
   one process, ~1 s.
2. **Greedy-token agreement across processes.** `scripts/head-handoff-race.sh` already
   runs *N* **exclusive** processes (`RACE_PROCS=8 RACE_RUNS=1 RACE_GEN=32`). Gate:
   **first 8 tokens identical in 8/8 processes; 32 tokens identical in ≥ 7/8** (one
   process may flip ≤ 1 token). The first-token rule is the strict one: top-1 is
   `13.170561` and its margin is the model's confidence, so a flip there is a real
   regression, not sampling noise.
3. **Logprob delta, so the gate measures margin and not luck.** On the first generated
   token: `max|Δlogit|` over the top-20 candidates **≤ 0.02**, and top-1 softmax
   probability delta **≤ 0.005**. Without this, a 32/32 pass cannot distinguish
   "identical for a good reason" from "identical by 1e-4".
4. **Perplexity on a fixed 4 K-token holdout within +0.3 %** of the f32-activation path.

**What the reference engines accept.** llama.cpp ships **weaker** than any of this: on
AMD, `ggml_vk_should_use_mmvq` returns true for `k ≥ 2048` at `n = 1`
(`ggml-vulkan.cpp:6696-6706`), so every k-quant and IQ4_XS/Q6_K decode on this box
already runs int8 dots against `q8_1` activations with no gate beyond llama.cpp's own
perplexity CI; `mul_mmq` uses `q8_1` for *all* batch sizes including the IQ types.
`ggml_vk_src1_prec_allows_q8` exists precisely so a user can *opt out*
(`GGML_PREC_F32`). MLC-LLM's Vulkan path has no int8 dot at all (f16/bf16 activations
only). So the industry position is "q8_1 activations are the default; make them
opt-out". Our position can be stricter *and* cheaper to defend: keep the f32 path as the
default and the parity reference (it is already the sanctioned shape — a flag, default
off, bit-identical incumbent, one-line revert), and gate the int8 path on the four tests
above.

---

# Part B — llama.cpp's MMQ, in depth

## B1. Three decode kernels, and which one is ours

| shader | shape | activation | MAC | when |
|---|---|---|---|---|
| `mul_mat_vec_<t>.comp` | 1 row/… per WG, `BLOCK_SIZE` threads | **fp32** | fp32 FMA after dequant | fallback, and **all IQ2/IQ3 types** |
| `mul_mat_vecq.comp` (MMVQ) | same | **q8_1** | `dotPacked4x8EXT` | AMD, `k ≥ 2048`, types: legacy, Q2_K…Q6_K, IQ1_S/M, **IQ4_XS**, mxfp4 |
| `mul_mmq.comp` (MMQ) | tile `BM×BN`, `BK`-chunked | **q8_1** | `dotPacked4x8EXT` from **LDS** | `n ≥ 2`, i.e. prefill / batched |

The third table row is the one that matters for our **multi-user GEMM**; the second is the
one that matters for **decode**. And the first row is the answer to a question worth
asking explicitly: **for IQ2_XXS/IQ2_XS/IQ3_S at batch 1, llama.cpp has no int8 path
either** — `mul_mat_vecq.comp` is generated only for
`is_legacy_quant || mxfp4 || is_k_quant || iq1_s || iq1_m || iq4_xs`
(`vulkan-shaders-gen.cpp:817`), so IQ3_S decode uses `mul_mat_vec_iq3_s.comp`:
grid lookup → `vec4` → **8 named `fma()`** → `temp += dscale*sum`. That is structurally
what we do. Their *MMQ* does have IQ3_S (`sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1},
…, tc_mmq_int_k, "matmul_iq3_s_q8_1", …)`, `ggml-vulkan.cpp:2646`), which tells you the
int8 form of IQ3_S exists and is worth having — it just needs the tile to amortise the
grid lookups.

## B2. The block→fragment layout

Decode is split in two halves that never meet in the inner loop:

**Weights (`A`) → int8 fragments.** Per type, `block_a_to_shmem()` dequantises the block
**once per (row, BK-chunk)** into an int8 `block_a_cache` in LDS, with the per-block scale
folded into a `dm`/`d`/`d_scales` field at the same time:

```glsl
// Q5_K, mul_mmq_funcs.glsl:459-525  (BK=32; qs_idx/qs_shift walk the 4-bit stream)
buf_a[buf_ib].qs[iqs] = int(((qs[qs_idx] >> qs_shift) & 0x0F0F0F0F)      // 4 nibbles, 1 instr
                          | (((qh[qh_idx] >> qh_shift) & 0x01010101) << 4));  // 5th bits
if (iqs == 0) buf_a[buf_ib].dm = FLOAT_TYPEV2(dm) * vec2(scale_dm);    // scale × 6-bit sc
...
q_sum += dotPacked4x8EXT(cache_a.qs[iqs], cache_b.qs[iqs]);            // 4 MACs/slot
return float(cache_b.ds.x) * float(cache_a.dm.x) * float(q_sum)
     - float(cache_a.dm.y) * float(cache_b.ds.y);                      // min: ONE multiply
```

`QUANT_R_MMQ` (`mul_mmq_shmem_types.glsl`) is the packing factor — 1 for Q5_K/Q6_K
(one `qs` word per 32 elements), 2 for Q4_K/IQ4_XS/IQ3_S, 4 for Q2_K — and it is what makes
the shift/mask count per 32 elements rather than per element. Our `blocks.glsl` has no
such constant: every unit is 8 elements, unpacked element by element.

**Activations (`B`) → q8_1, reused across the tile.** `block_q8_1` = `f16vec2 ds` +
`int8 qs[32]`, stored 4-blocks-to-a-`block_q8_1_x4` for 128-bit alignment
(`types.glsl:242-271`). `ds.x` is the scale, **`ds.y` is the sum of the 32 quants** — which
is what collapses every `- dmin·m` term to a scalar.

**Fragment map.** `{BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP}`; on AMD GCN
(`ggml-vulkan.cpp:1844`) `m = {256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16}` and on the
generic path `{128, 128, 128, 32, 64, 64, 2, 4, 4, 1, 32}` with `BK_STEP 4` (four BK
chunks per `barrier()`, i.e. 128 K-elements per round trip). Per thread the accumulator
is `sums[WMITER·TM·WNITER·TN]` = 16 floats, filled by
`block_a_to_registers` × `block_b_to_registers` → `mmq_dot_product`. `BK` is fixed at 32
for the integer path (`#define BK 32`, `mul_mmq.comp:73`) because 32 int8 = one q8_1 block
= one byte per lane.
**There is no `fixup` and no `K_QUANTS_PER_ITERATION` in the Vulkan backend** (`grep`
returns nothing) — those are CUDA MMQ concepts; the Vulkan analogues are `QUANT_R_MMQ`
(packing factor) and `BK_STEP` (chunks per barrier).

## B3. Why it beats dequant-then-FMA — four wins, only two of them about the MAC

1. **4 MACs per slot** instead of 1 (`v_dot4_i32_i8`).
2. **No `u2f`, no `fmul`, no `fadd`** — 3 instructions per element removed, replaced by
   0.25 of a dot.
3. **The scale is hoisted out of the element loop** (`d·q·x` accumulated as int, `d` applied
   once per block). In fp32 it cannot be hoisted: rounding after each element is what makes
   the result bit-reproducible, so the multiply is per element by construction.
4. **The offset collapses** (`Σ(d·q·x) − dm·Σx`, `Σx` read from `ds.y`). For Q4_K/Q5_K this
   removes 8 `fadd` per unit; for Q2_K/Q3_K it removes a *whole second dot* per 4 elements
   (`sum_m += dotPacked4x8EXT(scale_m, cache_b.qs[iqs])` still exists there because the
   scale varies *inside* the block, but the min does not).
   **Wins 3 and 4 are algebraic and are unavailable to any fp32 path** — which is why the
   fp32 floor (item A3.3, ~70–90) is above the int8 floor (~55–65) and not just slower per MAC.

LDS staging is the enabler, not the speedup, for `BN > 1`: the int8 fragment of a
`(BM, BK)` tile is read by `BN` accumulator columns, so the dequant cost divides by `BN`.
At `BN = 1` (our decode) there is nothing to amortise, which is exactly why llama.cpp
ships no MMQ for `n = 1` and why our `_moe` GEMV must get its win from items 1–4 per row,
not from tiling.

## B4. What transfers to SPIR-V/RADV on gfx1030/gfx906, and what does not

**Transfers:**
* `GL_EXT_integer_dot_product` / `dotPacked4x8EXT` — present on both cards (llama.cpp
  requires the extension on every device it runs on, and it runs here). Rate is the open
  question (§A3.1).
* `shared` (LDS) staging with `barrier()` — `shared uint32_t iq3s_grid[512]` (2 KiB) for
  the codebook; `BK_STEP = 4` chunks per barrier; `block_a_cache` of `int32 qs[8]` +
  `dm`. All plain SPIR-V.
* The packed-nibble extraction (`& 0x0F0F0F0F`, `>> 4`, `<< 4` on the `qh` bits), the
  int8 sign trick `(v ^ m) - m` (`mul_mmq_funcs.glsl:615`), the `ds.y` sum trick.
* The `m = 4`/`m = 8` register-tile shape (`WMITER·TM` × `WNITER·TN`) — directly usable by
  our `gemm_body.glsl` at `m ≤ 8`, and it is *cheaper* for us than for them because our
  `B` fragment is one 8-token slab, not a 64-token tile.
* Per-architecture row counts: `rm_kq = 4`, `rm_iq = 8` on AMD GCN, `2`/`4` elsewhere
  (`ggml-vulkan.cpp:2866-2892`) — our fixed `ROWS_PER_WG 8` is right for IQ on the MI50
  and 2× high for RDNA2, which is a one-line template parameter, not a round.

**Does not transfer:**
* `mul_mmq_cm1/cm2` — `VK_KHR_cooperative_matrix` is **absent on both cards** (round 06),
  so `TK` (coopmat-only), `cm1_bs()`, `itm/itn/itk` and the whole matrix-core path are
  dead code for us.
* The `BN = 64` token tile, `k_split` and `MUL_MAT_ID`'s `row_ids`/`row_map` indirection —
  our `moe_map.comp` + `X_BASE` already does the expert indirection more cheaply.
* RDNA3/4 tuning (`is_rdna3_or_4` 4-row overrides, `cm1_use_wave32`,
  `MixedSignednessAccelerated` gating) and the NVIDIA-pre-Turing type selection.
* `MUL_MAT_FUSION_FLAGS` bias/scale epilogues — worth copying in spirit (we already do it
  for `silu_div`), but their flag words are for `ggml`'s graph, not ours.

## B5. What our kernel becomes

Two files, two changes, both *wired* (same bindings and resources — the sanctioned
lowest-risk class), both behind one flag:

**`gemv_body.glsl` / `gemm_body.glsl`:** replace `DEQUANT_UNIT(..., float v[8])` with

```glsl
// PACK_UNIT: the 8-element unit becomes TWO packed int32 words (4 int8 each),
// plus the block's fp32 scale (and the min, where the type has one).
struct Unit { int q0; int q1; float d; float dm; };
Unit PACK_UNIT(uint word_off, uint lane);
...
Unit u = PACK_UNIT(row_base + b*BLOCK_WORDS, u_lane);
uint xb = ...;                                  // q8_1: 4 int8 + ds per 32 elements
q_sum += dotPacked4x8EXT(u.q0, XQ[xb + 0]);
q_sum += dotPacked4x8EXT(u.q1, XQ[xb + 1]);
// epilogue, once per block — not per element:
acc0 += u.d * float(q_sum) - u.dm * XSUM[b];     // XSUM = Σq_x of this 32-el group
```

with `XQ` = the q8_1 form of the activation row and `XSUM` its per-32 sums, both written
by the *producer* (§A3.2). `IQ3_S` keeps its two grid lookups but returns them **already
int8-packed** and applies the sign with `(v ^ m) − m`, exactly as
`mul_mmq_funcs.glsl:608-616` does.

**`gemm_body.glsl` at `m ≥ 4`:** add llama.cpp's tile on top — `BK = 32` chunks staged in
LDS as int8 for all `m` users, so `BM` weight rows share one dequant and one barrier per
4 chunks. With `m = 8` that divides the dequant cost by up to 8 where the *same* weight
row-block is reused — which for us means the **prefill chunk** path, not decode.

**Expected gain, from §A2's law:** `rate = 62 400/I_unit` (MI50) — `IQ3_S` 177 → ~95 is
**1.87×**, `Q5_K` 202 → ~60 is **3.3×** (capped by DRAM at 1 200), `IQ4_XS` 133 → ~50 is
**2.7×** plus the removal of the 8-lookup stall. On the RX: `127 000/I_unit`, so the same
change is 1.5× on `IQ3_S` *if* the instruction slots are real there, and ~1.0× on `Q5_K`
(already 97 % of DRAM).

---

# Top-3 experiments for the next implementation round

Predicted deltas assume the current token: **56.09 ms / 17.83 tok/s**, single stream,
dual, 30 resident blocks, `sum` 32 089 µs (§8ax).

| # | experiment | flag / revert | predicted | basis |
|---|---|---|---:|---|
| **X1** | **`PACK_UNIT` + int8 dot for IQ3_S / IQ3_XXS / IQ2_* (+ the fused q8_1 producer)** — `bench gemv --type IQ3_S` first, then the engine | `BRECCIA_GEMV_INT8=1`, default off, f32 path untouched | **−3.5 … −4.5 ms/token** | I_unit 177 → ~95 = 1.87× on `attn_qkv` 3 518 → 1 880, `qkv` 1 966 → 1 050, `gdn-z` 1 809 → 970, `moe` 4 021 → 2 300 (IQ3_XXS 417 → 780, it sits at 366), `gdn-out` 2 873 → 2 400 (it is at 56 % of its ceiling, latency-floored). Sum −4.7, discounted for the latency-floored rows |
| **X2** | **`Q5_K`/`Q4_K` int8 with the `Σx` min-collapse + `--head-device`** — the head, and the shared expert's `IQ4_XS` | `BRECCIA_GEMV_INT8_QK=1`, default off | **−1.1 … −1.9 ms/token** | head 1 772 → ~640 on the MI50 (I_unit 202 → 60); **0 on the RX** (630 of a 647 DRAM ceiling), so this item is only worth it *with* the head on the MI50, which inverts §8ax's placement advice for this one stage — measure both. Shared 1 508 → 900 (IQ4_XS 156 → ~260 M el/ms on its 0.6 MB) |
| **X3** | **the no-new-extension group**: (a) `gated` → `up` epilogue fusion, already queued (§8ay, −1.5); (b) merge `router` into the `hc-gated-mean` dispatch, or run it on the RX (47 M el/ms = 64 WGs = 0.8 WG/CU); (c) codebook → LDS for `IQ4_XS` only | three independent flags | **−2.2 … −2.7 ms/token** | (a) 1 575 µs → 0 (b) 1 345 → ~800 (c) 221 → 270-300 M el/ms on a 0.6 MB tensor ≈ −0.1 |

**If X1+X2+X3 all land:** 56.1 − 6.8…9.1 = **47–49 ms/token, 20.4–21.3 tok/s**. Order
matters for attributing the K law: X1 first (it changes `I_unit` on the biggest stage
group), then re-run the `bench gemv` sweep — if `IQ3_S` does **not** reach ≥ 600 M el/ms on
the MI50, the int8 dot is not 4×-per-slot there and X2 must not be attempted on the RX.

**Decision inputs owned by the concurrent probe (not this document):** whether
`OpSDot`-packed lowers to `v_dot4_i32_i8` at full rate on gfx906 and gfx1030, and the
`integerDotProduct4x8BitPackedSignedAccelerated` property on each card. If gfx1030 reports
`false` and the MI50 rates well, the int8 path is a **MI50-only** pipeline and §A3.8's
placement rule becomes part of the same commit.

---

## Sources

* New this round (host only, no GPU): `/var/tmp/users/korphaus/tmp/opencode/research-harnesses/r12-instr-mix/`
  — `gen.sh` / the inline Python in §A1, `out/*.spv`, `out/*.dis`, `out/counts.json`.
  Reproduce: `glslc -O --target-env=vulkan1.3 -I <repo>/crates/breccia-kernels/shaders …`.
* Our code: `crates/breccia-kernels/shaders/blocks.glsl` (all decode units; `_wide`
  variants at `:273-292`, `:347-378`), `gemv_body.glsl:108-142` (block loop, `vec4` dot),
  `gemm_body.glsl:70-102` (multi-user), `build.rs` (`-O`, `WIDE_SHADERS` list),
  `crates/breccia-kernels/src/…` (no integer-dot query exists yet — it has to be added
  with `VK_KHR_shader_integer_dot_product` + the `VkPhysicalDeviceShaderIntegerDotProduct
  FeaturesKHR` struct).
* llama.cpp (`Reference Projects/llama.cpp-ggml-vulkan`, read-only):
  `vulkan-shaders/mul_mmq.comp` (tile, `BK 32`, `BK_STEP 4`), `mul_mmq_funcs.glsl`
  (`block_a_to_shmem`/`mmq_dot_product` per type — IQ3_S `:595-648`, Q4_K/Q5_K `:457-525`,
  Q6_K `:528-590`, IQ4_XS `:279-320`, `block_b_to_shmem` `:651-682`),
  `mul_mmq_shmem_types.glsl` (`QUANT_R_MMQ`), `mul_mat_vecq.comp` + `mul_mat_vecq_funcs.glsl`
  (MMVQ: `repack`, `mmvq_dot_product`, `mul_q8_1`), `mul_mat_vec_iq3_s.comp` (the IQ3_S
  decode-GEMV, per-lane 32-element group, 8 independent grid loads), `types.glsl:242-271`
  (`block_q8_1_x4`), `:1577`/`:1749-1757` (LDS codebook + `init_iq_shmem`),
  `vulkan-shaders-gen.cpp:786-825` (which types get MMVQ), `ggml-vulkan.cpp:1-120`
  (arch detection: `integerDotProduct4x8BitPackedMixedSignednessAccelerated` separates
  RDNA3 from RDNA2), `:1700-1850` (warptile tables, AMD GCN override), `:2866-2892`
  (`rm_kq`/`rm_iq` per arch), `:6023-6070` (`src1_prec_allows_q8`), `:6335`
  (`ggml_vk_quantize_q8_1`), `:6656-6720` (`ggml_vk_should_use_mmvq`: AMD `k ≥ 2048`).
* Prior measurements reused: `docs/phase-4/01-forward-pass.md` §8av (per-stage GPU table),
  §8ax (the shape-controlled type sweep, the Q5_K `_wide` fix, the per-stage element
  rates), §8ay (the `hc-read` sub-stage table, the GDN merge, the small-dispatch floor);
  `docs/research/11` (record floor), `06`/`06a` (geometry and barrier prices),
  `09` (host-memory lane rates).

## Not done here, and why

* **Any GPU measurement.** Both cards are owned by the concurrent
  `VK_KHR_shader_integer_dot_product` probe (round 12 is paired; duplicating its sweep
  was explicitly out of scope), and the AGENTS.md rule is one GPU process at a time.
  Every rate above is either already recorded in `docs/phase-4/01` or derived from a
  static instruction count — both reproducible without a device.
* **An ISA dump.** `RADV_DEBUG=shaders` / `ACO_DEBUG` would confirm the SPIR-V →
  `v_dot4_i32_i8` mapping and the real VALU counts, but it requires creating a device and
  compiling pipelines, i.e. a GPU process. That is X1's first step, not this document's.
