# 06a — independent check on small-M GEMM shapes, and the *cause* of the barrier price

**Round:** paired research 6 (second harness, 2026-10-27 00:0x). **Read-only**: no engine crate touched.
**Why a separate file:** `06-vulkan-gemm-shapes-and-ncnn.md` (the round's main deliverable) was already
written by a parallel harness when this one finished; this file is the *independent* arm — a different
harness (`/var/tmp/users/korphaus/tmp/opencode/research-harnesses/vk-gemm-shapes`), a different quant
(**IQ4_XS**, which `06` did not measure: its sweep is Q4_K), and one result that changes a price in
`06` §3. It does not restate `06`; where the two agree, it says so in one line.

## 0. Method (and the three traps it hit, two of which `06` also flagged)

| | |
|---|---|
| Harness | standalone `ash` binary, own `VkDevice` with `VK_EXT_subgroup_size_control` + `VK_EXT_robustness2`, `glslc -O` + `spirv-val` at build time, `#include` of the **engine's own** `blocks.glsl` (`-I crates/breccia-kernels/shaders`) so the decode under test is the shipped one |
| Card | RX 6950 XT `0000:03:00.0` only (headless). Nothing was run on the MI50 (display card) |
| Shape | shared expert gate/up `N=640, K=2560`, `bpr=10`, IQ4_XS (136 B/256 elem), m = 1/2/4/8 riding in `pc.splits` |
| Timing | queue timestamps around `iters` dispatches in **one** command buffer, `iters` sized for ~20 ms; **min of 5**; each pipeline pre-warmed with a dispatch + `vkDeviceWaitIdle` **and** a ~1.2 s clock-ramp pass before the first cell |
| Shapes per process | **≤ 4 fresh pipelines per invocation** (AGENTS.md incident-log rule), pre-warm before any timing |
| Clocks | read from `/sys/bus/pci/devices/<pci>/drm/card*/device/pp_dpm_sclk`: the card idles at **500 MHz** and boosts to **2720 MHz**. Every number below was taken at 2 255–2 685 MHz |

Three measurement facts that cost time and are worth carrying forward:

1. **The first dispatch of a process is ~5× slow** (8.45 µs → 1.69 µs for the same cell): the card is at
   500 MHz and a 640-row shape (80 workgroups) cannot ramp it. `06` §0.1 found the same independently;
   two harnesses, two quants.
2. **A `HOST_VISIBLE` allocation is not necessarily VRAM.** Sizing the weight buffer with
   `DEVICE_LOCAL|HOST_VISIBLE` put it in the **visible-VRAM aperture**, and the same kernel measured
   **8.44 µs instead of 1.69 µs** (5×) purely from uncached reads. Weights must be `DEVICE_LOCAL` with a
   staging upload — `breccia-vk`'s `Gpu::buf(req = DEVICE_LOCAL, …)` path does this correctly, so the
   trap only bites hand-rolled harnesses and any future "just map the weights" shortcut.
3. **A single-cell delta under ~10 % is not a result on this harness** (see the two runs in §1: the
   geometry ranking flips between them). Report the band, not the winner.

## 1. The shape sweep, independently (IQ4_XS) — `06`'s conclusion holds

Pipelined protocol (no barrier between the timed dispatches — a **throughput** number; §2 explains why it
is ~3× below the dependent-chain latency). µs per 640×2560 dispatch, RX at ≥2 255 MHz.

Run 1 (12 cells, 4 processes):

| WG / rows‑per‑WG / lanes‑per‑row / subgroup | m=1 | m=2 | m=4 | m=8 | GFLOP/s @8 | GB/s @m=1 |
|---|---:|---:|---:|---:|---:|---:|
| 64 / 2 / 32 / 32 | 1.84 | 2.23 | 3.31 | 5.85 | 4 482 | 474 |
| 128 / 4 / 32 / 32 | **1.64** | 2.24 | 3.21 | 5.03 | 5 212 | 530 |
| **256 / 8 / 32 / 32** (shipping) | 1.89 | 2.32 | **3.09** | **5.02** | **5 225** | 461 |
| 64 / 1 / 64 / 32 | 1.83 | 2.42 | 3.49 | 6.22 | 4 216 | 476 |
| 128 / 2 / 64 / 32 | 1.85 | 2.55 | 3.56 | 5.81 | 4 510 | 470 |
| 256 / 4 / 64 / 32 | 1.85 | 2.59 | 3.63 | 5.93 | 4 419 | 471 |
| 128 / 1 / 128 / 32 | 1.91 | 2.73 | 4.02 | 6.70 | 3 912 | 457 |
| 256 / 2 / 128 / 32 | 2.00 | 2.84 | 4.12 | 6.72 | 3 899 | 434 |
| 256 / 1 / 256 / 32 | 2.20 | 3.19 | 5.02 | 9.20 | 2 848 | 395 |
| 64 / 1 / 64 / **64** (native) | 1.85 | 2.48 | 3.47 | 5.79 | 4 525 | 470 |
| 128 / 2 / 64 / **64** | 1.91 | 2.45 | 3.57 | 5.70 | 4 599 | 455 |
| 256 / 4 / 64 / **64** | 1.93 | 2.62 | 3.56 | 5.72 | 4 583 | 451 |

Run 2, the same three shipping candidates in one process, after a longer ramp:

| cell | m=1 | m=2 | m=4 | m=8 |
|---|---:|---:|---:|---:|
| 64 / 2 / 32 / 32 | 1.84 | 2.26 | 3.24 | 5.49 |
| 128 / 4 / 32 / 32 | 1.87 | 2.24 | 3.23 | 5.64 |
| **256 / 8 / 32 / 32** | 1.86 | 2.25 | 3.27 | **5.41** |

**The rank order flips between runs**: run 1 puts `128/4` 14 % ahead of `256/8` at m=1, run 2 puts all
three within **2 %** (1.84 / 1.87 / 1.86). The honest reading is `06`'s: **all three are the same speed.**

* **Geometry is not a lever — confirmed on a second harness and a second quant.** Every fully occupied
  cell from 64 to 256 threads sits in a **±5 %** band; `ls256/r8` (what we ship) is best-or-tied at
  m=4/m=8. Lane counts per row **> 32** (128/1, 256/1) are monotonically worse (up to +70 % at m=8):
  the extra cross-subgroup LDS reduction plus a 10-deep dependent block chain per row is not hidden by
  more lanes. Same conclusion as `06` §1 (Q4_K) from a different decode.
* **Forced subgroup 32 vs native 64: flat** (`64/1/64`: 1.84/2.23/3.31/5.85 with sg 32 vs
  1.85/2.48/3.47/5.79 with sg 64; `128/2/64`: 1.85/2.55/3.56/5.81 vs 1.91/2.45/3.57/5.70). No consistent
  sign, worst cell 4 %. `06` §1 measured forced-32 as 5–8 % *slower* on Q4_K. Operationally the same
  answer: **`VK_EXT_subgroup_size_control` is not a lever here**, and the RX's default is already 64.
* **m=8 buys 2.9–3.1× over eight m=1 calls** (5.41 vs 8 × 1.86 = 14.9 µs). Priced as a stage:
  gate+up+down at m=8 ≈ 5.4 + 5.4 + 9.1 = **20 µs/block = 0.95 ms/token over 48 blocks**, i.e. **~1 %**
  of the 92 ms/token prefill. The shared expert's 15.4–55.3 ms in the trace is therefore **not kernel
  work**, and no shape/occupancy change can move it.

## 2. The barrier price is a **cache-invalidation** price, and it scales with `bpr`, not grid size

`06` §8 lists this as its first open question: *"why a dependent 2-dispatch chain costs 4× the sum of its
parts … candidates: wave-drain per dispatch at 80 workgroups on 60 CUs, or the descriptor-set rebind …
worth one measurement with a 3 200-row shape (320 WGs)"*. That measurement — plus the `bpr` axis, which
turns out to be the one that matters — was run here.

Dependent chain: `iters` dispatches in one command buffer with **one `COMPUTE→COMPUTE` sync2 barrier**
after each, so dispatch *k+1* really depends on *k*. µs **per dispatch**, min of 3, 20 000-iteration
chains, IQ4_XS, m=1:

| rows | K | bpr | workgroups | chain µs/dispatch | no-barrier baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 2560 | 10 | 1 | 7.40 | 0.01 |
| 8 | 2560 | 10 | 4 | 5.63 | 0.02 |
| 160 | 2560 | 10 | 80 | 5.56 | 0.41 |
| 640 | 2560 | 10 | 320 | 5.84 | 1.71 |
| 640 | **640** | **3** | 320 | **2.48** | 0.60 |
| 640 | **512** | **2** | 320 | **1.62** | 0.38 |
| 1 | 256 (trivial kernel) | 1 | 1 | 1.15 | 0.036 |

**Reading**

* **Grid size is irrelevant**: 1 → 4 → 80 → 320 workgroups is flat (7.40 / 5.63 / 5.56 / 5.84 µs). The
  wave-drain hypothesis is *refuted* — at 320 workgroups there is no drain to amortise and no extra cost
  either.
* **`bpr` is the axis**: `bpr=10 → 5.6 µs`, `bpr=3 → 2.5 µs`, `bpr=2 → 1.6 µs`, `bpr=1 → 1.15 µs`.
  Fits **1.0–1.15 µs + ≈0.45 µs per dependent block**. That is the signature of a **cold first touch
  per block**: the barrier's acquire invalidates the caches, so the next dispatch re-fetches its weight
  blocks from L2/DRAM and each of the `bpr` dependent block loads is a round trip. It is *not* the
  access-mask scope and *not* the descriptor rebind (~20 ns, round 4).
* **Barrier *form* is irrelevant on RADV**: `COMPUTE→COMPUTE` with `write→read`, with
  `read|write→read|write`, and the legacy `ALL_COMMANDS→ALL_COMMANDS` memory barrier measure
  **5.624 / 5.624 / 5.624 µs** (0.0 % apart); on the trivial kernel all three are 1.150 µs. So the
  narrowing that round 2 credited with −2.5…−6.2 ms/token does **not** reproduce at this working set —
  the cost is the invalidation, which every legal form performs. `06` §3's own table shows v1 = 1 749 vs
  v2 = 1 116 ns on a 256 KiB set; the two harnesses differ in working set, not in mechanism, and both
  agree that `write→read` == `read|write`.
* **The no-barrier baseline is not a kernel duration.** 20 000 identical dispatches with no barrier
  measure **0.021 µs** each (disjoint resources and aliased resources alike) — they overlap almost
  completely. So "barrier cost = chain − baseline" is meaningless, and any throughput number taken
  without a dependency understates a real graph leg ~3× (`06` §0.2 says 4×; same trap, same cause).

**What this changes**

1. **Barrier pruning is worth ~4× what `06` §3 prices.** With the real `bpr=10` shared-expert kernels a
   removed dependent barrier is worth **≈5.6 µs**, not 1.07 µs: 1 509 → ~400 barriers ⇒ **−4.5 ms/token**
   (≈5 % of the GPU half). It is worth *more* on `bpr=10` kernels than on `bpr=3` ones, so prune the
   gate/up/HC-read/GDN dispatches first, not `down`/`mixer`.
2. **Tracked barriers (ncnn's pattern, `06` §3 / §5.1.1) remain the right mechanism**, and now with a
   second, measurable benefit: a *skipped* barrier also skips the invalidation, so the next dispatch
   keeps L2-resident weights. That is the same term `UNROLL=2` attacks (§4) — measure them together.
3. **`06` §8's remaining 4× mystery is answered**: the dependent chain exposes the **per-dispatch latency
   that pipelining hides** — fixed floor ≈1.1 µs (barrier + launch) plus ≈0.45 µs per dependent block
   load — and it does not depend on occupancy, so a 3 200-row shape does not help.

## 3. Crash safety — what this harness did *not* run, and why that is the finding

`06` §4.3/§4.5 already records the reset forensics (ring timeout with one outstanding packet,
`vkWaitForFences(u64::MAX)`, BACO reset on the display card) and that `VK_EXT_robustness2` is present
and costs ≤1 % on the real GEMM. This harness was built with a deliberate out-of-bounds probe
(`shaders/oob.comp`: one invocation reads `X[32000]` and writes `O[32000]` past two small buffers) and
**ran only the `robustBufferAccess2 = true` arm.**

* Verified present on all three devices with `vulkaninfo` on this host (not inferred): `VK_EXT_robustness2`
  (`nullDescriptor = true`), `VK_KHR_robustness2`, `VK_KHR_maintenance7`, `VK_EXT_subgroup_size_control`
  (rev 2), `VK_KHR_compute_shader_derivatives`, `VK_KHR_shader_subgroup_uniform_control_flow`,
  `VK_EXT_memory_priority`. Mesa 26.2.4 / RADV, Vulkan 1.4.354.
* **The un-robust arm was deliberately not run.** The one prior RX-side fault (2026-10-25, illegal
  opcode) *hung the machine*, and AGENTS.md makes any reset a stop-the-line event. A causation
  experiment that can wedge the box — to convert "we believe OOB faults" into "OOB faults" — is not
  worth it when the mitigation can be turned on and measured *without* a fault (`06` did: ≤1 %), and the
  fault class is already established by two incidents.
* Testing order for any new kernel, in priority order: **build-time `spirv-val`** (cheap, no device
  needed) → **`robustBufferAccess2 = true` in the probe harness** (OOB becomes a clamp/zero) →
  **bounded fence waits, never `u64::MAX`** (a hang becomes a `VkResult`) → **`timeout` around every
  process** → only then performance. Note the two bounds differ: `spirv-val`/validation cannot catch a
  computed index that is simply wrong, which is exactly what `robustness2` covers.

## 4. What this arm adds to the top-3 (it does not replace `06` §7)

| # | Item | Delta vs `06` |
|---|---|---|
| 1 | Resource-tracked barriers in `dispatch_batch` | **repriced**: −0.7…−1.0 → **≈−4.5 ms/token**, and prune `bpr=10` dispatches first (§2) |
| 2 | Fused gate+up+SwiGLU · `UNROLL=2` | unchanged; both now have one mechanism (fewer dependent round trips), so measure them **together** |
| 3 | Shape/occupancy re-tuning | **do not start it**: two harnesses, two quants, flat within 5 % (§1) |

## 5. Reproduction (harness kept out of the repo, per AGENTS.md)

```
H=/var/tmp/users/korphaus/tmp/opencode/research-harnesses/vk-gemm-shapes   # standalone ash crate
cd $H && cargo build --release
./target/release/vk-gemm-shapes sweep   --pci 0000:03:00.0 \
    --cells wg256_r8_l32_s32,wg128_r4_l32_s32,wg64_r2_l32_s32,wg64_r1_l64_s64   # <=4 shapes/process
./target/release/vk-gemm-shapes barrier --pci 0000:03:00.0                      # + bpr scaling law
SWEEP_DEBUG=1 ./target/release/vk-gemm-shapes sweep --pci 0000:03:00.0 --cells wg256_r8_l32_s32
```

The binary refuses more than four fresh pipelines per invocation, prints the measured `sclk` at start and
end, and takes min-of-5 batches of ~20 ms after a ~1.2 s clock ramp. `shaders/sweep_gemm.comp` is the
parameterised GEMM (`-DWG_SIZE -DLPR -DROWS_PER_WG -DSG`); `shaders/oob.comp` is the OOB probe (§3,
robust arm only). Raw transcripts: `results/shapes-iq4xs.txt`, `results/barrier.txt`
(`/var/tmp/users/korphaus/tmp/opencode/research-harnesses/vk-gemm-shapes/results/`).
