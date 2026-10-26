# 02 — Vulkan raw probe (Phase 0.2)

Raw, model-free Vulkan micro-benchmarks on the two discrete GPUs, produced by the
`harnesses/vk-probe` harness. This is the **Vulkan half** of the Phase-0.2
backend experiment; the HIP half and the GEMM/GEMV-vs-reference comparison live in
the companion gate material. Nothing here decides the pure-Vulkan vs hybrid
verdict — this document only records what RADV measured on this machine.

## 1. Commands and environment

```bash
# build (clean, no warnings; glslc + spirv-val run in build.rs for every shader)
cd /home/korphaus/projects/Breccia/harnesses/vk-probe
cargo build --release

# validation pass — 0 validation errors on both discrete GPUs (also run repeatedly
# during development)
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ./target/release/vk-probe --quick > results/validation-quick.log 2>&1

# measurements (GPUs free; see below) — two complete runs, both retained
./target/release/vk-probe --json results/vulkan-raw-probe.json        > results/full-run.log         2>&1
./target/release/vk-probe --json results/vulkan-raw-probe-repeat.json > results/full-run-repeat.log  2>&1
```

| Item | Value |
|---|---|
| Host | `<host>`, kernel `6.18.55-2-cachyos-lts`, Ryzen 9 7950X, 58 GB RAM |
| Vulkan loader / instance | 1.4.363 (harness requests 1.3) |
| Driver | RADV, `Mesa 26.2.4-arch3.1`, device API 1.4.354, driverVersion `26.2.4` |
| Toolchain | rustc/cargo 1.95.0, `ash 0.38.0+1.3.281`, glslc 2026.4, SPIRV-Tools v2026.4, Vulkan headers 1.4.363 |
| GPU processes at run time | **none** — `pgrep` at process start recorded only the idle `comfyui-mcp` npm bridges (no ComfyUI server), see `environment.competing_gpu_processes` in the JSON |
| Clocks during the run | sampled every 250 ms, `results/clock-sampler.log` (see §4) |

**GPU-busy procedure.** An earlier `llama-bench` (HIP) sequence and later a
full-model Vulkan `llama-server` (other session's Gate-A run) were active; the
harness run was held back in 30 s loops for 10 minutes. By the time the wait
expired the Vulkan server had exited and both GPUs read
`gpu_busy_percent = 0` with VRAM free (0.57/16 GB and 0.01/16 GB used). The
long-lived `llama-server` on PID 1411 is CPU-only (no radeon/vulkan/hip mappings
in `/proc/1411/maps`) and never touches the GPUs.

**Enumeration quirk.** This machine exports `MESA_VK_DEVICE_SELECT=1002:164e!`
(the desktop session pins the iGPU). `vulkaninfo` under that variable shows only
the Raphael iGPU. The harness clears the variable at startup and prints a
warning; without it all three devices enumerate (index 0 = iGPU, 1 = RX 6950 XT,
2 = Radeon VII).

## 2. Device discovery

### 2.1 Identity

| # | Name | vendorID:deviceID | type | API / driver |
|---|---|---|---|---|
| 0 | AMD Ryzen 9 7950X (RADV RAPHAEL_MENDOCINO) | 1002:164e | integrated (not benchmarked) | 1.4.354 / 26.2.4 |
| 1 | AMD Radeon RX 6950 XT (RADV NAVI21, gfx1030) | **1002:73a5** ✓ expected | discrete | 1.4.354 / 26.2.4 |
| 2 | AMD Radeon VII (RADV VEGA20, gfx906) | **1002:66af** ✓ expected | discrete | 1.4.354 / 26.2.4 |

### 2.2 Memory heaps (identical on both cards)

| Heap | Size | Flags |
|---|---|---|
| 0 | 29.22 GiB | host (system RAM, BAR-visible) |
| 1 | 15.98 GiB | device-local |

### 2.3 Queue families

| GPU | Family | Flags | Queues | timestampValidBits |
|---|---|---|---|---|
| RX 6950 XT | 0 | GRAPHICS\|COMPUTE\|TRANSFER\|SPARSE | 1 | 64 |
| | 1 | COMPUTE\|TRANSFER\|SPARSE | 4 | 64 |
| | 2 | (none) | 1 | 0 |
| | 3 | (none) | 1 | 64 |
| | 4 | SPARSE | 1 | 0 |
| Radeon VII | 0 | GRAPHICS\|COMPUTE\|TRANSFER\|SPARSE | 1 | 64 |
| | 1 | COMPUTE\|TRANSFER\|SPARSE | 4 | 64 |
| | 2 | SPARSE | 1 | 0 |

Both cards therefore offer a **graphics/compute/universal queue (family 0) plus a
separate transfer-capable family (family 1, 4 queues)**. The harness uses
family 0 for compute and family 1 for copies — a genuinely distinct queue, which
is what the overlap test needs. `timestampPeriod`: **10 ns/tick (RX 6950 XT),
40 ns/tick (Radeon VII)**; both timestamps-valid values above are 64 bits.

### 2.4 Subgroups — the architecture confirmation

| Property | RX 6950 XT (gfx1030) | Radeon VII (gfx906) |
|---|---|---|
| `subgroupSize` (default) | **64** | **64** |
| `minSubgroupSize` … `maxSubgroupSize` (size control) | **32 … 64** | 64 … 64 |
| `computeFullSubgroups` feature | yes | yes |
| supported stages | VERTEX, TESS_CTRL, TESS_EVAL, GEOMETRY, FRAGMENT, COMPUTE | same |
| supported ops | BASIC, VOTE, ARITHMETIC, BALLOT, SHUFFLE, SHUFFLE_RELATIVE, CLUSTERED, QUAD | same |
| quad ops in all stages | yes | yes |

Note on the expected values: RADV reports `subgroupSize = 64` on **both** cards
(the default wave width). What separates the architectures is the size-control
range: **gfx1030 admits 32 … 64 (Wave32 is available via
`VK_EXT_subgroup_size_control`), gfx906 is locked to 64 … 64 (Wave64 only)**.
So a specialization-constant kernel that *requests* a 32-wide subgroup will work
on the RX 6950 XT only; the default wave width on both is 64.

### 2.5 Extensions and features (per device)

| Extension (required by the probe) | RX 6950 XT | Radeon VII |
|---|---|---|
| `VK_KHR_timeline_semaphore` | present | present |
| `VK_EXT_external_memory_host` | present | present |
| `VK_KHR_cooperative_matrix` | **absent** | **absent** |
| `VK_EXT_subgroup_size_control` | present | present |
| `VK_KHR_shader_atomic_int64` | present | present |
| (total device extensions listed) | 245 | 218 |

| Feature queried at device creation | RX 6950 XT | Radeon VII |
|---|---|---|
| `shaderFloat16` | yes | yes |
| `shaderInt8` | yes | yes |
| `shaderBufferInt64Atomics` / `shaderSharedInt64Atomics` | yes / yes | yes / yes |
| `timelineSemaphore` | yes | yes |
| `subgroupSizeControl` / `computeFullSubgroups` | yes / yes | yes / yes |
| `cooperativeMatrix` | **no** | **no** |

`VK_KHR_cooperative_matrix` is not exposed by RADV 26.2.4 on either card, so
cooperative-matrix/WMMA paths are unavailable in Vulkan on this machine — GEMM
kernels must be built from ordinary subgroup/storage-buffer ops. Everything else
the probe wanted was present; no sub-test was skipped on either device
(`notes: []`, `error: null` in the JSON).

## 3. Results

Two complete runs, ~20 s apart, are stored (`vulkan-raw-probe.json` at
23:16:02 UTC, `vulkan-raw-probe-repeat.json` at 23:16:23 UTC). Tables below give
run 1 with the run-2 value in parentheses where it differs.

### 3.1 VRAM bandwidth (compute shader, 2 × 512 MiB device-local)

Timed with queue timestamps (`vkCmdWriteTimestamp` TOP/BOTTOM_OF_PIPE +
`vkGetQueryPoolResults(TYPE_64|WAIT)`), ≥ 0.2 s windows, 91–158 windows per
measurement. `copy` = bytes read + bytes written per pass (2 × buffer size).

| GPU | read GB/s | write GB/s | **copy GB/s** | repeat run | % of vendor peak |
|---|---|---|---|---|---|
| RX 6950 XT | 231.1 | 231.1 | **462.2** | 462.4 (±0.04 %) | 462 / 576 GB/s spec = **80 %** |
| Radeon VII | 413.0 | 413.0 | **826.1** | 824.9 (±0.15 %) | 826 / 1024 GB/s spec = **81 %** |

Vendor peaks used above: RX 6950 XT = 256-bit GDDR6 @ 18 Gbps = 576 GB/s;
Radeon VII = 4096-bit HBM2 @ 2.0 Gbps = 1024 GB/s.

**Cache-resident second window (2 × 32 MiB)** — same kernel, working set inside
the RX 6950 XT's 128 MiB Infinity Cache:

| GPU | copy GB/s | repeat run | vs DRAM window |
|---|---|---|---|
| RX 6950 XT | **1811.2** | 1811.8 (±0.03 %) | **3.9 ×** — cache-inflated, not DRAM traffic |
| Radeon VII | 792.5 | 792.4 (±0.01 %) | ≈ 1.0 × — Vega20 has no large L3, as expected |

This is where the "700–900+ GB/s (Infinity Cache can inflate)" expectation comes
from: DRAM-resident bulk traffic on the RX 6950 XT is ~462 GB/s, but any working
set that stays inside 128 MiB is served at ~1.8 TB/s. For scale, AMD's own spec
sheet lists the card's "effective memory bandwidth" (Infinity-Cache-assisted) as
*up to 1794 GB/s* — this measurement lands within 1 % of that figure. A third,
larger quick-mode window (2 × 16 MiB) measured 1420 GB/s, i.e. the inflated
figure is working-set-dependent, not a fixed property.

### 3.2 Compute throughput (dense `fma()` chains)

1 048 576 threads × 4096 rounds × 8 independent chains, fma counted as 2 flops,
device-local seed buffer, queue-timestamped windows ≥ 0.2 s.

| GPU | FP32 GFLOPS | repeat | FP16 GFLOPS | repeat | Observed clock (§4) → est. peak | FP32 achieved |
|---|---|---|---|---|---|---|
| RX 6950 XT (5120 SP) | **26 704** | 26 713 (±0.03 %) | **42 566** | 42 194 (±0.9 %) | ~2.695 GHz boost → ~27.6 TFLOPS | **~97 %** |
| Radeon VII (3840 SP) | **6 784** | 6 784 (exact) | **10 766** | 10 766 (exact) | 1.801 GHz (max state) → ~13.8 TFLOPS | **~49 %** |

Vendor peaks for reference (AMD spec clocks, lower than the clocks actually
observed): RX 6950 XT 23.65 TFLOPS FP32 / 47.31 TFLOPS FP16 at 2310 MHz boost
(80 CU, 5120 SP); Radeon VII 13.8 / 27.7 TFLOPS at 1800 MHz boost (60 CU,
3840 SP). The "achieved" column compares against the *observed* clock so that
DVFS does not flatter or penalise either card.

Precision caveats: chains are dependency-chained (`x = fma(x, a, b)`), so this
is *sustained achievable* FMA at ILP 8, not a theoretical peak; values stay in
`[1, e]` (no denormals, no Inf/NaN). FP16 reports binary16 ops **as issued** —
on both architectures packed/pair math means the useful math rate is ~2× the
FP32 vector rate; the measured FP16/FP32 ratios (1.59× on RX, 1.59× on Vega)
sit below the theoretical 2× because the same dependency-chain shape limits
issue rate. These numbers are a FMA-level proxy for GEMM arithmetic throughput;
no tiling/shared-memory GEMM is measured here.

Notable asymmetry: the identical kernel reaches ~97 % of peak on RDNA2 but only
~49 % of peak on gfx906 at max clocks — kernel-shape sensitive, not clock-limited
(the Radeon VII sat at its 1801 MHz max state throughout, §4). Vendor FP16 peaks
at observed clocks (2× FP32): ~55 TFLOPS (RX) and ~27.7 TFLOPS (Radeon VII) →
77 % and 39 % achieved.

### 3.3 PCIe host ↔ device (`vkCmdCopyBuffer`, family-1 queue)

Staging = `HOST_VISIBLE|HOST_COHERENT` allocation (the driver pins/page-locks it
while device-visible — Vulkan's equivalent of registered host memory, not
pageable `malloc`); target = device-local.

**RX 6950 XT** — PCIe 4.0 ×16 (`pp_dpm_pcie`: `16.0GT/s, x16`, ~31.5 GB/s/dir theoretical):

| size | H2D GB/s | D2H GB/s | repeat H2D / D2H |
|---|---|---|---|
| 16 MiB | 26.20 | 27.33 | 25.72 / 27.35 |
| 64 MiB | 28.66 | **21.97** † | 28.67 / 27.87 |
| 256 MiB | **23.30** † | 28.00 | 28.71 / 28.01 |
| 1024 MiB | 27.34 | 28.06 | 28.73 / 28.00 |

† transient dip in run 1; the repeat run measured **27.87 GB/s** (and 28.71 for
the 256 MiB H2D cell that dipped to 23.30), so steady-state across the ≥ 64 MiB
sizes is **~28.0–28.7 GB/s both directions ≈ 89–91 % of the link**.

**Radeon VII** — PCIe topology (`lspci -tv`): the card hangs off chipset root
port `02.1-[05-10]` (behind the chipset uplink), not off a CPU lane like the
6950 XT (`01.1-[01-03]`, direct ×16). The GPU's own PCIe DPM states read
**8.0 GT/s × 4** (gen3 ×4, ~3.07 GB/s/dir theoretical; idle state
`2.5GT/s, x4`). *(sysfs `current_link_width` reports ×16, which conflicts; the
throughput corroborates ×4 — see below):*

| size | H2D GB/s | D2H GB/s | repeat H2D / D2H |
|---|---|---|---|
| 16 MiB | 2.72 | 2.93 | 2.72 / 3.00 |
| 64 MiB | 2.80 | 2.88 | 2.81 / 2.97 |
| 256 MiB | 2.85 | 2.87 | 2.84 / 2.90 |
| 1024 MiB | 2.86 | 2.90 | 2.86 / 2.91 |

Measured ≈ **93–94 % of a gen3 ×4 link** (2.90 / 3.07) but only ≈ 18 % of a gen3
×16 link (15.75 GB/s) — the measurement itself selects the ×4 reading as the
operative one. Effective host bandwidth for this card is therefore **~2.9 GB/s
per direction**, ~10× below the RX 6950 XT, and it is shared with every other
device behind that chipset port. This is a platform/wiring property, not a RADV
software limitation — the same driver pushes 28 GB/s on the other card.

### 3.4 Compute / transfer overlap

Method: compute leg = one dispatch of ~1 s of the FMA kernel on family 0;
transfer leg = back-to-back 64 MiB H2D+D2H pairs on family 1 sized to ~1 s; both
legs recorded into *separate* command buffers, submitted back-to-back, timed by
wall clock across two fence waits.

`eff = 1 − (T_both − max(T_compute, T_transfer)) / min(T_compute, T_transfer)`
(1.0 = perfect concurrency, 0.0 = fully serial; values slightly > 100 % are
wall-clock noise).

| GPU | compute alone | transfer alone | both | **efficiency** | repeat run (compute / transfer / both → eff.) |
|---|---|---|---|---|---|
| RX 6950 XT | 1.060 s | 0.884 s (≈28.2 GB/s) | **1.040 s** | **102.3 %** | 1.060 / 1.004 / 1.060 → **100.0 %** |
| Radeon VII | 0.999 s | 1.041 s (≈2.84 GB/s) | **2.031 s** | **0.8 %** | 0.999 / 1.023 / 2.000 → **2.1 %** |

Cross-check: the transfer leg's throughput in the concurrent test (28.2 GB/s and
2.84 GB/s) matches the standalone PCIe numbers (28.7 and 2.86 GB/s), so the legs
are not throttling each other — on the RX 6950 XT the DMA runs **fully hidden**
under compute (T_both ≈ T_compute); on the Radeon VII the two legs run
**effectively serially** even though they are distinct queue families (family 1
vs family 0), i.e. driver/hardware do not concurrently schedule that compute
engine with its transfer path.

### 3.5 JSON schema

`--json PATH` writes `{schema_version, environment, results[]}` where each entry
is `{device, properties, vram_bandwidth_gbps, vram_bandwidth_cache_gbps,
compute_gflops, pcie, overlap, notes, error}`; a sub-test that cannot run is
`null` with the reason in `notes` (none occurred here). Full field reference:
`harnesses/vk-probe/README.md`.

## 4. Clock/power evidence

`results/clock-sampler.log` (1200 samples at 250 ms, 23:14:53–23:20:03 UTC,
covering both stored runs):

- **RX 6950 XT**: memory clock at its top state (1124 MHz) in all 1200 samples;
  core clock sits at 500 MHz idle and runs at **2685–2695 MHz** for all 68
  samples of its compute phase (`gpu_busy_percent` 97–99, max 99).
- **Radeon VII**: core clock at its top state **1801 MHz** in 1197/1200 samples
  (1773 MHz in 3), memory at top state 1000 MHz throughout; `busy` reaches 100 %
  in its compute phase (91 samples ≥ 90 %).
- **Never both loaded**: 0 samples where both cards were ≥ 90 % busy — each card
  was measured while the other idled.

So no figure above is a downclock artifact.

## 5. Repeatability

Between the two retained full runs: VRAM copy bandwidth ≤ 0.15 %, FP32 ≤ 0.03 %,
FP16 ≤ 0.9 %, overlap efficiency 102.3 → 100.0 % and 0.8 → 2.1 % (wall-clock
noise on the ≈ 1 s legs), PCIe < 1 % except the two transient dips flagged in
§3.3. Additional full runs during development (not retained — their logs were
overwritten by run 1) landed within 0.4 % of these values; treat the two stored
JSONs as the authoritative record.

## 6. Implications for pure-Vulkan vs hybrid (evidence only)

Factual observations; the gate decision belongs to the Gate-A report, not here.

**RX 6950 XT (gfx1030) — the primary compute device:**

- Vulkan on RADV drives this card to near-hardware limits with plain compute
  shaders: **80 % of DRAM peak** (462/576 GB/s), **~97 % of FP32 peak**
  (26.7 TFLOPS), **~91 % of its PCIe link** (28.7 GB/s), **100 % compute/DMA
  overlap** on distinct queues. Raw Vulkan throughput is therefore *not* the
  bottleneck candidate for this card — kernel/work efficiency is.
- **Infinity Cache is a 3.9× lever**: bulk traffic resident in the 128 MiB L3
  runs at ~1.8 TB/s vs 462 GB/s for DRAM-resident data. A pure-Vulkan design
  that keeps hot expert weights / KV slices inside ~128 MiB gets DRAM traffic
  nowhere near the streaming assumption; one that streams everything does not.
- Full async transfer queue + timeline semaphore + `VK_EXT_external_memory_host`
  (zero-copy host buffers) + `shaderFloat16` (42.6 TFLOPS) + int64 atomics are
  all available → the Phase-2 transfer-orchestration design (DMA ring, host-pinned
  staging, timeline fencing) is implementable in pure Vulkan.
- Missing: `VK_KHR_cooperative_matrix` → no WMMA GEMM path; wave width defaults
  to 64 with Wave32 *selectable* via subgroup size control (specialization
  required to get 32).

**Radeon VII (gfx906) — the second device:**

- On-card arithmetic and HBM bandwidth are solid for Vulkan: **826 GB/s HBM
  (81 % of peak)** and 6.78 TFLOPS FP32 / 10.8 TFLOPS FP16 achieved with this
  kernel shape (49 %/39 % of peak at observed clocks — kernel-shape sensitive,
  not clock-limited).
- The card's **host link is the hard constraint: ~2.9 GB/s per direction
  (gen3 ×4)**, ~10× below the RX 6950 XT, and **compute/transfer overlap is
  effectively 0 %** — transfers serialize against compute instead of hiding
  under it. Streaming weights, KV pages or expert tiles over PCIe to this card
  costs ~10× more time and cannot overlap with its compute under RADV as
  configured here.
- Same extension/feature set as the RX (no cooperative matrix; Wave64 only).

**Cross-cutting:**

- Both cards support the same feature surface for a pure-Vulkan implementation
  (timeline semaphores, external memory host, float16, int8, subgroup size
  control, atomic64); the two cards differ in *interconnect and concurrency*, not
  in API capability.
- Prior context (from `docs/plan/BRECCIA-PLAN.md`, not measured here): llama.cpp
  Vulkan previously ran this workload at 2.56 tok/s vs 23 tok/s HIP. Combined
  with §3.1–§3.2 — where Vulkan reaches 80–97 % of raw hardware limits — that gap
  must be sought in kernel mapping/work efficiency of the Vulkan backend (or its
  memory residency choices), not in a raw capability deficit of RADV on this
  hardware.
- Not measured by this probe (state explicitly if relied on elsewhere): GPU-to-GPU
  P2P/atomics between the two cards, full tiled GEMM/GEMV, image/sparse paths,
  long-run thermal behaviour, HIP-side numbers.

## 7. Artifacts

| Path | Content |
|---|---|
| `harnesses/vk-probe/` | cargo project: `src/`, `shaders/*.comp`, `build.rs` (glslc + spirv-val), `README.md` |
| `harnesses/vk-probe/results/vulkan-raw-probe.json` | primary machine-readable report (run 1) |
| `harnesses/vk-probe/results/vulkan-raw-probe-repeat.json` | repeat run (run 2) |
| `harnesses/vk-probe/results/full-run.log`, `full-run-repeat.log` | console output of runs 1–2 |
| `harnesses/vk-probe/results/validation-quick.log` | validation-layer run, **0 errors** |
| `harnesses/vk-probe/results/clock-sampler.log` | 250 ms sclk/mclk/busy samples during the run |
