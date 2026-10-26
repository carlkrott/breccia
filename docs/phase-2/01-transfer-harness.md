# 01 — Asynchronous transfer harness (Phase 2 / M1b)

Engine-facing transfer layer plus the measurement that decides **"can DMA be hidden
under compute on this hardware?"** (the M3 expert-streaming go/no-go). Everything
below runs through the same API M3/M5 will drive — `breccia-vk::transfer::TransferRing`
— so the numbers describe the engine, not a side harness.

Date: 2026-10-26 · GPUs idle (`pgrep llama-server` empty, `gpu_busy_percent` 0/1 at
start) · no inference service was started or stopped for this run.

## 1. What landed

`crates/breccia-vk` (new `transfer` + `transfer_bench` modules, one new shader):

| Piece | Detail |
|---|---|
| Timeline-semaphore DMA ring | `N` slots × fixed `slot_bytes` device-local buffers + `N` pinned host staging chunks + `N` transfer command buffers on the **dedicated transfer queue family**. Slot reuse is gated by timeline values; no `vkQueueWaitIdle` in steady state. |
| Two timelines, one value per round | `arrive` (transfer queue signals after the copy into a slot) and `release` (compute queue signals after consuming it). A slot may only be rewritten once `release >= v`, so producer and consumer never need a CPU round-trip. |
| Pinned host staging | `HOST_VISIBLE \| HOST_COHERENT`, allocated once, mapped once and kept mapped (the driver pins it while device-visible). `stage()` + `submit_upload()` give the engine a host-slice → slot upload in one call; `submit_copy()` is the pre-staged/pinned-ring path. |
| Compute/transfer overlap primitive | `submit_upload(slot, src, v)` → `submit_compute(slot, v, |dev, cmd| ..)` (waits `arrive>=v`, signals `release>=v`) → `wait(v)` / `wait_release(v)`. Slot buffers are `VK_SHARING_MODE_CONCURRENT` over both families (no ownership transfers per reuse). |
| Stats | `RingStats { submits, consumer_submits, bytes, slot_reuses, stalls, stall_seconds, host_waits, host_wait_seconds }` + `effective_gbps(seconds)`. |
| Capability probe | `VK_EXT_external_memory_host` presence/enablement, `minImportedHostPointerAlignment`, and a **real** import: aligned host allocation → `vkGetMemoryHostPointerPropertiesEXT` → `vkImportMemoryHostPointerInfoEXT` → DMA out of it → read-back verified. |
| Expert-slot churn (M3 shape) | GPU consumer kernel (`shaders/slot_consumer.comp`: strided vec4 read of the whole slot + 4 dependent FMA chains) eats a 6-deep slot ring while the host streams the seeded 4 MiB / 16 MiB blob sequence through it. Consumer work is **calibrated against the measured DMA rate** (two-point: `iters=0` read cost, `iters=1024` slope) so both legs describe the same number of MiB. |
| Fallback | The ring uses whatever `ctx::Gpu` selected: `distinct_transfer_queue_family`, else `second_queue_of_compute_family`, else `same_queue_as_compute`. The mode is printed and stored in JSON, with a note that overlap is then expected to be ~0. No device on this host hits the fallback (see §7). |

Not done here (deliberately): io_uring source feeding (Phase 4), D2H ring, P2P.

## 2. Commands

```bash
# build (release; the CLI is what runs the bench)
cargo build --release -p breccia-cli

# full measurement, both discrete GPUs, JSON to /tmp/transfer.json
./target/release/breccia bench transfer --json /tmp/transfer.json
# repeat run (variance check)
./target/release/breccia bench transfer --json /tmp/transfer-repeat.json

# single device, by cached-profile index (resolved to PCI identity) or PCI address
./target/release/breccia bench transfer --device 1 --quick
./target/release/breccia bench transfer --device 0000:0e:00.0 --quick

# validation layer (both discrete GPUs)
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./target/release/breccia bench transfer --quick
```

`breccia bench transfer` resolves `--device <profile-index|pci>` through the cached
`hw_profile.json`: an index is mapped to that device's **PCI address** first
(`device 1 -> AMD Radeon VII [pci 0000:0e:00.0]`), because Vulkan ordinals are not
stable on this host (docs/phase-1/profile-report.md). Deviceless default = every
discrete GPU. `MESA_VK_DEVICE_SELECT` is cleared inside `breccia-vk` before instance
creation (no warning was emitted in these runs, i.e. it was not set in the
benchmark environment).

Environment: `<host>`, Vulkan loader/instance 1.4.363, RADV Mesa 26.2.4,
RX 6950 XT `0000:03:00.0`, Radeon VII `0000:0e:00.0`. The desktop compositor holds
`renderD*` (chromium/plasmashell/Xwayland); both cards read 0–1 % busy at start and
the two runs agree within noise (§5), so the desktop is not measurably perturbing
the link.

## 3. Ring throughput (H2D, timeline-gated, no `vkQueueWaitIdle`)

Full runs, 4 slots (3 for the 256 MiB cell) on transfer family 1 index 0. `DMA` =
pre-staged pinned chunk → device-local slot (the steady state with an io_uring-fed
pinned ring); `+staging` = the same loop including the host memcpy into the pinned
chunk every round.

**RX 6950 XT (PCIe 4.0 ×16)**

| size | DMA run 1 | DMA run 2 | +staging run 1 | +staging run 2 | phase-0 H2D |
|---|---|---|---|---|---|
| 16 MiB | **28.25 GB/s** | 27.08 GB/s | 28.21 | 28.14 | 26.20 (repeat 25.72) |
| 64 MiB | **28.38 GB/s** | 28.39 GB/s | 21.60 | 21.51 | 28.66 (repeat 28.67) |
| 256 MiB | **28.71 GB/s** | 28.71 GB/s | 19.42 | 19.54 | 23.30 † (repeat 28.71) |

† the phase-0 run-1 dip; its repeat measured 28.71 GB/s, the same as here.

**Radeon VII / MI50 (gen3 ×4 behind the chipset)**

| size | DMA run 1 | DMA run 2 | +staging run 1 | +staging run 2 | phase-0 H2D |
|---|---|---|---|---|---|
| 16 MiB | **2.71 GB/s** | 2.72 GB/s | 2.71 | 2.72 | 2.72 |
| 64 MiB | **2.80 GB/s** | 2.80 GB/s | 2.77 | 2.78 | 2.80 |
| 256 MiB | **2.84 GB/s** | 2.85 GB/s | 2.74 | 2.74 | 2.85 |

The ring reproduces the Phase-0 `vkCmdCopyBuffer` numbers on both cards (RX
28.2–28.7 vs 28.7 GB/s; MI50 2.71–2.85 vs 2.86), i.e. **adding slot gating, two
timeline semaphores and per-slot command buffers costs no measurable bandwidth**.
The one 27.08 GB/s cell (16 MiB, run 2) is the only outlier; run 1 and the
validation run measured 28.2–28.3 GB/s on the same cell.

Two accounting notes from the JSON: the producer blocks on nearly every round
(556/560 stalls, 0.326 s of a 0.333 s window) — that is a 4-deep ring self-throttling
at line rate, not contention — and the `+staging` column at ≥64 MiB (19–22 GB/s) is
the **host memcpy tax**: a 64–256 MiB memcpy no longer fits in L3, so the memcpy,
not PCIe, becomes the limit. That is the argument for the zero-copy import path (§6).

## 4. Churn: expert-slot streaming under a running consumer (the M3 shape)

6 slots × 16 MiB (96 MiB device-local + 96 MiB pinned), 415 rounds, 4108 MiB streamed,
blobs alternating 4 MiB (211) and 16 MiB (204) from a seeded LCG, pipeline depth 6.
Consumer: 262 144 threads, calibrated to ≈ the measured DMA rate (384–400 FMA
iterations per MiB per thread on the RX, 1404 on the MI50 — same *ratio* on both,
i.e. the calibration is device-relative).

| metric | RX 6950 XT | MI50 |
|---|---|---|
| compute leg alone (same blob sequence, slots resident) | 0.1893 s / 0.1972 s | 1.9591 s / 1.9539 s |
| transfer leg alone (ring streaming, no consumer) | 0.1542 / 0.1543 s | 1.5871 / 1.5890 s |
| both, pipelined through the ring | **0.2034 / 0.2108 s** | **2.6240 / 2.6136 s** |
| overlap efficiency (phase-0 formula) | **90.8 % / 91.2 %** | 58.1 % / 58.5 % |
| consumer stall (`stream − compute_only`) | 0.014 / 0.014 s → **0.003 ms per MB** | 0.665 / 0.660 s → **0.154 ms per MB** |
| effective streaming rate during the pipelined run | 21.2 / 20.4 GB/s | 1.64 / 1.65 GB/s |
| full-rate streaming, no consumer | 27.94 / 27.91 GB/s | 2.71 GB/s |
| compute : transfer per blob | 1.23 : 1 | 1.23 : 1 |
| producer stalls in the pipelined run | 409 / 415 rounds, 0.193 s blocked | 409 / 415 rounds, 2.581 s blocked |

`overlap eff = 1 − (T_both − max(T_compute, T_transfer)) / min(T_compute, T_transfer)`
(identical to docs/phase-0/02 §3.4). Consumer stall is the primary number: it is the
time the consumer actually spent waiting for data the hardware could not hide.

**RX 6950 XT — DMA is hidden under compute.** With compute ≈1.2× the DMA leg, the
pipelined run costs 1.07–1.08× the compute-only run: the consumer loses ~3 µs per
MiB streamed, i.e. effectively nothing. The producer is the throttled party (blocked
0.193 s of a 0.203 s window) — the ring, not the consumer, absorbs the flow control.

**Radeon VII — partially hidden, and not enough.** The pipelined run exceeds
compute-only by 0.66 s (0.154 ms per MB, ≈2.4 ms of unhidden DMA per 16 MiB blob):
both legs wait, and the effective streaming rate during compute is 1.64 GB/s of the
2.71 GB/s the link can do alone. Note this is *more* overlap than Phase-0 measured
for the MI50 (0.8–2.1 %): there the compute leg was one ~1 s dispatch against ~1 s
of 64 MiB H2D+D2H pairs, i.e. sustained saturation of both engines; here short
4–16 MiB bursts find gaps between dispatches and partially overlap. The Phase-0
conclusion is not contradicted either way — at 2.8 GB/s per direction and ≤60 %
hiding, this card cannot be an expert-streaming target.

## 5. Repeatability

Run 1 / run 2 (consecutive invocations, same command):

| | RX | MI50 |
|---|---|---|
| ring DMA 16 / 64 / 256 MiB | 28.25 / 28.38 / 28.71 → 27.08 / 28.39 / 28.71 | 2.71 / 2.80 / 2.84 → 2.72 / 2.80 / 2.85 |
| compute-only churn | 0.1893 → 0.1972 s (+4 %) | 1.9591 → 1.9539 s (−0.3 %) |
| overlap | 90.8 → 91.2 % | 58.1 → 58.5 % |
| consumer stall per MB | 0.003 → 0.003 ms | 0.154 → 0.153 ms |
| zero-copy import H2D | 25.80 → 25.77 GB/s | 2.84 → 2.86 GB/s |

Everything except the RX 16 MiB cell and the RX compute leg (≤4 %) is within 1 %.
The `--quick` mode (1024 MiB, depth 4) gives the same verdict from a ~4× shorter run:
RX overlap 87.5–91.4 %, MI50 61.8–63.6 %, stalls 0.00 / 0.13–0.14 ms per MB.

## 6. Capability findings

| Capability | RX 6950 XT | MI50 | Notes |
|---|---|---|---|
| `VK_KHR_timeline_semaphore` (feature) | yes | yes | two timeline semaphores drive every submission; `timeline_semaphore_used = true` in JSON |
| distinct transfer queue family | family 1, **4 queues** | family 1, **4 queues** | copies go to family 1 index 0, compute to family 0 index 0 |
| `VK_EXT_external_memory_host` | listed + enabled | listed + enabled | `HOST_ALLOCATION_EXT` |
| `minImportedHostPointerAlignment` | **4096 B** | **4096 B** | page granularity, so any page-aligned pinned Tier-2 range imports directly |
| importable memory type bits | `0x20` | `0x20` | host-visible type accepted the host pointer |
| zero-copy import proven | **yes, verified** | **yes, verified** | DMA out of the imported range read back byte-identical to the host pattern |
| zero-copy H2D rate | **25.80 / 25.77 GB/s** | 2.84 / 2.86 GB/s | 91 % / 100 % of the staged DMA rate; no host memcpy involved |

Zero-copy import is therefore **implemented and measured**, not just reported: the
harness needs nothing but a 4096-byte-aligned host allocation, so a Tier-2 pinned
ring can be imported once and streamed for the lifetime of the process. The staged
path costs 19–22 GB/s at ≥64 MiB (§3) versus 25.8 GB/s imported, which is the
concrete reason to prefer the import path in M3.

## 7. Validation layer

```bash
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./target/release/breccia bench transfer --quick
```

**0 validation errors and 0 warnings** on both discrete GPUs (both ran the full
quick suite: capability probe, two ring cells, churn). Connection to develop:
the first version produced one error —
`VUID-vkBindBufferMemory-memory-02985` (imported memory bound to a buffer that did
not declare `HOST_ALLOCATION_EXT` in `VkExternalMemoryBufferCreateInfo::handleTypes`);
fixed by declaring the handle type on the imported buffer. The log in §2 is the
clean rerun.

Distinct-transfer-family fallback was not exercisable on this machine: **even the
Raphael iGPU exposes a family 1 (compute|transfer, 4 queues)** under RADV 26.2.4, so
all three devices take the `distinct_transfer_queue_family` path. A spot-check run on
the iGPU (`--device 0000:75:00.0 --quick`: 37.1 GB/s "DMA" over system memory,
overlap −1.1 % from millisecond-scale legs) confirms the ring only consumes
`gpu.queue_x`/`gpu.family_x`, so the `second_queue_of_compute_family` /
`same_queue_as_compute` fallbacks inherited from `ctx` need no ring changes — they
just report their mode and serialize.

## 8. Comparison to Phase-0 (`docs/phase-0/02-vulkan-raw-probe.md`)

| Measurement | Phase 0 | This harness | Verdict |
|---|---|---|---|
| RX H2D peak (64 MiB / 1 GiB copies) | 28.66 / 27.34 GB/s (89–91 % of the link) | 28.25–28.71 GB/s through a gated 3–4 slot ring | reproduced |
| MI50 H2D | 2.80–2.86 GB/s (gen3 ×4) | 2.71–2.85 GB/s | reproduced |
| RX compute/transfer overlap | 100.0–102.3 % | 90.8–91.2 % with a real consumer loop and a calibrated 1.2 : 1 compute leg | same conclusion |
| MI50 overlap | 0.8–2.1 % | 58.1–58.5 % for 4/16 MiB bursts | **different shape, same decision** (§4) |
| timeline semaphores / external memory host | "implementable" (capability only) | used on every submit; import proven + verified + throughput measured | upgraded to measured |

## 9. Implications for M3 (expert streaming) — factual

- **RX 6950 XT is a valid expert-streaming target.** 28.2–28.7 GB/s H2D through the
  ring = 0.57 ms per 16 MiB expert blob, and with a 6-deep ring the consumer stall is
  ~3 µs per MiB (≈0) while compute runs: at a 22 ms/token budget (≈45 tok/s) the link
  could in principle deliver ~38 × 16 MiB blobs/token if nothing else needed it and
  compute per blob kept up.
- **The MI50 must stay a resident-weights device.** Its link delivers 16 MiB in
  5.9 ms and, even with partial hiding, costs the consumer 0.154 ms per MB
  (≈2.4 ms per 16 MiB blob of unhidden time) — i.e. streaming even a handful of
  experts per token would rival or exceed the compute for that layer. This is the
  Phase-0/Gate-A "expert-parallel / disjoint placement by default" conclusion,
  re-derived on the engine's own code path.
- **Ring depth, not bandwidth, is the control knob.** With 6 slots the producer was
  back-pressured on 409/415 rounds on both cards: the ring's `stalls` counter is the
  flow-control signal M3's scheduler should watch, and 96 MiB of device-local slots
  + 96 MiB pinned staging is affordable.
- **Use the zero-copy import for Tier-2 experts.** Imported pinned RAM streams at
  91 % of the staged rate on the RX with no memcpy, while the staged path loses
  25–30 % once the chunk leaves L3. A pinned Tier-2 ring imported once is the
  intended M3 data path; the staging copy is the fallback (and is what makes the
  16 MiB cell look artificially good: 16 MiB still fits in cache).
- **Caveat, stated plainly:** the consumer is a GEMV-like read + FMA proxy, not an
  expert kernel, and its length was calibrated from the measured DMA rate. The
  claim supported here is only "the RX link can stay busy while a compute queue runs
  and the consumer need not wait"; absolute M3 throughput depends on the real
  dequant/GEMV kernels (Phase 3) and the route predictor (Phase 4).

## 10. Artifacts

| Path | Content |
|---|---|
| `crates/breccia-vk/src/transfer.rs` | ring, pinned staging, timeline gating, stats, `VK_EXT_external_memory_host` probe |
| `crates/breccia-vk/src/transfer_bench.rs` | benchmark orchestration, churn, JSON schema (`schema_version = 1`) |
| `crates/breccia-vk/shaders/slot_consumer.comp` | slot consumer (read + calibrated FMA), compiled + `spirv-val`'d by `build.rs` |
| `crates/breccia-cli/src/main.rs` | `breccia bench transfer` (+ profile-index/PCI resolution + human summary) |
| `/tmp/transfer.json`, `/tmp/transfer-repeat.json` | the two full runs behind every table above (`--json` output) |
| `/tmp/bench/validation-quick.log`, `/tmp/bench/full-run*.log` | console logs of the validation and full runs |

Verify with `cargo test --workspace`, `cargo clippy --workspace --all-targets --
-D warnings`, `cargo fmt --check`; regenerate the tables with the §2 commands
(`breccia bench transfer --json /tmp/transfer.json` takes ~25 s for both cards).
