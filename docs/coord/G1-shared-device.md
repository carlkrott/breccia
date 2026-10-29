# G1: two streams on one shared VkDevice (`BRECCIA_SHARED_DEVICE=1`)

Status: built 2026-10-29 (flag off by default); gate results below.

## Today
`bench token --streams N` loads one `Model` per stream (`breccia-cli` main.rs ~1835), even when two streams land on the same card (x3r9: RX, MI50, RX). Each `Model` owns its own `Gpu` (VkDevice), dense weights (3.6 GiB), head, and VRAM expert arena, so the second RX stream duplicates all of it and spills to GTT. `Multi` then runs the streams on scoped threads (multi.rs ~337/~409).

## Measured premise (2026-10-29, p1s50-x3r9, Phase 1 build)
During x3r9 decode the RX (card1, 03:00.0) holds 15.97 of 16.37 GiB VRAM **and 8.56 GiB GTT** (sampled every 0.5 s by `gttsamp.sh`; MI50 card0 11.8 GiB VRAM, 24 MiB GTT). The second RX stream does spill, so the plan's fence target stands. Full-arm trace: `out/gtt-p1s100-x3r9.tsv`. Bench note: the bench arms already leave the RX < 2.5 GiB free (gm pool idle, Qwen-Dual off); G1 should give that back.

## Goal
Streams that share a PCI device share one `Model`; each `Session` keeps only its KV, scratch and staging (~133 MiB, plan estimate, UNVERIFIED). `Session<'m>` already borrows `&'m Model`, so the type system allows it. The blockers are the per-`Gpu` objects that today assume one recording thread per device.

## Shared per-Gpu state that two threads would race on
| Object | Where | Rule broken | Fix |
|---|---|---|---|
| `pool_c` + `cmds_c[1+slot]` (batch ring) | ctx.rs `Gpu`, dispatch_batch.rs `begin` (`gpu.cmds_c[1 + slot]`) | A VkCommandPool is externally synchronised, and the per-thread ring state (`OPEN/NEXT/FENCES`) would hand both threads the same slot | Thread-local per-device pool + `RING` buffers (Changes 2) |
| `cmd_c` one-shot (`record`, `wall_run`, readbacks) | ctx.rs 1439/1576 | Same pool and buffer | Thread-local one-shot buffer per device (Changes 2): `record` and `submit` are separate calls, so a lock would have to span them |
| `queue_c` | `submit_cmd` (ctx.rs ~1458) | vkQueueSubmit is externally synchronised on the queue | Option A: `Mutex<()>` around every `queue_submit`. Option B: give each session its own queue from the 4-queue async-compute family, never index `queue_x_index` (queue_x already lives there). Start with A (smallest); B only if the counters show submit contention |
| `queue_x` + `pool_x`/`cmd_x` | `TransferRing` (transfer.rs:337, own pool at :566, submit :623), `copy_once` (ctx.rs ~1655, `wall_run(Qc::X)`) | Same: queue and one-shot buffer shared | Same queue lock; `cmd_x` thread-local like `cmd_c`. `TransferRing` already has its own pool per instance |
| `cmd_c` via `Staging` | xfer.rs `read_batch`/`write`/`write_regions` call `gpu.record(Qc::C)`; `Staging` is per session but the buffer is not | Shared one-shot buffer, used during decode | Covered by the thread-local one-shot buffer |
| `timer_c` / `probe` | ctx.rs `parts()` (timer_c: `time_once`/`run_timed`, bench-only); dispatch_batch `PROBE_POOL` | — | **Not shared on the decode path**: the `BRECCIA_GPU_TIME` probe pools are already thread-local per ring slot, so the probe stays on (run.sh sets it). `timer_c` is only used by single-thread benches |
| `Kernel` set + `dpool` | — | — | **Not shared**: every kernel lives in the session's `Venue` (`PipelineSet::build` per session, graph.rs:660), so the non-batched `bind_kernel` path writes a per-session set. No fix needed |
| `Timeline` | — | — | **Not used by the engine** (`end_submit(gpu, None)` everywhere), no fix |
| `device_wait_idle` | `Gpu::wait_idle`, `TransferRing::destroy` | Needs every queue of the device externally synchronised | Takes the queue lock |

## Changes (as built)
1. `breccia-vk` ctx.rs: `queue_lock: Mutex<()>` on `Gpu`, taken around every `vkQueueSubmit`/`vkQueueSubmit2` (`submit_cmd`, both `TransferRing` submits) and every `vkDeviceWaitIdle` (`Gpu::wait_idle`, `TransferRing::destroy`; `Gpu::destroy` owns the `Gpu`). Always on: uncontended it is one lock/unlock per submit, and it changes no numerics.
2. Under `BRECCIA_SHARED_DEVICE=1`, command buffers come from **per-thread pools** registered on the `Gpu` (`thread_cmds: Mutex<Vec<ThreadCmds>>`, keyed by `ThreadId`, created on first use): `parts()` returns the thread's one-shot buffer for `Qc::C`/`Qc::X`, and `dispatch_batch::begin` takes `gpu.ring_cmd(slot)` and stores it in the thread-local `Open` entry, so the per-dispatch path reads it from `Open` with no lock. Pools are freed in `Gpu::destroy` after `device_wait_idle` (not a TLS destructor, which could run after the device is gone). Flag off: `ring_cmd` returns `cmds_c[1 + slot]` and `parts` returns `cmd_c`/`cmd_x`, exactly as before. The shared `timer_c`/`timer_x` stay as they are (bench-only `time_*` path).
   shortcut: one registry entry per thread that ever recorded (each `generate` spawns new scoped threads), bounded for the bench; reuse dead entries before a long-running server shares a device.
3. CLI `bench_token_multi`: with the flag, streams on the same `--devices` entry share the first such stream's `Model`; different `--gpu-experts` on one card are rejected. Two streams on one card also require `BRECCIA_MULTI_DECODE_CONCURRENT=1` and `--prefill-chunk > 0`: the lockstep decode and the token-by-token prompt pass interleave the streams on one thread, and the second `dispatch_batch::begin` on that device would fail. `bench_agents` is unchanged (one model per stream; the flag only switches it to per-thread pools).
4. `Multi::new` unchanged (it already shares the expert pool). `Model` is read-only after load (weights, placed experts, arena), so a repeated `&Model` is safe.

## Gates (plan Phase 2a)
- v2r: 2 streams on the RX, 0 GTT spill (`amdgpu` GTT used ~0), parity per session vs the single-stream sequence.
- x3r9 with the flag: aggregate >= 28.27 tok/s (10-28) and >= the Gen4 p0off-x3r9 ~27.9; parity per session.
- Flag off: bit-identical to `6fc5368`.

## Risks
- Two streams' dense phases on one queue serialise on the GPU (they already do today across two VkDevices, via the kernel scheduler), so the win is residency (no spill, room for more experts in VRAM), not GPU overlap.
- RADV's single graphics+compute queue (family 0, count 1) vs the 4-queue async family: if Option A shows lock wait, move sessions to async queues (B).

## Gate results (2026-10-29, Gen4 MI50 66a1, binary breccia-g1; warm = mean of runs 2-3; parity = every stream's 160 tokens of every run identical to the reference arm)
| Arm | Flag | Warm agg tok/s | vs | Parity | Peak RX GTT / VRAM MiB | Fence / lane ms/step (run 1) |
|---|---|---|---|---|---|---|
| x1r | off | 17.30 | p0off-x1r 17.03 | True | 144 / 4638 | — |
| x3r9 | off | 28.13 | p0off-x3r9 27.94 | True | 8561 / 15968 | — |
| v2r (2 streams, RX) | off (2 copies) | 19.54 | — | True vs x1r | 8558 / 15968 | — |
| v2r (2 streams, RX) | **on** | **31.82** | 19.54 (+63%) | True vs x1r | **476** / 12748 | 58.6 / 53.0 |
| x3r9 | **on** | **39.51** | 27.94 (+41%) | True | **563** / 12749 | 97.3 / 81.4 (p0off 181.5 / 73.4) |
| x3r9 (final build, + TransferRing lock) | on | 39.41 | 27.94 | True | — | — |

All gates PASS: flag off is token-identical to 6fc5368; the shared RX copy removes the ~8.5 GiB GTT spill (residual ~0.5 GiB vs 144 MiB on x1r) and halves the fence wait per step; x3r9 clears 28.27 by 11 tok/s. With the flag the RX keeps ~3.2 GiB free during x3r9, inside the >= 2.5 GiB rule the flag-off arms break. Same foreign-VRAM condition (802 MiB on the RX) in every arm. Logs/reports: `/var/tmp/breccia-coord/out/{report,unit,gtt}-g1*`.
