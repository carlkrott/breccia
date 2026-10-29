# Phase 5 (L3): host expert pool per-byte and sync

Status: premise measured 2026-10-29; pinning, thread count, spin and a combining pool measured at x4b (all no gain at the 30-thread default); no code change (combine patch kept as `p5-combine.patch`).

## Where the pool sits at x4b
x4b = 4 streams (RX, MI50, RX, RX), G1 shared device, concurrent decode, `--threads 30`, binary `breccia-mx` (3524b4c). Lane stats (`BRECCIA_LANE_STATS=1`, report `ls-x4b`, identical prompts: that arm ran without `BRECCIA_BENCH_MIXED_PROMPTS`; the mixed-prompt lane-stats arm `cmbls-x4b` below is close) per run:

| | value |
|---|---|
| decode wall | ~13.5 s (159 steps) |
| pool phase wall | 11.6 s (86 % of decode) |
| phases | 48 336 (76 per stream-token: 38 host blocks x 2 phases) |
| per phase: wall / max worker busy / mean busy | 240 / 186 / 166 us |
| sync share (wall - max busy) / wall | 22.5 % (~54 us per phase) |
| imbalance (max - mean busy) | ~8.6 % of wall |
| pool lock wait per stream | ~3.0 s (x3r9: 0.5-0.7 s) |
| RX busy | 78 % (MI50 47 %) |

At x4b the four streams queue on one pool, so pool phase wall is on the token's critical path (it was not at x1r/x3r9, which is why Phase 1 spin did not move tok/s there). The RX at 78 % caps any CPU-side win at roughly +28 % before the RX is the limit (UNVERIFIED projection).

## Microbench (`bench cpu-experts`, avx2)
- 30 threads: 35.7 GB/s, 23.5 ms/token, 838 MB/token. The live pool moves ~34-36 GB/s, the same.
- Single thread 3.15 GB/s; per type 1t GB/s: IQ4_NL down 3.19, Q2_0 down 2.30, IQ2_S 3.02/3.07, IQ2_XS 3.04/2.92, IQ2_XXS 2.86/**1.97** (up, anomaly), IQ3_S 3.62/3.67, IQ3_XXS 3.96/3.90 (gate/up).
- Scaling, unpinned best ms (GB/s): 4t 69.0 (12.2), 8t 53.0 (15.8), 16t 36.2 (23.2), 24t 28.9 (29.0), 32t 22.6 (37.0). DRAM ceiling ~65-70 GB/s, so 30 threads is not DRAM-bound. The usable per-core rate is ~1.95 GB/s (16 pinned physical cores: 838 MB / 26.9 ms = 31 GB/s), not the 3.15 GB/s single-thread figure, because SMT siblings share a core: 30 threads at 35.7 GB/s is ~1.2 GB/s per thread, ~2.2 GB/s per physical core.
- Pinned (`BRECCIA_CPU_PIN`) best ms: 8t 38.7 (CPUs 0-7) / 39.5 (0-3,8-11), 16t 26.9 (0-15), 30t 23.0, 32t 23.9. Pinning is ~25 % faster at 8-16 threads (scheduler placement under scx_lavd) and equal at 30-32.

## Live levers measured
| Arm (x4b, mixed) | warm tok/s |
|---|---|
| mx-x4b, 30t unpinned (ref) / control re-run | 47.70 / 47.56 |
| pn30: workers 0-29, coord+streams 30,31 | 40.71 |
| pn28: workers 1-15,17-29, coord 0,16,30,31 | 40.45 |
| un28 / un26 / un24 / un32 (unpinned) | 41.95 / 40.62 / 40.00 / 46.48 |

- **Pinning: no.** Confining the coordinator and the four stream threads to 2-4 CPUs costs 15 %; the stream threads do real host work. Live x4b is already at the 30-thread point where the microbench shows no pinning gain.
- **Thread count is a packing cliff, not a scaling curve.** Top-k is 10, so phase 1 has 20 role jobs and phase 2 has 10. `choose_shards` (MAX_SHARDS 4, tolerance 1.15) gives makespan in expert-units:

  | threads | phase 1 | phase 2 |
  |---|---|---|
  | 24 / 26 | 1.0 | 0.5 |
  | 28 | 0.75 | 0.5 |
  | 30 / 32 | 0.667 | 0.333 |

  Predicted before the sweep (32 = 30, 24/26 below 28) and confirmed. At 30 both phases pack perfectly (60 and 30 equal shards), so packing has nothing left to give at the default; a row-granular split (any row partition stays bit-identical: per-row accumulation over experts keeps its order) would only help other thread counts.

## Remaining levers (bit-identical unless marked)
1. **Dispatch/wake (sync 22.5 %).** `run_phase` sends one message per worker in sequence; the last worker starts ~tens of us after the first. Candidates: the Phase 1 spin-then-park patch re-tested at x4b (result below); a single published batch with workers polling an atomic (L1 proper); fewer phases by merging queued requests (below). The coordinator running shards itself gains nothing at 30 threads without re-splitting: shards already pack exactly 2 (phase 1) and 1 (phase 2) per worker.
2. **Imbalance (~8.6 %).** Equal row counts but unequal time per shard (quant types differ by role, memory latency). Not addressed by packing.
3. **Per-byte kernel.** `iq4_nl` (down) runs one FMA dependency chain (`a0`) with `mul` then `fmadd`; a second accumulator or 2-4 row interleave sharing the `x` loads keeps each row's op order, so it can stay bit-identical. IQ2_XXS up at 1.97 GB/s vs gate 2.86 needs a look. VNNI / q8 activations change numerics (logit report, not token parity). Per-byte gains only matter once sync and imbalance stop dominating: at 30 threads each worker runs at ~1.2 GB/s against a 3.15 GB/s single-thread kernel.

## Gates
- Order-preserving changes (dispatch, interleave, packing): token parity per stream vs `mx-x4b`, x1r, x3r9.
- Anything that changes arithmetic order: logit report, not parity.
- Flag default off; pass rule = warm tok/s up on x4b and not down on x1r / x3r9.

## Results
Spin-then-park re-tested at x4b (`expert.rs.p1`, binary `breccia-p5spin`, `--threads 30`, lane stats on, parity True in every arm):

| `BRECCIA_LANE_SPIN_US` | warm tok/s | sync share | max busy s | lock wait s/stream |
|---|---|---|---|---|
| 0 | 47.29 | 22.3 % | 9.0 | 3.0 |
| 50 | 45.51 | 24.7 % | 9.3 | 3.4 |
| 200 | 43.73 | 10.6 % | 10.4 | 3.2 |

FAIL. Spinning halves the sync share at n = 200 but worker busy time rises 15 % and tok/s falls 8 %: 30 workers + 4 stream threads + the coordinator already fill the 32 logical CPUs, so a spinning worker takes an SMT sibling or a stream thread's slot. Any sync fix that burns CPU while idle loses here; the remaining sync lever that costs no CPU is running fewer phases.

Combining pool (`BRECCIA_POOL_COMBINE=1`, `p5-combine.patch`, binary `breccia-cmb`, `matrix-cmb.sh`): per-token callers post their request to a queue on the pool; whoever holds the pool lock drains every queued request (any block, any stream) and runs them as one phase-1 + one phase-2 pass. Each request keeps its own (expert, row) accumulation in router order, so it is bit-identical (unit test with 4 threads x 20 rounds over 4 blocks vs `moe_topk_sequential`; token parity True in every arm).

| Arm (mixed prompts) | warm tok/s | ref |
|---|---|---|
| cmb-x4b | 47.30 | mx-x4b 47.70 |
| cmbls-x4b (lane stats on) | 46.75 | — |
| cmboff-x4b (same binary, flag off) | 46.16 | mx-x4b 47.70 (not a clean control: flag-off `moe_topk` also went through the new request path) |
| cmb-x3r9 | 37.74 | mx-x3r9 39.53 |
| cmb-x1r | 17.12 | g1off-x1r 17.30 |

| Lane stats per run, x4b | ls-x4b (identical prompts) | cmbls-x4b |
|---|---|---|
| phases | 48 336 | 38 500-38 900 |
| pool phase wall s | 11.6 | 11.3 |
| max busy s | 9.0 | 9.1-9.3 |
| sync share | 22.5 % | 18.0-19.0 % |

FAIL. Merging is opportunistic: a holder finds on average ~1.25 requests queued, not the ~3 the estimate assumed, because the streams rarely post at the same moment. Sync share falls 4 points but worker busy rises ~0.2 s, and the phase wall saves ~0.3 s of ~13.5 s decode. Per-stream `lane_s` falls ~0.5 s, but tok/s does not move, and RX busy (rocm-smi samples with any GPU > 20 %, crude) stays within 3 points in every arm, so where that saving goes is unexplained. In combine mode `lock_wait_s` runs from queue push to lock, which includes time another holder spent on this request: not comparable with the flag-off figure. Bigger merges need the streams to arrive together, i.e. 2b lockstep; the patch is input for that, not live code.

