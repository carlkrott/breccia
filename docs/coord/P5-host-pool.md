# Phase 5 (L3): host expert pool per-byte and sync

Status: premise measured 2026-10-29; pinning, thread count and spin measured at x4b (all no gain at the 30-thread default); no code change.

## Where the pool sits at x4b
x4b = 4 streams (RX, MI50, RX, RX), G1 shared device, concurrent decode, mixed prompts, `--threads 30`, binary `breccia-mx` (3524b4c). Lane stats (`BRECCIA_LANE_STATS=1`, report `ls-x4b`) per run:

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
- Scaling, unpinned best ms (GB/s): 4t 69.0 (12.2), 8t 53.0 (15.8), 16t 36.2 (23.2), 24t 28.9 (29.0), 32t 22.6 (37.0). DRAM ceiling ~65-70 GB/s, so 30 threads is neither DRAM- nor compute-bound (30 x 3.15 = 94 GB/s of kernel): the loss is parallel efficiency.
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
1. **Dispatch/wake (sync 22.5 %).** `run_phase` sends one message per worker in sequence; the last worker starts ~tens of us after the first. Candidates: the Phase 1 spin-then-park patch re-tested at x4b (result below); a single published batch with workers polling an atomic (L1 proper); the coordinator taking a share of the shards instead of idling on `recv`.
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

FAIL. Spinning halves the sync share at n = 200 but worker busy time rises 15 % and tok/s falls 8 %: 30 workers + 4 stream threads + the coordinator already fill the 32 logical CPUs, so a spinning worker takes an SMT sibling or a stream thread's slot. Any sync fix that burns CPU while idle loses here; the remaining dispatch idea that does not is the coordinator running a share of the shards itself (it idles on `recv` today).
