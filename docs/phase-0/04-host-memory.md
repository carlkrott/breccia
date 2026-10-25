# Phase 0.5 — Host memory & Tier-2/3 reality check

Constraints that the plan's optimistic "30 GB pinned pool" numbers must respect, measured 2026-10-25.

## Measurements

| Item | Value | Source |
|---|---|---|
| RAM | 58.4 GB total; **43 GB MemAvailable idle** | `free -h` |
| MemAvailable under load (qwen-dual running) | **7–9 GB** | observed during 0.1 window |
| Swap | zram0 58.4 GB (zstd), **20.5 GB data → 7.8 GB physical** (2.6:1); idle si/so at rest | `zramctl`, `vmstat` |
| `ulimit -l` (mlock) | **8192 KB = 8 MiB** for korphaus | `ulimit -l` |
| cgroup user.slice limits | `memory.max=max`, `memory.swap.max=max` | sysfs |
| NUMA nodes | **1** (no NUMA partitioning needed) | `lscpu` |
| io_uring | **available** — `io_uring_setup` OK, `features=0x3ffff`, liburing 2.15 headers | syscall probe |
| max_map_count | 1,048,576 | procfs |
| Tier-3 candidate A | `/mnt/agent-storage-a` (sdb1, 916 GB): **673 GB free**, O_DIRECT write 375 MB/s / read 349 MB/s (dd, 128 MiB) — kv-slots already live here | dd oflag/iflag=direct |
| Tier-3 candidate B | `/mnt/agent-storage-b` (sdc1): 431 GB free | df |
| System NVMe `/srv`,`/home` | 121 GB free, 85% used, `/srv` root-owned (not writable; `/srv/strata-data` is) | df, touch test |

## Findings

1. **mlock is the hard blocker for Tier-2.** 8 MiB default lock limit → any pinned-RAM pool must run with `LimitMEMLOCK=infinity` (Strata's unit already does this — precedent: `/etc/systemd/system/strata-rx.service` drop-in) or an explicit `setrlimit` at startup. breccia-profile should *probe and report* the effective limit instead of assuming.
2. **gpu-manager evicts GPU tenants when MemAvailable < 4 GB** (observed: it stopped an idle qwen-dual at 3002 MiB). A Tier-2 pool of N GB counts against MemAvailable as it's pinned — so **pin budget must stay coordinated with that watermark**: measured headroom pattern = 43 GB idle − ~33 GB model/file-cache footprint ≈ 7–9 GB *while a model runs*. Realistic Tier-2 pool under load: **~16–24 GB**, not 30+, unless breccia itself is registered as a gpu-manager tenant with its own budget.
3. **Swap (zram) is resident, not thrashing**: 20.5 GB of cold pages sit compressed at rest with zero si/so — normal here, but a Tier-2 pool allocation should be sized against `MemAvailable`, not `MemTotal`.
4. **Tier-3 belongs on `/mnt/agent-storage-a`**: 673 GB free, O_DIRECT confirmed (~350 MB/s sequential; 4K-random to be measured with the io_uring harness in Phase 1), io_uring fully supported. System disk is too full (85%) for any cache growth.
5. Single NUMA node — the hwloc/NUMA probe in `breccia-profile` can stay simple on this host (still worth emitting for portability).

## Sizing recommendation (feeds `hw_profile.json` defaults)

```
Tier 1 (VRAM):   16+16 GB, minus ~1.5 GB display/composition reserve on the RX
Tier 2 (pinned): start 16 GB, watermark-configurable up to 24 GB;
                 requires LimitMEMLOCK=infinity; trigger eviction review at MemAvailable < 6 GB
Tier 3 (SSD):    /mnt/agent-storage-a, cap 100 GB for breccia cold KV/experts (673 GB free)
```
