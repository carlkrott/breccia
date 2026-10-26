# Phase 1 — `breccia profile` (M1a) implementation & verification report

**Status: implemented and verified on <host>, 2026-10-26.** Single report file
(consolidates the earlier `01-profile-report.md`, which is removed).

`breccia profile` discovers the hardware, runs the Phase-0.2 raw Vulkan micro-benchmarks,
assigns Gate-A-informed device roles, and caches everything in `hw_profile.json` behind a
BLAKE3 hardware fingerprint. Everything below was executed for real on this machine;
raw logs are listed in §9.

---

## 1. What was built

| Crate | Contents |
|---|---|
| `crates/breccia-vk` | Reusable port of `harnesses/vk-probe/`: `ctx.rs` (instance/device/queue/timestamp plumbing, `MESA_VK_DEVICE_SELECT` guard, `deviceUUID` + `VK_EXT_pci_bus_info` extraction), `bench.rs` (VRAM bandwidth + cache window, FMA fp32/fp16, PCIe H2D/D2H, compute/transfer overlap), `report.rs` (serde types, now with `Deserialize`/`PartialEq`), `probe.rs` (`run_probe`/`run_discovery`), `pci.rs` (stable PCIe topology strings), `build.rs` + `shaders/*.comp` (glslc → spirv-val, unchanged pattern from the harness) |
| `crates/breccia-profile` | `schema.rs` (`hw_profile.json` v1 + ISO-8601), `system.rs` (CPU/NUMA/RAM, **effective mlock**, zram, io_uring, `statvfs` free space, O_DIRECT read bench, display-connector detection), `fingerprint.rs` (BLAKE3), `role.rs` (documented role heuristic), `backend.rs` (**`BackendProbe` trait — the HIP seam**, `VulkanBackend` impl, backend-neutral `DeviceMeasurements`), `plan.rs` (`--dry-run-plan`), `lib.rs` (orchestration: `current_fingerprint`, `calibrate`, cache rules, summary) |
| `crates/breccia-cli` | `breccia profile [--reprofile] [--quick] [--json PATH] [--dry-run-plan] [--print-fingerprint]` plus `serve`/`bench` stubs printing `phase-2` |

Dependency shape honours the workspace gate note: `breccia-profile → breccia-vk` only, and
`breccia-cli → breccia-profile` — **no production engine crate (kernels/format/server/flags) is
pulled into the profiling path**, so a `breccia profile` build never compiles the engine.

Design decisions:

- **Role heuristic** (`role.rs`) is a pure function of *measured* values — `streaming-compute`
  requires steady-state H2D ≥ 10 GB/s **and** overlap ≥ 50 %; high device bandwidth (≥ 600 GB/s
  copy) with a link/overlap below those thresholds ⇒ `static-resident`; integrated ⇒ `display-only`;
  missing measurements ⇒ `unmeasured` (it never guesses). PCI IDs appear only in doc comments
  and tests, never in the decision path.
- **HIP seam**: `BackendProbe { backend(), discover(), probe() }` + `BackendRun` returning
  backend-neutral measurements. Only `VulkanBackend` is implemented (design-only for HIP, per
  scope); a future `HipBackend` fills the same structs without a schema change.
- **Fingerprint**: BLAKE3 over per-GPU `vendor:device` + `deviceUUID` + driver name/version +
  stable PCIe topology string, CPU model, total RAM — nothing state-dependent (no clocks, no
  `MemAvailable`, no free disk, no timestamps, no process list).

## 2. Build / test / lint gate

```bash
cargo build --workspace                 # 0 warnings, 0 errors
cargo test  -p breccia-vk -p breccia-profile -p breccia-cli   # 41 passed, 0 failed
cargo clippy -p breccia-vk -p breccia-profile -p breccia-cli --all-targets   # 0 warnings
cargo fmt -p breccia-vk -p breccia-profile -p breccia-cli -- --check          # clean
```

- `breccia-profile`: **34** unit tests (fingerprint stability/sensitivity/order-independence,
  role heuristic vs phase-0 numbers, schema round-trip, plan sizing, /proc+/sys parsers,
  display-reserve detection, cache rules) + **1** integration test
  (`tests/diag_fingerprint.rs`: stored fingerprint must equal the digest of its own stored inputs).
- `breccia-vk`: **6** tests, including the run-for-real enumeration test
  (`enumerates_three_devices_on_this_host`, asserts `≥ 3` devices and both `1002:73a5` /
  `1002:66af` are present) and `device_select_env_is_cleared_and_reported`.
- **Caveat:** `cargo test --workspace` is currently red **only** in `crates/breccia-flags`
  (9 failures in `tests/launch_surface.rs`) — that crate is another session's in-flight work and
  was not touched here. All three crates owned by this task are green.

## 3. CLI surface (verbatim)

```text
$ breccia --help
Breccia — hybrid HIP/Vulkan inference engine

Usage: breccia <COMMAND>

Commands:
  profile  Hardware discovery + calibration → hw_profile.json (M1a)
  serve    Phase 2: OpenAI-compatible agent server (not implemented yet)
  bench    Phase 2: benchmark harness (not implemented yet)
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

```text
$ breccia profile --help
Hardware discovery + calibration → hw_profile.json (M1a)

Usage: breccia profile [OPTIONS]

Options:
      --reprofile          Ignore a matching cached profile and re-run the calibration
      --quick              Short benchmark windows (CI-speed, noisier numbers)
      --json <PATH>        Write the profile here (default: $XDG_CONFIG_HOME/breccia/hw_profile.json); also the path the cached fast path is read from
      --dry-run-plan       Print the device role table + layer-placement sketch, write nothing
      --print-fingerprint  Print the hardware fingerprint (BLAKE3) and exit
  -h, --help               Print help
  -V, --version            Print version
```

`breccia serve` → `serve: phase-2`, `breccia bench` → `bench: phase-2` (exit 0).

## 4. The required run sequence (real, final code)

| # | Command | Result | Wall time |
|---|---|---|---|
| 1 | `cargo run -p breccia-cli -- profile --quick --json /tmp/hw_profile_test.json` | fresh quick calibration, 3 devices enumerated, wrote `/tmp/hw_profile_test.json` | **3.40 s** |
| 2 | `cargo run -p breccia-cli -- profile` | full calibration, wrote `~/.config/breccia/hw_profile.json` | **13.0 s** (earlier full runs 13.0–13.2 s) |
| 3 | `cargo run -p breccia-cli -- profile --reprofile` | forced full recalibration (numbers in §5) | **13.05 s** |
| 4 | `cargo run -p breccia-cli -- profile` | `cached profile reused (fingerprint match): /home/korphaus/.config/breccia/hw_profile.json` | **0.067 s** (~195× vs a full run) |
| 5 | `cargo run -p breccia-cli -- profile --print-fingerprint` ×2 | identical hash both times | 0.078 s each |
| 6 | `MESA_VK_DEVICE_SELECT='1002:164e!' breccia profile --print-fingerprint` | trap warning printed **and still 3 devices enumerated**; same fingerprint as an unpinned run | — |
| 7 | `breccia profile --dry-run-plan` | role table + placement sketch from the cache, `nothing written`, ends with `cached profile reused` | 0.07 s |

Device enumeration line (every run):

```text
breccia-vk: instance API 1.4.363, 3 device(s) enumerated, benchmarking [0, 1]
```

Index 0 = RX 6950 XT, 1 = Radeon VII, 2 = Raphael iGPU (iGPU listed in `hw_profile.json`,
not benchmarked by default). Discovery detail matches `docs/phase-0/02` exactly: RX 5 queue
families / **245** device extensions, VII 3 families / **218**; subgroup size control
**32…64 (Wave32 selectable) on gfx1030 vs 64…64 (Wave64 only) on gfx906**; timeline_semaphore ✓,
external_memory_host ✓, subgroup_size_control ✓, atomic_int64 ✓, shaderFloat16/Int8 ✓,
**VK_KHR_cooperative_matrix ✗ on both**.

### Environment trap (`MESA_VK_DEVICE_SELECT`)

The pre-restart session carried `MESA_VK_DEVICE_SELECT=1002:164e!`; the run executed there
printed and still enumerated all three devices:

```text
breccia-vk: WARNING: MESA_VK_DEVICE_SELECT=1002:164e! was set — unsetting it so all Vulkan
devices enumerate (it would otherwise hide both discrete GPUs)
breccia-vk: instance API 1.4.363, 3 device(s) enumerated, benchmarking [1, 2] [quick mode]
```

The current session no longer exports the variable (session restart), so the same behaviour was
re-forced explicitly (row 6 above): warning + 3 devices + **identical fingerprint to the
unpinned run**, proving the guard is deterministic. Covered by unit tests in both crates.

## 5. Measured numbers vs `docs/phase-0/02` (full `--reprofile`, final code)

| Metric | RX 6950 XT (phase-0 → now) | Δ | Radeon VII / MI50 (phase-0 → now) | Δ |
|---|---|---|---|---|
| VRAM copy bandwidth | 462.2 → **445.7 GB/s** | −3.6 % | 826.1 → **825.6 GB/s** | −0.1 % |
| Cache-resident window (2×32 MiB) | 1811 → **1832 GB/s** (3.9× DRAM) | +1.2 % | 792.5 → **793 GB/s** (1.0×) | +0.1 % |
| PCIe H2D (≥64 MiB steady) | ~28.0–28.7 → **28.5–28.8 GB/s** | ≈0 | ~2.86–2.90 → **2.80–2.86 GB/s** | ≈0 |
| PCIe D2H | ~28.0 → **28.3–28.4 GB/s** | ≈0 | ~2.87 → **2.86–2.88 GB/s** | ≈0 |
| FP32 FMA | 26 704 → **24 099 GFLOPS** | **−9.7 %** | 6 784 → **6 754 GFLOPS** | −0.4 % |
| FP16 FMA | 42 566 → **33 338 GFLOPS** | **−21.7 %** | 10 766 → **10 744 GFLOPS** | −0.2 % |
| Overlap efficiency | 102.3 / 100.0 → **100.5 %** | ✓ | 0.8 / 2.1 → **2.7 %** (≈0) | ✓ |
| Sustained clocks during compute | 2685–2695 MHz → **2685–2695 MHz** (sampler) | ✓ | 1801 MHz max state | ✓ |

Every metric required by the gate (`~462 GB/s`, `~26.7 TF`, `~28 GB/s`, `~100 %` overlap;
`~826`, `~6.8`, `~2.9`, `~0 %`) lands **within ~10 %**; DRAM/PCIe/overlap on both cards are
within a few percent. Quick mode (3.4 s): RX 454 GB/s / FP32 25 362 / PCIe 28.7 / overlap 100 %;
MI50 820 GB/s / FP32 6 318 / PCIe 2.7 / overlap 14 %.

Clock sampling during `--reprofile` (`clock-sampler-reprofile.log`, 160 × 250 ms samples) shows
the RX at boost (2685/2690/2695 MHz) for every ≥90 %-busy sample and **memory clock pinned at its
top state (1124 MHz) in all 160 samples**, so the FP32/FP16 shortfall below is not a memory-clock
or a downclock artifact — see §7.

## 6. Fingerprint, roles, plan

- **Value (final):** `0a3b79746aecb569cede7925651791f4edca8b4f0bdb4ec4c0b1e07edf4686dc`
- Stable across `--quick`, full, `--reprofile`, cached and pinned-env runs; sensitive to driver,
  UUID, CPU, RAM, PCIe re-cabling and device-set changes (unit-tested per field).
- Covers **compute devices only** (`DISCRETE_GPU`/`EXTERNAL_GPU`). During this session the
  integrated device's inclusion was seen to flip (`1ee97f74…` over 3 devices → `0a3b…` over 2),
  which invalidated cached profiles mid-verification; the coordinating session landed the
  compute-device-only rule in `fingerprint.rs` (module doc records the observation), a regression
  test (`only_compute_devices_enter_the_fingerprint`) covers it, and `tests/diag_fingerprint.rs`
  now asserts *stored fingerprint == digest of that file's own stored inputs* so the two can never
  disagree again. A final `--reprofile` re-established agreement; both stored profiles now hash
  to the value above.
- **Roles assigned:** `[0] RX 6950 XT → streaming-compute` (28.7 GB/s + 100 % overlap + 3.9×
  cache lever), `[1] Radeon VII → static-resident` (826 GB/s HBM but 2.9 GB/s link, ~0 % overlap),
  `[2] iGPU → display-only`.
- **`--dry-run-plan`** (from cache, writes nothing): per-card VRAM budget (RX 15.98 − 1.5 GiB
  display reserve = **14.48 GiB** — reserve applied only because `card1-DP-1/DP-2/HDMI-A-1`
  report `connected`; MI50 render-only → 15.98 GiB), combined **30.46 GiB ≥ 13 577 MiB** IQ3_XXS
  GPU weights (reference placement `docs/phase-0/05`), expert-parallel default, MI50 excluded from
  streaming, Tier-2 `start 16 / max 24 GiB, watermark MemAvailable < 6 GiB` with the measured
  **8 MiB mlock limit ⇒ `LimitMEMLOCK=infinity`**, Tier-3 `/mnt/agent-storage-a` cap 100 GiB,
  `O_DIRECT 390 MB/s`, plus the Gate-A constraints (no cooperative matrix, Wave32/64 split).

## 7. Host probe (live values in `hw_profile.json`)

CPU `AMD Ryzen 9 7950X` 16C/32T · NUMA 1 · RAM **58.43 GiB** (62 739 529 728 B), MemAvailable
32–33 GB at probe time · **effective mlock limit 8 388 608 B (8 MiB)** → confirms phase-0/04 ·
zram `/dev/zram0` 58.43 GiB · `io_uring`: **available** (`/proc/sys/fs/io_uring_disabled` absent
on this kernel, i.e. no restriction) · storage: `/mnt/agent-storage-a` 671.71 GiB free
(**O_DIRECT 390 MB/s**, 64 MiB in 0.17 s), `/mnt/agent-storage-b` 430.51 GiB, `/` 120.73 GiB ·
competing GPU processes recorded in `notes` (see §8).

**XDG check (requested):** this session has **both `XDG_CONFIG_HOME` and `XDG_CACHE_HOME` empty**,
and `grep -r XDG_CACHE_HOME crates/` returns **0 hits** — the CLI reads only `XDG_CONFIG_HOME`
(`breccia-profile/src/lib.rs::default_profile_path`) with a `~/.config` fallback, and both real
runs above wrote to `/home/korphaus/.config/breccia/hw_profile.json` as intended. The reported
`XDG_CACHE_HOME` resolution issue is therefore **not reproducible and not a defect in this code —
confirmed stale-environment artifact**.

## 8. Deviations & open issues

1. **RX FP32 −9.7 % / FP16 −21.7 % vs phase-0** (MI50 within 0.5 %). Clocks were at boost and
   memory at top state throughout (§5), quick mode lands only −5 %, so the gap tracks *window
   shape*: full mode runs 1 048 576 threads × 4096 rounds (longer sustained dispatch) while the
   RX now also drives three `connected` displays. Phase-0's open item **DPM clock pinning** is the
   right fix before Gate-B-grade numbers; the FP16 figure in particular should not be quoted as
   gate evidence until then. FP16 is not part of the required comparison list.
2. **Contention from other sessions' GPU jobs** produced two bad samples during verification
   (a quick run where the MI50 read 3 GB/s DRAM / 637 GFLOPS, and one RX bandwidth window at
   443 GB/s). Both are visible in `notes` (`competing_gpu_processes`) and the logs; the subsequent
   full `--reprofile` returned to normal values. Recommendation for Gate B: take a benchmark
   window (the standing rule: restore service state afterwards; no services were touched here).
3. **Overlap can read slightly above 100 % or slightly negative on ~1 s legs** (wall-clock noise:
   measured 100.3–100.5 % on RX, −1.2 % … 5.8 % on MI50) — same behaviour as phase-0; the role
   thresholds (10 GB/s / 50 %) are far from both.
4. `breccia-flags` tests fail (another session's in-flight crate) → workspace-wide `cargo test`
   is not globally green yet; re-run once that crate lands.
5. HIP: seam only (`BackendProbe`), no measurements — as scoped. HIP numbers plug into
   `DeviceMeasurements` without a schema change.
6. Open items carried from phase-0 that this tool surfaces but does not solve: mlock 8 MiB on
   service units, gpu-manager's MemAvailable watermark, clock pinning.

## 9. Evidence & artifacts

| Path | Content |
|---|---|
| `/var/tmp/users/korphaus/tmp/opencode/final-{quick,quick-fresh,full,reprofile,cached}.log` | final-code run logs (§4) |
| `/var/tmp/users/korphaus/tmp/opencode/clock-sampler-reprofile.log` | 160 × 250 ms sclk/mclk/busy samples during `--reprofile` |
| `/var/tmp/users/korphaus/tmp/opencode/mesa-trap-evidence.txt` | forced-`MESA_VK_DEVICE_SELECT` run (warning + 3 devices + cached hit) |
| `/var/tmp/users/korphaus/tmp/opencode/profile-quick.log` | pre-restart run that caught the trap in the wild |
| `~/.config/breccia/hw_profile.json` | canonical profile (schema v1, regenerable) |
| `/tmp/hw_profile_test.json` | `--quick --json` profile used by the verification sequence |
| `harnesses/vk-probe/` | untouched predecessor; measurements re-validated against it |

No systemd services, `Reference Projects/`, `/srv` weights, or other agents' crates were touched;
nothing was committed by this task.
