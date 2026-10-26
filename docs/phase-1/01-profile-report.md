# Phase 1.1 — `breccia profile` implementation report

M1a hardware discovery + calibration tool. Implemented across `breccia-vk` (Vulkan layer, ~2,100 lines incl. shaders), `breccia-profile` (orchestration/schema/roles/plan, ~2,500 lines), and `breccia-cli` (command surface).

## Commands

```bash
cargo run -p breccia-cli -- profile [--reprofile] [--quick] [--json PATH] [--dry-run-plan] [--print-fingerprint]
./scripts/verify.sh          # workspace gate: build + test + clippy/fmt when present
```

Default profile path: `$XDG_CONFIG_HOME/breccia/hw_profile.json` (fallback `~/.config/breccia/hw_profile.json`).

## Verified behaviours (on this machine, 2026-10-26, GPUs otherwise idle)

| Behaviour | Evidence |
|---|---|
| Enumerates **3 devices** (RX 6950 XT, Radeon VII, Raphael iGPU) | `3 device(s) enumerated, benchmarking [0, 1]` |
| RX: DRAM copy **450 GB/s**, cache window **1841 GB/s** (4.1×), PCIe **28.7/28.4 GB/s**, overlap **100%**, FP32 **24.3 TF** | matches probe `docs/phase-0/02` within ~5% |
| MI50: DRAM **825 GB/s**, PCIe **2.9/2.9 GB/s**, overlap **0%**, FP32 **6.7 TF** | matches probe |
| Role assignment | RX → `streaming-compute` (streaming/expert-paging home, prefill, drafter); MI50 → `static-resident` (no PCIe streaming; bandwidth-bound decode); iGPU → `display-only, not benchmarked` |
| `hw_profile.json` written, schema v1 | `schema_version/fingerprint/timestamp/devices/host/notes`, 18 KB |
| Fingerprint (BLAKE3, 64 hex) | `0a3b7974…f4686dc`, stable across consecutive runs |
| **Cached fast path** | 2nd invocation: **0.069 s**, "cached profile reused (fingerprint match)" |
| `--reprofile` | forces full calibration |
| `--dry-run-plan` | instant from cache; prints role table + IQ3_XXS placement sketch (13,577 MiB GPU weights → fits 30.5 GiB combined budget; expert-parallel default; Tier-3 from 04 defaults); writes nothing |
| `--print-fingerprint` | stable across runs |
| Env-trap guard | `MESA_VK_DEVICE_SELECT` unset before instance creation (warn if present) |
| Machine notes | mlock **8 MiB** detected, io_uring available, zram, O_DIRECT **390 MB/s** on `/mnt/agent-storage-a`, competing GPU processes recorded |

Full JSON examples live at `/tmp/verify2.json` during the run (not committed — regenerable).

## Fixes applied during verification

1. **Fingerprint flakiness (bug, fixed):** the first full run recorded only the 2 discrete GPUs while the next recorded 3 — RADV's enumeration of the integrated device transiently dropped out. Since the iGPU is never scheduled (plan: unused), the fingerprint now covers **compute devices only** (`DISCRETE_GPU`/`EXTERNAL_GPU`); a changed discrete-GPU set, driver, topology, CPU or RAM still invalidates. Regression test added (`only_compute_devices_enter_the_fingerprint`). Without this, every other run would force a needless recalibration.
2. Cache/`--dry-run-plan` behaviour was correct once the fingerprint was stable — no code change needed.

## Tests

`cargo test --workspace`: **40 tests green** (34 in breccia-profile incl. fingerprint stability/sensitivity/order-independence, role heuristics, plan sizing, schema round-trip; 6 in breccia-vk incl. PCI topology + sysfs discovery; CLI smoke).

## Open items

- Overlap efficiency can read slightly >100% (wall-clock noise on ~1 s legs) — acceptable; consider clamping display at 100%.
- Role thresholds are documented constants in `role.rs`; first real workload tuning belongs to Phase 2+.
- HIP-side measurements: `backend.rs` seam exists; no HIP probe in Phase 1 (Vulkan is the primary backend per the 2026-10-26 decision).
