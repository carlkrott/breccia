# Breccia

A hardware-adaptive MoE inference runtime for heterogeneous AMD GPUs (and beyond), built in Rust with a Vulkan/HIP hybrid execution core. Named for the rock composed of mismatched fragments fused into one resilient body — a metaphor for binding gfx906 Wave64 and gfx1030 Wave32 silicon into a single engine.

## Why

Current state of the art on this class of hardware (MI50 16 GB + RX 6950 XT 16 GB, Qwen3.8-Flash-Next IQ3_XXS @ 75 GB):

| Engine | Solo decode | Parallel |
|---|---|---|
| llama.cpp (dual-arch HIP build) | ~23.6 tok/s | 43–45 tok/s @ 4 slots |
| Strata | ~23 tok/s | 47 tok/s @ batch 8 |

Target: **~85 tok/s single-stream decode and 4–6 concurrent agent slots** inside 32 GB VRAM + ~30 GB usable host RAM, via:

- Split-residency MoE expert streaming (VRAM LRU cache → pinned RAM → NVMe)
- Tiered paged KV cache (VRAM → pinned RAM → NVMe) with radix-tree prefix reuse
- Asymmetric dual-GPU scheduling (compute role vs bandwidth role, measured per-device)
- Speculative drafting (MTP head + n-gram drafter) that rides the expert cache
- Hardware auto-profiling with persistent fingerprints (`--reprofile`, `--dry-run-plan`)

The tailnet/QUIC mesh extension (Tier 4) is explicitly deferred until the local engine passes its acceptance gate.

## Repository layout

```
Breccia/
├── README.md               ← this file
├── AGENTS.md               ← agent/developer rules for this repo
├── docs/
│   ├── GIT-WORKFLOW.md     ← branch/commit/PR procedure
│   ├── PROJECT-SETUP.md    ← toolchain, frameworks, build/test commands
│   ├── reference-engines.md← manifest of third-party engines under study
│   ├── plan/               ← phased milestone plan + gate reports
│   ├── phase-0/            ← baselines, backend probes, raw measurements
│   └── specs/              ← model/artifact architecture specs
├── Reference Projects/     ← local clones of engines under study (git-ignored)
├── harnesses/              ← small measurement programs (Phase 0 probes)
└── crates/                 ← Rust workspace (from Phase 1)
```

## Status

| Phase | State |
|---|---|
| 0 — baselines, backend risk, specs, reference engines | **complete — Gate A passed** (`docs/phase-0/gate-A-report.md`) |
| 1 — workspace, `breccia-profile` (M1a), launch-flag surface | in progress — `breccia profile` verified on hardware (`docs/phase-1/01-profile-report.md`); flag parser landing |
| 2 — DMA/transfer harness + zero-copy loader | **complete** — transfer ring measured (`docs/phase-2/01`), GGUF/pack/MTP loader validated on the real 75 GB shards (`docs/phase-2/02`) |
| 3 — dequant/GEMV kernels + forward pass (M2) | in progress — kernel work order item #1 (`docs/phase-3/`) |

Backend decision (2026-10-26): **Vulkan primary**, HIP-style launch flags preserved, HIP secondary. Architecture rules from Gate A: RX 6950 XT = streaming/compute home; MI50 = static-resident only (PCIe ≈2.9 GB/s, no DMA overlap); expert-parallel placement by default.

```bash
cargo run -p breccia-cli -- profile --quick          # calibrate → hw_profile.json
cargo run -p breccia-cli -- profile --dry-run-plan   # role table + placement sketch
./scripts/verify.sh                                  # pre-merge gate (build/test/clippy/fmt)
```
